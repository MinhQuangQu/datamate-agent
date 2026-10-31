"""Measure the DETERMINISTIC layers (ingestion, semantic views, metrics, time, formatting) on a split
with answers, without any LLM (dev tool, never used at inference).

For frequent question families it routes the question to the `dm` toolkit with the real NLU output
(periods, entities) and scores the result with the official scorer. A wrong value where the routing is
right points at a data/semantic bug; the per-family / per-business tables show where to look.

    python scripts/probe_deterministic.py --split data/raw/train --work /tmp/dma_build_train [--family revenue]
(--work = the work dir of a `dma.run --no-llm` build of the same split)
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dma.eval.data import answers_path  # noqa: E402
from dma.eval.scorer import read_jsonl, score_item  # noqa: E402
from dma.nlu.analyze import analyze_question  # noqa: E402
from dma.solvers.code.toolkit import load_dm  # noqa: E402
from dma.textnorm import fold  # noqa: E402

REV = r"doanh thu|dthu|doanh so|revenue|ban duoc .*tien|ban dc .*tien|thu ve|tien hang|tien ban"
FAMILIES = [   # (name, gold types, regex on folded question, negative regex)
    ("cancel_rate", {"percent"}, r"huy", r"so voi|thay doi"),
    ("revenue_pct_change", {"percent"}, rf"({REV}).*(so voi|thay doi|tang|giam)|(so voi|thay doi).*({REV})", r"chiem|ty trong|ti trong"),
    ("revenue_argmax_day", {"date"}, rf"ngay nao.*({REV}|nhieu tien|ban dat)|({REV}).*ngay nao", r"thang nao"),
    ("best_seller", {"list", "string"}, r"ban chay|ban (duoc|dc) nhieu nhat|hot nhat|chay nhat", r"loi nhat|it nhat|cham"),
    ("order_count", {"number"}, r"bao nhieu don|bnhiu don|bn don|may don|so don|so bill|so hoa don|nhieu don", r"huy|tra|ty le|ti le|%"),
    ("product_quantity", {"number"}, r"ban (duoc|dc|ra) (bao nhieu|bn|bnhiu|may|nhieu)", r"tien|doanh|don\b"),
    ("gross_profit", {"number"}, r"lai gop|loi nhuan gop|lai (duoc|dc) bao nhieu", r"mat hang|san pham|nao|rong|chi phi|ty|ti le|%"),
    ("stock_product", {"number"}, r"(ton kho|kho con ton|con ton).*(cuoi ngay|hom nay|hnay)", r"tien|gia von|bao nhieu mat hang|may mat hang|chan"),
    ("debt_total", {"number"}, r"(khach|khach hang) (con )?no (cua hang )?tong cong|tong (so tien )?(khach )?(con )?no", r"thang|tra no|thu"),
    ("debtor_count", {"number"}, r"(bao nhieu|may|bn|bnhiu) khach (con no|chua tra het no|con cong no)|so khach con (cong )?no", r"thang"),
    ("top_customer_orders", {"string"}, r"khach.*(nhieu don|nhieu lan|mua nhieu don)", r"tien|doanh"),
    ("revenue_total", {"number"}, REV, r"ngay nao|thang nao|trung binh|so voi|chenh|hon|kem|moi|tung|nhom|kenh|ca |hinh thuc|khach|nhan vien|ty|ti le"),
]


def route(fam: str, qa, dm, gold_type: str):
    ps = [p for p in qa.time.periods]
    p = ps[0] if ps else None
    if fam == "cancel_rate":
        return "percent", dm.metric("cancel_rate", period=p)
    if fam == "revenue_total":
        return "number", dm.metric("revenue", period=p)
    if fam == "order_count":
        return "number", dm.metric("order_count", period=p)
    if fam == "revenue_argmax_day":
        r = dm.rank("revenue", by="day", period=p, k=1)
        return "date", r.iloc[0]["code"] if len(r) else None
    if fam == "best_seller":
        k = qa.k or 1
        r = dm.rank("quantity", by="product", period=p, k=k)
        names = r["name"].tolist()
        return (gold_type, names if gold_type == "list" else (names[0] if names else None))
    if fam == "product_quantity":
        ent = next((m.best for m in qa.entities if m.best and m.best.kind == "product"), None)
        if ent is None:
            return None, None
        return "number", dm.metric("quantity", period=p, filters={"product": ent.code or ent.name})
    if fam == "gross_profit":
        return "number", dm.metric("gross_profit", period=p)
    if fam == "stock_product":
        ent = next((m.best for m in qa.entities if m.best and m.best.kind == "product"), None)
        if ent is None:
            return None, None
        return "number", dm.stock(at=p.end if p else None, product=ent.code or ent.name)
    if fam in ("debt_total", "debtor_count"):
        old = dm._as_of
        dm._as_of = p.end if p else dm.as_of
        try:
            return "number", dm.metric("debt_outstanding" if fam == "debt_total" else "debtor_count")
        finally:
            dm._as_of = old
    if fam == "top_customer_orders":
        r = dm.rank("order_count", by="customer", period=p, k=1)
        return "string", r.iloc[0]["name"] if len(r) else None
    if fam == "revenue_pct_change":
        if len(ps) >= 2:
            a, b = ps[0], ps[1]
        elif p is not None and re.search(r"thang truoc|thang lien truoc|ky truoc", qa.folded):
            from dma.timeparse.resolver import month
            y, m = (p.start.year, p.start.month - 1) if p.start.month > 1 else (p.start.year - 1, 12)
            a, b = p, month(y, m)
        else:
            return None, None
        return "percent", dm.compare("revenue", a, b)
    return None, None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--family")
    ap.add_argument("--show", type=int, default=6)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    qs = read_jsonl(os.path.join(a.split, "questions.jsonl"))
    gold = {g["id"]: g for g in read_jsonl(answers_path(a.split))}
    split_dir = os.path.join(a.work, "split")
    dms: dict = {}
    res = collections.defaultdict(lambda: [0, 0, 0])          # family -> [n, ok, err]
    byb = collections.defaultdict(lambda: [0, 0])
    bad = collections.defaultdict(list)
    rows = []
    import io
    import contextlib
    for q in qs:
        g = gold[q["id"]]
        f = fold(q["question"].replace("%", " phan tram "))
        fam = next((n for n, types, rx, neg in FAMILIES
                    if g["type"] in types | {"unanswerable"} and re.search(rx, f) and not (neg and re.search(neg, f))), None)
        if fam is None or (a.family and fam != a.family):
            continue
        b = q["business_id"]
        bdir = os.path.join(a.work, "biz", b)
        if not os.path.exists(os.path.join(bdir, "bkp.json")):
            continue
        try:
            if b not in dms:
                dms[b] = load_dm(bdir, split_dir)
                dms[b].bkp.attach_lexicon(dms[b].split.rulebook_text)
                dms[b].vindex = dms[b].bkp.vindex
            dm = dms[b]
            dm._bind_question(q["as_of_date"], q["question"])
            qa = analyze_question(q["question"], date.fromisoformat(q["as_of_date"]), dm.vindex, dm.bkp.bsm.gaps,
                                  dm.trules, dm.cal, dm.bkp.coverage)
            if "out_of_coverage" in qa.gaps_hit:
                t, v = "unanswerable", None
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    t, v = route(fam, qa, dm, g["type"])
            if t is None:
                continue
            rec = {"id": q["id"], "type": t}
            if t != "unanswerable":
                from dma.output.formatter import FormatConfig, safe_format
                rec, err = safe_format(q["id"], t, v, cfg=FormatConfig(rounding="half_up"))
            s, why = score_item(rec, g)
        except Exception as e:  # noqa: BLE001
            s, why, rec = -1, f"error {type(e).__name__}: {str(e)[:120]}", {}
        res[fam][0] += 1
        res[fam][1] += s == 1
        res[fam][2] += why.startswith("error")
        byb[b][0] += 1
        byb[b][1] += s == 1
        rows.append({"id": q["id"], "family": fam, "business": b, "ok": s == 1, "why": why, "q": q["question"],
                     "pred": rec.get("answer", rec.get("type")), "gold": g.get("answer", g["type"])})
        if s != 1:
            bad[fam].append(rows[-1])
    tot_n = sum(v[0] for v in res.values())
    tot_ok = sum(v[1] for v in res.values())
    print(f"probe: {tot_ok}/{tot_n} correct on routed questions ({len(qs)} total)")
    for fam, (n, ok, err) in sorted(res.items(), key=lambda kv: -kv[1][0]):
        print(f"  {fam:20s} n={n:4d} acc={ok / n:.3f} errors={err}")
    print("by business:", " ".join(f"{b}:{ok}/{n}" for b, (n, ok) in sorted(byb.items())))
    for fam, xs in bad.items():
        print(f"--- {fam} wrong ({len(xs)})")
        for r in xs[: a.show]:
            print(f"   {r['business']} | {r['q'][:80]} | pred={json.dumps(r['pred'], ensure_ascii=False)[:60]} | "
                  f"gold={json.dumps(r['gold'], ensure_ascii=False)[:60]} | {r['why'][:60]}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fo:
            for r in rows:
                fo.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
