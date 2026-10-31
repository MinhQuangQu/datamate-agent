"""FakeBackend — a deterministic TEST DOUBLE for the LLM (dev / CI only, never in the image).

It reads the real prompts produced by the pipeline (L4 question block, heuristic proposal,
...) and answers with simple keyword rules, so every stage — gateway, structured output
parsing, DMQL validation/compilation, CodeAct sandbox loop, arbitration, logging — runs end
to end without a GPU. Its answers are NOT a model's: accuracy numbers obtained with it only
measure the deterministic plumbing.
"""
from __future__ import annotations

import json
import re
from typing import Any

from dma.llm import prompts as P
from dma.textnorm import fold


def _section(text: str, start: str, end: str | None = None) -> str:
    i = text.find(start)
    if i < 0:
        return ""
    i += len(start)
    j = text.find(end, i) if end else -1
    return text[i:j if j >= 0 else None].strip()


def _parse_question(user: str) -> dict:
    raw = _section(user, "question (raw):", "\n")
    periods = json.loads(_section(user, "TIME CANDIDATES:", "\n") or "[]")
    ents = []
    for line in _section(user, "ENTITY CANDIDATES:", "CONCEPTS:").splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 5:
            ents.append({"mention": parts[0], "kind": parts[1], "code": None if parts[2] == "-" else parts[2],
                         "name": parts[3], "score": float(parts[4])})
    metrics = _section(user, "## metrics:", "\n")
    return {"raw": raw, "f": fold(raw), "periods": periods, "ents": ents,
            "metrics": {m.strip() for m in metrics.split(",") if m.strip()}}


def _ent(q: dict, kind: str) -> dict | None:
    c = [e for e in q["ents"] if e["kind"] == kind and e["score"] >= 0.7]
    return max(c, key=lambda e: e["score"]) if c else None


def fake_plan(q: dict) -> dict:
    f, ps = q["f"], q["periods"]
    t = {"start": ps[0]["start"], "end": ps[0]["end"], "expr": ps[0].get("text")} if ps else None
    k = re.search(r"\btop ?(\d+)|\b(\d+) (mat hang|mon|san pham|sp)\b", f)
    k = int(next(g for g in k.groups() if g and g.isdigit())) if k else None
    prod, cat, cust = _ent(q, "product"), _ent(q, "category"), _ent(q, "customer")
    if re.search(r"loi nhuan|\blai\b", f):
        if "gross_profit" in q["metrics"]:
            return {"answer_type": "number", "op": "aggregate", "measure": {"metric": "gross_profit"}, "time": t}
        return {"answerable": False, "unanswerable_reason": "Không có dữ liệu giá vốn", "answer_type": "unanswerable",
                "op": "aggregate"}
    if re.search(r"(tang|giam).*(phan tram|%)|so voi", f) and len(ps) >= 2:
        return {"answer_type": "percent", "op": "compare", "measure": {"metric": "revenue"},
                "time": {"start": ps[0]["start"], "end": ps[0]["end"]},
                "compare_time": {"start": ps[1]["start"], "end": ps[1]["end"]}, "compare_mode": "pct_change"}
    if re.search(r"chiem|ty trong|ti trong", f) and cat:
        return {"answer_type": "percent", "op": "share", "measure": {"metric": "revenue"},
                "filters": [{"field": "category", "op": "=", "value": cat["name"]}], "time": t}
    if re.search(r"\bhuy\b", f):
        return {"answer_type": "boolean", "op": "exists", "measure": {"metric": "order_count_all"},
                "filters": [{"field": "sem.sales_lines.order_status", "op": "=", "value": "canceled"}], "time": t}
    if re.search(r"ngay nao", f):
        return {"answer_type": "date", "op": "argmax_time", "measure": {"metric": "revenue"}, "entity": "day", "time": t}
    if re.search(r"thu may", f):
        return {"answer_type": "string", "op": "argmax_time", "measure": {"metric": "order_count"}, "entity": "weekday", "time": t}
    if re.search(r"khach (nao|hang nao)|kh nao", f):
        return {"answer_type": "string", "op": "rank", "measure": {"metric": "revenue"}, "entity": "customer", "time": t, "k": 1}
    if re.search(r"ban chay|ban (duoc|dc) nhieu", f):
        return {"answer_type": "list" if k else "string", "op": "rank", "measure": {"metric": "quantity"},
                "entity": "product", "time": t, "k": k or 1}
    if re.search(r"bao nhieu don|may don|so don", f):
        return {"answer_type": "number", "op": "aggregate", "measure": {"metric": "order_count"}, "time": t}
    if prod and re.search(r"bao nhieu", f):
        return {"answer_type": "number", "op": "aggregate", "measure": {"metric": "quantity"},
                "filters": [{"field": "product", "op": "=", "value": prod["code"] or prod["name"]}], "time": t}
    return {"answer_type": "number", "op": "aggregate", "measure": {"metric": "revenue"}, "time": t}


def fake_code(q: dict) -> str:
    plan = fake_plan(q)
    ps = q["periods"]
    p0 = f"dm.days({ps[0]['start']!r}, {ps[0]['end']!r})" if ps else "None"
    op, at = plan.get("op"), plan.get("answer_type")
    m = (plan.get("measure") or {}).get("metric")
    if at == "unanswerable":
        return "final_answer('unanswerable', None, reason='Không có dữ liệu giá vốn')"
    if op == "compare":
        return (f"a = dm.days({ps[0]['start']!r}, {ps[0]['end']!r}); b = dm.days({ps[1]['start']!r}, {ps[1]['end']!r})\n"
                f"final_answer('percent', dm.compare('revenue', a, b))")
    if op == "share":
        cat = plan["filters"][0]["value"]
        return f"final_answer('percent', dm.share('revenue', part={{'category': {cat!r}}}, period={p0}))"
    if op == "exists":
        return f"final_answer('boolean', dm.metric('canceled_order_count', period={p0}) > 0)"
    if op == "argmax_time":
        by = plan["entity"]
        col = "code" if by == "day" else "name"
        return f"r = dm.rank({m!r}, by={by!r}, period={p0}, k=1)\nfinal_answer({at!r}, r.iloc[0][{col!r}])"
    if op == "rank":
        k = plan.get("k") or 1
        if at == "list":
            return f"r = dm.rank({m!r}, by={plan['entity']!r}, period={p0}, k={k})\nfinal_answer('list', r['name'].tolist())"
        return f"r = dm.rank({m!r}, by={plan['entity']!r}, period={p0}, k=1)\nfinal_answer('string', r.iloc[0]['name'])"
    filt = {f["field"]: f["value"] for f in plan.get("filters", [])}
    return f"v = dm.metric({m!r}, period={p0}, filters={filt!r})\nprint(v)\nfinal_answer('number', v)"


def _proposal_to_bsm(user: str) -> dict:
    prop = json.loads(_section(user, "HEURISTIC PROPOSAL:", "\nREADME"))
    return {"tables": [{"name": t["name"], "pack_entity": t["pack_entity"], "grain": t.get("grain") or "other",
                        "primary_key": t.get("primary_key") or [], "description": "fake"} for t in prop["tables"]],
            "roles": prop["roles"], "relationships": prop.get("relationships", []), "gaps": prop.get("gaps", []),
            "domain_guess": prop.get("domain_guess", "other")}


class FakeBackend:
    def __init__(self):
        self.n_calls = 0

    async def complete(self, body: dict) -> dict:
        self.n_calls += 1
        msgs = body["messages"]
        system = msgs[0]["content"] if msgs and msgs[0]["role"] == "system" else ""
        user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        last = msgs[-1]["content"] if msgs else ""
        if system == P.PLAN_SYSTEM:
            text = json.dumps(fake_plan(_parse_question(user)), ensure_ascii=False)
        elif system.startswith(P.CODE_SYSTEM.split("{max_steps}")[0][:200]):
            if "[observation]" in last and "error" in last.lower():
                text = "```python\nfinal_answer('number', 0)\n```"
            else:
                text = "Tính bằng toolkit.\n```python\n" + fake_code(_parse_question(user)) + "\n```"
        elif system == P.BSM_SYSTEM:
            text = json.dumps(_proposal_to_bsm(user), ensure_ascii=False)
        elif system == P.RULE_SYSTEM:
            text = json.dumps({"patch": {}, "unmapped": [], "notes": "fake"})
        elif system == P.JUDGE_SYSTEM:
            text = json.dumps({"winner": "A", "rule_violations": [], "reason": "fake"})
        elif "Map each raw status value" in user:
            text = "{}"
        elif "Your previous plan failed" in last:
            text = json.dumps({"answer_type": "number", "op": "custom"})
        else:
            text = "OK"
        n_prompt = sum(len(m.get("content") or "") for m in msgs) // 3
        return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": n_prompt, "completion_tokens": len(text) // 3}}


_ = Any
