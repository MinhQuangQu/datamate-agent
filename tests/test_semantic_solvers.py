"""Semantic layer + toolkit + plan normalization + arbitration on the synthetic split."""
import asyncio
import json
import os
from datetime import date

import pytest

from dma.config import load_config
from dma.eval.scorer import read_jsonl, score_item
from dma.nlu.analyze import analyze_question
from dma.solvers.code.toolkit import load_dm
from dma.solvers.common import SolverContext
from dma.solvers.plan.dmql import Plan
from dma.solvers.plan.solver import PlanError, normalize_plan
from dma.types import Candidate
from dma.verify.arbiter import arbitrate, fallback_prob, final_decision
from dma.verify.calibrator import Calibrator
from dma.verify.canonical import canonicalize


def test_every_layout_gets_sales_views(built):
    for b, bkp in built["bkps"].items():
        assert "sem.sales_lines_valid" in bkp.views, (b, bkp.warnings)
        assert {"revenue", "quantity", "order_count"} <= set(bkp.catalog.metrics)
        assert bkp.quality >= 0.9, bkp.warnings
        assert "product" in bkp.catalog.entities and bkp.vindex.kinds().get("product")


def test_valid_order_rule_excludes_cancelled_and_returned(built):
    for bkp in built["bkps"].values():
        c = bkp.cursor()
        bad = c.execute("SELECT COUNT(*) FROM sem.sales_lines_valid v JOIN sem.order_flags f USING (order_id) "
                        "WHERE f.is_canceled OR f.has_return").fetchone()[0]
        assert bad == 0
        st = {r[0] for r in c.execute("SELECT DISTINCT order_status FROM sem.sales_lines").fetchall()}
        assert st <= {"completed", "canceled", "returned"}


def test_gaps(built):
    kinds = {b: set(bkp.bsm.gaps) for b, bkp in built["bkps"].items()}
    assert any("cost" in g for g in kinds.values())                  # a business without purchases
    assert any("cost" not in g for g in kinds.values())              # the one with purchase lots


def test_toolkit_matches_oracle_on_all_answerable_templates(built):
    """The toolkit (= compiler) reproduces the independent oracle on every non-trivial template."""
    data = built["data"]
    qs = {r["id"]: r for r in read_jsonl(os.path.join(data, "questions.jsonl"))}
    gold = {r["id"]: r for r in read_jsonl(os.path.join(data, "answers.jsonl"))}
    from dma.llm.fake import fake_code, _parse_question  # the fake planner only routes the question to dm calls
    n_ok = n = 0
    for qid, q in qs.items():
        bkp = built["bkps"][q["business_id"]]
        dm = load_dm(bkp.dir, built["split"].dir)
        dm._bind_question(q["as_of_date"], q["question"])
        qa = analyze_question(q["question"], date.fromisoformat(q["as_of_date"]), bkp.vindex, bkp.bsm.gaps,
                              built["split"].time_rules, built["split"].calendar, bkp.coverage)
        block = ("question (raw): " + q["question"] + "\nTIME CANDIDATES: " + json.dumps(qa.periods_json())
                 + "\nENTITY CANDIDATES:\n" + "\n".join(f"{r['mention']} | {r['kind']} | {r['code'] or '-'} | {r['name']} | {r['score']}"
                                                         for r in qa.entity_rows()) + "\nCONCEPTS:\n## metrics: "
                 + ", ".join(bkp.catalog.metrics) + "\n")
        code = fake_code(_parse_question(block))
        final = {}

        def final_answer(t, v=None, reason=None, **kw):
            final.update(type=t, value=v)
        exec(code, {"dm": dm, "final_answer": final_answer})
        rec = {"id": qid, "type": final["type"]}
        if final["type"] != "unanswerable":
            rec["answer"] = round(final["value"], 2) if isinstance(final["value"], float) else final["value"]
        n += 1
        n_ok += score_item(rec, gold[qid])[0] == 1
    assert n_ok == n, f"{n_ok}/{n}"


def _ctx(built, biz):
    cfg = load_config(None, {})
    return SolverContext(cfg=cfg, bkp=built["bkps"][biz], split=built["split"], llm=None, memory=None)


def test_normalize_plan_maps_names_status_weekday(built):
    biz = sorted(built["bkps"])[0]
    ctx = _ctx(built, biz)
    bkp = ctx.bkp
    pname = bkp.vindex.entries[0].name
    pcode = bkp.vindex.entries[0].code
    qa = analyze_question("x", date(2026, 3, 4), bkp.vindex, [], built["split"].time_rules, built["split"].calendar)
    p = Plan.model_validate({"answer_type": "number", "op": "aggregate", "measure": {"metric": "quantity"},
                             "filters": [{"field": "product", "op": "=", "value": pname},
                                         {"field": "status", "op": "=", "value": "Đã hủy"},
                                         {"field": "weekday", "op": "in", "value": ["thứ 7", "chủ nhật"]}]})
    p2, notes = normalize_plan(p, qa, ctx)
    vals = {f.field: f.value for f in p2.filters}
    assert vals["product"] == pcode and vals["sem.sales_lines.order_status"] == "canceled" and vals["weekday"] == [6, 7]
    with pytest.raises(PlanError):
        normalize_plan(Plan.model_validate({"answer_type": "number", "op": "aggregate", "measure": {"metric": "nope"}}), qa, ctx)
    with pytest.raises(PlanError):
        normalize_plan(Plan.model_validate({"answer_type": "percent", "op": "aggregate", "measure": {"metric": "revenue"}}), qa, ctx)


def test_final_answer_guards(built):
    bkp = next(iter(built["bkps"].values()))
    dm = load_dm(bkp.dir, built["split"].dir)
    dm._bind_question("2026-03-04", "tăng bao nhiêu %")
    name = bkp.vindex.entries[0].name
    assert dm._final_check("string", name, None, {})[1] == name
    from dma.textnorm import strip_accents
    if strip_accents(name) != name:
        with pytest.raises(ValueError):
            dm._final_check("string", strip_accents(name), None, {})
    with pytest.raises(ValueError):
        dm._final_check("number", "123", None, {})
    assert "percent_scale_suspect" in dm._final_check("percent", 0.12, None, {})[2]["warnings"]


def test_canonical_entity_identity(built):
    bkp = next(iter(built["bkps"].values()))
    e = bkp.vindex.entries[0]
    a = canonicalize({"type": "string", "answer": e.name}, bkp.vindex)
    b = canonicalize({"type": "string", "answer": e.code}, bkp.vindex)
    assert a == b
    assert canonicalize({"type": "number", "answer": 1.0}) == canonicalize({"type": "number", "answer": 1})
    t1 = canonicalize({"type": "table", "answer": {"columns": ["a", "b"], "rows": [["x", 1], ["y", 2]]}})
    t2 = canonicalize({"type": "table", "answer": {"columns": ["k", "v"], "rows": [["y", 2], ["x", 1]]}})
    assert t1 == t2


def test_arbitration_and_decision(built):
    biz = sorted(built["bkps"])[0]
    ctx = _ctx(built, biz)
    qa = analyze_question("doanh thu tháng này", date(2026, 3, 4), ctx.bkp.vindex, [], built["split"].time_rules,
                          built["split"].calendar)

    def cand(s, v):
        return Candidate(solver=s, answer_type="number", value=v, canonical=canonicalize({"type": "number", "answer": v}))
    chosen, info = asyncio.run(arbitrate(qa, ctx, [cand("plan", 10), cand("code", 10)], []))
    assert info["source"] == "agree_ab" and chosen.solver == "plan"

    async def extra():
        return cand("code#2", 12)
    chosen, info = asyncio.run(arbitrate(qa, ctx, [cand("plan", 10), cand("code", 12)], [extra, extra]))
    assert info["source"] == "vote" and chosen.value == 12
    cal = Calibrator()
    d = final_decision(chosen, {"agree_ab": 1}, cal, "agree_ab")
    assert d.action == "answer" and d.p_correct == 0.95
    assert final_decision(chosen, {"agree_ab": 0, "vote_share": 0.4}, cal, "none").action == "abstain"
    assert fallback_prob({"agree_ab": 0, "vote_share": 0.8, "n_candidates": 3}, "vote") == 0.65


def test_bsm_vote_keeps_proposal_when_samples_omit_roles():
    # a refine sample that leaves out `roles` (or a whole table) means "keep the proposal", not "no role"
    from dma.semantic.bsm import BSM, ColumnRole
    from dma.semantic.builder import vote
    prop = BSM(roles=[ColumnRole(table="orders", column="total", role="order_total"),
                      ColumnRole(table="lines", column="qty", role="quantity")])
    partial = BSM(roles=[ColumnRole(table="lines", column="qty", role="custom:qty_raw")])
    voted = vote(prop, [BSM(), BSM(), partial])
    got = {(r.table, r.column): r for r in voted.roles}
    assert got[("orders", "total")].role == "order_total" and got[("orders", "total")].confidence == 1.0
    assert got[("lines", "qty")].role == "quantity"      # 1.5 vs 1.0
