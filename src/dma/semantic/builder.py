"""BSM refinement with the LLM: P-BSM ×N samples (thinking) -> field-wise vote with the
heuristic proposal as a weighted voter (05.2 §4.2). Gaps are recomputed deterministically
from the voted roles. Status values left unmapped get one P-STATUS-MAP call (V6 repair)."""
from __future__ import annotations

import asyncio
import json
from collections import Counter, defaultdict

from dma.llm import prompts as P
from dma.semantic.bsm import BSM, ColumnRole, PACK_ENTITIES, load_ontology
from dma.semantic.heuristics import compute_gaps, distinct_values, status_map
from dma.semantic.validate import sanitize_roles

PROPOSAL_WEIGHT = 1.5


def roles_compact() -> str:
    ont = load_ontology()
    return "\n".join(f"{name}: {spec['dtype']}" for name, spec in ont["roles"].items())


def proposal_json(b: BSM) -> str:
    d = {"tables": [{"name": t.name, "pack_entity": t.pack_entity, "grain": t.grain, "primary_key": t.primary_key}
                    for t in b.tables],
         "roles": [{"table": r.table, "column": r.column, "role": r.role, "unit_multiplier": r.unit_multiplier,
                    **({"value_map": r.value_map} if r.value_map else {})} for r in b.roles],
         "relationships": [r.model_dump() for r in b.relationships], "gaps": b.gaps, "domain_guess": b.domain_guess}
    return json.dumps(d, ensure_ascii=False, sort_keys=True)


def _bsm_schema() -> dict:
    return BSM.model_json_schema()


async def refine_bsm(proposal: BSM, datacard: str, readme: str, prof: dict, con, llm, n_samples: int,
                     seed: int, replica: int = 0, calls: list | None = None) -> tuple[BSM, dict]:
    report = {"samples_ok": 0, "samples": n_samples, "changed_roles": [], "changed_packs": []}
    if llm is None or not getattr(llm, "enabled", False) or n_samples <= 0:
        return proposal, report
    excerpt = (readme or "")[:4500]
    msgs = [{"role": "system", "content": P.BSM_SYSTEM},
            {"role": "user", "content": P.BSM_USER.format(roles_compact=roles_compact(), proposal_json=proposal_json(proposal),
                                                          readme_excerpt=excerpt or "(empty)", data_card=datacard)}]
    samples: list[BSM] = []
    # independent samples (own seeds) -> concurrent; gather keeps sample order, so the vote is unchanged
    results = await asyncio.gather(*(
        llm.json("bsm", msgs, schema=_bsm_schema(), seed=seed + 104729 * (i + 1), replica=replica,
                 validate=lambda o: BSM.model_validate(_coerce(o)), calls=calls)
        for i in range(n_samples)))
    for obj, err, _ in results:
        if obj is not None:
            sanitize_roles(obj, prof)
            samples.append(obj)
        elif err:
            report.setdefault("errors", []).append(err[:300])
    report["samples_ok"] = len(samples)
    if not samples:
        return proposal, report
    voted = vote(proposal, samples)
    voted.gaps = compute_gaps(voted, prof, con)
    for r in voted.roles:
        p = proposal.roles_of(r.table).get(r.role)
        if p is None or p.column != r.column:
            report["changed_roles"].append(f"{r.table}.{r.column}->{r.role}")
    for t in voted.tables:
        pt = proposal.table(t.name)
        if pt and pt.pack_entity != t.pack_entity:
            report["changed_packs"].append(f"{t.name}: {pt.pack_entity}->{t.pack_entity}")
    return voted, report


def _coerce(o: dict) -> dict:
    """Tolerate common LLM shape slips before pydantic validation."""
    if not isinstance(o, dict):
        raise ValueError("BSM must be an object")
    o = dict(o)
    for k in ("tables", "roles", "relationships"):
        if not isinstance(o.get(k), list):
            o[k] = []
    o["roles"] = [r for r in o["roles"] if isinstance(r, dict) and r.get("table") and r.get("column") and r.get("role")]
    for r in o["roles"]:
        r["table"] = str(r["table"]).removeprefix("typed.")
        if isinstance(r.get("value_map"), dict):
            r["value_map"] = {str(k): str(v) for k, v in r["value_map"].items()}
        else:
            r.pop("value_map", None)
    o["tables"] = [t for t in o["tables"] if isinstance(t, dict) and t.get("name")]
    for t in o["tables"]:
        t["name"] = str(t["name"]).removeprefix("typed.")
        if t.get("pack_entity") not in PACK_ENTITIES:
            t["pack_entity"] = None
        if not isinstance(t.get("primary_key"), list):
            t["primary_key"] = [t["primary_key"]] if t.get("primary_key") else []
    o["relationships"] = [r for r in o["relationships"] if isinstance(r, dict) and
                          all(r.get(k) for k in ("from_table", "from_column", "to_table", "to_column"))]
    for r in o["relationships"]:
        r["from_table"] = str(r["from_table"]).removeprefix("typed.")
        r["to_table"] = str(r["to_table"]).removeprefix("typed.")
        if r.get("kind") not in ("many_to_one", "one_to_one"):
            r["kind"] = "many_to_one"
    for k in ("gaps", "warnings", "readme_facts", "packs"):
        if not isinstance(o.get(k), list):
            o[k] = []
    o.pop("date_coverage", None)
    o.pop("interpretation", None)
    o["version"] = "bsm/1"
    return o


def vote(proposal: BSM, samples: list[BSM]) -> BSM:
    voters = [(proposal, PROPOSAL_WEIGHT)] + [(s, 1.0) for s in samples]
    # tables -> pack_entity
    tables = []
    for t in proposal.tables:
        c: Counter = Counter()
        for b, w in voters:
            bt = b.table(t.name)
            c[bt.pack_entity if bt else t.pack_entity] += w
        pack, _ = sorted(c.items(), key=lambda kv: (-kv[1], str(kv[0])))[0]
        desc = next((b.table(t.name).description for b, _ in voters[1:] if b.table(t.name) and b.table(t.name).description), t.description)
        tables.append(t.model_copy(update={"pack_entity": pack, "description": desc}))
    # (table, column) -> role
    cols = {(r.table, r.column) for b, _ in voters for r in b.roles}
    by_col: dict[tuple[str, str], Counter] = defaultdict(Counter)
    meta: dict[tuple[str, str, str], list[ColumnRole]] = defaultdict(list)
    for b, w in voters:
        assigned = {(r.table, r.column): r for r in b.roles}
        # a sample votes only on tables it assigned roles in: omitting a table (or all roles) means "keep the
        # proposal" (BSM_SYSTEM), not "these columns have no role"
        voted_tables = {t for t, _ in assigned} if b is not proposal else {t for t, _ in cols}
        for tc in cols:
            if tc[0] not in voted_tables:
                continue
            r = assigned.get(tc)
            by_col[tc][r.role if r else "<none>"] += w
            if r:
                meta[(tc[0], tc[1], r.role)].append(r)
    picks = []
    for tc, c in by_col.items():
        role, w = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        if role != "<none>":
            picks.append((w / sum(c.values()), tc, role))
    picks.sort(key=lambda x: (-x[0], x[1], x[2]))
    used, roles = set(), []
    for w, (t, col), role in picks:
        if (t, role) in used and not role.startswith("custom:"):
            continue
        used.add((t, role))
        rs = meta[(t, col, role)]
        mult = Counter(r.unit_multiplier for r in rs).most_common(1)[0][0]
        vm = _vote_value_map([r.value_map for r in rs if r.value_map])
        roles.append(ColumnRole(table=t, column=col, role=role, confidence=round(w, 3), unit_multiplier=mult,
                                value_map=vm, note=rs[0].note))
    facts = list(dict.fromkeys(proposal.readme_facts + [f for s in samples for f in s.readme_facts]))[:12]
    dom = Counter(b.domain_guess for b, _ in voters).most_common(1)[0][0]
    out = proposal.model_copy(update={"tables": tables, "roles": roles, "readme_facts": facts, "domain_guess": dom})
    low = [f"{r.table}.{r.column}->{r.role} ({r.confidence:.2f})" for r in roles if r.confidence < 2 / 3]
    if low:
        out.warnings = list(out.warnings) + ["low-agreement roles: " + ", ".join(low[:8])]
    return out


def _vote_value_map(maps: list[dict[str, str]]) -> dict[str, str] | None:
    if not maps:
        return None
    keys = sorted({k for m in maps for k in m})
    out = {}
    for k in keys:
        c = Counter(m[k] for m in maps if k in m)
        out[k] = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return out


async def repair_status_maps(bsm: BSM, con, llm, readme: str, seed: int, replica: int = 0, calls=None) -> list[str]:
    """Deterministic map first; values still 'other' (>1% of rows) -> one P-STATUS-MAP call."""
    notes = []
    for r in bsm.roles:
        if r.role not in ("order_status", "line_status", "appointment_status"):
            continue
        rows = distinct_values(con, r.table, r.column)
        vals = [str(v) for v, _ in rows]
        base, _ = status_map(vals)
        vm = {**base, **(r.value_map or {})}
        total = sum(n for _, n in rows) or 1
        other_share = sum(n for v, n in rows if vm.get(str(v), "other") == "other") / total
        if other_share > 0.01 and llm is not None and getattr(llm, "enabled", False):
            msg = P.STATUS_MAP_USER.format(values_with_counts=json.dumps([[str(v), n] for v, n in rows], ensure_ascii=False),
                                           column_name=r.column, notes=(readme or "")[:600])
            obj, err, _ = await llm.json("status_map", [{"role": "user", "content": msg}],
                                         schema={"type": "object"}, seed=seed, replica=replica, calls=calls)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if str(v) in ("completed", "canceled", "returned", "pending", "other") and str(k) in vm and vm[str(k)] == "other":
                        vm[str(k)] = str(v)
                notes.append(f"status map for {r.table}.{r.column} repaired by LLM")
        r.value_map = vm
    return notes
