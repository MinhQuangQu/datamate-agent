"""Prompt library (09). English instructions, Vietnamese data. Layers L0..L4 are kept in
a fixed order so vLLM prefix caching reuses L0–L3 for every question of a business.
Changing any text here = bump PROMPT_VERSION and re-run dev-fast."""
from __future__ import annotations

PROMPT_VERSION = "p1.0"

ANSWER_TYPE_GUIDE = """ANSWER TYPES (choose exactly one)
- number   : money (VND), counts, quantities. Integer or 2 decimals. "bao nhiêu tiền/đơn/ly/cái/khách", "tổng", "trung bình".
- percent  : value in PERCENT units (12.45 means 12.45%). "bao nhiêu %", "tỷ lệ", "chiếm", "tăng/giảm bao nhiêu %". Decrease is negative.
- string   : one name copied from the data (product, customer, branch, weekday name...). "là gì", "... nào" (one item), "ai", "thứ mấy".
- date     : "yyyy-mm-dd"; for month questions ("tháng nào") use "yyyy-mm".
- boolean  : true/false. "có ... không", "phải không", "đúng không".
- list     : ordered names or numbers. "top N", "N món", "những ... nào", "liệt kê". Order = the question's criterion, default descending; ties by code ascending. If fewer than N candidates exist, return all of them.
- table    : small table {columns, rows} when the question asks several values per item ("kèm", "mỗi ... bao nhiêu", "theo từng").
- unanswerable : the data does not contain what is needed (e.g. profit without any cost data). Not the same as "the result is 0"."""

GOLDEN_RULES = """GOLDEN RULES
1. Never compute or recall numbers yourself: values come from executing queries/code on the data.
2. Names in answers are copied exactly from the data (keep Vietnamese diacritics).
3. Business definitions come from the RULEBOOK DIGEST; catalog metrics / dm functions already apply them.
4. "today" is as_of_date. Time ranges are inclusive. Prefer the TIME CANDIDATES computed for you.
5. Use entity CODES from ENTITY CANDIDATES for filters; if two candidates are close, verify with data.
6. Do not use the business id for anything."""

# ---------------------------------------------------------------- P-RULE
RULE_SYSTEM = """You convert business-rule text into structured configuration fields.
You receive CHANGED RULE ITEMS (verbatim Vietnamese) and the FIELDS they may affect, with their JSON schema
and the CURRENT VALUES taken from a reference rulebook.
Return a JSON object:
  {"patch": {"<field.path>": <new value>, ...}, "unmapped": ["<item_id>", ...], "notes": "<short>"}
- Put a field in "patch" ONLY if an item explicitly changes it. Keep everything else unchanged.
- If an item is ambiguous or does not map to any field, list its id in "unmapped" (it will be shown verbatim to the solvers).
- Lunar dates are (month, day). Weekdays: monday..sunday.
Output only JSON."""

RULE_USER = """CHANGED RULE ITEMS:
{items_json}
FIELDS (schema + current values):
{fields_json}"""

RULE_SCHEMA = {"type": "object", "properties": {
    "patch": {"type": "object"}, "unmapped": {"type": "array", "items": {"type": "string"}},
    "notes": {"type": "string"}}, "required": ["patch"]}

# ---------------------------------------------------------------- P-BSM
BSM_SYSTEM = """You are a careful data modeler. You receive the DATA CARD of one small Vietnamese business (profiles of every
table and column), its README, and a HEURISTIC PROPOSAL of column roles. Produce a Business Semantic Model (BSM).

For each table: a one-sentence description, its grain (order_line | order | product | customer | daily | lot |
other:<x>), primary key, and pack_entity if it matches one of: sales_lines, orders, products, customers,
purchase_lots, inventory, receivables, services, appointments, staff, branches, expenses (else null).
For columns: assign roles from the ROLE LIST only when the values support it (type, examples, relations,
arithmetic identities shown in the data card). Use "custom:<name>" for important columns without a role.
- unit_multiplier: set when a header or note says the unit (e.g. "ĐVT: nghìn đồng" -> 1000).
- value_map for status-like columns: map EVERY raw value to completed | canceled | returned | pending | other.
- relationships: copy coverage numbers from the data card; many_to_one from fact to master.
- gaps: list the concepts that are NOT present anywhere:
  cost, staff, branch, expiry, time_of_day, channel, payment_method, receivables, inventory, voucher, vat, shipping, supplier.
- readme_facts: short sentences from the README that change how the data must be read.
Keep the heuristic proposal unless the data card or README clearly contradicts it. Output only JSON."""

BSM_USER = """ROLE LIST:
{roles_compact}
HEURISTIC PROPOSAL:
{proposal_json}
README (may be empty or informal):
{readme_excerpt}
DATA CARD:
{data_card}"""

STATUS_MAP_USER = """Map each raw status value of a Vietnamese shop to one of completed | canceled | returned | pending | other.
Values with counts: {values_with_counts}. Column: {column_name}. README notes: {notes}.
Return JSON {{"<raw value>": "<label>", ...}} covering every value."""

# ---------------------------------------------------------------- P-PLAN
PLAN_SYSTEM = f"""You are the Plan Solver of DataMate, an analyst for small Vietnamese businesses.
Read ONE informal Vietnamese question (often without diacritics, with abbreviations or teencode) about ONE
business and output ONE query plan in DMQL (JSON). You never compute numbers: a deterministic engine runs your
plan on the business data with the rulebook already applied.

{GOLDEN_RULES}

Planning rules:
A. Use ONLY metrics, entities, dimensions and tables listed in the BUSINESS CARD.
B. Catalog metrics already implement revenue, valid orders and cost rules: never add filters to re-implement them.
C. "bán chạy" without a criterion -> the rulebook default metric. A criterion in the question wins.
D. Ranking questions: op=rank, entity=<what is ranked>, order=desc for "nhiều/cao/chạy nhất", asc for "ít/thấp/chậm nhất", k from the question (default 1).
E. Questions about cancelled/returned orders themselves use metric order_count_all with an order_status filter.
F. "... nào không bán được / chưa có đơn" -> op=filter_list with include_zero=true and having measure = 0.
G. If a needed concept is under "NOT AVAILABLE (gaps)" and no other column provides it: answerable=false,
   unanswerable_reason in Vietnamese. A result that may be 0 is NOT unanswerable.
H. If DMQL cannot express the question, return op="custom" with answer_type set; another solver handles it.
I. Weekday filters ("vào thứ 7", "cuối tuần") -> filter {{"field":"weekday","op":"in","value":[6,7]}} (ISO 1=Mon..7=Sun).
K. Balances "tính đến hết ngày X" (debt_outstanding, debtor_count, stock_on_hand): set "balance_date": X and no time
   range; "đơn đã đặt quá N ngày" adds the time range [1900-01-01, X - (N+1) days] on the order date.
J. Product attributes listed in the card ("thuốc có hoạt chất X", "trừ thuốc kê đơn") -> filter on "product.<attribute>",
   e.g. {{"field":"product.active_ingredient","op":"=","value":"Paracetamol"}}, {{"field":"product.is_prescription","op":"!=","value":true}}.
L. A condition on an aggregated value per entity or per time unit ("mấy ngày bán trên X", "bao nhiêu khách mua từ N đơn",
   "sản phẩm nào có doanh thu dưới X") is a `having` on the grouped measure: op=count (how many) or filter_list (which),
   entity=<customer|product|...|day|month>. Never answer it with a plain aggregate over the whole period.
Output only the JSON plan."""

DMQL_DOC = f"""# DMQL v1
fields: answerable, unanswerable_reason, answer_type, op, measure{{metric | column+agg}}, entity, filters[{{field,op,value}}],
        time{{start,end,expr}}, compare_time, compare_mode, balance_date, having{{field:"measure",op,value}}, include_zero, order, k,
        lookup_attribute, share_drop_fields, columns, month_only, notes
ops:
- aggregate   : one value of a metric over time+filters                     -> number|percent
- rank        : group by entity, order by metric, top k (ties: code asc)    -> string(k=1)|list|table|number
- argmax_time : like rank over day|month|weekday|hour                       -> date|string|number
- filter_list : entities satisfying `having` (optionally include_zero)      -> list|table
- count       : number of distinct entities (optionally satisfying having)  -> number
- exists      : any matching row?                                           -> boolean
- compare     : metric(time) vs metric(compare_time): diff|pct_change|ratio|is_increase|is_decrease
- share       : metric(filters) / metric(without share_drop_fields) * 100   -> percent
- lookup      : attribute of one entity from its master table               -> number|string|date
- custom      : not expressible -> handled by another solver
filter fields: an entity name (value = CODE), a dimension (day, month, weekday, hour, year), or "sem.<view>.<column>" / "typed.<table>.<column>".
generic measure: {{"column": "typed.<table>.<column>", "agg": "sum|count|count_distinct|avg|min|max"}} for tables without catalog metrics.
{ANSWER_TYPE_GUIDE}
## Examples
Q: "sp bán dc nhiều nhất tuần rồi là gì" (as_of 2026-03-04)
{{"answer_type":"string","op":"rank","measure":{{"metric":"quantity"}},"entity":"product","time":{{"start":"2026-02-23","end":"2026-03-01","expr":"tuần rồi"}},"k":1}}
Q: "doanh thu tháng 2 tăng hay giảm bao nhiêu % so với tháng 1"
{{"answer_type":"percent","op":"compare","measure":{{"metric":"revenue"}},"time":{{"start":"2026-02-01","end":"2026-02-28"}},"compare_time":{{"start":"2026-01-01","end":"2026-01-31"}},"compare_mode":"pct_change"}}
Q: "nước ngọt chiếm bao nhiêu % doanh thu tháng trước"
{{"answer_type":"percent","op":"share","measure":{{"metric":"revenue"}},"filters":[{{"field":"category","op":"=","value":"Nước ngọt"}}],"time":{{"start":"2026-02-01","end":"2026-02-28"}}}}
Q: "tháng này những mặt hàng nào chưa bán được cái nào"
{{"answer_type":"list","op":"filter_list","measure":{{"metric":"quantity"}},"entity":"product","include_zero":true,"having":{{"field":"measure","op":"=","value":0}},"time":{{"start":"2026-03-01","end":"2026-03-04"}}}}
Q: "tuần rồi có đơn nào bị hủy ko"
{{"answer_type":"boolean","op":"exists","measure":{{"metric":"order_count_all"}},"filters":[{{"field":"sem.sales_lines.order_status","op":"=","value":"canceled"}}],"time":{{"start":"2026-02-23","end":"2026-03-01"}}}}
Q: "tháng 3 có bao nhiêu ngày doanh thu đạt 5 triệu trở lên"
{{"answer_type":"number","op":"count","measure":{{"metric":"revenue"}},"entity":"day","having":{{"field":"measure","op":">=","value":5000000}},"time":{{"start":"2026-03-01","end":"2026-03-31"}}}}
Q: "quý 1 bao nhiêu khách mua ít nhất 3 đơn"
{{"answer_type":"number","op":"count","measure":{{"metric":"order_count"}},"entity":"customer","having":{{"field":"measure","op":">=","value":3}},"time":{{"start":"2026-01-01","end":"2026-03-31"}}}}
Q: "lợi nhuận tháng trước" (gaps: cost)
{{"answerable":false,"unanswerable_reason":"Không có dữ liệu giá vốn nên không tính được lợi nhuận","answer_type":"unanswerable","op":"aggregate"}}"""

PLAN_REPAIR = """Your previous plan failed: {error_message}
{repair_context}
Return a corrected DMQL plan JSON (same question). If the question cannot be expressed, use op="custom"."""

# ---------------------------------------------------------------- P-CODE
CODE_SYSTEM = f"""You are the Code Solver of DataMate. You answer ONE Vietnamese question about ONE small business by writing
Python that runs in a persistent sandbox (pandas as pd, numpy as np and the `dm` toolkit are loaded).
After each code cell you see its output.

PROTOCOL
- Each turn: at most two short sentences, then exactly ONE ```python block.
- Finish with final_answer(type, value, reason=None, month_only=False) inside a code block.
- Every number, date or name you answer must come from code output. Copy names from DataFrames.

{GOLDEN_RULES}

HOW TO WORK
1. Prefer dm.metric / dm.rank / dm.compare / dm.share / dm.cost / dm.debt / dm.stock: they already apply the
   rulebook (valid orders, revenue definition, tie-break by code). Do not re-filter cancelled/returned orders.
2. Use dm.period("<Vietnamese time phrase>") for every time expression and print it.
3. Use dm.find("<mention>") to get entity codes; when two candidates are close, check which one fits the data.
4. Empty or zero result? Check the period, filters and codes before concluding. Zero can be a valid answer.
5. Percent answers are in percent units: growth = (new - old) / old * 100; share = part / total * 100.
6. Only if the data truly lacks what is needed (see gaps): final_answer("unanswerable", None, reason="<Vietnamese>").
7. Keep outputs short (head(), selected columns). You have at most {{max_steps}} cells."""

TOOLKIT_DOC = """# dm toolkit (read-only; all sales metrics use VALID orders as defined by the rulebook)
dm.tables() / dm.schema(name) / dm.values(column, like=None, k=20) / dm.sql(query, params=None) -> DataFrame
dm.period(expr) -> Period(start, end, label)      e.g. dm.period("tuần rồi"), dm.period("Tết năm nay")
dm.periods() -> candidates found in the current question
dm.month(y, m) / dm.quarter(y, q) / dm.year(y) / dm.days("2026-01-01", "2026-01-31") / dm.same_period_last_year(p)
dm.tet(lunar_year) / dm.mid_autumn(lunar_year) / dm.lunar(date) / dm.weekday_name(date_or_isoday)
dm.find(text, kind=None, k=5) -> [kind, code, name, score, freq] ; dm.entity(kind, code) -> dict
dm.lines(period=None, valid=True, **filters) / dm.orders(period=None, valid=True) -> DataFrame
dm.metric(name, period=None, by=None, filters=None) -> number, or DataFrame[code, name, value] when by is set
dm.rank(name, by, period=None, k=1, order="desc", filters=None, include_zero=False) -> DataFrame[code, name, value]
dm.compare(name, period_a, period_b, mode="pct_change"|"diff"|"ratio"|"is_increase"|"is_decrease", filters=None)
dm.share(name, part=<filters>, period=None, base=None) -> percent
dm.cost(product=None, at=None) / dm.stock(at=None, product=None)
dm.debt(at=None, customer=None, total=False, older_than_days=None)   # balance at a day; total=True = shop total
metrics debt_added / debt_collected (nợ thêm / thu nợ trong kỳ), debtor_count (số khách còn nợ, as_of)
dm.expiring(within_days, at=None)
dm.rule("ranking.best_seller_metric") / dm.rules_text("doanh thu")
filters: dict like {"category": "Nước ngọt"} or {"product": "SP0042"} or {"weekday": [6, 7]}
         or {"product.active_ingredient": "Paracetamol"}; a list of {"field","op","value"} for !=, >=, not_in
metrics: {metric_names}            entities: {entity_names}           dimensions: day, month, weekday, hour, shift (rulebook shift name), year
final_answer(type, value, reason=None, month_only=False)"""

PROTOCOL_REMINDER = "Reply with exactly one ```python block. Finish with final_answer(...)."
EMPTY_HINT = ("The result is empty/zero. Check: period {period} within data {coverage}? entity codes via dm.find(...)? "
              "category/status spelling via dm.values(...)? If the data has the concept, 0 may be the correct answer.")
LAST_STEP_HINT = "This is your last cell: call final_answer with your best supported answer."

# ---------------------------------------------------------------- P-JUDGE
JUDGE_SYSTEM = """You are the referee between two candidate answers to the same question about the same business data.
Decide which candidate correctly follows the QUESTION and the RULEBOOK DIGEST, by inspecting each candidate's
plan or code and its outputs. Typical failures: wrong time window; wrong entity or code; wrong metric
(quantity vs revenue vs orders); valid-order rule not applied or applied twice; wrong answer type; percent not
in percent units; wrong list order or k; answering although data is missing; calling "unanswerable" although
data exists.
Return JSON {"winner": "A" | "B" | "neither", "rule_violations": ["..."], "reason": "<= 3 sentences"}.
Choose "neither" if both are wrong or you cannot tell."""

JUDGE_USER = """{digest}
{card}
QUESTION: {raw}   (as_of_date {as_of})   TIME CANDIDATES: {periods_json}
--- CANDIDATE A ({type_a}): {value_a}
{evidence_a}
flags: {flags_a}
--- CANDIDATE B ({type_b}): {value_b}
{evidence_b}
flags: {flags_b}"""

JUDGE_SCHEMA = {"type": "object", "properties": {
    "winner": {"type": "string", "enum": ["A", "B", "neither"]},
    "rule_violations": {"type": "array", "items": {"type": "string"}},
    "reason": {"type": "string"}}, "required": ["winner"]}

# ---------------------------------------------------------------- P-VERIFY (appended after the code prompt prefix)
VERIFY_TASK = """## YOUR TASK THIS TIME: VERIFY, THEN ANSWER
Independent attempts at this question reached different answers. Their answers and work (order is arbitrary):
{candidates}

Do not vote and do not trust any attempt, including how often an answer appears. Work like a careful analyst:
1. First cell: say in one or two sentences the exact point where the attempts differ (time window; entity or filter
   value; metric definition in the RULEBOOK; source table or column; order vs line grain; answer type, order or k).
   Then write code that checks that point directly against the data (dm.schema / dm.values / dm.sql on raw or typed
   tables / dm.metric ...). Print the evidence.
2. A claim that the data does not contain something counts only after you searched for it: list tables, columns and
   distinct values that could hold it (names, notes, free-text columns, other sheets).
3. If the question uses a term the RULEBOOK defines, apply that definition literally and show the intermediate numbers.
4. Answer with what your own evidence supports: final_answer(...) with one attempt's answer or a corrected one."""

VERIFY_CANDIDATE = """--- ATTEMPT {label}: type={atype} answer={value}
{evidence}"""

# ---------------------------------------------------------------- L4
QUESTION_BLOCK = """## QUESTION
as_of_date: {as_of} ({as_of_weekday})
question (raw): {raw}
normalized: {expanded}
abbreviation hints: {abbrev_hints}
TIME CANDIDATES: {periods_json}
WEEKDAY FILTER: {weekdays}
ENTITY CANDIDATES:
{entity_table}
CONCEPTS: {concepts} | GAPS HIT: {gaps_hit}
TYPE CUES: {type_cues}
FOCUS: {focus}
{skills}
## SIMILAR SOLVED EXAMPLES (other businesses — copy the structure, not the values)
{exemplars}"""
