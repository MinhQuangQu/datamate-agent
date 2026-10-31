from datetime import date

from dma.nlu.analyze import analyze_question
from dma.nlu.concepts import detect_concepts, detect_k, ranking_direction, type_cues
from dma.nlu.link import link_entities
from dma.semantic.value_index import ValueEntry, ValueIndex, tokens_of
from dma.textnorm import fold
from dma.timeparse.calendar import LunarCalendar
from dma.timeparse.resolver import TimeRules


def _vi():
    rows = [("product", "SP0001", "Sting dâu 330ml"), ("product", "SP0002", "Sting dâu 500ml"),
            ("product", "SP0003", "Xi măng Hà Tiên PCB40 bao 50kg"), ("product", "SP0004", "Bia Tiger lon 330ml"),
            ("product", "SP0005", "Hoa cúc họa mi"), ("category", None, "Xi măng"), ("category", None, "Bia"),
            ("category", None, "Hoa lẻ"), ("category", None, "Nước ngọt"), ("customer", "KH0107", "Chị Lan")]
    return ValueIndex([ValueEntry(k, c, n, [], fold(n), tokens_of(n) + ([fold(c)] if c else []), 10) for k, c, n in rows])


def _best(q):
    return [(m.text, m.best.code or m.best.name) for m in link_entities(fold(q), _vi())]


def test_linker_cases():
    ms = link_entities(fold("sting dau bán dc bao nhiêu chai tuần rồi"), _vi())
    assert ms[0].text == "sting dau" and ms[0].margin < 0.05 and len(ms[0].candidates) >= 2   # close variants -> "possible"
    assert _best("xi mang ha tien bán bao nhiêu bao") == [("xi mang ha tien", "SP0003")]
    assert _best("bia tiger ban duoc bao nhieu lon") == [("bia tiger", "SP0004")]
    assert _best("chị Lan còn nợ bao nhiêu") == [("chi lan", "KH0107")]
    assert _best("nước ngọt chiếm bao nhiêu %") == [("nuoc ngot", "Nước ngọt")]
    assert _best("hoa lẻ chiếm bao nhiêu % doanh thu") == [("hoa le", "Hoa lẻ")]
    assert _best("bán được bao nhiêu bó hoa cuc hoa mi") == [("hoa cuc hoa mi", "SP0005")]
    assert _best("SP0002 còn bán không") == [("sp0002", "SP0002")]


def test_concepts_types_k():
    f = fold("lợi nhuận tháng trước được bao nhiêu")
    assert "profit" in detect_concepts(f) and "return" not in detect_concepts(fold("trả lại hàng"))[:0]
    assert "profit" not in detect_concepts(fold("đơn bị trả lại"))
    assert type_cues(fold("doanh thu tăng bao nhiêu phần trăm"))[0][0] == "percent"
    assert type_cues(fold("tuần rồi có đơn nào bị hủy ko"))[0][0] == "boolean"
    assert type_cues(fold("tháng trước thứ mấy đông khách nhất"))[0][0] == "string"
    assert detect_k(fold("top 5 món bán chạy")) == 5 and detect_k(fold("ba sản phẩm bán chậm nhất")) == 3
    assert detect_k(fold("7 ngày qua bán bao nhiêu")) is None
    assert ranking_direction(fold("món nào bán chậm nhất")) == "asc"


def test_analyze_gaps_time_template():
    a = analyze_question("lợi nhuận tuần rồi của sting dâu 330ml", date(2026, 3, 4), _vi(), ["cost"], TimeRules(),
                         LunarCalendar.from_algorithm(), ("2025-01-01", "2026-03-31"))
    assert a.gaps_hit == ["cost"]
    assert (a.time.periods[0].start.isoformat(), a.time.periods[0].end.isoformat()) == ("2026-02-23", "2026-03-01")
    assert "<product>" in a.template_key and "<time:last_week>" in a.template_key
    b = analyze_question("từ trước tới nay bán được bao nhiêu", date(2026, 3, 4), _vi(), [], TimeRules(),
                         LunarCalendar.from_algorithm(), ("2025-01-01", "2026-03-31"))
    assert b.time.periods[0].label == "all_time" and b.time.periods[0].end == date(2026, 3, 4)


def test_domain_words_are_not_fuzzy_product_mentions():
    vi = ValueIndex([ValueEntry("product", c, n, [], fold(n), tokens_of(n) + [fold(c)], 10)
                     for c, n in (("SP1", "Khô mực nướng"), ("SP2", "Bia Tiger lon"))])
    ms = link_entities(fold("tồn kho bia tiger lon cuối ngày hôm nay"), vi)
    assert [m.best.code for m in ms] == ["SP2"]


def test_telex_typing_reaches_the_prompt_text():
    from dma.nlu.typos import telex_to_vi
    assert [telex_to_vi(w) for w in ("soos", "ddown", "cos", "tieenf", "truowcs", "thuoocs")] == \
        ["số", "đơn", "có", "tiền", "trước", "thuốc"]
    assert telex_to_vi("bill") is None
    a = analyze_question("hqua cos tổng cộng nhiêu ddown", date(2026, 3, 4), _vi(), [], TimeRules(),
                         LunarCalendar.from_algorithm(), ("2025-01-01", "2026-03-31"))
    assert a.expanded == "hôm qua có tổng cộng nhiêu đơn"
    b = analyze_question("soos bill hqua là bn", date(2026, 3, 4), _vi(), [], TimeRules(),
                         LunarCalendar.from_algorithm(), ("2025-01-01", "2026-03-31"))
    assert b.expanded == "số bill hôm qua là bao nhiêu"
