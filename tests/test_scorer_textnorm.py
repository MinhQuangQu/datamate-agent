from dma.eval.scorer import ScoreOptions, score, score_item
from dma.textnorm import expand_abbrev, fold, has_vietnamese_diacritics, scorer_norm


def test_scorer_norm_matches_portal_examples():
    assert scorer_norm("Sting dâu 330ml") == "sting dâu 330ml"
    assert scorer_norm("SP0042") == "sp0042"
    assert scorer_norm("  Xi măng Hà Tiên PCB40 (bao 50kg) ") == "xi măng hà tiên pcb40 bao 50kg"
    assert scorer_norm("so_tien_no") == "so tien no"          # underscore is not a letter/digit
    assert scorer_norm("Sting dau 330ml") != scorer_norm("Sting dâu 330ml")   # diacritics kept


def test_nfc_equivalence():
    nfd = "Cà phê sữa"   # decomposed "Cà phê sữa"
    assert scorer_norm(nfd) == scorer_norm("Cà phê sữa")


def test_fold_and_abbrev():
    assert fold("Trà Đào Cam Sả") == "tra dao cam sa"
    assert not has_vietnamese_diacritics("sp ban dc nhieu nhat")
    assert has_vietnamese_diacritics("sp bán dc nhiều nhất")
    text, hints = expand_abbrev("sp bán dc nhiều nhất tuần rồi là j, kh nào nợ")
    assert text.startswith("sản phẩm bán được nhiều nhất")
    assert "gì" in text
    assert any("'kh'" in h for h in hints)


G = [
    {"id": "n", "type": "number", "answer": 48250000},
    {"id": "f", "type": "number", "answer": 1234.57},
    {"id": "p", "type": "percent", "answer": -12.45},
    {"id": "s", "type": "string", "answer": "Xi măng Hà Tiên PCB40 bao 50kg", "alias": ["SP0001"]},
    {"id": "d", "type": "date", "answer": "2025-09"},
    {"id": "b", "type": "boolean", "answer": True},
    {"id": "l", "type": "list", "answer": ["Cà phê sữa", "Bạc xỉu", "Trà đào"]},
    {"id": "t", "type": "table", "answer": {"columns": ["khach_hang", "so_tien_no"],
                                            "rows": [["Chị Lan", 3200000], ["Anh Tú", 1500000]]}},
    {"id": "u", "type": "unanswerable", "answer": None},
]


def test_each_type_correct():
    P = [
        {"id": "n", "type": "number", "answer": 48250000.0},
        {"id": "f", "type": "number", "answer": 1234.57},
        {"id": "p", "type": "percent", "answer": -12.449},          # rounded to -12.45 by the rule
        {"id": "s", "type": "string", "answer": "sp0001"},          # code accepted via alias
        {"id": "d", "type": "date", "answer": "2025-09"},
        {"id": "b", "type": "boolean", "answer": True},
        {"id": "l", "type": "list", "answer": ["cà phê sữa", "Bạc  xỉu", "Trà đào!"]},
        {"id": "t", "type": "table", "answer": {"columns": ["so_tien_no", "khach_hang"],
                                                "rows": [[1500000, "anh tú"], [3200000.0, "Chị Lan"]]}},
        {"id": "u", "type": "unanswerable", "reason": "x"},
    ]
    rep = score(P, G)
    assert rep.correct == 9 and rep.wrong == 0 and rep.pa == 1.0


def test_penalties_and_abstain():
    P = [
        {"id": "n", "type": "percent", "answer": 48250000},          # type mismatch -> -1
        {"id": "f", "type": "number", "answer": 1234.5678},          # not rounded -> -1 (strict)
        {"id": "p", "type": "abstain"},                              # 0
        {"id": "s", "type": "string", "answer": "Xi mang Ha Tien PCB40 bao 50kg"},   # no accents -> -1
        {"id": "l", "type": "list", "answer": ["Bạc xỉu", "Cà phê sữa", "Trà đào"]},  # order -> -1
        {"id": "u", "type": "number", "answer": 0},                  # answering an unanswerable -> -1
        # d, b, t missing -> 0
    ]
    rep = score(P, G)
    assert rep.correct == 0 and rep.wrong == 5 and rep.abstain == 4
    assert abs(rep.pa - (-5 / 9)) < 1e-12


def test_number_rounding_option_and_bool_not_number():
    g = {"id": "x", "type": "number", "answer": 1234.57}
    assert score_item({"id": "x", "type": "number", "answer": 1234.5678}, g,
                      ScoreOptions(round_number_pred=True))[0] == 1
    assert score_item({"id": "x", "type": "number", "answer": True}, {"id": "x", "type": "number", "answer": 1})[0] == -1


def test_table_positional_fallback_and_set_rows():
    g = {"id": "t", "type": "table", "answer": {"columns": ["khach_hang", "so_tien_no"],
                                                "rows": [["Chị Lan", 3200000], ["Anh Tú", 1500000]]}}
    by_pos = {"id": "t", "type": "table", "answer": {"columns": ["Khách", "Nợ"],
                                                     "rows": [["Anh Tú", 1500000], ["Chị Lan", 3200000]]}}
    assert score_item(by_pos, g)[0] == 1
    swapped = {"id": "t", "type": "table", "answer": {"columns": ["Nợ", "Khách"],
                                                      "rows": [[1500000, "Anh Tú"], [3200000, "Chị Lan"]]}}
    assert score_item(swapped, g)[0] == -1     # names don't match -> positional -> wrong
