"""Rulebook -> RuleSpec on the REAL train rulebook (assets/train_rulebook.md): reuse, diff, perturbations (05.3 §7)."""
import asyncio
import os

import pytest

from dma.rules.extract import bootstrap_rulespec, build_split_rulespec, load_reference
from dma.rules.parse import parse_rulebook
from dma.rules.rulespec import RuleSpec, render_digest, time_rules_from

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MD = open(os.path.join(ROOT, "assets/train_rulebook.md"), encoding="utf-8").read()


@pytest.fixture(scope="module")
def ref():
    return load_reference(os.path.join(ROOT, "assets/train_rulebook.md"), os.path.join(ROOT, "assets/train_rulespec.json"))


def test_reference_reads_train_rules():
    s = bootstrap_rulespec(MD)
    assert s.valid_order.exclude_order_if_any_line_returned and not s.valid_order.exclude_returned_lines_only
    assert not s.revenue.include_vat and not s.revenue.include_shipping and not s.revenue.subtract_order_voucher
    assert s.ranking.best_seller_metric == "quantity" and s.time.week_start == "monday"
    assert tuple(s.time.tet_start_lunar) == (12, 23) and tuple(s.time.tet_end_lunar) == (1, 10)
    assert not s.vocabulary or all("tu ngu" in "x" or e.term for e in s.vocabulary)


def test_parse_tables_and_bullets():
    items = parse_rulebook("# A\n- x một\n- y hai\n\n| Luật | Mặc định |\n|---|---|\n| Doanh thu | sau CK |\n"
                           "\n| type | Format | Đúng |\n|---|---|---|\n| number | JSON | `1` |\n")
    assert [i.text for i in items] == ["x một", "y hai", "Doanh thu: sau CK", "type: number | Format: JSON | Đúng: `1`"]


@pytest.mark.parametrize("old,new,expect", [
    ("bắt đầu **thứ Hai**", "bắt đầu **Chủ nhật**", {"time.week_start": "sunday"}),
    ("| Bán chạy | theo **số lượng quy về đơn vị cơ bản**", "| Bán chạy | theo **doanh thu**",
     {"ranking.best_seller_metric": "revenue"}),
    ("**không** cộng thuế VAT", "**có** cộng thuế VAT", {"revenue.include_vat": True}),
    ("Đơn có trả hàng, **kể cả chỉ trả một phần**, là đơn trả hàng: bỏ **cả đơn**",
     "Đơn có trả hàng: chỉ bỏ các dòng bị trả, không bỏ cả đơn", {"valid_order.exclude_returned_lines_only": True}),
    ("**23 tháng Chạp**", "**25 tháng Chạp**", {"time.tet_start_lunar": (12, 25)}),
    ("**mùng 10 tháng Giêng**", "**mùng 7 tháng Giêng**", {"time.tet_end_lunar": (1, 7)}),
])
def test_perturbations(ref, old, new, expect):
    items, spec = ref
    assert old in MD, old
    rs, rep = asyncio.run(build_split_rulespec(MD.replace(old, new), items, spec))
    for k, v in expect.items():
        assert rs.get(k) == v, (k, rs.get(k), rep)


def test_same_rulebook_reuses_reference(ref):
    items, spec = ref
    rs, rep = asyncio.run(build_split_rulespec(MD, items, spec))
    assert rep["patch"] == {} and rep["same"] == len(items)


def test_public_style_additions_are_kept_verbatim(ref):
    items, spec = ref
    extra = MD + "\n## 11. Khách\n| Luật | Mặc định |\n|---|---|\n| Khách sộp nhất | khách có **giá trị trung bình một đơn** cao nhất trong kỳ |\n"
    rs, rep = asyncio.run(build_split_rulespec(extra, items, spec))
    assert any("Khách sộp nhất" in t for t in rs.free_text_rules)


def test_missing_rulebook_uses_reference(ref):
    items, spec = ref
    rs, rep = asyncio.run(build_split_rulespec(None, items, spec))
    assert "rulebook_missing" in rs.uncertain_fields


def test_time_rules_and_digest():
    rs = RuleSpec().with_patch({"time.week_start": "sunday"})
    assert time_rules_from(rs.time).week_start_isoday == 7
    assert "Chủ nhật .. Thứ Bảy" in render_digest(rs)
