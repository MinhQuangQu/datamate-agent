import json
from datetime import date

import numpy as np

from dma.output.formatter import FormatConfig, format_answer, round2, safe_format
from dma.output.writer import SubmissionWriter, write_log


def test_numbers_and_rounding_modes():
    assert format_answer("q", "number", 48250000.0)["answer"] == 48250000
    assert isinstance(format_answer("q", "number", np.int64(7))["answer"], int)
    assert format_answer("q", "number", 1234.5678)["answer"] == 1234.57
    assert format_answer("q", "percent", -12.4499)["answer"] == -12.45
    assert format_answer("q", "number", -0.001)["answer"] == 0
    assert round2(2.675, "python") == 2.67       # binary float: 2.67499999...
    assert round2(2.675, "half_up") == 2.68      # decimal semantics
    assert round2(0.125, "python") == 0.12 and round2(0.125, "half_up") == 0.13


def test_other_types():
    assert format_answer("q", "date", date(2026, 8, 15))["answer"] == "2026-08-15"
    assert format_answer("q", "date", "2025-09-30", month_only=True)["answer"] == "2025-09"
    assert format_answer("q", "boolean", np.bool_(True))["answer"] is True
    assert format_answer("q", "list", ["Cà phê sữa", np.float64(3.0)])["answer"] == ["Cà phê sữa", 3]
    t = format_answer("q", "table", {"columns": ["khach_hang", "so_tien_no"], "rows": [["Chị Lan", 3200000.0]]})
    assert t["answer"]["rows"] == [["Chị Lan", 3200000]]
    assert format_answer("q", "unanswerable", reason="Không có giá vốn") == \
        {"id": "q", "type": "unanswerable", "reason": "Không có giá vốn"}


def test_safe_format_falls_back_to_abstain():
    rec, err = safe_format("q", "number", "abc")
    assert rec == {"id": "q", "type": "abstain"} and err
    rec, err = safe_format("q", "boolean", 1)
    assert rec["type"] == "abstain"


def test_writer_resume_and_dedup(tmp_path):
    p = tmp_path / "result" / "submission.jsonl"
    w = SubmissionWriter(str(p))
    assert w.write({"id": "a", "type": "number", "answer": 1})
    assert not w.write({"id": "a", "type": "number", "answer": 2})     # duplicate ignored
    w.close()
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"id": "torn')                                          # simulated crash mid-line
    w2 = SubmissionWriter(str(p), resume=True)
    assert w2.done == {"a"}
    w2.write({"id": "b", "type": "string", "answer": "Bạc xỉu"})
    w2.close()
    lines = p.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[-1])["answer"] == "Bạc xỉu"
    path = write_log(str(tmp_path / "logs"), "q/1", {"answer": 1, "d": date(2026, 1, 1)})
    assert json.load(open(path))["d"] == "2026-01-01"


def test_config_rounding_used():
    assert format_answer("q", "number", 2.675, cfg=FormatConfig(rounding="half_up"))["answer"] == 2.68
