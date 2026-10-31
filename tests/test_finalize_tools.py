import json
import subprocess
import sys
from pathlib import Path

from dma.io.finalize import finalize, find_questions, validate_record

ROOT = Path(__file__).resolve().parents[1]


def _split(tmp_path: Path) -> Path:
    d = tmp_path / "data" / "split"
    d.mkdir(parents=True)
    qs = [{"id": f"q{i}", "business_id": "b", "question": "x", "as_of_date": "2026-03-04"} for i in range(4)]
    (d / "questions.jsonl").write_text("\n".join(json.dumps(q) for q in qs), encoding="utf-8")
    return tmp_path / "data"


def test_validate_record():
    assert validate_record({"id": "a", "type": "number", "answer": 1.5}) is None
    assert validate_record({"id": "a", "type": "number", "answer": True})
    assert validate_record({"id": "a", "type": "date", "answer": "2025-09"}) is None
    assert validate_record({"id": "a", "type": "date", "answer": "09/2025"})
    assert validate_record({"id": "a", "type": "table", "answer": {"columns": ["x"], "rows": [[1], [2, 3]]}})
    assert validate_record({"id": "a", "type": "unanswerable", "reason": "thiếu giá vốn"}) is None
    assert validate_record({"id": "a", "type": "abstain", "answer": 3})


def test_finalize_fills_dedups_and_drops_invalid(tmp_path):
    data = _split(tmp_path)
    assert find_questions(str(data)).endswith("split/questions.jsonl")
    res = tmp_path / "result"
    res.mkdir()
    (res / "submission.jsonl").write_text("\n".join([
        json.dumps({"id": "q0", "type": "number", "answer": 5}),
        json.dumps({"id": "q0", "type": "number", "answer": 6}),          # duplicate -> dropped
        json.dumps({"id": "q1", "type": "boolean", "answer": "yes"}),     # invalid -> dropped -> abstain
        json.dumps({"id": "zz", "type": "string", "answer": "x"}),        # unknown id
        '{"id": "q2", "type": "stri',                                     # torn line
    ]), encoding="utf-8")
    stats = finalize(str(data), str(res))
    lines = [json.loads(x) for x in (res / "submission.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in lines] == ["q0", "q1", "q2", "q3"]
    assert lines[0]["answer"] == 5 and all(r["type"] == "abstain" for r in lines[1:])
    assert stats["duplicate"] == 1 and stats["unknown_id"] == 1 and stats["invalid"] == 2
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_submission.py"),
                          str(res / "submission.jsonl"), str(data / "split" / "questions.jsonl")],
                         capture_output=True, text=True)
    assert out.returncode == 0 and "VALID" in out.stdout


def test_no_id_logic_guard(tmp_path):
    good = tmp_path / "src" / "dma"
    good.mkdir(parents=True)
    (good / "ok.py").write_text("def f(q):\n    return q.business_id + '/db.duckdb'\n")
    run = lambda: subprocess.run([sys.executable, str(ROOT / "scripts" / "check_no_id_logic.py"), str(tmp_path / "src")],
                                 capture_output=True, text=True)
    assert run().returncode == 0
    (good / "bad.py").write_text("def f(q):\n    if q.business_id == 'biz_118':\n        return 1\n")
    r = run()
    assert r.returncode == 1 and "branch on business_id" in r.stdout
