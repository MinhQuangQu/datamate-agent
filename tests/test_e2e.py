"""End-to-end: dma.run on a synthetic split with the FakeBackend test double — every stage
(ingest, BSM refine, rules, NLU, both solvers incl. sandbox, arbitration, writer, logs,
manifest) runs; reproducibility (2 runs identical); kill test leaves a valid submission."""
import json
import os
import signal
import subprocess
import sys
import time

import pytest

from dma.eval.scorer import read_jsonl, score
from dma.io.finalize import finalize, validate_record

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src") + os.pathsep + ROOT, PYTHONHASHSEED="0")


def _run(split, out, work, extra=()):
    cmd = [sys.executable, "-m", "dma.run", "--config", os.path.join(ROOT, "configs", "dev.yaml"), "--data", split,
           "--result", out, "--work", work, "--fake-llm", *extra]
    return subprocess.run(cmd, env=ENV, capture_output=True, text=True, timeout=900)


@pytest.fixture(scope="module")
def run_a(synth_split, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("res_a"))
    r = _run(synth_split, out, str(tmp_path_factory.mktemp("work_a")))
    assert r.returncode == 0, r.stderr[-2000:]
    return out


def test_e2e_outputs_logs_manifest(run_a, synth_split):
    subs = read_jsonl(os.path.join(run_a, "submission.jsonl"))
    qs = read_jsonl(os.path.join(synth_split, "questions.jsonl"))
    assert [s["id"] for s in subs] and {s["id"] for s in subs} == {q["id"] for q in qs}
    assert all(validate_record(s) is None for s in subs)
    rep = score(subs, read_jsonl(os.path.join(synth_split, "answers.jsonl")))
    assert rep.pa >= 0.95, rep.pa
    for q in qs:
        tr = json.load(open(os.path.join(run_a, "logs", f"{q['id']}.json"), encoding="utf-8"))
        assert {"id", "final", "timing_ms", "candidates", "decision", "llm_calls"} <= set(tr)
        assert "total" in tr["timing_ms"]
    man = json.load(open(os.path.join(run_a, "logs", "run_manifest.json"), encoding="utf-8"))
    assert man["n_questions"] == len(qs) and man["businesses"] and man["config_sha256"]
    assert os.path.exists(os.path.join(run_a, "logs", "events.jsonl"))
    assert os.path.getsize(os.path.join(run_a, "submission.jsonl")) < 5 * 1024 * 1024


def test_reproducible(run_a, synth_split, tmp_path):
    r = _run(synth_split, str(tmp_path / "b"), str(tmp_path / "wb"))
    assert r.returncode == 0
    a = sorted(read_jsonl(os.path.join(run_a, "submission.jsonl")), key=lambda x: x["id"])
    b = sorted(read_jsonl(str(tmp_path / "b" / "submission.jsonl")), key=lambda x: x["id"])
    assert a == b


def test_kill_leaves_valid_submission(synth_split, tmp_path):
    out = tmp_path / "k"
    cmd = [sys.executable, "-m", "dma.run", "--config", os.path.join(ROOT, "configs", "dev.yaml"), "--data", synth_split,
           "--result", str(out), "--work", str(tmp_path / "wk"), "--fake-llm"]
    p = subprocess.Popen(cmd, env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(4)
    p.send_signal(signal.SIGKILL)
    p.wait()
    stats = finalize(synth_split, str(out))
    subs = read_jsonl(str(out / "submission.jsonl"))
    assert len(subs) == stats["questions"] and all(validate_record(s) is None for s in subs)


def test_no_llm_mode_abstains_or_rule_answers(synth_split, tmp_path):
    out = tmp_path / "n"
    cmd = [sys.executable, "-m", "dma.run", "--config", os.path.join(ROOT, "configs", "dev.yaml"), "--data", synth_split,
           "--result", str(out), "--work", str(tmp_path / "wn"), "--no-llm"]
    r = subprocess.run(cmd, env=ENV, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0
    subs = read_jsonl(str(out / "submission.jsonl"))
    assert subs and all(s["type"] in ("abstain", "unanswerable") for s in subs)
