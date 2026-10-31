"""Crash-safe outputs.

* SubmissionWriter: one JSON line per finished question, flushed + fsync'ed, so an
  OOM/timeout at hour 5:59 still leaves every finished answer on disk (Rule 8:
  only lines written before the stop are scored).
* write_log(): per-question trace in /result/logs/<id>.json, atomic (tmp + rename).
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Iterable


class SubmissionWriter:
    def __init__(self, path: str, resume: bool = False):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self.done: set[str] = set()
        if resume and os.path.exists(path):
            with open(path, "rb+") as f:          # drop a torn last line left by a crash
                data = f.read()
                if data and not data.endswith(b"\n"):
                    f.truncate(data.rfind(b"\n") + 1)
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        self.done.add(json.loads(line)["id"])
                    except (json.JSONDecodeError, KeyError):
                        continue
        self._f = open(path, "a" if resume else "w", encoding="utf-8")

    def write(self, rec: dict) -> bool:
        line = json.dumps(rec, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            if rec["id"] in self.done:
                return False
            self._f.write(line + "\n")
            self._f.flush()
            os.fsync(self._f.fileno())
            self.done.add(rec["id"])
            return True

    def fill_abstain(self, ids: Iterable[str]) -> int:
        n = 0
        for qid in ids:
            n += self.write({"id": qid, "type": "abstain"})
        return n

    def close(self) -> None:
        with self._lock:
            self._f.close()

    def size_bytes(self) -> int:
        return os.path.getsize(self.path)


def write_log(log_dir: str, qid: str, payload: dict[str, Any]) -> str:
    os.makedirs(log_dir, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in qid)
    path = os.path.join(log_dir, f"{safe}.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, path)
    return path
