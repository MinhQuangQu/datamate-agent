"""Persistent, killable Python kernel for the Code Solver (CodeAct loop).

Parent  <--multiprocessing.Pipe-->  worker process (spawn)
  * variables persist between cells of the same question (Jupyter-like),
  * `reset()` restores the post-bootstrap namespace between questions,
  * a cell that exceeds its timeout kills the worker; it is respawned automatically,
  * an audit hook blocks subprocesses, sockets and writes outside the work dir/tmp,
  * the generated code calls final_answer(type, value, reason=None) to finish.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from typing import Any

MAX_OUT = 4000


@dataclass
class ExecResult:
    ok: bool
    stdout: str = ""
    value: str | None = None
    error: str | None = None
    final: dict | None = None
    elapsed: float = 0.0
    timed_out: bool = False

    def observation(self) -> str:
        """Text fed back to the LLM."""
        parts = []
        if self.stdout.strip():
            parts.append(self.stdout.rstrip())
        if self.value is not None:
            parts.append(f"[value] {self.value}")
        if self.error:
            parts.append(f"[error] {self.error}")
        if self.final is not None:
            parts.append(f"[final_answer] {self.final}")
        return "\n".join(parts) if parts else "[no output]"


# ----------------------------------------------------------------- worker side
class _Final(BaseException):
    pass


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n // 2] + f"\n... [{len(s) - n} chars truncated] ...\n" + s[-n // 2:]


def _jsonable(x: Any) -> Any:
    try:
        import numpy as np
        import pandas as pd
        if isinstance(x, pd.DataFrame):
            return {"columns": [str(c) for c in x.columns], "rows": [[_jsonable(v) for v in r] for r in x.itertuples(index=False)]}
        if isinstance(x, (pd.Series, np.ndarray)):
            return [_jsonable(v) for v in list(x)]
        if isinstance(x, pd.Timestamp):
            return x.to_pydatetime().date().isoformat() if x.hour == x.minute == 0 else x.isoformat()
        if isinstance(x, np.generic):
            return x.item()
    except ImportError:
        pass
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "isoformat"):
        return x.isoformat()
    return x


def _install_guards(workdir: str) -> None:
    import sys
    allowed = (os.path.realpath(workdir), "/tmp", "/dev/null", "/dev/shm")
    blocked_events = {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.fork",
                      "os.forkpty", "socket.connect", "socket.bind", "pty.spawn"}
    wflags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC

    def hook(event: str, args: tuple) -> None:
        if event in blocked_events:
            raise PermissionError(f"sandbox: '{event}' is not allowed")
        if event == "open" and args:
            path, mode, flags = (list(args) + [None, None, None])[:3]
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or \
                      (mode is None and isinstance(flags, int) and flags & wflags)
            if writing and isinstance(path, (str, bytes)):
                p = os.path.realpath(path.decode() if isinstance(path, bytes) else path)
                if not p.startswith(allowed):
                    raise PermissionError(f"sandbox: write outside work dir blocked: {p}")

    sys.addaudithook(hook)


def worker_main(conn, bootstrap: str, workdir: str, mem_mb: int | None) -> None:  # pragma: no cover (child)
    import ast
    import contextlib
    import io
    import traceback
    if mem_mb:
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (mem_mb << 20, mem_mb << 20))
        except (ImportError, ValueError, OSError):
            pass
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)
    base: dict[str, Any] = {"__name__": "__sandbox__"}
    boot_err = None
    try:
        exec(compile(bootstrap, "<bootstrap>", "exec"), base)
    except Exception:  # noqa: BLE001
        boot_err = traceback.format_exc(limit=3)
    _install_guards(workdir)
    ns = dict(base)
    conn.send({"ready": True, "boot_error": boot_err})
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        op = msg.get("op")
        if op == "close":
            return
        if op == "reset":
            ns = dict(base)
            conn.send({"ok": True})
            continue
        if op == "rebase":   # bind the kernel to another business without respawning (05.8 §6)
            err = None
            try:
                exec(compile(msg["code"], "<rebase>", "exec"), base)
            except Exception:  # noqa: BLE001
                err = traceback.format_exc(limit=3)[-2000:]
            ns = dict(base)
            conn.send({"ok": err is None, "error": err})
            continue
        final: dict[str, Any] = {}

        def final_answer(type: str, value: Any = None, reason: str | None = None, **extra: Any) -> None:  # noqa: A002
            check = ns.get("__final_check__")
            if check is not None:     # toolkit validation: raises ValueError the agent sees and fixes
                type, value, extra = check(type, value, reason, extra)
            final.update({"type": type, "value": value, "reason": reason, **extra})
            raise _Final()

        ns["final_answer"] = final_answer
        buf = io.StringIO()
        value_repr, err = None, None
        try:
            tree = ast.parse(msg["code"], mode="exec")
            last = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = ast.Expression(tree.body.pop().value)
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                exec(compile(tree, "<cell>", "exec"), ns)
                if last is not None:
                    v = eval(compile(last, "<cell>", "eval"), ns)
                    if v is not None:
                        value_repr = _truncate(repr(v), msg.get("max_output", MAX_OUT))
        except _Final:
            pass
        except BaseException:  # noqa: BLE001 - report everything incl. SystemExit
            tb = traceback.format_exc()
            lines = [ln for ln in tb.splitlines() if "sandbox.py" not in ln]
            err = _truncate("\n".join(lines[-12:]), 2000)
        try:
            payload = {"ok": err is None, "stdout": _truncate(buf.getvalue(), msg.get("max_output", MAX_OUT)),
                       "value": value_repr, "error": err, "final": _jsonable(final) if final else None}
            conn.send(payload)
        except Exception as e:  # noqa: BLE001 - unpicklable final value
            conn.send({"ok": False, "stdout": "", "value": None, "final": None,
                       "error": f"final_answer value not serialisable: {e}"})


# ----------------------------------------------------------------- parent side
class Sandbox:
    def __init__(self, bootstrap: str = "", workdir: str = "/tmp/dma/sandbox", timeout_s: float = 20.0,
                 mem_mb: int | None = None, max_output: int = MAX_OUT, start_timeout_s: float = 60.0):
        self.bootstrap, self.workdir, self.timeout_s = bootstrap, workdir, timeout_s
        self.mem_mb, self.max_output, self.start_timeout_s = mem_mb, max_output, start_timeout_s
        self._ctx = mp.get_context("spawn")
        self._proc = None
        self._conn = None
        self.boot_error: str | None = None
        self.restarts = 0
        self._rebase_src: str | None = None    # replayed after every respawn so a restarted kernel keeps its business

    def start(self) -> None:
        parent, child = self._ctx.Pipe()
        self._proc = self._ctx.Process(target=worker_main, args=(child, self.bootstrap, self.workdir, self.mem_mb),
                                       daemon=True)
        self._proc.start()
        child.close()
        self._conn = parent
        if not parent.poll(self.start_timeout_s):
            self._kill()
            raise RuntimeError("sandbox worker did not start")
        hello = parent.recv()
        self.boot_error = hello.get("boot_error")
        if self._rebase_src is not None:
            self._send_rebase(self._rebase_src, 120.0)

    def _alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def _kill(self) -> None:
        if self._proc is not None:
            self._proc.kill()
            self._proc.join(timeout=5)
        self._proc, self._conn = None, None

    def run(self, code: str, timeout_s: float | None = None) -> ExecResult:
        if not self._alive():
            self.start()
            if not self._alive():
                return ExecResult(False, error="KernelDied: restart failed", elapsed=0.0)
        t0 = time.monotonic()
        limit = timeout_s or self.timeout_s
        self._conn.send({"op": "exec", "code": code, "max_output": self.max_output})
        if not self._conn.poll(limit):
            self._kill()
            self.restarts += 1
            return ExecResult(False, error=f"TimeoutError: cell exceeded {limit:.0f}s; kernel restarted, variables lost",
                              elapsed=time.monotonic() - t0, timed_out=True)
        try:
            r = self._conn.recv()
        except EOFError:
            self._kill()
            self.restarts += 1
            return ExecResult(False, error="KernelDied: worker crashed (likely MemoryError); kernel restarted",
                              elapsed=time.monotonic() - t0)
        return ExecResult(r["ok"], r["stdout"], r["value"], r["error"], r["final"], time.monotonic() - t0)

    def rebase(self, code: str, timeout_s: float = 120.0) -> str | None:
        """Run `code` in the bootstrap namespace and make it the new reset point. -> error or None."""
        self._rebase_src = None
        if not self._alive():
            self.start()
        err = self._send_rebase(code, timeout_s)
        self._rebase_src = code if err is None else None
        return err

    def _send_rebase(self, code: str, timeout_s: float) -> str | None:
        self._conn.send({"op": "rebase", "code": code})
        if not self._conn.poll(timeout_s):
            self._kill()
            self.restarts += 1
            return "rebase timed out"
        r = self._conn.recv()
        return r.get("error")

    def reset(self) -> None:
        if not self._alive():
            self.start()
            return
        self._conn.send({"op": "reset"})
        if not self._conn.poll(10):
            self._kill()
            self.start()
            return
        self._conn.recv()

    def close(self) -> None:
        if self._alive():
            try:
                self._conn.send({"op": "close"})
            except (BrokenPipeError, OSError):
                pass
            self._proc.join(timeout=3)
        self._kill()

    def __enter__(self) -> "Sandbox":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
