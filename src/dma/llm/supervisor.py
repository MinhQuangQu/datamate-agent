"""vLLM supervisor (05.8 §1–2): one `vllm serve` per GPU (data parallel by hand), started
without blocking so ingestion overlaps model load; health checks; watchdog restarts a dead
replica (<= max_restarts); with one visible GPU only one replica is started (T5)."""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
import urllib.request

from dma.config import Cfg


def visible_gpus() -> int | None:
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None and env.strip() != "":
        return len([x for x in env.split(",") if x.strip() and x.strip() != "-1"])
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20).stdout
        return sum(1 for ln in out.splitlines() if ln.startswith("GPU "))
    except (OSError, subprocess.SubprocessError):
        return None


class Replica:
    def __init__(self, idx: int, gpu: str, port: int, cmd: list[str], env: dict, log_path: str):
        self.idx, self.gpu, self.port, self.cmd, self.env, self.log_path = idx, gpu, port, cmd, env, log_path
        self.proc: subprocess.Popen | None = None
        self.restarts = 0
        self.ready = False

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def start(self) -> None:
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        log = open(self.log_path, "ab", buffering=0)
        self.proc = subprocess.Popen(self.cmd, env=self.env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        self.ready = False

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def healthy(self, timeout: float = 3.0) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=timeout) as r:
                return r.status == 200
        except Exception:  # noqa: BLE001
            return False

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, 15)
                self.proc.wait(timeout=20)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(self.proc.pid, 9)
                except Exception:  # noqa: BLE001
                    pass


class Supervisor:
    def __init__(self, cfg: Cfg, log_dir: str, on_event=None):
        self.cfg = cfg
        self.log_dir = log_dir
        self.on_event = on_event or (lambda *a, **k: None)
        self.replicas: list[Replica] = []
        self._stop = threading.Event()
        self._watch: threading.Thread | None = None
        self.gateway = None

    def start(self) -> list[str]:
        lc = self.cfg.llm
        reps = list(lc.replicas)
        n = visible_gpus()
        if n is not None and 0 < n < len(reps):
            self.on_event("gpu_fallback", {"visible": n, "configured": len(reps)})
            reps = reps[:n]
        env_base = dict(os.environ)
        env_base.update({str(k): str(v) for k, v in (lc.env or {}).items()})
        env_base.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1"})
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        ids = [x.strip() for x in cvd.split(",")] if cvd else None
        for i, r in enumerate(reps):
            gpu = ids[i] if ids and i < len(ids) else str(r["gpu"])
            cmd = [lc.vllm_bin, "serve", lc.model_path, "--served-model-name", lc.served_name, "--host", "127.0.0.1",
                   "--port", str(r["port"]), *[str(a) for a in (lc.serve_args or [])]]
            env = dict(env_base, CUDA_VISIBLE_DEVICES=gpu)
            rep = Replica(i, gpu, int(r["port"]), cmd, env, os.path.join(self.log_dir, f"vllm_gpu{i}.log"))
            rep.start()
            self.on_event("vllm_start", {"replica": i, "gpu": gpu, "port": rep.port, "cmd": cmd})
            self.replicas.append(rep)
        return [r.url for r in self.replicas]

    def wait_ready(self, timeout: float) -> list[bool]:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            for r in self.replicas:
                if not r.ready and r.healthy():
                    r.ready = True
                    self.on_event("vllm_ready", {"replica": r.idx, "after_s": round(time.monotonic() - t0, 1)})
            if all(r.ready or not r.alive() for r in self.replicas):
                break
            time.sleep(3)
        return [r.ready for r in self.replicas]

    def watch(self, gateway, interval: float = 15.0) -> None:
        self.gateway = gateway
        self._watch = threading.Thread(target=self._loop, args=(interval,), daemon=True)
        self._watch.start()

    def _loop(self, interval: float) -> None:
        max_restarts = int(self.cfg.llm.max_restarts)
        while not self._stop.wait(interval):
            for r in self.replicas:
                if r.alive() and r.healthy():
                    if self.gateway:
                        self.gateway.mark_healthy(r.idx)
                    continue
                if self.gateway:
                    self.gateway.mark_unhealthy(r.idx, "health check failed")
                if not r.alive() and r.restarts < max_restarts:
                    r.restarts += 1
                    self.on_event("vllm_restart", {"replica": r.idx, "n": r.restarts})
                    r.start()

    def stop_all(self) -> None:
        self._stop.set()
        for r in self.replicas:
            r.stop()
