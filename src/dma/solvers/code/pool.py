"""Async pool of sandbox kernels with business affinity (05.8 §6): acquire(biz) prefers an
idle kernel already bound to that business, else rebinds the least-recently used one."""
from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager

from dma.solvers.code.sandbox import Sandbox
from dma.solvers.code.toolkit import BOOTSTRAP_BASE, rebase_code


class SandboxPool:
    def __init__(self, size: int, work_dir: str, cell_timeout_s: float = 20.0, recycle_after: int = 200,
                 mem_mb: int | None = None):
        self.size = max(1, size)
        self.work_dir = work_dir
        self.cell_timeout_s = cell_timeout_s
        self.recycle_after = recycle_after
        self.mem_mb = mem_mb
        self._idle: list[dict] = []
        self._all: list[dict] = []
        self._cond = asyncio.Condition()
        self._created = 0

    def _new(self) -> dict:
        i = self._created
        self._created += 1
        sb = Sandbox(bootstrap=BOOTSTRAP_BASE, workdir=os.path.join(self.work_dir, f"w{i}"), timeout_s=self.cell_timeout_s,
                     mem_mb=self.mem_mb)
        slot = {"sb": sb, "biz": None, "uses": 0, "last": 0.0, "id": i}
        self._all.append(slot)
        return slot

    @asynccontextmanager
    async def acquire(self, business_id: str, bkp_dir: str, split_dir: str):
        async with self._cond:
            while True:
                slot = next((s for s in self._idle if s["biz"] == business_id), None)
                if slot is None and len(self._all) < self.size:
                    slot = self._new()
                elif slot is None and self._idle:
                    slot = min(self._idle, key=lambda s: s["last"])
                if slot is not None:
                    if slot in self._idle:
                        self._idle.remove(slot)
                    break
                await self._cond.wait()
        try:
            sb: Sandbox = slot["sb"]
            if slot["uses"] >= self.recycle_after:
                await asyncio.to_thread(sb.close)
                slot["biz"], slot["uses"] = None, 0
            if slot["biz"] != business_id or not sb._alive():
                err = await asyncio.to_thread(sb.rebase, rebase_code(bkp_dir, split_dir))
                if err:
                    raise RuntimeError(f"sandbox bootstrap failed: {err[-800:]}")
                slot["biz"] = business_id
            else:
                await asyncio.to_thread(sb.reset)
            slot["uses"] += 1
            yield sb
        finally:
            slot["last"] = time.monotonic()
            if not slot["sb"]._alive():
                slot["biz"] = None      # killed by a timeout: rebind next time
            async with self._cond:
                self._idle.append(slot)
                self._cond.notify()

    async def close(self) -> None:
        for s in self._all:
            await asyncio.to_thread(s["sb"].close)
