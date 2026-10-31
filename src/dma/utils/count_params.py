"""Count parameters of every model shipped in the image, from safetensors HEADERS only
(no weights loaded). Rule 2: the sum over ALL models used at inference must be ≤ 15B
(MoE: total params; models not loaded simultaneously still count).

    python -m dma.utils.count_params /opt/models/main /opt/models/lora_x --limit 15e9 --json params.json

Notes
* Count the ORIGINAL bf16/fp16 checkpoint: packed int4 tensors (AWQ/GPTQ) under-count.
* Tied embeddings: if lm_head is absent from the files it is tied -> counted once (correct).
* Vision towers / MTP heads shipped in the checkpoint are counted (conservative).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import struct
import sys
from collections import defaultdict

PACKED = {"I32", "U32", "I8", "U8", "I4", "U4"}


def read_header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def count_dir(model_dir: str, group_depth: int = 2) -> dict:
    files = sorted(glob.glob(os.path.join(model_dir, "**", "*.safetensors"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no .safetensors under {model_dir}")
    total, groups, dtypes, packed = 0, defaultdict(int), defaultdict(int), 0
    for fp in files:
        for name, meta in read_header(fp).items():
            if name == "__metadata__":
                continue
            n = math.prod(meta["shape"]) if meta["shape"] else 1
            total += n
            groups[".".join(name.split(".")[:group_depth])] += n
            dtypes[meta["dtype"]] += n
            if meta["dtype"] in PACKED and not name.endswith(("scales", "zeros", "g_idx")):
                packed += n
    return {"dir": model_dir, "files": len(files), "total": total,
            "groups": dict(sorted(groups.items(), key=lambda kv: -kv[1])), "dtypes": dict(dtypes),
            "warning": "packed integer tensors found: count the unquantized checkpoint" if packed else None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--limit", type=float, default=15e9)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    reports = [count_dir(d) for d in a.dirs]
    grand = sum(r["total"] for r in reports)
    for r in reports:
        print(f"{r['dir']}: {r['total']:,} params ({r['total'] / 1e9:.3f} B) in {r['files']} files")
        for g, n in list(r["groups"].items())[:6]:
            print(f"    {g:<40} {n / 1e9:8.3f} B")
        if r["warning"]:
            print("    WARNING:", r["warning"])
    ok = grand <= a.limit
    print(f"TOTAL = {grand:,} ({grand / 1e9:.3f} B)  limit {a.limit / 1e9:.1f} B  ->  {'OK' if ok else 'OVER LIMIT'}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"total": grand, "limit": a.limit, "ok": ok, "models": reports}, f, indent=1)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
