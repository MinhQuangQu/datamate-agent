"""OpenAI-compatible HTTP server backed by the FakeBackend TEST DOUBLE (dev/CI only).

Exercises the real HTTP path of the gateway (retries, health, failover) without a GPU:
    python scripts/fake_llm_server.py --port 18001 &
    python -m dma.run --config configs/dev.yaml --endpoints http://127.0.0.1:18001/v1 ...
Replace it by `vllm serve` (scripts/serve_llm.sh) to plug the real model in.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dma.llm.fake import FakeBackend  # noqa: E402

BACKEND = FakeBackend()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, obj: dict) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/health", "/v1/models"):
            self._send(200, {"ok": True, "data": [{"id": "main"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.endswith("/chat/completions"):
            return self._send(404, {"error": "not found"})
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        self._send(200, asyncio.run(BACKEND.complete(body)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18001)
    a = ap.parse_args()
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
