"""LLM Gateway (05.8 §4): one async client for every stage, OpenAI-compatible
`/chat/completions` (vLLM serve). Explicit sampling, content-derived seeds, retry with the
SAME seed, replica failover, token accounting, thinking-truncation fallback, JSON parsing.

Plugging in a model = pointing `llm.endpoints` (or the supervisor) at a server that speaks
the OpenAI chat API with `served_name` as model name. Nothing else depends on the model.
"""
from __future__ import annotations

import asyncio
import os
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from dma.config import Cfg


def content_seed(*parts: Any, base: int = 0) -> int:
    """Seed from CONTENT (question text, as_of, stage, attempt) — never from ids (Rule 4)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return (int(h[:8], 16) + base) % (2 ** 31 - 1)


@dataclass
class LLMResponse:
    text: str
    reasoning: str = ""
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    replica: int = 0
    seed: int = 0
    stage: str = ""
    thinking: bool = False
    fallback: str | None = None

    def log(self) -> dict:
        d = asdict(self)
        d.pop("text")
        d.pop("reasoning")
        return d


@dataclass
class ReplicaState:
    url: str
    healthy: bool = True
    inflight: int = 0
    failures: int = 0
    calls: int = 0


class LLMUnavailable(RuntimeError):
    pass


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> Any:
    """First JSON object/array in `text` (fenced or bare). Raises ValueError."""
    if text is None:
        raise ValueError("empty completion")
    s = text.strip()
    s = re.sub(r"^<think>.*?</think>", "", s, flags=re.S).strip()
    for cand in [m.group(1) for m in _FENCE.finditer(s)] + [s]:
        cand = cand.strip()
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
        start = min([i for i in (cand.find("{"), cand.find("[")) if i >= 0], default=-1)
        if start < 0:
            continue
        depth, instr, esc = 0, False, False
        opener = cand[start]
        closer = "}" if opener == "{" else "]"
        for i in range(start, len(cand)):
            ch = cand[i]
            if instr:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    instr = False
                continue
            if ch == '"':
                instr = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cand[start:i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError("no JSON object found in completion")


def _repetitive(s: str, n: int = 40, k: int = 4) -> bool:
    if len(s) < n * k:
        return False
    tail = s[-n:]
    return s.count(tail) >= k


def _approx_tokens(body: dict) -> int:
    """Prompt size estimate for the client-side limiter (Vietnamese ≈ 3.2 chars/token on Qwen tokenizers)."""
    return int(sum(len(str(m.get("content") or "")) for m in body.get("messages", [])) / 3.2) + 16


def _retry_after(resp) -> float:
    h = resp.headers.get("retry-after")
    try:
        if h:
            return max(1.0, float(h))
    except ValueError:
        pass
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", resp.text or "")
    return (60.0 * int(m.group(1) or 0) + float(m.group(2)) + 0.5) if m else 10.0


class RateLimiter:
    """Sliding one-minute window on input tokens and requests (0 disables a limit)."""

    def __init__(self, tokens_per_min: float = 0, requests_per_min: float = 0):
        self.tpm, self.rpm = tokens_per_min, requests_per_min
        self.events: list[tuple[float, int]] = []
        self.lock = asyncio.Lock()

    async def acquire(self, tokens: int) -> None:
        if not (self.tpm or self.rpm):
            return
        async with self.lock:
            while True:
                now = time.monotonic()
                self.events = [e for e in self.events if now - e[0] < 60.0]
                used = sum(t for _, t in self.events)
                ok_t = not self.tpm or used + tokens <= self.tpm or not self.events
                ok_r = not self.rpm or len(self.events) < self.rpm
                if ok_t and ok_r:
                    self.events.append((now, tokens))
                    return
                await asyncio.sleep(max(0.2, 60.0 - (now - self.events[0][0])))


class LLMGateway:
    def __init__(self, cfg: Cfg, endpoints: list[str] | None = None, backend: Any = None):
        self.cfg = cfg
        lc = cfg.llm
        eps = endpoints if endpoints is not None else list(lc.endpoints or [])
        self.replicas = [ReplicaState(u.rstrip("/")) for u in eps]
        self.backend = backend                     # in-process backend (tests/fake), else HTTP
        self.enabled = bool(lc.enabled) and (bool(self.replicas) or backend is not None)
        self._client = None
        self._sems: list[asyncio.Semaphore] = []
        self.stats: dict[str, dict[str, int]] = {}
        self.model = lc.served_name
        self.on_event: Callable[[str, dict], None] | None = None
        self._limiter: RateLimiter | None = None
        key_env = str(lc.api_key_env or "")
        self._headers = {"Authorization": f"Bearer {os.environ[key_env]}"} if key_env and os.environ.get(key_env) else {}

    @property
    def n_replicas(self) -> int:
        return max(1, len(self.replicas)) if self.backend is None else 1

    async def _ensure(self) -> None:
        if self.backend is None and self._client is None:
            import httpx
            limits = httpx.Limits(max_connections=256, max_keepalive_connections=64)
            self._client = httpx.AsyncClient(timeout=self.cfg.llm.request_timeout_s, limits=limits)
        if not self._sems:
            self._sems = [asyncio.Semaphore(int(self.cfg.llm.max_inflight_per_replica)) for _ in range(self.n_replicas)]
        if self._limiter is None:
            rl = dict(self.cfg.llm.rate_limit or {})
            self._limiter = RateLimiter(float(rl.get("input_tokens_per_min") or 0), float(rl.get("requests_per_min") or 0))

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------ routing
    def pick(self, preferred: int) -> int:
        if self.backend is not None or not self.replicas:
            return 0
        n = len(self.replicas)
        r = preferred % n
        if not self.replicas[r].healthy:
            alive = [i for i in range(n) if self.replicas[i].healthy]
            if not alive:
                raise LLMUnavailable("no healthy LLM replica")
            r = alive[0]
        if n > 1:   # spill when the sticky replica is much busier than the other
            other = min((i for i in range(n) if i != r and self.replicas[i].healthy),
                        key=lambda i: self.replicas[i].inflight, default=None)
            if other is not None and self.replicas[r].inflight > 1.5 * self.replicas[other].inflight + 8:
                r = other
        return r

    def mark_unhealthy(self, r: int, why: str) -> None:
        if 0 <= r < len(self.replicas):
            self.replicas[r].healthy = False
            if self.on_event:
                self.on_event("replica_unhealthy", {"replica": r, "why": why})

    def mark_healthy(self, r: int) -> None:
        if 0 <= r < len(self.replicas):
            self.replicas[r].healthy = True

    # ------------------------------------------------------------ calls
    def _body(self, stage: str, messages: list[dict], seed: int, max_tokens: int, thinking: bool,
              json_schema: dict | None) -> dict:
        lc = self.cfg.llm
        samp = dict(lc.sampling.thinking if thinking else lc.sampling.non_thinking)
        body: dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": int(max_tokens),
                                "seed": int(seed), **samp}
        if lc.thinking_style == "qwen":
            body["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
        if json_schema is not None:
            mode = lc.structured_output
            if mode == "response_format":
                body["response_format"] = {"type": "json_schema",
                                           "json_schema": {"name": stage, "schema": json_schema, "strict": False}}
            elif mode == "guided_json":
                body["guided_json"] = json_schema
        for k in lc.drop_params or []:
            body.pop(k, None)
        return body

    async def _post(self, r: int, body: dict) -> dict:
        if self.backend is not None:
            return await self.backend.complete(body)
        url = f"{self.replicas[r].url}/chat/completions"
        waited = 0.0
        while True:
            await self._limiter.acquire(_approx_tokens(body))
            resp = await self._client.post(url, json=body, headers=self._headers)
            if resp.status_code != 429:
                break
            # provider rate limit: wait what it asks (Retry-After / "try again in 31.6s"), not a replica failure
            wait = _retry_after(resp)
            if waited + wait > float(self.cfg.llm.max_rate_wait_s) * 10:
                break
            if self.on_event:
                self.on_event("llm_rate_limited", {"wait_s": round(wait, 1)})
            waited += wait
            await asyncio.sleep(wait)
        resp.raise_for_status()
        return resp.json()

    async def chat(self, stage: str, messages: list[dict], *, seed: int, replica: int = 0,
                   max_tokens: int | None = None, thinking: bool | None = None, json_schema: dict | None = None,
                   calls: list | None = None) -> LLMResponse:
        if not self.enabled:
            raise LLMUnavailable("LLM disabled")
        await self._ensure()
        lc = self.cfg.llm
        thinking = bool(lc.thinking.get(stage, False)) if thinking is None else thinking
        max_tokens = int(max_tokens or lc.max_tokens.get(stage, 1024))
        resp = await self._call(stage, messages, seed, replica, max_tokens, thinking, json_schema)
        if thinking and (not resp.text.strip() or resp.finish_reason == "length" and json_schema is not None
                         or _repetitive(resp.reasoning)):
            # thinking ran out of budget / looped: one non-thinking retry (05.8 §4, 06 §5)
            hint = resp.reasoning[-1500:] if resp.reasoning else ""
            msgs = list(messages)
            if hint:
                msgs = msgs + [{"role": "assistant", "content": ""},
                               {"role": "user", "content": "Your reasoning was cut. Notes so far:\n" + hint +
                                "\nNow answer directly in the required format."}]
            r2 = await self._call(stage, msgs, seed, replica, max_tokens, False, json_schema)
            r2.fallback = "non_thinking_after_truncation"
            r2.prompt_tokens += resp.prompt_tokens
            r2.completion_tokens += resp.completion_tokens
            resp = r2
        if calls is not None:
            calls.append(resp.log())
        st = self.stats.setdefault(stage, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
        st["calls"] += 1
        st["prompt_tokens"] += resp.prompt_tokens
        st["completion_tokens"] += resp.completion_tokens
        return resp

    async def _call(self, stage, messages, seed, replica, max_tokens, thinking, json_schema) -> LLMResponse:
        lc = self.cfg.llm
        body = self._body(stage, messages, seed, max_tokens, thinking, json_schema)
        last_err: Exception | None = None
        for attempt in range(int(lc.max_retries) + 1):
            try:
                r = self.pick(replica)
            except LLMUnavailable:
                # no supervisor re-marks replicas when the servers are external: probe them before giving up
                if not any(await self.health()):
                    raise
                r = self.pick(replica)
            t0 = time.monotonic()
            sem = self._sems[r % len(self._sems)]
            try:
                async with sem:
                    if self.replicas:
                        self.replicas[r].inflight += 1
                    try:
                        data = await self._post(r, body)
                    finally:
                        if self.replicas:
                            self.replicas[r].inflight -= 1
                            self.replicas[r].calls += 1
                ch = (data.get("choices") or [{}])[0]
                msg = ch.get("message") or {}
                usage = data.get("usage") or {}
                text = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                if not reasoning and "<think>" in text:
                    m = re.match(r"\s*<think>(.*?)(</think>|$)(.*)", text, re.S)
                    if m:
                        reasoning, text = m.group(1), m.group(3)
                return LLMResponse(text=text, reasoning=reasoning, finish_reason=ch.get("finish_reason"),
                                   prompt_tokens=int(usage.get("prompt_tokens") or 0),
                                   completion_tokens=int(usage.get("completion_tokens") or 0),
                                   latency_ms=int(1000 * (time.monotonic() - t0)), replica=r, seed=seed,
                                   stage=stage, thinking=thinking)
            except LLMUnavailable:
                raise
            except Exception as e:  # noqa: BLE001 - transport / HTTP errors
                last_err = e
                if self.replicas:
                    self.replicas[r].failures += 1
                    if _is_transport(e) and self.replicas[r].failures >= 2:
                        self.mark_unhealthy(r, repr(e)[:200])
                if attempt < int(lc.max_retries):
                    await asyncio.sleep(min(8, 2 ** attempt))
        raise LLMUnavailable(f"LLM call failed after retries: {last_err!r}")

    async def json(self, stage: str, messages: list[dict], *, schema: dict | None, seed: int, replica: int = 0,
                   validate: Callable[[Any], Any] | None = None, calls: list | None = None,
                   max_tokens: int | None = None, thinking: bool | None = None) -> tuple[Any, str | None, LLMResponse | None]:
        """-> (validated object | None, error message | None, response)."""
        try:
            resp = await self.chat(stage, messages, seed=seed, replica=replica, max_tokens=max_tokens,
                                   thinking=thinking, json_schema=schema, calls=calls)
        except LLMUnavailable as e:
            return None, f"LLMUnavailable: {e}", None
        try:
            obj = extract_json(resp.text)
            return (validate(obj) if validate else obj), None, resp
        except Exception as e:  # noqa: BLE001 - pydantic ValidationError / ValueError
            return None, f"{type(e).__name__}: {str(e)[:1000]}", resp

    async def health(self, timeout: float = 5.0) -> list[bool]:
        if self.backend is not None:
            return [True]
        await self._ensure()
        out = []
        for i, rep in enumerate(self.replicas):
            base = rep.url[:-3] if rep.url.endswith("/v1") else rep.url
            try:
                r = await self._client.get(f"{base}/health", timeout=timeout)
                ok = r.status_code == 200
                if not ok:      # hosted APIs have no /health: the model list answers when the key is valid
                    r = await self._client.get(f"{rep.url}/models", timeout=timeout, headers=self._headers)
                    ok = r.status_code == 200
            except Exception:  # noqa: BLE001
                ok = False
            (self.mark_healthy if ok else (lambda j: None))(i)
            out.append(ok)
        return out


def _is_transport(e: Exception) -> bool:
    """A failure that says the replica is down. A read timeout only says it is busy (long thinking under load)."""
    try:
        import httpx
        if isinstance(e, httpx.ReadTimeout):
            return False
        return isinstance(e, (httpx.TransportError, httpx.TimeoutException)) or \
            (isinstance(e, httpx.HTTPStatusError) and e.response.status_code >= 500)
    except ImportError:
        return True


@dataclass
class CallLog:
    calls: list = field(default_factory=list)
