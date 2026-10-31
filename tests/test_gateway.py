import asyncio

import pytest

from dma.config import load_config
from dma.llm.gateway import LLMGateway, content_seed, extract_json


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Plan:\n```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert extract_json('<think>hmm {x}</think>\nresult {"b": "}"} trailing') == {"b": "}"}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_content_seed_is_deterministic_and_id_free():
    assert content_seed("q", "2026-03-04", "plan", 0) == content_seed("q", "2026-03-04", "plan", 0)
    assert content_seed("q", "2026-03-04", "plan", 0) != content_seed("q", "2026-03-04", "plan", 1)


class Flaky:
    """Fails the first call, then returns a thinking-truncated answer, then a good one."""

    def __init__(self):
        self.calls = []

    async def complete(self, body):
        self.calls.append(body)
        if len(self.calls) == 1:
            raise ConnectionError("boom")
        if body.get("chat_template_kwargs", {}).get("enable_thinking"):
            return {"choices": [{"message": {"content": "", "reasoning_content": "long..."}, "finish_reason": "length"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 100}}
        return {"choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3}}


def test_retry_same_seed_and_thinking_fallback():
    cfg = load_config(None, {"llm": {"max_retries": 2}})
    be = Flaky()
    gw = LLMGateway(cfg, endpoints=[], backend=be)
    calls = []
    obj, err, resp = asyncio.run(gw.json("plan", [{"role": "user", "content": "x"}], schema={"type": "object"},
                                         seed=42, calls=calls))
    assert obj == {"ok": True} and err is None
    assert resp.fallback == "non_thinking_after_truncation"
    assert {b["seed"] for b in be.calls} == {42}                     # retries reuse the seed (D4)
    assert be.calls[0]["response_format"]["type"] == "json_schema"
    assert be.calls[-1]["temperature"] == 0.0                        # non-thinking = greedy
    assert calls and calls[0]["completion_tokens"] == 103


def test_disabled_gateway_reports_error():
    cfg = load_config(None, {"llm": {"enabled": False}})
    gw = LLMGateway(cfg, endpoints=[])
    obj, err, _ = asyncio.run(gw.json("plan", [], schema=None, seed=1))
    assert obj is None and "LLMUnavailable" in err


def test_failover_skips_unhealthy_replica():
    cfg = load_config(None, {})
    gw = LLMGateway(cfg, endpoints=["http://a/v1", "http://b/v1"])
    gw.mark_unhealthy(0, "test")
    assert gw.pick(0) == 1
