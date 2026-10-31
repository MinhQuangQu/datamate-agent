"""Run configuration (configs/*.yaml) -> attribute-access tree.

Every business parameter comes from the split's RuleSpec, never from here; the config
only holds paths, budgets, thresholds and serving options (03 §6).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "version": 1,
    "run": {
        "data_root": "/data", "result_dir": "/result", "work_dir": "/tmp/dma",
        "profile": "accuracy", "soft_deadline_s": 18000, "hard_deadline_s": 20400,
        "max_concurrency": 48, "max_concurrent_builds": 4, "resume": False,
        "ingest_workers": 0,          # 0 -> min(16, cpu-2)
        "question_timeout_s": 900,    # per question; raise for rate-limited hosted APIs
    },
    "llm": {
        "enabled": True,
        "backend": "openai",          # openai (vLLM / any OpenAI-compatible server) | fake (tests only)
        "start_servers": True,        # supervisor launches vLLM; False -> use `endpoints`
        "endpoints": [],              # e.g. ["http://127.0.0.1:8001/v1"]
        "vllm_bin": "vllm",
        "model_path": "/opt/models/main", "served_name": "main",
        "replicas": [{"gpu": 0, "port": 8001}, {"gpu": 1, "port": 8002}],
        "serve_args": [], "env": {},
        "startup_timeout_s": 900, "request_timeout_s": 180, "max_retries": 2, "max_restarts": 2,
        "seed_base": 20261116,
        "thinking_style": "qwen",     # qwen (chat_template_kwargs.enable_thinking) | none
        "structured_output": "response_format",   # response_format | guided_json | none
        "sampling": {"thinking": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
                     "non_thinking": {"temperature": 0.0}},
        "max_tokens": {"rule": 6144, "bsm": 8192, "status_map": 1024, "plan": 3072, "plan_repair": 1024,
                       "code_first": 2048, "code_step": 768, "verify_first": 2048, "verify_step": 768,
                       "judge": 1536},
        "thinking": {"rule": True, "bsm": True, "status_map": False, "plan": True, "plan_repair": False,
                     "code_first": True, "code_step": False, "verify_first": True, "verify_step": False,
                     "judge": True},
        "max_inflight_per_replica": 64,
        # hosted OpenAI-compatible APIs (Groq, ...): key from this env var, params the provider rejects, client-side
        # rate limit (0 = off) and the longest single wait on HTTP 429 before giving up
        "api_key_env": "",
        "drop_params": [],
        "rate_limit": {"input_tokens_per_min": 0, "requests_per_min": 0},
        "max_rate_wait_s": 120,
    },
    "semantic": {"bsm_samples": 3, "min_fk_coverage": 0.95, "arithmetic_tolerance_vnd": 1,
                 "max_card_tokens": 3000, "use_llm": True},
    "rules": {"reference_rulebook": "assets/train_rulebook.md", "reference_rulespec": "assets/train_rulespec.json",
              "extraction_samples": 5, "min_vote_share": 0.8, "use_llm": True,
              "prompt_rulebook": "full"},   # full | relevant (rulebook rows retrieved per question)
    "nlu": {"link_threshold": 0.90, "possible_threshold": 0.60, "min_margin": 0.05},
    "memory": {"path": "assets/exemplars.jsonl", "top_k": 3, "max_per_template": 1, "exclude_businesses": []},
    "solvers": {
        "plan": {"enabled": True, "repairs": 1, "sql_timeout_s": 10},
        "code": {"enabled": True, "max_steps": 6, "cell_timeout_s": 20, "max_tokens_per_question": 9000,
                 "sandbox_pool": 24, "recycle_after": 200},
        "arbitration": {"extra_code_samples": 2, "extra_plan_samples": 1, "vote_margin": 0.8, "judge": True,
                        "verify": True},   # tool-using verifier agent on disagreement, judge is its fallback
    },
    "format": {"rounding": "python", "weekday_names": "assets/weekday_names.json",
               "max_list_len": 100, "max_table_rows": 200},
    "decision": {"calibrator": "assets/calibrator.json"},
    "logging": {"per_question": True, "max_observation_chars": 4000, "save_bkp": True},
}

PROFILES: dict[str, dict[str, Any]] = {
    "accuracy": {},
    "fast": {"solvers": {"code": {"max_steps": 4, "cell_timeout_s": 10, "max_tokens_per_question": 4000},
                         "arbitration": {"extra_code_samples": 1, "extra_plan_samples": 0}},
             "llm": {"max_tokens": {"code_first": 1024, "code_step": 512}}},
    "minimal": {"solvers": {"code": {"max_steps": 3},
                            "arbitration": {"extra_code_samples": 0, "extra_plan_samples": 0, "judge": False,
                                            "verify": False}}},
}


class Cfg(dict):
    """dict with attribute access; nested dicts become Cfg."""

    def __getattr__(self, k: str) -> Any:
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) and not isinstance(v, Cfg) else v

    def __setattr__(self, k: str, v: Any) -> None:
        self[k] = v


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _to_cfg(d: Any) -> Any:
    if isinstance(d, dict):
        return Cfg({k: _to_cfg(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_cfg(v) for v in d]
    return d


def load_config(path: str | None = None, overrides: dict | None = None) -> Cfg:
    data: dict = {}
    if path:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    merged = deep_merge(DEFAULTS, data)
    if overrides:
        merged = deep_merge(merged, overrides)
    merged["_path"] = os.path.abspath(path) if path else None
    merged["_app_root"] = app_root(path)
    return _to_cfg(merged)


def apply_profile(cfg: Cfg, profile: str) -> Cfg:
    merged = deep_merge(dict(cfg), PROFILES.get(profile, {}))
    merged["run"]["profile"] = profile
    return _to_cfg(merged)


def app_root(config_path: str | None) -> str:
    """Directory that relative asset paths resolve against: /app in the image, repo root in dev."""
    if os.environ.get("DMA_APP_ROOT"):
        return os.environ["DMA_APP_ROOT"]
    if config_path:
        d = os.path.dirname(os.path.abspath(config_path))
        return os.path.dirname(d) if os.path.basename(d) == "configs" else d
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def asset_path(cfg: Cfg, p: str | None) -> str | None:
    if not p:
        return None
    return p if os.path.isabs(p) else os.path.join(cfg["_app_root"], p)


def config_sha256(cfg: Cfg) -> str:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, default=str).encode()).hexdigest()
