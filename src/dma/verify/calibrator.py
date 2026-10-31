"""Confidence calibration + PA-optimal abstention.

Expected score of answering = P(correct)*1 + (1-P)*(-1) = 2P - 1  -> answer iff P > 0.5.
We learn P(correct | features) on OUT-OF-FOLD predictions (leave-business-out CV),
then pick the threshold that maximises PA on those predictions, preferring the
middle of a flat optimum (robust to the private-test shift).

Inference needs only numpy: coefficients are stored as JSON (no pickle).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

FEATURES = [
    "agree_ab",            # Plan Solver and Code Solver produced the same canonical answer
    "vote_share",          # share of all candidates agreeing with the chosen one
    "n_candidates",
    "plan_ok",             # plan compiled & executed without repair
    "code_steps",          # number of CodeAct steps used
    "exec_errors",         # failed cells / repairs
    "entity_score",        # best entity-link score of the mentions used (0..1)
    "entity_margin",       # top1 - top2 link score
    "time_deterministic",  # time window came from the deterministic resolver
    "empty_or_zero",       # result empty / zero
    "tie_flag",            # tie at ranking boundary
    "answer_type_prior",   # out-of-fold accuracy of this answer type
    "intent_prior",        # out-of-fold accuracy of this intent/op
    "unanswerable_pred",   # chosen answer is 'unanswerable'
    "self_conf",           # model-stated confidence (0..1), weak
    "mean_logprob",        # mean token logprob of the plan JSON (if available)
    "bsm_quality",         # share of BSM validation rules V1..V9 passed (0..1)
    "rule_uncertain",      # a RuleSpec field used by this question is uncertain
    "judge_used",          # decision needed the pairwise judge
    "type_conflict",       # answer_type contradicts a strong type cue
    "heavy_flags",         # number of heavy sanity flags (05.7 §3) on the chosen candidate
    "light_flags",         # number of light flags (time not from resolver, close entity candidates, repaired)
]


@dataclass
class Calibrator:
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    mean: list[float] = field(default_factory=list)
    scale: list[float] = field(default_factory=list)
    coef: list[float] = field(default_factory=list)
    intercept: float = 0.0
    threshold: float = 0.5
    threshold_unanswerable: float = 0.6

    # ----------------------------------------------------------- inference
    def vector(self, feats: dict[str, float]) -> np.ndarray:
        return np.array([float(feats.get(f, 0.0) or 0.0) for f in self.features], dtype=float)

    def prob(self, feats: dict[str, float]) -> float:
        if not self.coef:
            # untrained fallback: agreement heuristic
            return 0.9 if feats.get("agree_ab") else 0.45
        x = (self.vector(feats) - np.array(self.mean)) / np.array(self.scale)
        z = float(np.dot(x, np.array(self.coef)) + self.intercept)
        return 1.0 / (1.0 + math.exp(-z))

    def decide(self, feats: dict[str, float]) -> tuple[bool, float]:
        p = self.prob(feats)
        thr = self.threshold_unanswerable if feats.get("unanswerable_pred") else self.threshold
        return p > thr, p

    # ------------------------------------------------------------ training
    def fit(self, X: list[dict[str, float]], y: list[int], C: float = 1.0) -> "Calibrator":
        from sklearn.linear_model import LogisticRegression
        A = np.vstack([self.vector(r) for r in X])
        mu, sd = A.mean(0), A.std(0)
        sd[sd == 0] = 1.0
        lr = LogisticRegression(C=C, max_iter=2000).fit((A - mu) / sd, np.array(y))
        self.mean, self.scale = mu.tolist(), sd.tolist()
        self.coef, self.intercept = lr.coef_[0].tolist(), float(lr.intercept_[0])
        return self

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=1)

    @classmethod
    def load(cls, path: str) -> "Calibrator":
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))


def pa_curve(probs: list[float], correct: list[int], grid: np.ndarray | None = None) -> list[tuple[float, float]]:
    """PA obtained when answering only items with p > t. `correct` = 1/0 for the
    answer we WOULD give (abstained items score 0)."""
    p, c = np.asarray(probs), np.asarray(correct)
    grid = np.linspace(0.0, 0.95, 96) if grid is None else grid
    n = len(p)
    return [(float(t), float(((2 * c - 1) * (p > t)).sum() / n)) for t in grid]


def choose_threshold(probs: list[float], correct: list[int], tol: float = 0.002) -> tuple[float, float]:
    """Middle of the plateau within `tol` of the best PA; never below 0.5."""
    curve = pa_curve(probs, correct)
    best = max(v for _, v in curve)
    good = [t for t, v in curve if v >= best - tol and t >= 0.5]
    if not good:
        return 0.5, best
    return float(np.median(good)), best


def expected_calibration_error(probs: list[float], correct: list[int], bins: int = 10) -> float:
    p, c = np.asarray(probs), np.asarray(correct)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.any():
            ece += m.mean() * abs(p[m].mean() - c[m].mean())
    return float(ece)
