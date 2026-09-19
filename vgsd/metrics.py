"""Medical VQA metrics in the LLaVA-Med style.

closed accuracy : yes/no answer judged correct if the ground-truth word appears in the prediction.
                  mode="word"      -> whole-word match (default; "no" does not match "normal")
                  mode="substring" -> LLaVA-Med's lenient `gt in pred` string test
                  mode="first"     -> first yes/no word in the prediction must equal gt
open recall     : token-level recall |pred ∩ gt| / |gt| after normalisation (multiset)
overall         : sample-weighted average of the two (reproduces the paper's Overall column:
                  (200*34.45 + 251*68.92)/451 = 53.64 for LLaVA-Med greedy on VQA-RAD)
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_PUNCT = re.compile(r"[^\w\s]")


def normalize(text: str) -> list[str]:
    text = _PUNCT.sub(" ", str(text).lower())
    return text.split()


def closed_correct(pred: str, gt: str, mode: str = "word") -> float:
    gt_n = " ".join(normalize(gt))
    toks = normalize(pred)
    if mode == "substring":
        return float(gt_n in " ".join(toks))
    if mode == "word":
        return float(gt_n in toks)
    if mode == "first":
        for t in toks:
            if t in ("yes", "no"):
                return float(t == gt_n)
        return 0.0
    raise ValueError(mode)


def open_recall(pred: str, gt: str) -> float:
    g = Counter(normalize(gt))
    if not g:
        return 0.0
    p = Counter(normalize(pred))
    return sum(min(c, p[t]) for t, c in g.items()) / sum(g.values())


def per_sample_scores(records, closed_mode="word"):
    """records: dicts with pred, answer, is_closed. Returns np.array of scores in [0,1]."""
    return np.array([closed_correct(r["pred"], r["answer"], closed_mode) if r["is_closed"]
                     else open_recall(r["pred"], r["answer"]) for r in records], dtype=float)


def summarize(records, closed_mode="word"):
    s = per_sample_scores(records, closed_mode)
    closed = np.array([r["is_closed"] for r in records], dtype=bool)
    res = {"n": len(records), "n_open": int((~closed).sum()), "n_closed": int(closed.sum()),
           "open": 100 * s[~closed].mean() if (~closed).any() else float("nan"),
           "closed": 100 * s[closed].mean() if closed.any() else float("nan"),
           "overall": 100 * s.mean() if len(s) else float("nan")}
    return res


def mcnemar_exact(a_correct, b_correct) -> float:
    """Two-sided exact McNemar p-value on binary correctness vectors."""
    a, b = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired_bootstrap(a_scores, b_scores, n_boot=10000, seed=0) -> float:
    """One-sided p-value that mean(b) > mean(a), resampling questions with replacement."""
    a, b = np.asarray(a_scores, float), np.asarray(b_scores, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = b[idx].mean(1) - a[idx].mean(1)
    return float((diffs <= 0).mean())
