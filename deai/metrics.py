"""Small detector metrics shared by the scripts."""

from __future__ import annotations

import bisect
import math


def auroc(pos: list[float], neg: list[float]) -> float:
    """P(random positive score > random negative score); ties count half."""
    if not pos or not neg:
        return math.nan
    neg_sorted = sorted(neg)
    wins = 0.0
    for p in pos:
        lo = bisect.bisect_left(neg_sorted, p)
        hi = bisect.bisect_right(neg_sorted, p)
        wins += lo + 0.5 * (hi - lo)
    return wins / (len(pos) * len(neg))


def threshold_at_fpr(neg: list[float], fpr: float) -> float:
    """Smallest threshold t with share(neg >= t) <= fpr ("flag if score >= t")."""
    s = sorted(neg, reverse=True)
    k = int(math.floor(fpr * len(s)))  # this many negatives may be flagged
    return s[k] + 1e-9 if k < len(s) else -math.inf


def tpr_at_fpr(pos: list[float], neg: list[float], fpr: float = 0.01) -> float:
    if not pos or not neg:
        return math.nan
    t = threshold_at_fpr(neg, fpr)
    return sum(p >= t for p in pos) / len(pos)
