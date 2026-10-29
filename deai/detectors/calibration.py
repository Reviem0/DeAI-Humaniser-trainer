"""SlopTotal's overall score: a logistic model over engine scores, fitted on SlopBench.

Exact port of ``app/calibration.py`` from SlopTotal
(https://github.com/pablocaeg/sloptotal, MIT License, Copyright (c) 2025-2026
SlopTotal), plus the pieces of ``app/analyzer.py``, ``app/schemas.py`` and
``app/config.py`` that turn its output into what the API returns: the text cut
(``fit_to_limit``) and the verdict strings. ``calibration.json`` is SlopTotal's
file copied unchanged; it has "english" and "multilingual" sections for the full
report and a "quick" section for /api/quick-score.

Engine-score dicts map SlopTotal engine keys to 0-1 scores, already rounded to
3 decimals as SlopTotal's engines report them; a missing engine counts as 0.5.
Pass the ``fit_to_limit``-cut text, as SlopTotal does for full analyses.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from deai.detectors.language import count_words, detect_language, is_unspaced

_CALIBRATION = json.loads(
    (Path(__file__).parent / "calibration.json").read_text(encoding="utf-8")
)
SHORT_TEXT_WORDS = 80
UNCERTAIN = (35.0, 65.0)
CONFIDENT = (15.0, 80.0)

# app/config.py: longer submissions are cut at a word boundary before scoring.
MAX_ANALYSED_CHARS = 100_000
# app/config.py: full-report verdict bands and the quick path's three states.
SCORE_CLEAN = 30
SCORE_LOW_RISK = 45
SCORE_SUSPICIOUS = 55
SCORE_LIKELY_AI = 80
QUICK_CLEAN_MAX = 35
QUICK_AI_MIN = 65


def fit_to_limit(text: str, limit: int = MAX_ANALYSED_CHARS) -> str:
    """Cut text longer than limit at the last whitespace before it (app/analyzer.py)."""
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    return text[: cut if cut > limit // 2 else limit].rstrip()


def _logit(p: float) -> float:
    p = min(max(p, 0.01), 0.99)
    return math.log(p / (1 - p))


def _weighted(section: dict, engine_scores: dict[str, float]) -> float:
    return section["intercept"] + sum(
        weight * _logit(engine_scores.get(key, 0.5))
        for key, weight in section["weights"].items()
    )


def _interpolate(z: float, knots: list[list[float]]) -> float:
    """Piecewise-linear map from the model's log-odds to the 0-100 scale."""
    if z <= knots[0][0]:
        return knots[0][1]
    for (z0, s0), (z1, s1) in zip(knots, knots[1:], strict=False):
        if z <= z1:
            return s0 + (s1 - s0) * (z - z0) / (z1 - z0)
    return knots[-1][1]


def scoring_engines(text: str) -> frozenset[str]:
    """The engines with a non-zero weight in the full-report fit for this text."""
    section = "english" if detect_language(text) == "en" else "multilingual"
    weights = _CALIBRATION[section]["weights"]
    return frozenset(key for key, weight in weights.items() if weight > 0)


def score_range(engine_scores: dict[str, float], text: str) -> tuple[float, float]:
    """Lowest and highest overall score still possible while some scoring
    engines have not reported (missing ones at 0 and at 1)."""
    missing = scoring_engines(text) - engine_scores.keys()
    low, _, _ = calibrated_score({**engine_scores, **dict.fromkeys(missing, 0.0)}, text)
    high, _, _ = calibrated_score(
        {**engine_scores, **dict.fromkeys(missing, 1.0)}, text
    )
    return low, high


def language_support(lang: str) -> str:
    """How far detection is trusted in a language: supported, experimental or unsupported."""
    if lang == "en":
        return "supported"
    return (
        _CALIBRATION["multilingual"]["languages"]
        .get(lang, {})
        .get("status", "unsupported")
    )


def calibrated_score(
    engine_scores: dict[str, float], text: str
) -> tuple[float, str, str]:
    """Full-report 0-100 score (1 dp), confidence label and detected language.

    English: weighted log-odds pulled toward neutral by ``words / (words + k)``,
    then mapped through the band knots. Other languages: the multilingual fit
    plus a per-language offset, a short-text penalty for spaced scripts, and a
    sigmoid.
    """
    lang = detect_language(text)
    words = count_words(text)
    if lang == "en":
        english = _CALIBRATION["english"]
        z = _weighted(english, engine_scores)
        k = english["length_k"]
        reliability = words / (words + k) if k else 1.0
        z = reliability * z + (1 - reliability) * english["length_neutral"]
        score = _interpolate(z, english["band_knots"])
    else:
        multilingual = _CALIBRATION["multilingual"]
        offset = (
            multilingual["languages"]
            .get(lang, {})
            .get("offset", multilingual["default_offset"])
        )
        z = _weighted(multilingual, engine_scores) + offset
        reference = multilingual["short_ref_words"]
        if not is_unspaced(text) and 0 < words < reference:
            z -= multilingual["short_alpha"] * (1 / words - 1 / reference)
        score = 100 / (1 + math.exp(-z))

    return round(score, 1), confidence(score, words, language_support(lang)), lang


def confidence(score: float, words: int, support: str) -> str:
    """Never above "low" for a short text, a score in the uncertain middle, or a
    language whose detection is not supported. ``score`` is the unrounded one."""
    if (
        words < SHORT_TEXT_WORDS
        or support == "unsupported"
        or UNCERTAIN[0] <= score <= UNCERTAIN[1]
    ):
        return "low"
    if support == "experimental" or not (score < CONFIDENT[0] or score >= CONFIDENT[1]):
        return "medium"
    return "high"


def quick_score(engine_scores: dict[str, float], text: str) -> tuple[float, str]:
    """0-100 score (1 dp) and confidence for /api/quick-score, from the quick
    path's four classifiers and two heuristics. No non-English fit; confidence
    follows the language's support. SlopTotal does not cut this text."""
    quick = _CALIBRATION["quick"]
    words = count_words(text)
    z = _weighted(quick, engine_scores)
    k = quick["length_k"]
    reliability = words / (words + k) if k else 1.0
    z = reliability * z + (1 - reliability) * quick["length_neutral"]
    score = _interpolate(z, quick["band_knots"])
    support = language_support(detect_language(text))
    return round(score, 1), confidence(score, words, support)


def quick_verdict(score: float) -> str:
    """The quick path's three states: "clean", "mixed" or "ai" (app/analyzer.py)."""
    if score <= QUICK_CLEAN_MAX:
        return "clean"
    if score <= QUICK_AI_MIN:
        return "mixed"
    return "ai"


def overall_verdict(score: float) -> str:
    """The full report's ``overall_verdict`` band (app/schemas.py score_to_verdict_str)."""
    if score <= SCORE_CLEAN:
        return "Clean — likely human-written"
    elif score <= SCORE_LOW_RISK:
        return "Low risk"
    elif score <= SCORE_SUSPICIOUS:
        return "Suspicious"
    elif score <= SCORE_LIKELY_AI:
        return "Likely AI-generated"
    else:
        return "Slop detected"
