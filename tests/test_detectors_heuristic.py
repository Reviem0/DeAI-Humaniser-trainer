"""Golden parity of the heuristic engines, language detection and calibration with SlopTotal.

``fixtures/golden_heuristic.json`` was produced by running SlopTotal's own code
(its venv, at D:\\AI\\sloptotal) on ``fixtures/detector_texts.json`` plus the
extra texts stored in the golden file; SlopTotal is not needed here.
"""

import hashlib
import json
from pathlib import Path

import pytest

from deai.detectors import calibration, heuristic, language
from deai.detectors.heuristic import HEURISTIC_ENGINES

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = json.loads((FIXTURES / "golden_heuristic.json").read_text(encoding="utf-8"))
TEXTS = {
    t["id"]: t["text"]
    for t in json.loads((FIXTURES / "detector_texts.json").read_text(encoding="utf-8"))
}
TEXTS.update(GOLDEN["extra_texts"])
ENGINES = {engine.key: engine for engine in HEURISTIC_ENGINES}


def test_engine_metadata_matches_sloptotal():
    assert [e.key for e in HEURISTIC_ENGINES] == [
        "linguistic", "structural", "vocabulary", "formulaic", "readability", "sentiment"
    ]
    for key, meta in GOLDEN["engines_meta"].items():
        assert (ENGINES[key].name, ENGINES[key].engine_type) == (
            meta["name"], meta["engine_type"]
        )


def test_lists_are_verbatim():
    lists = GOLDEN["lists"]
    assert [list(m) for m in heuristic.AI_MARKERS] == lists["AI_MARKERS"]
    for name in ("AI_OPENERS", "AI_CLOSERS", "FILLER_PATTERNS", "HEDGING_PHRASES",
                 "BALANCE_PATTERNS", "VAGUE_POSITIVITY", "FILLER_QUALIFIERS"):
        assert getattr(heuristic, name) == lists[name], name
    assert language.FUNCTION_WORDS == lists["FUNCTION_WORDS"]


@pytest.mark.parametrize("rec", GOLDEN["texts"], ids=lambda r: r["id"])
def test_engine_scores(rec):
    text = TEXTS[rec["id"]]
    for key, expected in rec["engines"].items():
        raw = ENGINES[key].score(text)
        assert raw == pytest.approx(expected["raw"], abs=1e-9), key
        assert round(raw, 3) == expected["rounded"], key


@pytest.mark.parametrize("rec", GOLDEN["texts"], ids=lambda r: r["id"])
def test_language(rec):
    text = TEXTS[rec["id"]]
    assert language.detect_language(text) == rec["detect_language"]
    assert language.count_words(text) == rec["count_words"]
    assert language.is_unspaced(text) == rec["is_unspaced"]
    assert calibration.language_support(rec["detect_language"]) == rec["language_support"]
    assert sorted(calibration.scoring_engines(text)) == rec["scoring_engines"]


def test_formulaic_closer_quirk():
    """SlopTotal sets the closer when the opener's match contains "closer"."""
    assert ENGINES["formulaic"].score(TEXTS["x-closer-quirk"]) == pytest.approx(0.35)


@pytest.mark.parametrize(
    "case", GOLDEN["calibration"], ids=lambda c: f"{c['text_id']}-{len(c['engine_scores'])}"
)
def test_calibration(case):
    text = TEXTS[case["text_id"]]
    cut = calibration.fit_to_limit(text)
    scores = case["engine_scores"]
    assert list(calibration.calibrated_score(scores, cut)) == case["calibrated_score"]
    assert list(calibration.quick_score(scores, text)) == case["quick_score"]
    assert list(calibration.score_range(scores, cut)) == case["score_range"]
    assert calibration.quick_verdict(case["quick_score"][0]) == case["quick_verdict"]
    assert calibration.overall_verdict(case["calibrated_score"][0]) == case["overall_verdict"]


def test_confidence():
    for score, words, support, expected in GOLDEN["confidence"]:
        assert calibration.confidence(score, words, support) == expected


def test_verdicts():
    for score, (quick, overall) in GOLDEN["verdicts"].items():
        assert calibration.quick_verdict(float(score)) == quick
        assert calibration.overall_verdict(float(score)) == overall


@pytest.mark.parametrize("case", GOLDEN["fit_to_limit"], ids=lambda c: str(c["limit"]))
def test_fit_to_limit(case):
    text = case["text"] if "text" in case else TEXTS[case["text_id"]]
    cut = calibration.fit_to_limit(text, case["limit"])
    assert len(cut) == case["result_len"]
    assert hashlib.sha256(cut.encode("utf-8")).hexdigest() == case["result_sha256"]

