"""deai.detectors: the registry with stub engines, and end-to-end parity with SlopTotal.

golden_overall.json holds SlopTotal's own quick-score and full-analyze results
on tests/fixtures/detector_texts.json (all 23 engines, its analyzer code).
"""

import json
from pathlib import Path

import pytest

from deai.detectors import QUICK_ENGINES, DetectorError, Detectors, TextRejected, calibration

FIXTURES = Path(__file__).parent / "fixtures"
TEXT = "This passage is long enough to be scored by every engine in the registry. " * 3


class Stub:
    def __init__(self, key, engine_type="neural", value=0.5, fail=False):
        self.key, self.name, self.engine_type = key, key.upper(), engine_type
        self.value, self.fail, self.calls = value, fail, 0

    def score(self, text):
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        return self.value


def stubs(**over):
    keys = list(QUICK_ENGINES) + ["classifier_desklib", "structural", "perplexity"]
    return [Stub(k, **over.get(k, {})) for k in keys]


def test_quick_runs_only_weighted_engines_and_matches_calibration():
    engines = stubs()
    det = Detectors(engines)
    s = det.score(TEXT, "quick")
    ran = {e.key for e in engines if e.calls}
    assert ran == {"classifier_fakespot", "classifier_tmr", "linguistic", "formulaic"}
    assert s.overall == calibration.quick_score(s.engines, TEXT.strip())[0]
    assert s.verdict in ("clean", "mixed", "ai")


def test_engine_subset_without_overall():
    det = Detectors(stubs())
    s = det.score(TEXT, "analyze", engines=["perplexity"], overall=False)
    assert s.overall is None and s.engines == {"perplexity": 0.5}
    with pytest.raises(ValueError):
        det.score(TEXT, "quick", engines=["perplexity"])  # quick-score doesn't run it


def test_scores_are_clamped_and_rounded():
    det = Detectors([Stub("perplexity", value=0.12345), Stub("structural", value=1.7)])
    s = det.score(TEXT, "analyze", engines=["perplexity", "structural"], overall=False)
    assert s.engines == {"perplexity": 0.123, "structural": 1.0}


def test_failures_and_short_texts_are_per_text_exceptions():
    det = Detectors(stubs(classifier_fakespot={"fail": True}))
    out = det.score_many([TEXT, "too short"], "quick")
    assert isinstance(out[0], DetectorError) and "classifier_fakespot" in str(out[0])
    assert isinstance(out[1], TextRejected)


def _engine_cases(endpoint):
    golden = json.loads((FIXTURES / "golden_overall.json").read_text(encoding="utf-8"))[endpoint]
    texts = {t["id"]: t["text"] for t in json.loads((FIXTURES / "detector_texts.json").read_text(encoding="utf-8"))}
    return [(texts[i], g) for i, g in golden.items()]


@pytest.mark.models
@pytest.mark.parametrize("endpoint", ["quick", "analyze"])
def test_overall_matches_sloptotal(endpoint):
    try:
        det = Detectors()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"engines unavailable: {e}")
    cases = _engine_cases(endpoint)
    out = det.score_many([t for t, _ in cases], endpoint, engines=det.endpoint_engines(endpoint))
    worst_overall, worst_engine = 0.0, 0.0
    for (_, g), s in zip(cases, out):
        assert not isinstance(s, BaseException), s
        worst_overall = max(worst_overall, abs(s.overall - g["score"]))
        worst_engine = max(worst_engine, *(abs(s.engines[k] - v) for k, v in g["engines"].items()))
        assert s.confidence == g["confidence"]
    print(f"{endpoint}: max |overall diff| {worst_overall:.2f}, max |engine diff| {worst_engine:.4f}")
    assert worst_engine <= 0.002
    assert worst_overall <= 0.1  # measured 0.0; one 1-dp rounding step of slack for fp32 noise
