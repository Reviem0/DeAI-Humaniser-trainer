"""deai.detectors.neural: the windowing and label lookup, and golden parity with SlopTotal.

``golden_neural.json`` holds SlopTotal's own engine scores (3 dp) for the texts in
``detector_texts.json``, made by running its classifier engines in its venv. The
``models`` tests load the 9 real classifiers (skipped if the weights can't load)
and must reproduce every score within 0.002.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from deai.detectors.base import ENGINE_TYPES
from deai.detectors.neural import (ENGINE_CLASSES, NEURAL_ENGINES, ChatGPTEngine, E5Engine,
                                   FakespotEngine, OpenAIEngine, TMREngine, ai_label_index,
                                   window_starts, window_texts)

FIXTURES = Path(__file__).parent / "fixtures"
TOLERANCE = 0.002


# ----------------------------------------------------------------------
# window_starts
# ----------------------------------------------------------------------

def test_window_starts_short_texts_get_every_window():
    assert window_starts(600, 510, 256) == [0, 256, 512]
    assert window_starts(1, 510, 256) == [0]
    assert window_starts(8 * 256, 510, 256) == [256 * i for i in range(8)]  # exactly 8


def test_window_starts_long_texts_spread_max_windows_evenly():
    # 12 strides > 8: 8 starts from 0 to 3000 - 510 = 2490.
    assert window_starts(3000, 510, 256) == [0, 356, 711, 1067, 1423, 1779, 2134, 2490]
    assert window_starts(2100, 510, 510, 4) == [0, 530, 1060, 1590]
    assert window_starts(2000, 510, 510, 4) == [0, 510, 1020, 1530]


def test_window_starts_rounds_halves_to_even():
    assert window_starts(515, 510, 100, 3) == [0, 2, 5]  # 2.5 -> 2
    assert window_starts(517, 510, 100, 3) == [0, 4, 7]  # 3.5 -> 4
    assert window_starts(512, 510, 100, 5) == [0, 0, 1, 2, 2]  # 0.5 -> 0, 1.5 -> 2


class WordTokenizer:
    """One token per word, decoded back with spaces."""

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


def test_window_texts():
    tok = WordTokenizer()
    short = " ".join(f"w{i}" for i in range(510))
    assert window_texts(short, tok) == [short]
    # Starts 0, 256, 512; the window at 512 has 18 tokens, so the loop stops.
    words = [f"w{i}" for i in range(530)]
    assert window_texts(" ".join(words), tok) == [" ".join(words[:510]), " ".join(words[256:530])]
    assert len(window_texts(" ".join(f"w{i}" for i in range(5000)), tok)) == 8


# ----------------------------------------------------------------------
# AI label index
# ----------------------------------------------------------------------

def test_ai_label_index():
    assert ai_label_index({0: "Human", 1: "AI"}) == 1
    assert ai_label_index({0: "machine-generated", 1: "human"}) == 0
    assert ai_label_index({0: "LABEL_0", 1: "LABEL_1"}) == 1  # default
    assert ai_label_index({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}, default=None) == 2
    assert ai_label_index({0: "Fake", 1: "Real"}, ("fake",)) == 0
    assert ai_label_index({0: "human", 1: "gpt"}) == 1  # default, not a keyword match
    assert ai_label_index({0: "gpt", 1: "human"}, ("gpt",)) == 0


def prepared(engine_cls, id2label):
    engine = engine_cls()
    engine._prepare(SimpleNamespace(config=SimpleNamespace(id2label=id2label)))
    return engine._ai_idx


def test_engine_label_quirks():
    assert prepared(FakespotEngine, {0: "Human", 1: "AI"}) == 1
    assert prepared(E5Engine, {0: "GPT", 1: "human"}) == 0
    assert prepared(OpenAIEngine, {0: "Fake", 1: "Real"}) == 0
    assert prepared(ChatGPTEngine, {0: "Human", 1: "ChatGPT"}) == 1
    assert prepared(ChatGPTEngine, {0: "x", 1: "y", 2: "z"}) == 2  # last label
    # SlopTotal's TMR reads probs[ai_idx or 1]: an AI label at 0 is read at 1.
    assert prepared(TMREngine, {0: "AI", 1: "human"}) == 1


def test_engine_list_matches_sloptotal():
    keys = [e.key for e in NEURAL_ENGINES]
    assert len(set(keys)) == len(keys) == 9
    assert all(e.engine_type in ENGINE_TYPES for e in NEURAL_ENGINES)
    golden = json.loads((FIXTURES / "golden_neural.json").read_text(encoding="utf-8"))
    for e in NEURAL_ENGINES:
        meta = golden["meta"][e.key]
        assert (e.name, e.engine_type) == (meta["name"], meta["engine_type"])


# ----------------------------------------------------------------------
# Golden parity with SlopTotal (real models)
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def texts():
    return json.loads((FIXTURES / "detector_texts.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def golden():
    return json.loads((FIXTURES / "golden_neural.json").read_text(encoding="utf-8"))["engines"]


def clamp3(x):
    return round(min(max(x, 0.0), 1.0), 3)


@pytest.fixture(params=ENGINE_CLASSES, ids=lambda c: c.key)
def engine(request):
    engine = request.param()
    try:
        engine.load()
    except Exception as e:  # noqa: BLE001 - offline, missing weights, OOM
        pytest.skip(f"{engine.key} could not load: {e}")
    yield engine
    engine.unload()


@pytest.mark.models
def test_golden_parity(engine, texts, golden, capsys):
    expected = golden[engine.key]
    single = {t["id"]: clamp3(engine.score(t["text"])) for t in texts}
    batched = dict(zip([t["id"] for t in texts],
                       map(clamp3, engine.score_many([t["text"] for t in texts]))))
    diff = max(abs(single[i] - expected[i]) for i in expected)
    diff_batched = max(abs(batched[i] - expected[i]) for i in expected)
    with capsys.disabled():
        print(f"\n{engine.key}: max |diff| {diff:.3f} single, {diff_batched:.3f} score_many")
    bad = {i: (single[i], expected[i]) for i in expected if abs(single[i] - expected[i]) > TOLERANCE}
    assert not bad, f"{engine.key} score() off golden: {bad}"
    assert diff_batched <= TOLERANCE
