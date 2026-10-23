"""deai.reward: every hand-built shortcut must score 0.

The ``models`` tests load the real embedding/CoLA/NLI models (downloaded on first
run, GPU if available) and are skipped if they can't be loaded. The detector is
a stub that always says "human" (0), the best case for the attacker: anything
that gets through the gates would earn the maximum reward.
"""

import asyncio
import random

import pytest

from deai.reward import (Reward, RewardConfig, chunk_sentences, split_sentences,
                         unicode_problems, word_count)

SOURCE = (
    "Urban beekeeping has grown rapidly over the past decade, as city residents look for ways "
    "to support pollinators and produce local honey. Rooftop hives are now common in London, "
    "Paris and New York, where parks and gardens provide a surprising variety of flowers. "
    "However, experts warn that adding too many honeybee colonies can harm wild bees, which "
    "compete for the same limited supply of nectar and pollen. Some researchers argue that "
    "planting more native flowers would do more for biodiversity than installing new hives. "
    "Cities such as Zurich have begun to track hive numbers so that the density of colonies "
    "stays within what the local landscape can support."
)
PARAPHRASE = (
    "Over the last ten years, beekeeping in cities has expanded quickly, because people living "
    "in urban areas want to help pollinators and make honey close to home. Hives on rooftops "
    "are now a familiar sight in New York, Paris and London, where gardens and parks offer an "
    "unexpectedly wide range of flowers. Specialists caution, though, that too many honeybee "
    "colonies can hurt wild bees, since both compete for the same scarce nectar and pollen. "
    "Some scientists say that planting more native flowers would help biodiversity more than "
    "putting in additional hives. Zurich and other cities have started counting hives to keep "
    "the number of colonies within what the surrounding landscape can sustain."
)
# Human-written, similar length, different topic.
UNRELATED = (
    "My grandmother kept a notebook of every recipe she ever cooked, and most of them were "
    "written in pencil on the back of envelopes before she copied them out. When she moved "
    "into the small flat by the harbour, she gave the notebook to my mother, who promptly "
    "lost the page for the lemon cake everyone actually liked. We spent one whole summer "
    "trying to rebuild it from memory. Nobody agreed on whether it had poppy seeds, and my "
    "uncle insisted there was a splash of gin in the icing, which nobody believed. In the end "
    "we found the envelope tucked inside a library book she had never returned."
)


class AlwaysHuman:
    def __init__(self):
        self.calls: list[str] = []

    async def __call__(self, texts):
        self.calls.extend(texts)
        return [0.0] * len(texts)


def truncate(text: str, frac: float = 0.6) -> str:
    words = text.split()
    return " ".join(words[: int(len(words) * frac)])


def homoglyphs(text: str) -> str:
    return text.translate(str.maketrans({"a": "а", "e": "е", "o": "о"}))  # Cyrillic


def shuffled(text: str, seed: int = 0) -> str:
    words = text.split()
    random.Random(seed).shuffle(words)
    return " ".join(words)


# ----------------------------------------------------------------------
# Pure gates (no models)
# ----------------------------------------------------------------------

class NoModels:
    """Fails loudly if a model-backed gate is reached."""
    def embed(self, texts):
        raise AssertionError("embedder called")
    acceptability = entailment = embed


def gates_only(source, rewrite):
    return Reward(detector=AlwaysHuman(), models=NoModels()).check_gates([source], [rewrite])[0]


def test_fixtures_are_long_enough():
    assert word_count(SOURCE) >= 100
    assert 0.8 <= word_count(PARAPHRASE) / word_count(SOURCE) <= 1.2
    assert 0.8 <= word_count(UNRELATED) / word_count(SOURCE) <= 1.2


def test_homoglyphs_fail_unicode():
    r = gates_only(SOURCE, homoglyphs(PARAPHRASE))
    assert r.failed_gate == "unicode" and r.reward == 0.0


@pytest.mark.parametrize("trick", ["​", "‍", "­", "‮", "﻿", ""])
def test_invisible_characters_fail_unicode(trick):
    words = PARAPHRASE.split()
    words[5] = words[5][:2] + trick + words[5][2:]
    assert unicode_problems(" ".join(words), SOURCE)


def test_nfkc_unstable_characters_fail_unicode():
    assert unicode_problems(PARAPHRASE.replace("London", "Ｌｏｎｄｏｎ"), SOURCE)  # fullwidth
    assert unicode_problems(PARAPHRASE.replace(" ", " ", 3), SOURCE)  # no-break space
    assert unicode_problems(PARAPHRASE.replace("e", "é", 2), SOURCE)  # combining accent


def test_characters_from_the_source_are_allowed():
    src = SOURCE + " The Lyman‑α line and café owners agree."
    rw = PARAPHRASE + " Café owners and the Lyman‑α line agree."
    assert unicode_problems(rw, src) == []
    assert unicode_problems(rw, SOURCE)  # same text, but the source didn't use them


def test_truncation_fails_length():
    r = gates_only(SOURCE, truncate(PARAPHRASE))
    assert r.failed_gate == "length" and r.reward == 0.0


def test_padding_fails_length():
    r = gates_only(SOURCE, PARAPHRASE + " " + PARAPHRASE)
    assert r.failed_gate == "length"


def test_min_words_applies_even_when_ratio_is_fine():
    short = " ".join(SOURCE.split()[:70])
    r = gates_only(short, short)
    assert r.failed_gate == "length"


def test_failed_gates_never_reach_the_detector():
    det = AlwaysHuman()
    reward = Reward(detector=det, models=NoModels())
    res = asyncio.run(reward.score([SOURCE, SOURCE], [truncate(PARAPHRASE), homoglyphs(PARAPHRASE)]))
    assert [r.reward for r in res] == [0.0, 0.0]
    assert det.calls == []


class PassAll:
    """Every model-backed gate passes with similarity 1."""
    def embed(self, texts):
        return [[1.0, 0.0]] * len(texts)

    def acceptability(self, sentences):
        return [1.0] * len(sentences)


def test_sloptotal_detector_overall_and_engine_subset():
    import httpx
    from deai.harness import SlopTotalClient
    from deai.reward import sloptotal_detector

    def handler(req):
        if "fail" in req.content.decode():
            return httpx.Response(500, json={"error": "boom"})
        return httpx.Response(200, json={"score": 60.0, "engines": [
            {"engine": "classifier_fakespot", "score": 80.0}, {"engine": "linguistic", "score": 20.0}]})

    async def go(engines):
        async with SlopTotalClient(transport=httpx.MockTransport(handler), max_retries=1) as st:
            reward = Reward(sloptotal_detector(st, "quick", engines), models=PassAll())
            return await reward.score([SOURCE, SOURCE], [SOURCE, SOURCE + " fail"])

    ok, failed = asyncio.run(go(None))
    assert ok.detector_score == 60.0 and ok.reward == pytest.approx(0.4)
    assert failed.reward is None and "HarnessError" in failed.error
    ok, _ = asyncio.run(go(["classifier_fakespot", "linguistic"]))
    assert ok.detector_score == pytest.approx(50.0) and ok.reward == pytest.approx(0.5)


def test_sentence_splitting_keeps_closing_quotes():
    assert split_sentences('He said "stop." Then he left. (Fine.) Next') == [
        'He said "stop."', "Then he left.", "(Fine.)", "Next"]
    assert chunk_sentences(["a b c", "d e", "f g h i"], 5) == ["a b c d e", "f g h i"]


# ----------------------------------------------------------------------
# Shortcuts against the real models
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def reward():
    try:
        r = Reward(detector=AlwaysHuman(), config=RewardConfig(use_nli=True))
        r.models.embed(["warm-up"])
    except Exception as e:  # offline, no weights
        pytest.skip(f"reward models unavailable: {e}")
    return r


def score_one(reward, rewrite, source=SOURCE):
    return asyncio.run(reward.score([source], [rewrite]))[0]


@pytest.mark.models
def test_faithful_paraphrase_is_rewarded(reward):
    r = score_one(reward, PARAPHRASE)
    assert r.passed, r.gates
    assert r.reward == pytest.approx(r.similarity) and r.reward > 0.85


@pytest.mark.models
@pytest.mark.parametrize("name, rewrite, gate", [
    ("truncated", truncate(PARAPHRASE), "length"),
    ("homoglyphs", homoglyphs(PARAPHRASE), "unicode"),
    ("shuffled", shuffled(SOURCE), None),  # meaning or fluency, see below
    ("unrelated human text", UNRELATED, "meaning"),
])
def test_shortcuts_get_zero(reward, name, rewrite, gate):
    r = score_one(reward, rewrite)
    assert r.reward == 0.0, f"{name}: {r.gates}"
    if gate:
        assert r.failed_gate == gate


@pytest.mark.models
def test_shuffled_words_fail_fluency_on_their_own(reward):
    # Shuffling keeps every word, so similarity alone is not a reliable defence.
    for seed in range(3):
        assert reward.fluency([shuffled(SOURCE, seed)])[0] < reward.config.min_fluency


@pytest.mark.models
@pytest.mark.parametrize("old, new", [
    ("hurt wild bees", "help wild bees"),
    ("help biodiversity more than", "help biodiversity less than"),
])
def test_nli_catches_one_changed_fact(reward, old, new):
    # Embedding similarity barely notices (~0.98); the NLI gate must.
    r = score_one(reward, PARAPHRASE.replace(old, new))
    assert r.failed_gate == "nli" and r.reward == 0.0, r.gates
