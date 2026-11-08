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


class StubEngine:
    def __init__(self, key, engine_type, value):
        self.key, self.name, self.engine_type, self.value = key, key, engine_type, value

    def score(self, text):
        if "fail" in text:
            raise RuntimeError("boom")
        return self.value


def stub_detectors():
    from deai.detectors import Detectors
    return Detectors([StubEngine("classifier_fakespot", "neural", 0.8),
                      StubEngine("classifier_tmr", "neural", 0.4),
                      StubEngine("classifier_e5", "embedding", 0.6),
                      StubEngine("linguistic", "linguistic", 0.2),
                      StubEngine("formulaic", "linguistic", 0.2),
                      StubEngine("perplexity", "statistical", 0.1)])


def test_local_detector_overall_and_engine_subset():
    from deai.detectors import calibration
    from deai.reward import local_detector

    det = stub_detectors()

    def go(engines):
        reward = Reward(local_detector(det, "quick", engines), models=PassAll())
        return asyncio.run(reward.score([SOURCE, SOURCE], [SOURCE, SOURCE + " fail"]))

    ok, failed = go(None)
    expected = calibration.quick_score({"classifier_fakespot": 0.8, "classifier_tmr": 0.4,
                                        "linguistic": 0.2, "formulaic": 0.2}, SOURCE)[0]
    assert ok.detector_score == expected and ok.reward == pytest.approx(1 - expected / 100)
    assert failed.reward is None and "DetectorError" in failed.error
    ok, _ = go(["classifier_fakespot", "linguistic"])
    assert ok.detector_score == pytest.approx(50.0) and ok.reward == pytest.approx(0.5)


def test_family_engines_follow_the_endpoint():
    from deai.reward import family_engines, make_detector

    det = stub_detectors()
    # perplexity is analyze-only, so the quick families never include it.
    assert family_engines(det, "quick", "statistical") == []
    assert family_engines(det, "analyze", "neural") == ["classifier_e5", "classifier_fakespot", "classifier_tmr"]
    assert family_engines(det, "analyze", "statistical") == ["perplexity"]
    scores = asyncio.run(make_detector(det, "stat")([SOURCE]))
    assert scores == [pytest.approx(10.0)]


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


# A real reward hack: runs/dpo-r-stat-qwen3.5-9b round 5, val id ai-qa-stackexchange-0090
# (first paragraph). The rewrite turns "your marriage is your priority" into "your marriage
# remains her top priority", i.e. whose priority it is, and passed every gate
# (NLI src->rw 1.000, rw->src 0.969). The rest of the rewrite is the policy's own text.
ADVICE_SOURCE = (
    "The core issue here isn’t whether you and your friend are “just friends”—it’s that your "
    "wife feels something is off, and she needs reassurance that your marriage is your priority. "
    "Trying to convince her with logic or repeated denials often backfires because it can come "
    "across as dismissive of her feelings. Instead, shift your goal from winning the argument to "
    "rebuilding trust and safety. Start by listening without defending. Ask her what specifically "
    "bothers her: is it the amount of time you spend with your friend, the topics you share, "
    "physical affection, private jokes, late-night texting, or something else? Let her finish "
    "without interrupting or correcting her. Then reflect back what you heard: “So it worries you "
    "when I text her late at night and don’t mention it.” That alone can lower the temperature, "
    "because she feels seen rather than managed."
)
ADVICE_REWRITE = (
    "The central problem isn't if you and your friend are merely pals—it's that your wife senses "
    "something amiss and requires reassurance your marriage remains her top priority. Attempting "
    "to sway her via logic or constant denials frequently rebounds since it risks seeming "
    "dismissive of her emotions. Rather, pivot your aim from winning the debate to reconstructing "
    "trust and safety. Begin by listening without defending yourself. Query her exactly what "
    "disturbs her: is it time spent with your friend, shared topics, physical touch, private "
    "jokes, late-night messaging, or another factor? Permit her to finish without interrupting or "
    "correcting. Then mirror back what you grasped: \"So it worries you when I text her late and "
    "omit mentioning it.\" That single act can cool the heat, as she feels witnessed instead of "
    "managed."
)
SWAPPED = "your marriage remains her top priority"


@pytest.mark.models
def test_nli_catches_whose_priority_swap(reward):
    assert SWAPPED in ADVICE_REWRITE
    # Control: the same rewrite with only the pronoun fixed must pass, so a failure below
    # is the swap and not the policy's other liberties ("merely pals", "witnessed").
    fixed = score_one(reward, ADVICE_REWRITE.replace(SWAPPED, "your marriage remains your top priority"),
                      ADVICE_SOURCE)
    assert fixed.passed, fixed.gates
    r = score_one(reward, ADVICE_REWRITE, ADVICE_SOURCE)
    assert r.failed_gate == "nli" and r.reward == 0.0, r.gates


# ----------------------------------------------------------------------
# Mesmer gate (stubbed: the real model is 9B and runs on the HPC)
# ----------------------------------------------------------------------

class StubMesmer(PassAll):
    """PassAll whose Mesmer returns a fixed verdict and records what it was asked."""
    PROBS = {"human": (0.9, 0.05, 0.05), "ai": (0.05, 0.9, 0.05), "ai_edited": (0.05, 0.05, 0.9)}

    def __init__(self, verdict):
        self.verdict, self.calls = verdict, []

    def ai_verdicts(self, texts):
        self.calls.extend(texts)
        return [(self.verdict, self.PROBS[self.verdict])] * len(texts)


def test_mesmer_off_by_default_never_loads_it():
    models = StubMesmer("ai")
    r = asyncio.run(Reward(detector=AlwaysHuman(), models=models).score([SOURCE], [PARAPHRASE]))[0]
    assert r.passed and r.reward == pytest.approx(1.0)
    assert models.calls == []


@pytest.mark.parametrize("mode, verdict, passed", [
    ("ai", "human", True), ("ai", "ai_edited", True), ("ai", "ai", False),
    ("any", "human", True), ("any", "ai_edited", False), ("any", "ai", False),
])
def test_mesmer_gate(mode, verdict, passed):
    det = AlwaysHuman()
    reward = Reward(detector=det, models=StubMesmer(verdict), config=RewardConfig(mesmer=mode))
    r = asyncio.run(reward.score([SOURCE], [PARAPHRASE]))[0]
    assert r.gate("mesmer").passed is passed and r.gate("mesmer").value == StubMesmer.PROBS[verdict][0]
    assert verdict in r.gate("mesmer").detail
    if passed:
        assert r.reward == pytest.approx(1.0) and det.calls == [PARAPHRASE]
    else:
        assert r.failed_gate == "mesmer" and r.reward == 0.0 and det.calls == []


def test_mesmer_runs_last_and_only_on_survivors():
    models = StubMesmer("human")
    reward = Reward(detector=AlwaysHuman(), models=models, config=RewardConfig(mesmer="any"))
    res = reward.check_gates([SOURCE, SOURCE], [truncate(PARAPHRASE), PARAPHRASE])
    assert res[0].failed_gate == "length" and res[0].gate("mesmer") is None
    assert res[1].gates[-1].name == "mesmer" and models.calls == [PARAPHRASE]


def test_unknown_mesmer_mode_is_refused():
    reward = Reward(detector=AlwaysHuman(), models=StubMesmer("human"), config=RewardConfig(mesmer="on"))
    with pytest.raises(ValueError):
        reward.check_gates([SOURCE], [PARAPHRASE])
