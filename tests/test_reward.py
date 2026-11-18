"""deai.reward: every hand-built shortcut must score 0.

The ``models`` tests load the real embedding/CoLA/NLI models (downloaded on first
run, GPU if available) and are skipped if they can't be loaded. The detector is
a stub that always says "human" (0), the best case for the attacker: anything
that gets through the gates would earn the maximum reward.
"""

import asyncio
import math
import random
import re

import numpy as np
import pytest

from deai.reward import (Reward, RewardConfig, chunk_sentences, lm_stats, mesmer_factor, split_sentences,
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


def test_mesmer_factor_is_log_scaled_with_a_floor():
    assert mesmer_factor(1.0) == pytest.approx(1.0)
    assert mesmer_factor(1e-6) == pytest.approx(0.5) and mesmer_factor(1e-12) == pytest.approx(0.5)
    assert mesmer_factor(1e-3) == pytest.approx(0.75)
    # each decade counts the same, so saturated p_human still gives signal
    assert mesmer_factor(1e-4) - mesmer_factor(1e-5) == pytest.approx(mesmer_factor(1e-1) - mesmer_factor(1e-2))
    assert mesmer_factor(1e-3, weight=0.2) == pytest.approx(0.9)


class PerTextMesmer(PassAll):
    """PassAll whose Mesmer reads (verdict, p_human) from a dict keyed by rewrite text."""

    def __init__(self, by_text):
        self.by_text, self.calls = by_text, []

    def ai_verdicts(self, texts):
        self.calls.extend(texts)
        out = []
        for t in texts:
            verdict, h = self.by_text[t]
            out.append((verdict, (h, 1 - h, 0.0)))
        return out


def test_mesmer_soft_scales_reward_without_gating():
    det = AlwaysHuman()
    reward = Reward(detector=det, models=StubMesmer("ai"), config=RewardConfig(mesmer="soft"))
    r = asyncio.run(reward.score([SOURCE], [PARAPHRASE]))[0]
    assert r.passed and r.gate("mesmer") is None and det.calls == [PARAPHRASE]
    assert r.mesmer_verdict == "ai" and r.mesmer_probs == StubMesmer.PROBS["ai"]
    assert r.reward == pytest.approx(mesmer_factor(0.05))


def test_mesmer_soft_only_scores_rewrites_that_passed_the_gates():
    models = StubMesmer("human")
    reward = Reward(detector=AlwaysHuman(), models=models, config=RewardConfig(mesmer="soft"))
    res = asyncio.run(reward.score([SOURCE, SOURCE], [truncate(PARAPHRASE), PARAPHRASE]))
    assert res[0].failed_gate == "length" and res[0].reward == 0.0 and res[0].mesmer_probs is None
    assert models.calls == [PARAPHRASE]


def test_mesmer_soft_ranks_by_p_human_when_everything_is_ai():
    a, b = PARAPHRASE, PARAPHRASE + " Indeed."
    models = PerTextMesmer({a: ("ai", 1e-5), b: ("ai", 1e-3)})
    reward = Reward(detector=AlwaysHuman(), models=models, config=RewardConfig(mesmer="soft"))
    ra, rb = asyncio.run(reward.score([SOURCE, SOURCE], [a, b]))
    assert ra.passed and rb.passed and 0 < ra.reward < rb.reward


def test_mesmer_adaptive_gates_a_source_once_one_rewrite_gets_past():
    a, b = PARAPHRASE, PARAPHRASE + " Indeed."
    other = SOURCE + " More."
    models = PerTextMesmer({a: ("ai", 1e-5), b: ("ai_edited", 1e-2)})
    det = AlwaysHuman()
    reward = Reward(detector=det, models=models, config=RewardConfig(mesmer="adaptive"))
    # SOURCE has a rewrite Mesmer doesn't call "ai", so its "ai" one fails the gate;
    # `other` has only "ai" rewrites, so it stays soft.
    ra, rb, rc = asyncio.run(reward.score([SOURCE, SOURCE, other], [a, b, a]))
    assert ra.failed_gate == "mesmer" and ra.reward == 0.0
    assert rb.passed and rb.gate("mesmer").passed and rb.reward == pytest.approx(mesmer_factor(1e-2))
    assert rc.passed and rc.gate("mesmer") is None and rc.reward == pytest.approx(mesmer_factor(1e-5))
    assert sorted(det.calls) == sorted([b, a])  # the gated rewrite never reaches the detector


def test_unknown_mesmer_mode_is_refused():
    reward = Reward(detector=AlwaysHuman(), models=StubMesmer("human"), config=RewardConfig(mesmer="on"))
    with pytest.raises(ValueError):
        reward.check_gates([SOURCE], [PARAPHRASE])


# ----------------------------------------------------------------------
# Perplexity / burstiness gates (stub LM: one token per word, NLL by word)
# ----------------------------------------------------------------------

class StubLM(PassAll):
    """PassAll whose LM gives each whitespace word one token, with NLL ``nll(word)``."""

    def __init__(self, nll=lambda w: 2.0):
        self.nll, self.calls = nll, []

    def token_nll(self, texts):
        self.calls.extend(texts)
        out = []
        for t in texts:
            spans = [m.span() for m in re.finditer(r"\S+", t)]
            out.append(([self.nll(t[a:b]) for a, b in spans], spans))
        return out


def test_lm_stats_assigns_tokens_to_sentences():
    text = "One two three four. Five six seven eight. A b c d. W x y z."
    nll, spans = StubLM(lambda w: 1.0 if w[0].isupper() or w[0] in "otfs" else 3.0).token_nll([text])[0]
    st = lm_stats(text, nll, spans)
    assert len(st.sentence_perplexities) == 4
    assert st.perplexity == pytest.approx(math.exp(np.mean(nll)))
    pps = st.sentence_perplexities
    assert st.burstiness == pytest.approx(np.std(pps, ddof=1) / np.mean(pps))
    # Leading whitespace tokens count for the sentence they end in, not the previous one.
    spaced = [(a - 1, b) if a else (a, b) for a, b in spans]
    assert lm_stats(text, nll, spaced).sentence_perplexities == pps


def test_lm_stats_too_few_sentences_leaves_burstiness_unmeasured():
    text = "One two three four five. Six seven eight nine ten."
    assert lm_stats(text, *StubLM().token_nll([text])[0]).burstiness is None
    # Sentences under min_sentence_tokens don't count.
    text = "One two three four. Five. Six. Seven. Eight."
    assert len(lm_stats(text, *StubLM().token_nll([text])[0]).sentence_perplexities) == 1


def test_lm_gates_off_by_default_never_load_the_lm():
    models = StubLM(lambda w: 50.0)
    r = asyncio.run(Reward(detector=AlwaysHuman(), models=models).score([SOURCE], [PARAPHRASE]))[0]
    assert r.passed and r.gate("perplexity") is None and models.calls == []


def test_ordinary_text_passes_lm_gates():
    reward = Reward(detector=AlwaysHuman(), models=StubLM(), config=RewardConfig(use_lm_gates=True))
    r = asyncio.run(reward.score([SOURCE], [PARAPHRASE]))[0]
    assert r.passed and r.reward == pytest.approx(1.0)
    assert r.gate("perplexity").value == pytest.approx(math.exp(2.0))
    assert r.gate("burstiness").value == pytest.approx(0.0)


def test_inflated_perplexity_fails_and_skips_the_detector():
    det = AlwaysHuman()
    reward = Reward(detector=det, models=StubLM(lambda w: 5.0), config=RewardConfig(use_lm_gates=True))
    r = asyncio.run(reward.score([SOURCE], [PARAPHRASE]))[0]
    assert r.failed_gate == "perplexity" and r.reward == 0.0 and det.calls == []
    assert r.gate("burstiness") is None  # gates stop at the first failure


def test_one_planted_odd_sentence_fails_burstiness():
    # Mean perplexity stays human-range; one sentence of rare words makes it bursty.
    # 12 sentences: a sample CV can't exceed sqrt(n), so short texts can't fail this gate.
    text = PARAPHRASE + " " + UNRELATED
    odd_sentence = "We spent one whole summer trying to rebuild it from memory."
    others = set(text.replace(odd_sentence, "").split())
    odd = set(odd_sentence.split()) - others
    assert len(odd) >= 5
    reward = Reward(detector=AlwaysHuman(), models=StubLM(lambda w: 8.0 if w in odd else 1.5),
                    config=RewardConfig(use_lm_gates=True))
    r = reward.check_gates([text], [text])[0]
    assert r.gate("perplexity").passed, r.gates
    assert r.failed_gate == "burstiness" and r.reward == 0.0, r.gates


def test_burstiness_unmeasured_passes():
    text = " ".join(SOURCE.split()).replace(". ", "; ")  # one long sentence
    reward = Reward(detector=AlwaysHuman(), models=StubLM(), config=RewardConfig(use_lm_gates=True))
    r = reward.check_gates([SOURCE], [text])[0]
    assert r.passed and r.gate("burstiness").value is None and "too few" in r.gate("burstiness").detail


def test_lm_gates_run_after_fluency_and_before_nli():
    models = StubLM()
    models.entailment = lambda p, h: [1.0] * len(p)
    reward = Reward(detector=AlwaysHuman(), models=models, config=RewardConfig(use_lm_gates=True, use_nli=True))
    r = reward.check_gates([SOURCE], [PARAPHRASE])[0]
    names = [g.name for g in r.gates]
    assert names.index("fluency") < names.index("perplexity") < names.index("burstiness") < names.index("nli")


@pytest.mark.models
def test_real_lm_gates(reward):
    lm = Reward(detector=AlwaysHuman(), models=reward.models, config=RewardConfig(use_lm_gates=True))
    st = lm.lm_stats([PARAPHRASE, UNRELATED, shuffled(SOURCE)])
    cfg = lm.config
    for s in st[:2]:  # a faithful paraphrase and real human text sit inside the human range
        assert s.perplexity <= cfg.max_perplexity and s.burstiness <= cfg.max_burstiness, s
    assert st[2].perplexity > cfg.max_perplexity  # word salad is far outside it


# ----------------------------------------------------------------------
# Structure gate (no models)
# ----------------------------------------------------------------------

def _paras(text, n):
    sents = split_sentences(text)
    k = -(-len(sents) // n)
    return "\n\n".join(" ".join(sents[i:i + k]) for i in range(0, len(sents), k))


def structure_only(source, rewrite, **cfg):
    return Reward(detector=AlwaysHuman(), models=PassAll(),
                  config=RewardConfig(use_structure_gate=True, **cfg)).check_gates([source], [rewrite])[0]


def test_structure_gate_off_by_default():
    r = Reward(detector=AlwaysHuman(), models=PassAll()).check_gates([SOURCE], [PARAPHRASE.replace(". ", "; ")])[0]
    assert r.passed and r.gate("structure") is None


def test_identity_and_paraphrase_pass_structure():
    src = _paras(SOURCE, 3)
    assert structure_only(src, src).gate("structure").passed
    assert structure_only(src, _paras(PARAPHRASE, 3)).gate("structure").passed


def test_semicolon_stitching_and_merged_paragraphs_fail_structure():
    # The round 7 shortcut: sentences joined with semicolons, paragraphs merged into one.
    src = _paras(SOURCE, 3)
    r = structure_only(src, PARAPHRASE.replace(". ", "; "))
    assert r.failed_gate == "structure" and r.reward == 0.0
    detail = r.gate("structure").detail
    assert "semicolons" in detail and "1 paragraphs" in detail
    # Each rule on its own.
    assert "semicolons" in structure_only(SOURCE, PARAPHRASE.replace(". ", "; ", 3)).gate("structure").detail
    assert "paragraphs" in structure_only(src, PARAPHRASE).gate("structure").detail
    one_per_sentence = "\n\n".join(split_sentences(PARAPHRASE))
    assert "paragraphs" in structure_only(SOURCE, one_per_sentence, max_extra_paragraphs=2).gate("structure").detail
    long = structure_only(SOURCE, PARAPHRASE, max_sentence_words=20)
    assert "-word sentence" in long.gate("structure").detail


def test_structure_bounds_rise_to_the_sources_own():
    src = SOURCE.replace(". ", "; ", 3)  # a source that already uses semicolons
    assert structure_only(src, src).gate("structure").passed
    assert not structure_only(SOURCE, src).gate("structure").passed


def test_structure_runs_after_length_and_before_meaning():
    r = structure_only(SOURCE, truncate(PARAPHRASE))
    assert r.failed_gate == "length" and r.gate("structure") is None
    r = Reward(detector=AlwaysHuman(), models=NoModels(),
               config=RewardConfig(use_structure_gate=True)).check_gates([SOURCE], [PARAPHRASE.replace(". ", "; ")])[0]
    assert r.failed_gate == "structure"  # NoModels would raise if meaning ran
