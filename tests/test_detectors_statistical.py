"""deai.detectors.statistical: parity with SlopTotal's GPT-2 engines.

``tests/fixtures/golden_statistical.json`` holds SlopTotal's own 3-dp engine
scores for ``detector_texts.json`` (generated with its venv, device in ``meta``).
The ``models`` tests load gpt2-medium and distilgpt2 and are skipped if they
can't be loaded; the rest check the pure formulas on synthetic inputs.
"""

import json
import math
import statistics
import time
from pathlib import Path

import pytest
import torch

from deai.detectors import statistical as st
from deai.detectors.statistical import (
    STATISTICAL_ENGINES, DistilOutput, GPT2Output, LMCache, StatisticalEngine, TokenStats,
)

FIXTURES = Path(__file__).parent / "fixtures"
TOLERANCE = 0.002


# ----------------------------------------------------------------------
# Pure formulas
# ----------------------------------------------------------------------

def test_engine_metadata_matches_sloptotal():
    meta = [(e.key, e.name, e.engine_type) for e in STATISTICAL_ENGINES]
    assert meta == [
        ("binoculars", "Binoculars", "statistical"),
        ("fast_detectgpt", "Fast-DetectGPT", "statistical"),
        ("perplexity", "Perplexity", "statistical"),
        ("cross_perplexity", "Cross-Perplexity", "statistical"),
        ("gltr", "GLTR", "statistical"),
        ("log_rank", "Log-Rank", "statistical"),
        ("diveye", "DivEye", "statistical"),
        ("burstiness", "Burstiness", "linguistic"),
    ]
    assert len({id(e.cache) for e in STATISTICAL_ENGINES}) == 1


def test_ramps():
    assert st.perplexity_score(math.log(11.9)) == 1.0
    assert st.perplexity_score(math.log(50.1)) == 0.0
    assert st.perplexity_score(math.log(31)) == pytest.approx(0.5)
    assert st.binoculars_score(2.0, 2.0 * 1.26) == pytest.approx(0.5)
    assert st.binoculars_score(0.0, 1.0) == 1.0  # performer CE floored at 1e-6
    assert st.binoculars_score(2.0, 2.0) == 0.0
    assert st.fast_detectgpt_score(0.15) == pytest.approx(0.5)
    assert st.fast_detectgpt_score(-1) == 0.0 and st.fast_detectgpt_score(0.3) == 1.0
    assert st.log_rank_score(torch.full((8,), math.e**1.7 - 1)) == pytest.approx(0.5, abs=1e-6)


def test_cross_perplexity_shrinks_on_disagreement():
    # Both at the middle of their ramps: full agreement, plain mean.
    assert st.cross_perplexity_score(math.log(31), math.log(45)) == pytest.approx(0.5)
    # 1 vs 0: mean 0.5, agreement 0 -> 0.5 * 0.7.
    assert st.cross_perplexity_score(math.log(5), math.log(100)) == pytest.approx(0.35)


def test_gltr_uses_top1_share():
    ranks = torch.tensor([0, 0, 0, 0, 1, 2, 5, 9, 50, 3])  # 40% top-1
    assert st.gltr_score(ranks, 10) == pytest.approx(0.5)
    assert st.gltr_score(torch.zeros(10, dtype=torch.long), 10) == 1.0


def test_token_stats_on_synthetic_logits():
    logits = torch.tensor([[[2.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 3.0, 2.0], [9.0, 9.0, 9.0]]])
    ids = torch.tensor([[0, 1, 2, 1]])
    s = st.token_stats(logits, ids)
    lp = torch.log_softmax(logits[0, :-1], -1)
    assert torch.allclose(s.actual_lp, lp[[0, 1, 2], [1, 2, 1]])
    assert s.ranks.tolist() == [1, 0, 0]  # ties don't count as above
    # Uniform row: zero variance is clamped, curvature 0.
    assert s.curvature[1].item() == pytest.approx(0.0, abs=1e-4)
    p = lp[0].exp()
    mean, var = (p * lp[0]).sum(), (p * lp[0] ** 2).sum() - (p * lp[0]).sum() ** 2
    assert s.curvature[0].item() == pytest.approx(((lp[0, 1] - mean) / var.sqrt()).item(), rel=1e-5)


def test_diveye_stats_are_population_moments():
    xs = [1.0, 2.0, 2.0, 3.0, 7.0]
    mean, std, skew, kurt = st.diveye_stats(xs)
    assert mean == pytest.approx(statistics.mean(xs))
    assert std == pytest.approx(statistics.pstdev(xs))
    m3 = sum((x - mean) ** 3 for x in xs) / len(xs)
    m4 = sum((x - mean) ** 4 for x in xs) / len(xs)
    assert skew == pytest.approx(m3 / std**3) and kurt == pytest.approx(m4 / std**4 - 3)
    assert st.diveye_stats([1.0, 2.0, 3.0]) == (0.0, 0.0, 0.0, 0.0)
    assert st.diveye_stats([2.0] * 6)[1:] == (1e-10, 0.0, 0.0)


def test_diveye_score():
    # Symmetric values: skew 0 -> skew term 0.5 * 0.15.
    xs = [1.0, 1.0, 3.0, 3.0] * 5  # mean 2, pstdev 1, CV 0.5 -> cv term 0
    assert st.diveye_score(xs) == pytest.approx(0.075)
    xs = [0.0, 0.0, 4.0, 4.0] * 5  # mean 2, pstdev 2, CV 1.0
    assert st.diveye_score(xs) == pytest.approx((1.0 - 0.87) / 0.26 * 0.85 + 0.075)


def test_burstiness_sentences():
    text = "One two three four five. Short one! Six seven eight nine ten?  Eleven twelve thirteen fourteen fifteen"
    assert st.burstiness_sentences(text) == [
        "One two three four five.", "Six seven eight nine ten?",
        "Eleven twelve thirteen fourteen fifteen",
    ]


def test_sentence_alignment_is_approximate_offsets():
    encode = str.split  # one "token" per word
    losses = torch.tensor([1.0] * 5 + [2.0] * 3 + [3.0] * 4 + [4.0] * 2)
    sents = ["a b c d e", "f g h", "i j k l", "m n o p q r"]
    # 3-token sentence is skipped but still advances the offset; last is cut to 2 -> skipped.
    pps = st.sentence_perplexities(sents, losses, encode)
    assert pps == pytest.approx([math.e, math.e**3])
    # Stops once the offsets run past the text.
    assert st.sentence_perplexities(["a b c d e f g h i j k l m n o p"] + sents, losses, encode) == \
        pytest.approx([math.exp(losses.mean().item())])


def test_burstiness_score_uses_sample_stdev():
    pps = [10.0, 12.0, 14.0, 16.0]
    cv = statistics.stdev(pps) / statistics.mean(pps)
    assert st.burstiness_score(pps) == pytest.approx(1 - (cv - 0.15) / 0.65)
    assert st.burstiness_score([10.0] * 4) == 1.0


class FakeLM:
    """LMCache stand-in returning fixed outputs."""
    def __init__(self, n_tokens, loss=3.0, distil_loss=4.0):
        n = max(n_tokens, 0)
        stats = TokenStats(torch.full((n,), -2.0), torch.full((n,), 0.1), torch.zeros(n, dtype=torch.long))
        self.out = GPT2Output(None if n_tokens < 5 else loss, n_tokens, None if n_tokens < 5 else stats)
        self.d = DistilOutput(None if n_tokens + 1 < 10 else distil_loss, n_tokens)
        self.tokenizer = None

    def gpt2(self, text):
        return self.out

    def distil(self, text):
        return self.d


@pytest.mark.parametrize("n_tokens,nonzero", [
    (4, set()),
    (8, {"perplexity"}),
    (9, {"perplexity", "binoculars", "cross_perplexity"}),
    (19, {"perplexity", "binoculars", "cross_perplexity", "fast_detectgpt", "gltr", "log_rank",
          "diveye"}),
])
def test_minimum_token_counts(n_tokens, nonzero):
    scores = {e.key: StatisticalEngine(e.key, e.name, e.engine_type, e._score, FakeLM(n_tokens)).score("x")
              for e in STATISTICAL_ENGINES}
    assert {k for k, v in scores.items() if v > 0} == nonzero


# ----------------------------------------------------------------------
# Golden parity with SlopTotal (real models)
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def lm():
    cache = LMCache()
    try:
        cache.gpt2("Loading the models to see whether their weights are available here.")
        cache.distil("Loading the models to see whether their weights are available here.")
    except Exception as e:  # pragma: no cover - depends on the machine
        pytest.skip(f"GPT-2 weights unavailable: {e}")
    yield cache
    cache.unload()


@pytest.mark.models
def test_matches_sloptotal_golden(lm):
    texts = json.loads((FIXTURES / "detector_texts.json").read_text(encoding="utf-8"))
    golden = json.loads((FIXTURES / "golden_statistical.json").read_text(encoding="utf-8"))["texts"]
    engines = [StatisticalEngine(e.key, e.name, e.engine_type, e._score, lm) for e in STATISTICAL_ENGINES]
    diffs = {e.key: 0.0 for e in engines}
    worst = {}
    start = time.perf_counter()
    for item in texts:
        lm.clear()
        for e in engines:
            ours = round(min(max(e.score(item["text"]), 0.0), 1.0), 3)
            d = abs(ours - golden[item["id"]]["scores"][e.key])
            if d >= diffs[e.key]:
                diffs[e.key], worst[e.key] = d, item["id"]
    per_text = (time.perf_counter() - start) / len(texts)
    print(f"\nmax |diff| per engine over {len(texts)} texts ({per_text:.3f}s/text):")
    for k, d in diffs.items():
        print(f"  {k:18s} {d:.4f}  ({worst[k]})")
    bad = {k: (d, worst[k]) for k, d in diffs.items() if d > TOLERANCE}
    assert not bad


@pytest.mark.models
def test_one_forward_pass_per_model(lm, monkeypatch):
    calls = []
    forward = LMCache._forward
    monkeypatch.setattr(LMCache, "_forward", lambda self, name, text: calls.append(name) or forward(self, name, text))
    engines = [StatisticalEngine(e.key, e.name, e.engine_type, e._score, lm) for e in STATISTICAL_ENGINES]
    text = json.loads((FIXTURES / "detector_texts.json").read_text(encoding="utf-8"))[0]["text"]
    lm.clear()
    for e in engines:
        e.score(text)
    assert sorted(calls) == ["distilgpt2", "gpt2-medium"]
