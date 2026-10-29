"""SlopTotal's GPT-2 statistical engines, scored in-process.

Exact ports of SlopTotal's (https://github.com/pablocaeg/sloptotal)
``app/engines/{gpt2_cache,perplexity,cross_perplexity,binoculars,fast_detectgpt,
gltr,log_rank,diveye,burstiness}.py``: same models (gpt2-medium and distilgpt2,
fp32, no BOS, truncated at 1024 tokens), minimum token counts, ramps and
constants. SlopTotal's comments say "GPT-2 Large"; the model is gpt2-medium.

All eight engines share one ``LMCache``, so scoring them all on a text runs one
forward pass per model. The cache keeps per-token statistics, not logits: the
token log-probs, analytic curvature terms and ranks every engine is built from,
computed in fp32 like SlopTotal's CPU post-processing.

Differences from SlopTotal: a model failure raises instead of scoring 0, so the
reward can return ``None``; scores are not rounded here (the registry does it).
"""

from __future__ import annotations

import gc
import math
import re
import statistics
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import torch

from deai.detectors.base import detector_device

# Canonical repo ids: the hub still redirects "gpt2-medium" / "distilgpt2" (same commits),
# but downloads through the alias fail with a 404 on large files.
GPT2_MODEL = "openai-community/gpt2-medium"
DISTIL_MODEL = "distilbert/distilgpt2"
MAX_TOKENS = 1024
CACHE_SIZE = 5  # texts, as SlopTotal's _MAX_CACHE


def ramp_down(x: float, lo: float, hi: float) -> float:
    """1 at or below ``lo``, 0 at or above ``hi``, linear between."""
    if x <= lo:
        return 1.0
    if x >= hi:
        return 0.0
    return 1.0 - (x - lo) / (hi - lo)


def clamp01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


# --- per-token statistics -------------------------------------------------------

@dataclass(frozen=True)
class TokenStats:
    """GPT-2 per-position statistics for predicting token i+1 from its prefix (CPU, fp32)."""

    actual_lp: torch.Tensor  # log p(actual token)
    curvature: torch.Tensor  # (actual_lp - E[lp]) / std[lp], Fast-DetectGPT
    ranks: torch.Tensor  # number of vocabulary logits above the actual token's


def token_stats(logits: torch.Tensor, input_ids: torch.Tensor) -> TokenStats:
    """Everything the GPT-2 engines read from ``logits`` [1, T, V] and ``input_ids`` [1, T]."""
    pred_logits = logits[0, :-1].float()
    actual = input_ids[0, 1:].to(pred_logits.device).unsqueeze(1)
    log_probs = torch.log_softmax(pred_logits, dim=-1)
    actual_lp = log_probs.gather(1, actual).squeeze(1)
    # Fast-DetectGPT: SlopTotal takes softmax separately rather than exp(log_probs).
    probs = torch.softmax(pred_logits, dim=-1)
    expected_lp = (probs * log_probs).sum(dim=-1)
    expected_lp_sq = (probs * log_probs**2).sum(dim=-1)
    del probs, log_probs
    variance = (expected_lp_sq - expected_lp**2).clamp(min=1e-10)
    curvature = (actual_lp - expected_lp) / torch.sqrt(variance)
    ranks = (pred_logits > pred_logits.gather(1, actual)).sum(dim=-1)
    return TokenStats(actual_lp.cpu(), curvature.cpu(), ranks.cpu())


@dataclass(frozen=True)
class GPT2Output:
    loss: float | None  # None when n_tokens < 5, as SlopTotal
    n_tokens: int  # len(input_ids) - 1
    stats: TokenStats | None


@dataclass(frozen=True)
class DistilOutput:
    loss: float | None  # None when len(input_ids) < 10, as SlopTotal
    n_tokens: int


class LMCache:
    """Lazily loaded gpt2-medium + distilgpt2 with one forward pass per model per text.

    One lock covers loading and forward passes: SlopTotal found that concurrent
    loads corrupted GPT-2's tied lm_head.
    """

    def __init__(self, gpt2_model: str = GPT2_MODEL, distil_model: str = DISTIL_MODEL,
                 size: int = CACHE_SIZE) -> None:
        self.gpt2_model, self.distil_model, self.size = gpt2_model, distil_model, size
        self._lock = threading.RLock()
        self._models: dict[str, tuple] = {}
        self._gpt2: OrderedDict[str, GPT2Output] = OrderedDict()
        self._distil: OrderedDict[str, DistilOutput] = OrderedDict()

    def _load(self, name: str):
        with self._lock:
            if name not in self._models:
                from transformers import GPT2LMHeadModel, GPT2TokenizerFast
                tokenizer = GPT2TokenizerFast.from_pretrained(name)
                model = GPT2LMHeadModel.from_pretrained(name, dtype=torch.float32)
                self._models[name] = (model.eval().to(detector_device()), tokenizer)
            return self._models[name]

    @property
    def tokenizer(self):
        """gpt2-medium's tokenizer (Burstiness aligns sentences with it)."""
        return self._load(self.gpt2_model)[1]

    def _forward(self, name: str, text: str):
        model, tokenizer = self._load(name)
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=MAX_TOKENS).input_ids
        return model, ids

    def gpt2(self, text: str) -> GPT2Output:
        with self._lock:
            if text in self._gpt2:
                self._gpt2.move_to_end(text)
                return self._gpt2[text]
            model, ids = self._forward(self.gpt2_model, text)
            n_tokens = ids.size(1) - 1
            if n_tokens < 5:
                out = GPT2Output(None, n_tokens, None)
            else:
                ids = ids.to(next(model.parameters()).device)
                with torch.no_grad():
                    res = model(ids, labels=ids)
                    out = GPT2Output(res.loss.item(), n_tokens, token_stats(res.logits, ids))
                del res
            return _remember(self._gpt2, text, out, self.size)

    def distil(self, text: str) -> DistilOutput:
        with self._lock:
            if text in self._distil:
                self._distil.move_to_end(text)
                return self._distil[text]
            model, ids = self._forward(self.distil_model, text)
            loss = None
            if ids.size(1) >= 10:
                ids = ids.to(next(model.parameters()).device)
                with torch.no_grad():
                    loss = model(ids, labels=ids).loss.item()
            return _remember(self._distil, text, DistilOutput(loss, ids.size(1) - 1), self.size)

    def clear(self) -> None:
        """Drop cached per-text results."""
        with self._lock:
            self._gpt2.clear()
            self._distil.clear()

    def unload(self) -> None:
        """Drop the models and cache and return their GPU memory; they reload on next use."""
        with self._lock:
            self.clear()
            self._models.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _remember(cache: OrderedDict, key: str, value, size: int):
    cache[key] = value
    while len(cache) > size:
        cache.popitem(last=False)
    return value


# --- engine formulas (pure; scores before the registry's clamp and rounding) ----

def perplexity_score(loss: float) -> float:
    """GPT-2 Medium perplexity: <= 12 is AI (1), >= 50 human (0)."""
    return ramp_down(math.exp(loss), 12, 50)


def cross_perplexity_score(gpt2_loss: float, distil_loss: float) -> float:
    """Mean of the two models' perplexity ramps, shrunk by up to 30% when they disagree."""
    score_large = ramp_down(math.exp(gpt2_loss), 12, 50)
    score_distil = ramp_down(math.exp(distil_loss), 20, 70)
    avg = (score_large + score_distil) / 2
    agreement = 1.0 - abs(score_large - score_distil)
    return clamp01(avg * (0.7 + 0.3 * agreement))


def binoculars_score(gpt2_loss: float, distil_loss: float) -> float:
    """distilgpt2 / gpt2-medium cross-entropy ratio ramped up over 1.17..1.35.

    SlopTotal's calibrated variant, not the paper's perplexity/cross-perplexity.
    """
    ratio = distil_loss / max(gpt2_loss, 1e-6)
    return clamp01((ratio - 1.17) / (1.35 - 1.17))


def fast_detectgpt_score(mean_curvature: float) -> float:
    """Mean analytic curvature over 0..0.30."""
    if mean_curvature >= 0.30:
        return 1.0
    if mean_curvature <= 0.0:
        return 0.0
    return mean_curvature / 0.30


def gltr_score(ranks: torch.Tensor, n_tokens: int) -> float:
    """Share of top-1 (argmax) tokens over 0.30..0.50."""
    pct_top1 = (ranks == 0).sum().item() / n_tokens
    if pct_top1 >= 0.50:
        return 1.0
    if pct_top1 <= 0.30:
        return 0.0
    return (pct_top1 - 0.30) / 0.20


def log_rank_score(ranks: torch.Tensor) -> float:
    """Mean log(rank + 1): <= 1.2 is AI (1), >= 2.2 human (0)."""
    return ramp_down(torch.log(ranks.float() + 1).mean().item(), 1.2, 2.2)


def diveye_stats(values: list[float]) -> tuple[float, float, float, float]:
    """Population mean, std, skewness and excess kurtosis; zeros below 4 values."""
    n = len(values)
    if n < 4:
        return 0.0, 0.0, 0.0, 0.0
    mean = sum(values) / n
    diffs = [x - mean for x in values]
    var = sum(d**2 for d in diffs) / n
    std = math.sqrt(var) if var > 0 else 1e-10
    skew = (sum(d**3 for d in diffs) / n) / (std**3) if std > 1e-10 else 0.0
    kurt = (sum(d**4 for d in diffs) / n) / (std**4) - 3.0 if std > 1e-10 else 0.0
    return mean, std, skew, kurt


def diveye_score(surprisals: list[float]) -> float:
    """Surprisal CV ramped up over 0.87..1.13 (0.85) plus a weak low-|skew| term (0.15)."""
    mean, std, skew, _ = diveye_stats(surprisals)
    cv = std / mean if mean > 0 else 0.0
    cv_score = clamp01((cv - 0.87) / (1.13 - 0.87))
    abs_skew = abs(skew)
    if abs_skew <= 0.3:
        skew_score = 0.5
    elif abs_skew >= 1.5:
        skew_score = 0.0
    else:
        skew_score = 0.5 - ((abs_skew - 0.3) / 1.2) * 0.5
    return cv_score * 0.85 + skew_score * 0.15


def burstiness_sentences(text: str) -> list[str]:
    """Sentences of >= 5 words, split after . ! ? and whitespace."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip().split()) >= 5]


def sentence_perplexities(sentences: list[str], token_losses: torch.Tensor,
                          encode: Callable[[str], list[int]]) -> list[float]:
    """Perplexity of each sentence's slice of the whole-text token losses.

    SlopTotal's approximate alignment: sentence i takes the next
    ``len(encode(sentence))`` positions (cut at the end), and only slices of >= 4
    tokens are kept. Token boundaries around sentence breaks are not checked.
    """
    pps = []
    offset, total = 0, token_losses.size(0)
    for sent in sentences:
        sent_len = min(len(encode(sent)), total - offset)
        if sent_len >= 4:
            pps.append(math.exp(token_losses[offset:offset + sent_len].mean().item()))
        offset += sent_len
        if offset >= total:
            break
    return pps


def burstiness_score(sentence_pps: list[float]) -> float:
    """Sample-stdev CV of sentence perplexities: <= 0.15 flat/AI (1), >= 0.8 human (0)."""
    mean = statistics.mean(sentence_pps)
    cv = statistics.stdev(sentence_pps) / mean if mean > 0 else 0
    return ramp_down(cv, 0.15, 0.8)


# --- engines ------------------------------------------------------------------

@dataclass
class StatisticalEngine:
    key: str
    name: str
    engine_type: str
    _score: Callable[[LMCache, str], float]
    cache: LMCache

    def score(self, text: str) -> float:
        return self._score(self.cache, text)


def _perplexity(lm: LMCache, text: str) -> float:
    out = lm.gpt2(text)
    return 0.0 if out.loss is None else perplexity_score(out.loss)


def _two_model(fn: Callable[[float, float], float]) -> Callable[[LMCache, str], float]:
    def score(lm: LMCache, text: str) -> float:
        gpt2, distil = lm.gpt2(text), lm.distil(text)
        if gpt2.loss is None or distil.loss is None:
            return 0.0
        return fn(gpt2.loss, distil.loss)
    return score


def _min_tokens(n: int, fn: Callable[[GPT2Output], float]) -> Callable[[LMCache, str], float]:
    def score(lm: LMCache, text: str) -> float:
        out = lm.gpt2(text)
        return 0.0 if out.stats is None or out.n_tokens < n else fn(out)
    return score


def _burstiness(lm: LMCache, text: str) -> float:
    out = lm.gpt2(text)
    if out.stats is None or out.n_tokens < 20:
        return 0.0
    sentences = burstiness_sentences(text)
    if len(sentences) < 5:
        return 0.0
    tokenizer = lm.tokenizer
    pps = sentence_perplexities(sentences, -out.stats.actual_lp,
                                lambda s: tokenizer.encode(s, add_special_tokens=False))
    return 0.0 if len(pps) < 4 else burstiness_score(pps)


LM_CACHE = LMCache()

STATISTICAL_ENGINES: list[StatisticalEngine] = [
    StatisticalEngine("binoculars", "Binoculars", "statistical",
                      _two_model(binoculars_score), LM_CACHE),
    StatisticalEngine("fast_detectgpt", "Fast-DetectGPT", "statistical",
                      _min_tokens(10, lambda o: fast_detectgpt_score(o.stats.curvature.mean().item())),
                      LM_CACHE),
    StatisticalEngine("perplexity", "Perplexity", "statistical", _perplexity, LM_CACHE),
    StatisticalEngine("cross_perplexity", "Cross-Perplexity", "statistical",
                      _two_model(cross_perplexity_score), LM_CACHE),
    StatisticalEngine("gltr", "GLTR", "statistical",
                      _min_tokens(10, lambda o: gltr_score(o.stats.ranks, o.n_tokens)), LM_CACHE),
    StatisticalEngine("log_rank", "Log-Rank", "statistical",
                      _min_tokens(10, lambda o: log_rank_score(o.stats.ranks)), LM_CACHE),
    StatisticalEngine("diveye", "DivEye", "statistical",
                      _min_tokens(10, lambda o: diveye_score((-o.stats.actual_lp).tolist())),
                      LM_CACHE),
    StatisticalEngine("burstiness", "Burstiness", "linguistic", _burstiness, LM_CACHE),
]


def unload() -> None:
    """Free the shared GPT-2 models and cache (GPU memory included)."""
    LM_CACHE.unload()
