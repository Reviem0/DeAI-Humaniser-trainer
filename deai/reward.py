"""Phase 3 reward: quality gates, then ``(1 - detector/100) * similarity``.

A rewrite that fails any gate gets 0, so the policy can't trade quality for a
lower detector score. Gates run cheapest first and the detector is only called
for rewrites that passed all of them:

- ``unicode``: NFKC-stable; no format/control/private-use characters (zero-width,
  bidi, soft hyphen, ...); no word mixing Latin with another script; no
  non-ASCII letter or mark the source doesn't already use (homoglyphs,
  diacritic tricks).
- ``length``: 0.8-1.2x the source's word count and >= 80 words, so the detectors'
  short-text shrinkage toward neutral can't be used.
- ``structure`` (optional, ``use_structure_gate``): the rewrite keeps the source's
  layout. Semicolons and clause joiners (; : — –) per 100 words, and the longest
  sentence, stay within the human 99th percentile or the source's own value,
  whichever is higher; the paragraph count stays within half to double the
  source's (+3). Blocks stitching sentences together with semicolons and merging
  paragraphs, which starves the sentence-level and structural engines (a
  round 7 rewrite turned 6 paragraphs into 1 with 14 semicolons, PLAN.md).
- ``meaning``: cosine similarity of all-mpnet-base-v2 passage embeddings >= 0.85.
  mpnet truncates at 384 tokens, so passages are embedded in sentence-aligned
  chunks and mean-pooled; otherwise the second half of a rewrite would go unseen.
- ``nli`` (optional): the sentences of each text must be entailed by the other
  text, in both directions (rewrite->source catches dropped facts, source->rewrite
  added or changed ones). A sentence counts as entailed if its max over windows
  of the other text is >= 0.5; each direction's score is the share of entailed
  sentences. A mean probability would let one flipped fact in eight sentences
  through, and the single worst sentence fails long passages on one NLI false
  negative (Phase 4 zero-shot rewrites, see PLAN.md).
- ``fluency``: mean P(acceptable) per sentence from a CoLA classifier, above a
  threshold calibrated on human passages (``scripts/calibrate_reward.py``).
  Perplexity is deliberately not used for this: low perplexity is what
  statistical detectors flag, so a perplexity floor would be a detector.
- ``perplexity`` and ``burstiness`` (optional, ``use_lm_gates``): upper bounds
  only, set at a high quantile of human passages. Perplexity is the rewrite's
  per-token perplexity, and burstiness is the coefficient of variation of its
  sentence perplexities, both under an LM independent of the detectors' GPT-2
  (``LM_MODEL``). They block the opposite shortcut: inflating surprise with
  rare or unidiomatic words, or a few deliberately odd sentences, which the
  statistical engines read as human and CoLA doesn't catch. There is no lower
  bound. Texts with fewer than ``min_burstiness_sentences`` scorable sentences
  pass the burstiness gate unmeasured, and since a sample CV of n values is at
  most sqrt(n), the default bound (2.51) can only fail texts of 7+ sentences.
- ``mesmer`` (optional, off by default): Mesmer AI Detector 9B (a Qwen3.5-9B
  classifier, outside SlopTotal) must not flag the rewrite. ``ai`` fails only its
  "ai" verdict; ``any`` also fails "ai_edited". The verdict uses the model's own
  ``policy.json`` thresholds (<= 0.7% of human texts called AI, <= 1% flagged).
  It runs last because it is by far the most expensive gate (~16 GB bf16, ~9 GB
  int8), so it is meant for the HPC.

``mesmer="soft"`` uses Mesmer as a reward term instead of a gate: it scores the
rewrites that passed every gate (beside the detector), and multiplies their
reward by ``mesmer_factor(p_human)``. As a gate Mesmer failed every base-model
rewrite (all "ai", PLAN.md Phase 4), which leaves DPO no pairs; the factor is on
a log scale so moving p_human from 1e-4 to 1e-3 already counts, and it never
goes below ``1 - mesmer_weight`` so the detector still ranks rewrites Mesmer
calls equally AI.

``mesmer="adaptive"`` is soft, plus the ``ai`` gate within a source once any of
its rewrites (among those that passed the other gates) gets a verdict other than
"ai": then that source's "ai" rewrites score 0. Rewrites of the same source are
recognised by identical source text, so a rewrite's reward depends on its
siblings in the same ``score()`` call; with one rewrite per source (validation)
it is just soft.

Models load lazily and are injectable, so tests can swap in stubs.
"""

from __future__ import annotations

import asyncio
import importlib.util
import math
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Protocol, Sequence

import numpy as np

from deai.detectors import DetectorError, Detectors, TextRejected

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
# Krishna et al. 2020's fluency classifier; label 0 = acceptable.
COLA_MODEL = "cointegrated/roberta-large-cola-krishna2020"
COLA_ACCEPTABLE = 0
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
NLI_ENTAILMENT = 1  # {0: contradiction, 1: entailment, 2: neutral}
# The perplexity/burstiness gates' LM: neither GPT-2 (the statistical engines', which
# stay held-out detectors) nor Qwen (the policy's family). Apache 2.0, 8k context.
LM_MODEL = "HuggingFaceTB/SmolLM2-360M"
LM_MAX_TOKENS = 2048  # passages are <= 600 words, ~800 tokens
LM_BATCH = 8  # logits are batch x tokens x 49k vocab
# CC BY-NC-SA 4.0. Pinned: its loader (mesmer_detector.py) is code from the repo.
MESMER_MODEL = "mesmertech/mesmer-ai-detector-9b"
MESMER_REVISION = "27483f365c703c8706a45a40355f55b8f93f7014"
# off: no gate; ai: fail the "ai" verdict; any: fail "ai" and "ai_edited";
# soft: no gate, the reward is scaled by mesmer_factor(p_human); adaptive: soft, plus the
# "ai" gate among a source's rewrites once any of them gets a verdict other than "ai".
MESMER_MODES = ("off", "ai", "any", "soft", "adaptive")
# Mesmer's predict() reads only the first 1,023 tokens; longer rewrites go through
# its windowed analyze() so their end can't hide AI text. ~1.3 tokens per word.
MESMER_PREDICT_MAX_WORDS = 700


@dataclass(frozen=True)
class RewardConfig:
    min_ratio: float = 0.8
    max_ratio: float = 1.2
    min_words: int = 80
    min_similarity: float = 0.85
    # 2nd percentile of train+val human passages (scripts/calibrate_reward.py); see PLAN.md.
    min_fluency: float = 0.70
    # Share of sentences entailed, per direction: a 5-sentence passage must keep
    # all 5; 1 miss is tolerated from 10 sentences up.
    min_entailment: float = 0.9
    nli_sentence_threshold: float = 0.5
    use_nli: bool = False
    # Perplexity / burstiness upper bounds (LM_MODEL), at the 98th percentile of
    # train+val human passages (scripts/calibrate_reward.py); see PLAN.md.
    use_lm_gates: bool = False
    max_perplexity: float = 39.5
    max_burstiness: float = 2.51
    min_burstiness_sentences: int = 4
    min_sentence_tokens: int = 4
    # Structure: human 99th percentiles (train+val, scripts/calibrate_reward.py), each
    # raised to the source's own value when the source is higher.
    use_structure_gate: bool = False
    max_semicolons: float = 1.8  # per 100 words
    max_joiners: float = 3.3  # ; : — – per 100 words
    max_sentence_words: int = 105
    min_paragraph_ratio: float = 0.5  # of the source's paragraphs, rounded up
    max_paragraph_ratio: float = 2.0  # or the source's + max_extra_paragraphs if higher
    max_extra_paragraphs: int = 3
    mesmer: str = "off"  # one of MESMER_MODES
    # soft mode: share of the reward Mesmer controls, and the p_human at which its
    # term bottoms out (log scale from there up to p_human = 1).
    mesmer_weight: float = 0.5
    mesmer_floor: float = 1e-6
    chunk_words: int = 150  # mpnet's 384-token limit is ~250 words; leave margin


def mesmer_factor(p_human: float, weight: float = 0.5, floor: float = 1e-6) -> float:
    """Reward multiplier in [1 - weight, 1]: ``1 - weight`` at p_human <= floor, 1 at
    p_human = 1, linear in log(p_human) between (each decade adds the same amount)."""
    m = math.log(max(p_human, floor) / floor) / math.log(1 / floor)
    return 1 - weight + weight * min(1.0, m)


# ----------------------------------------------------------------------
# Text helpers
# ----------------------------------------------------------------------

_SENT_END = re.compile(r"(?:(?<=[.!?…])|(?<=[.!?…][\"'”’)\]]))\s+(?=[\"'“‘(\[]?[A-Z0-9])")


def word_count(text: str) -> int:
    return len(text.split())  # same definition as deai.data


def split_sentences(text: str) -> list[str]:
    out = []
    for para in re.split(r"\n\s*\n|\n", text):
        out.extend(s.strip() for s in _SENT_END.split(para) if s.strip())
    return out


def paragraphs(text: str) -> list[str]:
    """Non-empty lines (split_sentences' paragraph split)."""
    return [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]


JOINERS = ";:—–"


@dataclass(frozen=True)
class Structure:
    semicolons: float  # per 100 words
    joiners: float  # ; : — – per 100 words
    longest_sentence: int  # words
    paragraphs: int


def structure(text: str) -> Structure:
    words = max(word_count(text), 1)
    sents = split_sentences(text) or [text]
    return Structure(100 * text.count(";") / words, 100 * sum(text.count(c) for c in JOINERS) / words,
                     max(word_count(s) for s in sents), len(paragraphs(text)) or 1)


def structure_problems(rewrite: str, source: str, cfg: "RewardConfig") -> list[str]:
    """Reasons the rewrite fails the structure gate (empty = passes)."""
    rw, src = structure(rewrite), structure(source)
    problems = []
    if rw.semicolons > max(cfg.max_semicolons, src.semicolons):
        problems.append(f"{rw.semicolons:.2f} semicolons/100 words (source {src.semicolons:.2f})")
    if rw.joiners > max(cfg.max_joiners, src.joiners):
        problems.append(f"{rw.joiners:.2f} ;:—– per 100 words (source {src.joiners:.2f})")
    if rw.longest_sentence > max(cfg.max_sentence_words, src.longest_sentence):
        problems.append(f"{rw.longest_sentence}-word sentence (source {src.longest_sentence})")
    lo = math.ceil(cfg.min_paragraph_ratio * src.paragraphs)
    hi = max(math.floor(cfg.max_paragraph_ratio * src.paragraphs), src.paragraphs + cfg.max_extra_paragraphs)
    if not lo <= rw.paragraphs <= hi:
        problems.append(f"{rw.paragraphs} paragraphs (source {src.paragraphs}, allowed {lo}-{hi})")
    return problems


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of ``split_sentences(text)`` in ``text``."""
    spans, pos = [], 0
    for s in split_sentences(text):
        start = text.find(s, pos)
        if start < 0:  # can't happen (sentences are stripped substrings, in order)
            continue
        spans.append((start, start + len(s)))
        pos = start + len(s)
    return spans


@dataclass(frozen=True)
class LMStats:
    perplexity: float
    burstiness: float | None  # None: fewer than min_sentences scorable sentences
    sentence_perplexities: list[float]


def lm_stats(text: str, nll: Sequence[float], spans: Sequence[tuple[int, int]],
             min_sentence_tokens: int = 4, min_sentences: int = 4) -> LMStats:
    """Perplexity and burstiness from per-token NLLs (nats) and their character spans.

    A token belongs to the sentence its last character falls in, so a token that
    carries the space before a sentence counts for that sentence; whitespace
    tokens between sentences count for none. Burstiness is the sample-stdev CV of
    the per-sentence perplexities of sentences with >= ``min_sentence_tokens``.
    """
    if not len(nll):
        return LMStats(math.inf, None, [])
    nll = np.asarray(nll, dtype=np.float64)
    ppl = float(math.exp(nll.mean()))
    ends = np.array([e for _, e in spans])
    pps = []
    for s0, s1 in sentence_spans(text):
        sel = nll[(ends > s0) & (ends <= s1)]
        if len(sel) >= min_sentence_tokens:
            pps.append(float(math.exp(sel.mean())))
    burst = None
    if len(pps) >= max(min_sentences, 2):
        burst = float(np.std(pps, ddof=1) / np.mean(pps))
    return LMStats(ppl, burst, pps)


def chunk_sentences(sentences: list[str], max_words: int) -> list[str]:
    """Greedy runs of whole sentences up to ``max_words`` (a long sentence stands alone)."""
    chunks, cur, n = [], [], 0
    for s in sentences:
        w = word_count(s)
        if cur and n + w > max_words:
            chunks.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks


# ----------------------------------------------------------------------
# Cheap gates
# ----------------------------------------------------------------------

_ALLOWED_CONTROL = {"\n", "\r", "\t"}


def _script(ch: str) -> str:
    """First word of the Unicode name: LATIN, CYRILLIC, GREEK, ... (approximate but enough)."""
    return unicodedata.name(ch, "UNKNOWN").split(" ")[0]


def _strip_punct(word: str) -> str:
    return re.sub(r"^\W+|\W+$", "", word)


def unicode_problems(rewrite: str, source: str = "") -> list[str]:
    """Reasons the rewrite fails the unicode gate (empty = passes).

    Characters and words the source already contains are exempt: an AI source
    may legitimately use a non-breaking hyphen or ``ℝⁿ``, and keeping them isn't
    a trick the rewriter invented.
    """
    problems = []
    src_chars = set(source)
    if unicodedata.normalize("NFKC", rewrite) != rewrite:
        unstable = {c for c in rewrite if unicodedata.normalize("NFKC", c) != c}
        new = sorted(unstable - src_chars)
        if new or not unstable:  # no single unstable char = a composing sequence
            problems.append("changed by NFKC: " + " ".join(f"U+{ord(c):04X}" for c in new[:5]))
    bad_cats = sorted({f"U+{ord(c):04X}" for c in rewrite
                       if unicodedata.category(c) in ("Cf", "Co", "Cs", "Cn")
                       or (unicodedata.category(c) == "Cc" and c not in _ALLOWED_CONTROL)})
    if bad_cats:
        problems.append("invisible/format characters " + " ".join(bad_cats[:5]))
    src_words = {_strip_punct(w) for w in source.split()}
    for word in rewrite.split():
        scripts = {_script(c) for c in word if c.isalpha()}
        if "LATIN" in scripts and len(scripts) > 1 and _strip_punct(word) not in src_words:
            problems.append(f"mixed-script word {word!r}")
            break
    new = sorted({c for c in rewrite if ord(c) > 127 and unicodedata.category(c)[0] in "LM"
                  and c not in src_chars})
    if new:
        problems.append("non-ASCII letters not in source: " + "".join(new[:10]))
    return problems


# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------

@dataclass
class Gate:
    name: str
    passed: bool
    value: float | None = None
    detail: str = ""


@dataclass
class RewardResult:
    reward: float | None  # None only if the detector failed; gate failures give 0.0
    gates: list[Gate] = field(default_factory=list)
    similarity: float | None = None
    detector_score: float | None = None  # 0-100
    error: str | None = None
    # mesmer="soft": Mesmer's verdict and (p_human, p_ai, p_ai_edited), for rewrites that passed
    mesmer_verdict: str | None = None
    mesmer_probs: tuple[float, float, float] | None = None

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.gates)

    @property
    def failed_gate(self) -> str | None:
        return next((g.name for g in self.gates if not g.passed), None)

    def gate(self, name: str) -> Gate | None:
        return next((g for g in self.gates if g.name == name), None)


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------

class QualityModels(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...  # unit-normalised
    def acceptability(self, sentences: list[str]) -> list[float]: ...
    def entailment(self, premises: list[str], hypotheses: list[str]) -> list[float]: ...
    # Only needed with use_lm_gates: per text, the NLL (nats) of each token given its
    # prefix (the first after a BOS) and each token's (start, end) characters in the text.
    def token_nll(self, texts: list[str]) -> list[tuple[list[float], list[tuple[int, int]]]]: ...
    # Only needed when mesmer is on: (verdict, (p_human, p_ai, p_ai_edited)) per text.
    def ai_verdicts(self, texts: list[str]) -> list[tuple[str, tuple[float, float, float]]]: ...


class HFQualityModels:
    """The real models, loaded on first use.

    Mesmer is configured by environment, since it only runs on machines that fit it:
    ``DEAI_MESMER_DEVICE`` (default: this object's device), ``DEAI_MESMER_INT8=1``
    (8-bit weights, ~9 GB instead of ~16) and ``DEAI_MESMER_TOKENS_PER_BATCH``.
    """

    def __init__(self, device: str | None = None, batch_size: int = 32):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self._embedder = None
        self._cola = None
        self._nli = None
        self._lm = None
        self._mesmer = None

    def embed(self, texts):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(EMBED_MODEL, device=self.device)
        return self._embedder.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                     convert_to_numpy=True).tolist()

    def acceptability(self, sentences):
        if self._cola is None:
            self._cola = self._load_classifier(COLA_MODEL)
        return self._classify(self._cola, sentences, None, COLA_ACCEPTABLE)

    def entailment(self, premises, hypotheses):
        if self._nli is None:
            self._nli = self._load_classifier(NLI_MODEL)
        return self._classify(self._nli, premises, hypotheses, NLI_ENTAILMENT)

    def token_nll(self, texts):
        import torch
        import torch.nn.functional as F
        if self._lm is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(LM_MODEL)
            dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
            model = AutoModelForCausalLM.from_pretrained(LM_MODEL, dtype=dtype)
            self._lm = tok, model.to(self.device).eval()
        tok, model = self._lm
        bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        pad = tok.pad_token_id if tok.pad_token_id is not None else bos
        enc = [tok(t, add_special_tokens=False, return_offsets_mapping=True, truncation=True,
                   max_length=LM_MAX_TOKENS - 1) for t in texts]
        out = []
        for i in range(0, len(enc), LM_BATCH):
            batch = [[bos, *e["input_ids"]] for e in enc[i:i + LM_BATCH]]
            width = max(len(b) for b in batch)
            ids = torch.full((len(batch), width), pad, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for j, b in enumerate(batch):
                ids[j, :len(b)] = torch.tensor(b)
                mask[j, :len(b)] = 1
            ids, mask = ids.to(self.device), mask.to(self.device)
            with torch.no_grad():
                logits = model(input_ids=ids, attention_mask=mask).logits
                for j, (b, e) in enumerate(zip(batch, enc[i:i + LM_BATCH])):
                    n = len(b) - 1
                    nll = F.cross_entropy(logits[j, :n].float(), ids[j, 1:n + 1], reduction="none")
                    out.append((nll.tolist(), [tuple(o) for o in e["offset_mapping"]]))
            del logits
        return out

    def ai_verdicts(self, texts):
        if self._mesmer is None:
            self._mesmer = load_mesmer(os.environ.get("DEAI_MESMER_DEVICE") or self.device)
        det = self._mesmer
        out: list = [None] * len(texts)
        short = [i for i, t in enumerate(texts) if word_count(t) <= MESMER_PREDICT_MAX_WORDS]
        if short:
            for i, p in zip(short, det.predict([texts[i] for i in short])):
                out[i] = (det.verdict(p), tuple(float(x) for x in p))
        for i in (i for i, o in enumerate(out) if o is None):
            r = det.analyze(texts[i])
            out[i] = (r["verdict"], tuple(float(r["probs"][k]) for k in ("human", "ai", "ai_edited")))
        return out

    def release(self) -> None:
        """Drop the models and return their GPU memory; they reload on next use."""
        self._embedder = self._cola = self._nli = self._lm = self._mesmer = None
        import gc

        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_classifier(self, name):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name, dtype=dtype)
        return tok, model.to(self.device).eval()

    def _classify(self, tok_model, a, b, label):
        import torch
        tok, model = tok_model
        out = []
        for i in range(0, len(a), self.batch_size):
            args = [a[i:i + self.batch_size]]
            if b is not None:
                args.append(b[i:i + self.batch_size])
            enc = tok(*args, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                p = model(**enc.to(self.device)).logits.float().softmax(-1)[:, label]
            out.extend(p.tolist())
        return out


def mesmer_int8() -> bool:
    return os.environ.get("DEAI_MESMER_INT8", "").lower() in ("1", "true", "yes")


def mesmer_snapshot(int8: bool | None = None) -> str:
    """Local folder of Mesmer at the pinned revision (from the HF cache when offline)."""
    from huggingface_hub import snapshot_download
    int8 = mesmer_int8() if int8 is None else int8
    weights = "int8/model.safetensors" if int8 else "model.safetensors"
    return snapshot_download(MESMER_MODEL, revision=MESMER_REVISION,
                             allow_patterns=["*.json", "mesmer_detector.py", weights])


def load_mesmer(device: str | None = None):
    """Mesmer's own ``Detector`` (from the repo's mesmer_detector.py) at the pinned revision."""
    int8 = mesmer_int8()
    path = mesmer_snapshot(int8)
    spec = importlib.util.spec_from_file_location("mesmer_detector", os.path.join(path, "mesmer_detector.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    kw = {"tokens_per_batch": int(os.environ["DEAI_MESMER_TOKENS_PER_BATCH"])} \
        if os.environ.get("DEAI_MESMER_TOKENS_PER_BATCH") else {}
    return module.Detector.from_pretrained(path, int8=int8, device=device, **kw)


# ----------------------------------------------------------------------
# Detectors
# ----------------------------------------------------------------------

Detector = Callable[[list[str]], Awaitable[list[float]]]  # texts -> 0-100 each

# Detector families for the transfer study, by SlopTotal's engine_type. The
# "classifier" and "embedding" engines are fine-tuned transformers too.
FAMILIES: dict[str, tuple[str, ...]] = {
    "neural": ("neural", "classifier", "embedding"),
    "statistical": ("statistical",),
    "heuristic": ("linguistic",),
}

# Phase 4 reward sources: name -> (endpoint, family or None for the overall score).
REWARD_SOURCES: dict[str, tuple[str, str | None]] = {
    "quick": ("quick", None),             # R-quick, ~Fakespot
    "neural-quick": ("quick", "neural"),  # the 4 neural engines quick-score runs
    "neural": ("analyze", "neural"),      # R-neural
    "stat": ("analyze", "statistical"),   # R-stat
    "full": ("analyze", None),            # R-full
}


def family_engines(detectors: Detectors, endpoint: str, family: str) -> list[str]:
    """Keys of a family's engines among those the endpoint runs."""
    types = FAMILIES[family]
    info = detectors.engines()
    return sorted(k for k in detectors.endpoint_engines(endpoint) if info[k].engine_type in types)


def make_detector(detectors: Detectors, reward_source: str) -> Detector:
    endpoint, family = REWARD_SOURCES[reward_source]
    engines = family_engines(detectors, endpoint, family) if family else None
    if engines == []:
        raise ValueError(f"{endpoint} runs no {family} engines")
    return local_detector(detectors, endpoint, engines)


def local_detector(detectors: Detectors, endpoint: str = "quick",
                   engines: Sequence[str] | None = None) -> Detector:
    """Overall score, or with ``engines`` the mean of those engines (x100): R-neural, R-stat.

    A text that can't be scored comes back as its exception; ``Reward.score`` turns
    that into ``reward=None``.
    """
    async def detect(texts: list[str]) -> list[float]:
        results = await asyncio.to_thread(detectors.score_many, texts, endpoint,  # type: ignore[arg-type]
                                          engines or (), engines is None)
        out = []
        for r in results:
            if isinstance(r, BaseException):
                out.append(r)
            elif engines is None:
                out.append(r.overall)
            else:
                out.append(100 * sum(r.engines[k] for k in engines) / len(engines))
        return out
    return detect


# ----------------------------------------------------------------------
# Reward
# ----------------------------------------------------------------------

class Reward:
    def __init__(self, detector: Detector | None = None, models: QualityModels | None = None,
                 config: RewardConfig = RewardConfig()):
        self.detector = detector
        self.models = models or HFQualityModels()
        self.config = config

    async def score(self, sources: Sequence[str], rewrites: Sequence[str]) -> list[RewardResult]:
        results = self.check_gates(sources, rewrites)
        todo = [i for i, r in enumerate(results) if r.passed]
        if not todo:
            return results
        if self.detector is None:
            raise RuntimeError("Reward has no detector; use check_gates() for gates only")
        cfg = self.config
        if cfg.mesmer in ("soft", "adaptive"):
            for i, (verdict, probs) in zip(todo, self.models.ai_verdicts([rewrites[i].strip() for i in todo])):
                results[i].mesmer_verdict, results[i].mesmer_probs = verdict, tuple(probs)
        if cfg.mesmer == "adaptive":
            # Gate within each source's rewrites once at least one of them gets past Mesmer.
            clean = {sources[i] for i in todo if results[i].mesmer_verdict != "ai"}
            for i in todo:
                if sources[i] in clean:
                    r = results[i]
                    h, a, e = r.mesmer_probs
                    r.gates.append(Gate("mesmer", r.mesmer_verdict != "ai", h,
                                        f"{r.mesmer_verdict}: human {h:.3f}, ai {a:.3f}, ai_edited {e:.3f}"))
                    if not r.passed:
                        r.reward = 0.0  # a gate failure, not a detector failure (None)
            todo = [i for i in todo if results[i].passed]
        texts = [rewrites[i].strip() for i in todo]
        try:
            scores = await self.detector(texts)
        except (DetectorError, TextRejected) as e:
            scores = [e] * len(todo)
        for i, s in zip(todo, scores):
            r = results[i]
            if isinstance(s, BaseException):
                r.reward, r.error = None, f"{type(s).__name__}: {s}"
                continue
            r.detector_score = float(s)
            r.reward = (1 - r.detector_score / 100) * r.similarity
            if r.mesmer_probs is not None:
                r.reward *= mesmer_factor(r.mesmer_probs[0], cfg.mesmer_weight, cfg.mesmer_floor)
        return results

    def check_gates(self, sources: Sequence[str], rewrites: Sequence[str],
                    use_nli: bool | None = None) -> list[RewardResult]:
        """Run every gate up to the first failure. Passing results have ``reward=None`` until scored."""
        if len(sources) != len(rewrites):
            raise ValueError("sources and rewrites differ in length")
        cfg = self.config
        if cfg.mesmer not in MESMER_MODES:
            raise ValueError(f"mesmer must be one of {MESMER_MODES}, got {cfg.mesmer!r}")
        use_nli = cfg.use_nli if use_nli is None else use_nli
        results = [RewardResult(reward=0.0) for _ in rewrites]

        def alive() -> list[int]:
            return [i for i, r in enumerate(results) if r.passed]

        for i, (src, rw) in enumerate(zip(sources, rewrites)):
            problems = unicode_problems(rw, src)
            results[i].gates.append(Gate("unicode", not problems, detail="; ".join(problems)))
            if problems:
                continue
            n_src, n_rw = word_count(src), word_count(rw)
            ratio = n_rw / n_src if n_src else math.inf
            ok = cfg.min_ratio <= ratio <= cfg.max_ratio and n_rw >= cfg.min_words
            results[i].gates.append(Gate("length", ok, ratio, f"{n_rw} words vs {n_src}"))
            if ok and cfg.use_structure_gate:
                problems = structure_problems(rw, src, cfg)
                results[i].gates.append(Gate("structure", not problems, detail="; ".join(problems)))

        idx = alive()
        if idx:
            sims = self.similarity([sources[i] for i in idx], [rewrites[i] for i in idx])
            for i, sim in zip(idx, sims):
                results[i].similarity = sim
                results[i].gates.append(Gate("meaning", sim >= cfg.min_similarity, sim))

        idx = alive()
        if idx:
            for i, f in zip(idx, self.fluency([rewrites[i] for i in idx])):
                results[i].gates.append(Gate("fluency", f >= cfg.min_fluency, f))

        idx = alive()
        if cfg.use_lm_gates and idx:
            for i, st in zip(idx, self.lm_stats([rewrites[i] for i in idx])):
                g = results[i].gates
                g.append(Gate("perplexity", st.perplexity <= cfg.max_perplexity, st.perplexity,
                              f"max {cfg.max_perplexity:g}"))
                if g[-1].passed:
                    b = st.burstiness
                    g.append(Gate("burstiness", b is None or b <= cfg.max_burstiness, b,
                                  f"{len(st.sentence_perplexities)} sentences, max {cfg.max_burstiness:g}"
                                  + (" (too few to measure)" if b is None else "")))

        idx = alive()
        if use_nli and idx:
            for i, (fwd, back) in zip(idx, self.entailment([sources[i] for i in idx],
                                                          [rewrites[i] for i in idx])):
                results[i].gates.append(Gate("nli", min(fwd, back) >= cfg.min_entailment,
                                             min(fwd, back), f"src->rw {fwd:.3f}, rw->src {back:.3f}"))

        idx = alive()
        if cfg.mesmer in ("ai", "any") and idx:
            flagged = ("ai",) if cfg.mesmer == "ai" else ("ai", "ai_edited")
            for i, (verdict, (h, a, e)) in zip(idx, self.models.ai_verdicts([rewrites[i].strip() for i in idx])):
                results[i].gates.append(Gate("mesmer", verdict not in flagged, h,
                                             f"{verdict}: human {h:.3f}, ai {a:.3f}, ai_edited {e:.3f}"))

        for i in alive():
            results[i].reward = None
        return results

    # -- model-backed measures, batched over passages -------------------

    def similarity(self, a: Sequence[str], b: Sequence[str]) -> list[float]:
        """Cosine of mean-pooled chunk embeddings."""
        chunks = [chunk_sentences(split_sentences(t), self.config.chunk_words) or [t]
                  for t in [*a, *b]]
        vecs = np.asarray(self.models.embed([c for cs in chunks for c in cs]), dtype=np.float64)
        bounds = np.cumsum([0] + [len(cs) for cs in chunks])
        pooled = np.stack([vecs[s:e].mean(0) for s, e in zip(bounds[:-1], bounds[1:])])
        pooled /= np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-12)
        n = len(a)
        return (pooled[:n] * pooled[n:]).sum(1).tolist()

    def fluency(self, texts: Sequence[str]) -> list[float]:
        """Mean P(acceptable) over each text's sentences."""
        sents = [split_sentences(t) or [t] for t in texts]
        probs = self.models.acceptability([s for ss in sents for s in ss])
        bounds = np.cumsum([0] + [len(ss) for ss in sents])
        return [float(np.mean(probs[s:e])) for s, e in zip(bounds[:-1], bounds[1:])]

    def lm_stats(self, texts: Sequence[str]) -> list[LMStats]:
        """Perplexity and burstiness of each text under ``LM_MODEL``."""
        cfg = self.config
        return [lm_stats(t, nll, spans, cfg.min_sentence_tokens, cfg.min_burstiness_sentences)
                for t, (nll, spans) in zip(texts, self.models.token_nll(list(texts)))]

    def entailment(self, sources: Sequence[str], rewrites: Sequence[str]) -> list[tuple[float, float]]:
        """(source entails rewrite, rewrite entails source) per pair: share of entailed sentences."""
        # Premises are ~3-sentence windows of the other text, which keeps pairs under 512 tokens.
        premises, hyps, owner = [], [], []
        for p_i, (src, rw) in enumerate(zip(sources, rewrites)):
            for d, (prem, hyp) in enumerate(((src, rw), (rw, src))):
                windows = _windows(split_sentences(prem) or [prem])
                for h_i, h in enumerate(split_sentences(hyp) or [hyp]):
                    premises.extend(windows)
                    hyps.extend([h] * len(windows))
                    owner.extend([(p_i, d, h_i)] * len(windows))
        best: dict[tuple[int, int, int], float] = {}
        for key, p in zip(owner, self.models.entailment(premises, hyps)):
            best[key] = max(best.get(key, 0.0), p)
        per_dir: dict[tuple[int, int], list[float]] = {}
        for (p_i, d, _), v in best.items():
            per_dir.setdefault((p_i, d), []).append(v)
        t = self.config.nli_sentence_threshold
        return [tuple(float(np.mean(np.array(per_dir.get((p_i, d), [0.0])) >= t)) for d in (0, 1))
                for p_i in range(len(sources))]


def _windows(sentences: list[str], size: int = 3, stride: int = 2) -> list[str]:
    if len(sentences) <= size:
        return [" ".join(sentences)]
    return [" ".join(sentences[i:i + size]) for i in range(0, len(sentences) - size + stride, stride)]
