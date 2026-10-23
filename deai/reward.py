"""Phase 3 reward: quality gates, then ``(1 - detector/100) * similarity``.

A rewrite that fails any gate gets 0, so the policy can't trade quality for a
lower detector score. Gates run cheapest first and the detector is only called
for rewrites that passed all of them:

- ``unicode``: NFKC-stable; no format/control/private-use characters (zero-width,
  bidi, soft hyphen, ...); no word mixing Latin with another script; no
  non-ASCII letter or mark the source doesn't already use (homoglyphs,
  diacritic tricks).
- ``length``: 0.8-1.2x the source's word count and >= 80 words, so SlopTotal's
  short-text shrinkage toward neutral can't be used.
- ``meaning``: cosine similarity of all-mpnet-base-v2 passage embeddings >= 0.85.
  mpnet truncates at 384 tokens, so passages are embedded in sentence-aligned
  chunks and mean-pooled; otherwise the second half of a rewrite would go unseen.
- ``nli`` (optional, for validation): every sentence of each text must be
  entailed by the other text, in both directions. A sentence's entailment is its
  max over windows of the other text; the passage score is the *least* entailed
  sentence, since one flipped fact in eight sentences barely moves a mean.
- ``fluency``: mean P(acceptable) per sentence from a CoLA classifier, above a
  threshold calibrated on human passages (``scripts/calibrate_reward.py``).
  Perplexity is deliberately not used: low perplexity is what statistical
  detectors flag.

Models load lazily and are injectable, so tests can swap in stubs.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Protocol, Sequence

import numpy as np

from deai.harness import HarnessError, SlopTotalClient, TextRejected

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
# Krishna et al. 2020's fluency classifier; label 0 = acceptable.
COLA_MODEL = "cointegrated/roberta-large-cola-krishna2020"
COLA_ACCEPTABLE = 0
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
NLI_ENTAILMENT = 1  # {0: contradiction, 1: entailment, 2: neutral}


@dataclass(frozen=True)
class RewardConfig:
    min_ratio: float = 0.8
    max_ratio: float = 1.2
    min_words: int = 80
    min_similarity: float = 0.85
    # 2nd percentile of train+val human passages (scripts/calibrate_reward.py); see PLAN.md.
    min_fluency: float = 0.70
    # Provisional: re-check on Phase 4's zero-shot rewrites.
    min_entailment: float = 0.5
    use_nli: bool = False
    chunk_words: int = 150  # mpnet's 384-token limit is ~250 words; leave margin


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


class HFQualityModels:
    """The real models, loaded on first use."""

    def __init__(self, device: str | None = None, batch_size: int = 32):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self._embedder = None
        self._cola = None
        self._nli = None

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


# ----------------------------------------------------------------------
# Detectors
# ----------------------------------------------------------------------

Detector = Callable[[list[str]], Awaitable[list[float]]]  # texts -> 0-100 each


def sloptotal_detector(client: SlopTotalClient, endpoint: str = "quick",
                       engines: Sequence[str] | None = None) -> Detector:
    """Overall score, or with ``engines`` the mean of those engines (x100): R-neural, R-stat.

    A text the server can't score raises; ``Reward.score`` turns that into ``reward=None``.
    """
    async def detect(texts: list[str]) -> list[float]:
        results = await client.score_many(texts, endpoint)  # type: ignore[arg-type]
        out = []
        for r in results:
            if isinstance(r, BaseException):
                out.append(r)
            elif engines is None:
                out.append(r.overall)
            else:
                vals = [r.engines[k] for k in engines if k in r.engines]
                out.append(100 * sum(vals) / len(vals) if vals else HarnessError(
                    f"none of {list(engines)} in {endpoint} result"))
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
        try:
            scores = await self.detector([rewrites[i].strip() for i in todo])
        except (HarnessError, TextRejected) as e:
            scores = [e] * len(todo)
        for i, s in zip(todo, scores):
            r = results[i]
            if isinstance(s, BaseException):
                r.reward, r.error = None, f"{type(s).__name__}: {s}"
                continue
            r.detector_score = float(s)
            r.reward = (1 - r.detector_score / 100) * r.similarity
        return results

    def check_gates(self, sources: Sequence[str], rewrites: Sequence[str],
                    use_nli: bool | None = None) -> list[RewardResult]:
        """Run every gate up to the first failure. Passing results have ``reward=None`` until scored."""
        if len(sources) != len(rewrites):
            raise ValueError("sources and rewrites differ in length")
        cfg = self.config
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
        if use_nli and idx:
            for i, (fwd, back) in zip(idx, self.entailment([sources[i] for i in idx],
                                                          [rewrites[i] for i in idx])):
                results[i].gates.append(Gate("nli", min(fwd, back) >= cfg.min_entailment,
                                             min(fwd, back), f"src->rw {fwd:.3f}, rw->src {back:.3f}"))

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

    def entailment(self, sources: Sequence[str], rewrites: Sequence[str]) -> list[tuple[float, float]]:
        """(source entails rewrite, rewrite entails source) per pair, each the worst sentence's."""
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
        return [tuple(float(min(per_dir.get((p_i, d), [0.0]))) for d in (0, 1))
                for p_i in range(len(sources))]


def _windows(sentences: list[str], size: int = 3, stride: int = 2) -> list[str]:
    if len(sentences) <= size:
        return [" ".join(sentences)]
    return [" ".join(sentences[i:i + size]) for i in range(0, len(sentences) - size + stride, stride)]
