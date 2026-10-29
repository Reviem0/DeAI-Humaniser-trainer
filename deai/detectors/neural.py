"""The 9 transformer classifier engines of SlopTotal, run in-process.

Ports of SlopTotal's (https://github.com/pablocaeg/sloptotal) ``app/engines/
classifier_*.py`` and the windowing in ``app/engines/base.py``. ``score(text)``
reproduces its single-text scores (fp32, no padding); ``score_many`` batches the
windows of many texts for throughput, which padding moves by well under 1e-3.

Each model loads on its engine's first call, onto ``detector_device()``, and
``unload()`` frees it. Loads share one module lock: SlopTotal found concurrent
``from_pretrained`` calls could leave weights randomly initialised.
"""

from __future__ import annotations

import gc
import threading
from collections.abc import Iterable, Mapping

from deai.detectors.base import detector_device

WINDOW = 510  # tokens per window, leaving room for the 2 special tokens of 512
STRIDE = 256
MAX_WINDOWS = 8
# SlopTotal's 3 large models step a whole window at a time and score at most 4.
LARGE_MODEL_MAX_WINDOWS = 4
MIN_WINDOW_TOKENS = 20  # a shorter tail window ends the loop

_LOAD_LOCK = threading.RLock()

AI_KEYWORDS = ("ai", "fake", "generated", "machine")


def window_starts(n_tokens: int, window: int = WINDOW, stride: int = STRIDE,
                  max_windows: int = MAX_WINDOWS) -> list[int]:
    """Token offsets of the windows scored over a long text.

    Up to ``max_windows`` windows every ``stride`` tokens; beyond that,
    ``max_windows`` spread evenly from 0 to ``n_tokens - window`` (Python's
    round, so halves go to even).
    """
    starts = list(range(0, n_tokens, stride))
    if len(starts) <= max_windows:
        return starts
    last = max(0, n_tokens - window)
    return [round(i * last / (max_windows - 1)) for i in range(max_windows)]


def ai_label_index(id2label: Mapping, keywords: Iterable[str] = AI_KEYWORDS,
                   default: int | None = 1) -> int | None:
    """First label index whose lower-cased name contains a keyword, else ``default``.

    ``default=None`` falls back to the largest index (SlopTotal's ChatGPT engine).
    """
    keywords = tuple(keywords)
    for idx, label in id2label.items():
        if any(kw in str(label).lower() for kw in keywords):
            return int(idx)
    if default is None:
        return max(int(k) for k in id2label)
    return default


def window_texts(text: str, tokenizer, stride: int = STRIDE,
                 max_windows: int = MAX_WINDOWS) -> list[str]:
    """The texts a classifier scores for ``text``: itself if it fits one window,
    else each window's ids decoded back to text. Empty means score 0."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= WINDOW:
        return [text]
    chunks = []
    for start in window_starts(len(tokens), WINDOW, stride, max_windows):
        ids = tokens[start:start + WINDOW]
        if len(ids) < MIN_WINDOW_TOKENS:
            break
        chunks.append(tokenizer.decode(ids, skip_special_tokens=True))
    return chunks


class ClassifierEngine:
    """One HF sequence classifier, windowed and averaged as SlopTotal does."""

    key: str
    name: str
    engine_type: str = "neural"
    model_name: str
    stride: int = STRIDE
    max_windows: int = MAX_WINDOWS

    def __init__(self, device: str | None = None, batch_size: int = 16):
        self.device = device
        self.batch_size = batch_size  # windows per forward pass in score_many
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.key!r})"

    # -- loading ------------------------------------------------------------

    def _load_hf(self):
        """(tokenizer, model) on the CPU; overridden by the custom-head engines."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        return (AutoTokenizer.from_pretrained(self.model_name),
                AutoModelForSequenceClassification.from_pretrained(self.model_name))

    def load(self):
        """Load the model if needed; returns (tokenizer, model)."""
        if self._model is None:
            with _LOAD_LOCK:
                if self._model is None:
                    tokenizer, model = self._load_hf()
                    self._prepare(model)
                    self._tokenizer = tokenizer
                    # Publish the model last: callers test it to skip the lock.
                    self._model = model.to(self.device or detector_device()).eval()
        return self._tokenizer, self._model

    def _prepare(self, model) -> None:
        """Read anything the scoring needs from the loaded config."""

    def unload(self) -> None:
        """Drop the model and return its GPU memory."""
        with self._lock:  # waits for a score in progress
            self._model = self._tokenizer = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- scoring ------------------------------------------------------------

    def _ai_prob(self, logits):
        """P(AI) per row of a (batch, num_labels) logits tensor."""
        raise NotImplementedError

    def _probs(self, tokenizer, model, chunks: list[str], batch_size: int) -> list[float]:
        import torch
        device = next(model.parameters()).device
        # Similar lengths together, so padding stays short.
        order = sorted(range(len(chunks)), key=lambda i: len(chunks[i]))
        probs = [0.0] * len(chunks)
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            enc = tokenizer([chunks[i] for i in idx], return_tensors="pt", truncation=True,
                            max_length=512, padding=len(idx) > 1)
            with torch.no_grad():
                res = model(**{k: v.to(device) for k, v in enc.items()})
            logits = getattr(res, "logits", res)
            for i, p in zip(idx, self._ai_prob(logits).tolist()):
                probs[i] = p
        return probs

    def score(self, text: str) -> float:
        """Mean P(AI) over the text's windows, one unpadded forward each."""
        tokenizer, model = self.load()
        with self._lock:
            chunks = window_texts(text, tokenizer, self.stride, self.max_windows)
            probs = self._probs(tokenizer, model, chunks, 1)
        return sum(probs) / len(probs) if probs else 0.0

    def score_many(self, texts: list[str]) -> list[float]:
        """``score`` for each text, with windows batched across texts."""
        tokenizer, model = self.load()
        with self._lock:
            per_text = [window_texts(t, tokenizer, self.stride, self.max_windows)
                        for t in texts]
            flat = [c for chunks in per_text for c in chunks]
            probs = self._probs(tokenizer, model, flat, self.batch_size) if flat else []
        out, i = [], 0
        for chunks in per_text:
            p = probs[i:i + len(chunks)]
            i += len(chunks)
            out.append(sum(p) / len(p) if p else 0.0)
        return out


class _SoftmaxIndexEngine(ClassifierEngine):
    """Softmax over the labels, read at the AI label found in ``id2label``."""

    keywords = AI_KEYWORDS
    default_index: int | None = 1
    _ai_idx = 1

    def _prepare(self, model) -> None:
        self._ai_idx = ai_label_index(getattr(model.config, "id2label", {}),
                                      self.keywords, self.default_index)

    def _ai_prob(self, logits):
        return logits.softmax(-1)[:, self._ai_idx]


class _SigmoidEngine(ClassifierEngine):
    """A single logit; its sigmoid is P(AI)."""

    def _ai_prob(self, logits):
        return logits[:, 0].sigmoid()


class FakespotEngine(_SoftmaxIndexEngine):
    key, name = "classifier_fakespot", "Fakespot"
    model_name = "fakespot-ai/roberta-base-ai-text-detection-v1"


class TMREngine(_SoftmaxIndexEngine):
    key, name = "classifier_tmr", "TMR Detector"
    model_name = "Oxidane/tmr-ai-text-detector"

    def _prepare(self, model) -> None:
        super()._prepare(model)
        # SlopTotal reads probs[ai_idx or 1], so an AI label at index 0 becomes 1.
        self._ai_idx = self._ai_idx or 1

    def _ai_prob(self, logits):
        if logits.shape[-1] == 1:
            return logits[:, 0].sigmoid()
        return super()._ai_prob(logits)


class BERTRaidEngine(ClassifierEngine):
    key, name = "classifier_bert_raid", "BERT-tiny RAID"
    model_name = "ShantanuT01/BERT-tiny-RAID"

    def _ai_prob(self, logits):
        return 1.0 - logits[:, 0].sigmoid()  # its single logit is P(human)


class E5Engine(_SoftmaxIndexEngine):
    key, name, engine_type = "classifier_e5", "E5-Small", "embedding"
    model_name = "MayZhou/e5-small-lora-ai-generated-detector"
    keywords = AI_KEYWORDS + ("gpt",)


class OpenAIEngine(_SoftmaxIndexEngine):
    key, name, engine_type = "classifier_openai", "OpenAI Detector", "classifier"
    model_name = "openai-community/roberta-base-openai-detector"  # alias "roberta-base-openai-detector"
    keywords = ("fake",)


class ChatGPTEngine(_SoftmaxIndexEngine):
    key, name, engine_type = "classifier_chatgpt", "ChatGPT Detector", "classifier"
    model_name = "Hello-SimpleAI/chatgpt-detector-roberta"
    keywords = ("chatgpt", "ai", "fake")
    default_index = None  # the last label


class ReMoDetectEngine(ClassifierEngine):
    key, name = "classifier_remodetect", "ReMoDetect"
    model_name = "hyunseoki/ReMoDetect-deberta"
    stride, max_windows = WINDOW, LARGE_MODEL_MAX_WINDOWS

    def _ai_prob(self, logits):
        if logits.shape[-1] == 1:  # a reward model: higher is more RLHF-like
            return logits[:, 0].sigmoid()
        return logits.softmax(-1)[:, -1]


def _desklib_class():
    import torch.nn as nn
    from transformers import AutoConfig, AutoModel, PreTrainedModel

    class DesklibAIDetectionModel(PreTrainedModel):
        """DeBERTa-v3-large under ``model.*``, a Linear(hidden, 1) ``classifier``
        on the masked mean of the last hidden state."""

        config_class = AutoConfig

        def __init__(self, config):
            super().__init__(config)
            self.model = AutoModel.from_config(config)
            self.classifier = nn.Linear(config.hidden_size, 1)
            self.post_init()

        def forward(self, input_ids, attention_mask=None, **kwargs):
            last = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(last.dtype)
            pooled = (last * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return self.classifier(pooled)

    return DesklibAIDetectionModel


def _superannotate_class():
    import torch.nn as nn
    from transformers import AutoConfig, AutoModel, PreTrainedModel

    class SuperAnnotateDetector(PreTrainedModel):
        """RoBERTa-large under ``roberta.*`` (no pooler), a Linear(hidden, 1)
        ``dense`` head on the <s> hidden state."""

        config_class = AutoConfig

        def __init__(self, config):
            super().__init__(config)
            self.roberta = AutoModel.from_config(config, add_pooling_layer=False)
            self.dense = nn.Linear(config.hidden_size, 1)
            self.post_init()

        def forward(self, input_ids, attention_mask=None, **kwargs):
            out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            return self.dense(out.last_hidden_state[:, 0])

    return SuperAnnotateDetector


class DesklibEngine(_SigmoidEngine):
    key, name = "classifier_desklib", "Desklib DeBERTa"
    model_name = "desklib/ai-text-detector-v1.01"
    stride, max_windows = WINDOW, LARGE_MODEL_MAX_WINDOWS

    def _load_hf(self):
        from transformers import AutoConfig, AutoTokenizer
        config = AutoConfig.from_pretrained(self.model_name)
        return (AutoTokenizer.from_pretrained(self.model_name),
                _desklib_class().from_pretrained(self.model_name, config=config))


class SuperAnnotateEngine(_SigmoidEngine):
    key, name = "classifier_superannotate", "SuperAnnotate"
    model_name = "SuperAnnotate/ai-detector-low-fpr"
    base_name = "FacebookAI/roberta-large"  # the repo's config has no model_type
    stride, max_windows = WINDOW, LARGE_MODEL_MAX_WINDOWS

    def _load_hf(self):
        from transformers import AutoConfig, AutoTokenizer
        config = AutoConfig.from_pretrained(self.base_name)
        return (AutoTokenizer.from_pretrained(self.base_name),
                _superannotate_class().from_pretrained(self.model_name, config=config))


ENGINE_CLASSES: tuple[type[ClassifierEngine], ...] = (
    FakespotEngine, TMREngine, BERTRaidEngine, E5Engine, OpenAIEngine, ChatGPTEngine,
    ReMoDetectEngine, DesklibEngine, SuperAnnotateEngine,
)

# Shared instances; nothing loads until an engine's first score.
NEURAL_ENGINES: list[ClassifierEngine] = [cls() for cls in ENGINE_CLASSES]
