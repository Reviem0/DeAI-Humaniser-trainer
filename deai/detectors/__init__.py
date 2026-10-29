"""SlopTotal's 23 detector engines and its two overall scores, in-process.

Ports of SlopTotal (MIT, https://github.com/pablocaeg/sloptotal), checked against
its own code on shared texts (tests/test_detectors_*.py), so no server is needed:

- ``quick``: the quick-score path. Six engines (4 classifiers, 2 heuristics),
  ``calibration.quick_score``, verdict clean/mixed/ai.
- ``analyze``: the full path. All 23 engines on the text cut to 100k chars,
  ``calibration.calibrated_score`` (English or multilingual fit), the long verdict.

Engine scores are clamped and rounded to 3 decimals before calibration, as
SlopTotal does. Only the engines a request needs are run: the ones with a
non-zero calibration weight for the overall score, plus any asked for by key.
Zero-weight engines contribute nothing to the overall score, so skipping them
doesn't change it.

One deliberate difference: SlopTotal scores a failed engine 0.0 (its engines catch
their own errors; 0.5 only if one escapes on the quick path) and carries on; here
the failure, or a non-finite score, is raised for that text, so the reward becomes
None and the trainer skips the sample instead of learning from a fake "human" score.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Collection, Literal, Sequence

from deai.detectors import calibration
from deai.detectors.base import Engine

Endpoint = Literal["quick", "analyze"]

MIN_TEXT_LENGTH = 50  # SlopTotal's MIN_TEXT_LENGTH, after strip()
# Recorded with stored scores (run config.json, baseline cache) so they're never
# mixed with scores from the SlopTotal server, which stored none.
BACKEND = "in-process"

QUICK_ENGINES = ("classifier_bert_raid", "classifier_e5", "classifier_tmr", "classifier_fakespot",
                 "linguistic", "formulaic")
# Quick-score weights (calibration.json "quick"); bert_raid and e5 are fitted to 0.
QUICK_SCORING = frozenset(k for k, w in calibration._CALIBRATION["quick"]["weights"].items() if w > 0)


class TextRejected(ValueError):
    """The text can't be scored (under 50 chars after strip, as SlopTotal's 400)."""


class DetectorError(RuntimeError):
    """An engine failed on a text."""


@dataclass
class Score:
    overall: float | None  # 0-100, higher = more likely AI; None when not asked for
    engines: dict[str, float]  # engine key -> 0-1, only the engines that ran
    verdict: str
    endpoint: Endpoint
    latency_s: float
    confidence: str = ""
    language: str = ""  # analyze only


@dataclass(frozen=True)
class EngineInfo:
    key: str
    name: str
    engine_type: str


def default_engines() -> list[Engine]:
    """All 23, in SlopTotal's engine order (app/analyzer.py ``_engines``)."""
    from deai.detectors.heuristic import HEURISTIC_ENGINES
    from deai.detectors.neural import NEURAL_ENGINES
    from deai.detectors.statistical import STATISTICAL_ENGINES
    order = ["classifier_tmr", "classifier_remodetect", "binoculars", "fast_detectgpt", "perplexity",
             "cross_perplexity", "classifier_fakespot", "classifier_e5", "classifier_bert_raid",
             "classifier_openai", "classifier_chatgpt", "classifier_desklib", "classifier_superannotate",
             "gltr", "log_rank", "diveye", "burstiness", "linguistic", "structural", "vocabulary",
             "formulaic", "readability", "sentiment"]
    by_key = {e.key: e for e in [*NEURAL_ENGINES, *STATISTICAL_ENGINES, *HEURISTIC_ENGINES]}
    return [by_key[k] for k in order]


@dataclass
class _Job:
    text: str
    keys: list[str]
    scores: dict[str, float] = field(default_factory=dict)
    error: BaseException | None = None


class Detectors:
    """``Detectors().score(text, "quick")``; engines are injectable for tests."""

    def __init__(self, engines: Sequence[Engine] | None = None):
        self._engines = {e.key: e for e in (engines if engines is not None else default_engines())}

    def engines(self) -> dict[str, EngineInfo]:
        return {k: EngineInfo(k, e.name, e.engine_type) for k, e in self._engines.items()}

    def endpoint_engines(self, endpoint: Endpoint) -> tuple[str, ...]:
        """Every engine the endpoint runs in SlopTotal (of those this registry has)."""
        if endpoint == "quick":
            return tuple(k for k in QUICK_ENGINES if k in self._engines)
        return tuple(self._engines)

    def score(self, text: str, endpoint: Endpoint, engines: Collection[str] = (),
              overall: bool = True) -> Score:
        r = self.score_many([text], endpoint, engines, overall)[0]
        if isinstance(r, BaseException):
            raise r
        return r

    def score_many(self, texts: Sequence[str], endpoint: Endpoint, engines: Collection[str] = (),
                   overall: bool = True) -> list[Score | BaseException]:
        """Score each text; a text that's rejected or hits an engine failure yields the exception.

        ``engines`` are run in addition to those the overall score needs; with
        ``overall=False`` only they run and ``Score.overall`` is None.
        """
        allowed = set(self.endpoint_engines(endpoint))
        unknown = set(engines) - allowed
        if unknown:
            raise ValueError(f"{endpoint} doesn't run {sorted(unknown)}")
        t0 = time.perf_counter()
        jobs: list[_Job] = []
        for text in texts:
            text = text.strip()
            if endpoint == "analyze":
                text = calibration.fit_to_limit(text)
            need = set(engines)
            if overall:
                need |= QUICK_SCORING if endpoint == "quick" else calibration.scoring_engines(text)
            job = _Job(text, [k for k in self._engines if k in need])
            if len(text) < MIN_TEXT_LENGTH:
                job.error = TextRejected(f"text is {len(text)} chars; scoring needs {MIN_TEXT_LENGTH}")
            jobs.append(job)
        self._run(jobs)
        latency = (time.perf_counter() - t0) / max(len(texts), 1)
        return [job.error or self._finish(job, endpoint, overall, latency) for job in jobs]

    def unload(self) -> None:
        """Free every loaded model; they reload on next use."""
        seen = set()
        for e in self._engines.values():
            target = getattr(e, "cache", e)  # the GPT-2 engines share one cache
            if id(target) not in seen and hasattr(target, "unload"):
                seen.add(id(target))
                target.unload()

    # ------------------------------------------------------------------

    def _run(self, jobs: list[_Job]) -> None:
        # Engines that batch (the classifiers) go engine by engine across texts; the
        # rest go text by text, so the GPT-2 engines share one forward pass per text.
        batched = [k for k, e in self._engines.items() if hasattr(e, "score_many")]
        for key in batched:
            todo = [j for j in jobs if j.error is None and key in j.keys]
            if not todo:
                continue
            try:
                vals = self._engines[key].score_many([j.text for j in todo])
            except Exception:  # find which text it was
                vals = [self._one(key, j) for j in todo]
            for j, v in zip(todo, vals):
                if v is not None:
                    self._keep(key, j, v)
        for j in jobs:
            for key in j.keys:
                if j.error is not None:
                    break
                if key not in batched:
                    v = self._one(key, j)
                    if v is not None:
                        self._keep(key, j, v)

    @staticmethod
    def _keep(key: str, job: _Job, v: float) -> None:
        if not math.isfinite(v):
            job.error = DetectorError(f"{key}: non-finite score {v}")
        else:
            job.scores[key] = round(min(max(float(v), 0.0), 1.0), 3)

    def _one(self, key: str, job: _Job) -> float | None:
        try:
            return self._engines[key].score(job.text)
        except Exception as e:
            job.error = DetectorError(f"{key}: {type(e).__name__}: {e}")
            return None

    @staticmethod
    def _finish(job: _Job, endpoint: Endpoint, overall: bool, latency: float) -> Score:
        if not overall:
            return Score(None, job.scores, "", endpoint, latency)
        if endpoint == "quick":
            score, conf = calibration.quick_score(job.scores, job.text)
            return Score(score, job.scores, calibration.quick_verdict(score), endpoint, latency, conf)
        score, conf, lang = calibration.calibrated_score(job.scores, job.text)
        return Score(score, job.scores, calibration.overall_verdict(score), endpoint, latency, conf, lang)
