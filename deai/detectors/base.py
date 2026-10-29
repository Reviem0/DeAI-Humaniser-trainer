"""What every in-process detector engine provides, and where its models run.

The engines reproduce SlopTotal's (https://github.com/pablocaeg/sloptotal) so the
reward no longer needs its server. Each engine returns its raw score on 0-1;
``deai.detectors.registry`` clamps it and rounds it to 3 decimals, as SlopTotal
does before calibration.
"""

from __future__ import annotations

import os
from functools import cache
from typing import Protocol

# SlopTotal's engine_type values, which the transfer-study families group.
ENGINE_TYPES = ("neural", "classifier", "embedding", "statistical", "linguistic")


class Engine(Protocol):
    key: str  # SlopTotal's engine key, e.g. "classifier_fakespot"
    name: str  # its display name, e.g. "Fakespot"
    engine_type: str  # one of ENGINE_TYPES

    def score(self, text: str) -> float:
        """P(AI) on 0-1 for a stripped text of >= 50 chars. Models load on first call."""
        ...


@cache
def detector_device() -> str:
    """``DEAI_DETECTOR_DEVICE`` (cuda/cpu) if set, else cuda when available.

    SlopTotal runs fp32 and its CPU and GPU scores agree to 3 decimals, so the
    device changes speed and memory, not scores.
    """
    explicit = os.getenv("DEAI_DETECTOR_DEVICE")
    if explicit in ("cuda", "cpu"):
        return explicit
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
