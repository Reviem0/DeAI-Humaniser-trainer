"""VRAM modes: how the policy, the reward's quality models and the detectors share one GPU.

Measured on the worst case (600-word sources, Qwen3-4B bf16, ``--max-length 2048``):

- policy: DPO training 12.3 GB allocated; sampling 16 sequences 11.6 GB (8: 10.5 GB)
- detectors, fp32: 8.4 GB if all 23 load, 11.3 GB peak while scoring (quick: 4 engines, ~1 GB)
- quality models (mpnet, CoLA, NLI): 1.5 GB

So the policy and the scorers fit a 16 GB card only one at a time:

- ``high``: everything stays loaded. Fastest; needs ~24 GB.
- ``medium`` (default, fits 16 GB): the detectors and quality models are freed
  before every policy stage (sample, train, eval generation) and reload to score.
- ``low`` (~12 GB, or GRPO on 16 GB): ``medium`` plus activation offloading in
  DPO and at most 8 sequences per ``generate()``. GRPO, which holds the policy
  and the scorers at once, runs the detectors on the CPU in medium and low.

With several GPUs (``deai/multigpu.py``) each worker process has a GPU of its
own, and the mode applies to each of them as to a single card.

None of this changes a score: the detectors run fp32 on any device (CPU and GPU
agree to 0.001) and the quality models stay on the GPU. ``deai/__init__.py``
picks the cudaMallocAsync allocator, which keeps the training peak from
fragmenting past 16 GB (18.3 GB reserved for 12.3 GB used with the default one).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VramMode:
    name: str
    swap_scorers: bool  # free detectors + quality models before each policy stage
    activation_offloading: bool  # DPO: checkpointed activations wait in CPU memory
    max_gen_batch: int | None  # cap on sequences per generate() call
    grpo_detectors_on_cpu: bool


MODES: dict[str, VramMode] = {
    "high": VramMode("high", swap_scorers=False, activation_offloading=False, max_gen_batch=None,
                     grpo_detectors_on_cpu=False),
    "medium": VramMode("medium", swap_scorers=True, activation_offloading=False, max_gen_batch=None,
                       grpo_detectors_on_cpu=True),
    "low": VramMode("low", swap_scorers=True, activation_offloading=True, max_gen_batch=8,
                    grpo_detectors_on_cpu=True),
}
DEFAULT = "medium"


def gen_batch(mode: VramMode, requested: int) -> int:
    return min(requested, mode.max_gen_batch) if mode.max_gen_batch else requested


def peak_gb() -> str:
    """Peak allocated / reserved since the last reset, then reset (for per-stage logging)."""
    import torch
    if not torch.cuda.is_available():
        return "no GPU"
    out = (f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB allocated, "
           f"{torch.cuda.max_memory_reserved() / 2**30:.1f} GB reserved")
    torch.cuda.reset_peak_memory_stats()
    return out
