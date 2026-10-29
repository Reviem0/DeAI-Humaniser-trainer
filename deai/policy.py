"""Phase 4 policy: the rewriter model, its prompt, and batched sampling.

A trained policy is the base model plus one LoRA adapter per DPO round,
applied in order. For generation the adapters are merged into the weights;
for training only the latest adapter is trainable.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"  # non-thinking Qwen3; Apache-2.0
INSTRUCTION = (
    "Rewrite the following passage so it reads naturally; keep every fact. "
    "Keep it about the same length as the original. Reply with the rewritten passage only."
)
# Qwen3 templates think unless told not to; other templates ignore the flag.
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
TOKENS_PER_WORD = 1.6  # generous for English with Qwen's tokenizer
# LoRA goes on every attention and MLP projection. Qwen3.5 (e.g. Qwen/Qwen3.5-9B) has
# full attention in every 4th layer only; the others are linear attention (Gated
# DeltaNet), whose projections are the in_proj_* / out_proj names. A name the base
# doesn't have matches nothing, so Qwen3 adapters are unchanged. in_proj_a/b (a few
# outputs per head, the decay and gate) are left out.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                "in_proj_qkv", "in_proj_z", "out_proj"]


def messages(source: str) -> list[dict]:
    return [{"role": "user", "content": f"{INSTRUCTION}\n\n{source.strip()}"}]


def max_new_tokens(source: str, max_ratio: float = 1.2) -> int:
    """Room for the longest rewrite the length gate accepts, plus a margin."""
    return int(len(source.split()) * max_ratio * TOKENS_PER_WORD) + 48


def load_tokenizer(base: str = BASE_MODEL):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_policy(base: str = BASE_MODEL, adapters: Sequence[Path] = (), trainable: bool = False):
    """Base + adapters. With ``trainable`` the last adapter stays a live, trainable
    PEFT adapter (earlier ones are merged); otherwise everything is merged."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda")
    adapters = [Path(a) for a in adapters]
    frozen, live = (adapters[:-1], adapters[-1:]) if trainable else (adapters, [])
    for a in frozen:
        model = PeftModel.from_pretrained(model, str(a)).merge_and_unload()
    if live:
        model = PeftModel.from_pretrained(model, str(live[0]), is_trainable=True)
    return model


def release_gpu() -> None:
    """Call after dropping the last reference to a model."""
    gc.collect()
    torch.cuda.empty_cache()


@dataclass
class Sample:
    text: str
    finished: bool  # ended with EOS rather than hitting max_new_tokens
    n_tokens: int


@torch.no_grad()
def generate(model, tok, sources: Sequence[str], n: int = 1, temperature: float = 0.9,
             top_p: float = 1.0, max_seqs_per_batch: int = 16, seed: int = 0,
             progress: bool = True) -> list[list[Sample]]:
    """``n`` rewrites per source (greedy if ``temperature == 0``). Sources are batched by length."""
    model.eval()
    tok.padding_side = "left"
    order = sorted(range(len(sources)), key=lambda i: len(sources[i]))
    per_batch = max(1, max_seqs_per_batch // n)
    out: list[list[Sample]] = [[] for _ in sources]
    torch.manual_seed(seed)
    batches = [order[i:i + per_batch] for i in range(0, len(order), per_batch)]
    for b_i, idx in enumerate(batches):
        prompts = [tok.apply_chat_template(messages(sources[i]), tokenize=False, add_generation_prompt=True,
                                           **CHAT_TEMPLATE_KWARGS) for i in idx]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        limit = max(max_new_tokens(sources[i]) for i in idx)
        sampling = dict(do_sample=True, temperature=temperature, top_p=top_p) if temperature > 0 \
            else dict(do_sample=False, temperature=None, top_p=None, top_k=None)
        gen = model.generate(**enc, max_new_tokens=limit, num_return_sequences=n,
                             pad_token_id=tok.pad_token_id, **sampling)
        gen = gen[:, enc["input_ids"].shape[1]:].cpu()
        stop_ids = {tok.eos_token_id, tok.pad_token_id, *_extra_stop_ids(model)}
        for row, ids in enumerate(gen.tolist()):
            src_i = idx[row // n]
            end = next((k for k, t in enumerate(ids) if t in stop_ids), None)
            content = ids if end is None else ids[:end]
            out[src_i].append(Sample(tok.decode(content, skip_special_tokens=True).strip(),
                                     finished=end is not None, n_tokens=len(content)))
        if progress:
            print(f"    generated {min((b_i + 1) * per_batch, len(order))}/{len(order)} sources", flush=True)
    return out


def length_chunks(sources: Sequence[str], n: int = 1, max_seqs_per_batch: int = 16,
                  batches_per_chunk: int = 2) -> list[list[int]]:
    """Source indices in chunks that ``generate`` batches exactly as it batches all of them.

    ``generate`` sorts by length and cuts batches from that order, so contiguous
    slices of it that hold whole batches give the same batches, padding and (greedy)
    outputs whether the chunks run in one call, one after another, or on several GPUs.
    """
    order = sorted(range(len(sources)), key=lambda i: len(sources[i]))
    size = max(1, max_seqs_per_batch // n) * batches_per_chunk
    return [order[i:i + size] for i in range(0, len(order), size)]


def _extra_stop_ids(model) -> list[int]:
    eos = getattr(model.generation_config, "eos_token_id", None)
    return [eos] if isinstance(eos, int) else list(eos or [])
