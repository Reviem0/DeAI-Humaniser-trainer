"""Multi-GPU plumbing without GPUs: chunking that keeps generate()'s batches, and the worker pool."""

import os
from types import SimpleNamespace

import pytest
import torch

from deai import multigpu, policy

HERE = __name__  # the spawned workers import their functions from this module


class _RecordingTok:
    pad_token_id, eos_token_id = 0, 1

    def __init__(self):
        self.batches = []

    def apply_chat_template(self, msgs, **kw):
        return msgs[0]["content"]

    def __call__(self, prompts, **kw):
        self.batches.append(tuple(prompts))
        return SimpleNamespace(to=lambda device: {"input_ids": torch.zeros((len(prompts), 1), dtype=torch.long)})

    def decode(self, ids, **kw):
        return "rewrite"


class _FakeModel:
    device = "cpu"
    generation_config = SimpleNamespace(eos_token_id=1)

    def eval(self):
        return self

    def generate(self, input_ids, num_return_sequences, **kw):
        return torch.ones((len(input_ids) * num_return_sequences, 2), dtype=torch.long)


def test_length_chunks_reproduce_generates_batches():
    sources = [f"passage {'x' * (i * 37 % 101)} {i}" for i in range(23)]
    whole = _RecordingTok()
    policy.generate(_FakeModel(), whole, sources, n=1, temperature=0.0, max_seqs_per_batch=4, progress=False)

    chunked = _RecordingTok()
    chunks = policy.length_chunks(sources, 1, 4)
    assert sorted(i for c in chunks for i in c) == list(range(len(sources)))
    for idx in reversed(chunks):  # any order, as on several GPUs
        policy.generate(_FakeModel(), chunked, [sources[i] for i in idx], n=1, temperature=0.0,
                        max_seqs_per_batch=4, progress=False)
    assert sorted(chunked.batches) == sorted(whole.batches)


def _init(offset):
    return offset


def _work(state, x):
    return x + state, os.environ.get("CUDA_VISIBLE_DEVICES")


def _fail(state, x):
    raise ValueError(f"bad task {x}")


def test_imap_one_device_runs_here_in_order(monkeypatch):
    freed = []
    monkeypatch.setattr(policy, "release_gpu", lambda: freed.append(True))
    tasks = [(i, {"x": i}) for i in range(5)]
    out = list(multigpu.imap(f"{HERE}:_work", tasks, ["0"], f"{HERE}:_init", (10,)))
    assert [k for k, _ in out] == list(range(5)) and [r[0] for _, r in out] == list(range(10, 15))
    assert freed == [True]  # state it built is freed

    out = list(multigpu.imap(f"{HERE}:_work", tasks, [], f"{HERE}:_init", (10,), state=100))
    assert [r[0] for _, r in out] == list(range(100, 105)) and freed == [True]  # caller's state is kept


def test_imap_spawns_one_process_per_device_each_seeing_its_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "outer")
    tasks = [(i, {"x": i}) for i in range(12)]
    out = dict(multigpu.imap(f"{HERE}:_work", tasks, ["5", "7"], f"{HERE}:_init", (10,), state="ignored"))
    assert {k: v[0] for k, v in out.items()} == {i: i + 10 for i in range(12)}
    assert {v[1] for v in out.values()} <= {"5", "7"}
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "outer"


def test_imap_raises_a_workers_error():
    with pytest.raises(RuntimeError, match="ValueError: bad task"):
        list(multigpu.imap(f"{HERE}:_fail", [(0, {"x": 0})], ["0", "1"], f"{HERE}:_init", (0,)))


def test_gpus_follow_cuda_visible_devices(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,4,6")
    assert multigpu.gpus() == ["2", "4", "6"] and multigpu.gpus(2) == ["2", "4"]
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert multigpu.gpus() == ["0", "1", "2"]
    with pytest.raises(SystemExit):
        multigpu.gpus(4)
