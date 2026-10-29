"""Data parallelism over several GPUs for the stages that dominate a DPO round.

Sampling is most of a round (838 sources x 8 rewrites took ~2.5 h on one card,
against ~27 min to score with the full reward, ~10 min to train and ~4 min to
evaluate), so sampling, scoring and eval generation split their chunks across
one process per GPU. Each process sees only its own GPU (``CUDA_VISIBLE_DEVICES``),
so it runs the one-GPU code unchanged with its own copy of the model it needs.

The chunks and their seeds are the ones a single GPU uses (``train_dpo``), so
results don't depend on the number of GPUs, up to GPU nondeterminism; only the
order rows are written in changes. With one GPU, or none, the work runs in this
process as before. DPO training stays on the first GPU.

Workers print to the console only; what belongs in ``train.log`` (errors, peak
memory) comes back through the result queue and is printed here.
"""

from __future__ import annotations

import importlib
import os
import queue
import traceback
from typing import Any, Callable, Iterable, Iterator

Task = tuple[Any, dict]  # (key, keyword arguments for the work function)


def gpus(requested: int | None = None) -> list[str]:
    """The first ``requested`` visible GPUs (default: all), as ``CUDA_VISIBLE_DEVICES`` names them."""
    import torch
    n = torch.cuda.device_count()
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    names = [s.strip() for s in env.split(",")][:n] if env else [str(i) for i in range(n)]
    if requested is None:
        return names
    if requested > len(names):
        raise SystemExit(f"--gpus {requested}: only {len(names)} GPU(s) visible")
    return names[:requested]


def imap(work: str, tasks: Iterable[Task], devices: list[str], init: str, init_args: tuple = (),
         state: Any = None) -> Iterator[tuple[Any, Any]]:
    """Yield ``(key, work(state, **kwargs))`` for every ``(key, kwargs)`` task as it finishes.

    ``work`` and ``init`` are ``"module:function"`` names, so a spawned process can
    import them. On one device (or none) the tasks run here, in order, on ``state``
    if given, else on ``init(*init_args)``, which is freed afterwards. On several,
    every device gets a process that calls ``init`` once and takes tasks from a
    shared queue, so results come back out of order.
    """
    tasks = list(tasks)
    if len(devices) <= 1:
        yield from _imap_here(work, tasks, init, init_args, state)
    else:
        yield from _imap_spawned(work, tasks, devices, init, init_args)


def _fn(ref: str) -> Callable:
    module, name = ref.split(":")
    return getattr(importlib.import_module(module), name)


def _imap_here(work, tasks, init, init_args, state):
    own = state is None
    if own:
        state = _fn(init)(*init_args)
    fn = _fn(work)
    try:
        for key, kwargs in tasks:
            yield key, fn(state, **kwargs)
    finally:
        if own:
            del state
            from deai.policy import release_gpu
            release_gpu()


def _imap_spawned(work, tasks, devices, init, init_args):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")  # CUDA can't be forked, and Windows has no fork
    todo, done = ctx.Queue(), ctx.Queue()
    for t in tasks:
        todo.put(t)
    for _ in devices:
        todo.put(None)

    procs = []
    old = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        for d in devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = d  # a spawned child starts with this environment
            p = ctx.Process(target=_worker, args=(d, init, init_args, work, todo, done), name=f"gpu-{d}")
            p.start()
            procs.append(p)
    finally:
        if old is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old

    finished = False
    try:
        remaining, running = len(tasks), len(procs)
        while remaining or running:
            try:
                kind, who, body = done.get(timeout=10)
            except queue.Empty:
                crashed = next((p for p in procs if p.exitcode not in (None, 0)), None)
                if crashed:  # killed (e.g. out of host memory) before it could report
                    raise RuntimeError(f"worker {crashed.name} exited with code {crashed.exitcode}") from None
                continue
            if kind == "error":
                raise RuntimeError(f"worker on GPU {who} failed:\n{body}")
            if kind == "done":
                running -= 1
                print(f"    GPU {who} peak: {body}", flush=True)
                continue
            remaining -= 1
            yield who, body
        finished = True
    finally:
        for p in procs:
            if finished:
                p.join(timeout=60)
            if p.is_alive():
                p.terminate()
                p.join()


def _worker(device: str, init: str, init_args: tuple, work: str, todo, done) -> None:
    try:
        state, fn = _fn(init)(*init_args), _fn(work)
        while (task := todo.get()) is not None:
            key, kwargs = task
            done.put(("result", key, fn(state, **kwargs)))
        from deai import vram
        done.put(("done", device, vram.peak_gb()))
    except BaseException:
        done.put(("error", device, traceback.format_exc()))
