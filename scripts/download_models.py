"""Download every Hugging Face model the pipeline loads, for machines without internet.

On an HPC whose GPU nodes are offline, run this on the login node; the job then
sets HF_HUB_OFFLINE=1 and reads everything from the HF cache (HF_HOME). The repo
ids come from the code (policy, reward gates, detector engines), so the list
can't drift from what training loads. A repo that is a LoRA adapter pulls in its
base model too.

    python scripts/download_models.py                    # download + offline check
    python scripts/download_models.py --base Qwen/Qwen2.5-1.5B-Instruct   # extra policy base
    python scripts/download_models.py --check            # offline check only (e.g. on a GPU node)

The check runs in a fresh process with HF_HUB_OFFLINE=1 on the CPU: it scores a
text with each detector engine and each quality model through deai's own loaders,
one at a time, and checks the policy's tokenizer, config and weight shards.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Formats nothing here loads; skipping them roughly halves some downloads.
IGNORE = ["*.onnx", "onnx/*", "openvino/*", "coreml/*", "*.h5", "tf_model*", "*.msgpack",
          "flax_model*", "*.tflite", "*.ot", "*.mlmodel"]
CHECK_TEXT = ("The committee met on Tuesday to review the budget. After a long discussion, "
              "members agreed to postpone the vote until the auditors had finished their report, "
              "which is expected early next month.")


def model_repos(bases: list[str]) -> list[str]:
    from deai import policy, reward
    from deai.detectors import neural, statistical
    repos = [policy.BASE_MODEL, *bases, reward.EMBED_MODEL, reward.COLA_MODEL, reward.NLI_MODEL,
             statistical.GPT2_MODEL, statistical.DISTIL_MODEL]
    for cls in neural.ENGINE_CLASSES:
        repos.append(cls.model_name)
        if hasattr(cls, "base_name"):
            repos.append(cls.base_name)
    return list(dict.fromkeys(repos))


def download(repos: list[str]) -> int:
    from huggingface_hub import snapshot_download
    queue, done, failed = list(repos), set(), []
    while queue:
        repo = queue.pop(0)
        if repo in done:
            continue
        done.add(repo)
        t0 = time.perf_counter()
        try:
            path = Path(snapshot_download(repo, ignore_patterns=IGNORE))
        except Exception as e:
            print(f"  FAIL {repo}: {type(e).__name__}: {e}", flush=True)
            failed.append(repo)
            continue
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9
        print(f"  ok   {repo} ({size:.2f} GB, {time.perf_counter() - t0:.0f}s)", flush=True)
        adapter = path / "adapter_config.json"
        if adapter.exists():
            base = json.loads(adapter.read_text()).get("base_model_name_or_path")
            if base:
                print(f"       adapter -> also fetching base {base}")
                queue.append(base)
    if failed:
        print(f"\n{len(failed)} repo(s) failed: {', '.join(failed)}")
        print("Gated repos need `huggingface-cli login` (or HF_TOKEN) first.")
    return len(failed)


def check(bases: list[str]) -> int:
    """Load everything with the hub offline, through the same code training uses."""
    assert os.environ.get("HF_HUB_OFFLINE") == "1", "run --check with HF_HUB_OFFLINE=1"
    from deai import policy
    from deai.detectors import Detectors
    from deai.reward import HFQualityModels

    failed = []

    def run(label, fn):
        t0 = time.perf_counter()
        try:
            fn()
            print(f"  ok   {label} ({time.perf_counter() - t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  FAIL {label}: {type(e).__name__}: {e}", flush=True)
            failed.append(label)

    print("detectors (analyze, each engine alone):")
    det = Detectors()
    for key in det.endpoint_engines("analyze"):
        run(key, lambda: det.score(CHECK_TEXT, "analyze", engines=(key,), overall=False))
        det.unload()

    print("quality models:")
    q = HFQualityModels(device="cpu")
    run("embed", lambda: q.embed([CHECK_TEXT]))
    run("acceptability", lambda: q.acceptability([CHECK_TEXT]))
    run("entailment", lambda: q.entailment([CHECK_TEXT], [CHECK_TEXT]))
    q.release()

    print("policy bases (tokenizer, config, weight files):")
    for base in dict.fromkeys([policy.BASE_MODEL, *bases]):
        run(base, lambda: _check_policy(base))

    print(f"\n{'all models load offline' if not failed else f'{len(failed)} failed: ' + ', '.join(failed)}")
    return len(failed)


def _check_policy(base: str) -> None:
    """Tokenizer and config load; every weight file is cached. Loading the weights
    themselves would need ~8 GB of RAM for a 4B model, too much for a login node."""
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoConfig

    from deai import policy
    policy.load_tokenizer(base)
    AutoConfig.from_pretrained(base)

    def cached(name: str) -> str | None:
        p = try_to_load_from_cache(base, name)
        return p if isinstance(p, str) else None

    index = cached("model.safetensors.index.json")
    if index:
        shards = set(json.loads(Path(index).read_text())["weight_map"].values())
        missing = sorted(s for s in shards if not cached(s))
        if missing:
            raise FileNotFoundError(f"missing shards {missing}")
    elif not cached("model.safetensors"):
        raise FileNotFoundError("no model.safetensors(.index.json) in the cache")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", action="append", default=[],
                    help="extra policy base model (repeatable; default base is always included)")
    ap.add_argument("--check", action="store_true", help="only check that everything loads offline")
    ap.add_argument("--no-check", action="store_true", help="download without the offline check")
    args = ap.parse_args()
    if os.environ.get("DEAI_BASE"):
        args.base.append(os.environ["DEAI_BASE"])

    from huggingface_hub import constants
    print(f"HF cache: {constants.HF_HUB_CACHE}")

    if args.check:
        return 1 if check(args.base) else 0

    repos = model_repos(args.base)
    print(f"downloading {len(repos)} repos:")
    if download(repos):
        return 1
    if args.no_check:
        return 0

    print("\noffline check (fresh process, HF_HUB_OFFLINE=1, CPU):")
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
           "DEAI_DETECTOR_DEVICE": "cpu", "CUDA_VISIBLE_DEVICES": ""}
    cmd = [sys.executable, __file__, "--check", *[a for b in args.base for a in ("--base", b)]]
    return subprocess.run(cmd, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
