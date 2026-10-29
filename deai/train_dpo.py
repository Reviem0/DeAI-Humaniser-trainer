"""Phase 4A: best-of-N + DPO, repeated for a few rounds.

Each round k:
  1. sample  N rewrites per train source from the current policy  -> samples.jsonl
  2. score   every rewrite with the reward (gates incl. NLI, then detector) -> scored.jsonl
  3. pairs   chosen = best, rejected = worst; skip sources whose best is 0
  4. train   one DPO epoch of a fresh LoRA on top of the previous policy -> adapter/
  5. eval    greedy rewrites of val, gates incl. NLI, detector     -> eval.jsonl, metrics.json

The policy after round k is the base model plus adapters 1..k merged in order,
so DPO's reference (the adapter switched off) is always the previous round's
policy. Round 0 is the zero-shot base model, evaluated only. Every stage skips
work whose output already exists, so rerunning the same command resumes.

    .venv\\Scripts\\python -m deai.train_dpo --run r-quick --reward quick --rounds 3
    .venv\\Scripts\\python -m deai.train_dpo --run smoke --reward quick --rounds 1 \\
        --train-limit 8 --eval-limit 8 --n 4        # end-to-end smoke test
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from deai import consolelog, policy, vram
from deai.detectors import BACKEND
from deai.reward import REWARD_SOURCES
from deai.tblog import RunLog
from deai.rollout import (Scorer, evaluate, load_sources, read_jsonl, result_row,
                          source_scores, write_jsonl)

RUNS = Path(__file__).resolve().parents[1] / "runs"
STAGES = ("sample", "score", "pairs", "train", "eval")


def adapters_up_to(run_dir: Path, k: int) -> list[Path]:
    return [run_dir / f"round_{i}" / "adapter" for i in range(1, k + 1)]


# ----------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------

def stage_sample(args, run_dir: Path, k: int, train) -> None:
    path = run_dir / f"round_{k}" / "samples.jsonl"
    done = {r["id"] for r in read_jsonl(path)}
    todo = train[~train.id.isin(done)]
    if todo.empty:
        return
    print(f"[round {k}] sampling {args.n} rewrites for {len(todo)} sources", flush=True)
    tok = policy.load_tokenizer(args.base)
    model = policy.load_policy(args.base, adapters_up_to(run_dir, k - 1))
    chunk = 32  # write as we go so an interruption loses little
    for start in range(0, len(todo), chunk):
        part = todo.iloc[start:start + chunk]
        outs = policy.generate(model, tok, part.text.tolist(), n=args.n, temperature=args.temperature,
                               max_seqs_per_batch=args.gen_batch, seed=k * 100_000 + start, progress=False)
        write_jsonl(path, [{"id": r.id, "source_text": r.text,
                            "samples": [s.__dict__ for s in samples]}
                           for r, samples in zip(part.itertuples(), outs)], mode="a")
        print(f"    {min(start + chunk, len(todo))}/{len(todo)}", flush=True)
    del model
    policy.release_gpu()


def stage_score(args, run_dir: Path, k: int, scorer: Scorer, tb: RunLog) -> None:
    rows = read_jsonl(run_dir / f"round_{k}" / "samples.jsonl")
    path = run_dir / f"round_{k}" / "scored.jsonl"
    done = {r["id"] for r in read_jsonl(path)}
    todo = [r for r in rows if r["id"] not in done]
    if not todo:
        return
    print(f"[round {k}] scoring {sum(len(r['samples']) for r in todo)} rewrites "
          f"({args.reward} reward)", flush=True)
    t0 = time.perf_counter()
    chunk = 16
    for start in range(0, len(todo), chunk):
        part = todo[start:start + chunk]
        srcs = [r["source_text"] for r in part for _ in r["samples"]]
        rws = [s["text"] for r in part for s in r["samples"]]
        fin = [s["finished"] for r in part for s in r["samples"]]
        results = iter(scorer.score(srcs, rws, fin, use_nli=args.train_nli))
        out = []
        for r in part:
            out.append({"id": r["id"], "samples": [{**s, **result_row(next(results))} for s in r["samples"]]})
        write_jsonl(path, out, mode="a")
        n = min(start + chunk, len(todo))
        tb.log_scoring(k, (len(done) + n) * args.n, [s for r in out for s in r["samples"]])
        el = time.perf_counter() - t0
        print(f"    {n}/{len(todo)} sources, {el / 60:.1f} min, ETA {(len(todo) - n) * el / n / 60:.0f} min",
              flush=True)


def stage_pairs(args, run_dir: Path, k: int) -> dict:
    scored = read_jsonl(run_dir / f"round_{k}" / "scored.jsonl")
    sources = {r["id"]: r["source_text"] for r in read_jsonl(run_dir / f"round_{k}" / "samples.jsonl")}
    rng = random.Random(k)
    pairs, all_rewards, best = [], [], []
    gate_fail: dict[str, int] = {}
    for r in scored:
        ok = [s for s in r["samples"] if s["reward"] is not None]  # None = detector failed
        for s in r["samples"]:
            name = s["failed_gate"] or ("detector_failed" if s["reward"] is None else "passed")
            gate_fail[name] = gate_fail.get(name, 0) + 1
        if len(ok) < 2:
            continue
        rewards = [s["reward"] for s in ok]
        all_rewards += rewards
        hi, lo = max(rewards), min(rewards)
        best.append(hi)
        if hi <= 0 or hi - lo < args.min_margin:
            continue
        chosen = rng.choice([s for s in ok if s["reward"] == hi])
        rejected = rng.choice([s for s in ok if s["reward"] == lo])
        pairs.append({"id": r["id"], "prompt": policy.messages(sources[r["id"]]),
                      "chosen": [{"role": "assistant", "content": chosen["text"]}],
                      "rejected": [{"role": "assistant", "content": rejected["text"]}],
                      "chosen_reward": hi, "rejected_reward": lo})
    write_jsonl(run_dir / f"round_{k}" / "pairs.jsonl", pairs)
    stats = {"sources": len(scored), "pairs": len(pairs),
             "mean_reward": round(float(np.mean(all_rewards)), 4) if all_rewards else None,
             "mean_best_of_n": round(float(np.mean(best)), 4) if best else None,
             "samples_by_failed_gate": gate_fail}
    (run_dir / f"round_{k}" / "round_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(f"[round {k}] {json.dumps(stats)}", flush=True)
    return stats


def stage_train(args, run_dir: Path, k: int, tb: RunLog) -> None:
    from datasets import Dataset
    from peft import LoraConfig
    from torch.utils.tensorboard import SummaryWriter
    from transformers.integrations import TensorBoardCallback
    from trl import DPOConfig, DPOTrainer

    out = run_dir / f"round_{k}" / "adapter"
    pairs = read_jsonl(run_dir / f"round_{k}" / "pairs.jsonl")
    if not pairs:
        raise SystemExit(f"round {k}: no DPO pairs (every source's best rewrite scored 0). "
                         "Try a warm start or a looser prompt; see PLAN.md Phase 4 step 0.")
    print(f"[round {k}] DPO on {len(pairs)} pairs", flush=True)
    ds = Dataset.from_list([{"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"],
                             "chat_template_kwargs": policy.CHAT_TEMPLATE_KWARGS} for p in pairs])
    tok = policy.load_tokenizer(args.base)
    model = policy.load_policy(args.base, adapters_up_to(run_dir, k - 1))
    loss_type = args.loss_type.split(",")
    cfg = DPOConfig(
        output_dir=str(run_dir / f"round_{k}" / "dpo"),
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, args.effective_batch // args.batch_size),
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.1,  # a float < 1 is a ratio in transformers 5
        beta=args.beta,
        loss_type=loss_type,
        loss_weights=[1.0] + [args.sft_weight] * (len(loss_type) - 1) if len(loss_type) > 1 else None,
        max_length=args.max_length,
        bf16=True,
        gradient_checkpointing=True,
        activation_offloading=vram.MODES[args.vram].activation_offloading,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=k,
    )
    peft_cfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"])
    tb_cb = TensorBoardCallback(SummaryWriter(tb.trainer_dir(f"train_round_{k}")))
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=peft_cfg,
                         callbacks=[tb_cb])
    trainer.train()
    # Save first: the weights only exist in memory until now. TensorBoardCallback closes
    # its writer itself in on_train_end (and sets tb_writer to None).
    trainer.model.save_pretrained(str(out))
    (run_dir / f"round_{k}" / "train_log.json").write_text(
        json.dumps(trainer.state.log_history, indent=1), encoding="utf-8")
    del trainer, model
    policy.release_gpu()


def stage_eval(args, run_dir: Path, k: int, scorer: Scorer, val, src_scores) -> dict:
    out_dir = run_dir / f"round_{k}"
    if (out_dir / "metrics.json").exists():
        return json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    print(f"[round {k}] evaluating on {len(val)} val sources", flush=True)
    tok = policy.load_tokenizer(args.base)
    model = policy.load_policy(args.base, adapters_up_to(run_dir, k))
    metrics = evaluate(model, tok, val, scorer, out_dir, src_scores, args.gen_batch)
    del model
    policy.release_gpu()
    return metrics


# ----------------------------------------------------------------------

def print_summary(run_dir: Path, rounds: int) -> None:
    keys = ["pass_all_gates", "mean_reward", "source_detector_mean", "rewrite_detector_mean",
            "rewrite_detector_mean_passing", "rewrite_below_likely_ai", "evasion_and_quality",
            "similarity_mean", "length_ratio_median"]
    print(f"\n{'round':>5} " + " ".join(f"{k[:14]:>14}" for k in keys))
    for k in range(rounds + 1):
        p = run_dir / f"round_{k}" / "metrics.json"
        if p.exists():
            m = json.loads(p.read_text(encoding="utf-8"))
            print(f"{k:>5} " + " ".join(f"{str(m.get(x)):>14}" for x in keys))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run name; outputs go to runs/dpo-<run>/")
    ap.add_argument("--reward", choices=sorted(REWARD_SOURCES), default="quick")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--base", default=policy.BASE_MODEL)
    ap.add_argument("--n", type=int, default=8, help="rewrites sampled per source")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--gen-batch", type=int, default=16,
                    help="sequences per generate() call; 16 fits Qwen3-4B's KV cache, 1.5B takes 64")
    ap.add_argument("--train-limit", type=int, default=None, help="train sources per round (default: all)")
    ap.add_argument("--eval-limit", type=int, default=None, help="val sources (default: all)")
    ap.add_argument("--min-margin", type=float, default=0.05, help="min chosen-rejected reward gap")
    ap.add_argument("--no-train-nli", dest="train_nli", action="store_false",
                    help="skip the NLI gate when scoring training samples (val always uses it)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--loss-type", default="sigmoid", help="e.g. 'sigmoid,sft' for RPO-style regularisation")
    ap.add_argument("--sft-weight", type=float, default=0.2, help="weight of extra loss types")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=1, help="pairs per step; 1 fits a 4B policy in 16 GB")
    ap.add_argument("--effective-batch", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--skip-zero-shot", action="store_true", help="don't evaluate the base model (round 0)")
    ap.add_argument("--stop-after", choices=STAGES, default=None, help="stop the last round after this stage")
    ap.add_argument("--vram", choices=sorted(vram.MODES), default=vram.DEFAULT,
                    help="high: keep every model loaded (~24 GB); medium: swap the scorers out for policy "
                         "stages (fits 16 GB); low: also offload activations, generate <= 8 (~12 GB)")
    args = ap.parse_args()
    args.gen_batch = vram.gen_batch(vram.MODES[args.vram], args.gen_batch)

    run_dir = RUNS / f"dpo-{args.run}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config.json"
    # "detectors": runs scored by the SlopTotal server have no such key, so they can't be
    # resumed and silently mix its scores (failed engines read as 0.0) with in-process ones.
    cfg = {**vars(args), "instruction": policy.INSTRUCTION, "detectors": BACKEND}
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text(encoding="utf-8"))
        for key in ("reward", "base", "n", "train_limit", "train_nli", "instruction", "detectors"):
            if old.get(key) != cfg[key]:
                raise SystemExit(f"{run_dir} was started with {key}={old.get(key)!r}; "
                                 f"got {cfg[key]!r}. Use another --run name.")
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    consolelog.start(run_dir)

    endpoint = REWARD_SOURCES[args.reward][0]
    scorer = Scorer(args.reward)
    val = load_sources("val", args.eval_limit)
    src_scores = source_scores(scorer, val, run_dir / "val_source_scores.json")
    print(f"run {run_dir.name}: reward {args.reward} ({endpoint}), base {args.base}, "
          f"{len(val)} val sources", flush=True)

    tb = RunLog(run_dir)
    print(f"TensorBoard: .venv\\Scripts\\tensorboard --logdir {RUNS}", flush=True)
    mode = vram.MODES[args.vram]
    print(f"VRAM mode {mode.name}: generate() batch {args.gen_batch}"
          f"{', scorers swapped out for policy stages' if mode.swap_scorers else ', everything stays loaded'}"
          f"{', activation offloading' if mode.activation_offloading else ''}", flush=True)
    vram.peak_gb()
    try:
        if not args.skip_zero_shot:
            if mode.swap_scorers:
                scorer.release()
            stage_eval(args, run_dir, 0, scorer, val, src_scores)
            print(f"[round 0] eval peak: {vram.peak_gb()}", flush=True)
            tb.log_round(run_dir, 0)
            print_summary(run_dir, 0)

        for k in range(1, args.rounds + 1):
            if (run_dir / f"round_{k}" / "metrics.json").exists():
                tb.log_round(run_dir, k)  # backfills rounds finished before TensorBoard logging
                continue
            # A different sample of sources each round when --train-limit is set.
            train = load_sources("train", args.train_limit, seed=k)
            last = k == args.rounds
            for stage in STAGES:
                if mode.swap_scorers and stage in ("sample", "train", "eval"):
                    scorer.release()  # the policy needs the GPU memory the scorers hold
                if stage == "sample":
                    stage_sample(args, run_dir, k, train)
                elif stage == "score":
                    stage_score(args, run_dir, k, scorer, tb)
                elif stage == "pairs":
                    stage_pairs(args, run_dir, k)
                elif stage == "train" and not (run_dir / f"round_{k}" / "adapter").exists():
                    stage_train(args, run_dir, k, tb)
                elif stage == "eval":
                    stage_eval(args, run_dir, k, scorer, val, src_scores)
                if stage != "pairs":
                    print(f"[round {k}] {stage} peak: {vram.peak_gb()}", flush=True)
                tb.log_round(run_dir, k)
                if last and stage == args.stop_after:
                    print(f"stopped after {stage}")
                    return 0
            print_summary(run_dir, k)
    finally:
        tb.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
