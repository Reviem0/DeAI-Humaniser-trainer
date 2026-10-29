"""Phase 4B: GRPO with the Phase 3 reward (only if 4A plateaus and throughput allows).

Every group of ``--num-generations`` completions for one source is scored with
the same gates + detector as 4A. Completions cut off at the length limit get 0.
A detector failure returns None, which TRL leaves out of the group's mean.
Each reward call appends its rollouts to ``rollouts.jsonl`` for spot checks.

Optionally start from a 4A policy with ``--init-from runs/dpo-<run> --init-round k``.
At the end the adapter is saved and val is evaluated like a 4A round.

    .venv\\Scripts\\python -m deai.train_grpo --run r-quick --reward quick
    .venv\\Scripts\\python -m deai.train_grpo --run smoke --reward quick --train-limit 4 \\
        --eval-limit 4 --num-generations 4 --max-steps 2      # smoke test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from deai import consolelog, policy, vram
from deai.reward import REWARD_SOURCES
from deai.rollout import Scorer, evaluate, load_sources, source_scores, write_jsonl
from deai.tblog import RunLog
from deai.detectors import BACKEND
from deai.train_dpo import RUNS, adapters_up_to


class GateReward:
    """TRL reward function: (prompts, completions, completion_ids, source, ...) -> rewards."""

    def __init__(self, scorer: Scorer, max_completion_length: int, log_path: Path, use_nli: bool = True,
                 tb: RunLog | None = None):
        self.scorer = scorer
        self.use_nli = use_nli
        self.tb = tb
        self.rewrites_scored = 0
        self.max_len = max_completion_length
        self.log_path = log_path
        self.calls = 0
        self.__name__ = f"reward_{scorer.reward_source}"  # TRL logs it under this name

    def __call__(self, prompts, completions, completion_ids, source, id, **kwargs) -> list[float | None]:
        texts = [c[0]["content"].strip() if isinstance(c, list) else c.strip() for c in completions]
        finished = [len(ids) < self.max_len for ids in completion_ids]
        t0 = time.perf_counter()
        results = self.scorer.score(source, texts, finished, use_nli=self.use_nli)
        self.calls += 1
        write_jsonl(self.log_path, [
            {"call": self.calls, "id": i, "rewrite": t, "reward": r.reward, "failed_gate": r.failed_gate,
             "similarity": r.similarity, "detector": r.detector_score}
            for i, t, r in zip(id, texts, results)], mode="a")
        self.rewrites_scored += len(results)
        if self.tb:
            self.tb.log_scoring(0, self.rewrites_scored, [
                {"reward": r.reward, "passed": r.passed, "detector": r.detector_score} for r in results])
        passed = sum(r.passed for r in results)
        print(f"  reward call {self.calls}: {passed}/{len(results)} passed gates, "
              f"{time.perf_counter() - t0:.1f}s", flush=True)
        return [r.reward for r in results]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="outputs go to runs/grpo-<run>/")
    ap.add_argument("--reward", choices=sorted(REWARD_SOURCES), default="quick")
    ap.add_argument("--base", default=policy.BASE_MODEL)
    ap.add_argument("--init-from", type=Path, default=None, help="a 4A run dir to start from")
    ap.add_argument("--init-round", type=int, default=None, help="its round (default: last with an adapter)")
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-completion-length", type=int, default=1200, help="fits 1.2x a 600-word source")
    ap.add_argument("--batch-size", type=int, default=1, help="completions per forward pass")
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--gen-batch", type=int, default=16, help="sequences per generate() call in eval")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.0, help="KL coefficient (TRL default 0)")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--no-train-nli", dest="train_nli", action="store_false")
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--eval-limit", type=int, default=None)
    ap.add_argument("--vram", choices=sorted(vram.MODES), default=vram.DEFAULT,
                    help="high: detectors on the GPU beside the policy (~24 GB); medium/low: detectors "
                         "on the CPU, low also generates <= 8 in eval")
    args = ap.parse_args()
    mode = vram.MODES[args.vram]
    args.gen_batch = vram.gen_batch(mode, args.gen_batch)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from torch.utils.tensorboard import SummaryWriter
    from transformers.integrations import TensorBoardCallback
    from trl import GRPOConfig, GRPOTrainer

    run_dir = RUNS / f"grpo-{args.run}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(
        {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
         "instruction": policy.INSTRUCTION, "detectors": BACKEND}, indent=1), encoding="utf-8")
    consolelog.start(run_dir)
    init = []
    if args.init_from:
        k = args.init_round or max(int(p.parent.name.split("_")[1])
                                   for p in args.init_from.glob("round_*/adapter"))
        init = adapters_up_to(args.init_from, k)
        print(f"starting from {args.init_from.name} round {k}")

    if mode.grpo_detectors_on_cpu:
        # GRPO holds the policy, its generations and the scorers at once. An explicit
        # DEAI_DETECTOR_DEVICE wins; scores agree across devices to 0.001.
        os.environ.setdefault("DEAI_DETECTOR_DEVICE", "cpu")
    scorer = Scorer(args.reward)
    print(f"VRAM mode {mode.name}: detectors on {os.environ.get('DEAI_DETECTOR_DEVICE', 'cuda')}", flush=True)
    train = load_sources("train", args.train_limit)
    ds = Dataset.from_list([{"prompt": policy.messages(r.text), "source": r.text, "id": r.id}
                            for r in train.itertuples()])
    tok = policy.load_tokenizer(args.base)
    model = policy.load_policy(args.base, init)

    cfg = GRPOConfig(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        mask_truncated_completions=True,
        temperature=args.temperature,
        chat_template_kwargs=policy.CHAT_TEMPLATE_KWARGS,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,
        beta=args.beta,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        report_to="none",
        log_completions=False,
    )
    peft_cfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
                          target_modules=policy.LORA_TARGETS)
    tb = RunLog(run_dir)
    print(f"TensorBoard: .venv\\Scripts\\tensorboard --logdir {RUNS}", flush=True)
    reward_fn = GateReward(scorer, args.max_completion_length, run_dir / "rollouts.jsonl", args.train_nli, tb)
    tb_cb = TensorBoardCallback(SummaryWriter(tb.trainer_dir("train")))
    trainer = GRPOTrainer(model=model, reward_funcs=reward_fn, args=cfg, train_dataset=ds,
                          processing_class=tok, peft_config=peft_cfg, callbacks=[tb_cb])
    last = max((p for p in (run_dir / "checkpoints").glob("checkpoint-*")),
               key=lambda p: int(p.name.split("-")[1]), default=None)
    trainer.train(resume_from_checkpoint=str(last) if last else None)
    # Save first; TensorBoardCallback closes its writer itself in on_train_end.
    adapter = run_dir / "adapter"
    trainer.model.save_pretrained(str(adapter))
    (run_dir / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=1), encoding="utf-8")
    del trainer, model
    policy.release_gpu()
    torch.cuda.reset_peak_memory_stats()

    if mode.swap_scorers:
        scorer.release()
    val = load_sources("val", args.eval_limit)
    src_scores = source_scores(scorer, val, run_dir / "val_source_scores.json")
    model = policy.load_policy(args.base, [*init, adapter])
    metrics = evaluate(model, tok, val, scorer, run_dir / "eval", src_scores, args.gen_batch)
    tb.log_round(run_dir, 1, run_dir / "eval")  # x = 1: after training (compare with a DPO run's round 0)
    tb.close()
    print(json.dumps(metrics, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
