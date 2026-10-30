"""Phase 4 logic that doesn't need a GPU or a server: DPO pair selection, eval summary."""

from types import SimpleNamespace

from deai import policy
from deai.rollout import summarize_eval, write_jsonl
from deai.train_dpo import adapters_up_to, stage_pairs


def sample(text, reward, gate=None):
    return {"text": text, "reward": reward, "failed_gate": gate, "finished": True}


def make_round(tmp_path, scored):
    rd = tmp_path / "round_1"
    write_jsonl(rd / "samples.jsonl", [{"id": i, "source_text": f"source {i}", "samples": []} for i in scored])
    write_jsonl(rd / "scored.jsonl", [{"id": i, "samples": s} for i, s in scored.items()])
    return rd


def test_pairs_pick_best_and_worst_and_skip_useless_sources(tmp_path):
    rd = make_round(tmp_path, {
        "a": [sample("a-mid", 0.4), sample("a-best", 0.7), sample("a-zero", 0.0, "length")],
        "b": [sample("b1", 0.0, "nli"), sample("b2", 0.0, "meaning")],          # nothing passed
        "c": [sample("c1", 0.50), sample("c2", 0.52)],                          # gap < margin
        "d": [sample("d1", None), sample("d2", 0.6)],                           # detector failed on d1
        "e": [sample("e1", 0.3), sample("e2", None), sample("e3", 0.1)],
    })
    stats = stage_pairs(SimpleNamespace(min_margin=0.05), tmp_path, 1)
    from deai.rollout import read_jsonl
    pairs = {p["id"]: p for p in read_jsonl(rd / "pairs.jsonl")}
    assert set(pairs) == {"a", "e"}
    assert pairs["a"]["chosen"][0]["content"] == "a-best"
    assert pairs["a"]["rejected"][0]["content"] == "a-zero"
    assert pairs["a"]["prompt"] == policy.messages("source a")
    assert pairs["e"]["rejected_reward"] == 0.1
    assert stats["pairs"] == 2 and stats["samples_by_failed_gate"]["nli"] == 1


def test_passing_pairs_compare_rewrites_that_passed_every_gate(tmp_path):
    from deai.rollout import read_jsonl
    rd = make_round(tmp_path, {
        # two passing rewrites 0.3 apart in reward (detector score): a passing pair
        "a": [sample("a-best", 0.7), sample("a-ok", 0.4), sample("a-nli", 0.0, "nli")],
        # one passing rewrite: extremes in "passing", nothing in "passing-only"
        "b": [sample("b-pass", 0.5), sample("b-nli", 0.0, "nli")],
        # passing rewrites too close together; extremes would pair b-pass vs the gate failure
        "c": [sample("c1", 0.50), sample("c2", 0.52), sample("c-len", 0.0, "length")],
    })
    by_mode = {}
    for mode in ("extremes", "passing", "passing-only"):
        stats = stage_pairs(SimpleNamespace(min_margin=0.05, pairs=mode), tmp_path, 1)
        by_mode[mode] = ({p["id"]: (p["chosen"][0]["content"], p["rejected"][0]["content"], p["kind"])
                          for p in read_jsonl(rd / "pairs.jsonl")}, stats)

    pairs, stats = by_mode["extremes"]
    assert pairs == {"a": ("a-best", "a-nli", "extremes"), "b": ("b-pass", "b-nli", "extremes"),
                     "c": ("c2", "c-len", "extremes")}

    pairs, stats = by_mode["passing"]
    assert pairs == {"a": ("a-best", "a-ok", "passing"), "b": ("b-pass", "b-nli", "extremes")}
    assert stats["pairs_by_kind"] == {"passing": 1, "extremes": 1} and stats["pair_mode"] == "passing"

    pairs, _ = by_mode["passing-only"]
    assert pairs == {"a": ("a-best", "a-ok", "passing")}


def test_adapter_chain_is_ordered(tmp_path):
    assert adapters_up_to(tmp_path, 0) == []
    assert [p.parent.name for p in adapters_up_to(tmp_path, 3)] == ["round_1", "round_2", "round_3"]


def test_summarize_eval_separates_evasion_from_quality():
    def row(passed, det, gate=None):
        gates = {"unicode": {"passed": True}, "length": {"passed": gate != "length"}}
        if gate != "length":
            gates["meaning"] = {"passed": passed}
        return {"passed": passed, "failed_gate": gate, "rewrite_detector": det, "source_detector": 80.0,
                "reward": 0.5 if passed else 0.0, "gates": gates, "finished": True,
                "similarity": 0.9 if gate != "length" else None, "words": 100, "source_words": 100}
    m = summarize_eval([row(True, 20.0), row(True, 70.0), row(False, 10.0, "meaning"),
                        row(False, 10.0, "length")])
    assert m["pass_all_gates"] == 0.5
    assert m["rewrite_below_likely_ai"] == 0.75   # 3 of 4 evade...
    assert m["evasion_and_quality"] == 0.25       # ...but only 1 also passes the gates
    assert m["gate_pass_rate_when_reached"] == {"length": 0.75, "meaning": 0.667, "unicode": 1.0}


def _scalars(logdir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(logdir))
    ea.Reload()
    return {tag: [(e.step, round(e.value, 4)) for e in ea.Scalars(tag)] for tag in ea.Tags()["scalars"]}


def test_tensorboard_round_logging_is_written_once(tmp_path):
    import json
    from deai.tblog import RunLog
    for k, (passed, det) in enumerate([(0.13, 67.3), (0.40, 50.0)]):
        rd = tmp_path / f"round_{k}"
        rd.mkdir()
        (rd / "metrics.json").write_text(json.dumps({
            "n": 10, "pass_all_gates": passed, "rewrite_detector_mean": det,
            "gate_pass_rate_when_reached": {"nli": 0.2}, "failed_gate_counts": {"nli": 5, "passed": 5}}))
    (tmp_path / "round_1" / "round_stats.json").write_text(json.dumps(
        {"sources": 10, "pairs": 7, "mean_reward": 0.1, "samples_by_failed_gate": {"passed": 2, "nli": 6}}))

    for _ in range(2):  # a resumed run logs the same rounds again: no duplicates
        log = RunLog(tmp_path)
        log.log_round(tmp_path, 0)
        log.log_round(tmp_path, 1)
        log.log_scoring(1, 16, [{"reward": 0.5, "passed": True, "detector": 40.0},
                                {"reward": 0.0, "passed": False, "detector": None}])
        log.close()

    s = _scalars(tmp_path / "tb" / "rounds")
    assert s["val/pass_all_gates"] == [(0, 0.13), (1, 0.4)]
    assert s["val/rewrite_detector_mean"] == [(0, 67.3), (1, 50.0)]
    assert s["val_failed_at/nli"] == [(0, 0.5), (1, 0.5)]
    assert s["best_of_n/pair_yield"] == [(1, 0.7)]
    live = _scalars(tmp_path / "tb" / "scoring")
    assert live["round_1/pass_all_gates"][0] == (16, 0.5)
    assert live["round_1/detector_mean_passing"][0] == (16, 40.0)


def test_stage_train_saves_adapter_after_callbacks_close(tmp_path, monkeypatch):
    # TensorBoardCallback.on_train_end closes its writer and sets tb_writer to None; code after
    # train() that touched it crashed before save_pretrained and lost a 4 h DPO epoch.
    import torch
    import trl
    from deai import train_dpo
    from deai.tblog import RunLog

    saved = []

    class FakeTrainer:
        def __init__(self, model, args, callbacks, **kw):
            self.callbacks, self.state = callbacks, SimpleNamespace(log_history=[{"loss": 0.69}])
            self.model = SimpleNamespace(save_pretrained=saved.append)

        def train(self):
            for cb in self.callbacks:
                cb.on_train_end(None, self.state, None)

    monkeypatch.setattr(trl, "DPOTrainer", FakeTrainer)
    monkeypatch.setattr(trl, "DPOConfig", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(policy, "load_tokenizer", lambda base: None)
    monkeypatch.setattr(policy, "load_policy", lambda base, adapters: None)
    monkeypatch.setattr(policy, "release_gpu", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    write_jsonl(tmp_path / "round_1" / "pairs.jsonl", [{"prompt": policy.messages("s"), "chosen": [{"role": "assistant", "content": "a"}],
                                                        "rejected": [{"role": "assistant", "content": "b"}]}])
    args = SimpleNamespace(base="m", batch_size=1, effective_batch=16, lr=2e-5, beta=0.1, loss_type="sigmoid",
                           sft_weight=0.2, max_length=2048, lora_r=16, vram="medium", gpus=["0"])
    tb = RunLog(tmp_path)
    train_dpo.stage_train(args, tmp_path, 1, tb)
    tb.close()
    assert saved == [str(tmp_path / "round_1" / "adapter")]
    assert (tmp_path / "round_1" / "train_log.json").exists()


def test_max_new_tokens_covers_the_length_gate():
    src = "word " * 500
    assert policy.max_new_tokens(src) >= 500 * 1.2 * 1.3
