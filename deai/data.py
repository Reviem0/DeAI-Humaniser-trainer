"""Phase 2 data: SlopBench passages, topic/generator splits, and the baseline table.

Every AI passage in SlopBench has a human counterpart written for the same seed
(headline, question, prompt, book, ...); ``pair`` links them. Splits are made
over *topic groups*, so a topic never appears in two splits:

- the group is the source plus the seed, so the three passages from one
  Gutenberg book, or essays answering the same prompt, stay together;
- ``essay-hewlett`` is one group: 84 of its 88 prompts are variations on
  "computers and society";
- every group containing a passage by a held-out generator goes to test, so
  test also measures generalisation to a generator the rewriter never saw;
- the remaining groups are hashed into train/val/test.

Human passages follow their pair's split. They're never rewritten; they serve
as the false-positive reference and as SFT targets.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

SLOPBENCH = Path(r"D:\AI\sloptotal\tests\eval\slopbench\corpus")
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PASSAGES = DATA_DIR / "passages.parquet"
BASELINE = DATA_DIR / "baseline_scores.parquet"

HELD_OUT_GENERATORS = ("mistralai/mistral-small-2603",)
SINGLE_TOPIC_SOURCES = ("essay-hewlett",)
SPLIT_FRACTIONS = {"train": 0.7, "val": 0.1, "test": 0.2}

# Rewrite sources: the length gate wants 0.8x source and >= 80 words, so
# shorter sources can't pass it. SlopBench caps texts at 600 words.
MIN_SOURCE_WORDS = 100
MAX_SOURCE_WORDS = 600


def load_slopbench(corpus: Path = SLOPBENCH) -> pd.DataFrame:
    """Human and AI passages, one row each. Hard cases and grey-zone sets are left out."""
    rows = []
    for path in sorted(corpus.glob("human-*.jsonl")) + sorted(corpus.glob("ai-*.jsonl")):
        if "mixed" in path.name:
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                human = r["label"] == "human"
                rows.append({
                    "id": r["id"],
                    "pair": r["id"] if human else r["pair"],
                    "label": r["label"],
                    "source": r["source"],
                    "domain": r["domain"],
                    "model": "human" if human else r["model"],
                    "text": r["text"],
                    "words": len(r["text"].split()),
                    "seed": json.dumps(r["seed"], sort_keys=True) if human else None,
                })
    df = pd.DataFrame(rows)
    # AI rows carry no seed; take it from their human counterpart.
    seeds = df.loc[df.label == "human"].set_index("pair")["seed"]
    df["seed"] = df["pair"].map(seeds)
    return df


def topic_group(source: str, seed_json: str) -> str:
    if source in SINGLE_TOPIC_SOURCES:
        return source
    seed = {k: v for k, v in json.loads(seed_json).items() if k != "prompt_inferred_by"}
    return f"{source}|{json.dumps(seed, sort_keys=True)}"


def _hash_split(group: str, salt: str) -> str:
    x = int(hashlib.sha256(f"{salt}|{group}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    edge = 0.0
    for split, frac in SPLIT_FRACTIONS.items():
        edge += frac
        if x < edge:
            return split
    return "test"


def assign_splits(df: pd.DataFrame, held_out: tuple[str, ...] = HELD_OUT_GENERATORS,
                  salt: str = "deai-v1") -> pd.DataFrame:
    """Add ``group``, ``split`` (train/val/test/excluded) and ``test_kind``."""
    df = df.copy()
    df["group"] = [topic_group(s, seed) for s, seed in zip(df.source, df.seed)]

    held_groups = set(df.loc[df.model.isin(held_out), "group"])
    pinned = {g: "train" for g in SINGLE_TOPIC_SOURCES}
    group_split = {}
    for g in df.group.unique():
        if g in pinned:
            group_split[g] = pinned[g]
        elif g in held_groups:
            group_split[g] = "test"
        else:
            group_split[g] = _hash_split(g, salt)

    df["split"] = df.group.map(group_split)
    # A held-out generator's passage in a pinned train group would leak it.
    leak = df.model.isin(held_out) & (df.split != "test")
    df.loc[leak, "split"] = "excluded"
    # Human rows follow their pair, so a pair is never split across sets.
    excluded_pairs = set(df.loc[leak, "pair"])
    df.loc[(df.label == "human") & df.pair.isin(excluded_pairs), "split"] = "excluded"

    held_pairs = set(df.loc[df.model.isin(held_out), "pair"])
    df["test_kind"] = None
    test = df.split == "test"
    df.loc[test, "test_kind"] = "in_dist"
    df.loc[test & df.pair.isin(held_pairs), "test_kind"] = "ood_generator"
    return df


def build_passages(corpus: Path = SLOPBENCH) -> pd.DataFrame:
    df = assign_splits(load_slopbench(corpus))
    df["is_source"] = (
        (df.label == "ai")
        & (df.split != "excluded")
        & df.words.between(MIN_SOURCE_WORDS, MAX_SOURCE_WORDS)
    )
    return df.drop(columns=["seed"]).sort_values(["split", "label", "id"]).reset_index(drop=True)


def load_passages(path: Path = PASSAGES) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_baseline(path: Path = BASELINE) -> pd.DataFrame:
    return pd.read_parquet(path)
