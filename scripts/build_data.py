"""Phase 2: build data/passages.parquet from SlopBench with topic/generator splits.

    .venv\\Scripts\\python scripts\\build_data.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deai.data import HELD_OUT_GENERATORS, PASSAGES, SLOPBENCH, build_passages  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slopbench", type=Path, default=SLOPBENCH)
    ap.add_argument("--out", type=Path, default=PASSAGES)
    args = ap.parse_args()

    df = build_passages(args.slopbench)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    print(f"{len(df)} passages -> {args.out}")
    print(f"held-out generator(s): {', '.join(HELD_OUT_GENERATORS)}\n")
    print("Rows by split and label:")
    print(df.pivot_table(index="split", columns="label", values="id", aggfunc="count", fill_value=0))
    print("\nTest rows by kind:")
    print(df[df.split == "test"].pivot_table(index="test_kind", columns="label", values="id",
                                             aggfunc="count", fill_value=0))
    src = df[df.is_source]
    print(f"\nRewrite sources (AI, {src.words.min()}-{src.words.max()} words): "
          f"{dict(src.split.value_counts())}")
    print("\nRewrite sources by generator and split:")
    print(src.pivot_table(index="model", columns="split", values="id", aggfunc="count", fill_value=0))
    print("\nRewrite sources by source and split:")
    print(src.pivot_table(index="source", columns="split", values="id", aggfunc="count", fill_value=0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
