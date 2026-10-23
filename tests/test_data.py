"""Split invariants for deai.data (needs the SlopBench corpus on disk)."""

import pytest

from deai import data
from deai.metrics import auroc, tpr_at_fpr

pytestmark = pytest.mark.skipif(not data.SLOPBENCH.exists(), reason="SlopBench corpus not found")


@pytest.fixture(scope="module")
def df():
    return data.build_passages()


def test_ids_unique_and_pairs_complete(df):
    assert df.id.is_unique
    counts = df.groupby("pair").label.agg(sorted)
    assert all(c == ["ai", "human"] for c in counts)


def test_topic_group_in_one_split(df):
    used = df[df.split != "excluded"]
    assert (used.groupby("group").split.nunique() == 1).all()


def test_pair_in_one_split(df):
    assert (df.groupby("pair").split.nunique() == 1).all()


def test_held_out_generator_only_in_test(df):
    held = df[df.model.isin(data.HELD_OUT_GENERATORS)]
    assert set(held.split) <= {"test", "excluded"}
    assert (held[held.split == "test"].test_kind == "ood_generator").all()
    # and no group it's used in has rows outside test
    groups = set(held[held.split == "test"].group)
    assert set(df[df.group.isin(groups)].split) <= {"test", "excluded"}


def test_sources_are_ai_and_long_enough(df):
    src = df[df.is_source]
    assert (src.label == "ai").all()
    assert src.words.between(data.MIN_SOURCE_WORDS, data.MAX_SOURCE_WORDS).all()
    assert set(src.split) == {"train", "val", "test"}


def test_deterministic():
    a, b = data.build_passages(), data.build_passages()
    assert a[["id", "split"]].equals(b[["id", "split"]])


def test_metrics():
    assert auroc([3, 4], [1, 2]) == 1.0
    assert auroc([1], [1]) == 0.5
    neg = list(range(100))  # 0..99
    assert tpr_at_fpr([99.5, 50], neg, 0.01) == 0.5
