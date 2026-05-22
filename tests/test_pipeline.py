"""End-to-end smoke tests for the pipeline.

These tests exist so a CI run can catch obvious regressions before the
product is rebuilt or redeployed. They are deliberately fast — they
fit one tiny pipeline on the training split and assert that the model
beats a "predict the majority class" baseline.

Run with ``pytest -q tests/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load import load_raw, label_distribution
from src.data.split import make_splits
from src.features.preprocess import PreprocessingConfig, clean_text
from src.models.registry import ExperimentSpec, build_pipeline


def test_load_raw_returns_expected_schema():
    df = load_raw()
    for col in ("text", "date", "label", "id", "label_name"):
        assert col in df.columns, f"missing column: {col}"
    assert df["label"].nunique() == 6
    assert (df["text"].str.len() > 0).all()


def test_label_distribution_sums_to_100_percent():
    dist = label_distribution(load_raw())
    assert pytest.approx(dist["pct"].sum(), rel=1e-3) == 100.0


def test_clean_text_removes_entity_markup():
    raw = "The {@Clinton LumberKings@} beat the {@Cedar Rapids Kernels@}"
    cleaned = clean_text(raw, PreprocessingConfig())
    assert "{@" not in cleaned and "@}" not in cleaned
    assert "clinton lumberkings" in cleaned


def test_clean_text_replaces_urls_and_mentions():
    raw = "check this https://example.com from @somebody"
    cleaned = clean_text(raw, PreprocessingConfig())
    assert "http" not in cleaned
    assert "@somebody" not in cleaned
    assert "url" in cleaned
    assert "user" in cleaned


def test_splits_are_stratified_and_disjoint():
    splits = make_splits(force=True)
    ids_per_split = {name: set(df["id"]) for name, df in splits.items()}
    assert ids_per_split["train"].isdisjoint(ids_per_split["val"])
    assert ids_per_split["train"].isdisjoint(ids_per_split["test"])
    assert ids_per_split["val"].isdisjoint(ids_per_split["test"])
    # stratification: every class appears in every split
    for df in splits.values():
        assert df["label"].nunique() == 6


def test_tiny_pipeline_beats_majority_baseline():
    splits = make_splits()
    train, val = splits["train"], splits["val"]
    pipe = build_pipeline(ExperimentSpec("smoke", "tfidf", "logreg"))
    pipe.fit(train["text"].tolist(), train["label"].to_numpy())
    score = pipe.score(val["text"].tolist(), val["label"].to_numpy())
    majority_rate = val["label"].value_counts(normalize=True).max()
    assert score > majority_rate, (
        f"Pipeline accuracy {score:.3f} did not beat majority-class "
        f"baseline {majority_rate:.3f}"
    )
