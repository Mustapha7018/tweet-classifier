"""Load and validate the raw tweet dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import load_config, resolve_path

REQUIRED_COLUMNS = {"text", "date", "label", "id", "label_name"}


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Load ``Data.json`` as a validated DataFrame."""
    cfg = load_config()
    data_path = Path(path) if path is not None else resolve_path(cfg["paths"]["raw_data"])

    if not data_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {data_path}. "
            "Place Data.json in data/raw/ or override the path in config.yaml."
        )

    with data_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    df = pd.DataFrame.from_records(records)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Required columns missing from raw data: {missing}")

    # Normalise types.
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    df["label_name"] = df["label_name"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["id"] = df["id"].astype("int64")

    # Remove duplicate ids and blank text.
    n0 = len(df)
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)
    if len(df) != n0:
        print(f"[load_raw] Dropped {n0 - len(df)} duplicate/empty rows.")

    return df


def label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy DataFrame summarising the class balance."""
    counts = df["label_name"].value_counts().rename("count")
    pct = (counts / counts.sum() * 100).round(2).rename("pct")
    return pd.concat([counts, pct], axis=1).reset_index(names="label_name")


if __name__ == "__main__":
    # Command-line sanity check.
    df = load_raw()
    print(f"Loaded {len(df):,} tweets, {df['label'].nunique()} classes.")
    print(label_distribution(df).to_string(index=False))
