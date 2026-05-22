"""Load the raw tweet dataset into a clean, validated DataFrame.

The source file (``Data.json``) is a JSON array of records with keys:
``text, date, label, id, label_name``. The loader normalises types,
validates the schema, drops duplicates, and exposes a single
``load_raw()`` entry point used by every downstream module.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import load_config, resolve_path

REQUIRED_COLUMNS = {"text", "date", "label", "id", "label_name"}


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw JSON dataset into a DataFrame.

    Parameters
    ----------
    path
        Optional override for the raw data path. Defaults to the value
        in ``config.yaml`` (``paths.raw_data``).

    Returns
    -------
    pandas.DataFrame
        Validated DataFrame with one row per tweet.

    Raises
    ------
    FileNotFoundError
        If the data file does not exist.
    ValueError
        If required columns are missing.
    """
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

    # Type normalisation
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    df["label_name"] = df["label_name"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["id"] = df["id"].astype("int64")

    # Drop fully duplicate rows and empty texts (defensive — none expected)
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
    # Quick sanity check from the command line
    df = load_raw()
    print(f"Loaded {len(df):,} tweets, {df['label'].nunique()} classes.")
    print(label_distribution(df).to_string(index=False))
