"""Create stratified train / validation / test splits and persist them.

A single, frozen split is created once and reused across every experiment
so that all model comparisons score on the *same* held-out test set —
this is what makes the leaderboard in ``reports/results.csv`` meaningful.

Stratification on the ``label`` column preserves the (heavily imbalanced)
class proportions in every split, which is essential for honest reporting
of macro-F1 on minority classes (Forman & Scholz, 2010).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import load_config, resolve_path
from .load import load_raw


def make_splits(force: bool = False) -> dict[str, pd.DataFrame]:
    """Build train/val/test splits and cache them to ``data/processed/``.

    Parameters
    ----------
    force
        If True, regenerate even if cached splits already exist.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of ``"train" | "val" | "test"`` to DataFrame.
    """
    cfg = load_config()
    processed_dir = resolve_path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    paths = {name: processed_dir / f"{name}.parquet" for name in ("train", "val", "test")}

    if not force and all(p.exists() for p in paths.values()):
        return {name: pd.read_parquet(p) for name, p in paths.items()}

    df = load_raw()

    random_state = cfg["random_state"]
    test_size = cfg["split"]["test_size"]
    val_size = cfg["split"]["val_size"]
    stratify = df["label"] if cfg["split"]["stratify"] else None

    # First split off the test set
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # Then split train_val into train and val.
    # val_size is given as a fraction of the *original* dataset, so we adjust.
    val_fraction_of_remaining = val_size / (1.0 - test_size)
    stratify_tv = train_val["label"] if cfg["split"]["stratify"] else None
    train, val = train_test_split(
        train_val,
        test_size=val_fraction_of_remaining,
        random_state=random_state,
        stratify=stratify_tv,
    )

    splits = {"train": train.reset_index(drop=True),
              "val": val.reset_index(drop=True),
              "test": test.reset_index(drop=True)}

    for name, frame in splits.items():
        frame.to_parquet(paths[name], index=False)

    return splits


def split_summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Tidy summary of split sizes and class balance for the report."""
    rows = []
    for name, frame in splits.items():
        row = {"split": name, "n": len(frame)}
        for cls, count in frame["label_name"].value_counts().items():
            row[cls] = count
        rows.append(row)
    return pd.DataFrame(rows).fillna(0)


if __name__ == "__main__":
    splits = make_splits(force=True)
    print(split_summary(splits).to_string(index=False))
