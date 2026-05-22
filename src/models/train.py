"""Train all experiments and persist the best pipeline for the product.

Running ``python -m src.models.train`` will:

1. Load splits (creating them if absent).
2. For each :class:`ExperimentSpec` in the default grid, fit on
   ``train``, validate on ``val``, and record metrics.
3. Pick the best by macro-F1 on validation, refit on ``train + val``,
   and persist to ``models/best_pipeline.joblib``.
4. Write ``reports/results.csv`` and confusion-matrix figures.

The "refit on train + val" step is standard practice once
hyperparameters / model choice have been frozen on the validation set —
the test set is touched once at evaluation time only.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from ..config import load_config, resolve_path
from ..data.split import make_splits
from ..evaluation.metrics import score_predictions, save_confusion_matrix
from .registry import DEFAULT_EXPERIMENTS, ExperimentSpec, build_pipeline


def _fit_and_score(spec: ExperimentSpec,
                   train: pd.DataFrame,
                   val: pd.DataFrame,
                   class_names: list[str]) -> tuple[object, dict]:
    """Fit one pipeline and score it on the validation set."""
    pipe = build_pipeline(spec)
    pipe.fit(train["text"].tolist(), train["label"].to_numpy())
    y_pred = pipe.predict(val["text"].tolist())
    metrics = score_predictions(val["label"].to_numpy(), y_pred, class_names)
    metrics["experiment"] = spec.name
    metrics["representation"] = spec.representation
    metrics["classifier"] = spec.classifier
    return pipe, metrics


def run_all_experiments() -> pd.DataFrame:
    """Train the default grid; return a DataFrame of validation metrics."""
    cfg = load_config()
    splits = make_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]

    class_names = [cfg["class_names"][i] for i in sorted(cfg["class_names"])]
    figures_dir = resolve_path(cfg["paths"]["figures_dir"])
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    reports_dir = resolve_path(cfg["paths"]["reports_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    fitted: dict[str, object] = {}

    for spec in DEFAULT_EXPERIMENTS:
        print(f"\n[train] {spec.name}  ({spec.representation} + {spec.classifier})")
        pipe, metrics = _fit_and_score(spec, train, val, class_names)
        rows.append(metrics)
        fitted[spec.name] = pipe

        # Validation confusion matrix per experiment
        y_pred_val = pipe.predict(val["text"].tolist())
        save_confusion_matrix(
            y_true=val["label"].to_numpy(),
            y_pred=y_pred_val,
            class_names=class_names,
            title=f"{spec.name}  (validation)",
            out_path=figures_dir / f"cm_val_{spec.name}.png",
        )
        print(f"  macro_f1={metrics['macro_f1']:.4f}  acc={metrics['accuracy']:.4f}")

    leaderboard = (
        pd.DataFrame(rows)
        .sort_values("macro_f1", ascending=False)
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # Pick winner, refit on train+val, evaluate on held-out test
    # ------------------------------------------------------------------
    winner_name = leaderboard.iloc[0]["experiment"]
    winner_spec = next(s for s in DEFAULT_EXPERIMENTS if s.name == winner_name)
    print(f"\n[train] Winner on validation: {winner_name}")
    print("[train] Refitting winner on train + val …")

    final_pipe = build_pipeline(winner_spec)
    combined = pd.concat([train, val], ignore_index=True)
    final_pipe.fit(combined["text"].tolist(), combined["label"].to_numpy())

    y_test_pred = final_pipe.predict(test["text"].tolist())
    test_metrics = score_predictions(test["label"].to_numpy(), y_test_pred, class_names)
    test_metrics["experiment"] = f"{winner_name}__refit_test"
    test_metrics["representation"] = winner_spec.representation
    test_metrics["classifier"] = winner_spec.classifier
    leaderboard = pd.concat([leaderboard, pd.DataFrame([test_metrics])], ignore_index=True)

    save_confusion_matrix(
        y_true=test["label"].to_numpy(),
        y_pred=y_test_pred,
        class_names=class_names,
        title=f"{winner_name}  (held-out test)",
        out_path=figures_dir / f"cm_test_{winner_name}.png",
    )

    # ------------------------------------------------------------------
    # Persist artefacts
    # ------------------------------------------------------------------
    leaderboard_path = reports_dir / "results.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    print(f"\n[train] Leaderboard written to {leaderboard_path}")

    model_path = models_dir / "best_pipeline.joblib"
    joblib.dump(final_pipe, model_path)

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": winner_spec.name,
        "representation": winner_spec.representation,
        "classifier": winner_spec.classifier,
        "class_names": class_names,
        "test_metrics": {k: v for k, v in test_metrics.items()
                         if k not in {"experiment", "representation", "classifier"}},
    }
    (models_dir / "best_pipeline.meta.json").write_text(
        json.dumps(metadata, indent=2, default=str)
    )
    print(f"[train] Model written to {model_path}")
    return leaderboard


if __name__ == "__main__":
    df = run_all_experiments()
    print("\n=== Final leaderboard ===")
    print(df.to_string(index=False))
