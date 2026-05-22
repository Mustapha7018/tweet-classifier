"""Evaluation metrics and figure helpers.

The chosen metric for the leaderboard is **macro-F1**: it averages F1
across classes *without* weighting by support, which is appropriate when
classes are heavily imbalanced and minority-class performance matters
(Sokolova & Lapalme, 2009). We also report:

- accuracy            — overall agreement (the lay metric)
- weighted F1         — sensible when one cares about overall throughput
- per-class precision/recall/F1 — diagnostic for minority-class behaviour
- TP / TN / FP / FN per class — required by the rubric, derived from the
  confusion matrix

Confusion matrices are written as PNG into ``reports/figures/`` with a
consistent style so the report can include them verbatim.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering for CI / Streamlit
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def per_class_counts(y_true, y_pred, n_classes: int) -> dict[int, dict[str, int]]:
    """Compute TP/FP/FN/TN per class from a confusion matrix.

    Uses the standard one-vs-rest decomposition:
    for each class c, TP = correctly predicted as c; FP = predicted c but
    truly other; FN = truly c but predicted other; TN = the remainder.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    out: dict[int, dict[str, int]] = {}
    total = cm.sum()
    for c in range(n_classes):
        tp = int(cm[c, c])
        fn = int(cm[c, :].sum() - tp)
        fp = int(cm[:, c].sum() - tp)
        tn = int(total - tp - fp - fn)
        out[c] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return out


def score_predictions(y_true, y_pred, class_names: list[str]) -> dict:
    """Compute the headline metrics plus per-class P / R / F1 / counts."""
    n_classes = len(class_names)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred,
        labels=list(range(n_classes)),
        zero_division=0,
    )
    counts = per_class_counts(y_true, y_pred, n_classes)

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(sup[i]),
            **counts[i],
        }

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
    }


def save_confusion_matrix(
    y_true,
    y_pred,
    class_names: list[str],
    title: str,
    out_path: str | Path,
    *,
    normalize: bool = False,
) -> Path:
    """Render and save a confusion-matrix heatmap to disk."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
