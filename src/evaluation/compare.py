"""Cross-validated comparison of all experiments with significance testing.

The headline leaderboard in ``reports/results.csv`` reports validation-set
metrics from a single split. That is enough for picking a deployment
candidate, but to defend the *ranking* in the report we need to show the
gaps are not just noise from the particular train/val split. This module
adds:

1. **Stratified 5-fold cross-validation** over the *training* set so we
   get a mean ± standard deviation for each experiment. This is the
   standard reliability check for text-classification papers
   (Dietterich, 1998).

2. **McNemar's test** on the test-set predictions of the two top
   experiments, which is the recommended significance test for
   comparing two classifiers evaluated on the same data
   (Dietterich, 1998; Raschka, 2018).

Run with ``python -m src.evaluation.compare``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_val_score

from ..config import load_config, resolve_path
from ..data.split import make_splits
from ..models.registry import DEFAULT_EXPERIMENTS, build_pipeline


def cross_validate_experiments(scoring: str = "f1_macro") -> pd.DataFrame:
    """Run stratified k-fold CV on the training split for every experiment."""
    cfg = load_config()
    splits = make_splits()
    train = splits["train"]
    X = train["text"].tolist()
    y = train["label"].to_numpy()

    skf = StratifiedKFold(
        n_splits=cfg["cv"]["n_splits"],
        shuffle=cfg["cv"]["shuffle"],
        random_state=cfg["random_state"],
    )

    rows = []
    for spec in DEFAULT_EXPERIMENTS:
        pipe = build_pipeline(spec)
        scores = cross_val_score(pipe, X, y, cv=skf, scoring=scoring, n_jobs=-1)
        rows.append({
            "experiment": spec.name,
            "representation": spec.representation,
            "classifier": spec.classifier,
            f"{scoring}_mean": float(scores.mean()),
            f"{scoring}_std": float(scores.std(ddof=1)),
            "n_splits": cfg["cv"]["n_splits"],
        })
    return (
        pd.DataFrame(rows)
        .sort_values(f"{scoring}_mean", ascending=False)
        .reset_index(drop=True)
    )


def mcnemar_test(y_true: np.ndarray,
                 y_pred_a: np.ndarray,
                 y_pred_b: np.ndarray) -> dict:
    """McNemar's chi-squared test (with continuity correction) for paired
    classifier predictions on the same test set.

    Returns the 2×2 contingency table, the test statistic, and the p-value.
    """
    a_correct = (y_pred_a == y_true)
    b_correct = (y_pred_b == y_true)

    n00 = int(np.sum(~a_correct & ~b_correct))   # both wrong
    n01 = int(np.sum(~a_correct &  b_correct))   # only B right
    n10 = int(np.sum( a_correct & ~b_correct))   # only A right
    n11 = int(np.sum( a_correct &  b_correct))   # both right

    # Use the exact binomial form when n01 + n10 is small;
    # else use the chi-squared form with continuity correction.
    if (n01 + n10) < 25:
        # Exact: P(K >= max | n=n01+n10, p=0.5) two-sided
        k = max(n01, n10)
        n = n01 + n10
        if n == 0:
            p_value = 1.0
            statistic = 0.0
        else:
            p_one_side = stats.binom.sf(k - 1, n, 0.5)
            p_value = min(1.0, 2 * p_one_side)
            statistic = float(k)
        method = "exact_binomial"
    else:
        statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
        p_value = float(1.0 - stats.chi2.cdf(statistic, df=1))
        method = "chi_squared_with_continuity"

    return {
        "method": method,
        "table": {"n00": n00, "n01": n01, "n10": n10, "n11": n11},
        "statistic": float(statistic),
        "p_value": float(p_value),
    }


def main() -> None:
    cfg = load_config()
    reports_dir = resolve_path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("[compare] Running 5-fold CV across all experiments …")
    cv_df = cross_validate_experiments()
    cv_path = reports_dir / "cv_results.csv"
    cv_df.to_csv(cv_path, index=False)
    print(f"[compare] CV results → {cv_path}")
    print(cv_df.to_string(index=False))

    # ------------------------------------------------------------------
    # McNemar between top-2 experiments on the held-out test set
    # ------------------------------------------------------------------
    splits = make_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]
    combined = pd.concat([train, val], ignore_index=True)

    top2 = cv_df.head(2)["experiment"].tolist()
    print(f"\n[compare] McNemar on test set between top-2: {top2}")

    preds = {}
    for name in top2:
        spec = next(s for s in DEFAULT_EXPERIMENTS if s.name == name)
        pipe = build_pipeline(spec)
        pipe.fit(combined["text"].tolist(), combined["label"].to_numpy())
        preds[name] = pipe.predict(test["text"].tolist())

    result = mcnemar_test(
        y_true=test["label"].to_numpy(),
        y_pred_a=preds[top2[0]],
        y_pred_b=preds[top2[1]],
    )
    result["model_a"] = top2[0]
    result["model_b"] = top2[1]

    out_path = reports_dir / "mcnemar.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"[compare] McNemar result → {out_path}")


if __name__ == "__main__":
    main()
