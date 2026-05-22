"""Inference helper — load the persisted product model once.

The app should never re-fit a model at request time. This module
provides a small, cached loader and a simple ``predict`` wrapper that
returns labels *plus* confidence scores.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from ..config import load_config, resolve_path


@lru_cache(maxsize=1)
def load_model():
    """Load the persisted product pipeline (cached for the process lifetime)."""
    cfg = load_config()
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    model_path = models_dir / "best_pipeline.joblib"
    meta_path = models_dir / "best_pipeline.meta.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}. "
            "Run `python -m src.models.train` first."
        )
    pipeline = joblib.load(model_path)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return pipeline, meta


def predict(texts: list[str]) -> list[dict]:
    """Predict topic labels for a list of texts.

    Returns
    -------
    list[dict]
        One dict per input text with keys: ``text``, ``label_id``,
        ``label_name``, ``confidence`` (or ``decision_score`` if the
        underlying classifier does not expose ``predict_proba``).
    """
    pipeline, meta = load_model()
    class_names: list[str] = meta.get("class_names", [])
    labels = pipeline.predict(texts)

    # Calibration: not every estimator gives probabilities (LinearSVC
    # doesn't). When unavailable, fall back to the absolute distance
    # from the decision boundary so the UI can still show a magnitude.
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(texts)
        confidences = probs.max(axis=1)
        score_key = "confidence"
    elif hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(texts)
        # Multi-class one-vs-rest: pick the margin of the chosen class
        if scores.ndim == 2:
            confidences = np.take_along_axis(
                scores, labels.reshape(-1, 1), axis=1
            ).ravel()
        else:
            confidences = np.abs(scores)
        score_key = "decision_score"
    else:
        confidences = np.full(len(texts), np.nan)
        score_key = "confidence"

    results = []
    for text, label_id, conf in zip(texts, labels, confidences):
        results.append({
            "text": text,
            "label_id": int(label_id),
            "label_name": class_names[int(label_id)] if class_names else str(int(label_id)),
            score_key: float(conf),
        })
    return results
