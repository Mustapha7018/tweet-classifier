"""Cached inference helpers for the persisted model."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from ..config import load_config, resolve_path


@lru_cache(maxsize=1)
def load_model():
    """Load the cached product pipeline."""
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
    """Predict topic labels and confidence-like scores."""
    pipeline, meta = load_model()
    class_names: list[str] = meta.get("class_names", [])
    labels = pipeline.predict(texts)

    # Fall back to decision margins when probabilities are unavailable.
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
