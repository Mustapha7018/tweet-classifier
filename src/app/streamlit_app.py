"""Streamlit UI for the tweet topic classifier."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make ``src`` importable when Streamlit launches this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predict import load_model, predict  # noqa: E402


# Page setup.
st.set_page_config(
    page_title="Tweet Topic Classifier",
    page_icon="🗂️",
    layout="centered",
)

st.title("Tweet Topic Classifier")
st.caption(
    "Paste a tweet or upload a CSV — get an instant topic prediction. "
    "No data-science knowledge required."
)


@st.cache_resource(show_spinner="Loading model…")
def _bootstrap():
    """Cached model load — Streamlit only pays the joblib cost once."""
    return load_model()


try:
    _pipeline, _meta = _bootstrap()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()


with st.sidebar:
    st.subheader("About this model")
    st.write(f"**Best pipeline:** `{_meta.get('experiment', 'n/a')}`")
    st.write(f"**Representation:** {_meta.get('representation', 'n/a')}")
    st.write(f"**Classifier:** {_meta.get('classifier', 'n/a')}")
    if "test_metrics" in _meta:
        tm = _meta["test_metrics"]
        st.metric("Held-out accuracy", f"{tm.get('accuracy', 0):.3f}")
        st.metric("Held-out macro F1", f"{tm.get('macro_f1', 0):.3f}")
    st.divider()
    st.caption(
        "Trained on TweetTopic 2019–2021. Predictions on text from other "
        "platforms or much newer time periods may be less reliable."
    )


# Single-message tab.
tab_single, tab_batch = st.tabs(["📝 Single message", "📂 Batch CSV"])

with tab_single:
    text = st.text_area(
        "Tweet or message",
        placeholder="Paste the text you want to categorise here…",
        height=130,
    )
    if st.button("Classify", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Please paste some text first.")
        else:
            result = predict([text])[0]
            score_key = "confidence" if "confidence" in result else "decision_score"
            st.success(f"Predicted topic: **{result['label_name']}**")
            score = result[score_key]
            if score_key == "confidence":
                st.progress(min(max(score, 0.0), 1.0),
                            text=f"Confidence: {score:.2%}")
            else:
                st.caption(f"Decision margin: {score:+.3f} "
                           "(higher = more confident)")


# Batch CSV tab.
with tab_batch:
    st.write("Upload a CSV with a column named `text`. "
             "The app appends `predicted_label` and `confidence` columns.")
    uploaded = st.file_uploader("Choose CSV file", type=["csv"])

    if uploaded is not None:
        df_in = pd.read_csv(uploaded)
        if "text" not in df_in.columns:
            st.error("Your CSV must have a `text` column.")
        else:
            with st.spinner(f"Classifying {len(df_in):,} rows…"):
                results = predict(df_in["text"].astype(str).tolist())
            scores_col = (
                "confidence" if "confidence" in results[0] else "decision_score"
            )
            df_out = df_in.copy()
            df_out["predicted_label"] = [r["label_name"] for r in results]
            df_out[scores_col] = [r[scores_col] for r in results]

            st.dataframe(df_out.head(20), use_container_width=True)

            buf = io.BytesIO()
            df_out.to_csv(buf, index=False)
            st.download_button(
                "Download predictions",
                buf.getvalue(),
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
