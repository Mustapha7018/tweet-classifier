# Tweet Topic Classifier — Product Prototype

A domain-specific data science product that automatically categorises social-media
posts (tweets) into one of six topic classes. Built for end-users with limited
data-science knowledge (e.g. social-media managers, comms teams), the prototype
exposes its model behind a one-click Streamlit web app while keeping a fully
reproducible training and evaluation pipeline underneath.

This prototype was developed for **CETM46 — Data Science Product Development,
Assignment 2** at the University of Sunderland.

---

## 1. What it does

Given a short piece of text (a tweet, a headline, a social-media post),
the product returns the predicted **topic class** plus a confidence score:

- `arts_&_culture`
- `business_&_entrepreneurs`
- `daily_life`
- `pop_culture`
- `science_&_technology`
- `sports_&_gaming`

Two interaction modes are supported:

1. **Single-message classification** — paste one piece of text, get the label.
2. **Batch classification** — upload a CSV with a `text` column, get a CSV
   back with predictions and confidence scores attached.

---

## 2. Project structure

```
tweet_classifier/
├── README.md
├── requirements.txt
├── config.yaml                 # All tunable parameters in one place
├── data/
│   ├── raw/                    # Drop Data.json here
│   └── processed/              # train/val/test splits (created by pipeline)
├── models/                     # Persisted sklearn pipelines (.joblib)
├── reports/
│   ├── figures/                # Confusion matrices, charts
│   └── results.csv             # Experiment leaderboard
├── notebooks/
│   └── 01_exploration_and_experiments.ipynb
├── src/
│   ├── data/
│   │   ├── load.py             # JSON → DataFrame
│   │   └── split.py            # Stratified train/val/test split
│   ├── features/
│   │   └── preprocess.py       # Tweet-specific text cleaning
│   ├── models/
│   │   ├── registry.py         # Model + vectoriser definitions
│   │   └── train.py            # Train / persist sklearn pipelines
│   ├── evaluation/
│   │   ├── metrics.py          # Macro F1, per-class, confusion matrix
│   │   └── compare.py          # Build leaderboard CSV
│   └── app/
│       └── streamlit_app.py    # End-user UI
└── tests/
    └── test_pipeline.py        # Smoke tests
```

---

## 3. Install and run (end-user, 60 seconds)

```bash
# 1. Create env
python -m venv .venv && source .venv/bin/activate

# 2. Install
pip install -r requirements.txt

# 3. Place the dataset
cp /Users/mustaphi/Downloads/477/Data.json data/raw/Data.json

# 4. Train and persist the best model (one-off, ~2 minutes)
python -m src.models.train

# 5. Launch the app
streamlit run src/app/streamlit_app.py
```

The app opens at `http://localhost:8501`.

---

## 4. Reproducing the experiments

The full experiment grid (multiple text representations × multiple
classifiers, with stratified k-fold CV and statistical comparison) is in
`notebooks/01_exploration_and_experiments.ipynb`.

To regenerate `reports/results.csv` and the confusion-matrix figures from
the command line:

```bash
python -m src.evaluation.compare
```

---

## 5. Reproducibility

- All randomness is seeded via `config.yaml → random_state`.
- Train / validation / test splits are stratified by class and persisted
  to `data/processed/` so every experiment scores on the same held-out test set.
- Each trained pipeline is serialised with `joblib` and tagged with the
  config hash used to create it.

---

## 6. Limitations

- Trained on tweets from 2019–2021; topic drift on newer data is expected.
- Tweet-specific markup (`{@handle@}`) is normalised during preprocessing;
  inputs from other platforms may need their own cleaner.
- The largest class (`pop_culture`, 39%) is six-times the size of the
  smallest (`arts_&_culture`, 2.2%); class-weighted training and macro-F1
  are used to compensate but minority-class recall remains the weakest link.

See the technical report for a full discussion.
