"""Registry of NLP-representation × classifier pipelines.

Each entry returns a fully-configured sklearn ``Pipeline`` so the rest of
the codebase treats every experiment identically: fit, predict, score.

Three text representations are supported:

- **bow**   — sparse bag-of-words counts
- **tfidf** — TF-IDF weighted unigrams + bigrams (the standard baseline
  from Manning et al., 2008)
- **char_tfidf** — character n-grams (3–5) which are robust to the
  spelling noise and slang typical of social-media text
  (Joulin et al., 2017)

Four classifiers are supported:

- **logreg**  — multinomial logistic regression with L2 regularisation,
  the canonical strong linear baseline for sparse text features
- **svm**     — linear SVM (LinearSVC), historically the top performer
  on TF-IDF text features (Joachims, 1998)
- **nb**      — multinomial Naive Bayes, fast generative baseline
- **rf**      — random forest, non-linear sanity check

The product itself ships a single ``"best"`` model; the registry is what
the experiment notebook iterates over.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from ..config import load_config
from ..features.preprocess import PreprocessingConfig, clean_series


# ---------------------------------------------------------------------------
# Top-level cleaner function.
#
# Defined at module scope (not as a lambda inside ``build_pipeline``) so that
# ``joblib.dump`` can pickle the resulting Pipeline. Reads the current
# preprocessing config at call time; this keeps the persisted artefact
# config-driven without baking the config dict into the pickle.
# ---------------------------------------------------------------------------
def _clean_step(texts):
    cfg = load_config()
    prep_cfg = PreprocessingConfig.from_dict(cfg["preprocessing"])
    return clean_series(texts, prep_cfg)

# ---------------------------------------------------------------------------
# Text representations
# ---------------------------------------------------------------------------


def _make_vectoriser(kind: str):
    """Factory for the supported text-representation steps."""
    if kind == "bow":
        return CountVectorizer(
            ngram_range=(1, 1),
            min_df=2,
            max_df=0.95,
            strip_accents="unicode",
        )
    if kind == "tfidf":
        return TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents="unicode",
        )
    if kind == "char_tfidf":
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )
    raise ValueError(f"Unknown vectoriser: {kind!r}")


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

def _make_classifier(kind: str, random_state: int):
    """Factory for the supported classifiers.

    ``class_weight="balanced"`` is set where the API exposes it so that the
    minority classes (``arts_&_culture``, ``business_&_entrepreneurs``) are
    not ignored by the optimiser — see King & Zeng (2001) for the
    rationale on rare-event classification.
    """
    if kind == "logreg":
        return LogisticRegression(
            max_iter=2000,
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            random_state=random_state,
        )
    if kind == "svm":
        return LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=random_state,
        )
    if kind == "nb":
        # Multinomial NB has no class_weight parameter; we pass uniform priors
        # by leaving fit_prior=True and rely on the training distribution.
        return MultinomialNB(alpha=0.3)
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced",
            random_state=random_state,
        )
    raise ValueError(f"Unknown classifier: {kind!r}")


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentSpec:
    """Declarative description of a single experiment."""

    name: str
    representation: str
    classifier: str

    def __post_init__(self) -> None:
        if self.representation not in REPRESENTATIONS:
            raise ValueError(f"Unknown representation: {self.representation}")
        if self.classifier not in CLASSIFIERS:
            raise ValueError(f"Unknown classifier: {self.classifier}")


REPRESENTATIONS = ("bow", "tfidf", "char_tfidf")
CLASSIFIERS = ("logreg", "svm", "nb", "rf")


def build_pipeline(spec: ExperimentSpec) -> Pipeline:
    """Build a fresh sklearn Pipeline for the given experiment spec.

    The pipeline encapsulates the *entire* inference path so the persisted
    artefact at deployment time only needs ``pipeline.predict(raw_texts)``.
    """
    cfg = load_config()
    return Pipeline(
        steps=[
            ("clean", FunctionTransformer(func=_clean_step, validate=False)),
            ("vectorise", _make_vectoriser(spec.representation)),
            ("classify", _make_classifier(spec.classifier, cfg["random_state"])),
        ]
    )


# A reasonable default experiment grid for the report.
DEFAULT_EXPERIMENTS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec("baseline_tfidf_logreg", "tfidf", "logreg"),
    ExperimentSpec("bow_logreg",            "bow",   "logreg"),
    ExperimentSpec("tfidf_svm",             "tfidf", "svm"),
    ExperimentSpec("tfidf_nb",              "tfidf", "nb"),
    ExperimentSpec("tfidf_rf",              "tfidf", "rf"),
    ExperimentSpec("char_tfidf_logreg",     "char_tfidf", "logreg"),
    ExperimentSpec("char_tfidf_svm",        "char_tfidf", "svm"),
)
