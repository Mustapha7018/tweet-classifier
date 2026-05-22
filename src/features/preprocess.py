"""Tweet-specific text preprocessing.

The cleaner is intentionally light-touch:

- Tweets are short, so aggressive stop-word removal can strip topical signal.
- The dataset uses a ``{@entity@}`` markup convention (e.g.
  ``{@Clinton LumberKings@}``) which must be normalised so that proper-noun
  entities are still tokenised consistently across splits.
- URLs and @mentions are replaced with placeholder tokens rather than
  removed outright — their *presence* is informative even when the
  specific URL or user-handle is not.
- Hashtags are preserved (they carry strong topical signal in this corpus).

Every step is controlled by ``config.yaml → preprocessing`` so that
ablation experiments can be run without code changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

# Regex patterns compiled once at import time.
_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_MENTION_RE = re.compile(r"(?<!\w)@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_ENTITY_MARKUP_RE = re.compile(r"\{@\s*(.*?)\s*@\}")
_MULTISPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class PreprocessingConfig:
    """Strongly-typed view of the preprocessing config block."""

    lowercase: bool = True
    strip_entity_markup: bool = True
    replace_urls: bool = True
    replace_mentions: bool = True
    replace_hashtags: bool = False
    collapse_whitespace: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "PreprocessingConfig":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def clean_text(text: str, cfg: PreprocessingConfig | None = None) -> str:
    """Apply the configured cleaning pipeline to one string.

    Parameters
    ----------
    text
        Raw tweet text.
    cfg
        Optional preprocessing config. Defaults to the standard config.

    Returns
    -------
    str
        Cleaned text, ready for tokenisation by the vectoriser.
    """
    if cfg is None:
        cfg = PreprocessingConfig()

    if cfg.strip_entity_markup:
        text = _ENTITY_MARKUP_RE.sub(r"\1", text)

    if cfg.replace_urls:
        text = _URL_RE.sub(" URL ", text)

    if cfg.replace_mentions:
        text = _MENTION_RE.sub(" USER ", text)

    if cfg.replace_hashtags:
        text = _HASHTAG_RE.sub(r"\1", text)
    # else: leave "#word" intact — sklearn's default token_pattern strips the '#'
    # but the word itself is preserved.

    if cfg.lowercase:
        text = text.lower()

    if cfg.collapse_whitespace:
        text = _MULTISPACE_RE.sub(" ", text).strip()

    return text


def clean_series(texts: Iterable[str], cfg: PreprocessingConfig | None = None) -> list[str]:
    """Vectorised wrapper over ``clean_text`` for a pandas Series / list."""
    if cfg is None:
        cfg = PreprocessingConfig()
    return [clean_text(t, cfg) for t in texts]
