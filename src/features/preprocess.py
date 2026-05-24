"""Light-touch tweet text preprocessing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

# Compile regexes once at import time.
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
    """Clean one text value using the configured preprocessing steps."""
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
    # Keep "#word"; sklearn strips the hash but keeps the token.

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
