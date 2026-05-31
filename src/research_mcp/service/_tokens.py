"""Shared text tokenization for cross-source matching.

The same NFKD-fold + lowercase + split-on-non-alphanumeric routine was
copied across three call sites — cross-source title dedup
(`service/search.py`), `find_paper` Jaccard matching (`service/discovery.py`),
and the deterministic test reranker (`reranker/fake.py`). They have to fold
text identically or dedup and matching silently disagree, so the routine
lives in one place.
"""

from __future__ import annotations

import re
import unicodedata

NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_unicode(text: str) -> str:
    """NFKD-fold to ASCII, dropping diacritics and other non-ASCII bytes.

    "Schölkopf" becomes "Scholkopf" so accented and unaccented spellings of
    the same name collide as intended during matching.
    """
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def tokenize(text: str, *, stopwords: frozenset[str] = frozenset()) -> list[str]:
    """Fold, lowercase, and split `text` into alphanumeric tokens, in order.

    Empty fragments and any token in `stopwords` are dropped. Lowercasing
    happens before the split because `NON_ALNUM_RE` only spans `a-z0-9`;
    callers that want a set wrap the result.
    """
    folded = normalize_unicode(text).lower()
    return [t for t in NON_ALNUM_RE.split(folded) if t and t not in stopwords]
