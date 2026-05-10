"""Curated venue lists for the heuristic scorer.

`TOP_VENUE_PATTERNS` matches strings (case-insensitive) that strongly
imply a top-tier conference or journal. The list is conservative — we
prefer to under-flag than over-credit a no-name workshop.

`PREDATORY_PATTERNS` matches generic-sounding multi-discipline open-
access journals that frequently appear on Beall's list updates. A
match here is a flag, not a rejection — we surface it as a warning
and zero the venue dimension. The user decides whether to trust the
paper.

`ARXIV_CATEGORY_BOOST` are arXiv subject classifications associated
with high-quality preprints (most CS-AI / stat-ML papers are also
NeurIPS / ICML).
"""

from __future__ import annotations

# Distinctive multi-word venue names. Match via word-boundary regex so
# "ieee tpami" matches "IEEE TPAMI" and "IEEE Transactions on Pattern
# Analysis and Machine Intelligence (TPAMI)" alike. The full-form
# entries ("neural information processing systems") were added after
# the chaos test caught Vaswani scoring 50/100 — S2's enrichment
# populated venue as the full long form, which didn't match the
# short-form acronyms-only list.
TOP_VENUE_PARTIAL_PATTERNS: frozenset[str] = frozenset(
    {
        # CS / ML conference acronyms
        "neurips", "nips", "icml", "iclr", "cvpr", "iccv", "eccv",
        "emnlp", "naacl", "aaai", "ijcai", "siggraph", "icra",
        "popl", "pldi", "osdi", "sosp", "vldb", "sigmod",
        # CS / ML conference long forms — S2 returns these on enrichment
        "neural information processing systems",
        "international conference on machine learning",
        "international conference on learning representations",
        "annual meeting of the association for computational linguistics",
        "conference on empirical methods in natural language processing",
        "conference on computer vision and pattern recognition",
        "international conference on computer vision",
        "european conference on computer vision",
        "aaai conference on artificial intelligence",
        # CS / ML journals
        "ieee tpami", "jmlr", "tacl",
        "computational linguistics",
        "acm computing surveys",
        "transactions on machine learning",
        # Biology / Medicine — multi-word, distinctive
        "nature genetics", "nature biotechnology", "nature medicine",
        "nature physics", "nature methods", "nature chemistry",
        "current biology", "embo journal",
        "n engl j med", "annals of internal medicine",
        "journal of clinical oncology",
        # Physics
        "physical review letters", "physical review x",
        "reviews of modern physics",
        # Mathematics
        "annals of mathematics", "inventiones mathematicae",
        "journal of the american mathematical society",
        "acta mathematica",
    }
)

# Single-word and ambiguous patterns that must match the *whole* venue
# string after lowercasing+stripping. "science" the journal vs.
# "Computer Science" / "Procedia Computer Science" needs this stricter
# matching to avoid false positives. Common ML conferences with
# 3-letter acronyms ("acl", "kdd", "sigir") also live here because
# those letter sequences appear inside longer names.
TOP_VENUE_EXACT_PATTERNS: frozenset[str] = frozenset(
    {
        "nature", "science", "cell",
        "pnas", "elife",
        "lancet", "nejm", "jama", "bmj",
        "circulation",
        "acl", "kdd", "sigir", "rss", "icse",
        "stoc", "focs", "soda",
    }
)

# Backwards-compat alias kept so callers that imported the union name
# still work; new code uses the two more-specific names above.
TOP_VENUE_PATTERNS: frozenset[str] = (
    TOP_VENUE_PARTIAL_PATTERNS | TOP_VENUE_EXACT_PATTERNS
)

# Common patterns in predatory venue names. Several are intentionally
# loose because that's how the actual venues mask themselves.
PREDATORY_PATTERNS: tuple[str, ...] = (
    "open access journal of multidisciplinary",
    "international journal of advanced research",
    "global journal of",
    "world journal of advances",
)

# arXiv subject categories that correlate with peer-reviewed venue
# matches at a higher rate (lots of NeurIPS/ICML papers also live here).
# Used to give arXiv-only papers a venue baseline above 0.
ARXIV_CATEGORY_BOOST: frozenset[str] = frozenset(
    {"cs.lg", "cs.ai", "cs.cl", "cs.cv", "stat.ml"}
)
