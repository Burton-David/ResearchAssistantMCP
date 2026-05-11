"""Field detection for the field-aware citation scorer.

Reads two metadata signals populated by the source adapters:

  * `arxiv_primary_category` — populated by `ArxivSource._parse_entry`
    from the Atom feed (`cs.LG`, `math.AG`, etc.). Most specific
    signal; tried first.
  * `openalex_field` — populated by `OpenAlexSource._parse_work` from
    `primary_topic.field.display_name` ("Computer Science", "Medicine",
    "Mathematics", "Physical Sciences"). Coarser-grained but available
    for non-arXiv records.

Falls back to `Field.DEFAULT` when neither key is set or when the
value doesn't map to one of our coarse buckets. The five fields are
chosen because they have distinctively different citation-velocity
and longevity profiles — adding "Biology" would require deciding
whether it tracks CS (fast-moving) or Medicine (slow), and neither
fits cleanly. Conservative: when in doubt, fall through to default.
"""

from __future__ import annotations

from enum import StrEnum

from research_mcp.domain.paper import Paper


class Field(StrEnum):
    """Coarse-grained research fields that drive scorer overrides.

    Values are stable strings so they can ride in `CitationQualityScore.
    factors` for telemetry / debugging without exposing the enum type.
    """

    CS = "cs"
    MEDICINE = "medicine"
    MATH = "math"
    PHYSICS = "physics"
    DEFAULT = "default"


# arXiv primary-category PREFIX -> Field. The category is dotted
# ("cs.LG", "physics.flu-dyn"); we match on the bit before the dot.
# `stat.*` and `q-fin.*` are statistical methods used heavily in ML
# so they map to CS; `q-bio.*` is quantitative biology, closest to
# medicine. Astrophysics, condensed matter, high-energy variants all
# share the physics citation pattern.
_ARXIV_PREFIX_MAP: dict[str, Field] = {
    "cs": Field.CS,
    "stat": Field.CS,
    "math": Field.MATH,
    "physics": Field.PHYSICS,
    "cond-mat": Field.PHYSICS,
    "astro-ph": Field.PHYSICS,
    "hep-ph": Field.PHYSICS,
    "hep-th": Field.PHYSICS,
    "hep-ex": Field.PHYSICS,
    "hep-lat": Field.PHYSICS,
    "quant-ph": Field.PHYSICS,
    "gr-qc": Field.PHYSICS,
    "nucl-th": Field.PHYSICS,
    "nucl-ex": Field.PHYSICS,
    "q-bio": Field.MEDICINE,
}


# OpenAlex `primary_topic.field.display_name` -> Field. The display
# names come from OpenAlex's controlled vocabulary; lowercased before
# lookup. Multiple aliases collapse to MEDICINE so anything in the
# clinical/biomedical stack is treated the same.
_OPENALEX_FIELD_MAP: dict[str, Field] = {
    "computer science": Field.CS,
    "medicine": Field.MEDICINE,
    "nursing": Field.MEDICINE,
    "dentistry": Field.MEDICINE,
    "health professions": Field.MEDICINE,
    "pharmacology, toxicology and pharmaceutics": Field.MEDICINE,
    "immunology and microbiology": Field.MEDICINE,
    "biochemistry, genetics and molecular biology": Field.MEDICINE,
    "mathematics": Field.MATH,
    "physics and astronomy": Field.PHYSICS,
}


def detect_field(paper: Paper) -> Field:
    """Resolve the paper's field from metadata hints.

    Priority order: arXiv primary_category > OpenAlex field > DEFAULT.
    arXiv is more specific (subdiscipline-level) so we prefer it when
    both are present.
    """
    arxiv_cat = (paper.metadata.get("arxiv_primary_category") or "").lower()
    if arxiv_cat:
        prefix = arxiv_cat.split(".", 1)[0]
        field = _ARXIV_PREFIX_MAP.get(prefix)
        if field is not None:
            return field

    openalex_field = (paper.metadata.get("openalex_field") or "").lower().strip()
    if openalex_field:
        field = _OPENALEX_FIELD_MAP.get(openalex_field)
        if field is not None:
            return field

    return Field.DEFAULT
