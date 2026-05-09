"""Shared formatting helpers for citation renderers.

Author parsing is intentionally simple: split on whitespace, treat the last
token as the surname and all preceding tokens as given-names. Sources rarely
expose richly-structured names; this is good enough for AMA-style "Smith JD"
and APA-style "Smith, J. D." output.
"""

from __future__ import annotations

from research_mcp.domain.paper import Author


def split_name(name: str) -> tuple[str, list[str]]:
    """Return (surname, given_names) for a free-form name string.

    Handles the common "First Last" and "First Middle Last" cases. Names with
    a comma are treated as already in "Surname, Given" form.
    """
    raw = name.strip()
    if not raw:
        return "", []
    if "," in raw:
        surname, _, given = raw.partition(",")
        return surname.strip(), [g for g in given.strip().split() if g]
    parts = raw.split()
    if len(parts) == 1:
        return parts[0], []
    return parts[-1], parts[:-1]


def initials(given: list[str]) -> str:
    """Return concatenated initials for AMA/Vancouver style ("J", "JD")."""
    return "".join(g[0].upper() for g in given if g)


def initials_dotted(given: list[str]) -> str:
    """Return dotted, space-separated initials for APA/Chicago ("J. D.")."""
    return " ".join(f"{g[0].upper()}." for g in given if g)


def ama_author(author: Author) -> str:
    surname, given = split_name(author.name)
    bits = initials(given)
    if not surname:
        return bits or author.name.strip()
    return f"{surname} {bits}".strip()


def apa_author(author: Author) -> str:
    surname, given = split_name(author.name)
    bits = initials_dotted(given)
    if not surname:
        return bits or author.name.strip()
    if not bits:
        return surname
    return f"{surname}, {bits}"


def chicago_author(author: Author) -> str:
    surname, given = split_name(author.name)
    if not surname:
        return author.name.strip()
    given_str = " ".join(given)
    return f"{surname}, {given_str}".rstrip(", ")


def mla_author(author: Author) -> str:
    return chicago_author(author)


def join_with_and(parts: list[str]) -> str:
    """Style: "A, B, and C". Used by APA / MLA / Chicago."""
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{', '.join(parts[:-1])}, and {parts[-1]}"
