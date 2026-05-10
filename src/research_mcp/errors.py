"""Cross-cutting exceptions for research-mcp.

Kept in a single small module so adapters and services can raise / catch
without circular imports. Domain types and protocols stay free of error
imports — the Source `fetch` contract still declares `Paper | None` as its
nominal return; `SourceUnavailable` is a side channel for distinguishing
transient failures from "id not owned by this Source."
"""

from __future__ import annotations

import re

_HTTP_CODE_RE = re.compile(r"\b([45]\d{2})\b")


class ResearchMCPError(Exception):
    """Base for all research-mcp-defined exceptions."""


class SourceUnavailable(ResearchMCPError):  # noqa: N818  # state-of-the-source name reads better than SourceUnavailableError
    """Raised by a `Source.fetch` when a transient failure prevents resolving
    the id — network errors, HTTP 429, 5xx, timeouts.

    Distinguishes "I tried and couldn't reach the upstream" from "I tried and
    confirmed this id isn't mine" (which Sources signal by returning None).
    Callers (LibraryService, the cite_paper MCP handler) use this distinction
    to render error messages that tell the user whether to retry or stop.

    Sources MAY swallow the exception inside their `search` implementation —
    the Source protocol still requires search to never raise on transient
    errors, returning an empty Sequence instead. But `fetch` callers benefit
    from knowing the difference, so the typed exception is the right tool
    there.
    """

    def __init__(self, source_name: str, reason: str) -> None:
        super().__init__(f"{source_name}: {reason}")
        self.source_name = source_name
        self.reason = reason

    def short_reason(self) -> str:
        """A user-friendly one-line summary, stripping URLs/stack traces.

        httpx raises errors like "Client error '429 ' for url
        'https://...long-url...?...' For more information check
        https://developer.mozilla.org/...". The whole ~600-char blob would
        end up in cite_paper / get_paper error responses verbatim. Collapse
        to a clean form by extracting the HTTP status code when present,
        else stripping URLs and quoting noise.
        """
        text = self.reason
        match = _HTTP_CODE_RE.search(text)
        if match:
            return f"HTTP {match.group(1)}"
        for marker in (" for url", "\nFor more information"):
            if marker in text:
                text = text.split(marker, 1)[0]
                break
        return text.strip().strip("'").strip()
