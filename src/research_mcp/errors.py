"""Cross-cutting exceptions for research-mcp.

Kept in a single small module so adapters and services can raise / catch
without circular imports. Domain types and protocols stay free of error
imports — the Source `fetch` contract still declares `Paper | None` as its
nominal return; `SourceUnavailable` is a side channel for distinguishing
transient failures from "id not owned by this Source."
"""

from __future__ import annotations


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

        httpx raises errors with messages like "Client error '429 ...' for
        url 'https://...long-url...?...&...' For more information check
        https://developer.mozilla.org/...". That entire ~600-char blob ends
        up in cite_paper / get_paper error responses; LLM clients render
        it inline. Collapse to a status-code-or-summary string."""
        text = self.reason
        # Most common case: httpx's error string starts with "<class>: <method>"
        # then a quoted code, then "for url '...'". Cut at "for url".
        for marker in (" for url", "\nFor more information"):
            if marker in text:
                text = text.split(marker, 1)[0]
                break
        return text.strip().strip("'").strip()
