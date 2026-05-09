"""Tiny stdlib-only `.env` loader for entry points.

Walks up from the current working directory looking for `.env` and sets any
keys it finds in `os.environ` — but only if they aren't already set, so a
real `export OPENAI_API_KEY=...` always wins. Intentionally avoids the
`python-dotenv` package so this stays a runtime-cost-free convenience.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(start: Path | None = None) -> Path | None:
    """Walk up looking for a `.env`. Returns the path loaded, or None.

    Tries the current working directory first, then the directory containing
    this module. The second path matters for editable installs where the
    package lives at `<project>/src/research_mcp/` — we can find a project-
    root `.env` even if the server was spawned with a foreign cwd by an MCP
    client.

    Skips comments, blank lines, malformed lines. Strips one layer of single
    or double quotes from values. Doesn't handle multi-line values, `export `
    prefixes, or interpolation — `python-dotenv` does all that, and projects
    that need it are past the convenience this helper exists to cover.
    """
    candidates: list[Path] = []
    cwd = (start or Path.cwd()).resolve()
    candidates.extend([cwd, *cwd.parents])
    package_dir = Path(__file__).resolve().parent
    candidates.extend([package_dir, *package_dir.parents])

    seen: set[Path] = set()
    for directory in candidates:
        if directory in seen:
            continue
        seen.add(directory)
        candidate = directory / ".env"
        if not candidate.is_file():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if value and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            os.environ.setdefault(key, value)
        return candidate
    return None
