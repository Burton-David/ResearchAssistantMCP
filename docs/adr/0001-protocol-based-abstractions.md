# ADR 0001: Protocol-based abstractions over ABCs

**Status:** Accepted, 2026-05-09
**Decider:** D. Burton

## Context

The core abstractions (`Source`, `Index`, `Embedder`, `CitationRenderer`) need to be extensible. Anyone should be able to write a new Source for IEEE papers, plug in a Chroma-backed Index, swap in local sentence-transformers — without subclassing or registering anything.

Two options were considered:

1. **Abstract base classes** (`abc.ABC` + `@abstractmethod`). Implementations must inherit explicitly.
2. **`typing.Protocol`** with `@runtime_checkable`. Implementations match by structural shape; no inheritance required.

## Decision

`typing.Protocol`.

## Reasoning

- Structural typing means a third-party class that already exposes a method shaped like our `Source.search` is automatically compatible. ABCs would require an adapter wrapper.
- Runtime `isinstance` checks still work via `@runtime_checkable`, so the wiring layer can validate dependencies at startup if needed.
- Tests can use plain dataclasses or simple classes as fakes without subclassing anything. The fake stays close to the test, not in a `mocks/` directory.
- Pyright and mypy enforce protocol conformance at type-check time — the safety guarantee is the same.

## Trade-offs

- Protocols are slightly less discoverable than ABCs. We mitigate this by keeping all four in `research_mcp.domain` and re-exporting them from the package root.
- Default method implementations (a real strength of ABCs) are not available. None of our four protocols benefit from defaults — they are pure interfaces, and a default `search()` implementation has no plausible body.

## Consequences

- New `Source` / `Index` / `Embedder` / `CitationRenderer` implementations don't have to live in this repo. Third parties can publish their own packages and they will work without us shipping a release.
- Refactoring an existing implementation no longer risks breaking inheritance — only the protocol contract.
- If we later need shared default behavior (unlikely), we can add a concrete base class that implements the Protocol and have implementations subclass it for convenience. Protocols don't preclude inheritance; they just don't require it.
