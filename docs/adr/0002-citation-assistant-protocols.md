# ADR 0002: Four protocols for the citation-assistant mission

**Status:** Accepted, 2026-05-10
**Decider:** D. Burton

## Context

The original five protocols (`Source`, `Index`, `Embedder`,
`CitationRenderer`, `Reranker`) describe the *data plane* — where papers
come from, where they're indexed, how they become vectors, how they're
rendered as citations, how candidates are reranked. They support a
"search and ingest papers" workflow well.

The mission this repo aims at is a step bigger than that: a
citation-finding research assistant. A user pastes a draft paragraph;
the system identifies which sentences make claims that need citations,
finds candidate papers per claim across multiple databases, scores
each candidate's quality, and explains the recommendation. The
killer demo is one tool call (`assist_draft(text)`) that orchestrates
all of it.

That mission needs four new abstractions on top of the data plane.

## Decision

Add four protocols in `src/research_mcp/domain/`:

- `ClaimExtractor` — text → list of typed Claims
- `Chunker` — Paper → list of TextChunks (section-aware)
- `CitationScorer` — Paper → CitationQualityScore (with breakdown)
- `PaperAnalyzer` — Paper → PaperAnalysis (structured fields)

All four are `typing.Protocol` declarations following the precedent set
in ADR-0001.

## Reasoning

Each new protocol earns its place by sitting orthogonal to the existing
abstractions and to each other:

### ClaimExtractor is orthogonal to Source and Embedder

Claim extraction operates on user-supplied draft text — never on Papers
returned from a Source. The output (Claims with types and suggested
search terms) is the *input* to a search pipeline that uses Source +
Embedder. Conflating the two would force every Source implementation to
also know about claims. They shouldn't.

Different implementations also matter: the default `SpacyClaimExtractor`
is pattern-based + spaCy NER (no API cost, ~80% precision); a future
`LLMClaimExtractor` would be more accurate but cost real tokens per
draft. A user picks one via env, same as embedder/reranker today.

### Chunker is orthogonal to Embedder

A 30-page PDF must be chunked before embedding (each model has a token
limit) AND before LLM analysis. Putting chunking inside the Embedder
would force every Embedder implementation to re-implement
section-detection regex, and would couple the Embedder protocol to the
Paper data model (Embedder today only knows about strings).

Chunking is also useful without an Embedder — `analyze_paper` runs an
LLM over chunks; `chunk_paper` exposes chunks directly to the LLM
caller for inspection. The protocol earns its keep across all three
consumers.

### CitationScorer is orthogonal to Reranker

The Reranker scores `(query, paper)` pairs for relevance — "how related
is this paper to what the user is asking about?". The CitationScorer
scores papers for citation *quality* — "would this be a good paper to
cite, regardless of what the citing claim is?" Venue tier, author
h-index, citation velocity, predatory-journal flags. Two distinct
signals; combining them would lose information.

The two compose: a candidate's final ranking is some function of
relevance (from the Reranker) and quality (from the Scorer). The
weighting lives in the `CitationService`, not the protocols.

### PaperAnalyzer is orthogonal to everything

It's the only protocol that's strictly LLM-driven (the others have
non-LLM implementations). It produces a `PaperAnalysis` with
methodology, contributions, limitations, datasets, etc. — fields the
LLM caller and the citation explainer both consume. Not coupled to
the Embedder (analysis is structural, not vector-based) or the
Scorer (analysis answers "what does this paper say?", not "is it a
good citation?").

## Consequences

- Three new services compose the new protocols: `CitationService`
  (claim → candidate citations + scores + explanations),
  `AnalysisService` (paper → structured analysis), and `DraftService`
  (the assist_draft orchestrator). Same composition pattern as the
  existing `SearchService` / `LibraryService` / `DiscoveryService`.
- The MCP tool surface grows from 7 to 15 tools. Each new tool composes
  one or two of the new protocols. The killer-demo `assist_draft` tool
  composes all four new protocols plus the data-plane services.
- Implementations are optional. A user with no spaCy installed can
  still run search/ingest/cite — the citation-assistant tools simply
  refuse with a helpful hint, mirroring how `library_search` refuses
  when no embedder is configured.
- ADR-0001's structural-typing argument applies cleanly: a third party
  can implement `ClaimExtractor` for their own NER stack, or a domain-
  specific scorer for medicine vs. CS, without subclassing anything.

## Alternatives considered

- **Bake everything into one `CitationAssistant` class.** Simplest at
  first, but mixes orthogonal concerns and makes every change a
  whole-system change. Rejected.
- **Use one `Analyzer` protocol that returns claims OR analysis OR
  scores depending on a kind parameter.** Marginally fewer types in
  `domain/` but at the cost of an awkward union return type and a
  non-orthogonal contract. Rejected.
- **Keep claim extraction outside `domain/` because it's not
  paper-shaped.** Considered. But the domain *is* what the system is
  about, and "extracting claims that need citations" is core to the
  citation-assistant mission. Putting `ClaimExtractor` in `domain/`
  signals to a reader that this is a primary concern, not a utility.
