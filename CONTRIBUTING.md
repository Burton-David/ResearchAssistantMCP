# Contributing

Thanks for your interest in `research-mcp`. This document explains how to set up the project locally and the conventions the codebase follows.

## Setup

```bash
git clone https://github.com/Burton-David/ResearchAssistantMCP
cd ResearchAssistantMCP
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

You will not need any API keys to run the unit tests, the e2e test, or the REPL. To exercise the live integrations and the OpenAI embedder you need:

```bash
export OPENAI_API_KEY=sk-...                    # for the OpenAIEmbedder
export RESEARCH_MCP_INDEX_PATH=~/research_index # for FaissIndex persistence
export SEMANTIC_SCHOLAR_API_KEY=...              # optional, for higher S2 limits
```

## Running the checks

```bash
pytest                            # 67 unit + 1 e2e tests, all offline
pytest -m unit                    # fast subset
RESEARCH_MCP_INTEGRATION=1 pytest # add the live arXiv tests
ruff check src tests              # lint
mypy src                          # type check (strict)
```

A green run on all four is the floor for any PR.

## REPL-first development

Before adding a new module, exercise the API in `research-mcp repl`:

```bash
research-mcp repl
>>> papers = await q("attention is all you need")
>>> await library.ingest(papers[0].id)
>>> cite(papers[0], "ama")
```

If the API feels wrong inside the REPL, change the protocol BEFORE writing five files that depend on it. The four protocols in `src/research_mcp/domain/` are stable contracts — implementations conform to them, not the other way around. If a real implementation forces a protocol change, raise an issue first.

## Code conventions

- **Async everywhere I/O happens.** All four protocols are async; implementations must be too. No CPU-bound work in the protocols.
- **Python 3.11+** syntax: `X | None`, `StrEnum`, structural pattern matching where it helps.
- **No `# type: ignore` without a comment** explaining the suppression. The two existing ignores point at the IPython entry point and the mcp SDK decorators; both note why.
- **No mocks of protocols you own.** Tests use lightweight real implementations (`FakeEmbedder`, `MemoryIndex`, `StaticSource`) — they live in `tests/conftest.py` so they stay close to the test.
- **No invented abstractions.** No "Pipeline", "Workflow", or "Manager" classes unless usage clearly demands them.
- **Comments are sparse.** Only when the *why* is non-obvious — never to restate what the code does.

## Adding a new Source / Index / Embedder / CitationRenderer

Implement the protocol in a new module under the matching package, write a test next to it that uses the protocol surface (no monkey-patching), and re-export from the package `__init__`. The wiring lives in `cli.py` and `mcp/server.py`; add your new implementation there if it should be selectable at runtime.

## Commits

One logical change per commit, with a message that explains the motivation. Keep them honest — `wip` and `fix stuff` don't ship.

## Branches and pull requests

Every change ships through a feature branch off `main` and merges back as a squash-PR. This keeps `main` linear and lets parallel work (humans, cowork sessions, other agents) merge without conflicts.

- **Branch from latest `origin/main`** at the start of every task. Never branch off another in-flight feature branch — when that one merges and squashes, your base disappears and you inherit conflicts.
- **One concern per branch.** Name like `feat/<thing>`, `fix/<thing>`, or `docs/<thing>`. Keep scope tight so the PR can be reviewed in one sitting.
- **Open a PR back to `main`** when ready. Include a "Test plan" section listing what you verified locally (`pytest`, `ruff`, `mypy`).
- **Review the diff before merging,** especially for cowork or other automated PRs. `gh pr view <n>` shows the description; `gh pr diff <n>` shows the change.
- **Squash-merge** so each PR becomes one commit on `main`. The PR title becomes the commit subject.
- **Delete the branch on merge** — `gh pr merge --squash --delete-branch` deletes both the remote and the local branch in one step.
- If two branches touch overlapping code, the second to PR rebases onto updated `main` rather than stacking. Stacking only works when you control the merge order.

For parallel agent sessions specifically: each session branches off `origin/main` at start of its task and opens a PR back independently. Never have one session check out another session's branch as a starting point — that's how merge bombs get built.
