<!--
Thanks for the PR. A few conventions from CONTRIBUTING.md to keep in mind:

  - Branch off latest origin/main; squash-merge on landing.
  - One concern per branch — keep scope tight enough to review in one sitting.
  - Real implementations of protocols we own, not mocks (see tests/conftest.py).
  - Link any related issue with "Closes #N" so it auto-closes on merge.
-->

## Summary

<!-- One paragraph: what changed and why. The "why" matters more than the "what" -
     reviewers can read the diff for the what. -->

## Test plan

<!-- What did you actually run locally? Check the boxes that apply; add anything else. -->

- [ ] `pytest -q` — all green
- [ ] `ruff check src tests` — clean
- [ ] `mypy src` — clean
- [ ] REPL-verified the new behavior (paste the snippet you ran, briefly)
- [ ] Manual MCP test (Claude Desktop, Claude Code, or a custom client) — if user-facing
- [ ] Other: _(describe)_

## Notes for the reviewer

<!-- Anything non-obvious about the design? Tradeoffs you considered and ruled out?
     Subtle invariants the test suite doesn't cover? Drop them here so the reviewer
     doesn't have to re-derive them from the diff. -->
