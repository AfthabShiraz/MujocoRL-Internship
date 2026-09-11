## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## Task 1 — cube reorientation (no touch)

The policy stalls at 25–30° of orientation error with 0/32 deterministic successes. Thirty runs
are logged in `leapXelaMjLab/TRAINING_NOTES.md` (authoritative).

**If you are here to run experiments, read `research/NEXT_EXPERIMENTS.md` first.** It has the
queue, the exact code changes, the scoring rules, and — importantly — the list of things already
tested that must not be re-run. Supporting analysis: `research/STATE.md` (condensed state),
`research/verdicts/SYNTHESIS.md` (the ranking), `research/RESEARCH.md` (literature dossier).
