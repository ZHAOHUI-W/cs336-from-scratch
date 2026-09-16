# Design Decision Records (ADRs)

Write an ADR **before** (or while) implementing anything non-obvious. This is how we train architecture mindset.

## When to write one
- Choosing a data structure or algorithm (e.g. BPE pair index)
- Choosing a module boundary (what is a “block”?)
- Memory / precision / sharding choices
- Anything you debated for more than ~10 minutes

## Naming
`NNN-short-title.md` — e.g. `001-bpe-pair-index.md`

Copy [`000-template.md`](000-template.md).
