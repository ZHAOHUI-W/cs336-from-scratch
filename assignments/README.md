# Assignments

## A1 — Basics
Location: [`a1-basics/`](a1-basics/)

Upstream: [stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics) (pinned in `a1-basics/UPSTREAM.md`).

```sh
cd assignments/a1-basics
chmod +x scripts/fetch_large_assets.sh
./scripts/fetch_large_assets.sh   # fixtures, snapshots, handout PDF
uv sync
uv run pytest                     # expect NotImplementedError until you fill adapters
```

Implement in `cs336_basics/`. Wire tests through `tests/adapters.py`. Do not edit the test files.

## Later
- `a2-systems/`
- `a3-scaling/`
- `a4-data/`
- `a5-alignment/`
