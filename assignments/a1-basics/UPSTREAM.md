# Upstream pin

Vendored from [stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics) at commit `a158843b20107949f1a8d7df1b05cd33b9166712`.

Large fixtures, snapshots, handout PDF, `corpus.en`, and `uv.lock` are **not** stored in this study repo (API size limits). After clone, run:

```sh
chmod +x scripts/fetch_large_assets.sh
./scripts/fetch_large_assets.sh
```

Then:

```sh
uv sync
uv run pytest
```

Expect `NotImplementedError` until you fill `tests/adapters.py` and implement `cs336_basics/`.
