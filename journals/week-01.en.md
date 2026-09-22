# Week 1 journal (English)

Chinese version: [week-01.zh.md](./week-01.zh.md).

## Day 1 — Map A1 (2026-09-16)

### What I built
- No implementation yet (map day only).
- Oriented on Assignment 1 deliverables and the `tests/adapters.py` contract.

### Design decisions (link ADRs)
- None yet. Day 2 starts ADR `001-adamw-state-layout`.

### What I locked in
**A1 end-to-end pipeline**
1. Train BPE tokenizer (bytes → merges → vocab)
2. Encode text → integer token IDs
3. Transformer LM predicts next-token distribution
4. Cross-entropy loss + AdamW updates
5. Sample / evaluate perplexity (TinyStories)

**`adapters.py` as interface**
`adapters.py` is a thin facade: tests own the weights and shapes; my code owns the algorithms behind stable function signatures (`run_*` / `get_*`). I implement real modules in `cs336_basics/` and wire them through adapters.

**Week 1 vs Week 2 adapters**
- Week 1: `get_adamw_cls`, `get_tokenizer`, `run_train_bpe`, loss/schedule/checkpoint/batch helpers
- Week 2: linear, embedding, RMSNorm, SwiGLU, RoPE, attention, transformer block/LM

**Tokenizer reminder**
UTF-8 bytes → BPE merges → token IDs; encode/decode must round-trip; special tokens stay unsplit.

**Loss reminder**
Next-token CE $-\log p(\text{correct})$; uniform over vocab $V$ → $\log V$.

### Bugs that taught me something
- N/A (no coding day).

### Architecture takeaway
Interfaces (`adapters.py`) are part of the design — they freeze what must stay stable while internals can change.

### Next week’s risk / tomorrow
Day 2: draft ADR 001 (AdamW state layout + decoupled weight decay), then implement AdamW and match `torch.optim.AdamW` on a toy step.

## Day — BPE `train_bpe` (2026-09-21)

#### What I built
- Implemented `cs336_basics/train_bpe.py`, wired through `tests/adapters.py` → `run_train_bpe`.
- Accepted: `test_train_bpe`, `test_train_bpe_speed` (~0.47s < 1.5s), `test_train_bpe_special_tokens`.

#### Naive BPE (correctness first)

Pipeline:

1. Read corpus (UTF-8; avoid `errors='ignore'`, which silently drops bytes and changes merge stats).
2. Split on `special_tokens` (longer first if multiple). Specials do **not** enter pre-tokenization or pair counts; they are still added to `vocab`.
3. On each chunk, GPT-2 `PAT` via `regex.finditer` for pre-tokenization.
4. Each pre-token → UTF-8 bytes → word frequency table `dict[tuple[bytes, ...], int]` (tuple elements are byte-string pieces, not vocab ids).
5. Init `vocab = specials + 256` single bytes; loop until `len(vocab) == vocab_size`:
   - count adjacent pairs (or maintain counts);
   - pick highest frequency; ties → lexicographically **greater** pair;
   - `merges.append(pair)`;
   - replace that adjacent pair in word sequences with `C = A + B` (bytes concat);
   - add `C` to `vocab` **once per merge**;
6. Return `(vocab, merges)`.

#### Optimization (after correctness)

**Bottleneck (cProfile):** each merge rescanned the whole word table to rebuild pair counts — cost scales with (unique words × length × num merges). Pre-tokenization was cheap on `corpus.en`.

**Changes:**
- Maintain `pair_freq: pair → count` across merges.
- Maintain `pair_to_words: pair → set of words` containing that pair. After choosing `best_pair=(A,B)`, only process words in `pair_to_words[(A,B)]`; skip the rest.
- For each affected word: remove its contribution from `pair_freq` / index, apply left-to-right non-overlapping merge, write the new word back, add new pairs to `pair_freq` / index.
- Update `freq` **in place** (`pop` old word, add new); avoid full-table `dict` copy each merge.

#### Bugs that taught me something
- Keep sequences / merges as **`bytes`**, never mix vocab int ids into word tuples.
- `vocab[len(vocab)] = C` must sit **outside** the per-word loop; putting it inside grew vocab by (#affected words) per merge and stopped after ~2 merges vs ~243.
- `pair_freq` and `pair_to_words` must stay in sync; otherwise `KeyError` on `freq.pop` or `pair_to_words.pop(best_pair)`.
- Loop body indentation matters: only `freq.pop` inside `for` left other words half-updated.

#### Architecture takeaway
Indexes are part of the algorithm's contract: `pair_to_words` is not a micro-optimization bolted on — it defines which words are allowed to change each merge. Correctness = same merges as full recount; speed = touch only those words and keep counts consistent.

#### Next
- Remove debug `merges.txt` write if still present.
- Implement tokenizer **encode/decode** (`get_tokenizer`) and `tests/test_tokenizer.py`.

## Day — Tokenizer encode/decode（2026-09-22）

#### What I did
- Implemented `Tokenizer` in `cs336_basics/tokenizer.py`: `__init__`, `encode`, `decode`, `encode_iterable`.
- Wired `tests/adapters.py` `get_tokenizer` to `return Tokenizer(vocab, merges, special_tokens)`.
- Patched `tests/test_tokenizer.py` for Windows: optional `resource` import; skip hard `RLIMIT_AS` when unavailable.
- Fetched GPT-2 fixtures (`gpt2_vocab.json` / `gpt2_merges.txt`) for tiktoken-alignment tests.

#### Contract with `train_bpe`
- `train_bpe` returns `vocab: dict[int, bytes]` and `merges: list[tuple[bytes, bytes]]` (**in training order**).
- The tokenizer **consumes** these for encode/decode; it does not learn new merges.
- Specials are in `vocab`; at encode time a full special match becomes one id and does **not** enter BPE.

#### encode (text → ids)
1. **Special split** (longer first): matched specials → `bytes_to_id`; ordinary spans go next.
2. **GPT-2 PAT pre-tokenization** on ordinary spans (same regex as `train_bpe`).
3. **BPE**: each pre-token → UTF-8 → single-byte sequence; apply merges **in list order**.
4. **Lookup**: each resulting bytes piece → token id.

#### decode / encode_iterable
- `decode`: id → vocab bytes, concat, then UTF-8 decode (e.g. `errors="replace"` for bad sequences).
- `encode_iterable`: `encode` each string chunk, yield ids one-by-one (streaming).

#### Pitfalls
- Byte-only lookup: roundtrip can pass while tiktoken id alignment fails (words split to letters).
- `merges` is an ordered list applied in sequence—not “pick highest frequency again.”
- Do not nest a second `get_tokenizer` that returns `None`; remove duplicate `decode` / `encode_iterable`.

#### Why `merges` must be an ordered list
Training learns a *sequence* of merge rules—“who merges before whom.” Encoding must replay that same order so segmentation matches training.

Each chosen pair is counted only after earlier merges have already been applied. For example `(e, s)→es` may appear first; only then can `(es, t)→est` exist. Without `es`, the pair `(es, t)` never occurs. So merges are not an unordered bag of rules but an ordered list of dependent steps: smaller index = learned earlier = applied earlier at encode time.

If encode instead re-picks “highest frequency pair in the current word,” or applies merges as an unordered set, the segmentation diverges and token ids will not match. `list[tuple[bytes, bytes]]` stores that order; a `dict` or `set` would drop it.

In short: order is part of the tokenizer; dropping order is a different tokenizer.

#### Acceptance
- Roundtrip and tiktoken-alignment cases in `tests/test_tokenizer.py` passed.

#### Next
- Full TinyStories `train_bpe` experiment (e.g. compression), or A1 Transformer.

## Day — TinyStories `train_bpe` 实验（2026-09-22）

#### What I did
- Handout **train_bpe_tinystories**: byte-level BPE on `TinyStoriesV2-GPT4-train.txt`.
- Config: `vocab_size=10000`, `special_tokens=["<|endoftext|>"]`.
- Script `scripts/train_tinystories_bpe.py`; implementations:
  1. Incremental-merge `train_bpe`
  2. `train_bpe_multiprocessing.py` — chunked pretok via `find_chunk_boundaries` + `Pool`
  3. `train_bpe_multiprocessing_buckets.py` — `PairFrequencyBuckets` for best-pair selection
- Artifacts (not committed): `artifacts/tinystories_bpe/vocab_*.txt`, `merges_*.txt`.

#### Full-corpus numbers (vocab_size=10000, num_workers=4, train)

- **Buckets** (`train_bpe_multiprocessing_buckets`): wall **404.75 s (about 6.7 min)**; peak memory (incl. children) **4610.33 MB (about 4.61 GB)**; longest token `b' accomplishment'` (len 15).
- **Baseline** (same pretok workers, full-map `max` for best pair): about **450 s / 4.6 GB**, same longest token.
- Buckets about **45 s (~10%)** faster; memory similar. Still under the 30 min / 30 GB hard caps; above the under-2-min hint.
- Correctness: bucket vs non-bucket `merges_train_process4*.txt` identical; expect `len(vocab)=10000`, `len(merges)=9743`.
#### Multiprocess pretok
- Official `find_chunk_boundaries` on `<|endoftext|>`; top-level `_count_chunk` for Windows spawn; `num_workers=4`.
- Parent `peak_wset` omits workers — sample `rss(parent)+rss(children)` during the run; don’t label raw bytes as MB.

#### Profiling (deliverable b)
- `cProfile` is parent-only: Pool wait dominates visually while workers do pretok.
- Parent hotspot: per-merge `max(pair_freq)`. Prefer split timers (pretok vs merge) or profile with `num_workers=1`.

#### Bucketing
- Frequency → bucket of pairs; `best_pair` = lex-max in highest non-empty bucket (lazy `max_count`).
- Correctness: full-run `merges_train_process4*.txt` **byte-identical** with vs without buckets.
- Cost: updates ~O(1); `best_pair` is O(|bucket|), costly when many pairs share count 1 late in training.

#### Handout answer drafts
- **(a)** ~6.7 min (buckets), ~4.61 GB peak, longest `b' accomplishment'`.
- **(b)** Parent bottleneck is repeated full `max` over pairs; pretok cost sits in workers. Bucketing removes full-map scans each merge; use wall-clock splits for pretok vs merge.

#### Next
- Optional: chase &lt;2 min hint; then OWT 32k BPE or Transformer.
