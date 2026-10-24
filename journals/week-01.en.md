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


## Day — Linear and Embedding (2026-10-12 to 2026-10-23)

#### What I did
- §3.3.2 Linear (no bias): `cs336_basics/linear.py` + `adapters.run_linear`.
- §3.3.3 Embedding: `cs336_basics/embedding.py` + `adapters.run_embedding`.
- Fixed a pytest collection error: moved the demo code in `pretokenization_example.py` under `if __name__ == "__main__":` so importing it no longer runs `open(...)`.

#### Easy-to-confuse concepts
1. **Weight shapes**: Linear stores `(d_out, d_in)` and computes `y = x Wᵀ`; Embedding stores `(vocab_size, d_model)`, where row i is the vector for token i. They are different operations.
2. **Embedding forward is a lookup, not a matmul**: `self.weight[token_ids]`. An einsum would treat token ids as numeric weights and sum rows, which is wrong. Output shape = shape of `token_ids` + `d_model`, e.g. `(1, 3)` in, `(1, 3, d_model)` out.
3. **Parentheses vs brackets**: `self.weight(token_ids)` calls the Parameter like a function and raises `'Parameter' object is not callable`; use bracket indexing to fetch rows.
4. **einops.einsum argument order**: tensors first, pattern string last.
5. **Initialization**: Linear uses truncated normal with σ² = 2/(d_in+d_out), truncated at ±3σ, to keep forward and backward signal scale roughly constant (wider layers get smaller σ). Embedding uses N(0,1) truncated to [-3, 3]. A lookup does not sum over d_in inputs, so there is no variance build-up with width, and std=1 is fine.
6. **Position**: the same token gets the identical embedding at every position (indexing does not encode position). In A1, position enters later via RoPE (rotary position embedding) inside attention.
7. **The embedding table is trained**: it is an `nn.Parameter` updated by the optimizer; each step only rows for tokens present in the batch get gradients. Embedding and the LM head are separate parameters (no weight tying in A1).
8. **What the adapter does**: tests only call `adapters.run_*`; the adapter builds my module, copies the test weights into `.data`, and calls `forward`.
9. **Test assets**: if `tests/fixtures/ts_tests/model.pt` is missing, run `scripts/fetch_large_assets.sh` first.
10. **Parameter name**: name the Embedding parameter `weight` (not `weights`) so later full-model `load_state_dict` keys match.

#### Verification
- `uv run pytest -k test_linear`: PASSED.
- `uv run pytest -k test_embedding`: PASSED.

#### Next
- §3.4 RMSNorm.
- Small cleanup: unused `math` / `einsum` imports in `embedding.py`; dead `raise NotImplementedError` after `return` in `run_embedding`.
#### Recall questions
1. **Does the same token get the same embedding at different positions? How does the model tell positions apart?**
   - My answer: same vector; positions are told apart by the index.
   - Correction: same vector ✅. But the index only selects *which* token, not *where* it sits, so the Embedding layer knows nothing about position. In A1, position comes from RoPE (rotary position embedding) in attention: Q and K are rotated by position, so the same word at position 1 vs 5 yields different attention scores.
2. **Why does Embedding init use std=1 while Linear uses √(2/(d_in+d_out))?**
   - My answer: keep signal scale roughly constant in forward and backward passes; wider inputs get smaller σ.
   - Correction: the Linear part is right ✅. Missing the Embedding part: a Linear output sums d_in input×weight terms, so variance grows with width and weights must shrink; an Embedding just fetches one row with no summation, so variance does not grow with width and unit scale (std=1) is fine.

## Day — RMSNorm and SwiGLU (2026-10-23 ~ 2026-10-24)

#### What I did
- §3.4.1 RMSNorm: `cs336_basics/rmsnorm.py` + `adapters.run_rmsnorm`.
- §3.4.2 SwiGLU feed-forward: `cs336_basics/swiglu.py` (module-level `silu` function + `SwiGlu` module) + `adapters.run_silu` / `adapters.run_swiglu`.
- Small cleanup: removed unused `math` / `einsum` imports from `embedding.py`.

#### Common small mistakes
1. **`super.__init__()` missing parentheses**: must be `super().__init__()`. Without them the parent init never runs, and registering parameters fails later.
2. **Accepting `device` / `dtype` but not passing them on**: happened in both RMSNorm and SwiGLU. If `__init__` takes them, pass them to `torch.ones(...)` or to submodules `LinearNoBias(..., device=device, dtype=dtype)`; never hardcode `None`.
3. **Typo `x ** x`**: squaring is `x ** 2`. This kind of bug raises no error and just computes the wrong result.
4. **`math.*` cannot handle tensors**: `math.sqrt` and `math.sigmoid` (which does not even exist) work on single Python numbers only. For tensors use `torch.sqrt`, `torch.sigmoid`.
5. **Wrong argument in the adapter**: `run_rmsnorm` first passed the weights tensor as `d_model`, so `torch.ones()` raised a TypeError. Build the module (pass dimensions) and load the weights (pass tensors) as two separate steps.
6. **Writing einsum on a module**: `self.w1` is a `LinearNoBias` module whose forward already does the einsum, so just call `self.w1(x)`. `silu.einsum(...)` is not valid.
7. **Mixing up `@` and `*`**: the ⊙ in the formula is elementwise multiplication, i.e. `*`; `@` is matrix multiplication.
8. **Loading weights at the wrong level**: `swiglu.w1 = w1_weight` tries to replace the whole submodule with a tensor; the right way is `swiglu.w1.weight.data = w1_weight`, which loads into the submodule's parameter.

#### Things worth noting
1. **RMSNorm formula**: `x / sqrt(mean(x²) + eps) * g`. Compared with LayerNorm, there is no mean subtraction and no bias β. eps goes inside the square root to avoid division by zero.
2. **`keepdim=True`**: after averaging over `dim=-1` (d_model), keep that dimension so the shape goes from `(B, S, d)` to `(B, S, 1)` and broadcasts correctly with x. Without it you get an error when shapes mismatch, but when `S == d_model` by coincidence there is **no error and the result is silently wrong**.
3. **Upcast to float32, then cast back**: record `in_dtype = x.dtype`, compute the square and mean in float32 to avoid overflow or precision loss in low precision (e.g. bf16), then `.to(in_dtype)` at the end.
4. **g initialized to ones**: no rescaling at the start; training learns it. Name the parameter `weight` so it matches `load_state_dict` keys later.
5. **SiLU vs ReLU**: SiLU(x) = x·σ(x) is smooth everywhere; in the negative region it gives small negative values (minimum about -0.28) with nonzero gradient. ReLU outputs 0 with 0 gradient for all negatives.
6. **Gating**: SiLU(W1x) is the gate and W3x is the content. When a gate dimension is near 0, that output dimension is near 0 regardless of the content. The gate is computed from the input, so each token decides which dimensions to open or close.
7. **Why d_ff ≈ 8/3·d_model**: a plain FFN has 2 matrices of d×4d, 8d² in total; SwiGLU has 3 matrices, so setting 3·d·d_ff = 8d² gives d_ff = 8d/3 and keeps the parameter count the same. Then round to a multiple of 64 for efficient GPU computation.
8. **Parameter-free functions live outside the module**: SiLU has no learnable parameters, so it does not go in `__init__`; it is a module-level function that `run_silu` can import directly.
9. **Submodule names `w1` / `w2` / `w3`**: when loading full-model weights later, keys look like `...ffn.w1.weight`, so the names must match.
10. **No error ≠ correct**: a missing `keepdim` or `@` with `S == d_ff` can both fail silently. That is why every module needs a test behind it.

#### Verification
- `uv run pytest -k test_rmsnorm`: PASSED.
- `uv run pytest -k "test_swiglu or test_silu"`: 2 passed.

#### Recall questions
1. **What happens if `keepdim=True` is missing?**
   - My answer: the averaged dimension gets removed.
   - Correction: right ✅. The consequence is an error when shapes mismatch, or, when `S == d_model` by coincidence, no error but broadcasting along the wrong dimension and a wrong result.
2. **How does RMSNorm differ from LayerNorm?**
   - My answer: one normalizes before entering attention, the other after coming out?
   - Correction: that describes pre-norm vs post-norm (where the norm sits), not RMSNorm vs LayerNorm. RMSNorm does not subtract the mean and has no bias β; it only scales by the RMS. LayerNorm subtracts the mean, divides by the standard deviation, and has β.
3. **What is the difference between `swiglu.w1 = w1_weight` and `swiglu.w1.weight.data = w1_weight`?**
   - My answer: one refers to the module, the other to the weight values of the matrix inside the module.
   - Correction: right ✅. The first tries to replace the whole submodule with a tensor; only the second loads the data into the submodule's parameter.
4. **What happens if `*` is replaced with `@` on `(B, S, d_ff)`?**
   - My answer: it does an inner product or matrix multiply and raises a shape error.
   - Correction: mostly right ✅. `@` compares the first operand's last dim `d_ff` with the second's second-to-last dim `S`: an error when they differ, but when `S == d_ff` by coincidence there is no error and the result is wrong.

#### Next
- §3.4.3 RoPE (rotary position embedding).