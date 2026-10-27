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

## Day — RoPE rotary position embedding (2026-10-24 ~ 2026-10-26)

#### What I did
- §3.4.3 RoPE: `cs336_basics/rope.py` (`RoPE(theta, d_k, max_seq_len, device)`) + `adapters.run_rope`.

#### Common small mistakes
1. **Subtracting 1 from the `arange` end**: `arange` already excludes the end. `arange(0, d_k-2, 2)` drops the last pair and `arange(0, max_seq_len-1)` drops the last position. Use `d_k` and `max_seq_len` as the end.
2. **Writing the outer product as `*`**: frequencies `(d_k/2,)` times positions `(max_seq_len,)` with `*` errors when lengths differ, and silently multiplies elementwise when they happen to match. Use `torch.outer(positions, freqs)`, positions first so they become rows.
3. **Extra brackets in `torch.outer([a], [b])`**: the brackets turn tensors into Python lists; `outer` needs 1-D tensors.
4. **New tensors without `dtype` / `device`**: the position vector must be float and on the same device as the frequencies.
5. **Using `arange` instead of slicing x**: `x1 = torch.arange(...)` creates position numbers unrelated to the input. What gets rotated is x itself: `x[..., 0::2]`, `x[..., 1::2]`.
6. **Using an attribute never stored in `__init__`**: `self.max_seq_len` was never assigned, which raises AttributeError.
7. **Lookup result not assigned**: `self.cos[token_positions]` on its own line is computed and thrown away.
8. **Too few arguments to forward**: `rope(token_positions)` treats the positions as x; it must be `rope(in_query_or_key, token_positions)`.

#### Things worth noting
1. **Core idea**: instead of adding a position vector, treat each adjacent pair in q and k as a point in a plane and rotate it by position. (a, b) rotated by θ becomes (a·cosθ − b·sinθ, a·sinθ + b·cosθ).
2. **Why it is relative**: q rotates by iθ and k by jθ, so the dot product depends only on the contents of q, k and the distance j−i. Positions (3, 5) and (100, 102) give the same score.
3. **Why position info is needed**: attention is a weighted sum, which ignores order; Embedding, FFN and RMSNorm all work per position. Without injected position, "dog bites man" and "man bites dog" look the same to the model.
4. **Frequencies**: group k uses 1/Θ^((2k−2)/d); the numerators are 0, 2, …, d−2, i.e. `arange(0, d_k, 2)`. For d=4, Θ=10000 this gives [1, 0.01]. Early groups rotate fast and resolve short distances; later groups rotate slowly and encode long distances.
5. **cos/sin table shape**: `(max_seq_len, d_k/2)`, one angle per position per pair. Index it directly with `token_positions` to get `(..., seq_len, d_k/2)`, the same idea as the Embedding lookup.
6. **`register_buffer(..., persistent=False)`**: a buffer is not trained but belongs to the module and moves with `.to(device)`; `persistent=False` keeps it out of `state_dict`, since it can be recomputed any time and the official weights do not contain these keys.
7. **Adjacent pairs, not two halves**: the handout pairs (x₀, x₁), (x₂, x₃), …, hence `0::2` / `1::2` slicing. Splitting by `:d/2` / `d/2:` is the GPT-NeoX pairing and will not match the tests.
8. **Interleaving back**: `torch.stack((out1, out2), dim=-1)` gives `(..., d_k/2, 2)`, then `flatten(-2)` gives `(..., d_k)`. Example: [a, c] and [b, d] → [[a, b], [c, d]] → [a, b, c, d].
9. **Pitfall for MHA**: x will gain a head dim `(batch, heads, seq, d_k)` while the lookup is `(batch, seq, d_k/2)`. Broadcasting aligns from the right, so batch would line up against heads; cos/sin will need an extra head dim.

#### Verification
- `uv run pytest -k test_rope`: PASSED.

#### Recall questions
1. **Why slice with `x[..., 0::2]` and `x[..., 1::2]` instead of `x[..., :d/2]` and `x[..., d/2:]`?**
   - My answer: because we need to take the two elements within each pair.
   - Correction: right ✅. The handout pairs adjacent elements; splitting into halves would pair (x₀, x_{d/2}), a different convention (GPT-NeoX) that will not match the tests.
2. **When `token_positions` is `(batch, seq_len)`, what is the shape of `self.cos[token_positions]`, and why can it multiply `x1` directly?**
   - My answer: `(batch, seq_len, d_k/2)`, and `x1` is also `(batch, seq_len, d_k/2)`, so the shapes match.
   - Correction: right ✅. Note that MHA adds a head dim, which will require an extra dimension for broadcasting.

#### Next
- §3.4.4 softmax (subtract the max for numerical stability), then scaled dot-product attention.
## Day — softmax, scaled dot-product attention, and multi-head self-attention (2026-10-26)

#### What I did
- §3.4.4 softmax: `softmax(x, dim)` in `cs336_basics/nn_utils.py` plus `adapters.run_softmax`.
- §3.4.4 scaled dot-product attention (SDPA): `nn_utils.scaled_dot_product_attention(Q, K, V, mask)` plus `adapters.run_scaled_dot_product_attention`.
- §3.4.5 causal multi-head self-attention (MHA): `MultiHeadSelfAttention` in `cs336_basics/attention.py` plus `adapters.run_multihead_self_attention`.
- MHA with RoPE: optional `theta` / `max_seq_len` constructor args, plus `adapters.run_multihead_self_attention_with_rope`.

#### Common small mistakes
1. **Forgot to wire the adapter again (4 times in this section)**: after every module, import it in `tests/adapters.py` and replace `raise NotImplementedError` with the implementation. Check this before saying "done", and delete any leftover `raise` after `return`.
2. **Hardcoded `dim=-1` in softmax**: the function takes a `dim` argument, so use it.
3. **Summed the wrong thing in softmax**: the denominator sums the exp values, not the raw x. `torch.exp` takes a single tensor, with no `dim` / `keepdim`.
4. **`torch.max(..., dim)` returns (values, indices)**: take `.values` or use `amax`. `torch.sum` already returns a tensor, so there is no `.values` on it.
5. **`torch.sqrt` on an int**: `Q.shape[-1]` is a Python int, not a tensor; use `math.sqrt`.
6. **Dim size vs dim index**: softmax needs `dim=-1` (an index), not `K.shape[-1]` (a size).
7. **`masked_fill` value set to `scores`**: fill with `float('-inf')`, and only when `mask is not None`.
8. **Missing `self.` on submodules**: `q_proj = LinearNoBias(...)` is just a local variable; it is not registered as a submodule and forward can't see it.
9. **`/` vs `//`**: `d_model / num_heads` gives a float; use `//`.
10. **Commas in an einops pattern**: `"... h s d -> ... s (h d)"` has no commas; commas only separate inputs in einsum.
11. **Missing `output_proj`**: after merging heads, multiply by W_O once more.
12. **Wrong weight loading in the adapter (second time)**: `LinearNoBias(...)(q_proj_weight)` calls forward and multiplies the weight by a random matrix. The right way is `mha.q_proj.weight.data = q_proj_weight`.
13. **RoPE overwritten unconditionally**: `self.rope = RoPE(...)` followed directly by `self.rope = None`; it needs if/else.
14. **Precedence of `theta and max_seq_len is not None`**: it means `theta and (max_seq_len is not None)`; write `is not None` on both sides.
15. **RoPE applied before splitting heads**: the last dim is then d_model, but the RoPE table is built for d_k. Split first, then rotate.
16. **Default positions used `x.shape[-1]`**: that is d_model; the sequence length is `x.shape[-2]`. The tensor also belongs on `x.device`; there is no variable called `device` inside `forward`.
17. **Didn't keep the result of `unsqueeze`**: it is not in place; write `token_positions = token_positions.unsqueeze(-2)`.

#### Noteworthy points
1. **Subtract the max in softmax**: e^x overflows above about x = 88 in float32. Subtracting the same constant from every element doesn't change the result, so subtract the max; `keepdim=True` keeps it broadcastable.
2. **Attention softmax runs over the key dim (last dim)**: each query's weights over all keys sum to 1.
3. **Why divide by √d_k**: if q and k components have variance 1, q·k has variance d_k. Large scores saturate softmax toward one-hot, so gradients go to about 0. Dividing by √d_k brings the variance back to 1.
4. **Mask convention**: True means attend, False means blocked. `masked_fill(~mask, -inf)` makes those positions 0 after softmax. A row that is entirely False becomes all -inf, which gives NaN.
5. **Causal mask**: `torch.tril(torch.ones(S, S, dtype=torch.bool, device=x.device))`; the lower triangle is True, so token i sees only 0..i.
6. **Transpose in einsum without `.T`**: name the dims, `"... m d, ... n d -> ... m n"` is QKᵀ. With `@`, use `transpose(-2, -1)`; `.T` would also reverse batch dims.
7. **Project once, then split into heads**: one `(d_model, d_model)` matrix is mathematically the same as stacking each head's small matrix, but needs only one matmul. This is the standard approach (GPT-2 fuses q, k, v into one `c_attn`; LLaMA / Qwen use separate `q_proj`, `k_proj`, `v_proj`; GQA, MQA, and MLA vary the number or size of k and v heads).
8. **Head-split shapes**: `(B, S, d_model)` → `"... s (h d) -> ... h s d"` → `(B, h, S, d_k)`. The head dim must come before S: SDPA does the dot product and softmax over the last two dims and treats everything before as batch. Stopping at `(B, S, h, d_k)` makes heads of the same token attend to each other; the `(S, S)` mask usually fails to broadcast and errors, but if S happens to equal h it silently computes the wrong thing.
9. **RoPE rotates only q and k, not v**: the rotations of q and k cancel inside the dot product, leaving only the relative position i−j. v is not in the dot product, so rotating it would mix absolute positions straight into the output content.
10. **RoPE uses d_k, not d_model**: pairs rotate within each head, and all heads share the same angles.
11. **Head-dim broadcasting**: for `(B, S)` positions, `unsqueeze(-2)` gives `(B, 1, S)`, the lookup gives `(B, 1, S, d_k/2)`, and the 1 broadcasts over heads. The test passes `(1, S)`, which happens to work without the extra dim, but a real `(B, S)` would line B up against h.
12. **Why `token_positions` comes from the caller**: a position is not always the token's index in the tensor. With a KV cache only 1 new token is fed in, and its position is 10, not 0; left padding and packing several documents into one sequence (each restarting at 0) are other cases.
13. **Add features without breaking the old interface**: new args default to `None`, so old calls and tests are unaffected; rerun the old test after the change.

#### Verification
- `uv run pytest -k test_softmax_matches_pytorch`: PASSED.
- `uv run pytest -k test_scaled_dot_product_attention`: 3D and 4D both PASSED.
- `uv run pytest -k test_multihead_self_attention`: `test_multihead_self_attention` and `test_multihead_self_attention_with_rope` both PASSED.

#### Recall questions
1. **Why `math.sqrt(Q.shape[-1])` and not `torch.sqrt(Q.shape[-1])` in SDPA?**
   - My answer: a shape entry is a dim size, an integer, not a tensor.
   - Correction: right ✅.
2. **If a mask row is all False, what does softmax give?**
   - My answer: all zeros.
   - Correction: wrong ❌. The whole row is -inf; subtracting the max gives -inf − (−inf) = NaN, so the row is all NaN. The causal mask's diagonal is always True, so this doesn't happen there.
3. **Why must the head split be `(..., h, S, d_k)`?**
   - My answer: SDPA only does the dot product and softmax on the last two dims and treats the rest as batch.
   - Correction: right ✅, but I missed the consequence: `(..., S, h, d_k)` makes heads attend to each other; the mask usually errors, and when S equals h it is silently wrong.
4. **What is the difference between `LinearNoBias(...)(q_proj_weight)` and `mha.q_proj.weight.data = q_proj_weight`?**
   - My answer: initialize q_proj first, then assign the test matrix into it.
   - Correction: that only describes the `.data` version ✅. The first one calls forward, multiplying the weight by a random matrix, so nothing is loaded.
5. **Why isn't RoPE applied to v?**
   - My answer: v is the score matrix; the output is each position's weighted sum over all vectors.
   - Correction: partly right ⚠️. The scores are softmax(QKᵀ/√d_k); v is the content being weighted. v isn't rotated because it doesn't enter the dot product, so its rotation can't cancel into a relative position.
6. **With a KV cache, what happens to the 11th token if `token_positions` uses the default?**
   - My answer: it recomputes self-attention over tokens 0 to 11 and grows the KV cache.
   - Correction: wrong ❌. Only 1 token is fed in, so S=1 and the default is `[0]`. It gets rotated as position 0, its relative distances to the cached keys are all wrong, and nothing errors. The caller must pass `[[10]]`.

#### Next
- §3.5 Transformer block (pre-norm: `y = x + MHA(RMSNorm(x))`, then `z = y + FFN(RMSNorm(y))`).

## Day — Transformer block and TransformerLM (2026-10-26 ~ 2026-10-27)

#### What I did
- §3.5 Transformer block: pre-norm `TransformerBlock` in `cs336_basics/transformer.py` plus `adapters.run_transformer_block`.
- §3.5 TransformerLM: `TransformerLM` in `cs336_basics/transformer_lm.py` plus `adapters.run_transformer_lm`.

#### Common small mistakes
1. **Adapter didn't load the weights (third time)**: the block adapter only had a comment, so the test ran on a randomly initialized model. Once submodules are named after the keys, a single `load_state_dict(weights)` loads everything.
2. **Class named lowercase `transformer`**: the adapter imports `TransformerBlock`, so the class name must match (and classes use CamelCase by convention).
3. **Only one block in the `ModuleList`**: `num_layers` was passed in but never used. Write `[TransformerBlock(...) for _ in range(num_layers)]`. The error was `Unexpected key(s): layers.1.*, layers.2.*`.
4. **`forward` skipped the blocks entirely**: the embedding output went straight into `ln_final`, i.e. a 0-layer model. This only showed up once all keys matched, as an output mismatch against the snapshot.
5. **The loop must write back to `x`**: `for block in self.layers: x = block(x)`, so each layer gets the previous layer's output.
6. **Forgot to pass `device` when building blocks**: CPU tests still pass; it breaks on GPU.
7. **Argument name mapping**: the LM signature's `context_length` and `rope_theta` map to the block's `max_seq_len` and `theta`.
8. **Dead `raise NotImplementedError` after `return` again**: delete it.

#### Noteworthy points
1. **Pre-norm layout**: `y = x + attn(ln1(x), pos)`, `z = y + ffn(ln2(y))`. Only sublayer inputs are normalized; the residual trunk is not.
2. **Two separate RMSNorms**: each has its own learnable gain and serves a different sublayer (attention vs FFN), so they can't be shared.
3. **What the residual does**: output = x + f(x), derivative = 1 + f'(x), so the gradient always has an identity path and doesn't vanish easily even in deep stacks.
4. **Pre-norm vs post-norm**: post-norm (original Transformer) puts the norm after the residual add, so the trunk is normalized every layer; deep models train less stably and usually need warmup. Pre-norm keeps the trunk clean and trains more stably, at the cost of an extra `ln_final`.
5. **A block keeps the shape**: input and output are both (B, S, d_model), which is why blocks stack.
6. **LM data flow**: `(B, S)` ints → embedding `(B, S, d_model)` → N blocks → `ln_final` → `lm_head` `(B, S, vocab_size)`.
7. **What `lm_head` is**: a `LinearNoBias(d_model, vocab_size)` with weight shape (vocab_size, d_model), trainable and randomly initialized. It dots each position's hidden vector h with each word's row to score that word.
8. **Output is logits, no softmax**: logits are unnormalized scores; they can be negative and don't sum to 1. Softmax lives outside the model: inside cross-entropy during training, before sampling during generation.
9. **Logits at position t predict token t+1**: the causal mask lets position t see only 0..t, so its job is to guess the next token.
10. **Weight tying**: GPT-2 shares one matrix between `lm_head` and the embedding (same shape). A1 uses two independent weights.
11. **`nn.ModuleList` vs a Python list**: ModuleList registers block i as a submodule named `i`, producing keys like `layers.0.attn.q_proj.weight`. Modules in a plain list are not registered: `load_state_dict` reports Unexpected keys, `parameters()` doesn't see them (the optimizer won't update them), and `.to(device)` won't move them.
12. **Missing vs Unexpected**: Missing means the model needs a key the dict lacks (e.g. 5 layers built, 3 in the weights); Unexpected means the dict has a key the model lacks (e.g. 1 layer built, 3 in the weights).
13. **Why the truncated-input test passes**: with the sequence halved, the first half of the output is identical to the full-input run. Three reasons: the causal mask hides later tokens; RoPE's default positions `arange(S)` share the same prefix; RMSNorm and FFN are per-token.

#### Verification
- `uv run pytest -k "test_transformer_lm or test_transformer_block"`: `test_transformer_block`, `test_transformer_lm`, and `test_transformer_lm_truncated_input` all PASSED.

#### Recall questions
1. **Why two RMSNorms in a block?**
   - My answer: the two inputs being normalized have different means, so they need to be trained separately.
   - Correction: right direction ⚠️. RMSNorm doesn't subtract the mean; more precisely, each norm has its own gain serving a different sublayer.
2. **What does the residual connection do?**
   - My answer: stabilizes backpropagation.
   - Correction: right ✅. Specifically, the derivative is 1 + f', so the gradient always has an identity path.
3. **Pre-norm vs post-norm?**
   - My answer: forgot.
   - Correction: see noteworthy point 4.
4. **Block input/output shape?**
   - My answer: both (B, S, d_model); the dimension doesn't change.
   - Correction: right ✅, which is what makes stacking possible.
5. **What happens if `self.layers` is a plain list?**
   - My answer: ModuleList registers block i as submodule i so keys like `layers.0.attn...` are recognized; PyTorch doesn't see modules in a plain list, so state_dict can't see them and loading fails.
   - Correction: right ✅. Also: the error is Unexpected key, and the optimizer and `.to(device)` don't see those blocks either.
6. **Why is the first half identical with truncated input?**
   - My answer: the causal mask means each token only sees earlier tokens, so feeding half doesn't matter; not sure about RoPE.
   - Correction: partly right ⚠️. RoPE only depends on each token's own position, and `arange(6)` is exactly the first 6 of `arange(12)`, so the rotations match.
7. **Model built with 5 layers but weights have 3: what error?**
   - My answer: missing input?
   - Correction: mostly right ⚠️. Precisely: `Missing key(s): layers.3.*, layers.4.*`.

#### Next
- §4 cross-entropy, then AdamW, cosine learning-rate schedule, and gradient clipping.
## Day — Resource accounting: transformer_accounting (2026-10-27)

#### Common mistakes
1. **Counting the embedding inside the block**: the embedding (V·d) sits outside all blocks, so there is only one copy and it must not be multiplied by L.
2. **Writing output_proj as (d_k, d)**: after the heads are concatenated the width is d, so output_proj is (d, d).
3. **Forgetting the FFN shapes**: SwiGLU has three matrices. w1 and w3 are (d_ff, d) and w2 is (d, d_ff), so each block has 3·d·d_ff parameters.
4. **Missing ln1 and ln2 inside the block**: each block has two RMSNorms, which is 2d parameters.
5. **Multiplying 2Vd by L**: the embedding and lm_head exist once. Multiplying them by L gave 9.2B instead of the correct ~1.64B.
6. **Mixing up GB and GiB**: bytes ÷ 10⁹ is GB, and bytes ÷ 2³⁰ is GiB. The earlier 34.28 was really GiB.
7. **Reversing the matmul shape order**: with batch=1 the input is (S×d), so a projection is (S×d)·(d×d). Writing (d×d)·(S×d) gives shapes that don't line up.
8. **Not splitting QKᵀ and scores×V by head**: each head is (S×d_k)·(d_k×S), and there are h of them.
9. **Missing the output_proj and FFN matmuls**: the FFN turns out to be the largest part.
10. **Counting RoPE, RMSNorm, and softmax as matmuls**: they are elementwise or row-wise ops of order S·d or h·S². The embedding lookup is not a matmul either.
11. **Multiplying all four dimensions together**: (S×d)·(d×d) was written as S·d³. The shared inner dimension n counts only once, so it is 2·S·d·d = 2Sd².
12. **Dropping the 2 in 2mnp**: the multiply and the add each count as one FLOP.
13. **Thinking the FFN-to-projection ratio is f/d ≈ 2.68**: the coefficient matters, so 6Sdf / 8Sd² = (3/4)·(f/d) ≈ 2.
14. **Predicting that all four shares shrink as the model grows**: the shares always sum to 100%, so they cannot all shrink.

#### Noteworthy points
1. **The 2mnp rule**: (m×n)·(n×p) produces m·p entries. Each entry is a length-n dot product with n multiplies and n adds, so the total is 2mnp.
2. **Parameter count**: P = L(4d² + 3df + 2d) + 2Vd + d. GPT-2 XL has 1,640,452,800 ≈ 1.64B parameters, which is 6.56 GB (6.11 GiB) in fp32. Blocks are about 90% of it, and the FFN is about 2× the attention projections.
3. **Per-block matmul list (batch=1)**:

   | Matmul | Shape | Count |
   |---|---|---|
   | Q, K, V projections | (S×d)·(d×d) | 3 |
   | QKᵀ | (S×d_k)·(d_k×S) | h |
   | scores × V | (S×S)·(S×d_k) | h |
   | output_proj | (S×d)·(d×d) | 1 |
   | FFN w1, w3 | (S×d)·(d×f) | 2 |
   | FFN w2 | (S×f)·(f×d) | 1 |

   Outside the blocks there is only lm_head: (S×d)·(d×V).
4. **Forward FLOPs formula**: FLOPs = L(8Sd² + 4S²d + 6Sdf) + 2SdV. Using h·d_k = d merges the per-head terms.
5. **A simplification trick**: dividing all three block terms by 2Sd gives 3f : 4d : 2S, which shows at a glance which term is largest.
6. **GPT-2 XL (S=1024)**: 6.98×10¹⁰ per block, 3.35×10¹² for 48 layers, 1.65×10¹¹ for lm_head, and 3.52×10¹² in total. FFN is 57.5%, projections 28.6%, attention scores 9.2%, and lm_head 4.7%.
7. **How d_ff is chosen**: the multiple of 64 nearest to 8/3·d. That is 2048 for small, 2752 for medium, 3392 for large, and 4288 for XL.
8. **Trends as the model grows**:

   | Model | Total FLOPs | FFN | Projections | Attention scores | lm_head |
   |---|---|---|---|---|---|
   | small | 2.9×10¹¹ | 39.8% | 19.9% | 13.3% | 27.1% |
   | medium | 8.3×10¹¹ | 50.1% | 24.8% | 12.4% | 12.7% |
   | large | 1.8×10¹² | 54.3% | 27.3% | 10.9% | 7.4% |
   | XL | 3.5×10¹² | 57.5% | 28.6% | 9.2% | 4.7% |

   The FFN and projections scale with d², so their shares rise, and they stay at about 2:1. Attention scores scale with S²d, and lm_head scales with dV and exists once (its ratio to the blocks is about V/(L·d)), so both shares fall. The vocabulary is expensive for small models: lm_head is 27% of GPT-2 small.
9. **Long context**: raising XL's context from 1024 to 16384 (16×) makes attention scores 256× larger while every other term grows 16×. Total FLOPs grow about 38×, to 1.34×10¹⁴, and the attention-score share jumps from 9.2% to 61.7%. This is why long context is expensive, and it is what FlashAttention and sparse attention target.

#### Recall questions
1. **How many FLOPs does (m×n)·(n×p) take, and why?**
   - Answer: 2mnp. The result has mp entries, each a length-n dot product (n multiplies and n adds).
2. **What are GPT-2 XL's parameter count and fp32 memory?**
   - Answer: about 1.64B parameters, which is 6.56 GB.
3. **Why aren't RMSNorm and softmax counted in the FLOPs?**
   - Answer: they are not matmuls, and at order S·d or h·S² they are one or two orders of magnitude smaller.
4. **Does the FFN-to-projection ratio change as the model grows?**
   - My answer: it gets smaller.
   - Correction: it doesn't change ⚠️. Plugging f ≈ (8/3)d into 3f/(4d) cancels d and leaves about 2.
5. **When the context grows 16×, how do the terms and the total change?**
   - My answer: FFN, projections, and lm_head each grow 16×, attention scores grow 256×, the total grows more than 16×, and attention becomes the largest share.
   - Correction: correct ✅. The total grows about 38×.

#### Next step
- §4.1 cross-entropy: the concept questions are answered, so next is implementing `run_cross_entropy`.