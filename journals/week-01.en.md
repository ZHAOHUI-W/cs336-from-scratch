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

#### Background / motivation
- The model only accepts integer ids, so text has to be cut into tokens first. The trade-offs:
  - Characters / bytes: tiny vocabulary (256), but sequences get very long, and attention cost grows with the square of the length.
  - Whole words: short sequences, but the vocabulary explodes, and an unseen word (OOV, out-of-vocabulary) cannot be encoded at all.
  - **BPE (Byte Pair Encoding)**: start from the 256 single bytes and repeatedly merge the most frequent adjacent pair in the corpus into a new token. Common words end up as one token, rare words split into a few pieces, and because the base unit is a byte, any string can be encoded (no OOV).
- Where it sits in the pipeline: raw text → **`train_bpe` learns vocab and merges (run once, before training)** → the Tokenizer encodes text into ids → saved as a uint16 array → `get_batch` samples batches → model.

#### Formula / rule
- Each round picks the most frequent adjacent pair:

  $$(A,B)^* = \arg\max_{(A,B)} \sum_{w} \text{freq}(w)\cdot \#_{(A,B)}(w)$$

  where $w$ is the byte sequence of one pre-token (a piece produced by pre-tokenization), $\text{freq}(w)$ is how often it occurs in the corpus, and $\#_{(A,B)}(w)$ counts how often $A,B$ appear next to each other in $w$. Ties go to the lexicographically greater pair.
- The merged token $C = A + B$ (bytes concatenation) is added to vocab and recorded in merges.
- Number of merges = `vocab_size − 256 − len(special_tokens)`. For example vocab_size=500 with 1 special: 500 − 256 − 1 = 243.

#### Inputs / outputs / shapes
```python
train_bpe(input_path: str, vocab_size: int, special_tokens: list[str])
    -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]
```
- `input_path`: corpus path; `vocab_size`: target size (including specials and the 256 single bytes); `special_tokens`: e.g. `["<|endoftext|>"]`, added to vocab but never counted.
- Returns `vocab: dict[int, bytes]` (id → byte string) and `merges: list[tuple[bytes, bytes]]` (in the order they were learned).
- Internal structures: `freq: dict[tuple[bytes, ...], int]` (word frequency table), `pair_freq: dict[tuple[bytes, bytes], int]`, `pair_to_words: dict[pair, set[word]]`.
- No tensors and no learnable parameters.

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

#### Where it lives and tests
- Files: `cs336_basics/train_bpe.py` (naive + incremental), `train_bpe_native.py`, `train_bpe_multiprocessing.py`, `train_bpe_multiprocessing_buckets.py`. `run_train_bpe` in `tests/adapters.py` now imports the bucket version, `train_bpe_multiprocessing_buckets.train_bpe`.
- Incremental skeleton:
```python
vocab = {specials...} | {bytes([b]) for b in range(256)}
freq = count each pre-token's byte tuple
pair_freq, pair_to_words = initial counts
while len(vocab) < vocab_size:
    best = max(pair_freq, key=lambda p: (pair_freq[p], p))   # highest count, ties -> lexicographically greater
    merges.append(best); vocab[len(vocab)] = best[0] + best[1]   # once per round
    for w in list(pair_to_words[best]):                     # only words that contain best
        remove w's old pair counts -> merge -> add the new word's pair counts -> swap the word in freq
```
- Passing: `test_train_bpe`, `test_train_bpe_speed`, `test_train_bpe_special_tokens`.

#### Small numeric example
Word counts (already split into bytes): `low`×5, `lower`×2, `newest`×6, `widest`×3.

| Pair | Count |
|---|---|
| (e, s) | 6 + 3 = 9 |
| (s, t) | 6 + 3 = 9 |
| (w, e) | 2 + 6 = 8 |
| (l, o) | 5 + 2 = 7 |
| (o, w) | 5 + 2 = 7 |
| (n, e), (e, w) | 6 |

- Round 1: (e, s) and (s, t) tie at 9; the lexicographically greater (s, t) wins and gives `st`. `newest` becomes n e w e **st** and `widest` becomes w i d e **st**.
- Round 2: the new pair (e, st) has count 9, the highest, and merges into `est`. That pair did not exist before round 1, which is exactly why merges must stay ordered.

#### Things worth noting
1. **Byte-level means no OOV**: every Unicode character is 1–4 UTF-8 bytes, so the worst case is byte-by-byte encoding.
2. **Split out special tokens first**: otherwise PAT chops `<|endoftext|>` into pieces, and they may merge with the bytes on either side, crossing document boundaries.
3. **PAT pre-tokenization limits where merges happen**: merges only happen inside one pre-token. For example `dog!` is first split into `dog` and `!`, so a cross-category token like `g!` never appears.
4. **Side effect**: while adding comments on 2026-11-03 I found that every train_bpe version overwrites `merges.txt` in the current directory on each run (flagged with `# 注意`).

#### Recall questions
1. **With vocab_size=500 and one special token, how many merges are there?**
   <details><summary>Answer</summary>500 − 256 − 1 = 243.</details>
2. **Which pair wins a tie, and why is a rule needed at all?**
   <details><summary>Answer</summary>The lexicographically greater one. A fixed rule makes the result reproducible and lets it match the reference merge by merge.</details>
3. **Why does updating only the words in `pair_to_words[best]` give the same result as a full recount?**
   <details><summary>Answer</summary>Words without best don't change this round, so their contribution to pair_freq stays the same. Only words containing best need their old counts removed and new counts added.</details>
4. **What happens if `vocab[len(vocab)] = C` sits inside the per-word loop?**
   <details><summary>Answer</summary>It runs once per affected word, vocab explodes, and it fills up after about 2 merges (the reference does 243).</details>

#### Next
- Remove debug `merges.txt` write if still present.
- Implement tokenizer **encode/decode** (`get_tokenizer`) and `tests/test_tokenizer.py`.

## Day — Tokenizer encode/decode (2026-09-22)

#### What I did
- Implemented `Tokenizer` in `cs336_basics/tokenizer.py`: `__init__`, `encode`, `decode`, `encode_iterable`.
- Wired `tests/adapters.py` `get_tokenizer` to `return Tokenizer(vocab, merges, special_tokens)`.
- Patched `tests/test_tokenizer.py` for Windows: optional `resource` import; skip hard `RLIMIT_AS` when unavailable.
- Fetched GPT-2 fixtures (`gpt2_vocab.json` / `gpt2_merges.txt`) for tiktoken-alignment tests.

#### Background / motivation
- `train_bpe` only learns the rules. For real training and inference, the Tokenizer has to encode any text into ids (encode) and turn ids the model generates back into text (decode).
- Requirements: encode followed by decode must give back the original text (roundtrip), and segmentation must match training exactly, or the ids mean the wrong thing.
- Where it sits: corpus → Tokenizer.encode → one long id sequence (saved as a uint16 file for `get_batch`); during generation, ids → decode → text.

#### Formula / rule
- encode: `text → split on specials → PAT pre-tokenization → UTF-8 bytes → apply merges in order → bytes_to_id lookup`.
- decode: $\text{text} = \text{utf8\_decode}\big(\text{vocab}[i_1] \,\|\, \text{vocab}[i_2] \,\|\, \cdots\big)$, where $\|$ is byte concatenation. Concatenating before decoding means a multi-byte character split across two tokens still comes back correctly.

#### Inputs / outputs / shapes
```python
Tokenizer(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None)
encode(text: str) -> list[int]
decode(ids: list[int]) -> str
encode_iterable(iterable: Iterable[str]) -> Iterator[int]
```
- Internal attributes: `bytes_to_id` (reverse of vocab), `special_pattern` (regex compiled longest-first), `pre_token_re` (GPT-2 PAT).
- No tensors and no learnable parameters.

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

#### Where it lives
- File: `cs336_basics/tokenizer.py`. `encode` walks `special_pattern.finditer`, alternating between ordinary spans and special spans; ordinary spans go to `_encode_normal_text`, which calls `_apply_single_merge` for each merge on each pre-token.
- Adapter: `get_tokenizer` in `tests/adapters.py` simply does `return Tokenizer(vocab, merges, special_tokens)`.

#### Small numeric example
- With merges = `[(s, t), (e, st)]`, encode `"west"`:
  1. Bytes: `w e s t`
  2. Apply merge 1, (s, t): `w e st`
  3. Apply merge 2, (e, st): `w est`
  4. Look up 2 ids. In the reverse order, (e, st) is not found first, then (s, t) merges, leaving `w e st`, i.e. 3 tokens, which does not match training.
- Specials: `"hi<|endoftext|>"` is split into `"hi"` and `"<|endoftext|>"`; the latter is looked up whole as 1 id and never enters BPE.
- Overlapping specials: with both `<|endoftext|>` and `<|endoftext|><|endoftext|>`, matching longest-first keeps the latter from being split in two (`test_overlapping_special_tokens`).

#### Things worth noting
1. **Special tokens must already be in vocab**, or `bytes_to_id[...]` raises KeyError (flagged with `# 注意` on 2026-11-03).
2. **No cache**: every occurrence of the same word replays all merges, which is slow on a large corpus. Common fixes are a dict cache per pre-token, or using merge ranks and merging only the lowest-rank pair each time.
3. **`encode_iterable` encodes each chunk independently**: nothing merges across chunks. Reading a file line by line ends each line with a newline, so this is usually fine.
4. The old first-line comment in tokenizer.py ("byte-only lookup") is out of date; it was flagged with `# 注意` too.

#### Acceptance
- Roundtrip and tiktoken-alignment cases in `tests/test_tokenizer.py` passed.

#### Recall questions
1. **Why does decode concatenate all the bytes before decoding instead of decoding token by token?**
   <details><summary>Answer</summary>A multi-byte character (e.g. 3 bytes for a Chinese character) can be split across two tokens, and decoding one piece alone gives garbage. Decoding the concatenation restores it.</details>
2. **Why match multiple special tokens longest-first?**
   <details><summary>Answer</summary>A shorter one would grab part of a longer one; for example `<|endoftext|><|endoftext|>` would become two `<|endoftext|>`.</details>
3. **With byte-only lookup, why does the roundtrip test pass while the tiktoken alignment test fails?**
   <details><summary>Answer</summary>Bytes alone restore the text perfectly, but without merges every word becomes letter-level tokens, so the id sequence is completely different from tiktoken's.</details>

#### Next
- Full TinyStories `train_bpe` experiment (e.g. compression), or A1 Transformer.

## Day — TinyStories `train_bpe` experiment (2026-09-22)

#### What I did
- Handout **train_bpe_tinystories**: byte-level BPE on `TinyStoriesV2-GPT4-train.txt`.
- Config: `vocab_size=10000`, `special_tokens=["<|endoftext|>"]`.
- Script `scripts/train_tinystories_bpe.py`; implementations:
  1. Incremental-merge `train_bpe`
  2. `train_bpe_multiprocessing.py` — chunked pretok via `find_chunk_boundaries` + `Pool`
  3. `train_bpe_multiprocessing_buckets.py` — `PairFrequencyBuckets` for best-pair selection
- Artifacts (not committed): `artifacts/tinystories_bpe/vocab_*.txt`, `merges_*.txt`.

#### Background / motivation
- The handout asks for one BPE run on real data: how long full training takes, how much memory it uses, and whether the learned tokens look sensible.
- Why optimize: the TinyStories training set is about 2 GB, and the naive implementation is too slow at that scale. The two bottlenecks are pre-tokenization (regex splitting) and picking the most frequent pair every round.
- The resulting vocab and merges are the tokenizer for the later TinyStories language model, and vocab_size=10000 is also why token ids fit in uint16.

#### Inputs / outputs
- Input: `TinyStoriesV2-GPT4-train.txt`, `vocab_size=10000`, `special_tokens=["<|endoftext|>"]`, `num_workers=4`.
- Output: `vocab_*.txt` and `merges_*.txt` under `artifacts/tinystories_bpe/` (not committed), plus the wall time, peak memory and longest token.
- Bucket structure `PairFrequencyBuckets`: `counts: pair → freq`, `buckets: freq → set(pairs)`, with `update(...)`, `best_pair()` and `__bool__`.

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

#### Small numeric example (buckets)
- Counts `{(s,t): 9, (e,s): 9, (l,o): 7}` give buckets `{9: {(s,t), (e,s)}, 7: {(l,o)}}`. `best_pair` takes the lexicographic max from the highest non-empty bucket, 9, which is (s, t).
- After merging `st`, (e, s) drops to 0 and leaves bucket 9; the new (e, st)=9 goes into bucket 9. Each update moves one pair from one bucket to another, with no scan over all pairs.

#### Common mistakes
1. **Reading only the parent's peak memory**: `peak_wset` excludes worker processes; sample `rss(parent) + rss(all children)` periodically.
2. **Wrong units**: labelling a raw byte count as MB.
3. **Misreading cProfile**: it only profiles the parent, so Pool wait/terminate looks huge, but that is just waiting for the workers.
4. **On Windows the worker function must be top-level** (`_count_chunk`) so the spawn start method can pickle it.

#### Recall questions
1. **Why does cProfile mislead you about the bottleneck in a multiprocess program?**
   <details><summary>Answer</summary>It only sees the parent, which spends most of its time waiting on the Pool; the pretokenize work happens in the children.</details>
2. **Is best_pair strictly O(1) with buckets?**
   <details><summary>Answer</summary>No. It is O(size of the highest bucket). Late in training many pairs share the same count (e.g. 1), so the max inside a bucket can still be expensive.</details>
3. **How do you check the bucket version is correct?**
   <details><summary>Answer</summary>Run it and the non-bucket version on the same data and check the merges files are byte-identical (`fc` shows no difference), and that `len(merges)=9743`.</details>

#### Next
- Optional: chase &lt;2 min hint; then OWT 32k BPE or Transformer.


## Day — Linear and Embedding (2026-10-12 to 2026-10-23)

#### What I did
- §3.3.2 Linear (no bias): `cs336_basics/linear.py` + `adapters.run_linear`.
- §3.3.3 Embedding: `cs336_basics/embedding.py` + `adapters.run_embedding`.
- Fixed a pytest collection error: moved the demo code in `pretokenization_example.py` under `if __name__ == "__main__":` so importing it no longer runs `open(...)`.

### Linear (full picture)

#### Background / motivation
- Almost every learnable computation in a Transformer is a linear map: the Q, K, V, O projections, the three FFN matrices, and the final lm_head. Write it once as a general module and reuse it everywhere.
- Modern LLMs (LLaMA and others) usually drop the bias, so this is `LinearNoBias`.

#### Formula
- One vector: $y = Wx$, $W \in \mathbb{R}^{d_{out}\times d_{in}}$.
- A batch of row vectors: $Y = X W^\top$, with $X$ of shape $(\dots, d_{in})$.
- Init: $W_{ij} \sim \mathcal{N}(0, \sigma^2)$, $\sigma^2 = \frac{2}{d_{in}+d_{out}}$, truncated to $[-3\sigma, 3\sigma]$.

#### Inputs / outputs / shapes
- `LinearNoBias(d_in, d_out, device=None, dtype=None)`
- Learnable parameter: `weight`, shape `(d_out, d_in)`, initialized with `nn.init.trunc_normal_`.
- `forward(x)`: `(..., d_in)` → `(..., d_out)`, with any number of leading batch dims.

#### Implementation steps
```python
self.weight = nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype))
std = math.sqrt(2 / (d_in + d_out))
nn.init.trunc_normal_(self.weight, std=std, a=-3*std, b=3*std)
# forward
return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")
```
- File `cs336_basics/linear.py`; the adapter `run_linear` builds the module and sets `linear.weight.data = weights`; `test_linear` PASSED.

#### Small numeric example
- $d_{in}=2$, $d_{out}=3$, $W = \begin{bmatrix}1&2\\0&1\\1&0\end{bmatrix}$, $x=[3,4]$ → $y = [1\cdot3+2\cdot4,\ 4,\ 3] = [11, 4, 3]$.
- Init: with $d_{in}=d_{out}=512$, $\sigma=\sqrt{2/1024}\approx0.0442$ and the truncation range is about ±0.133.

### Embedding (full picture)

#### Background / motivation
- A token id is just a category number; its size means nothing (id 5 is not "bigger" than id 2), so it can't be fed to the network as a number. Each id gets its own learnable d_model-dim vector.
- It is the model's first layer: `(B, S)` integers → `(B, S, d_model)` vectors.

#### Formula
- Table $E \in \mathbb{R}^{V\times d}$, $\text{emb}(i) = E[i]$, i.e. row $i$.
- Equivalent to $\text{onehot}(i)\cdot E$, but a direct lookup needs no multiplication and is much faster.
- Init: $\mathcal{N}(0, 1)$ truncated to $[-3, 3]$.

#### Inputs / outputs / shapes
- `EmbeddingLayer(num_embeddings, embedding_dim, device=None, dtype=None)`
- Learnable parameter: `weight`, shape `(vocab_size, d_model)`.
- `forward(token_ids)`: long `(B, S)` → `(B, S, d_model)`.

#### Implementation steps
- `return self.weight[token_ids]`, using advanced indexing to fetch all rows at once.
- File `cs336_basics/embedding.py`; adapter `run_embedding`; `test_embedding` PASSED.

#### Small numeric example
- V=4, d=2, $E = [[0,0],[1,2],[3,4],[5,6]]$, ids = `[[2, 1, 2]]` with shape (1, 3).
- Output `[[[3,4],[1,2],[3,4]]]` with shape (1, 3, 2). Positions 0 and 2 both hold token 2 and get the identical vector; the Embedding itself knows nothing about position.

### Shared: confusing concepts, verification and recall

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
3. **Why is the Linear weight stored as (d_out, d_in), with forward x Wᵀ?**
   <details><summary>Answer</summary>It matches PyTorch's `nn.Linear` convention and the shape of the test weights. Each row holds one output neuron's weights, and x Wᵀ dots x with every row.</details>
4. **Do `onehot(i) @ E` and `E[i]` give the same result? Why use the latter?**
   <details><summary>Answer</summary>Yes. The latter needs no V-dim one-hot vector and no V×d multiply; it just fetches a row.</details>
5. **Embedding input ids of shape (2, 5), d_model=8: what is the output shape?**
   <details><summary>Answer</summary>(2, 5, 8): the input shape plus a trailing d_model dim.</details>

## Day — RMSNorm and SwiGLU (2026-10-23 ~ 2026-10-24)

#### What I did
- §3.4.1 RMSNorm: `cs336_basics/rmsnorm.py` + `adapters.run_rmsnorm`.
- §3.4.2 SwiGLU feed-forward: `cs336_basics/swiglu.py` (module-level `silu` function + `SwiGlu` module) + `adapters.run_silu` / `adapters.run_swiglu`.
- Small cleanup: removed unused `math` / `einsum` imports from `embedding.py`.

### RMSNorm (full picture)

#### Background / motivation
- In deep networks the scale of each layer's output drifts: too large blows up, too small vanishes, and training becomes unstable. Normalization pulls each token's vector back to a common scale.
- LLaMA and others replace LayerNorm with RMSNorm (Root Mean Square Normalization): no mean subtraction and no bias, cheaper to compute, and about as good.
- Where it sits: each block has ln1 (before attention) and ln2 (before the FFN), plus one ln_final at the end of the model (pre-norm layout).

#### Formula
$$\text{RMS}(a) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} a_i^2 + \varepsilon},\qquad \text{RMSNorm}(a)_i = \frac{a_i}{\text{RMS}(a)}\, g_i$$
- $a$: one token's d_model-dim vector; $g$: learnable per-dim gain; $\varepsilon = 10^{-5}$ prevents division by zero.

#### Inputs / outputs / shapes
- `RMSNorm(d_model, eps=1e-5, device=None, dtype=None)`
- Learnable parameter: `weight` (the g), shape `(d_model,)`, initialized to ones.
- `forward(x)`: `(B, S, d_model)` → same shape and dtype.

#### Implementation steps
```python
in_dtype = x.dtype
x = x.to(torch.float32)
rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)   # (B, S, 1)
return (x * self.weight / rms).to(in_dtype)
```
- File `cs336_basics/rmsnorm.py`; adapter `run_rmsnorm`; `test_rmsnorm` PASSED.

#### Small numeric example
- $a=[3,4]$, $g=[1,1]$: mean of squares = (9+16)/2 = 12.5, RMS ≈ 3.536, output ≈ [0.849, 1.131]. The output's mean square is exactly 1.
- With $g=[2,1]$ the output is ≈ [1.697, 1.131]: g lets the model decide how much to scale each dim.

### SwiGLU and SiLU (full picture)

#### Background / motivation
- Attention moves information between tokens; the FFN (Feed-Forward Network) processes each token on its own, independently per position. Most of the model's parameters live in the FFN.
- SwiGLU (Swish-Gated Linear Unit) is the gated FFN used by LLaMA and others, and empirically works better than a ReLU FFN.

#### Formula
$$\text{SiLU}(x) = x\cdot\sigma(x) = \frac{x}{1+e^{-x}}$$
$$\text{FFN}(x) = W_2\big(\text{SiLU}(W_1x)\odot W_3x\big)$$
- $W_1, W_3 \in \mathbb{R}^{d_{ff}\times d}$, $W_2 \in \mathbb{R}^{d\times d_{ff}}$, $\odot$ is elementwise multiplication, $d_{ff}\approx\frac{8}{3}d$.

#### Inputs / outputs / shapes
- `silu(x)`: any shape, elementwise, no parameters.
- `SwiGlu(d_model, d_ff, device=None, dtype=None)` with submodules `w1`, `w3` (weight `(d_ff, d_model)`) and `w2` (`(d_model, d_ff)`), all `LinearNoBias` with truncated-normal init.
- `forward(x)`: `(B, S, d)` → w1 and w3 give `(B, S, d_ff)` → elementwise product → w2 gives `(B, S, d)`.

#### Implementation steps
- `return self.w2(silu(self.w1(x)) * self.w3(x))`
- File `cs336_basics/swiglu.py`; adapters `run_silu` (calls `silu` directly) and `run_swiglu` (`swiglu.w1.weight.data = w1_weight` and so on); `test_silu_matches_pytorch` and `test_swiglu` PASSED.

#### Small numeric example
- SiLU(2) = 2 × 0.881 = 1.762; SiLU(0) = 0; SiLU(−1) = −1 × 0.269 = −0.269; the minimum is about SiLU(−1.28) ≈ −0.278.
- Gating: d_ff=2, $W_1x=[2,-5]$, $W_3x=[3,3]$. SiLU gives [1.762, −0.033]; times [3, 3] elementwise gives [5.29, −0.10]. The second gate is almost closed, so the content 3 barely gets through.
- d_model=512: 8/3 × 512 ≈ 1365, and the nearest multiple of 64 is 1344, the d_ff in the TinyStories config.

### Shared: mistakes, notes, verification and recall

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
5. **What does RMSNorm output for a=[3,4], g=[1,1]?**
   <details><summary>Answer</summary>RMS = √12.5 ≈ 3.536, output ≈ [0.849, 1.131].</details>
6. **In SwiGLU, which branch is the gate and which is the content? What does a closed gate mean?**
   <details><summary>Answer</summary>SiLU(W1x) is the gate and W3x is the content. When a gate dim is near 0, that product is near 0 whatever the content is.</details>
7. **Why does RMSNorm upcast to float32?**
   <details><summary>Answer</summary>Squares overflow or lose precision in bf16/fp16; compute in float32, then `.to(in_dtype)` back.</details>

#### Next
- §3.4.3 RoPE (rotary position embedding).

## Day — RoPE rotary position embedding (2026-10-24 ~ 2026-10-26)

#### What I did
- §3.4.3 RoPE: `cs336_basics/rope.py` (`RoPE(theta, d_k, max_seq_len, device)`) + `adapters.run_rope`.

#### Background / motivation
- Attention by itself ignores order (see noteworthy point 3 below), so position information has to be injected.
- The original Transformer added a position vector to the embedding (absolute position). RoPE (Rotary Position Embedding) instead rotates q and k by position; the rotation angles subtract inside the dot product, so the attention score depends only on relative distance. LLaMA, Qwen and others use it.
- Where it sits: inside each attention, after splitting heads and before the dot product, q and k are each rotated once.

#### Formula
- Group the d_k-dim vector into pairs $(x_0,x_1), (x_2,x_3), \dots$, $d_k/2$ pairs in total. At position $i$, pair $k$ ($k=1,\dots,d_k/2$) uses the angle

  $$\theta_{i,k} = \frac{i}{\Theta^{(2k-2)/d_k}}$$

- Each pair gets a 2×2 rotation:

  $$\begin{bmatrix}x'_{2k-2}\\x'_{2k-1}\end{bmatrix} = \begin{bmatrix}\cos\theta_{i,k} & -\sin\theta_{i,k}\\ \sin\theta_{i,k} & \cos\theta_{i,k}\end{bmatrix}\begin{bmatrix}x_{2k-2}\\x_{2k-1}\end{bmatrix}$$

- Key property: $\langle R_i q, R_j k\rangle = \langle q, R_{j-i}k\rangle$, so the score depends only on $j-i$. $\Theta$ is the base (10000) and $i$ is the token position.

#### Inputs / outputs / shapes
- `RoPE(theta, d_k, max_seq_len, device=None)`, **no learnable parameters**.
- Buffers: `cos`, `sin`, shape `(max_seq_len, d_k/2)`, `persistent=False`.
- `forward(x, token_positions)`: x is `(..., S, d_k)`, token_positions broadcasts to `(..., S)`, output `(..., S, d_k)`.

#### Implementation steps
```python
freq = 1 / theta ** (torch.arange(0, d_k, 2) / d_k)          # (d_k/2,)
angle = torch.outer(torch.arange(max_seq_len), freq)         # (max_seq_len, d_k/2)
register_buffer("cos", angle.cos()); register_buffer("sin", angle.sin())
# forward
cos, sin = self.cos[token_positions], self.sin[token_positions]
x1, x2 = x[..., 0::2], x[..., 1::2]
out = torch.stack((x1*cos - x2*sin, x1*sin + x2*cos), dim=-1).flatten(-2)
```
- File `cs336_basics/rope.py`; adapter `run_rope`; `test_rope` PASSED.

#### Small numeric example
- d_k=2 (a single pair, frequency 1), q=[1,0] at position 1, k=[1,0] at position 3: after rotation q=[cos1, sin1]=[0.540, 0.841] and k=[cos3, sin3]=[−0.990, 0.141]; dot = −0.535 + 0.119 = −0.416 = cos 2.
- At positions 0 and 2 instead: q=[1,0], k=[cos2, sin2], and the dot is still cos 2 = −0.416. Both have distance 2, so the scores match.
- With d_k=4 and Θ=10000 the frequencies are [1, 0.01]; at position 100 the two angles are 100 rad and 1 rad.

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
3. **Is the score the same for q at position 1 and k at 3 as for q at 0 and k at 2?**
   <details><summary>Answer</summary>Yes (for the same q and k content). The angles subtract inside the dot product, leaving only the distance 2.</details>
4. **Does RoPE have learnable parameters? Why aren't the cos/sin tables in the state_dict?**
   <details><summary>Answer</summary>No. The tables can always be recomputed from Θ, d_k and max_seq_len, so they are `persistent=False` buffers, and the official weights don't contain these keys.</details>
5. **Why is the frequency table `arange(0, d_k, 2)` and not `arange(0, d_k-2, 2)`?**
   <details><summary>Answer</summary>arange already excludes the end. The numerators must be 0, 2, …, d_k−2, d_k/2 values, so the end is d_k.</details>

#### Next
- §3.4.4 softmax (subtract the max for numerical stability), then scaled dot-product attention.
## Day — softmax, scaled dot-product attention, and multi-head self-attention (2026-10-26)

#### What I did
- §3.4.4 softmax: `softmax(x, dim)` in `cs336_basics/nn_utils.py` plus `adapters.run_softmax`.
- §3.4.4 scaled dot-product attention (SDPA): `nn_utils.scaled_dot_product_attention(Q, K, V, mask)` plus `adapters.run_scaled_dot_product_attention`.
- §3.4.5 causal multi-head self-attention (MHA): `MultiHeadSelfAttention` in `cs336_basics/attention.py` plus `adapters.run_multihead_self_attention`.
- MHA with RoPE: optional `theta` / `max_seq_len` constructor args, plus `adapters.run_multihead_self_attention_with_rope`.

### softmax (full picture)

#### Background / motivation
- Turns arbitrary real scores into a probability distribution: every entry is non-negative and they sum to 1. Attention weights and the model's next-token distribution both need it.
- Computing $e^x$ directly overflows for large x, so a numerically stable version is needed.

#### Formula
$$\text{softmax}(x)_i = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}},\qquad m = \max_j x_j$$
- Multiplying numerator and denominator by $e^{-m}$ changes nothing, but the largest term becomes $e^0=1$, so nothing overflows.

#### Inputs / outputs / shapes
- `softmax(x, dim)`: any shape, normalized along `dim`, output has the same shape. No parameters.

#### Implementation steps
- `max(dim, keepdim=True).values` → subtract → `exp` → `sum(dim, keepdim=True)` → divide.
- File `cs336_basics/nn_utils.py`; adapter `run_softmax`; `test_softmax_matches_pytorch` PASSED.

#### Small numeric example
- x = [1, 2, 3]: subtracting the max gives [−2, −1, 0], exp gives [0.135, 0.368, 1], the sum is 1.503, and the result is ≈ [0.090, 0.245, 0.665].
- x = [1000, 1001, 1002] gives exactly the same. Without the subtraction, exp(1000) = inf and inf/inf = NaN.

### Scaled dot-product attention, SDPA (full picture)

#### Background / motivation
- Each position compares its query with every position's key, then takes a similarity-weighted average of the values. This is the only way tokens exchange information.

#### Formula
$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$$
- $Q$: queries, $K$: keys, $V$: values; $d_k$: the q/k dimension; $M$: 0 where the mask is True and $-\infty$ where it is False.

#### Inputs / outputs / shapes
- `scaled_dot_product_attention(Q, K, V, mask=None)`
- Q `(..., m, d_k)`, K `(..., n, d_k)`, V `(..., n, d_v)`, boolean mask broadcastable to `(..., m, n)`.
- Intermediate scores `(..., m, n)`, output `(..., m, d_v)`. No parameters.

#### Implementation steps
```python
scores = einsum(Q, K, "... m d, ... n d -> ... m n") / math.sqrt(Q.shape[-1])
if mask is not None:
    scores = scores.masked_fill(~mask, float('-inf'))
return einsum(softmax(scores, dim=-1), V, "... m n, ... n d_v -> ... m d_v")
```
- Adapter `run_scaled_dot_product_attention`; `test_scaled_dot_product_attention` and `test_4d_scaled_dot_product_attention` PASSED.

#### Small numeric example
- One query $q=[1,0]$, two keys $k_1=[1,0]$ and $k_2=[0,1]$, values 10 and 20, $d_k=2$.
- Scores = [1, 0]/√2 = [0.707, 0], softmax ≈ [0.670, 0.330], output = 10×0.670 + 20×0.330 ≈ 13.3.
- With mask = [True, False] the second key is blocked and the output is exactly 10.

### Causal multi-head self-attention, MHA (with RoPE) (full picture)

#### Background / motivation
- One head can only learn one "where to look" pattern. Several heads run in parallel, each attending in its own d_k-dim subspace, and are then concatenated and mixed by W_O.
- Causal: a language model predicts the next word, so position i must not peek at tokens after i.

#### Formula
$$\text{MHA}(x) = W_O\,\text{Concat}(\text{head}_1,\dots,\text{head}_h),\quad \text{head}_j = \text{Attention}(Q_j, K_j, V_j, \text{mask}=\text{causal})$$
- $Q = xW_Q^\top$, $K = xW_K^\top$, $V = xW_V^\top$, each split into h parts, $d_k = d_{model}/h$. With RoPE, $Q_j$ and $K_j$ are rotated by position first.

#### Inputs / outputs / shapes
- `MultiHeadSelfAttention(d_model, num_heads, theta=None, max_seq_len=None, device=None, dtype=None)`
- Learnable parameters: `q_proj`, `k_proj`, `v_proj`, `output_proj`, each weight `(d_model, d_model)` (LinearNoBias truncated-normal init), 4·d_model² in total. When theta and max_seq_len are given, a parameter-free `RoPE(theta, d_k, max_seq_len)` is built too.
- `forward(x, token_positions=None)`: x `(B, S, d_model)` → `(B, S, d_model)`. Intermediates: `(B, h, S, d_k)` after the split, scores `(B, h, S, S)`.

#### Implementation steps
```python
q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)                     # (B, S, d)
q, k, v = [rearrange(t, "... s (h d) -> ... h s d", h=self.num_heads) for t in (q, k, v)]
mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=x.device))
if self.rope is not None:
    if token_positions is None: token_positions = torch.arange(S, device=x.device)
    token_positions = token_positions.unsqueeze(-2)                           # room for the head dim
    q, k = self.rope(q, token_positions), self.rope(k, token_positions)
out = scaled_dot_product_attention(q, k, v, mask)                             # (B, h, S, d_k)
return self.output_proj(rearrange(out, "... h s d -> ... s (h d)"))
```
- File `cs336_basics/attention.py`; adapters `run_multihead_self_attention` and `run_multihead_self_attention_with_rope` (`mha.q_proj.weight.data = ...`); both tests PASSED.

#### Small numeric example
- d_model=8, h=2, so d_k=4. x (1, 3, 8) → q (1, 3, 8) → split (1, 2, 3, 4) → scores (1, 2, 3, 3) → output (1, 2, 3, 4) → merged (1, 3, 8) → still (1, 3, 8) after output_proj. Parameters: 4 × 8² = 256.
- Causal mask for S=3:

  ```
  [[T, F, F],
   [T, T, F],
   [T, T, T]]
  ```

  Token 0 sees only itself; token 2 sees 0, 1 and 2.

### Shared: mistakes, notes, verification and recall

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
7. **What is softmax of x = [1000, 1001, 1002]? What happens without subtracting the max?**
   <details><summary>Answer</summary>The same as [1, 2, 3], about [0.090, 0.245, 0.665]. Without it, exp(1000)=inf and the result is NaN.</details>
8. **With d_model=8, h=2, S=3, what shape are the scores, and how many parameters does MHA have?**
   <details><summary>Answer</summary>(B, 2, 3, 3); 4 × 8² = 256.</details>
9. **Why divide by √d_k?**
   <details><summary>Answer</summary>With unit-variance q and k components, q·k has variance d_k. Large scores saturate softmax toward one-hot and the gradient goes to about 0; dividing by √d_k brings the variance back to 1.</details>

#### Next
- §3.5 Transformer block (pre-norm: `y = x + MHA(RMSNorm(x))`, then `z = y + FFN(RMSNorm(y))`).

## Day — Transformer block and TransformerLM (2026-10-26 ~ 2026-10-27)

#### What I did
- §3.5 Transformer block: pre-norm `TransformerBlock` in `cs336_basics/transformer.py` plus `adapters.run_transformer_block`.
- §3.5 TransformerLM: `TransformerLM` in `cs336_basics/transformer_lm.py` plus `adapters.run_transformer_lm`.

### Transformer block (full picture)

#### Background / motivation
- Combines "tokens talk to each other" (attention) and "each token processes itself" (FFN), plus norms and residuals, into one unit that can be stacked. The whole model is N blocks stacked up.

#### Formula (pre-norm)
$$y = x + \text{MHA}(\text{RMSNorm}_1(x)),\qquad z = y + \text{FFN}(\text{RMSNorm}_2(y))$$

#### Inputs / outputs / shapes
- `TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta, device=None, dtype=None)`
- Submodules and parameters: `ln1.weight` (d,), `attn.{q,k,v,output}_proj.weight` each (d, d), `ln2.weight` (d,), `ffn.w1.weight` and `ffn.w3.weight` (d_ff, d), `ffn.w2.weight` (d, d_ff). Each block has $4d^2 + 3d\,d_{ff} + 2d$ parameters.
- `forward(x, token_positions=None)`: `(B, S, d)` → `(B, S, d)`.

#### Implementation steps
```python
y = x + self.attn(self.ln1(x), token_positions)
return y + self.ffn(self.ln2(y))
```
- File `cs336_basics/transformer.py`; adapter `run_transformer_block` loads everything with one `block.load_state_dict(weights)`; `test_transformer_block` PASSED.

#### Small numeric example
- d=512, d_ff=1344: attention 4×512² = 1,048,576, FFN 3×512×1344 = 2,064,384, two norms 2×512 = 1,024, total 3,113,984 ≈ 3.11M, with the FFN about 2/3.

### TransformerLM (full picture)

#### Background / motivation
- The full language model: a sequence of token ids in, a score (logit) for the next token at every position out. Training compares the logits with y via cross-entropy; generation applies softmax and samples.

#### Formula
$$h_0 = E[\text{ids}],\quad h_l = \text{Block}_l(h_{l-1})\ (l=1..L),\quad \text{logits} = \text{RMSNorm}(h_L)\,W_{\text{lm}}^\top$$
- The probability $p(\text{next token}\mid\text{prefix}) = \text{softmax}(\text{logits})$ is computed outside the model.

#### Inputs / outputs / shapes
- `TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device=None, dtype=None)`
- Parameters: `token_embeddings.weight` (V, d), `layers.{i}.*` (L blocks), `ln_final.weight` (d,), `lm_head.weight` (V, d). Total $P = L(4d^2+3d\,d_{ff}+2d) + 2Vd + d$.
- `forward(in_indices)`: long `(B, S)` → logits `(B, S, V)`.

#### Implementation steps
```python
x = self.token_embeddings(in_indices)       # (B, S, d)
for block in self.layers:
    x = block(x)                             # write back into x
return self.lm_head(self.ln_final(x))       # (B, S, V)
```
- File `cs336_basics/transformer_lm.py`; adapter `run_transformer_lm` (`load_state_dict`); `test_transformer_lm` and `test_transformer_lm_truncated_input` PASSED.
- Flagged with `# 注意` on 2026-11-03: `forward` does not pass `token_positions` to the blocks, so RoPE always uses the default 0..S−1. Fine for training, but it must change for KV-cache inference or packing.

#### Small numeric example
- TinyStories config: V=10000, context_length=256, d=512, L=4, h=16, d_ff=1344.
- 4 blocks: 4 × 3,113,984 = 12,455,936; embedding and lm_head: 2 × 10000 × 512 = 10,240,000; ln_final: 512. Total 22,696,448 ≈ 22.7M.
- The handout's "about 17M non-embedding parameters" matches 12.46M in blocks plus 5.12M in lm_head ≈ 17.6M.

### Shared: mistakes, notes, verification and recall

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
8. **How many parameters does the TinyStories config (V=10000, d=512, L=4, d_ff=1344) have?**
   <details><summary>Answer</summary>About 22.7M: 12.46M in blocks + 10.24M for embedding and lm_head + 512 for ln_final.</details>
9. **What is the hidden risk in TransformerLM's forward right now?**
   <details><summary>Answer</summary>It doesn't pass token_positions to the blocks, so RoPE always uses 0..S−1. With a KV cache or multi-document packing the positions are wrong, silently.</details>

#### Next
- §4 cross-entropy, then AdamW, cosine learning-rate schedule, and gradient clipping.
## Day — Resource accounting: transformer_accounting (2026-10-27)

#### Background / motivation
- Before training, estimate on paper: how many parameters the model has (memory), how many FLOPs (floating-point operations) a forward pass takes (training time), and which part is most expensive (where to optimize).
- This is a written handout problem with no code or test.

#### Symbols
- V: vocabulary size; S: sequence length (context_length); L: number of layers; d: d_model; h: number of heads; $d_k = d/h$; f: d_ff.
- GPT-2 XL config: V=50257, S=1024, L=48, d=1600, h=25, f=4288 (the multiple of 64 nearest to 8/3×1600).

#### Formulas
- A matmul $(m\times n)\cdot(n\times p)$ costs $2mnp$ FLOPs.
- Parameters $P = L(4d^2 + 3df + 2d) + 2Vd + d$; forward FLOPs $= L(8Sd^2 + 4S^2d + 6Sdf) + 2SdV$ (derivation and numbers under the noteworthy points).

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

#### Small numeric example
- (2×3)·(3×4): the result has 2×4 = 8 entries, each a length-3 dot product with 3 multiplies and 3 adds, so 2×2×3×4 = 48 FLOPs.
- One Q projection with S=1024, d=1600: 2 × 1024 × 1600 × 1600 ≈ 5.24×10⁹ FLOPs; a block has four (Q, K, V, O), about 2.1×10¹⁰ in total, which is exactly 8Sd².

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
6. **How many parameters does one block have, in terms of d and f?**
   <details><summary>Answer</summary>4d² (four projections) + 3df (three SwiGLU matrices) + 2d (two RMSNorms).</details>
7. **Why is lm_head such a large share in small models?**
   <details><summary>Answer</summary>It scales with dV and exists once; its ratio to the blocks is about V/(L·d). Small models have small L and d, so lm_head is 27% of GPT-2 small.</details>

#### Next step
- §4.1 cross-entropy: the concept questions are answered, so next is implementing `run_cross_entropy`.

## Day — Training basics: cross-entropy, SGD, AdamW (2026-10-28 to 10-30)

### Cross-entropy

#### Background / motivation
- The model outputs logits at every position, and training needs a single number for "how good is the prediction" to minimize. Cross-entropy is the average negative log-probability of the correct token: the more probability the model gives the right answer, the smaller the loss.
- For evaluation, exponentiating it gives perplexity.
- Where it sits: the model's logits `(B, S, V)` and the labels y `(B, S)` (from `get_batch`) go into the loss, followed by `backward()`.

#### Formula
$$\ell_i = -\log \text{softmax}(o_i)[y_i] = \text{logsumexp}(o_i) - o_{i,y_i},\qquad \text{logsumexp}(o) = m + \log\sum_j e^{o_j - m}$$
$$\text{loss} = \frac{1}{N}\sum_{i=1}^{N}\ell_i,\qquad \text{PPL} = e^{\text{loss}}$$
- $o_i$: the V-dim logits at position i, $y_i$: the correct token id, $m = \max_j o_{i,j}$, N = B×S positions.

#### Inputs / outputs / shapes
- `cross_entropy(inputs, targets)`: logits `(..., V)` and integer targets `(...)`, scalar output. No parameters.

#### Implementation steps
```python
targets = targets.reshape(-1)                                  # (N,)
inputs = inputs.reshape(-1, inputs.shape[-1])                  # (N, V)
m = torch.max(inputs, dim=-1, keepdim=True).values             # (N, 1)
lse = m.squeeze(-1) + torch.log(torch.sum(torch.exp(inputs - m), dim=-1))   # (N,)
correct = inputs[torch.arange(N), targets]                     # (N,)
return (lse - correct).mean()
```
- File `cs336_basics/cross_entropy.py`; adapter `run_cross_entropy`; `test_cross_entropy` PASSED.

#### Small numeric example
- o = [2, 1, 0], y = 0: logsumexp = 2 + log(1 + e⁻¹ + e⁻²) = 2 + log 1.503 = 2.408, loss = 2.408 − 2 = 0.408, i.e. p_y = e^−0.408 ≈ 0.665.
- With y = 2: loss = 2.408 − 0 = 2.408, p_y ≈ 0.090.
- Two positions with losses 0.408 and 2.408 average to 1.408, so PPL = e^1.408 ≈ 4.09.

#### Common mistakes
1. **A leftover `self` parameter on a plain function**: `cross_entropy` is not a method and takes no `self`.
2. **Missing the sum in logsumexp**: it is log(Σ exp(x − m)), so sum first, then take the log.
3. **m not squeezed**: `max(..., keepdim=True)` gives (N,1), and adding it to (N,) broadcasts to (N,N).
4. **Calling `inputs.shape(-2)`**: shape is indexed, as in `shape[-2]`.
5. **reshape results not assigned**: `inputs.reshape(...)` is not in place, so an unassigned call does nothing.
6. **`reshape(-1,)` flattened the logits to 1D**: it should be `reshape(-1, inputs.shape[-1])`, which keeps the vocab dimension V.
7. **The adapter was not wired**, and a dead `raise NotImplementedError` was left after the `return`.

#### Noteworthy
1. **Per-position loss = −x[y] + logsumexp(x)**: the target is one-hot, so the cross-entropy sum reduces to −log p_y.
2. **Two numerical-stability tricks**: subtracting the max m prevents exp **overflow**. Using logsumexp directly lets log and exp cancel, which avoids softmax underflowing to 0 and then log 0. Computing a stable softmax first and then taking the log still fails on 1000× logits.
3. **Picking the correct logit**: `inputs[torch.arange(N), targets]` takes column targets[i] from row i.
4. **The mean is over all B×S positions**: flatten logits to (−1, V) and targets to (−1) first.
5. **Perplexity = exp(mean cross-entropy)**: average first, then exponentiate, never the other way round. Read it as how many words the model is hesitating among on average.
6. **Initial-loss sanity check**: uniform guessing gives loss = log V and perplexity = V. That is about 10.8 for V=50257 and about 9.2 for V=10000. If the first training loss is far from this, the init or the loss is probably wrong.

#### Recall questions
1. **What is the cross-entropy for o=[2,1,0], y=0?**
   <details><summary>Answer</summary>logsumexp ≈ 2.408, loss ≈ 0.408, p_y ≈ 0.665.</details>
2. **Why reshape to (N, V) and (N,) first?**
   <details><summary>Answer</summary>The loss averages over all B×S positions; after flattening, `inputs[arange(N), targets]` picks each row's correct logit in one go.</details>
3. **m uses keepdim=True; why squeeze it before adding it to the log term?**
   <details><summary>Answer</summary>(N,1) plus (N,) broadcasts to (N,N). Squeeze to (N,) first.</details>

### SGD and learning_rate_tuning

#### Background / motivation
- With a loss and gradients, a rule is still needed to update the parameters: the optimizer. SGD (Stochastic Gradient Descent) is the simplest: take a small step along the negative gradient.
- The handout uses SGD with a 1/√(t+1) decay to introduce how a `torch.optim.Optimizer` is written (param_groups, state, step), as groundwork for AdamW; the learning_rate_tuning experiment shows that too large an lr diverges.

#### Formula
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{t+1}}\,\nabla L(\theta_t)$$
- $\theta$: parameters, $\alpha$: initial learning rate, $t$: how many times this parameter has been updated (from 0), $\nabla L$: the gradient, same shape as θ.

#### Inputs / outputs / shapes
- `SGD(params, lr=1e-3)`; `step(closure=None)` returns the closure's loss or None.
- State: `self.state[p]["t"]`, an integer. No extra tensor state.

#### Implementation steps
```python
for group in self.param_groups:
    lr = group["lr"]
    for p in group["params"]:
        if p.grad is None: continue
        t = self.state[p].get("t", 0)
        p.data -= lr / math.sqrt(t + 1) * p.grad.data
        self.state[p]["t"] = t + 1
```
- File `cs336_basics/sgd.py`, with the experiment under `if __name__ == "__main__":` (`uv run python -m cs336_basics.sgd`). There is no pytest for this; read the printed losses.

#### Small numeric example
- loss = mean(w²) with w of size 10×10, so ∂loss/∂w = 2w/100 and each step multiplies w by $1 - \frac{2\alpha}{100\sqrt{t+1}}$.
- α=100, t=0: factor = 1 − 2 = −1, so w flips sign with the same size and the loss is unchanged; at t=1 the factor is 1 − 2/√2 ≈ −0.414 and it starts shrinking.
- α=1000, t=0: factor = −19, the loss is multiplied by 361, and it diverges from there.

#### Common mistakes
1. **`super().__init__(self, defaults)`**: the first argument must be `params`, not `self`.
2. **No `backward()` or `step()` in the experiment loop**: only the loss was computed, so the parameters never updated.
3. **lr hard-coded to 1**: the loop variable `lr_rate` was never passed to the optimizer.
4. **Weights and optimizer not rebuilt for each lr**: each run continued from the previous one, so the runs weren't comparable.
5. **f-string syntax error**: `f"lr={lr_rate}: "loss...` was missing either `{}` or a comma.
6. **Lowercase `callable` in the type hint**: it should be `Callable`.

#### Noteworthy
1. **∇ is the gradient**: the partial derivative of the loss for every parameter element, with the same shape as the parameter (`p.grad` has the shape of `p`).
2. **Each of three things handles one job**: `param_groups` holds hyperparameters, `p.grad` holds gradients, and `state[p]` holds history across steps.
3. **param_groups is a list of dicts**: different parameters can get different lr or weight_decay, and anything a group leaves out is filled from `defaults`.
4. **`p.data -= ...` updates in place**: `p = p - ...` only rebinds a local name to a new tensor, so the model parameter doesn't change.
5. **closure and `return loss`**: they exist for optimizers like L-BFGS that evaluate the loss several times per step. SGD and AdamW don't use them and return None.
6. **`backward()` accumulates gradients**: that is why `zero_grad()` comes before each step.
7. **lr experiment**: each step multiplies w by 1 − 2lr/(100√(t+1)), so the loss is multiplied by its square.
   - lr=10: stable but slow, from 22 to 3 in 10 steps.
   - lr=100: the step-0 factor is exactly −1, so the loss is unchanged, then it converges quickly to about 1e-23.
   - lr=1000: the step-0 factor is −19, so the loss is multiplied by 361 and diverges exponentially to about 1e18.
   - Conclusion: any factor with absolute value above 1 diverges.

#### Recall questions
1. **Why is the loss unchanged at step 0 with α=100?**
   <details><summary>Answer</summary>The factor is 1 − 2×100/100 = −1, so w only flips sign and w² stays the same.</details>
2. **What is the difference between `p.data -= ...` and `p = p - ...`?**
   <details><summary>Answer</summary>The first updates the parameter in place; the second only points the local name p at a new tensor, leaving the model parameter unchanged.</details>
3. **Why call `zero_grad()` before each step?**
   <details><summary>Answer</summary>`backward()` adds into p.grad, so without clearing it the previous step's gradient piles up.</details>

### AdamW

#### Background / motivation
- SGD uses one step size for every parameter, so dims with large gradients overshoot and dims with small gradients crawl. Adam tunes the step per element: m tracks direction, v tracks size, and m/√v makes each step roughly α.
- AdamW also "decouples" weight decay from the gradient update: it shrinks θ a little each step directly, instead of adding λθ into the gradient (where v would rescale it). It is the standard optimizer for training LLMs.

#### Formula (t starts at 1)
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,\qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$
$$\alpha_t = \alpha\frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t},\qquad \theta \leftarrow \theta - \alpha_t\frac{m_t}{\sqrt{v_t}+\varepsilon},\qquad \theta \leftarrow \theta - \alpha\lambda\theta$$
- $g$: gradient; $m$: first moment (moving average of the gradient); $v$: second moment (moving average of the squared gradient); $\beta_1=0.9$, $\beta_2=0.999$: decay rates; $\varepsilon=10^{-8}$; $\lambda$: weight-decay coefficient; $\alpha_t$: the bias-corrected lr (m and v start at 0, so they are too small in the first steps).

#### Inputs / outputs / shapes
- `AdamW(params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)`
- Per-parameter state: `state[p] = {"t": int, "m": same shape as p, "v": same shape as p}`, initialized with `torch.zeros_like(p)`. Optimizer state is twice the parameter count.

#### Implementation steps
```python
if len(state) == 0: state["t"] = 0; state["m"] = zeros_like(p); state["v"] = zeros_like(p)
t = state["t"] + 1
m = b1*m + (1-b1)*g;  v = b2*v + (1-b2)*g**2;  state["m"], state["v"] = m, v
lr_t = lr * math.sqrt(1 - b2**t) / (1 - b1**t)        # local variable
p.data -= lr_t * m / (torch.sqrt(v) + eps)
p.data *= 1 - lr * weight_decay                       # raw lr
state["t"] = t
```
- File `cs336_basics/adamw.py`; adapter `get_adamw_cls` returns the class `AdamW`; `test_adamw` PASSED.

#### Small numeric example
- One scalar parameter θ=1, g=0.5, α=0.1, λ=0.01, t=1:
  - m = 0.1 × 0.5 = 0.05; v = 0.001 × 0.25 = 0.00025, √v ≈ 0.0158
  - α_t = 0.1 × √0.001 / 0.1 ≈ 0.0316
  - step = 0.0316 × 0.05 / 0.0158 ≈ 0.1, so θ = 0.9
  - decay: θ × (1 − 0.1 × 0.01) = 0.8991
- With g = 50 instead the step is still about 0.1: the first step is about α regardless of the gradient size.

#### Common mistakes
1. **Adapter forgotten for the 7th time**: after editing `adamw.py` it still had `raise NotImplementedError`.
2. **The adapter returned an instance, `AdamW()`**: it should return the class `AdamW`, and the test constructs it with the parameters and hyperparameters.
3. **m and v written as `self.m = nn.Parameter(torch.zeros(params))`**:
   - Every parameter needs its own m and v, which one `self.m` cannot hold.
   - `params` is a sequence of tensors, not a shape, and a generator gets used up here.
   - m and v are optimizer state, not trainable parameters.
   - They belong in `self.state[p]`.
4. **m and v used before initialization and never stored back**: the first use is undefined.
5. **`torch.zeros(p)`**: zeros takes a shape. Use `torch.zeros_like(p)`, which also carries dtype and device.
6. **`math.sqrt(v)`**: v is a tensor, so use `torch.sqrt`.
7. **t incremented twice**: it was already `+1` at the start, and `t+1` was stored again at the end.
8. **defaults hard-coded as constants**: the passed-in betas, eps, and weight_decay were ignored.
9. **`ValueError(...)` without `raise`**.
10. **Imported as `adamW` while the file is `adamw.py`**: Windows is case-insensitive, but Linux will fail.

#### Noteworthy
1. **Algorithm (t starts at 1)**:
   - m ← β₁m + (1−β₁)g
   - v ← β₂v + (1−β₂)g²
   - α_t = α·√(1−β₂ᵗ)/(1−β₁ᵗ)
   - θ ← θ − α_t·m/(√v+ε)
   - θ ← θ − αλθ
2. **m gives direction, v gives scale**: m is a moving average of the gradient, so directions that flip back and forth cancel out. v is a moving average of the squared gradient. m/√v is about ±1, so the first step is about α regardless of how big the gradient is.
3. **Everything is elementwise**: m and v have the shape of p, and `g*g` is an elementwise square. Optimizer state is twice the parameter count.
4. **t must start at 1**: at t=0, α_t = 0/0 = nan. The pattern is increment-then-use: `t = state.get("t", 0) + 1`.
5. **Weight decay uses the raw α, not α_t**: decay has nothing to do with m and v, and α_t would make its strength dip and then recover over t. That is what "decoupled" means.
6. **Why weight decay**: it pulls unhelpful weights back toward 0, which limits overfitting and keeps the scale of logits and activations stable. In practice, biases and RMSNorm gains are usually not decayed.
7. **`super().__init__(params, defaults)`**: the base class turns params into `param_groups`, fills missing hyperparameters from defaults, and creates `self.state`.
8. **α_t is a local variable**: never write it back to `group["lr"]`, or the correction compounds every step.

#### Recall questions
1. **TinyStories loss = 1.5: what is the perplexity?**
   - Answer: e^1.5 ≈ 4.48, so the model is hesitating among about 4.5 words on average.
2. **What are the loss and perplexity for uniform guessing?**
   - My answer: 1 − log V.
   - Correction: loss = log V and perplexity = V ⚠️.
3. **What does `self.state[p]` store?**
   - My answer: the parameter's gradient.
   - Correction: the gradient is in `p.grad` ⚠️. state holds history across steps: t for SGD, and t, m, v for AdamW.
4. **Why can't m and v live on `self.m`?**
   - Answer: every parameter needs its own m and v, each with a different shape ✅.
5. **Why does the adapter return `AdamW` rather than `AdamW()`?**
   - Answer: the test constructs the optimizer itself with the model parameters and hyperparameters ✅.
6. **Why does weight decay use the raw lr?**
   - My answer: not sure.
   - Correction: α_t only corrects the bias in m and v. Decay is unrelated to them and should shrink θ by a steady (1−αλ) every step.
7. **With θ=1, g=0.5, α=0.1, λ=0.01, what is θ after the first step? And with g=50?**
   <details><summary>Answer</summary>About 0.8991; almost the same for g=50, because m/√v ≈ ±1.</details>
8. **Why must AdamW's t start at 1?**
   <details><summary>Answer</summary>At t=0, α_t = α·0/0 = nan. Increment first, then use it.</details>

#### Next step
- adamw_accounting: memory for AdamW training (parameters, gradients, optimizer state, activations) and FLOPs.
### adamw_accounting (2026-10-30)

#### Background / motivation
- transformer_accounting counted parameters and FLOPs for inference. Real training also has to answer: how big a batch fits in one GPU's memory, and how many days does a training run take?
- This is a written handout problem with no code or test.

#### Symbols
- P: parameter count; B: batch size (samples per batch); S, L, d, h, f, V as above; each float32 takes 4 bytes.
- MFU (Model FLOPs Utilization): achieved FLOP/s divided by the hardware peak.

#### Common mistakes
1. **Forgot gradients as one of the four categories**: memory holds parameters, gradients, optimizer state, and activations. Gradients are also P numbers, because every parameter has a same-shaped `p.grad`.
2. **Called activations "the input text"**: token ids are tiny. What takes memory is the forward intermediates, which backward needs.
3. **GB conversion**: 16P = 26.24 billion bytes = 26.24 GB, not 244 GB. Dividing by 10⁹ gives GB; dividing by 2³⁰ gives GiB (about 24.4).
4. **Put ln_final and lm_head inside the block**: they appear once, after all L blocks. The two residual adds were also missing.
5. **Treated B as GB**: B is a sample count and must be an integer, so 80 GB allows at most B=3.
6. **TFLOP/s conversion**: 495 TFLOP/s = 4.95e14, not 495e13. T is 10¹².
7. **No parentheses around the denominator**: `/50%*495e13` divides by 50% and then multiplies. It should be `/(0.5×4.95e14)`.

#### Noteworthy
1. **The four memory categories**:
   - Parameters: P
   - Gradients: P
   - AdamW state (m, v): 2P
   - The first three total 4P float32 values, or 16P bytes. For GPT-2 XL that is 26.24 GB, more than a 24 GB card holds on its own.
2. **Activation formula**: elements = L(8BSd + 3BSf + 2BhS²) + BSd + 2BSV. Multiply by 4 for bytes.
3. **GPT-2 XL memory ≈ 15.53·B + 26.24 GB**: 80 GB fits at most B=3 (72.8 GB). B=4 needs 88.4 GB, which is too much.
4. **Where activations go at S=1024**:
   - 2BhS² (QKᵀ and softmax): about 65%
   - 3BSf: about 16%
   - 8BSd: about 16%
   - Outside the blocks: about 3%
   - The (B,h,S,S) matrices dominate, which is why FlashAttention doesn't store them and recomputes them blockwise in backward.
5. **AdamW is about 14 FLOPs per parameter, 14P per step**: that is negligible next to forward plus backward at 6P·BS (about 6144P at B=1). α_t is a scalar computed once per step, not multiplied by P.
6. **Backward ≈ 2× forward**: the backward of Y=XW computes dX = dY·Wᵀ and dW = Xᵀ·dY, two matmuls of the same size. So training costs about 6 FLOPs per parameter per token. dW needs the forward input X, which is why activations must be stored.
7. **Training time**: GPT-2 XL, 400K steps, B=1024, S=1024, one H100 in fp32 (495 TFLOP/s), 50% MFU.
   - Total FLOPs = 3.52e12 × 3 × 1024 × 4e5 ≈ 4.33e21
   - ÷ 2.475e14 ≈ 1.75e7 s ≈ **202 days**
   - Hence multi-GPU parallelism and bf16 in real training.

#### Small numeric example
- A small model with P = 1,000,000 parameters in fp32: parameters 4 MB + gradients 4 MB + m and v 8 MB = 16 MB, i.e. 16P bytes.
- GPT-2 XL: P ≈ 1.64×10⁹, so 16P ≈ 26.24 GB; at B=1 activations add about 15.53 GB, about 41.8 GB in total.

#### Recall questions
1. **What four kinds of things does AdamW training keep in memory?**
   - My answer: inputs, parameters, optimizer state.
   - Correction: gradients were missing ⚠️, and "inputs" should be activations.
2. **How many GB is 16P bytes for GPT-2 XL?**
   - My answer: 244 GB.
   - Correction: 26.24 GB ⚠️.
3. **Which activation scales with S²?**
   - Answer: QKᵀ and softmax, with shape (B,h,S,S) ✅.
4. **Why is backward about 2× forward?**
   - Answer: it needs two matmuls, one for dX and one for dW.
5. **How long does GPT-2 XL take on one H100?**
   - Answer: about 202 days.
6. **About how many FLOPs does training take per parameter per token?**
   <details><summary>Answer</summary>About 6: 2 forward and 4 backward (two matmuls, for dX and dW).</details>
7. **Why must activations be stored?**
   <details><summary>Answer</summary>Backward computes dW = Xᵀ·dY, which needs the forward input X.</details>

#### Next step
- §4.4 cosine learning-rate schedule, starting from why warmup is needed.

## Day — LR schedule, gradient clipping and data loading (2026-10-30 ~ 2026-11-04)

#### What I did
- §4.4 cosine learning-rate schedule with warmup: `lr_schedule` in `cs336_basics/lr_schedule.py`, adapter `run_get_lr_cosine_schedule`, `test_get_lr_cosine_schedule` PASSED.
- §4.5 gradient clipping: `gradient_clipping` in `cs336_basics/grad_clip.py`, adapter `run_gradient_clipping`, `test_gradient_clipping` PASSED.
- §5.1 data loading: `data_loading` in `cs336_basics/load_dataset.py`, adapter `run_get_batch`, `test_get_batch` PASSED.
- 2026-11-03: added Chinese docstrings and inline comments to every module under `cs336_basics` (see the end of this section).

### §4.4 Cosine learning-rate schedule (lr_schedule)

#### Background / motivation
- The learning rate (lr) shouldn't stay fixed for the whole run. A scheduler is a function from the current step t to the lr for that step. LLaMA uses "linear warmup + cosine annealing".
- **Why warmup**:
  1. At the start the parameters are randomly initialized, the loss is large, and gradients are large and noisy. A large lr right away can push the parameters somewhere bad, or even turn the loss into NaN.
  2. In the first few steps Adam's v is estimated from only a handful of gradients and is unreliable. If v is underestimated, m/√v is too large and the actual step is too big.
  3. So start with a tiny lr, let m, v and the parameters settle, then ramp up to α_max.
- **Why decay**: near the minimum, a large step keeps jumping over it and oscillates from side to side. Only a smaller lr lets it settle in.
  - Example: loss = w², gradient 2w, update $w \leftarrow w - \alpha\cdot2w = (1-2\alpha)w$. With α=1 the factor is −1: w = 0.1 → −0.1 → 0.1 …, bouncing around the minimum forever. With α=0.1 the factor is 0.8: 0.1 → 0.08 → 0.064 …, converging. Real gradients are also noisy, so without lowering the lr late in training the parameters keep jittering around the minimum.
- Where it sits: each step of the training loop first computes `lr = lr_schedule(t, ...)`, writes it into `optimizer.param_groups[i]["lr"]`, then calls `optimizer.step()`.

#### Formula (three pieces)
$$\alpha_t = \begin{cases}\dfrac{t}{T_w}\,\alpha_{max} & t < T_w \quad\text{(warmup)}\\[2mm] \alpha_{min} + \dfrac12\left(1+\cos\!\Big(\dfrac{t-T_w}{T_c-T_w}\pi\Big)\right)(\alpha_{max}-\alpha_{min}) & T_w \le t \le T_c \quad\text{(cosine annealing)}\\[2mm] \alpha_{min} & t > T_c \quad\text{(post-annealing)}\end{cases}$$
- $t$: current step; $T_w$: number of warmup steps; $T_c$: the step where cosine annealing ends; $\alpha_{max}$, $\alpha_{min}$: the maximum and minimum learning rates.
- **Endpoint checks**:
  - $t=T_w$: cos 0 = 1, so $\alpha = \alpha_{min} + (\alpha_{max}-\alpha_{min}) = \alpha_{max}$ (not the midpoint value).
  - $t=T_c$: cos π = −1, so $\alpha = \alpha_{min}$.
  - Exactly halfway, $t=(T_w+T_c)/2$: cos(π/2) = 0, so $\alpha = (\alpha_{max}+\alpha_{min})/2$.
  - The warmup piece also reaches α_max at $t=T_w$, so the curve is continuous.

#### Inputs / outputs / shapes
- `lr_schedule(t, t_w, t_c, lr_min, lr_max) -> float`: pure scalar math, no tensors, no parameters.
- The adapter uses different names: `run_get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)`, passed through one by one by name.

#### Implementation steps
```python
def lr_schedule(t, t_w, t_c, lr_min, lr_max):
    if t < t_w:
        lr_t = t / t_w * lr_max
    elif t <= t_c:
        lr_t = lr_min + 0.5 * (1 + math.cos(math.pi * (t - t_w) / (t_c - t_w))) * (lr_max - lr_min)
    else:
        lr_t = lr_min
    return lr_t
```

#### Small numeric example
- α_max = 1e-3, α_min = 1e-4, and let $p = (t-T_w)/(T_c-T_w)$ be the annealing progress:

  | p | ½(1+cos pπ) | α_t |
  |---|---|---|
  | 0 | 1 | 1.00e-3 |
  | 0.25 | 0.854 | 8.68e-4 |
  | 0.5 | 0.5 | 5.50e-4 |
  | 0.75 | 0.146 | 2.32e-4 |
  | 1 | 0 | 1.00e-4 |

- Shape of the curve: slow decline at the start and end, fast in the middle, like half of an upside-down cosine wave.
- With warmup, T_w=100, T_c=1000: t=0 → 0; t=50 → 5e-4; t=100 → 1e-3; t=550 → 5.5e-4; t=1000 → 1e-4; t=2000 → 1e-4.

#### Common mistakes
1. **Couldn't say why warmup is needed**: the answer is the three points above (large noisy early gradients, Adam's v unreliable early so the step is too big, settle with small steps then ramp up).
2. **Thought the lr at t=T_w is the midpoint value**: at t=T_w, cos 0 = 1 and the lr is exactly α_max; the midpoint (α_max+α_min)/2 comes at t=(T_w+T_c)/2.
3. **Put `lr_t` in the function signature as an input**: lr_t is the output to compute, not a parameter.
4. **Forgot `import math`**: both `math.cos` and `math.pi` are needed.
5. **Left a dead `raise NotImplementedError` after `return` in the adapter** (still there; it can be deleted).
6. **T_c == T_w divides by zero**: at t = T_w it enters the middle branch with denominator T_c − T_w = 0 (flagged in the comments).

#### Things worth noting
1. The scheduler only returns a number and never touches the parameters. The training loop has to write it into the optimizer's `param_groups` itself.
2. Warmup starts at t=0, so step 0 has lr = 0 and doesn't actually update the parameters.
3. A handout footnote mentions schedules where the lr periodically goes back up (restarts) to escape local optima; A1 uses a single annealing pass.
4. AdamW's internal α_t bias correction is a separate thing: the schedule sets the base lr α, and AdamW multiplies that by √(1−β₂ᵗ)/(1−β₁ᵗ).

#### Recall questions
1. **Why is warmup needed?**
   <details><summary>Answer</summary>At random init gradients are large and noisy; Adam's v is unreliable early, so m/√v is too large and steps are too big. Start with a small lr to settle, then ramp up to α_max.</details>
2. **Why lower the lr late in training?**
   <details><summary>Answer</summary>Near the minimum a large step keeps jumping over it and oscillating. In the loss=w² example, α=1 gives factor −1, bouncing between ±w forever.</details>
3. **What is the lr at t=T_w, t=T_c and t=(T_w+T_c)/2?**
   <details><summary>Answer</summary>α_max, α_min, (α_max+α_min)/2.</details>
4. **With α_max=1e-3 and α_min=1e-4, what is the lr halfway through annealing?**
   <details><summary>Answer</summary>5.5e-4.</details>

### §4.5 Gradient clipping

#### Background / motivation
- Training occasionally hits an abnormal batch (very long repeated text, dirty data) and the gradient suddenly gets huge, say 1000 instead of the usual 1. That step pushes the parameters far away, the loss spikes, or even becomes NaN, and earlier progress is wrecked.
- Lowering the lr doesn't fix it: the lr is the same for every step, so lowering it slows down the normal steps too, and since the step is proportional to lr × gradient, the step with a 1000× gradient is still 1000× larger.
- Gradient clipping shortens the gradient to the threshold M only when it is "too long"; normal steps are unaffected.
- Where it sits: in the training loop after `loss.backward()` and before `optimizer.step()`.

#### Formula
- Global L2 norm: flatten every parameter's gradient and concatenate them into one long vector g:

  $$\|g\|_2 = \sqrt{\sum_{\text{param } i}\ \sum_{\text{elements}} g^2} = \sqrt{\|g_1\|^2 + \|g_2\|^2 + \cdots}$$

- If $\|g\|_2 > M$: multiply every gradient by the same factor $\dfrac{M}{\|g\|_2+\varepsilon}$, with $\varepsilon = 10^{-6}$. The clipped norm ends up just under M. Otherwise leave them alone.
- $M$: the `max_l2_norm` threshold; $g_i$: the gradient tensor of parameter i (same shape as the parameter).

#### Inputs / outputs / shapes
- `gradient_clipping(parameters, max_l2_norm, eps=1e-6) -> None`
- `parameters`: an iterable of parameters (possibly a generator such as `model.parameters()`); each `p.grad` has p's shape and may be None.
- No return value; each `p.grad` is modified in place.

#### Implementation steps
```python
grads = [p.grad for p in parameters if p.grad is not None]   # store as a list first, skip None
total_norm = torch.sqrt(sum((g ** 2).sum() for g in grads))   # per-tensor sums of squares, add them, then one sqrt
if total_norm > max_l2_norm:
    scale = max_l2_norm / (total_norm + eps)
    for g in grads:
        g *= scale                                             # in place, so p.grad changes too
```
- Why store a list first: `parameters` may be a generator that can only be iterated once, so a second pass to scale after computing the norm would find it empty.
- Why `g *= scale`: only an in-place update modifies `p.grad` itself; `g = g * scale` just points the local variable at a new tensor.

#### Small numeric example
- Two parameters whose gradients together are (30, 40): the norm is √(900+1600) = 50, M=5, factor = 5/50.000001 ≈ 0.1, giving (3, 4) with norm ≈ 5 and the same direction.
- Clipping each parameter separately (each parameter's norm at most 5): 30 → 5 and 40 → 5, giving (5, 5). The direction changes from about 53° to 45°, so the update direction is distorted.
- Why not compute an L2 per g: what gets clipped is the whole model's update for this step, i.e. the concatenated long vector. The squared total norm is the sum of the squared per-parameter norms. For example g₁ = [[1,2],[2,0]] (sum of squares 9, norm 3) and g₂ = [4,0,0] (norm 4) give a total norm of √(9+16) = 5, not 3+4=7. With M=5 no clipping is needed. The code does compute each g's sum of squares; it just doesn't take a sqrt per g and compare, it adds them first and takes one sqrt.
- One parameter being small on its own doesn't mean the whole is small. One shared factor guarantees two things: the direction is unchanged, and the total length of the step is at most M.

#### Common mistakes
1. **Used `torch.pow(grads, 2)` on the list directly**: grads is a Python list of tensors with different shapes; compute each g's sum of squares and add them up.
2. **Forgot the square root**: that gives the sum of squares, not the norm. For a gradient (3, 4) the sum of squares is 25 and the norm is 5; with threshold 5 nothing should be clipped, but comparing 25 clips wrongly.
3. **Misspelled the parameter `max_12_norm` (digits 12)**: it should be `max_l2_norm` (letter l); the adapter passes arguments by name, so they won't match.
4. **An unused `import math`**.
5. **The `raise NotImplementedError` after `return` in the adapter never runs** (still there; it can be deleted).
6. **Errors when every grad is None**: `sum` over an empty sequence returns the Python int 0, and `torch.sqrt(0)` gets an int and raises TypeError (flagged in the comments).

#### Things worth noting
1. Skip parameters with `p.grad is None`: parameters that took no part in the forward pass have no gradient.
2. The clipped norm is M·‖g‖/(‖g‖+ε), just under M, which is why the test matches PyTorch's `clip_grad_norm_` (PyTorch's ε is also 1e-6).
3. Clipping changes only the length, not the direction. Adam's m/√v already does some "automatic scaling"; clipping mainly stops a single abnormal step from dragging m, v and the parameters off course.
4. In practice you log the pre-clip total norm every step; a sudden jump is often a warning sign of a loss spike.

#### Recall questions
1. **Why doesn't lowering the lr fix an exploding gradient on one batch?**
   <details><summary>Answer</summary>The lr is the same every step, so lowering it slows the normal steps too, and the abnormal step is still 1000× a normal one. Clipping shortens only the step that is too long.</details>
2. **Two parameters have gradient norms 3 and 4. What is the total norm? Clip with M=5?**
   <details><summary>Answer</summary>√(9+16) = 5; no clipping (not greater than M).</details>
3. **For (30, 40) and M=5, what do global and per-parameter clipping give?**
   <details><summary>Answer</summary>Global gives (3, 4), same direction; per-parameter gives (5, 5), direction changed.</details>
4. **Why store grads as a list first? Why `g *= scale`?**
   <details><summary>Answer</summary>parameters may be a generator that can only be iterated once; only the in-place multiply modifies p.grad itself.</details>

### §5.1 Data loading (get_batch)

#### Background / motivation
- The training data is one very long sequence of token ids (the tokenized corpus concatenated, possibly hundreds of millions of tokens). Each step randomly takes a batch of (x, y) from it to feed the model.
- A language model's goal is to predict the next token, so **y is x shifted by one position**: each position of y is the "next token" of the corresponding position of x.
- Benefits of sampling this way: any start is a valid sample, so sampling is simple; every sample has the same length, so no padding; and the whole dataset never has to be in memory.
- Where it sits: `dataset` (memmap) → **get_batch** → x goes into the model to get logits, and y and the logits together give the cross-entropy.

#### Formula
- Data $d_0, d_1, \dots, d_{n-1}$ of length n, context_length = L. Pick a random start i:

  $$x = [d_i, \dots, d_{i+L-1}],\qquad y = [d_{i+1}, \dots, d_{i+L}]$$

- y's last index satisfies $i+L \le n-1$, so $i \in [0,\ n-L-1]$, $n-L$ valid starts in total.
- The upper bound of `np.random.randint(low, high, size)` is exclusive, so write `randint(0, n - L, size=B)`, whose maximum is n−L−1.

#### Inputs / outputs / shapes
- `data_loading(dataset, batch_size, context_length, device=None) -> (x, y)`
- `dataset`: a 1-D numpy integer array (or memmap) of length n.
- `x`, `y`: both `torch.long`, shape `(batch_size, context_length)`, on `device`. No parameters.
- Shapes downstream: x `(B, L)` → embedding `(B, L, d_model)` → … → logits `(B, L, V)` → `cross_entropy(logits, y)`. y never enters the model; it is only the label (target) when computing the loss.
- **batch_size is "how many samples in one batch"** (the number of rows), not "how many batches in total". The total number of batches is the number of training steps, one get_batch call per step. Total tokens seen = steps × B × L.
- Each row is one sample, but each row contains L small "predict the next token" tasks: thanks to the causal mask, position t only sees 0..t. So one batch contributes B×L predictions, and the cross-entropy averages over all of them.

#### Implementation steps
```python
def data_loading(dataset, batch_size, context_length, device=None):
    n = len(dataset)
    starts = np.random.randint(0, n - context_length, batch_size)          # B starts in [0, n-L-1]
    x = np.stack([dataset[i : i + context_length] for i in starts])        # (B, L)
    y = np.stack([dataset[i + 1 : i + 1 + context_length] for i in starts])  # (B, L), shifted by one
    x = torch.tensor(x, dtype=torch.long).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)
    return x, y
```
- The file was first called `get_batch.py` and later renamed to `cs336_basics/load_dataset.py`, with the function named `data_loading`; the adapter `run_get_batch` calls it.

#### Small numeric example
- Data `[5, 8, 2, 9, 4, 7, 1]`, n=7, L=3. Start i=1: x = `[8, 2, 9]`, y = `[2, 9, 4]`. Seeing 8 the model must predict 2; seeing 8, 2 it must predict 9; seeing 8, 2, 9 it must predict 4.
- Largest start: i + 3 ≤ 6, so i ≤ 3. With i=3, x = `[9, 4, 7]` and y = `[4, 7, 1]`, using exactly the last token. With i=5, x is already out of range.
- B=2 with starts 1 and 3:

  ```
  x = [[8, 2, 9],     y = [[2, 9, 4],
       [9, 4, 7]]          [4, 7, 1]]
  ```

  Both have shape (2, 3).

#### Large datasets: memmap
- Real datasets are too large to load into memory. Use `np.memmap("train.bin", dtype=np.uint16, mode="r")`, or `np.load("train.npy", mmap_mode="r")` if it was saved with `np.save`. The array looks complete, but only the slices you touch are read from disk. `get_batch` itself needs no changes.
- memmap takes a **file path**, not an array already in memory, so it belongs in the training script where data is loaded, not as `dataset = np.memmap(dataset)`.
- **The dtype must match the one used when saving**: a memmap file doesn't record its dtype, and reading it with the wrong one gives garbage. With vocab < 65536, uint16 (2 bytes) uses 4× less space than int64. The handout suggests checking after loading that the max value is below vocab_size.

#### Common mistakes
1. **Thought x and y had shapes seq_len and d_model**: both are token ids with shape (B, L). d_model only appears after the embedding.
2. **Thought batch_size meant "how many batches in total"**: it is the number of samples (rows) in one batch.
3. **Answered 5 for the largest start**: with n=7 and L=3 it is 3; in general n−L−1.
4. **First draft**: `torch.long(np.random.randint(batch_size), ...)`. randint returns a single integer in [0, B), unrelated to the data length; dataset is never used; the data type `torch.long` is called like a function; and it returns one value, not (x, y).
5. **Second draft**: `np.random.randint(0, batch_size, n - context_length)` swaps the upper bound and the count; `[i, context_length]` is just a two-element list, not a slice.
6. **Third draft**: `dataset[i, i+context_length]` uses a comma, which on a 1-D array is a 2-D index and raises `IndexError: too many indices`; it needs a colon. The upper bound was written as `n - context_length + 1`, which can pick i = n−L, leaving y with only L−1 elements so `np.stack` fails (off-by-one).
7. **An unused `import torch.nn as nn`** (still there).

#### Things worth noting
1. x is the model input and y is the label; y never enters the model.
2. Each row's start is drawn independently, so samples in one batch may overlap.
3. `torch.tensor(..., dtype=torch.long)` makes a copy. A uint16 memmap slice must become long before it can index the embedding.
4. Move to device: when the model is on the GPU, x and y must be on the same device.

#### Recall questions
1. **How are x and y related? What are their shapes?**
   <details><summary>Answer</summary>y is x shifted by one; both are (batch_size, context_length) token ids.</details>
2. **For data length n and context_length L, what is the valid range of starts? What upper bound goes into `randint`?**
   <details><summary>Answer</summary>[0, n−L−1], n−L of them; the upper bound is exclusive, so n−L.</details>
3. **batch_size=4, L=3, 1000 training steps: how many tokens are seen in total? How many predictions are in one batch?**
   <details><summary>Answer</summary>1000 × 4 × 3 = 12000; one batch has 4 × 3 = 12 predictions.</details>
4. **What does `np.memmap` take? Why must the dtype be right?**
   <details><summary>Answer</summary>A file path; the file doesn't record its dtype, so the wrong one turns the data into garbage.</details>
5. **What is the difference between `dataset[i, i+L]` and `dataset[i : i+L]`?**
   <details><summary>Answer</summary>The first is a 2-D index (row i, column i+L) and raises IndexError on a 1-D array; the second is a slice of L consecutive elements.</details>

### 2026-11-03: Chinese comments for every module

- Every module under `cs336_basics` got Chinese docstrings and inline comments: for each class and function, what it does, its inputs and outputs, tensor shapes, the shapes of learnable parameters, and the formulas used. Comments only, no logic changes, and the test results were the same before and after. The changes are not committed yet.
- Suspicious spots found along the way, all flagged with `# 注意` and left unchanged:
  1. `transformer_lm.py`: `forward` doesn't pass `token_positions` to the blocks, so RoPE always uses 0..S−1.
  2. `lr_schedule.py`: the middle branch divides by zero when T_c == T_w.
  3. `grad_clip.py`: `torch.sqrt(0)` raises when every grad is None.
  4. `tokenizer.py`: no cache, so it is slow; special tokens must already be in vocab or encode raises KeyError; the old first-line comment is out of date.
  5. Every train_bpe version overwrites `merges.txt` in the current directory on each run; the bucket version writes a `worker_*.prof` per worker (that is where the prof files under `a1-basics` come from).

#### Next
- §5.2 checkpointing (saving and resuming training): first work out what, besides the model parameters, must be saved to resume a run (the optimizer state m, v, t, and the current step).

## Day — Checkpointing, training loop, encoding, and generation (2026-11-04 ~ 2026-11-05)

#### What we did
- §5.2 Checkpointing: `save_checkpoint` / `load_checkpoint` in `cs336_basics/model_checkpoint.py`; `test_checkpointing` PASSED.
- Training loop: `cs336_basics/train.py`, wiring get_batch → forward → cross-entropy → backward → grad clip → AdamW → val / save.
- Switched BPE artifacts to pickle, and wrote `cs336_basics/encode_dataset.py` to encode TinyStories into `uint16` `.bin` files.
- Installed CUDA PyTorch (cu128, Aliyun mirror) and trained on an RTX 3060 Laptop 6GB to about steps 21000–22000, with val loss around 1.4.
- §6 Text generation: `cs336_basics/generate.py` with greedy / temperature / top-p (nucleus sampling), continuing stories from the iteration-22001 checkpoint.
- Journal and code-comment changes are still largely uncommitted; this turn does not commit either.

### §5.2 Checkpointing (`cs336_basics/model_checkpoint.py`)

#### Background / motivation
- Training runs for tens of thousands of steps; crashes, kills, and machine swaps are normal. Resuming must pick up where you left off — model weights alone are not enough.
- **Three things you must save**:
  1. **Model parameters**: `model.state_dict()` (weights, biases, …).
  2. **Optimizer state**: AdamW's per-parameter first moment m, second moment v, and timestep t. Without them, a resume is a cold Adam restart and the earlier warmup / adaptive scaling is wasted.
  3. **Current iteration**: the LR schedule is a function of step count; logging and save intervals depend on it too.
- **Do not save gradients**: `p.grad` is recomputed by the next `backward`. Saving it wastes disk and does nothing useful.

#### Inputs / outputs / shapes
- `save_checkpoint(model, optimizer, iteration, out)`: no return value. `out` may be a path string or an open file-like object.
- `load_checkpoint(src, model, optimizer) -> int`: fills an **already constructed** model and optimizer, returns the saved `iteration`.
- On disk, one dict:

  ```python
  {
      "model": model.state_dict(),          # name → tensor
      "optimizer": optimizer.state_dict(),  # state (m, v, t) and param_groups
      "iteration": iteration,              # int
  }
  ```

#### Implementation steps
```python
def save_checkpoint(model, optimizer, iteration, out):
    save_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(save_dict, out)

def load_checkpoint(src, model, optimizer):
    save_dict = torch.load(src)
    model.load_state_dict(save_dict["model"])
    optimizer.load_state_dict(save_dict["optimizer"])
    return save_dict["iteration"]
```
- A `state_dict` stores tensors **by name**, not the architecture. Before load you must build `TransformerLM` and `AdamW` with the same hyperparameters, then `load_state_dict` to fill the values in.
- PyTorch 2.6+ defaults to `weights_only=True`: do not put custom Python objects in `torch.save` — only tensors / primitives, or unpickling will fail or be refused.

#### Small numeric example
- Save at the end of step 2000 with `iteration=2001` (see the training loop: store “where the next step should start”). The file holds all model weights, AdamW's m/v per parameter, and t.
- After restart: construct `TransformerLM(...)` and `AdamW(...)`, then `start_it = load_checkpoint(...)` → 2001, and `for it in range(2001, max_iters)` continues without redoing step 2000.

#### Common mistakes
1. **Thought gradients must be saved too**: the next `backward` recomputes them.
2. **Type alias written as `dict[model, optimizer, iteration]`**: not a valid runtime type; if type objects actually land in the save dict, `torch.load` raises `UnpicklingError`.
3. **`load_checkpoint` forgot `return iteration`**: the caller cannot resume at the right step.
4. **Saved the whole optimizer object instead of `optimizer.state_dict()`**: a custom `AdamW` class will not unpickle cleanly → `UnpicklingError`. Persist only the state_dict (tensors + primitives).
5. **Did not understand why load still needs model and optimizer arguments**: the file has “name → value” only, not how layers stack or how the optimizer is built; the receiving objects must already exist.

#### Things worth noting
1. `out` / `src` support both paths and file objects; the tests use both.
2. On resume, do not freeze the LR at the optimizer's initial value: each step still writes `lr_schedule(t=it, ...)` into `param_groups`, so the LR curve stays continuous from step 2001.
3. `test_checkpointing` PASSED.

#### Recall questions
1. **What three things must a resume save? Do gradients belong?**
   <details><summary>Answer</summary>Model state_dict, optimizer state_dict (m, v, t), and iteration. Gradients do not — the next backward recomputes them.</details>
2. **Why must `load_checkpoint` take an already-built model and optimizer?**
   <details><summary>Answer</summary>A state_dict stores tensors by name, not architecture or optimizer class; build the shells first, then load the numbers.</details>
3. **What goes wrong if you save the whole optimizer object?**
   <details><summary>Answer</summary>Unpickling often fails when the class does not match; save only the state_dict.</details>

### Training loop (`cs336_basics/train.py`)

#### Background / motivation
- The earlier pieces (model, loss, AdamW, LR schedule, clipping, get_batch, checkpoint) are parts; `train.py` screws them together in the right order so a multi-ten-thousand-step run can finish.
- The wrong order shows up as silent bugs: updating with stale grads, clipping after the update, or an LR schedule that looks written but AdamW never reads.

#### Correct order each step
1. **Compute this step's lr and write `optimizer.param_groups[i]["lr"]`**. AdamW's `step()` reads `group["lr"]`, not an `optimizer.lr` attribute.
2. **`get_batch` / `data_loading`** for `(x, y)`.
3. **Forward**: `logits = model(x)`, `loss = cross_entropy(logits, y)`.
4. **`optimizer.zero_grad()`**: clear leftover `.grad` from the previous step.
5. **`loss.backward()`**: compute this step's gradients.
6. **`gradient_clipping(...)`**: shrink the global grad norm if needed.
7. **`optimizer.step()`**: AdamW update.
8. **Log / validate / save** on their intervals.

Validation: `model.eval()` + `torch.no_grad()`, average loss over `eval_batches`, then `model.train()`. When saving, use `iteration=it+1` so resume means “start from the next step” and does not redo the step just finished; call `os.makedirs(..., exist_ok=True)` before save.

#### Inputs / outputs / shapes (TinyStories defaults)
- Data: `data/tinystories_train.bin`, `data/tinystories_valid.bin` (`uint16` memmap).
- Model: `vocab_size=10000`, `context_length=256`, `d_model=512`, `d_ff=1344`, `num_layers=4`, `num_heads=16`, `rope_theta=10000`.
- Train: `batch_size=32`, `max_iters=40000`, `t_w=1000`, `t_c=40000`, `lr_max=1e-3`, `lr_min=1e-4`, `grad_clip=1.0`; `log_interval=100`, `valid_interval=1000`, `save_interval=2000`, `eval_batches=20`.

#### Implementation steps (skeleton)
```python
for it in range(start_it, args.max_iters):
    lr = lr_schedule(t=it, t_w=..., t_c=..., lr_min=..., lr_max=...)
    for g in optimizer.param_groups:
        g["lr"] = lr
    x, y = data_loading(train_dataset, batch_size, context_length, device)
    logits = model(x)
    loss = cross_entropy(logits, y)
    optimizer.zero_grad()
    loss.backward()
    gradient_clipping(model.parameters(), max_l2_norm=grad_clip)
    optimizer.step()
    # on intervals: print loss; eval; save_checkpoint(..., iteration=it+1, ...)
```

#### Small numeric example
- Step 0, random init, vocab 10000: cross-entropy should be about \(\ln(10000) \approx 9.21\); we saw ~9.28 — a healthy start.
- Warmup ends at step 1000 (`lr` reaches `lr_max`); cosine decay is steepest in the middle of the schedule.

#### Common mistakes
1. **First draft had the order backwards**: `step` before `backward`, clipping after `step`, LR set last — updates used empty/stale grads, and both clip and schedule were no-ops in practice.
2. **memmap `dtype` mismatched the file**: garbage values; must be `uint16` as at encode time.
3. **train and valid pointed at the same file**: validation loss had no independent meaning.
4. **Wrong constructor kwargs** for `TransformerLM` / `AdamW`.
5. **Saved `iteration=it` instead of `it+1`**: resume redid the step just completed.

#### Things worth noting
1. **Dropout, briefly**: at train time, zero activations with probability p and divide by \(1-p\) (inverted dropout) so \(\mathbb{E}[x]=x\). LLM (large language model) pretraining usually skips dropout; this assignment's default has none either.
2. Always `eval()` + `no_grad()` for val, then `train()` again — good habit even though this model has no BN/Dropout.
3. Per-step train loss is noisy (±0.1); trust the val_loss trend.

#### Recall questions
1. **What is the correct order of one training step? Where does the LR go?**
   <details><summary>Answer</summary>Set param_groups["lr"] → get_batch → forward+loss → zero_grad → backward → clip → step → log/val/save. AdamW.step reads group["lr"].</details>
2. **Why save `iteration=it+1`?**
   <details><summary>Answer</summary>It marks where the next step should start, so `range(start_it, max_iters)` on resume does not redo the finished step.</details>
3. **What must the validation loop do?**
   <details><summary>Answer</summary>`model.eval()` + `torch.no_grad()`, average several batches, then `model.train()`.</details>

### BPE pickle and dataset encoding (`cs336_basics/encode_dataset.py`)

#### Background / motivation
- The old `vocab*.txt` / `merges*.txt` decoded `bytes` to text for storage: spaces, `' t'` vs two spaces, and broken UTF-8 fragments could not round-trip, so reload disagreed with the BPE (Byte Pair Encoding) used at train time.
- Fix: retrain TinyStories BPE with vocab_size=10000 and special token `<|endoftext|>`, then `pickle.dump` the binary structure:

  ```python
  {"vocab": dict[int, bytes], "merges": list[tuple[bytes, bytes]]}
  ```

  Path: `artifacts/tinystories_bpe/bpe.pkl`. Result: vocab 10000, 9743 merges, `<|endoftext|>` id = 0.
- **A Tokenizer is not a neural net**: no learnable parameters, only fixed merge rules and a vocab lookup. BPE “training” only counts frequencies and chooses merge order.

#### Inputs / outputs / shapes
- `encode_file(tokenizer, in_path, out_path, chunk=1_000_000) -> total_tokens`
- Input: UTF-8 text; output: raw little-endian `uint16` stream (2 bytes per token id).
- TinyStories: the **train** split trains both BPE and the LM; the **valid** split is only encoded, not used for BPE training.

#### Implementation steps
1. `pickle.load` vocab / merges; build `Tokenizer(..., special_tokens=["<|endoftext|>"])`.
2. Stream ids with `tokenizer.encode_iterable(fin)` — do not load the ~2GB text at once.
3. Every `chunk` ids, `np.array(buf, dtype=np.uint16).tobytes()` to disk; a Python `list` int is ~28–36 bytes vs 2 for `uint16`.
4. Assert: file size == `total * 2`; memmap length == `total`; `max id < 10000`.

#### Small numeric example / measured
- valid: 5,461,210 tokens; train: 540,796,778 tokens; max id = 9999.
- About 4.12 raw bytes / token.
- Decoding a prefix matches the raw text; valid starting mid-sentence is a dataset split artifact (train was cut mid-word), not an encoder bug.
- A pretok cache was added to the Tokenizer for speed.

#### Common mistakes
1. **Kept text vocab/merges files**: space / broken-UTF-8 ambiguity; encode disagreed with BPE training.
2. **`encode` on the whole file at once**: memory and speed blow up.
3. **Wrote int64 or a pickled Python list**: far larger than uint16.
4. **Forgot the special token**: `<|endoftext|>` must be in vocab or encode raises KeyError.

#### Things worth noting
1. Pickle stores `bytes` directly — no lossy str round-trip.
2. eot id = 0; generation stops when 0 is sampled.
3. memmap of the `.bin` must use `dtype=uint16`.

#### Recall questions
1. **Why switch BPE storage to pickle instead of vocab.txt / merges.txt?**
   <details><summary>Answer</summary>Text round-trips make bytes ambiguous (spaces, broken UTF-8); pickle stores dict[int,bytes] and the merges list directly.</details>
2. **Is the Tokenizer a neural net? What does BPE “training” learn?**
   <details><summary>Answer</summary>No; no gradients. BPE training only counts frequencies and produces a fixed merge order and vocab.</details>
3. **Why stream uint16 to disk?**
   <details><summary>Answer</summary>vocab&lt;65536 so two bytes suffice; streaming avoids loading the whole corpus; list ints are an order of magnitude larger per id.</details>

### GPU PyTorch install (brief)

#### Background / motivation
- The env had `2.11.0+cpu`, so `torch.cuda.is_available()` was False and training was stuck on CPU.
- Hardware: RTX 3060 Laptop 6GB — enough for the TinyStories config (batch 32, context 256).

#### What we did
- Added a PyTorch cu128 index via `uv`. When `download.pytorch.org` was slow, switched to the Aliyun mirror with `format = "flat"` (a flat whl directory, not a standard pep503 index):

  ```toml
  [[tool.uv.index]]
  name = "pytorch-cu128"
  url = "https://mirrors.aliyun.com/pytorch-wheels/cu128/"
  format = "flat"
  explicit = true
  ```

- Raised `UV_HTTP_TIMEOUT` so large wheels would not time out mid-download. Afterward `cuda.is_available()` was True.

### Training run results

#### Process
- Smoke test first, then full run: laptop 3060, `batch_size=32`, `max_iters=t_c=40000`, `t_w=1000`, `lr_max=1e-3`.
- A one-time `kernel cache` directory warning appeared; it does not affect numerics and can be ignored.

#### Numbers
| step | train loss (approx) | val loss |
|------|---------------------|----------|
| 0 | 9.28 | 9.28 |
| 1000 | 2.30 | 2.35 |
| 2000 | 1.90 | **1.95** (already under the handout target 2.00) |
| 7000 | 1.68 | 1.63 |
| ~21000–22000 | ~1.4 | ~1.4 |

- train ≈ val → no overfitting (TinyStories is far larger than what this model can memorize).
- Stopped manually near step 21000; generation's `load_checkpoint` reported **iteration = 22001**.

#### Recall questions
1. **Why is step-0 loss about 9.2?**
   <details><summary>Answer</summary>With random init the model is roughly uniform over the vocab, so cross-entropy ≈ ln(vocab_size)=ln(10000)≈9.21.</details>
2. **What does train loss staying close to val loss mean?**
   <details><summary>Answer</summary>No clear overfitting; capacity is small relative to the data.</details>

### §6 Text generation (`cs336_basics/generate.py`)

#### Background / motivation
- A trained LM outputs **logits** (unnormalized scores) per position and vocab item. To write a story, turn the last position's logits into a next token id, append it to the prompt, and loop until `<|endoftext|>` or `max_new_tokens`.
- **Greedy decoding**: always `argmax` — same prompt always yields the same continuation, and loops are common.
- **Sampling**: softmax → probabilities, then `torch.multinomial`; **temperature** and **top-p / nucleus sampling** control how wild the text can get.

#### Formulas
- Softmax: \(\mathrm{softmax}(z)_i = e^{z_i}/\sum_j e^{z_j}\).
- Temperature: \(z \leftarrow z / \tau\) before softmax.
  - \(\tau \to 0\): sharper, ≈ greedy.
  - \(\tau \to \infty\): flatter, ≈ uniform.
- Top-p: sort probs descending, keep adding until cumulative mass reaches p, zero the rest, renormalize, then sample. Keep tokens while \((\mathrm{cumsum} - \mathrm{sorted\_probs}) < p\), so the **first token that pushes the cumulative sum over p is still kept**.

#### Inputs / outputs / shapes
- `generate(model, tokenizer, prompt, max_new_tokens, temperature, top_p=None, device=None) -> str`
- Each step: `window = ids[-context_length:]` → `x` shape `(1, S)` → `logits = model(x)[0, -1, :]` shape `(vocab_size,)` → scalar id, `ids.append`.
- Loading weights: build `TransformerLM` + a dummy `AdamW`, then `load_checkpoint` (it needs an optimizer shell); wrap inference in `@torch.no_grad()`.

#### Implementation steps
1. `ids = tokenizer.encode(prompt)` (instance method, not `Tokenizer.encode`).
2. Loop up to `max_new_tokens`: truncate window → forward → take last-position logits.
3. If `temperature <= 0`: greedy `argmax`; else `logits /= temperature`, then softmax.
4. If top-p is on: `sort` → `cumsum` → `mask = (cumsum - sorted_probs) < top_p` → zero → renormalize → `multinomial` → map back with `indices[...]`.
5. Stop if id == 0 (eot); else append. After the loop, `tokenizer.decode(ids)`.

#### Small numeric example
- logits `[2, 1, 0]`, \(\tau=1\): softmax ≈ `[0.67, 0.24, 0.09]`.
- \(\tau=0.1\): ≈ `[0.99995, 0.00005, …]`, almost always the first id.
- \(\tau=10\): ≈ `[0.37, 0.33, 0.30]`, nearly flat.
- Probs `[0.5, 0.3, 0.15, 0.05]`, p=0.9: cumsum 0.5→0.8→0.95, keep the first three; `cumsum - sorted` is `[0, 0.5, 0.8, 0.95]`, mask True on the first three.

#### Common mistakes
1. **`Tokenizer.encode(prompt)`**: that is the class; use the instance `tokenizer.encode`.
2. **`return` indented inside the `for`**: returned after the first token.
3. **`next_id` still a 0-d tensor**: breaks `ids.append` / comparisons — cast with `int(...)`.
4. **Else branch (no top-p) used `argmax`**: when `temperature>0` without top-p you still want `multinomial(softmax(...))`.
5. **Caller passed `top_k=` while the signature has `top_p`**: unexpected keyword.
6. **Thought `load_checkpoint` returned a dict**: it returns an **int** (iteration).
7. **Trailing space on the prompt**: BPE leaves a lone space token; the model often samples number-like merges (`8`, `8ic`). That is real generation, not a decode bug — keep prompts on full word boundaries.

#### Things worth noting
1. An empty prompt starting with `<|endoftext|>` makes the model open a new story, matching the training separator.
2. Greedy often starts with high-frequency names like Lily/Sue; sampling varies names and plots.
3. A thin prompt like “The little girl was” tends toward short or loopy continuations.

#### Recall questions
1. **What happens to the distribution when temperature is very small / very large?**
   <details><summary>Answer</summary>Small: peaked, near-greedy. Large: flat, near-uniform.</details>
2. **Why is the top-p mask `(cumsum - sorted_probs) < p`?**
   <details><summary>Answer</summary>That is the cumulative mass *before* adding the current token, so the first token that crosses p is still kept.</details>
3. **Without top-p, why softmax then multinomial instead of uniform over the vocab?**
   <details><summary>Answer</summary>multinomial needs non-negative probs that sum to 1; uniform sampling ignores the learned distribution — high-logit tokens should be more likely.</details>
4. **Is a trailing-space prompt that yields `8` a decode bug?**
   <details><summary>Answer</summary>No. The lone space token was encoded, and the model sampled a number-related token from the training distribution.</details>

#### Next
- Optional: LR sweep (`lr_max` in {3e-4, 1e-3, 3e-3}) and compare curves / final val loss.
- Optional: finish the remaining steps and see whether val loss drops further from ~1.4.
- If the handout still has OpenWebText experiments, those are next; before starting A2, check the career analysis for priority.
- Journals and code comments remain largely uncommitted; commit separately when wanted — this turn does not commit or push.
