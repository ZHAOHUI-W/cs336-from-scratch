# Week 1 journal

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

### 中文版

#### 做了什么
- 实现 `cs336_basics/train_bpe.py`，经 `tests/adapters.py` 的 `run_train_bpe` 接入测试。
- 已通过：`test_train_bpe`、`test_train_bpe_speed`（约 0.47s，小于 1.5s）、`test_train_bpe_special_tokens`。

#### 朴素 BPE（先正确性）

流程：

1. 读入语料（UTF-8；不要用 `errors='ignore'`，会静默丢字节、改变 merge 统计）。
2. 用 `special_tokens` 切分（多个时先长后短）。special **不进入** pre-tokenization，也 **不进入** pair 统计；但仍要写入 `vocab`。
3. 对每个 chunk，用 GPT-2 的 `PAT`（`regex.finditer`）做 pre-tokenization。
4. 每个 pre-token → UTF-8 bytes → 词频表 `dict[tuple[bytes, ...], int]`（元组元素是 bytes 片段，不是 vocab 的 int id）。
5. 初始化 `vocab = specials + 256` 个单字节；循环直到 `len(vocab) == vocab_size`：
   - 统计（或维护）所有相邻 pair 的频率；
   - 选频率最高的 pair；并列时取字典序 **更大** 的 pair；
   - `merges.append(pair)`；
   - 在词序列里把该相邻 pair 替换为 `C = A + B`（bytes 拼接）；
   - 每个 merge 只向 `vocab` **加入一次** `C`；
6. 返回 `(vocab, merges)`。

#### 优化（正确性过后再做）

**瓶颈（cProfile）：** 每次 merge 都整表重扫词频、重数 pair，代价随「词条数 × 平均长度 × merge 次数」上涨。在 `corpus.en` 上 pre-tokenization 不是大头。

**改动：**
- 跨轮维护 `pair_freq: pair → count`。
- 维护 `pair_to_words: pair → 含该 pair 的 word 集合`。选出 `best_pair=(A,B)` 后，只处理 `pair_to_words[(A,B)]` 里的词，其余跳过。
- 对每个受影响词：从 `pair_freq` / 索引中扣掉旧贡献 → 左到右、不重叠地应用 merge → 写回新词 → 把新 pair 加回 `pair_freq` / 索引。
- **原地** 更新 `freq`（`pop` 旧词再写入新词），避免每轮整表 `dict` 拷贝。

#### 踩坑
- 序列与 merges 必须始终是 **`bytes`**，不要把 vocab 的 int id 混进 word tuple。
- `vocab[len(vocab)] = C` 必须在「按词循环」**之外**；若放进内层，vocab 会按受影响词数量暴涨，merge 只做约 2 次就触顶（参考约 243 次）。
- `pair_freq` 与 `pair_to_words` 必须同步，否则会出现 `freq.pop` 或 `pair_to_words.pop(best_pair)` 的 `KeyError`。
- `for` 循环缩进错误时，若循环体只剩 `freq.pop`，其余词会半更新，索引彻底脏掉。

#### 架构收获
索引是算法契约的一部分：`pair_to_words` 不是外挂微优化，它定义了「这一轮允许改哪些词」。正确性 = 与全表重数得到相同 merges；速度 = 只碰这些词并保持计数一致。

#### 下一步
- 若仍在写调试用 `merges.txt`，关掉或删除。
- 实现 tokenizer 的 **encode / decode**（`get_tokenizer`）并过 `tests/test_tokenizer.py`。

---

### English

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


