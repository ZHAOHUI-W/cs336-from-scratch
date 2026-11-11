# Week 1 日志（中文）

英文版见 [week-01.en.md](./week-01.en.md)。

## Day 1 — Map A1 (2026-09-16)

> 本日笔记原为英文规划，详见同目录 `week-01.en.md` 对应章节。

## Day — BPE `train_bpe` (2026-09-21)

#### 做了什么
- 实现 `cs336_basics/train_bpe.py`，经 `tests/adapters.py` 的 `run_train_bpe` 接入测试。
- 已通过：`test_train_bpe`、`test_train_bpe_speed`（约 0.47s，小于 1.5s）、`test_train_bpe_special_tokens`。

#### 背景 / 动机
- 模型只认整数 id，文本要先切成 token。三种切法的取舍：
  - 按字符 / 字节：词表很小（256），但序列太长，注意力的计算量随长度平方增长。
  - 按整词：序列短，但词表爆炸，遇到没见过的词（OOV，out-of-vocabulary，词表外词）就没法编码。
  - **BPE（Byte Pair Encoding，字节对编码）**：从 256 个单字节出发，反复把语料里最常见的相邻对合并成新 token。常见词最后变成 1 个 token，罕见词拆成几段；因为底层是字节，任何字符串都能编码，没有 OOV。
- 在流水线里的位置：原始文本 → **`train_bpe` 学出 vocab 和 merges（只在训练前跑一次）** → Tokenizer 把文本编码成 id → 存成 uint16 数组 → `get_batch` 取 batch → 模型。

#### 公式 / 规则
- 每一轮选频率最高的相邻对：

  $$(A,B)^* = \arg\max_{(A,B)} \sum_{w} \text{freq}(w)\cdot \#_{(A,B)}(w)$$

  其中 $w$ 是一个 pre-token（预分词片段）的字节序列，$\text{freq}(w)$ 是它在语料里出现的次数，$\#_{(A,B)}(w)$ 是 $A,B$ 在 $w$ 里相邻出现的次数。频率并列时取字典序更大的那一对。
- 合并成新 token $C = A + B$（bytes 拼接），写进 vocab，记进 merges。
- 一共要做的 merge 次数 = `vocab_size − 256 − len(special_tokens)`。例如 vocab_size=500、1 个 special：500 − 256 − 1 = 243 次。

#### 输入 / 输出 / 形状
```python
train_bpe(input_path: str, vocab_size: int, special_tokens: list[str])
    -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]
```
- `input_path`：语料文件路径；`vocab_size`：目标词表大小（包含 special token 和 256 个单字节）；`special_tokens`：如 `["<|endoftext|>"]`，写进 vocab，但不参与统计。
- 返回 `vocab: dict[int, bytes]`（id → 字节串）和 `merges: list[tuple[bytes, bytes]]`（按学到的先后顺序）。
- 内部数据结构：`freq: dict[tuple[bytes, ...], int]`（词频表）、`pair_freq: dict[tuple[bytes, bytes], int]`、`pair_to_words: dict[pair, set[word]]`。
- 没有张量，也没有可学习参数。

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

#### 实现位置与测试
- 文件：`cs336_basics/train_bpe.py`（朴素 + 增量版）、`train_bpe_native.py`、`train_bpe_multiprocessing.py`、`train_bpe_multiprocessing_buckets.py`。现在 `tests/adapters.py` 的 `run_train_bpe` 导入的是分桶版 `train_bpe_multiprocessing_buckets.train_bpe`。
- 增量版骨架：
```python
vocab = {specials...} | {bytes([b]) for b in range(256)}
freq = 统计每个 pre-token 的字节元组出现次数
pair_freq, pair_to_words = 初始统计
while len(vocab) < vocab_size:
    best = max(pair_freq, key=lambda p: (pair_freq[p], p))   # 频率最高，并列取字典序大
    merges.append(best); vocab[len(vocab)] = best[0] + best[1]   # 每轮只加一次
    for w in list(pair_to_words[best]):                     # 只处理含 best 的词
        扣掉 w 的旧 pair 计数 → 合并 → 加回新词的 pair 计数 → freq 里换成新词
```
- 通过：`test_train_bpe`、`test_train_bpe_speed`、`test_train_bpe_special_tokens`。

#### 小数值例子
词频（已按字节拆开）：`low`×5、`lower`×2、`newest`×6、`widest`×3。

| 相邻对 | 计数 |
|---|---|
| (e, s) | 6 + 3 = 9 |
| (s, t) | 6 + 3 = 9 |
| (w, e) | 2 + 6 = 8 |
| (l, o) | 5 + 2 = 7 |
| (o, w) | 5 + 2 = 7 |
| (n, e)、(e, w) | 6 |

- 第 1 轮：(e, s) 和 (s, t) 并列 9，取字典序更大的 (s, t)，得到 `st`。`newest` 变成 n e w e **st**，`widest` 变成 w i d e **st**。
- 第 2 轮：新出现的 (e, st) 计数 9，最高，合并成 `est`。这一对在第 1 轮之前根本不存在，这就是 merges 必须有序的原因。

#### 值得注意的点
1. **byte-level 没有 OOV**：任何 Unicode 字符都能拆成 1~4 个 UTF-8 字节，最差情况就是逐字节编码。
2. **special token 先切出来**：否则 `<|endoftext|>` 会被 PAT 切碎，还可能和两边的字节合并，跨越文档边界。
3. **PAT 预分词限制了 merge 的范围**：merge 只在一个 pre-token 内部发生，例如 `dog!` 会先切成 `dog` 和 `!`，所以不会合并出 `g!` 这种跨类别的 token。
4. **副作用**：2026-11-03 加注释时发现，几个 train_bpe 版本每次运行都会在当前目录覆盖写 `merges.txt`（已用 `# 注意` 标出）。

#### 回忆题
1. **vocab_size=500、只有 1 个 special token，要做多少次 merge？**
   <details><summary>答案</summary>500 − 256 − 1 = 243 次。</details>
2. **频率并列时取哪个 pair？为什么一定要有规则？**
   <details><summary>答案</summary>取字典序更大的。有确定的规则，结果才可复现，才能和参考实现逐条对齐。</details>
3. **为什么只更新 `pair_to_words[best]` 里的词，结果仍和全表重数一样？**
   <details><summary>答案</summary>不含 best 的词这一轮不会变，它们对 pair_freq 的贡献也不变，只有含 best 的词需要扣旧计数、加新计数。</details>
4. **`vocab[len(vocab)] = C` 放进按词循环里会怎样？**
   <details><summary>答案</summary>每个受影响的词都加一次，vocab 暴涨，大约 2 次 merge 就满了（参考是 243 次）。</details>

#### 下一步
- 若仍在写调试用 `merges.txt`，关掉或删除。
- 实现 tokenizer 的 **encode / decode**（`get_tokenizer`）并过 `tests/test_tokenizer.py`。

## Day — Tokenizer encode/decode（2026-09-22）

#### 做了什么
- 实现 `cs336_basics/tokenizer.py` 的 `Tokenizer`：`__init__`、`encode`、`decode`、`encode_iterable`。
- 在 `tests/adapters.py` 的 `get_tokenizer` 中直接返回 `Tokenizer(vocab, merges, special_tokens)`。
- Windows 上为 `tests/test_tokenizer.py` 做了 `resource` 可选导入（无 `RLIMIT_AS` 时跳过硬内存上限）。
- 补齐 GPT-2 fixture（`gpt2_vocab.json` / `gpt2_merges.txt`），用于与 tiktoken 对齐的测例。

#### 背景 / 动机
- `train_bpe` 只负责"学规则"；真正训练和推理时，要用 Tokenizer 把任意文本编码成 id（encode），再把模型生成的 id 解码回文本（decode）。
- 要求：encode 后再 decode 必须还原原文（roundtrip，往返一致）；切分要和训练时完全一致，否则 id 的含义就错了。
- 流水线位置：语料 → Tokenizer.encode → 一长串 id（之后存成 uint16 文件给 `get_batch` 用）；生成时 id → decode → 文本。

#### 公式 / 规则
- encode：`text → 按 special 切段 → PAT 预分词 → UTF-8 字节 → 按 merges 顺序依次合并 → 查 bytes_to_id`。
- decode：$\text{text} = \text{utf8\_decode}\big(\text{vocab}[i_1] \,\|\, \text{vocab}[i_2] \,\|\, \cdots\big)$，$\|$ 表示字节拼接。先拼接再解码，一个多字节字符被拆在两个 token 里也能还原。

#### 输入 / 输出 / 形状
```python
Tokenizer(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None)
encode(text: str) -> list[int]
decode(ids: list[int]) -> str
encode_iterable(iterable: Iterable[str]) -> Iterator[int]
```
- 内部属性：`bytes_to_id`（vocab 的反向表）、`special_pattern`（按长度降序编译的正则）、`pre_token_re`（GPT-2 PAT）。
- 没有张量，没有可学习参数。

#### 与 `train_bpe` 的契约
- `train_bpe` 产出：`vocab: dict[int, bytes]`，`merges: list[tuple[bytes, bytes]]`（**按训练时的先后顺序**）。
- Tokenizer **消费**这两样做编解码；不再学新 merge。
- Special tokens 在训练时写入 vocab；编码时整段匹配进 id，**不**进入 BPE merge。

#### encode（text → ids）
1. **Special 切分**：按 special 字符串匹配（长的优先）；命中段直接 `bytes_to_id`；未命中的普通段进入下一步。
2. **GPT-2 PAT pre-tokenization**：对每个普通段用与 `train_bpe` 相同的 regex 切成 pre-token。
3. **BPE**：每个 pre-token → UTF-8 → 初始单字节序列；再 **按 `merges` 列表顺序** 把相邻 pair 合并成更长 bytes。
4. **查表**：合并结果中每个 bytes 片段 → token id。

#### decode / encode_iterable
- `decode`：id → vocab 中的 bytes，拼接后再 UTF-8 解码（坏序列可用 `errors="replace"`）。
- `encode_iterable`：对字符串流逐块 `encode`，再逐个 yield id（避免整文件进内存）。

#### 踩坑
- 只做「UTF-8 单字节查表」时：roundtrip 能绿，但与 tiktoken 的 id 对不齐（词被拆成字母级）。
- `merges` 必须是有序列表，按顺序应用，不能当成「再选一次最高频 pair」。
- `adapters.get_tokenizer` 不要嵌套同名函数导致返回 `None`；删掉重复的 `decode` / `encode_iterable`。

#### 为什么 merges 必须是有序列表
训练时学到的是「先合谁、后合谁」这条规则序列，编码必须按同一顺序重放，切分才和训练一致。

每一轮选出的 pair，都是在前面那些 merge 已经发生之后的语料上统计出来的。例如先有 `(e, s)→es`，后面才可能出现 `(es, t)→est`；没有先合出 `es`，`(es, t)` 这个 pair 根本不存在。因此 merges 不是一袋互不相关的规则，而是有依赖关系的步骤列表：下标越小，越早学到，编码时也越先应用。

若编码时再按「当前词里频率最高」选 pair，或把 merges 当成无序集合乱序应用，切分会和训练不同，token id 也对不齐。`list[tuple[bytes, bytes]]` 就是在保存这份顺序；`dict` / `set` 会丢掉先后。

一句话：顺序是 tokenizer 的一部分；丢掉顺序等于换了一套 tokenizer。

#### 实现位置
- 文件：`cs336_basics/tokenizer.py`。`encode` 先按 `special_pattern.finditer` 交替处理普通段和 special 段；普通段交给 `_encode_normal_text`，里面对每个 pre-token 逐条调用 `_apply_single_merge`。
- adapter：`tests/adapters.py` 的 `get_tokenizer` 直接 `return Tokenizer(vocab, merges, special_tokens)`。

#### 小数值例子
- 设 merges = `[(s, t), (e, st)]`，编码 `"west"`：
  1. 字节序列：`w e s t`
  2. 应用第 1 条 (s, t)：`w e st`
  3. 应用第 2 条 (e, st)：`w est`
  4. 查表得到 2 个 id。如果把顺序反过来，先找 (e, st) 找不到，再合 (s, t)，最后得到 `w e st`，是 3 个 token，和训练不一致。
- special：`"hi<|endoftext|>"` 先切成 `"hi"` 和 `"<|endoftext|>"`，后者整体查表得到 1 个 id，不进 BPE。
- 重叠的 special：同时有 `<|endoftext|>` 和 `<|endoftext|><|endoftext|>` 时，按长度降序匹配，后者才不会被拆成两个（`test_overlapping_special_tokens`）。

#### 值得注意的点
1. **special token 必须已经在 vocab 里**，否则 `bytes_to_id[...]` 会 KeyError（2026-11-03 注释时标了 `# 注意`）。
2. **没有缓存**：同一个词每次出现都要把所有 merges 重放一遍，大语料上很慢。常见的优化是给 pre-token 加 dict 缓存，或者用 merge 的 rank 每次只合并 rank 最小的那一对。
3. **`encode_iterable` 按块独立编码**：块与块之间不会合并。按行读文件时每行末尾带换行，一般没问题。
4. tokenizer.py 第一行"只按单字节查表"的旧注释已经过时，当时也用 `# 注意` 标出来了。

#### 验收
- `tests/test_tokenizer.py` 中 roundtrip 与 tiktoken 对齐相关用例已通过。

#### 回忆题
1. **decode 为什么要先把所有 bytes 拼起来再解码，而不是逐个 token 解码？**
   <details><summary>答案</summary>一个多字节字符（例如中文 3 个字节）可能被拆在两个 token 里，单独解码某一段会得到乱码。拼起来再整体解码才能还原。</details>
2. **多个 special token 为什么要按长度从长到短匹配？**
   <details><summary>答案</summary>短的会抢先匹配长的一部分，例如 `<|endoftext|><|endoftext|>` 会被拆成两个 `<|endoftext|>`。</details>
3. **只做单字节查表时，为什么 roundtrip 测试能过，tiktoken 对齐测试过不了？**
   <details><summary>答案</summary>单字节也能完整还原文本，但不做 merge，每个词都被拆成字母级 token，id 序列和 tiktoken 完全不同。</details>

#### 下一步
- TinyStories 全量 `train_bpe` 实验（压缩率等），或进入 A1 Transformer。

## Day — TinyStories `train_bpe` 实验（2026-09-22）

#### 做了什么
- Handout **Problem (train_bpe_tinystories)**：在 `TinyStoriesV2-GPT4-train.txt` 上训 byte-level BPE。
- 配置：`vocab_size=10000`，`special_tokens=["<|endoftext|>"]`。
- 脚本：`scripts/train_tinystories_bpe.py`；实现演进：
  1. 串行 / 增量 merge 的 `train_bpe`
  2. `train_bpe_multiprocessing.py`：`find_chunk_boundaries` + `Pool` 并行 pretokenize
  3. `train_bpe_multiprocessing_buckets.py`：`PairFrequencyBuckets` 加速选最优 pair
- 产物（不进 git）：`artifacts/tinystories_bpe/` 下的 `vocab_*.txt` / `merges_*.txt`。

#### 背景 / 动机
- handout 要求在真实数据上训一次 BPE：看完整训练要多久、占多少内存，再检查学出来的 token 是否合理。
- 为什么要优化：TinyStories 训练集约 2 GB，朴素实现在这个规模下太慢。瓶颈有两处：pre-tokenization（预分词，正则切词）和每一轮选最高频的 pair。
- 产出的 vocab 和 merges 就是后面 TinyStories 语言模型要用的 tokenizer，vocab_size=10000 也决定了 token id 可以用 uint16 存。

#### 输入 / 输出
- 输入：`TinyStoriesV2-GPT4-train.txt`，`vocab_size=10000`，`special_tokens=["<|endoftext|>"]`，`num_workers=4`。
- 输出：`vocab_*.txt` 和 `merges_*.txt`，放在 `artifacts/tinystories_bpe/` 下，不进 git；同时记录耗时、峰值内存和最长 token。
- 分桶结构 `PairFrequencyBuckets`：`counts: pair → freq`，`buckets: freq → set(pairs)`，接口有 `update(...)`、`best_pair()` 和 `__bool__`。

#### 全量结果（vocab_size=10000，num_workers=4，train）

- **分桶版**（`train_bpe_multiprocessing_buckets`）：Wall time **404.75 s（约 6.7 min）**；峰值内存（含 children）**4610.33 MB（约 4.61 GB）**；最长 token `b' accomplishment'`（len 15）。
- **对照（同分进程、全体 `max` 选 pair）**：约 **450 s / 4.6 GB**，最长 token 相同。
- 分桶相对对照约快 **45 s（约 10%）**；内存同量级。仍低于硬上限 30 min / 30 GB，未到 hint 的 2 min 以内。
- 正确性：`merges_train_process4.txt` 与 `merges_train_process4_buckets.txt` 一致；期望 `len(vocab)=10000`，`len(merges)=9743`。
#### 多进程 pretokenize
- 用官方 `find_chunk_boundaries`，按 `<|endoftext|>` 对齐切块；worker 顶层函数 `_count_chunk`（Windows spawn）。
- `num_workers=4` 为默认尝试值；valid 上可从 ~30s 降到 ~20s 量级。
- 注意：主进程 `peak_wset` **不含** worker；要用周期性 `rss(parent)+rss(children)` 才接近真实峰值。勿把「字节数」误标成 MB。

#### Profile（handout (b)）
- `python -m cProfile` **只看主进程**：Pool 的 wait/terminate 会显得很大，那是在等子进程，不是 Pool「算得慢」。
- 主进程可见热点：每轮 `max(pair_freq)` 选 merge（上亿次 lambda）。
- 更稳的拆分：对 `_build_frequency_table` vs merge 循环分别 `perf_counter`；或 `num_workers=1` 再 profile。

#### 分桶优化
- `PairFrequencyBuckets`：`counts[pair]→freq`，`buckets[freq]→set(pairs)`；更新时搬桶；`best_pair` 在最高非空桶内 `max(bucket)`（同频字典序最大）。
- 正确性：全量产物 `merges_train_process4.txt` 与 `merges_train_process4_buckets.txt` **二进制一致**（`fc` 无差异）。
- 复杂度：update 近似 O(1)；`best_pair` 是 O(该频桶大小)，不是严格 O(1)。后期大量 pair 频次为 1 时，桶内 `max` 仍可能偏贵；主要加速前中期「避免扫全体 pair」。

#### Handout 答句草稿
- **(a)** 全量 TinyStories、vocab 10k、4 进程 pretokenize，约 **6.7 分钟**、峰值约 **4.61 GB**（分桶版）；最长 token 为 `b' accomplishment'`，符合带空格的常见英文词干被合并的现象。
- **(b)** cProfile 显示主进程瓶颈在反复 `max` 选 pair；pretokenize 在子进程中，主进程表现为等待 Pool。分桶后消除「每轮扫全体 pair」；应用分段计时区分 pretok vs merge。

#### 小数值例子（分桶）
- 计数 `{(s,t): 9, (e,s): 9, (l,o): 7}`，对应的桶是 `{9: {(s,t), (e,s)}, 7: {(l,o)}}`。`best_pair` 从最高的非空桶 9 里取字典序最大的，得到 (s, t)。
- 合并 `st` 之后 (e, s) 降到 0，要从桶 9 里移走；新出现的 (e, st)=9 放进桶 9。每次更新只是把一个 pair 从一个桶搬到另一个桶，不用扫描全部 pair。

#### 常见错误
1. **只看主进程的峰值内存**：`peak_wset` 不包含 worker 子进程，要周期性地采样 `rss(父进程) + rss(所有子进程)`。
2. **单位标错**：把字节数直接标成 MB。
3. **误读 cProfile**：cProfile 只统计主进程，Pool 的 wait/terminate 显得很大，其实是在等子进程算完。
4. **Windows 下 worker 函数要写在模块顶层**（`_count_chunk`）：spawn 方式启动子进程时要能 pickle 它。

#### 回忆题
1. **为什么用 cProfile 看多进程程序会误判瓶颈？**
   <details><summary>答案</summary>它只看主进程，主进程大部分时间在等 Pool，真正做 pretokenize 的是子进程。</details>
2. **分桶之后选 best_pair 是严格 O(1) 吗？**
   <details><summary>答案</summary>不是。是 O(最高频桶的大小)。训练后期大量 pair 频次相同（比如都是 1）时，桶内取 max 仍然偏贵。</details>
3. **怎么验证分桶版是对的？**
   <details><summary>答案</summary>和不分桶版在同一份数据上跑，比较 merges 文件是否逐字节一致（`fc` 无差异），并检查 `len(merges)=9743`。</details>

#### 下一步
- 若追 &lt;2 min hint：分段计时 + 视情况再优 merge / worker 数。
- Handout 下一题：`train_bpe_expts_owt`（vocab 32k），或进入 Transformer。


## Day — Linear 与 Embedding（2026-10-12 ~ 2026-10-23）

#### 做了什么
- §3.3.2 Linear（无 bias）：`cs336_basics/linear.py` + `adapters.run_linear`。
- §3.3.3 Embedding：`cs336_basics/embedding.py` + `adapters.run_embedding`。
- 修测试收集报错：`pretokenization_example.py` 的演示代码放进 `if __name__ == "__main__":`，避免 pytest 导入时执行 `open(...)`。

### Linear（模块全貌）

#### 背景 / 动机
- Transformer 里几乎所有可学习的计算都是线性变换：Q、K、V、O 四个投影，FFN 的三个矩阵，最后的 lm_head。先把它写成一个通用模块，后面都复用它。
- 现代 LLM（LLaMA 等）的线性层一般不带 bias，所以这里实现的是 `LinearNoBias`。

#### 公式
- 单个向量：$y = Wx$，$W \in \mathbb{R}^{d_{out}\times d_{in}}$。
- 批量的行向量：$Y = X W^\top$，$X$ 的形状是 $(\dots, d_{in})$。
- 初始化：$W_{ij} \sim \mathcal{N}(0, \sigma^2)$，$\sigma^2 = \frac{2}{d_{in}+d_{out}}$，截断在 $[-3\sigma, 3\sigma]$。

#### 输入 / 输出 / 形状
- `LinearNoBias(d_in, d_out, device=None, dtype=None)`
- 可学习参数：`weight`，形状 `(d_out, d_in)`，用 `nn.init.trunc_normal_` 初始化。
- `forward(x)`：`(..., d_in)` → `(..., d_out)`，前面的 batch 维可以任意多。

#### 实现步骤
```python
self.weight = nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype))
std = math.sqrt(2 / (d_in + d_out))
nn.init.trunc_normal_(self.weight, std=std, a=-3*std, b=3*std)
# forward
return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")
```
- 文件 `cs336_basics/linear.py`；adapter `run_linear` 建模块后执行 `linear.weight.data = weights`；`test_linear` PASSED。

#### 小数值例子
- $d_{in}=2$，$d_{out}=3$，$W = \begin{bmatrix}1&2\\0&1\\1&0\end{bmatrix}$，$x=[3,4]$ → $y = [1\cdot3+2\cdot4,\ 4,\ 3] = [11, 4, 3]$。
- 初始化：$d_{in}=d_{out}=512$ 时 $\sigma=\sqrt{2/1024}\approx0.0442$，截断范围约 ±0.133。

### Embedding（模块全貌）

#### 背景 / 动机
- token id 只是类别编号，数值大小没有意义（id 5 并不比 id 2"大"），不能直接当数字喂给网络。要给每个 id 配一个可学习的 d_model 维向量。
- 它是模型的第一层：`(B, S)` 的整数 → `(B, S, d_model)` 的向量。

#### 公式
- 表 $E \in \mathbb{R}^{V\times d}$，$\text{emb}(i) = E[i]$，也就是第 $i$ 行。
- 等价于 $\text{onehot}(i)\cdot E$，但直接查表不用做乘法，快得多。
- 初始化：$\mathcal{N}(0, 1)$，截断在 $[-3, 3]$。

#### 输入 / 输出 / 形状
- `EmbeddingLayer(num_embeddings, embedding_dim, device=None, dtype=None)`
- 可学习参数：`weight`，形状 `(vocab_size, d_model)`。
- `forward(token_ids)`：long 型的 `(B, S)` → `(B, S, d_model)`。

#### 实现步骤
- `return self.weight[token_ids]`，用高级索引一次取出所有行。
- 文件 `cs336_basics/embedding.py`；adapter `run_embedding`；`test_embedding` PASSED。

#### 小数值例子
- V=4，d=2，$E = [[0,0],[1,2],[3,4],[5,6]]$，ids = `[[2, 1, 2]]`，形状 (1, 3)。
- 输出 `[[[3,4],[1,2],[3,4]]]`，形状 (1, 3, 2)。位置 0 和位置 2 都是 token 2，向量完全一样，Embedding 本身不知道位置。

### 本节共用：易混概念、验收与回忆

#### 容易弄混的概念
1. **权重形状**：Linear 存 `(d_out, d_in)`，前向是 `y = x Wᵀ`；Embedding 存 `(vocab_size, d_model)`，第 i 行就是 token i 的向量。两者不是同一种操作。
2. **Embedding 的 forward 是查表，不是矩阵乘法**：`self.weight[token_ids]`。用 einsum 会把 token id 当成数值去加权求和，结果是错的。输出形状 = `token_ids` 的形状 + `d_model`，例如 `(1, 3)` 进、`(1, 3, d_model)` 出。
3. **圆括号 vs 方括号**：`self.weight(token_ids)` 是把 Parameter 当函数调用，会报 `'Parameter' object is not callable`；取行要用方括号索引。
4. **einops.einsum 参数顺序**：先传张量，pattern 字符串放最后。
5. **初始化**：Linear 用截断正态，σ² = 2/(d_in+d_out)，截断在 ±3σ，目的是让前向和反向的信号尺度大致不变（输入越宽，σ 越小）。Embedding 用 N(0,1) 截断到 [-3, 3]。Embedding 只是查表，不会对 d_in 个输入求和，所以没有"方差随宽度累积"的问题，std=1 就行。
6. **位置信息**：同一个 token 在不同位置拿到的 embedding 完全相同（索引本身不区分位置）。A1 里位置信息靠后面注意力中的 RoPE（旋转位置编码）加入。
7. **Embedding 表要训练**：它是 `nn.Parameter`，和其他参数一起被优化器更新；每一步只有 batch 里出现过的 token 那几行拿到梯度。Embedding 和 LM head 是两套独立参数（A1 不做 weight tying）。
8. **adapter 的作用**：测试只调用 `adapters.run_*`；adapter 负责建你的模块、把测试给的权重写进 `.data`、再调 `forward`。
9. **测试依赖**：缺 `tests/fixtures/ts_tests/model.pt` 时先跑 `scripts/fetch_large_assets.sh`。
10. **参数命名**：Embedding 参数命名为 `weight`（不是 `weights`），后面整模型 `load_state_dict` 需要名字对上。

#### 验收
- `uv run pytest -k test_linear`：PASSED。
- `uv run pytest -k test_embedding`：PASSED。

#### 下一步
- §3.4 RMSNorm。
- 小清理：`embedding.py` 中未使用的 `math` / `einsum` import；`run_embedding` 里 `return` 之后多余的 `raise NotImplementedError`。
#### 回忆题
1. **同一个 token 出现在序列的不同位置，Embedding 输出的向量一样吗？模型靠什么区分位置？**
   - 我的答案：一样；靠索引区分。
   - 订正：向量一样 ✅。但索引只选「哪个 token」，不管它排第几位，所以 Embedding 层完全不知道位置。A1 里位置信息靠注意力中的 RoPE（旋转位置编码）：按位置旋转 Q 和 K，同一个词在第 1 位和第 5 位算出的注意力分数不同。
2. **为什么 Embedding 初始化用 std=1，而 Linear 要用 √(2/(d_in+d_out))？**
   - 我的答案：让信号在前向和反向传播时尺度大致保持不变，输入越宽，σ 越小。
   - 订正：Linear 部分对 ✅。补 Embedding 部分：Linear 的输出要把 d_in 个「输入×权重」加起来，项越多方差越大，所以权重按宽度缩小；Embedding 只是取出一行，不做求和，方差不随宽度变大，所以每个数直接用单位尺度（std=1）。
3. **Linear 的权重为什么存成 (d_out, d_in)，前向写 x Wᵀ？**
   <details><summary>答案</summary>和 PyTorch `nn.Linear` 的约定一致，测试给的权重也是这个形状。每一行是一个输出神经元的权重，x 乘 Wᵀ 就是 x 和每一行做点积。</details>
4. **`onehot(i) @ E` 和 `E[i]` 结果一样吗？为什么用后者？**
   <details><summary>答案</summary>一样。后者不用构造 V 维的 one-hot 向量，也不用做 V×d 的乘法，只是取一行。</details>
5. **Embedding 输入 (2, 5) 的 id，d_model=8，输出什么形状？**
   <details><summary>答案</summary>(2, 5, 8)：输入形状后面加一维 d_model。</details>

## Day — RMSNorm 与 SwiGLU（2026-10-23 ~ 2026-10-24）

#### 做了什么
- §3.4.1 RMSNorm：`cs336_basics/rmsnorm.py` + `adapters.run_rmsnorm`。
- §3.4.2 SwiGLU 前馈网络：`cs336_basics/swiglu.py`（模块级 `silu` 函数 + `SwiGlu` 模块）+ `adapters.run_silu` / `adapters.run_swiglu`。
- 小清理：删掉 `embedding.py` 里没用到的 `math` / `einsum` import。

### RMSNorm（模块全貌）

#### 背景 / 动机
- 深层网络里，每层输出的数值尺度会漂移：太大会爆，太小会消失，训练不稳定。归一化把每个 token 的向量拉回统一的尺度。
- LLaMA 等模型用 RMSNorm（Root Mean Square Normalization，均方根归一化）代替 LayerNorm：不减均值、没有偏置，计算更省，效果差不多。
- 位置：每个 block 里有 ln1（attention 前）和 ln2（FFN 前），整个模型最后还有一个 ln_final（pre-norm 结构）。

#### 公式
$$\text{RMS}(a) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} a_i^2 + \varepsilon},\qquad \text{RMSNorm}(a)_i = \frac{a_i}{\text{RMS}(a)}\, g_i$$
- $a$：一个 token 的 d_model 维向量；$g$：可学习的逐维缩放（gain）；$\varepsilon = 10^{-5}$，防止除以 0。

#### 输入 / 输出 / 形状
- `RMSNorm(d_model, eps=1e-5, device=None, dtype=None)`
- 可学习参数：`weight`（就是 g），形状 `(d_model,)`，初始化为全 1。
- `forward(x)`：`(B, S, d_model)` → 形状和 dtype 都不变。

#### 实现步骤
```python
in_dtype = x.dtype
x = x.to(torch.float32)
rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)   # (B, S, 1)
return (x * self.weight / rms).to(in_dtype)
```
- 文件 `cs336_basics/rmsnorm.py`；adapter `run_rmsnorm`；`test_rmsnorm` PASSED。

#### 小数值例子
- $a=[3,4]$，$g=[1,1]$：平方的平均 = (9+16)/2 = 12.5，RMS ≈ 3.536，输出 ≈ [0.849, 1.131]。输出的平方平均正好是 1。
- 如果 $g=[2,1]$，输出 ≈ [1.697, 1.131]：g 让模型自己决定每一维放大多少。

### SwiGLU 与 SiLU（模块全貌）

#### 背景 / 动机
- attention 负责 token 之间交换信息；FFN（Feed-Forward Network，前馈网络）负责每个 token 自己加工信息，各个位置独立计算。模型的参数大头在 FFN。
- SwiGLU（Swish-Gated Linear Unit，SiLU 门控线性单元）是 LLaMA 等用的门控 FFN，实验上比 ReLU 版 FFN 效果更好。

#### 公式
$$\text{SiLU}(x) = x\cdot\sigma(x) = \frac{x}{1+e^{-x}}$$
$$\text{FFN}(x) = W_2\big(\text{SiLU}(W_1x)\odot W_3x\big)$$
- $W_1, W_3 \in \mathbb{R}^{d_{ff}\times d}$，$W_2 \in \mathbb{R}^{d\times d_{ff}}$，$\odot$ 是逐元素相乘，$d_{ff}\approx\frac{8}{3}d$。

#### 输入 / 输出 / 形状
- `silu(x)`：任意形状，逐元素计算，没有参数。
- `SwiGlu(d_model, d_ff, device=None, dtype=None)`，子模块 `w1`、`w3`（weight 形状 `(d_ff, d_model)`）和 `w2`（`(d_model, d_ff)`），都是 `LinearNoBias`，用截断正态初始化。
- `forward(x)`：`(B, S, d)` → w1、w3 得到 `(B, S, d_ff)` → 逐元素相乘 → w2 得到 `(B, S, d)`。

#### 实现步骤
- `return self.w2(silu(self.w1(x)) * self.w3(x))`
- 文件 `cs336_basics/swiglu.py`；adapter `run_silu`（直接调 `silu`）和 `run_swiglu`（`swiglu.w1.weight.data = w1_weight` 等）；`test_silu_matches_pytorch`、`test_swiglu` PASSED。

#### 小数值例子
- SiLU(2) = 2 × 0.881 = 1.762；SiLU(0) = 0；SiLU(−1) = −1 × 0.269 = −0.269；最小值约为 SiLU(−1.28) ≈ −0.278。
- 门控：d_ff=2，$W_1x=[2,-5]$，$W_3x=[3,3]$。SiLU 得到 [1.762, −0.033]，逐元素乘以 [3, 3] 得到 [5.29, −0.10]。第二维的门几乎关上了，内容 3 基本传不过去。
- d_model=512：8/3 × 512 ≈ 1365，取最近的 64 的倍数得到 1344，也就是 TinyStories 配置里的 d_ff。

### 本节共用：错误、要点、验收与回忆

#### 常犯的小错误
1. **`super.__init__()` 少了括号**：要写 `super().__init__()`。少了括号不会调用父类初始化，后面注册参数时就会出错。
2. **`device` / `dtype` 接收了却没往下传**：RMSNorm 和 SwiGLU 都犯过。`__init__` 收了这两个参数，就要传给 `torch.ones(...)` 或子模块 `LinearNoBias(..., device=device, dtype=dtype)`，不能写死成 `None`。
3. **笔误 `x ** x`**：平方是 `x ** 2`。这种错误不报错，只会算出错误结果。
4. **`math.*` 处理不了张量**：`math.sqrt`、`math.sigmoid`（后者根本不存在）只能算单个 Python 数字。张量要用 `torch.sqrt`、`torch.sigmoid`。
5. **adapter 传错参数**：`run_rmsnorm` 一开始把权重张量当成 `d_model` 传进去，`torch.ones()` 报 TypeError。先建模块（传维度），再装权重（传张量），两件事不要混在一起。
6. **对模块再写 einsum**：`self.w1` 是 `LinearNoBias` 模块，它的 forward 里已经做了 einsum，直接 `self.w1(x)` 调用就行。`silu.einsum(...)` 这种写法不成立。
7. **`@` 和 `*` 混用**：公式里的 ⊙ 是逐元素相乘，对应 `*`；`@` 是矩阵乘法。
8. **权重装错层级**：`swiglu.w1 = w1_weight` 是想把整个子模块换成张量；正确做法是 `swiglu.w1.weight.data = w1_weight`，装进子模块里的参数。

#### 值得注意的地方
1. **RMSNorm 公式**：`x / sqrt(mean(x²) + eps) * g`。和 LayerNorm 相比，不减均值，也没有偏置 β。eps 放在根号里面，防止除以 0。
2. **`keepdim=True`**：沿 `dim=-1`（d_model）求平均后要保留这一维，形状从 `(B, S, d)` 变成 `(B, S, 1)`，才能和 x 正确广播。漏了它，形状不匹配时会报错；但碰巧 `S == d_model` 时**不报错，结果却是错的**。
3. **先升到 float32 再降回来**：先记下 `in_dtype = x.dtype`，用 float32 计算平方和平均，避免低精度（如 bf16）下平方溢出或丢精度，最后 `.to(in_dtype)` 转回原类型。
4. **g 初始化为 1**：一开始不改变尺度，之后由训练去学。参数名用 `weight`，和后面 `load_state_dict` 的 key 对上。
5. **SiLU 和 ReLU 的区别**：SiLU(x) = x·σ(x)，处处平滑；负数区会给出小的负值（最低约 -0.28），梯度也不为 0。ReLU 在负数区输出和梯度都是 0。
6. **门控**：SiLU(W1x) 是门，W3x 是内容。门的某一维接近 0，那一维的输出就接近 0，不管内容是多少。门由输入算出来，所以每个 token 自己决定开关哪些维度。
7. **为什么 d_ff ≈ 8/3·d_model**：普通 FFN 有 2 个 d×4d 矩阵，合计 8d²；SwiGLU 有 3 个矩阵，令 3·d·d_ff = 8d²，得 d_ff = 8d/3，总参数量保持一样。再取到 64 的倍数，方便 GPU 计算。
8. **无参数的函数写在模块外面**：SiLU 没有可学习参数，不放进 `__init__`，写成模块级函数；`run_silu` 可以直接 import 来用。
9. **子模块命名 `w1` / `w2` / `w3`**：后面整模型加载权重时，key 会是 `...ffn.w1.weight` 这种格式，名字要对上。
10. **不报错 ≠ 算对了**：`keepdim` 漏掉、`@` 碰上 `S == d_ff`，都可能静默出错。所以每个模块都要有测试兜底。

#### 验收
- `uv run pytest -k test_rmsnorm`：PASSED。
- `uv run pytest -k "test_swiglu or test_silu"`：2 passed。

#### 回忆题
1. **漏了 `keepdim=True` 会怎样？**
   - 我的答案：会把求平均的那一维消掉。
   - 订正：对 ✅。后果是形状对不上时报错；碰巧 `S == d_model` 时不报错，但会按错误的维度广播，结果是错的。
2. **RMSNorm 和 LayerNorm 有什么区别？**
   - 我的答案：一个在进入 attention 前归一化，一个在出来的时候归一化？
   - 订正：这说的是 pre-norm 和 post-norm 的区别（norm 放在哪里），不是 RMSNorm 和 LayerNorm 的区别。RMSNorm 不减均值、没有偏置 β，只按 RMS 缩放；LayerNorm 先减均值再除以标准差，还有 β。
3. **`swiglu.w1 = w1_weight` 和 `swiglu.w1.weight.data = w1_weight` 有什么区别？**
   - 我的答案：一个是取模块，一个是取模块内矩阵的权重值。
   - 订正：对 ✅。前者是想用张量替换整个子模块，后者才是把数据装进子模块的参数里。
4. **把 `*` 换成 `@`，在 `(B, S, d_ff)` 上会怎样？**
   - 我的答案：会做内积或矩阵乘，直接报形状错误。
   - 订正：基本对 ✅。`@` 比较的是前者最后一维 `d_ff` 和后者倒数第二维 `S`：不相等时报错；碰巧 `S == d_ff` 时不报错，但结果是错的。
5. **a=[3,4]、g=[1,1] 时 RMSNorm 的输出是多少？**
   <details><summary>答案</summary>RMS = √12.5 ≈ 3.536，输出 ≈ [0.849, 1.131]。</details>
6. **SwiGLU 里哪一路是门、哪一路是内容？门关上是什么意思？**
   <details><summary>答案</summary>SiLU(W1x) 是门，W3x 是内容。门的某一维接近 0，那一维的乘积就接近 0，不管内容是多少。</details>
7. **为什么 RMSNorm 计算时要先转 float32？**
   <details><summary>答案</summary>bf16/fp16 下平方容易溢出或损失精度，算完再 `.to(in_dtype)` 转回原类型。</details>

#### 下一步
- §3.4.3 RoPE（旋转位置编码）。

## Day — RoPE 旋转位置编码（2026-10-24 ~ 2026-10-26）

#### 做了什么
- §3.4.3 RoPE：`cs336_basics/rope.py`（`RoPE(theta, d_k, max_seq_len, device)`）+ `adapters.run_rope`。

#### 背景 / 动机
- 注意力本身不区分顺序（详见下面"值得注意"第 3 条），所以必须注入位置信息。
- 原始 Transformer 把一个位置向量加到 embedding 上（绝对位置）。RoPE（Rotary Position Embedding，旋转位置编码）改成按位置旋转 q 和 k，旋转角度在点积里相减，注意力分数只依赖相对距离。LLaMA、Qwen 等都用它。
- 位置：在每个 attention 里，切完头之后、做点积之前，对 q 和 k 各转一次。

#### 公式
- 把 d_k 维向量两两分组：$(x_0,x_1), (x_2,x_3), \dots$，共 $d_k/2$ 组。第 $i$ 个位置、第 $k$ 组（$k=1,\dots,d_k/2$）的角度是

  $$\theta_{i,k} = \frac{i}{\Theta^{(2k-2)/d_k}}$$

- 每组做 2×2 旋转：

  $$\begin{bmatrix}x'_{2k-2}\\x'_{2k-1}\end{bmatrix} = \begin{bmatrix}\cos\theta_{i,k} & -\sin\theta_{i,k}\\ \sin\theta_{i,k} & \cos\theta_{i,k}\end{bmatrix}\begin{bmatrix}x_{2k-2}\\x_{2k-1}\end{bmatrix}$$

- 关键性质：$\langle R_i q, R_j k\rangle = \langle q, R_{j-i}k\rangle$，分数只依赖 $j-i$。$\Theta$ 是底数（10000），$i$ 是 token 的位置。

#### 输入 / 输出 / 形状
- `RoPE(theta, d_k, max_seq_len, device=None)`，**没有可学习参数**。
- buffer：`cos`、`sin`，形状 `(max_seq_len, d_k/2)`，`persistent=False`。
- `forward(x, token_positions)`：x 是 `(..., S, d_k)`，token_positions 能广播到 `(..., S)`，输出 `(..., S, d_k)`。

#### 实现步骤
```python
freq = 1 / theta ** (torch.arange(0, d_k, 2) / d_k)          # (d_k/2,)
angle = torch.outer(torch.arange(max_seq_len), freq)         # (max_seq_len, d_k/2)
register_buffer("cos", angle.cos()); register_buffer("sin", angle.sin())
# forward
cos, sin = self.cos[token_positions], self.sin[token_positions]
x1, x2 = x[..., 0::2], x[..., 1::2]
out = torch.stack((x1*cos - x2*sin, x1*sin + x2*cos), dim=-1).flatten(-2)
```
- 文件 `cs336_basics/rope.py`；adapter `run_rope`；`test_rope` PASSED。

#### 小数值例子
- d_k=2（只有一组，频率 1），q=[1,0] 在位置 1，k=[1,0] 在位置 3：旋转后 q=[cos1, sin1]=[0.540, 0.841]，k=[cos3, sin3]=[−0.990, 0.141]，点积 = −0.535 + 0.119 = −0.416 = cos 2。
- 换成位置 0 和 2：q=[1,0]，k=[cos2, sin2]，点积还是 cos 2 = −0.416。距离都是 2，分数一样。
- d_k=4、Θ=10000 时频率是 [1, 0.01]；位置 100 的两组角度分别是 100 弧度和 1 弧度。

#### 常犯的小错误
1. **`arange` 终点多减了 1**：`arange` 本来就不包含终点。`arange(0, d_k-2, 2)` 会少掉最后一组，`arange(0, max_seq_len-1)` 会少掉最后一个位置。终点直接写 `d_k`、`max_seq_len`。
2. **外积写成 `*`**：频率 `(d_k/2,)` 和位置 `(max_seq_len,)` 用 `*` 相乘，长度不同会报错，长度碰巧相同则会逐元素相乘、静默算错。要用 `torch.outer(位置, 频率)`，位置在前，作为行。
3. **`torch.outer([a], [b])` 多了方括号**：外面套一层方括号就变成 Python 列表，`outer` 要的是一维张量。
4. **新建张量没给 `dtype` / `device`**：位置向量要和频率在同一设备上，并且是 float。
5. **用 `arange` 代替从 x 切片**：`x1 = torch.arange(...)` 是新造了一串位置编号，和输入无关。要旋转的是 x 本身：`x[..., 0::2]`、`x[..., 1::2]`。
6. **用了 `__init__` 里没存的属性**：`self.max_seq_len` 从来没有赋值过，运行会报 AttributeError。
7. **查表结果没赋值**：`self.cos[token_positions]` 单独一行，算完就丢了。
8. **调用 forward 少传参数**：`rope(token_positions)` 把位置当成了 x，要写 `rope(in_query_or_key, token_positions)`。

#### 值得注意的地方
1. **核心思想**：不加位置向量，而是把 q、k 每相邻两个数看成平面上的点，按位置旋转。(a, b) 旋转 θ 后变成 (a·cosθ − b·sinθ, a·sinθ + b·cosθ)。
2. **为什么是相对位置**：q 转 iθ，k 转 jθ，点积只取决于 q、k 的内容和距离 j−i。位置 (3, 5) 和 (100, 102) 的分数一样。
3. **为什么需要位置信息**：注意力是加权求和，与顺序无关；Embedding、FFN、RMSNorm 都逐位置独立计算。不注入位置，「狗咬人」和「人咬狗」对模型来说没有区别。
4. **频率**：第 k 组是 1/Θ^((2k−2)/d)，分子是 0, 2, …, d−2，即 `arange(0, d_k, 2)`。d=4、Θ=10000 时为 [1, 0.01]。前面的组转得快，分辨近距离；后面的组转得慢，编码远距离。
5. **cos/sin 表形状**：`(max_seq_len, d_k/2)`，每个位置、每一组各一个角度。用 `token_positions` 直接索引，得到 `(..., seq_len, d_k/2)`，和 Embedding 查表同一个思路。
6. **`register_buffer(..., persistent=False)`**：buffer 不训练，但属于模块，会跟着 `.to(device)` 一起搬；`persistent=False` 表示不存进 `state_dict`，因为随时能重算，而且官方权重里也没有这两个 key。
7. **相邻配对，不是前后两半**：handout 规定 (x₀, x₁)、(x₂, x₃)…… 一组，所以用 `0::2` / `1::2` 切片。按 `:d/2` / `d/2:` 拆是 GPT-NeoX 那种配对，测试会对不上。
8. **交错拼回**：`torch.stack((out1, out2), dim=-1)` 得到 `(..., d_k/2, 2)`，再 `flatten(-2)` 得到 `(..., d_k)`。例：[a, c] 和 [b, d] → [[a, b], [c, d]] → [a, b, c, d]。
9. **写 MHA 时的坑**：x 会多一个 head 维 `(batch, heads, seq, d_k)`，而查表结果是 `(batch, seq, d_k/2)`。广播从右往左对齐，batch 会去对 heads，需要给 cos/sin 补 head 维。

#### 验收
- `uv run pytest -k test_rope`：PASSED。

#### 回忆题
1. **为什么用 `x[..., 0::2]` 和 `x[..., 1::2]`，而不是 `x[..., :d/2]` 和 `x[..., d/2:]`？**
   - 我的答案：因为要取每组里面的两个。
   - 订正：对 ✅。handout 规定相邻两个一组；按前后两半拆会配成 (x₀, x_{d/2})，是另一种约定（GPT-NeoX），测试对不上。
2. **`token_positions` 是 `(batch, seq_len)` 时，`self.cos[token_positions]` 是什么形状？为什么能和 `x1` 直接相乘？**
   - 我的答案：`(batch, seq_len, d_k/2)`，`x1` 也是 `(batch, seq_len, d_k/2)`，形状一样。
   - 订正：对 ✅。注意到了 MHA 会多出 head 维，届时需要补维度才能广播。
3. **q 在位置 1、k 在位置 3，和 q 在位置 0、k 在位置 2，分数一样吗？**
   <details><summary>答案</summary>一样（q、k 内容不变时）。旋转角度在点积里相减，只剩距离 2。</details>
4. **RoPE 有可学习参数吗？cos/sin 表为什么不进 state_dict？**
   <details><summary>答案</summary>没有。表可以随时从 Θ、d_k、max_seq_len 重算，所以用 `persistent=False` 的 buffer，官方权重里也没有这两个 key。</details>
5. **频率表为什么是 `arange(0, d_k, 2)`，不是 `arange(0, d_k-2, 2)`？**
   <details><summary>答案</summary>arange 本来就不包含终点。分子要取 0, 2, …, d_k−2 共 d_k/2 个数，终点写 d_k 才对。</details>

#### 下一步
- §3.4.4 softmax（减最大值保证数值稳定），然后 scaled dot-product attention。
## Day — softmax、缩放点积注意力与多头自注意力（2026-10-26）

#### 做了什么
- §3.4.4 softmax：`cs336_basics/nn_utils.py` 的 `softmax(x, dim)` 和 `adapters.run_softmax`。
- §3.4.4 缩放点积注意力（SDPA，scaled dot-product attention）：`nn_utils.scaled_dot_product_attention(Q, K, V, mask)` 和 `adapters.run_scaled_dot_product_attention`。
- §3.4.5 因果多头自注意力（MHA，multi-head self-attention）：`cs336_basics/attention.py` 的 `MultiHeadSelfAttention`，以及 `adapters.run_multihead_self_attention`。
- 带 RoPE 的 MHA：构造时可选传入 `theta` / `max_seq_len`，对应 `adapters.run_multihead_self_attention_with_rope`。

### softmax（模块全貌）

#### 背景 / 动机
- 把一组任意实数分数变成概率分布：每个数非负，加起来等于 1。注意力权重和模型最后输出的下一词分布都要用到它。
- 直接算 $e^x$ 在 x 很大时会溢出，所以要写数值稳定版。

#### 公式
$$\text{softmax}(x)_i = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}},\qquad m = \max_j x_j$$
- 分子分母同时乘 $e^{-m}$，结果不变，但最大的一项变成 $e^0=1$，不会溢出。

#### 输入 / 输出 / 形状
- `softmax(x, dim)`：任意形状，沿 `dim` 归一化，输出形状不变。没有参数。

#### 实现步骤
- `max(dim, keepdim=True).values` → 减掉 → `exp` → `sum(dim, keepdim=True)` → 相除。
- 文件 `cs336_basics/nn_utils.py`；adapter `run_softmax`；`test_softmax_matches_pytorch` PASSED。

#### 小数值例子
- x = [1, 2, 3]：减最大值得到 [−2, −1, 0]，exp 后是 [0.135, 0.368, 1]，和为 1.503，结果 ≈ [0.090, 0.245, 0.665]。
- x = [1000, 1001, 1002]：结果完全一样。不减最大值的话，exp(1000) = inf，inf/inf = NaN。

### 缩放点积注意力 SDPA（模块全貌）

#### 背景 / 动机
- 每个位置用自己的 query 去和所有位置的 key 比相似度，再按相似度对 value 做加权平均。这是 token 之间交换信息的唯一途径。

#### 公式
$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$$
- $Q$：查询，$K$：键，$V$：值；$d_k$：每个 q/k 的维度；$M$：mask 为 True 的位置是 0，为 False 的位置是 $-\infty$。

#### 输入 / 输出 / 形状
- `scaled_dot_product_attention(Q, K, V, mask=None)`
- Q `(..., m, d_k)`，K `(..., n, d_k)`，V `(..., n, d_v)`，mask 是布尔型，能广播到 `(..., m, n)`。
- 中间的 scores 是 `(..., m, n)`，输出 `(..., m, d_v)`。没有参数。

#### 实现步骤
```python
scores = einsum(Q, K, "... m d, ... n d -> ... m n") / math.sqrt(Q.shape[-1])
if mask is not None:
    scores = scores.masked_fill(~mask, float('-inf'))
return einsum(softmax(scores, dim=-1), V, "... m n, ... n d_v -> ... m d_v")
```
- adapter `run_scaled_dot_product_attention`；`test_scaled_dot_product_attention`、`test_4d_scaled_dot_product_attention` PASSED。

#### 小数值例子
- 1 个 query $q=[1,0]$，2 个 key $k_1=[1,0]$、$k_2=[0,1]$，value 是 10 和 20，$d_k=2$。
- 分数 = [1, 0]/√2 = [0.707, 0]，softmax 后 ≈ [0.670, 0.330]，输出 = 10×0.670 + 20×0.330 ≈ 13.3。
- 如果 mask = [True, False]，第二个 key 被屏蔽，输出正好是 10。

### 因果多头自注意力 MHA（含 RoPE）（模块全貌）

#### 背景 / 动机
- 一个头只能学一种"看哪里"的模式。多个头并行，各自在 d_k 维子空间里做注意力，最后拼起来再用 W_O 混合。
- 因果（causal）：语言模型要预测下一个词，位置 i 不能偷看 i 之后的 token。

#### 公式
$$\text{MHA}(x) = W_O\,\text{Concat}(\text{head}_1,\dots,\text{head}_h),\quad \text{head}_j = \text{Attention}(Q_j, K_j, V_j, \text{mask}=\text{causal})$$
- $Q = xW_Q^\top$、$K = xW_K^\top$、$V = xW_V^\top$，再切成 h 份，$d_k = d_{model}/h$。带 RoPE 时，$Q_j$、$K_j$ 先按位置旋转。

#### 输入 / 输出 / 形状
- `MultiHeadSelfAttention(d_model, num_heads, theta=None, max_seq_len=None, device=None, dtype=None)`
- 可学习参数：`q_proj`、`k_proj`、`v_proj`、`output_proj`，每个 weight 都是 `(d_model, d_model)`（LinearNoBias 截断正态初始化），共 4·d_model² 个。传了 theta 和 max_seq_len 时，再建一个 `RoPE(theta, d_k, max_seq_len)`，没有参数。
- `forward(x, token_positions=None)`：x 是 `(B, S, d_model)`，输出 `(B, S, d_model)`。中间形状：切头后 `(B, h, S, d_k)`，分数 `(B, h, S, S)`。

#### 实现步骤
```python
q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)                     # (B, S, d)
q, k, v = [rearrange(t, "... s (h d) -> ... h s d", h=self.num_heads) for t in (q, k, v)]
mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=x.device))
if self.rope is not None:
    if token_positions is None: token_positions = torch.arange(S, device=x.device)
    token_positions = token_positions.unsqueeze(-2)                           # 给 head 维留位置
    q, k = self.rope(q, token_positions), self.rope(k, token_positions)
out = scaled_dot_product_attention(q, k, v, mask)                             # (B, h, S, d_k)
return self.output_proj(rearrange(out, "... h s d -> ... s (h d)"))
```
- 文件 `cs336_basics/attention.py`；adapter `run_multihead_self_attention` 和 `run_multihead_self_attention_with_rope`（`mha.q_proj.weight.data = ...`）；两个测试都 PASSED。

#### 小数值例子
- d_model=8，h=2，所以 d_k=4。x 是 (1, 3, 8) → q 是 (1, 3, 8) → 切头 (1, 2, 3, 4) → 分数 (1, 2, 3, 3) → 输出 (1, 2, 3, 4) → 拼回 (1, 3, 8) → output_proj 后还是 (1, 3, 8)。参数量 4 × 8² = 256。
- S=3 的因果 mask：

  ```
  [[T, F, F],
   [T, T, F],
   [T, T, T]]
  ```

  第 0 个 token 只看自己，第 2 个 token 能看 0、1、2。

### 本节共用：错误、要点、验收与回忆

#### 常犯的小错误
1. **又忘了接 adapter（这一节出现了 4 次）**：每写完一个模块，`tests/adapters.py` 里对应的函数都要 import 并把 `raise NotImplementedError` 换成实现。说"写好了"之前先检查这一步。`return` 后面残留的 `raise` 也要顺手删掉。
2. **softmax 把 `dim` 写死成 -1**：函数有 `dim` 参数，就要用它。
3. **softmax 求和对象写错**：分母要对 exp 之后的值求和，不是对原始 x 求和。`torch.exp` 只接一个张量，不接 `dim` / `keepdim`。
4. **`torch.max(..., dim)` 返回的是 (values, indices)**：要取 `.values`，或者直接用 `amax`。`torch.sum` 返回的就是张量，后面不能再接 `.values`。
5. **`torch.sqrt` 用在 int 上**：`Q.shape[-1]` 是 Python 整数，不是张量，要用 `math.sqrt`。
6. **把维度大小当成维度编号**：softmax 的 `dim` 要写 `-1`（维度编号），不是 `K.shape[-1]`（维度大小）。
7. **`masked_fill` 的填充值写成了 `scores`**：应该填 `float('-inf')`，并且只在 `mask is not None` 时才做。
8. **成员没加 `self.`**：`q_proj = LinearNoBias(...)` 只是局部变量，不会注册成子模块，forward 里也访问不到。
9. **`/` 和 `//` 混用**：`d_k = d_model / num_heads` 得到的是 float，要用 `//`。
10. **einops 模式写了逗号**：`"... h s d -> ... s (h d)"` 里没有逗号；逗号只在 einsum 里分隔多个输入。
11. **漏了 `output_proj`**：多头拼回去之后还要再乘一次 W_O。
12. **adapter 加载权重错误（第二次了）**：`LinearNoBias(...)(q_proj_weight)` 是调用 forward，把权重当输入去乘一个随机矩阵。正确写法是 `mha.q_proj.weight.data = q_proj_weight`。
13. **RoPE 被无条件覆盖**：`self.rope = RoPE(...)` 下一行紧跟 `self.rope = None`，要写成 if/else。
14. **`theta and max_seq_len is not None` 的优先级**：实际含义是 `theta and (max_seq_len is not None)`，两边都要写 `is not None`。
15. **RoPE 放在切头之前**：此时最后一维是 d_model，而 RoPE 表是按 d_k 建的，要先切头再旋转。
16. **默认位置用了 `x.shape[-1]`**：那是 d_model，序列长度是 `x.shape[-2]`；张量还要放在 `x.device` 上，`forward` 里没有叫 `device` 的变量。
17. **`unsqueeze` 没接住返回值**：它不是原地操作，要写 `token_positions = token_positions.unsqueeze(-2)`。

#### 值得注意的地方
1. **softmax 减最大值**：e^x 在 x 大于约 88 时会溢出（float32）。每个元素减去同一个常数，结果不变，所以先减最大值，`keepdim=True` 保证能广播回去。
2. **注意力的 softmax 沿 key 维（最后一维）做**：每个 query 对所有 key 的权重加起来等于 1。
3. **为什么除以 √d_k**：q、k 各分量方差为 1 时，q·k 的方差是 d_k。分数太大会让 softmax 饱和成接近 one-hot，梯度接近 0。除以 √d_k 把方差拉回 1。
4. **mask 约定**：True 表示可以看，False 表示屏蔽。用 `masked_fill(~mask, -inf)`，softmax 后这些位置为 0。如果一整行都是 False，会全是 -inf，结果是 NaN。
5. **因果 mask**：`torch.tril(torch.ones(S, S, dtype=torch.bool, device=x.device))`，下三角为 True，第 i 个 token 只能看到 0 到 i。
6. **einsum 里转置不用 `.T`**：直接给维度起名字，`"... m d, ... n d -> ... m n"` 就是 QKᵀ。用 `@` 时写 `transpose(-2, -1)`，`.T` 会把 batch 维也反转。
7. **一次投影再切头**：一个 `(d_model, d_model)` 的大矩阵，数学上等价于把每个头的小矩阵拼在一起，但只需一次矩阵乘，更快。这是现在的通用做法（GPT-2 把 q、k、v 合成一个 `c_attn`；LLaMA / Qwen 分成 `q_proj`、`k_proj`、`v_proj`；GQA、MQA、MLA 是在 k、v 的头数或维度上做变化）。
8. **切头的形状**：`(B, S, d_model)` → `"... s (h d) -> ... h s d"` → `(B, h, S, d_k)`。head 维必须在 S 前面：SDPA 只在最后两维做点积和 softmax，前面都当 batch。如果停在 `(B, S, h, d_k)`，就成了同一个 token 的头和头之间互相做注意力；`(S, S)` 的 mask 通常广播不上而报错，S 恰好等于 h 时则不报错但悄悄算错。
9. **RoPE 只转 q 和 k，不转 v**：q 和 k 的旋转在点积里相互抵消，只剩相对位置 i−j。v 不参与点积，转了 v 会把绝对位置直接混进输出内容。
10. **RoPE 用 d_k，不用 d_model**：每个头内部两两配对旋转，所有头共用同一套角度。
11. **head 维广播**：`token_positions` 是 `(B, S)` 时，先 `unsqueeze(-2)` 变成 `(B, 1, S)`，查表得到 `(B, 1, S, d_k/2)`，那个 1 广播到每个头。测试传的是 `(1, S)`，不补维也碰巧能过，但真实的 `(B, S)` 会让 B 去对 h。
12. **`token_positions` 为什么要由外部传入**：位置不一定等于 token 在张量里的下标。例如 KV cache 推理时只送进 1 个新 token，它的位置是 10 而不是 0；还有左侧 padding、多文档拼接（packing）时每篇文档位置从 0 重新计数。
13. **加功能不破坏旧接口**：新参数默认 `None`，旧的调用方式和测试都不受影响；改完要把旧测试一起跑一遍。

#### 验收
- `uv run pytest -k test_softmax_matches_pytorch`：PASSED。
- `uv run pytest -k test_scaled_dot_product_attention`：3D、4D 都 PASSED。
- `uv run pytest -k test_multihead_self_attention`：`test_multihead_self_attention` 和 `test_multihead_self_attention_with_rope` 都 PASSED。

#### 回忆题
1. **SDPA 里为什么写 `math.sqrt(Q.shape[-1])`，不能写 `torch.sqrt(Q.shape[-1])`？**
   - 我的答案：shape 是维度大小，是一个整数值，不是 tensor。
   - 订正：对 ✅。
2. **如果某一行 mask 全是 False，softmax 之后是什么？**
   - 我的答案：全 0。
   - 订正：不对 ❌。整行都是 -inf，减最大值得到 -inf −（-inf）= NaN，结果是一整行 NaN。因果 mask 的对角线永远是 True，所以不会遇到这种情况。
3. **切头后为什么必须是 `(..., h, S, d_k)`？**
   - 我的答案：SDPA 只对最后两个维度做点积和 softmax，前面都看作 batch。
   - 订正：对 ✅，但漏了后果：停在 `(..., S, h, d_k)` 会变成头和头之间做注意力，mask 一般会报错，S 等于 h 时会静默算错。
4. **`LinearNoBias(...)(q_proj_weight)` 和 `mha.q_proj.weight.data = q_proj_weight` 有什么区别？**
   - 我的答案：先对 q_proj 初始化，再把测试用的矩阵赋值进去。
   - 订正：这只描述了 `.data` 的写法 ✅。前者是调用 forward，把权重当成输入去乘一个随机矩阵，根本没有加载进去。
5. **为什么 v 不做 RoPE？**
   - 我的答案：v 是得分矩阵，输出是每个位置在所有向量上的加权和。
   - 订正：部分对 ⚠️。得分是 softmax(QKᵀ/√d_k)，v 是被加权的内容。不转 v，是因为它不参与点积，旋转没法抵消成相对位置。
6. **KV cache 推理时第 11 个 token 进来，用默认 `token_positions` 会怎样？**
   - 我的答案：会把 0 到 11 的自注意力重新算一遍，KV cache 变大。
   - 订正：不对 ❌。只送进 1 个 token，S=1，默认位置是 `[0]`，它会被当成位置 0 旋转，和缓存里的 k 的相对距离全错，而且不报错。必须由调用方传入 `[[10]]`。
7. **x = [1000, 1001, 1002] 的 softmax 是多少？不减最大值会怎样？**
   <details><summary>答案</summary>和 [1, 2, 3] 一样，约 [0.090, 0.245, 0.665]。不减的话 exp(1000)=inf，结果是 NaN。</details>
8. **d_model=8、h=2、S=3 时，分数矩阵是什么形状？MHA 有多少参数？**
   <details><summary>答案</summary>(B, 2, 3, 3)；4 × 8² = 256 个。</details>
9. **为什么除以 √d_k？**
   <details><summary>答案</summary>q、k 各分量方差为 1 时，q·k 的方差是 d_k。分数太大 softmax 会饱和成接近 one-hot，梯度接近 0。除以 √d_k 把方差拉回 1。</details>

#### 下一步
- §3.5 Transformer block（pre-norm：`y = x + MHA(RMSNorm(x))`，再 `z = y + FFN(RMSNorm(y))`）。

## Day — Transformer block 与 TransformerLM（2026-10-26 ~ 2026-10-27）

#### 做了什么
- §3.5 Transformer block：`cs336_basics/transformer.py` 的 `TransformerBlock`（pre-norm），以及 `adapters.run_transformer_block`。
- §3.5 TransformerLM：`cs336_basics/transformer_lm.py` 的 `TransformerLM`，以及 `adapters.run_transformer_lm`。

### Transformer block（模块全貌）

#### 背景 / 动机
- 把"token 之间交流"（attention）和"每个 token 自己加工"（FFN）两个子层，加上 norm 和残差，组成一个可以反复堆叠的单元。整个模型就是 N 个 block 叠起来。

#### 公式（pre-norm）
$$y = x + \text{MHA}(\text{RMSNorm}_1(x)),\qquad z = y + \text{FFN}(\text{RMSNorm}_2(y))$$

#### 输入 / 输出 / 形状
- `TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta, device=None, dtype=None)`
- 子模块和参数：`ln1.weight` (d,)，`attn.{q,k,v,output}_proj.weight` 各 (d, d)，`ln2.weight` (d,)，`ffn.w1.weight`、`ffn.w3.weight` (d_ff, d)，`ffn.w2.weight` (d, d_ff)。每个 block 共 $4d^2 + 3d\,d_{ff} + 2d$ 个参数。
- `forward(x, token_positions=None)`：`(B, S, d)` → `(B, S, d)`。

#### 实现步骤
```python
y = x + self.attn(self.ln1(x), token_positions)
return y + self.ffn(self.ln2(y))
```
- 文件 `cs336_basics/transformer.py`；adapter `run_transformer_block` 里一句 `block.load_state_dict(weights)` 加载全部权重；`test_transformer_block` PASSED。

#### 小数值例子
- d=512、d_ff=1344：attention 4×512² = 1,048,576，FFN 3×512×1344 = 2,064,384，两个 norm 2×512 = 1,024，合计 3,113,984 ≈ 311 万，FFN 约占 2/3。

### TransformerLM（模块全貌）

#### 背景 / 动机
- 完整的语言模型：输入一串 token id，输出每个位置对下一个 token 的打分（logits）。训练时和 y 算交叉熵，生成时 softmax 后采样。

#### 公式
$$h_0 = E[\text{ids}],\quad h_l = \text{Block}_l(h_{l-1})\ (l=1..L),\quad \text{logits} = \text{RMSNorm}(h_L)\,W_{\text{lm}}^\top$$
- 概率 $p(\text{下一个 token}\mid\text{前缀}) = \text{softmax}(\text{logits})$，在模型外面算。

#### 输入 / 输出 / 形状
- `TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device=None, dtype=None)`
- 参数：`token_embeddings.weight` (V, d)，`layers.{i}.*`（L 个 block），`ln_final.weight` (d,)，`lm_head.weight` (V, d)。总数 $P = L(4d^2+3d\,d_{ff}+2d) + 2Vd + d$。
- `forward(in_indices)`：`(B, S)` 的 long → logits `(B, S, V)`。

#### 实现步骤
```python
x = self.token_embeddings(in_indices)       # (B, S, d)
for block in self.layers:
    x = block(x)                             # 结果写回 x
return self.lm_head(self.ln_final(x))       # (B, S, V)
```
- 文件 `cs336_basics/transformer_lm.py`；adapter `run_transformer_lm`（`load_state_dict`）；`test_transformer_lm`、`test_transformer_lm_truncated_input` PASSED。
- 2026-11-03 加注释时标出（`# 注意`）：`forward` 没有把 `token_positions` 传给各个 block，所以 RoPE 永远用默认的 0..S−1。训练时没问题，以后做 KV cache 推理或 packing 时要改。

#### 小数值例子
- TinyStories 配置：V=10000、context_length=256、d=512、L=4、h=16、d_ff=1344。
- 4 个 block：4 × 3,113,984 = 12,455,936；embedding 和 lm_head：2 × 10000 × 512 = 10,240,000；ln_final：512。总计 22,696,448 ≈ 2270 万。
- handout 说约 17M non-embedding 参数，对应 block 的 12.46M 加上 lm_head 的 5.12M ≈ 17.6M。

### 本节共用：错误、要点、验收与回忆

#### 常犯的小错误
1. **adapter 没有加载权重（第三次了）**：block 的 adapter 里只留了一句注释，测试跑的是随机初始化的模型。子模块按 key 命名后，一句 `load_state_dict(weights)` 就能全部加载。
2. **类名写成小写 `transformer`**：adapter 里 import 的是 `TransformerBlock`，类名要和 import 对上（类名也按惯例用大驼峰）。
3. **`ModuleList` 里只放了一个 block**：`num_layers` 传进来却没用到，要写 `[TransformerBlock(...) for _ in range(num_layers)]`。报错是 `Unexpected key(s): layers.1.*, layers.2.*`。
4. **`forward` 完全跳过了 block**：embedding 出来直接进 `ln_final`，等于 0 层模型。key 全部对上之后才暴露出来，表现为输出和 snapshot 对不上。
5. **循环要把结果写回 `x`**：`for block in self.layers: x = block(x)`，下一层才能拿到上一层的输出。
6. **建 block 时漏传 `device`**：CPU 测试照样能过，上 GPU 才会出问题。
7. **参数名转换**：LM 签名里的 `context_length` 和 `rope_theta`，传给 block 时分别对应 `max_seq_len` 和 `theta`。
8. **`return` 后面又留了 `raise NotImplementedError`**：是死代码，删掉。

#### 值得注意的地方
1. **pre-norm 结构**：`y = x + attn(ln1(x), pos)`，`z = y + ffn(ln2(y))`。只对子层的输入做 norm，残差主干上不做 norm。
2. **两个 RMSNorm 要分开**：每个都有自己可学习的 gain，分别服务 attention 和 FFN 两个不同的子层，不能共用。
3. **残差的作用**：输出等于 x + f(x)，求导得到 1 + f'(x)，梯度始终有一条恒等通路，层数再深也不容易消失。
4. **pre-norm 与 post-norm**：post-norm（原始 Transformer）把 norm 放在残差相加之后，主干每层都被 norm 一次，深层训练不稳定，通常需要 warmup。pre-norm 主干是干净的，训练更稳定，代价是最后要多加一个 `ln_final`。
5. **block 不改变形状**：输入输出都是 (B, S, d_model)，所以可以一层接一层地堆叠。
6. **LM 的数据流**：`(B, S)` 整数 → embedding 得到 `(B, S, d_model)` → N 个 block → `ln_final` → `lm_head` 得到 `(B, S, vocab_size)`。
7. **`lm_head` 是什么**：一个 `LinearNoBias(d_model, vocab_size)`，权重形状是 (vocab_size, d_model)，可训练、随机初始化。它把每个位置的隐藏向量 h 和每个词的那一行做点积，得到这个词的打分。
8. **输出是 logits，不做 softmax**：logits 是未归一化的分数，可以是负数，加起来也不等于 1。softmax 放在模型外面：训练时在交叉熵里做，生成时在采样前做。
9. **位置 t 的 logits 预测的是第 t+1 个 token**：因果 mask 让位置 t 只看得到 0..t，所以它的任务是猜下一个。
10. **weight tying**：GPT-2 让 `lm_head` 和 embedding 共用同一个矩阵（形状正好相同）。A1 用的是两份独立的权重。
11. **`nn.ModuleList` 和 Python list 的区别**：ModuleList 会把第 i 个 block 注册成名为 `i` 的子模块，生成 `layers.0.attn.q_proj.weight` 这样的 key。普通 list 里的模块不会被注册：`load_state_dict` 报 Unexpected key，`parameters()` 看不到它们（优化器不会更新），`.to(device)` 也不会搬它们。
12. **Missing 与 Unexpected**：Missing 是模型需要、字典里没有（比如建了 5 层、权重只有 3 层）；Unexpected 是字典里有、模型没有（比如建了 1 层、权重有 3 层）。
13. **截断输入测试为什么能过**：序列减半后，前半段输出和完整输入时完全一样。原因有三：因果 mask 下每个位置看不到后面的 token；RoPE 的默认位置 `arange(S)` 前缀相同；RMSNorm 和 FFN 都是逐 token 计算。

#### 验收
- `uv run pytest -k "test_transformer_lm or test_transformer_block"`：`test_transformer_block`、`test_transformer_lm`、`test_transformer_lm_truncated_input` 全部 PASSED。

#### 回忆题
1. **block 里为什么要两个 RMSNorm？**
   - 我的答案：这两个要 norm 的矩阵均值不一样，需要分别训练。
   - 订正：方向对 ⚠️。RMSNorm 不减均值；准确地说，每个 norm 有自己的 gain，分别服务 attention 和 FFN 两个不同的子层。
2. **残差连接的作用？**
   - 我的答案：为了反向传播的稳定性。
   - 订正：对 ✅。具体是导数 1 + f'，梯度始终有一条恒等通路。
3. **pre-norm 和 post-norm 的区别？**
   - 我的答案：忘记了。
   - 订正：见上面第 4 条。
4. **block 的输入输出形状？**
   - 我的答案：都是 (B, S, d_model)，不改变维度。
   - 订正：对 ✅，所以才能堆叠。
5. **`self.layers` 用普通 list 会怎样？**
   - 我的答案：ModuleList 会把第 i 个 block 登记成名为 i 的子模块，key 能被识别为 `layers.0.attn...`；普通 list PyTorch 不认识，state_dict 看不到，load 不进去。
   - 订正：对 ✅。补充：报的是 Unexpected key，而且优化器和 `.to(device)` 也看不到这些 block。
6. **为什么截断输入后前半段输出完全一样？**
   - 我的答案：因果 mask 让当前 token 只能看到它之前的 token，所以只输入一半也不影响；RoPE 这块不太清楚。
   - 订正：部分对 ⚠️。RoPE 只看每个 token 自己的位置编号，`arange(6)` 正好是 `arange(12)` 的前 6 个，旋转结果相同。
7. **模型建了 5 层、权重只有 3 层，会报什么？**
   - 我的答案：会报缺少输入？
   - 订正：基本对 ⚠️。准确说法是 `Missing key(s): layers.3.*, layers.4.*`。
8. **TinyStories 配置（V=10000, d=512, L=4, d_ff=1344）一共多少参数？**
   <details><summary>答案</summary>约 2270 万：block 12.46M + embedding 和 lm_head 10.24M + ln_final 512。</details>
9. **TransformerLM 的 forward 现在有什么隐患？**
   <details><summary>答案</summary>没有把 token_positions 传给 block，RoPE 一直用 0..S−1。KV cache 推理或多文档 packing 时位置会错，而且不报错。</details>

#### 下一步
- §4 交叉熵（cross-entropy），然后是 AdamW、学习率余弦调度、梯度裁剪。
## Day — Resource accounting：transformer_accounting（2026-10-27）

#### 背景 / 动机
- 动手训练之前，先用纸笔估算：模型有多少参数（决定显存），前向要多少 FLOPs（floating-point operations，浮点运算次数，决定训练时间），哪一部分最贵（决定优化方向）。
- 这是 handout 的书面题，没有代码和测试。

#### 符号
- V：词表大小；S：序列长度（context_length）；L：层数；d：d_model；h：头数；$d_k = d/h$；f：d_ff。
- GPT-2 XL 配置：V=50257，S=1024，L=48，d=1600，h=25，f=4288（离 8/3×1600 最近的 64 的倍数）。

#### 公式
- 矩阵乘 $(m\times n)\cdot(n\times p)$ 需要 $2mnp$ 次 FLOPs。
- 参数量 $P = L(4d^2 + 3df + 2d) + 2Vd + d$；前向 FLOPs $= L(8Sd^2 + 4S^2d + 6Sdf) + 2SdV$（推导和数字见下面"值得注意"）。

#### 常见错误
1. **把 embedding 算进 block 里**：embedding（V·d）在所有 block 外面，只有一份，不能乘 L。
2. **output_proj 形状写成 (d_k, d)**：多头拼接后是 d 维，所以 output_proj 是 (d, d)。
3. **忘了 FFN 的形状**：SwiGLU 有三个矩阵，w1、w3 是 (d_ff, d)，w2 是 (d, d_ff)，每个 block 共 3·d·d_ff 个参数。
4. **漏了 block 里的 ln1、ln2**：每个 block 有两个 RMSNorm，共 2d 个参数。
5. **把 2Vd 乘了 L**：embedding 和 lm_head 只有一份，乘了 L 后得到 92 亿，正确答案约 16.4 亿。
6. **GB 和 GiB 混用**：字节数 ÷ 10⁹ 是 GB，÷ 2³⁰ 是 GiB。之前的 34.28 其实是 GiB。
7. **矩阵乘形状顺序写反**：batch=1 时输入是 (S×d)，投影是 (S×d)·(d×d)，写成 (d×d)·(S×d) 形状对不上。
8. **QKᵀ 和 分数×V 没有按头拆开**：每个头是 (S×d_k)·(d_k×S)，一共 h 个。
9. **漏了 output_proj 和 FFN 的矩阵乘**：其中 FFN 恰好是最大的一块。
10. **把 RoPE、RMSNorm、softmax 当成矩阵乘**：它们是逐元素或逐行运算，量级只有 S·d 或 h·S²。embedding 查表也不是矩阵乘。
11. **把四个维度全乘起来**：(S×d)·(d×d) 写成 S·d³。共享的中间维度 n 只乘一次，应该是 2·S·d·d = 2Sd²。
12. **丢了 2mnp 前面的 2**：乘法和加法各算一次 FLOP。
13. **以为 FFN 和投影的比值是 f/d ≈ 2.68**：要带上系数，6Sdf / 8Sd² = (3/4)·(f/d) ≈ 2。
14. **预测模型变大时四块占比都变小**：占比加起来永远是 100%，不可能都变小。

#### 值得注意
1. **2mnp 规则**：(m×n)·(n×p) 的结果有 m·p 个数，每个数是长度为 n 的点积，要做 n 次乘法和 n 次加法，所以总共 2mnp。
2. **参数量公式**：P = L(4d² + 3df + 2d) + 2Vd + d。GPT-2 XL 是 1,640,452,800 ≈ 16.4 亿，fp32 下占 6.56 GB（6.11 GiB）。block 约占 90%，其中 FFN 约是 attention 投影的 2 倍。
3. **每个 block 的矩阵乘清单（batch=1）**：

   | 矩阵乘 | 形状 | 次数 |
   |---|---|---|
   | Q、K、V 投影 | (S×d)·(d×d) | 3 |
   | QKᵀ | (S×d_k)·(d_k×S) | h |
   | 分数 × V | (S×S)·(S×d_k) | h |
   | output_proj | (S×d)·(d×d) | 1 |
   | FFN w1、w3 | (S×d)·(d×f) | 2 |
   | FFN w2 | (S×f)·(f×d) | 1 |

   block 外面只有 lm_head：(S×d)·(d×V)。
4. **前向 FLOPs 公式**：FLOPs = L(8Sd² + 4S²d + 6Sdf) + 2SdV。用 h·d_k = d 可以把多头项合并。
5. **一个化简技巧**：三项都除以 2Sd，得到 3f : 4d : 2S，一眼就能看出谁大。
6. **GPT-2 XL（S=1024）**：每个 block 6.98×10¹⁰，48 层共 3.35×10¹²，lm_head 1.65×10¹¹，总共 3.52×10¹²。FFN 占 57.5%，投影占 28.6%，attention 分数占 9.2%，lm_head 占 4.7%。
7. **d_ff 的取法**：取最接近 8/3·d 的 64 的倍数。small 是 2048，medium 是 2752，large 是 3392，XL 是 4288。
8. **模型变大时的趋势**：

   | 模型 | 总 FLOPs | FFN | 投影 | attention 分数 | lm_head |
   |---|---|---|---|---|---|
   | small | 2.9×10¹¹ | 39.8% | 19.9% | 13.3% | 27.1% |
   | medium | 8.3×10¹¹ | 50.1% | 24.8% | 12.4% | 12.7% |
   | large | 1.8×10¹² | 54.3% | 27.3% | 10.9% | 7.4% |
   | XL | 3.5×10¹² | 57.5% | 28.6% | 9.2% | 4.7% |

   FFN 和投影正比于 d²，占比上升，两者始终约 2:1。attention 分数正比于 S²d，lm_head 正比于 dV 且只有一份（相对 block 的比值约 V/(L·d)），所以这两项占比下降。小模型的词表很"贵"：small 里 lm_head 占 27%。
9. **长上下文**：XL 的 context 从 1024 变成 16384（16 倍）时，attention 分数涨 256 倍，其他项涨 16 倍。总 FLOPs 涨约 38 倍，到 1.34×10¹⁴；attention 分数占比从 9.2% 变成 61.7%。这就是长上下文贵的原因，也是 FlashAttention、稀疏 attention 要解决的问题。

#### 小数值例子
- (2×3)·(3×4)：结果有 2×4 = 8 个数，每个数是长度 3 的点积，要做 3 次乘法和 3 次加法，合计 2×2×3×4 = 48 FLOPs。
- 一个 Q 投影，S=1024、d=1600：2 × 1024 × 1600 × 1600 ≈ 5.24×10⁹ FLOPs；一个 block 有 Q、K、V、O 四个，合计约 2.1×10¹⁰，正好是 8Sd²。

#### 回忆题
1. **(m×n)·(n×p) 需要多少 FLOPs？为什么？**
   - 答案：2mnp。结果有 mp 个数，每个是长度为 n 的点积（n 次乘法和 n 次加法）。
2. **GPT-2 XL 的参数量和 fp32 内存？**
   - 答案：约 16.4 亿个参数，6.56 GB。
3. **为什么 RMSNorm、softmax 不计入 FLOPs？**
   - 答案：不是矩阵乘，量级只有 S·d 或 h·S²，比矩阵乘小一两个数量级。
4. **模型变大时，FFN 和投影的比值会变吗？**
   - 我的答案：会变小。
   - 订正：不变 ⚠️。f ≈ (8/3)d 代进 3f/(4d)，d 约掉，结果约等于 2。
5. **context 变成 16 倍，各项和总数怎么变？**
   - 我的答案：FFN、投影、lm_head 各 16 倍，attention 分数 256 倍，总数超过 16 倍，attention 占比最大。
   - 订正：对 ✅。总数约 38 倍。
6. **一个 block 有多少参数？用 d、f 表示。**
   <details><summary>答案</summary>4d²（四个投影）+ 3df（SwiGLU 三个矩阵）+ 2d（两个 RMSNorm）。</details>
7. **为什么 lm_head 在小模型里占比很高？**
   <details><summary>答案</summary>它正比于 dV，只有一份，和 block 的比值约 V/(L·d)。小模型 L、d 都小，所以 small 里 lm_head 占 27%。</details>

#### 下一步
- §4.1 交叉熵（cross-entropy）：已经答完概念题，接下来实现 `run_cross_entropy`。

## Day — 训练基础：交叉熵、SGD、AdamW（2026-10-28 ~ 10-30）

### 交叉熵（cross-entropy）

#### 背景 / 动机
- 语言模型的输出是每个位置的 logits，要有一个数来衡量"预测得多好"，作为训练要最小化的目标。交叉熵就是"正确 token 的负对数概率"的平均：模型给正确答案的概率越高，loss 越小。
- 评估时把它取 exp，就是困惑度（perplexity）。
- 位置：模型输出 logits `(B, S, V)` 和标签 y `(B, S)`（来自 `get_batch`）一起进 loss，然后 `backward()`。

#### 公式
$$\ell_i = -\log \text{softmax}(o_i)[y_i] = \text{logsumexp}(o_i) - o_{i,y_i},\qquad \text{logsumexp}(o) = m + \log\sum_j e^{o_j - m}$$
$$\text{loss} = \frac{1}{N}\sum_{i=1}^{N}\ell_i,\qquad \text{PPL} = e^{\text{loss}}$$
- $o_i$：第 i 个位置的 logits（V 维），$y_i$：正确 token id，$m = \max_j o_{i,j}$，N = B×S 个位置。

#### 输入 / 输出 / 形状
- `cross_entropy(inputs, targets)`：inputs `(..., V)` 的 logits，targets `(...)` 的整数，输出标量。没有参数。

#### 实现步骤
```python
targets = targets.reshape(-1)                                  # (N,)
inputs = inputs.reshape(-1, inputs.shape[-1])                  # (N, V)
m = torch.max(inputs, dim=-1, keepdim=True).values             # (N, 1)
lse = m.squeeze(-1) + torch.log(torch.sum(torch.exp(inputs - m), dim=-1))   # (N,)
correct = inputs[torch.arange(N), targets]                     # (N,)
return (lse - correct).mean()
```
- 文件 `cs336_basics/cross_entropy.py`；adapter `run_cross_entropy`；`test_cross_entropy` PASSED。

#### 小数值例子
- o = [2, 1, 0]，y = 0：logsumexp = 2 + log(1 + e⁻¹ + e⁻²) = 2 + log 1.503 = 2.408，loss = 2.408 − 2 = 0.408，对应 p_y = e^−0.408 ≈ 0.665。
- 如果 y = 2：loss = 2.408 − 0 = 2.408，p_y ≈ 0.090。
- 两个位置的 loss 分别是 0.408 和 2.408，平均 1.408，PPL = e^1.408 ≈ 4.09。

#### 常见错误
1. **普通函数里多写了 `self` 参数**：`cross_entropy` 不是类方法，不需要 `self`。
2. **logsumexp 漏了 sum**：应该是 log(Σ exp(x − m))，先求和再取 log。
3. **m 没有 squeeze**：`max(..., keepdim=True)` 得到 (N,1)，再和 (N,) 相加会广播成 (N,N)。
4. **`inputs.shape(-2)` 把 shape 当函数调用**：shape 要用下标 `shape[-2]`。
5. **reshape 结果没赋值回去**：`inputs.reshape(...)` 不是原地操作，不赋值等于没写。
6. **`reshape(-1,)` 把 logits 压成了一维**：应该是 `reshape(-1, inputs.shape[-1])`，保留词表维 V。
7. **adapter 没接**，`return` 后面又留了一个死掉的 `raise NotImplementedError`。

#### 值得注意
1. **单个位置的 loss = −x[y] + logsumexp(x)**：target 是 one-hot，交叉熵求和后只剩 −log p_y。
2. **数值稳定的两个技巧**：减去最大值 m 防止 exp **上溢**；直接用 logsumexp，让 log 和 exp 相互抵消，防止 softmax 结果下溢成 0 后出现 log 0。先算稳定的 softmax 再取 log 的写法，在 logits 放大 1000 倍时仍然会失败。
3. **取正确词的 logit**：`inputs[torch.arange(N), targets]`，第 i 行取第 targets[i] 列。
4. **平均是对全部 B×S 个位置取的**：先把 logits 展平成 (−1, V)，targets 展平成 (−1)。
5. **困惑度 = exp(平均交叉熵)**：先平均再取 exp，不能先 exp 再平均。可以理解为模型平均在几个词里犹豫。
6. **初始 loss 自检**：均匀瞎猜时 loss = log V，困惑度 = V。V=50257 时约 10.8，V=10000 时约 9.2。训练一开始的 loss 如果明显偏离这个值，多半是初始化或 loss 写错了。

#### 回忆题
1. **o=[2,1,0]、y=0 时交叉熵是多少？**
   <details><summary>答案</summary>logsumexp ≈ 2.408，loss ≈ 0.408，p_y ≈ 0.665。</details>
2. **为什么要先 reshape 成 (N, V) 和 (N,)？**
   <details><summary>答案</summary>loss 是对所有 B×S 个位置取平均；展平之后可以用 `inputs[arange(N), targets]` 一次取出每行正确词的 logit。</details>
3. **m 用了 keepdim=True，为什么加到 log 那一项上之前要 squeeze？**
   <details><summary>答案</summary>(N,1) 和 (N,) 相加会广播成 (N,N)。先 squeeze 成 (N,)。</details>

### SGD 与 learning_rate_tuning

#### 背景 / 动机
- 有了 loss 和梯度，还要一个规则来更新参数，这就是优化器。SGD（Stochastic Gradient Descent，随机梯度下降）是最简单的一种：沿负梯度方向走一小步。
- handout 用一个带 1/√(t+1) 衰减的 SGD 来介绍 `torch.optim.Optimizer` 的写法（param_groups、state、step），为后面的 AdamW 打基础；learning_rate_tuning 实验让我们看到 lr 太大会发散。

#### 公式
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{t+1}}\,\nabla L(\theta_t)$$
- $\theta$：参数，$\alpha$：初始学习率，$t$：这个参数已经更新过的次数（从 0 开始），$\nabla L$：梯度，形状和 θ 一样。

#### 输入 / 输出 / 形状
- `SGD(params, lr=1e-3)`；`step(closure=None)` 返回 closure 的 loss 或 None。
- 状态：`self.state[p]["t"]`，一个整数。没有额外张量状态。

#### 实现步骤
```python
for group in self.param_groups:
    lr = group["lr"]
    for p in group["params"]:
        if p.grad is None: continue
        t = self.state[p].get("t", 0)
        p.data -= lr / math.sqrt(t + 1) * p.grad.data
        self.state[p]["t"] = t + 1
```
- 文件 `cs336_basics/sgd.py`，实验写在 `if __name__ == "__main__":` 里（`uv run python -m cs336_basics.sgd`）。这一题没有 pytest，看打印出来的 loss。

#### 小数值例子
- loss = mean(w²)，w 是 10×10，所以 ∂loss/∂w = 2w/100，每步 w 乘以 $1 - \frac{2\alpha}{100\sqrt{t+1}}$。
- α=100、t=0：系数 = 1 − 2 = −1，w 变号但大小不变，loss 不变；t=1 时系数 = 1 − 2/√2 ≈ −0.414，开始变小。
- α=1000、t=0：系数 = −19，loss 乘以 361，之后一路发散。

#### 常见错误
1. **`super().__init__(self, defaults)`**：第一个参数应该是 `params`，不是 `self`。
2. **实验循环里没有 `backward()` 和 `step()`**：只算了 loss，参数从来没更新。
3. **lr 写死成 1**：循环变量 `lr_rate` 没有传进优化器。
4. **换 lr 时没有重建 weights 和优化器**：后一组会接着前一组的结果跑，没法对比。
5. **f-string 语法错误**：`f"lr={lr_rate}: "loss...` 中间少了 `{}` 或逗号。
6. **类型注解用了小写 `callable`**：应该用 `Callable`。

#### 值得注意
1. **∇ 是梯度**：loss 对每个参数元素分别求偏导，和参数同形状（`p.grad` 和 `p` 形状一样）。
2. **三个东西各管一样**：`param_groups` 管超参数，`p.grad` 管梯度，`state[p]` 管跨步骤的历史。
3. **param_groups 是列表套字典**：可以给不同参数分组设不同 lr 或 weight_decay，没写的超参数由 `defaults` 补上。
4. **`p.data -= ...` 是原地更新**：`p = p - ...` 只是让局部变量指向新张量，模型参数不会变。
5. **closure 和 `return loss`**：只为 L-BFGS 这类一步要多次算 loss 的优化器准备，SGD 和 AdamW 用不到，返回 None。
6. **`backward()` 是累加梯度**：所以每步之前要先 `zero_grad()`。
7. **lr 实验**：每步系数是 1 − 2lr/(100√(t+1))，loss 乘它的平方。
   - lr=10：稳定但慢，10 步从 22 降到 3。
   - lr=100：第 0 步系数正好是 −1，loss 不变，之后迅速收敛到约 1e-23。
   - lr=1000：第 0 步系数是 −19，loss 乘 361，指数式发散到约 1e18。
   - 结论：系数绝对值大于 1 就会发散。

#### 回忆题
1. **α=100 时第 0 步 loss 为什么不变？**
   <details><summary>答案</summary>系数 1 − 2×100/100 = −1，w 只是变号，w² 不变。</details>
2. **`p.data -= ...` 和 `p = p - ...` 有什么区别？**
   <details><summary>答案</summary>前者原地修改参数；后者只是让局部变量 p 指向一个新张量，模型参数不变。</details>
3. **为什么每步之前要 `zero_grad()`？**
   <details><summary>答案</summary>`backward()` 是把梯度累加到 p.grad 上，不清零就会叠上前一步的梯度。</details>

### AdamW

#### 背景 / 动机
- SGD 对所有参数用同一个步长。梯度大的维度会走过头，梯度小的维度又走得太慢。Adam 给每个元素单独调步长：m 记方向，v 记大小，m/√v 让每一步的步长大约都是 α。
- AdamW 再把权重衰减和梯度更新"解耦"，直接每步把 θ 缩小一点，而不是把 λθ 加进梯度里（那样会被 v 缩放）。这是训练 LLM 的标准优化器。

#### 公式（t 从 1 开始）
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,\qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$
$$\alpha_t = \alpha\frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t},\qquad \theta \leftarrow \theta - \alpha_t\frac{m_t}{\sqrt{v_t}+\varepsilon},\qquad \theta \leftarrow \theta - \alpha\lambda\theta$$
- $g$：梯度；$m$：一阶矩（梯度的滑动平均）；$v$：二阶矩（梯度平方的滑动平均）；$\beta_1=0.9$、$\beta_2=0.999$：衰减率；$\varepsilon=10^{-8}$；$\lambda$：权重衰减系数；$\alpha_t$：做了偏差修正（m、v 从 0 开始，前几步偏小）的学习率。

#### 输入 / 输出 / 形状
- `AdamW(params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)`
- 每个参数的状态：`state[p] = {"t": int, "m": 和 p 同形状, "v": 和 p 同形状}`，用 `torch.zeros_like(p)` 初始化。优化器状态是参数量的 2 倍。

#### 实现步骤
```python
if len(state) == 0: state["t"] = 0; state["m"] = zeros_like(p); state["v"] = zeros_like(p)
t = state["t"] + 1
m = b1*m + (1-b1)*g;  v = b2*v + (1-b2)*g**2;  state["m"], state["v"] = m, v
lr_t = lr * math.sqrt(1 - b2**t) / (1 - b1**t)        # 局部变量
p.data -= lr_t * m / (torch.sqrt(v) + eps)
p.data *= 1 - lr * weight_decay                       # 用原始 lr
state["t"] = t
```
- 文件 `cs336_basics/adamw.py`；adapter `get_adamw_cls` 返回类 `AdamW`；`test_adamw` PASSED。

#### 小数值例子
- 一个标量参数 θ=1，g=0.5，α=0.1，λ=0.01，t=1：
  - m = 0.1 × 0.5 = 0.05；v = 0.001 × 0.25 = 0.00025，√v ≈ 0.0158
  - α_t = 0.1 × √0.001 / 0.1 ≈ 0.0316
  - 更新量 = 0.0316 × 0.05 / 0.0158 ≈ 0.1，θ = 0.9
  - 衰减：θ × (1 − 0.1 × 0.01) = 0.8991
- 把 g 换成 50，更新量还是约 0.1：第一步的步长约等于 α，和梯度大小无关。

#### 常见错误
1. **adapter 第 7 次忘了接**：改完 `adamw.py` 还是 `raise NotImplementedError`。
2. **adapter 返回了实例 `AdamW()`**：应该返回类本身 `AdamW`，测试会自己用参数和超参数去构造。
3. **把 m、v 写成 `self.m = nn.Parameter(torch.zeros(params))`**：
   - 每个参数需要各自的 m、v，一个 `self.m` 放不下。
   - `params` 是张量序列，不是形状；而且生成器会被这里读空。
   - m、v 是优化器状态，不是可训练参数。
   - 正确位置是 `self.state[p]`。
4. **m、v 用前没初始化，算完也没存回**：第一次用到时会报未定义。
5. **`torch.zeros(p)`**：zeros 要的是形状，应该用 `torch.zeros_like(p)`，顺便带上 dtype 和 device。
6. **`math.sqrt(v)`**：v 是张量，要用 `torch.sqrt`。
7. **t 加了两次**：开头已经 `+1`，结尾又存了 `t+1`。
8. **defaults 写死成常数**：传进来的 betas、eps、weight_decay 被忽略了。
9. **`ValueError(...)` 前面少了 `raise`**。
10. **import 写成 `adamW`，文件名却是 `adamw.py`**：Windows 不区分大小写，Linux 上会报错。

#### 值得注意
1. **算法（t 从 1 开始）**：
   - m ← β₁m + (1−β₁)g
   - v ← β₂v + (1−β₂)g²
   - α_t = α·√(1−β₂ᵗ)/(1−β₁ᵗ)
   - θ ← θ − α_t·m/(√v+ε)
   - θ ← θ − αλθ
2. **m 管方向，v 管大小**：m 是梯度的滑动平均，来回抖动的方向会相互抵消；v 是梯度平方的滑动平均。m/√v 约等于 ±1，所以第一步的步长约等于 α，和梯度本身的大小无关。
3. **一切都是逐元素的**：m、v 和 p 同形状，`g*g` 是逐元素平方。优化器状态是参数量的 2 倍。
4. **t 必须从 1 开始**：t=0 时 α_t = 0/0 = nan。写法是"先加后用"：`t = state.get("t", 0) + 1`。
5. **权重衰减用原始 α，不用 α_t**：衰减和 m、v 无关，用 α_t 会让衰减强度随 t 先变弱再回升。这就是 decoupled（解耦）的意思。
6. **为什么要权重衰减**：把没用的权重拉回 0 附近，防止过拟合，也让 logits 和激活值的尺度保持稳定。实际中 bias 和 RMSNorm 的 gain 一般不做衰减。
7. **`super().__init__(params, defaults)`**：父类负责把 params 整理成 `param_groups`、用 defaults 补全缺失的超参数，并创建 `self.state`。
8. **α_t 是局部变量**：不要写回 `group["lr"]`，否则每一步会叠加修正。

#### 回忆题
1. **TinyStories 的 loss = 1.5，困惑度是多少？**
   - 答案：e^1.5 ≈ 4.48，模型平均在约 4.5 个词里犹豫。
2. **均匀瞎猜时的 loss 和困惑度？**
   - 我的答案：1 − log V。
   - 订正：loss = log V，困惑度 = V ⚠️。
3. **`self.state[p]` 存什么？**
   - 我的答案：参数的梯度。
   - 订正：梯度在 `p.grad` 里 ⚠️。state 存跨步骤的历史：SGD 存 t，AdamW 存 t、m、v。
4. **m、v 为什么不能存在 `self.m` 上？**
   - 答案：每个参数要有自己的 m、v，形状各不相同 ✅。
5. **adapter 为什么返回 `AdamW` 而不是 `AdamW()`？**
   - 答案：测试会自己用模型参数和超参数构造优化器 ✅。
6. **权重衰减为什么用原始 lr？**
   - 我的答案：不太清楚。
   - 订正：α_t 只是用来修正 m、v 的偏差的，衰减和它们无关，要每步稳定地乘 (1−αλ)。
7. **θ=1、g=0.5、α=0.1、λ=0.01 时，第一步之后 θ 是多少？g 换成 50 呢？**
   <details><summary>答案</summary>约 0.8991；g=50 时几乎一样，因为 m/√v ≈ ±1。</details>
8. **AdamW 的 t 为什么必须从 1 开始？**
   <details><summary>答案</summary>t=0 时 α_t = α·0/0 = nan。写法是先加 1 再用。</details>

#### 下一步
- adamw_accounting：AdamW 训练时的显存（参数、梯度、优化器状态、激活值）和 FLOPs。
### adamw_accounting（2026-10-30）

#### 背景 / 动机
- transformer_accounting 算的是"推理时"的参数和 FLOPs。真正训练时还要回答：一张卡的显存放得下多大的 batch？训练一次要多少天？
- 这是 handout 的书面题，没有代码和测试。

#### 符号
- P：参数量；B：batch size（每批样本数）；S、L、d、h、f、V 同上；每个 float32 占 4 字节。
- MFU（Model FLOPs Utilization，模型算力利用率）：实际达到的 FLOP/s 除以硬件峰值。

#### 常见错误
1. **漏了梯度这一类**：显存里有 4 类东西，分别是参数、梯度、优化器状态、激活值。梯度也是 P 个数，因为每个参数都有一个同形状的 `p.grad`。
2. **把激活值说成"输入的文本"**：token id 很小，真正占显存的是前向的中间结果，backward 要用它们。
3. **GB 换算错**：16P = 262.4 亿字节 = 26.24 GB，不是 244 GB。÷10⁹ 是 GB，÷2³⁰ 是 GiB（约 24.4）。
4. **block 流程里多放了 ln_final 和 lm_head**：它们在 L 个 block 之后只出现一次。还漏了两次残差相加。
5. **把 B 当成了 GB**：B 是样本数，必须是整数，80 GB 下最多 B=3。
6. **TFLOP/s 换算错**：495 TFLOP/s = 4.95e14，不是 495e13。T 是 10¹²。
7. **分母没加括号**：写成 `/50%*495e13`，会先除以 50% 再乘，要写成 `/(0.5×4.95e14)`。

#### 值得注意
1. **显存的 4 类**：
   - 参数：P
   - 梯度：P
   - AdamW 状态（m、v）：2P
   - 前三类合计 4P 个 float32，也就是 16P 字节。GPT-2 XL 是 26.24 GB，一张 24 GB 的卡光这些就放不下。
2. **激活值公式**：元素数 = L(8BSd + 3BSf + 2BhS²) + BSd + 2BSV，乘 4 得到字节数。
3. **GPT-2 XL 的显存 ≈ 15.53·B + 26.24 GB**：80 GB 最多放 B=3（72.8 GB），B=4 就是 88.4 GB，超了。
4. **S=1024 时激活值的分布**：
   - 2BhS²（QKᵀ 和 softmax）：约 65%
   - 3BSf：约 16%
   - 8BSd：约 16%
   - block 外：约 3%
   - (B,h,S,S) 矩阵是大头，这就是 FlashAttention 不存它、backward 时按块重算的原因。
5. **AdamW 每个参数约 14 FLOPs，一步共 14P**：相比前向加反向的 6P·BS（B=1 时约 6144P）可以忽略。α_t 是每步只算一次的标量，不乘 P。
6. **反向 ≈ 2 倍前向**：Y=XW 的反向要算 dX = dY·Wᵀ 和 dW = Xᵀ·dY，是两次同样大小的矩阵乘。所以训练每个参数、每个 token 约 6 FLOPs。dW 要用前向时的 X，所以激活值必须存下来。
7. **训练时间**：GPT-2 XL，40 万步，B=1024，S=1024，单张 H100 fp32（495 TFLOP/s），MFU 50%。
   - 总 FLOPs = 3.52e12 × 3 × 1024 × 4e5 ≈ 4.33e21
   - ÷ 2.475e14 ≈ 1.75e7 秒 ≈ **202 天**
   - 所以真实训练要多卡并行，还要用 bf16。

#### 小数值例子
- 一个只有 P = 1,000,000 个参数的小模型，fp32：参数 4 MB + 梯度 4 MB + m、v 8 MB = 16 MB，也就是 16P 字节。
- GPT-2 XL：P ≈ 1.64×10⁹，16P ≈ 26.24 GB；B=1 时激活值约 15.53 GB，合计约 41.8 GB。

#### 回忆题
1. **用 AdamW 训练时，显存里存哪 4 类东西？**
   - 我的答案：输入、参数、优化器状态。
   - 订正：漏了梯度 ⚠️。另外"输入"应该是激活值。
2. **GPT-2 XL 的 16P 字节是多少 GB？**
   - 我的答案：244 GB。
   - 订正：26.24 GB ⚠️。
3. **哪个激活值和 S² 成正比？**
   - 答案：QKᵀ 和 softmax，形状 (B,h,S,S) ✅。
4. **为什么反向约是前向的 2 倍？**
   - 答案：要同时算 dX 和 dW 两次矩阵乘。
5. **单张 H100 训练 GPT-2 XL 要多久？**
   - 答案：约 202 天。
6. **训练每个参数、每个 token 大约多少 FLOPs？**
   <details><summary>答案</summary>约 6：前向 2，反向 4（dX 和 dW 两次矩阵乘）。</details>
7. **为什么激活值必须存下来？**
   <details><summary>答案</summary>反向算 dW = Xᵀ·dY 要用前向时的输入 X。</details>

#### 下一步
- §4.4 cosine 学习率调度，从 warmup 的动机开始。

## Day — 学习率调度、梯度裁剪与数据加载（2026-10-30 ~ 2026-11-04）

#### 做了什么
- §4.4 带 warmup 的 cosine 学习率调度：`cs336_basics/lr_schedule.py` 的 `lr_schedule`，adapter `run_get_lr_cosine_schedule`，`test_get_lr_cosine_schedule` PASSED。
- §4.5 梯度裁剪：`cs336_basics/grad_clip.py` 的 `gradient_clipping`，adapter `run_gradient_clipping`，`test_gradient_clipping` PASSED。
- §5.1 数据加载：`cs336_basics/load_dataset.py` 的 `data_loading`，adapter `run_get_batch`，`test_get_batch` PASSED。
- 2026-11-03：给 `cs336_basics` 下所有模块加了中文 docstring 和行内注释（见本节最后）。

### §4.4 cosine 学习率调度（lr_schedule）

#### 背景 / 动机
- 学习率（lr，learning rate）不应该从头到尾不变。调度器（scheduler）就是一个函数：输入当前步数 t，输出这一步用的 lr。LLaMA 用的是"线性 warmup + cosine 退火"。
- **为什么要 warmup（预热）**：
  1. 刚开始参数是随机初始化的，loss 很大，梯度又大又乱，一上来就用大 lr 容易把参数推到很差的地方，甚至 loss 变成 NaN。
  2. Adam 的 v 在前几步只用很少几个梯度估计出来，还不可靠。v 估小了，m/√v 就偏大，实际步长过大。
  3. 所以先用很小的 lr，让 m、v 和参数先稳定下来，再逐步升到 α_max。
- **为什么要衰减（decay）**：接近最低点时，大步长会一直从最低点上跳过去，在两侧来回震荡。lr 变小后才能稳稳落进去。
  - 例子：loss = w²，梯度是 2w，更新 $w \leftarrow w - \alpha\cdot2w = (1-2\alpha)w$。α=1 时系数是 −1，w=0.1 → −0.1 → 0.1 …… 永远在最低点两边跳；α=0.1 时系数 0.8，0.1 → 0.08 → 0.064 …… 稳稳收敛。真实训练里梯度还有噪声，后期 lr 不降下来，参数会一直在最低点附近抖。
- 位置：训练循环里每一步先算 `lr = lr_schedule(t, ...)`，写进 `optimizer.param_groups[i]["lr"]`，再 `optimizer.step()`。

#### 公式（三段）
$$\alpha_t = \begin{cases}\dfrac{t}{T_w}\,\alpha_{max} & t < T_w \quad\text{(warmup)}\\[2mm] \alpha_{min} + \dfrac12\left(1+\cos\!\Big(\dfrac{t-T_w}{T_c-T_w}\pi\Big)\right)(\alpha_{max}-\alpha_{min}) & T_w \le t \le T_c \quad\text{(cosine 退火)}\\[2mm] \alpha_{min} & t > T_c \quad\text{(退火之后)}\end{cases}$$
- $t$：当前步数；$T_w$：warmup 的步数；$T_c$：cosine 退火结束的那一步；$\alpha_{max}$、$\alpha_{min}$：最大和最小学习率。
- **端点检查**：
  - $t=T_w$：cos 0 = 1，$\alpha = \alpha_{min} + (\alpha_{max}-\alpha_{min}) = \alpha_{max}$（不是中间值）。
  - $t=T_c$：cos π = −1，$\alpha = \alpha_{min}$。
  - 正中间 $t=(T_w+T_c)/2$：cos(π/2) = 0，$\alpha = (\alpha_{max}+\alpha_{min})/2$。
  - warmup 段在 $t=T_w$ 时也正好到 α_max，所以曲线是连续的。

#### 输入 / 输出 / 形状
- `lr_schedule(t, t_w, t_c, lr_min, lr_max) -> float`，纯标量计算，没有张量，没有参数。
- adapter 的参数名不一样：`run_get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)`，按名字一一传过去。

#### 实现步骤
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

#### 小数值例子
- α_max = 1e-3，α_min = 1e-4，令 $p = (t-T_w)/(T_c-T_w)$ 表示退火进度：

  | p | ½(1+cos pπ) | α_t |
  |---|---|---|
  | 0 | 1 | 1.00e-3 |
  | 0.25 | 0.854 | 8.68e-4 |
  | 0.5 | 0.5 | 5.50e-4 |
  | 0.75 | 0.146 | 2.32e-4 |
  | 1 | 0 | 1.00e-4 |

- 曲线形状：开头和结尾下降得慢，中间下降得快，像半个倒过来的 cos 波。
- 带上 warmup，T_w=100，T_c=1000：t=0 → 0；t=50 → 5e-4；t=100 → 1e-3；t=550 → 5.5e-4；t=1000 → 1e-4；t=2000 → 1e-4。

#### 常见错误
1. **说不出为什么要 warmup**：答案是上面的三条（初始梯度大且乱、Adam 的 v 前几步不可靠导致实际步长偏大、先小步稳定再升）。
2. **以为 t=T_w 时 lr 是中间值**：t=T_w 时 cos 0 = 1，lr 正好是 α_max；中间值 (α_max+α_min)/2 出现在 t=(T_w+T_c)/2。
3. **把 `lr_t` 写进函数签名当输入**：lr_t 是要算出来的输出，不是参数。
4. **忘了 `import math`**：`math.cos`、`math.pi` 都要用到。
5. **adapter 里 `return` 后面留了死掉的 `raise NotImplementedError`**（现在还在，可以删）。
6. **T_c == T_w 会除以 0**：t = T_w 时进入中间分支，分母 T_c − T_w = 0（注释里已标出）。

#### 值得注意的点
1. 调度器只返回一个数，不碰参数。训练循环要自己把这个数写进优化器的 `param_groups`。
2. warmup 从 t=0 开始，第 0 步 lr = 0，这一步实际上不更新参数。
3. handout 脚注提到也有让 lr 周期性回升的调度（restarts），用来跳出局部最优；A1 只用单次退火。
4. AdamW 内部的 α_t 偏差修正和这里的调度是两回事：调度决定"基础 lr"α，AdamW 再在它上面乘 √(1−β₂ᵗ)/(1−β₁ᵗ)。

#### 回忆题
1. **为什么需要 warmup？**
   <details><summary>答案</summary>随机初始化时梯度大而乱；Adam 的 v 前几步不可靠，m/√v 偏大，实际步长过大。先用小 lr 稳定，再升到 α_max。</details>
2. **为什么后期要把 lr 降下来？**
   <details><summary>答案</summary>接近最低点时大步长会一直跳过最低点、来回震荡。loss=w² 的例子里 α=1 时系数 −1，永远在 ±w 之间跳。</details>
3. **t=T_w、t=T_c、t=(T_w+T_c)/2 时 lr 分别是多少？**
   <details><summary>答案</summary>α_max、α_min、(α_max+α_min)/2。</details>
4. **α_max=1e-3、α_min=1e-4，退火进行到一半时 lr 是多少？**
   <details><summary>答案</summary>5.5e-4。</details>

### §4.5 梯度裁剪（gradient clipping）

#### 背景 / 动机
- 训练中偶尔会碰到一个异常的 batch（比如很长的重复文本、脏数据），梯度突然变得很大，比如平常是 1，这一步变成 1000。这一步会把参数推得很远，loss 突然飙升（loss spike），甚至变成 NaN，之前的训练成果被破坏。
- 调小 lr 解决不了：lr 对每一步都一样，调小了正常的步也变慢；而且步长正比于 lr × 梯度，梯度大 1000 倍的那一步照样大 1000 倍。
- 梯度裁剪只在梯度"过长"时把它缩短到阈值 M，正常的步不受影响。
- 位置：训练循环里 `loss.backward()` 之后、`optimizer.step()` 之前。

#### 公式
- 全局 L2 范数：把所有参数的梯度拉平、拼成一个长向量 g：

  $$\|g\|_2 = \sqrt{\sum_{\text{参数 } i}\ \sum_{\text{元素}} g^2} = \sqrt{\|g_1\|^2 + \|g_2\|^2 + \cdots}$$

- 如果 $\|g\|_2 > M$：每个梯度都乘以同一个系数 $\dfrac{M}{\|g\|_2+\varepsilon}$，$\varepsilon = 10^{-6}$。裁剪后范数略小于 M。否则不动。
- $M$：`max_l2_norm` 阈值；$g_i$：第 i 个参数的梯度张量（形状和参数一样）。

#### 输入 / 输出 / 形状
- `gradient_clipping(parameters, max_l2_norm, eps=1e-6) -> None`
- `parameters`：可迭代的参数（可能是生成器，例如 `model.parameters()`），每个 `p.grad` 和 p 同形状，也可能是 None。
- 没有返回值，直接原地修改每个 `p.grad`。

#### 实现步骤
```python
grads = [p.grad for p in parameters if p.grad is not None]   # 先存成 list，跳过 None
total_norm = torch.sqrt(sum((g ** 2).sum() for g in grads))   # 先各自平方和，加起来，最后开一次根号
if total_norm > max_l2_norm:
    scale = max_l2_norm / (total_norm + eps)
    for g in grads:
        g *= scale                                             # 原地缩放，p.grad 跟着变
```
- 为什么先存成 list：`parameters` 可能是生成器，只能遍历一次，算完范数再遍历一次去缩放时就空了。
- 为什么用 `g *= scale`：原地修改才会改到 `p.grad` 本身；`g = g * scale` 只是让局部变量指向一个新张量。

#### 小数值例子
- 两个参数的梯度合起来是 (30, 40)：范数 = √(900+1600) = 50，M=5，系数 = 5/50.000001 ≈ 0.1，得到 (3, 4)，范数 ≈ 5，方向不变。
- 如果按参数各自裁剪（每个参数的范数各自不超过 5）：30 → 5，40 → 5，得到 (5, 5)。方向从约 53° 变成了 45°，更新方向被扭曲了。
- 为什么不对每个 g 各自算 L2：裁剪的对象是整个模型这一步的更新，也就是拼起来的长向量。总范数的平方等于各参数范数的平方和：例如 g₁ = [[1,2],[2,0]]（平方和 9，范数 3），g₂ = [4,0,0]（范数 4），总范数 = √(9+16) = 5，不是 3+4=7。M=5 时不用裁剪。代码里其实也算了每个 g 的平方和，只是不各自开根号比较，而是先加起来再开一次根号。
- 某个参数单独看不大，不代表整体不大；统一缩放同时保证两点：方向不变，整步总长度不超过 M。

#### 常见错误
1. **对列表直接用 `torch.pow(grads, 2)`**：grads 是 Python 列表，里面的张量形状各不相同，要对每个 g 各自求平方和再相加。
2. **漏了开根号**：算出来的是平方和不是范数。梯度 (3, 4) 时平方和 25、范数 5，阈值 5 本来不该裁剪，拿 25 去比就会误裁。
3. **参数名写成 `max_12_norm`（数字 12）**：应该是 `max_l2_norm`（字母 l），adapter 按名字传参会对不上。
4. **没用到的 `import math`**。
5. **adapter 里 `return` 后面的 `raise NotImplementedError` 执行不到**（现在还在，可以删）。
6. **所有 grad 都是 None 时会报错**：`sum` 对空序列返回 Python 整数 0，`torch.sqrt(0)` 收到的是 int，会报 TypeError（注释里已标出）。

#### 值得注意的点
1. 跳过 `p.grad is None` 的参数：没参与前向的参数没有梯度。
2. 裁剪后的范数是 M·‖g‖/(‖g‖+ε)，略小于 M，所以测试和 PyTorch 的 `clip_grad_norm_` 对得上（PyTorch 默认 ε 也是 1e-6）。
3. 裁剪不改方向，只改长度；而 Adam 的 m/√v 本身也有一定的"自动缩放"，裁剪主要是防止个别异常步把 m、v 和参数一起带歪。
4. 实践中会打印每一步裁剪前的总范数，突然变大往往是 loss spike 的前兆。

#### 回忆题
1. **为什么调小 lr 解决不了个别 batch 梯度爆炸的问题？**
   <details><summary>答案</summary>lr 对每步都一样，调小了正常步也变慢；异常那一步的步长照样是正常步的 1000 倍。裁剪只缩短过长的那一步。</details>
2. **两个参数的梯度范数分别是 3 和 4，总范数是多少？M=5 要不要裁？**
   <details><summary>答案</summary>√(9+16) = 5，不用裁（不大于 M）。</details>
3. **(30, 40)、M=5，全局裁剪和逐参数裁剪分别得到什么？**
   <details><summary>答案</summary>全局得到 (3, 4)，方向不变；逐参数得到 (5, 5)，方向变了。</details>
4. **为什么先把 grads 存成 list？为什么用 `g *= scale`？**
   <details><summary>答案</summary>parameters 可能是只能遍历一次的生成器；原地乘才会修改 p.grad 本身。</details>

### §5.1 数据加载（get_batch）

#### 背景 / 动机
- 训练数据是一整条很长的 token id 序列（tokenize 完的语料拼起来，可能有几亿个 token）。每一步要从里面随机取一个 batch 的 (x, y) 送进模型。
- 语言模型的目标是预测下一个 token，所以 **y 就是 x 整体往后错一位**：y 的每个位置都是 x 对应位置的"下一个 token"。
- 这样取数据的好处：任何起点都是合法样本，采样很简单；所有样本一样长，不用 padding；不用把整个数据集读进内存。
- 位置：`dataset`（memmap）→ **get_batch** → x 进模型得到 logits，y 和 logits 一起算交叉熵。

#### 公式
- 数据 $d_0, d_1, \dots, d_{n-1}$，长度 n，context_length = L。随机选起点 i：

  $$x = [d_i, \dots, d_{i+L-1}],\qquad y = [d_{i+1}, \dots, d_{i+L}]$$

- y 的最后一个下标 $i+L \le n-1$，所以 $i \in [0,\ n-L-1]$，一共 $n-L$ 个合法起点。
- `np.random.randint(low, high, size)` 的上界是开区间，所以写 `randint(0, n - L, size=B)`，最大能取到 n−L−1。

#### 输入 / 输出 / 形状
- `data_loading(dataset, batch_size, context_length, device=None) -> (x, y)`
- `dataset`：一维 numpy 整数数组（或 memmap），长度 n。
- `x`、`y`：都是 `torch.long`，形状 `(batch_size, context_length)`，放在 `device` 上。没有参数。
- 后面的形状：x `(B, L)` → embedding `(B, L, d_model)` → … → logits `(B, L, V)` → `cross_entropy(logits, y)`。y 不进模型，只在算 loss 时当标签（target）。
- **batch_size 是"一批里有几条样本"**（行数），不是"一共有几批"。一共多少批是训练步数，每一步调用一次 get_batch。总共看过的 token 数 = 步数 × B × L。
- 每一行是一条样本，但一行里同时有 L 个"预测下一个 token"的小任务：因为有因果 mask，位置 t 只看得到 0..t。所以一个 batch 一共贡献 B×L 个预测，交叉熵对这么多个位置取平均。

#### 实现步骤
```python
def data_loading(dataset, batch_size, context_length, device=None):
    n = len(dataset)
    starts = np.random.randint(0, n - context_length, batch_size)          # B 个起点，范围 [0, n-L-1]
    x = np.stack([dataset[i : i + context_length] for i in starts])        # (B, L)
    y = np.stack([dataset[i + 1 : i + 1 + context_length] for i in starts])  # (B, L)，错一位
    x = torch.tensor(x, dtype=torch.long).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)
    return x, y
```
- 文件一开始叫 `get_batch.py`，后来改名成 `cs336_basics/load_dataset.py`，函数叫 `data_loading`；adapter `run_get_batch` 调用它。

#### 小数值例子
- 数据 `[5, 8, 2, 9, 4, 7, 1]`，n=7，L=3。起点 i=1：x = `[8, 2, 9]`，y = `[2, 9, 4]`。模型看到 8 要预测 2，看到 8、2 要预测 9，看到 8、2、9 要预测 4。
- 起点最大值：i + 3 ≤ 6，所以 i ≤ 3。i=3 时 x = `[9, 4, 7]`，y = `[4, 7, 1]`，正好用到最后一个 token。i=5 时 x 就已经越界了。
- B=2，起点 1 和 3：

  ```
  x = [[8, 2, 9],     y = [[2, 9, 4],
       [9, 4, 7]]          [4, 7, 1]]
  ```

  形状都是 (2, 3)。

#### 大数据集：memmap
- 真实数据集太大，读不进内存。用 `np.memmap("train.bin", dtype=np.uint16, mode="r")`，或者保存时用了 `np.save` 就用 `np.load("train.npy", mmap_mode="r")`。得到的数组看起来是完整的，只有切到的那一小段才会从磁盘读进来。`get_batch` 本身不用改。
- memmap 接收的是**文件路径**，不是已经读进内存的数组，所以是在训练脚本加载数据时用，不是写成 `dataset = np.memmap(dataset)`。
- **dtype 必须和存文件时一致**：memmap 文件本身不记录 dtype，读错了数全是乱的。vocab < 65536 时用 uint16（2 字节），比 int64 省 4 倍空间。handout 建议读完检查一下最大值是否小于 vocab_size。

#### 常见错误
1. **以为 x、y 的形状是 seq_len 和 d_model**：两者都是 token id，形状都是 (B, L)。d_model 是进了 embedding 之后才出现的。
2. **以为 batch_size 是"一共多少批"**：它是一批里的样本数（行数）。
3. **起点最大值答成 5**：n=7、L=3 时最大是 3，一般式 n−L−1。
4. **第一版**：`torch.long(np.random.randint(batch_size), ...)`。randint 只返回一个 [0, B) 的整数，和数据长度无关；完全没用到 dataset；把数据类型 `torch.long` 当函数调用；只返回一个值，不是 (x, y)。
5. **第二版**：`np.random.randint(0, batch_size, n - context_length)` 把上界和个数颠倒了；`[i, context_length]` 只是一个两元素列表，不是切片。
6. **第三版**：`dataset[i, i+context_length]` 用了逗号，在一维数组上当成二维索引，报 `IndexError: too many indices`，要用冒号；上界写成 `n - context_length + 1`，可能取到 i = n−L，这时 y 只剩 L−1 个元素，`np.stack` 报错（差一，off-by-one）。
7. **没用到的 `import torch.nn as nn`**（现在还在）。

#### 值得注意的点
1. x 是模型输入，y 是标签；y 不进模型。
2. 每一行的起点独立随机，同一个 batch 里的样本可以重叠。
3. `torch.tensor(..., dtype=torch.long)` 会复制一份数据。uint16 的 memmap 切片转成 long 后才能当 embedding 的索引。
4. 放到 device 上：模型在 GPU 上时，x、y 也必须在同一个 device 上。

#### 回忆题
1. **x 和 y 是什么关系？形状是什么？**
   <details><summary>答案</summary>y 是 x 往后错一位；都是 (batch_size, context_length) 的 token id。</details>
2. **数据长度 n、context_length L，起点的合法范围？`randint` 的上界写什么？**
   <details><summary>答案</summary>[0, n−L−1]，共 n−L 个；上界是开区间，写 n−L。</details>
3. **batch_size=4、L=3、训练 1000 步，一共看过多少个 token？一个 batch 有多少个预测？**
   <details><summary>答案</summary>1000 × 4 × 3 = 12000；一个 batch 有 4 × 3 = 12 个预测。</details>
4. **`np.memmap` 接收什么？dtype 为什么要写对？**
   <details><summary>答案</summary>文件路径；文件不记录 dtype，读错了数据全是乱的。</details>
5. **`dataset[i, i+L]` 和 `dataset[i : i+L]` 有什么区别？**
   <details><summary>答案</summary>前者是二维索引（第 i 行第 i+L 列），一维数组上报 IndexError；后者是切片，取 L 个连续元素。</details>

### 2026-11-03：全部模块加中文注释

- `cs336_basics` 下的各个模块都加了中文 docstring 和行内注释：每个类和函数写了功能、输入输出、张量形状、可学习参数的形状和用到的公式。只加注释，逻辑没动，测试结果改前改后一致。改动还没提交。
- 顺手发现的可疑写法，都用 `# 注意` 标出来，没有改：
  1. `transformer_lm.py`：`forward` 没有把 `token_positions` 传给各个 block，RoPE 永远用 0..S−1。
  2. `lr_schedule.py`：T_c == T_w 时中间分支除以 0。
  3. `grad_clip.py`：所有 grad 都是 None 时 `torch.sqrt(0)` 报错。
  4. `tokenizer.py`：没有缓存，比较慢；special token 必须已经在 vocab 里，否则 encode 报 KeyError；第一行的旧注释已经过时。
  5. train_bpe 几个版本每次运行都会覆盖当前目录下的 `merges.txt`；分桶版会给每个 worker 写一个 `worker_*.prof`（`a1-basics` 下那些 prof 文件就是这么来的）。

#### 下一步
- §5.2 checkpointing（保存和恢复训练）：先想清楚断点续训时，除了模型参数还要存什么（优化器状态 m、v、t，以及当前步数）。

## Day — Checkpointing、训练循环、编码与生成（2026-11-04 ~ 2026-11-05）

#### 做了什么
- §5.2 Checkpointing：`cs336_basics/model_checkpoint.py` 的 `save_checkpoint` / `load_checkpoint`，`test_checkpointing` PASSED。
- 训练循环：`cs336_basics/train.py`，串起 get_batch → 前向 → 交叉熵 → 反传 → 梯度裁剪 → AdamW → 验证 / 存盘。
- BPE 改存 pickle，并写 `cs336_basics/encode_dataset.py` 把 TinyStories 编成 `uint16` 的 `.bin`。
- 装上带 CUDA 的 PyTorch（cu128，阿里云镜像），在 RTX 3060 Laptop 6GB 上训练到约 21000–22000 步，验证集 loss 约 1.4。
- §6 文本生成：`cs336_basics/generate.py`，贪心 / temperature / top-p（nucleus sampling，核采样），从第 22001 步 checkpoint 续写故事。
- 日志和代码注释改动大多还没提交（git commit），本轮也不提交。

### §5.2 Checkpointing（`cs336_basics/model_checkpoint.py`）

#### 背景 / 动机
- 训练动辄几万步，中途断电、杀进程、换机器都很常见。断点续训（resume）要能从上次停的地方接着跑，不能只靠「模型参数」。
- **一定要存的三样**：
  1. **模型参数**：`model.state_dict()`（权重 W、偏置等）。
  2. **优化器状态**：AdamW 的每个参数的一阶矩 m、二阶矩 v，以及时间步 t。少了它们，续训等于重新冷启动 Adam，前面 warmup / 自适应缩放都白做。
  3. **当前 iteration（迭代步数）**：学习率调度按步数算，日志和存盘间隔也靠它。
- **不用存梯度**：`p.grad` 是这一步 `backward` 临时算出来的，下一步会重新算。存了也没用，还白白占盘。

#### 输入 / 输出 / 形状
- `save_checkpoint(model, optimizer, iteration, out)`：无返回值。`out` 可以是文件路径字符串，也可以是已经打开的文件对象（file-like）。
- `load_checkpoint(src, model, optimizer) -> int`：把状态装进**已经建好**的 model 和 optimizer，返回保存时的 `iteration`。
- 存盘内容是一个 dict：

  ```python
  {
      "model": model.state_dict(),          # 名字 → 张量
      "optimizer": optimizer.state_dict(),  # 含 state（m、v、t）和 param_groups
      "iteration": iteration,              # int
  }
  ```

#### 实现步骤
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
- `state_dict` 只按**名字**存张量，不存网络结构。所以 load 之前必须先用同样的超参把 `TransformerLM` 和 `AdamW` 构造出来，再 `load_state_dict` 把数值填回去。
- PyTorch 2.6+ 默认 `weights_only=True`：`torch.save` 里不要塞自定义 Python 对象，只放张量 / 基本类型，否则反序列化（unpickle）会报错或被拒绝。

#### 小数值例子
- 训到第 2000 步存盘：`iteration=2001`（见训练循环：存「下一步要从哪开始」）。文件里有 model 的全部权重、AdamW 每个参数的 m/v、以及 t。
- 重启后：先 `TransformerLM(...)`、`AdamW(...)`，再 `start_it = load_checkpoint(...)` 得到 2001，`for it in range(2001, max_iters)` 接着跑，不会把第 2000 步再做一遍。

#### 常见错误
1. **以为梯度也要存**：梯度下一步 `backward` 会重算，存了浪费。
2. **类型别名写成 `dict[model, optimizer, iteration]`**：这不是合法的运行时类型，而且若真把类型对象写进 save 字典，`torch.load` 会 `UnpicklingError`。
3. **`load_checkpoint` 忘了 `return iteration`**：调用方拿不到步数，续训只能从 0 开始或自己猜。
4. **存了整个 optimizer 对象而不是 `optimizer.state_dict()`**：自定义 `AdamW` 类反序列化时对不上，`UnpicklingError`。只存 state_dict（纯张量 + 基本类型）才稳。
5. **不明白为什么 load 还要传入 model、optimizer**：因为文件里只有「名字 → 数值」，没有层怎么叠、优化器怎么构造；接收对象必须先建好壳子。

#### 值得注意的点
1. `out` / `src` 支持路径或文件对象，测试里两种都会用到。
2. 续训时学习率不要写死在 optimizer 初始 lr 上：每步仍用 `lr_schedule(t=it, ...)` 写进 `param_groups`，这样从 2001 步接上时 lr 曲线连续。
3. `test_checkpointing` PASSED。

#### 回忆题
1. **断点续训最少要存哪三样？梯度要不要存？**
   <details><summary>答案</summary>模型 state_dict、优化器 state_dict（含 m、v、t）、iteration。梯度不用存，下一步 backward 会重算。</details>
2. **为什么 `load_checkpoint` 必须传入已经建好的 model 和 optimizer？**
   <details><summary>答案</summary>state_dict 只按名字存张量，不存网络结构和优化器类；要先构造好对象再把数值 load 进去。</details>
3. **存整个 optimizer 对象会怎样？**
   <details><summary>答案</summary>反序列化时类对不上，容易 UnpicklingError；应只存 state_dict。</details>

### 训练循环（`cs336_basics/train.py`）

#### 背景 / 动机
- 前面的模块（模型、loss、AdamW、lr 调度、裁剪、get_batch、checkpoint）都是零件；`train.py` 把它们按正确顺序拧成一条能跑完几万步的流水线。
- 顺序错了会出现「用旧梯度更新」「裁剪没生效」「lr 调度看起来写了但 AdamW 根本没读到」这类隐性 bug。

#### 每一步的正确顺序
1. **算本步 lr，写入 `optimizer.param_groups[i]["lr"]`**。AdamW 的 `step()` 读的是 `group["lr"]`，不是 `optimizer.lr` 这种属性。
2. **`get_batch` / `data_loading`** 取 `(x, y)`。
3. **前向**：`logits = model(x)`，`loss = cross_entropy(logits, y)`。
4. **`optimizer.zero_grad()`**：清掉上一步残留的 `.grad`。
5. **`loss.backward()`**：算本步梯度。
6. **`gradient_clipping(...)`**：必要时缩小总梯度范数。
7. **`optimizer.step()`**：用 AdamW 更新参数。
8. **日志 / 验证 / 存盘**（按 interval）。

验证集：`model.eval()` + `torch.no_grad()`，对 `eval_batches` 个 batch 取平均 loss，再 `model.train()`。存盘时 `iteration=it+1`，表示「下一步从哪开始」，避免 resume 把刚做完的那一步再跑一遍；`os.makedirs(..., exist_ok=True)` 先建好目录。

#### 输入 / 输出 / 形状（默认超参，TinyStories）
- 数据：`data/tinystories_train.bin`、`data/tinystories_valid.bin`（`uint16` memmap）。
- 模型：`vocab_size=10000`，`context_length=256`，`d_model=512`，`d_ff=1344`，`num_layers=4`，`num_heads=16`，`rope_theta=10000`。
- 训练：`batch_size=32`，`max_iters=40000`，`t_w=1000`，`t_c=40000`，`lr_max=1e-3`，`lr_min=1e-4`，`grad_clip=1.0`；`log_interval=100`，`valid_interval=1000`，`save_interval=2000`，`eval_batches=20`。

#### 实现步骤（骨架）
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
    # 按 interval：print loss；eval；save_checkpoint(..., iteration=it+1, ...)
```

#### 小数值例子
- 第 0 步、随机初始化、词表 10000：交叉熵期望约 \(\ln(10000) \approx 9.21\)，实测约 9.28，说明起步正常。
- 第 1000 步结束 warmup，`lr` 到 `lr_max`；之后余弦下降，中段降得最快。

#### 常见错误
1. **第一版顺序写反**：先 `step` 再 `backward`，裁剪写在 `step` 之后，lr 最后才设——等于用空/旧梯度更新，裁剪和调度都白写。
2. **memmap `dtype` 与存盘不一致**：数全是乱的；必须和编码时一样用 `uint16`。
3. **train / valid 指到同一个文件**：验证集 loss 没有独立意义。
4. **构造模型 / 优化器参数名或个数不对**：和 `TransformerLM`、`AdamW` 签名对不上。
5. **存盘 `iteration=it` 而不是 `it+1`**：resume 会把刚完成的那一步再做一次。

#### 值得注意的点
1. **Dropout（随机失活）简述**：训练时以概率 p 把激活置 0，再除以 \(1-p\)（inverted dropout，反转 dropout），保证 \(\mathbb{E}[x]=x\)。大语言模型（LLM，large language model）预训练通常不用 dropout，本作业默认也没有。
2. 验证时务必 `eval()` + `no_grad()`，结束要 `train()`，否则 BatchNorm/Dropout 类行为会一直停在推理模式（本模型虽无 BN/Dropout，养成习惯）。
3. 训练 loss 单 batch 噪声大（±0.1 来回跳），看趋势以 val_loss 为准。

#### 回忆题
1. **一步训练的正确顺序是什么？lr 要写到哪里？**
   <details><summary>答案</summary>设 param_groups["lr"] → get_batch → forward+loss → zero_grad → backward → clip → step → 日志/验证/存盘。AdamW.step 读的是 group["lr"]。</details>
2. **为什么 save 时用 `iteration=it+1`？**
   <details><summary>答案</summary>表示下一步要从哪开始，resume 时 `range(start_it, max_iters)` 不会重复做刚完成的那一步。</details>
3. **验证循环要注意什么？**
   <details><summary>答案</summary>`model.eval()` + `torch.no_grad()`，多 batch 取平均，然后 `model.train()`。</details>

### BPE pickle 与数据集编码（`cs336_basics/encode_dataset.py`）

#### 背景 / 动机
- 旧的 `vocab*.txt` / `merges*.txt` 把 `bytes` 解码成文本再存：空格、`' t'` 和两个空格、以及坏掉的 UTF-8 片段会对不上，读回来和训练时的 BPE（Byte Pair Encoding，字节对编码）不一致。
- 解决办法：重新训练 vocab_size=10000 的 TinyStories BPE（带特殊 token `<|endoftext|>`），用 `pickle.dump` 直接存二进制结构：

  ```python
  {"vocab": dict[int, bytes], "merges": list[tuple[bytes, bytes]]}
  ```

  路径：`artifacts/tinystories_bpe/bpe.pkl`。结果：词表 10000，merges 9743 条，`<|endoftext|>` 的 id = 0。
- **Tokenizer 不是神经网络**：没有可学习参数，只有固定的 merge 规则和词表查表。所谓 BPE「训练」只是统计频次、决定合并顺序。

#### 输入 / 输出 / 形状
- `encode_file(tokenizer, in_path, out_path, chunk=1_000_000) -> total_tokens`
- 输入：UTF-8 文本；输出：裸 `uint16` 小端字节流（每个 token id 占 2 字节）。
- TinyStories：**训练集**既用来训 BPE，也用来训语言模型；**验证集**只做编码，不参与 BPE 训练。

#### 实现步骤
1. `pickle.load` 出 vocab / merges，构造 `Tokenizer(..., special_tokens=["<|endoftext|>"])`。
2. `tokenizer.encode_iterable(fin)` 流式产出 id，不要一次性读进整个 2GB 文本。
3. 攒满 `chunk` 个就 `np.array(buf, dtype=np.uint16).tobytes()` 写入；Python `list` 里每个 int 大约 28–36 字节，`uint16` 只有 2 字节。
4. 断言：文件大小 == `total * 2`；`memmap` 长度 == `total`；`max id < 10000`。

#### 小数值例子 / 实测
- valid：5,461,210 个 token；train：540,796,778 个 token；最大 id = 9999。
- 约 4.12 字节原文 / token（和 BPE 压缩率观感一致）。
- 解码前缀能对上原文；valid 从半句话开始，是因为 train 文件在词中间被切到下一个文件——数据集本身的切分现象，不是编码器 bug。
- 为加速，给 Tokenizer 加了缓存（cache）：相同 pretok 命中时直接返回 id 列表。

#### 常见错误
1. **继续用文本格式存 vocab/merges**：空格和坏 UTF-8 歧义，编码结果和 BPE 训练不一致。
2. **一次性 `encode(整个文件)`**：内存和速度都爆。
3. **存成 int64 / Python list 落盘**：体积比 uint16 大很多。
4. **忘记 special token**：`<|endoftext|>` 必须在词表里，否则 encode 报 KeyError。

#### 值得注意的点
1. pickle 存的是 `bytes`，没有「解码成 str 再编码回来」的往返损失。
2. eot id = 0，生成时抽到 0 就停。
3. memmap 读 `.bin` 时 dtype 必须是 `uint16`。

#### 回忆题
1. **为什么改用 pickle 存 BPE，而不是 vocab.txt / merges.txt？**
   <details><summary>答案</summary>文本往返会把 bytes 歧义化（空格、残缺 UTF-8）；pickle 直接存 dict[int,bytes] 和 merges 列表。</details>
2. **Tokenizer 是神经网络吗？BPE「训练」在学什么？**
   <details><summary>答案</summary>不是；没有梯度。BPE 训练只统计频次、得到固定 merge 顺序和词表。</details>
3. **为什么用 uint16 流式写盘？**
   <details><summary>答案</summary>vocab&lt;65536，两字节够用；流式避免整份语料进内存；list 里每个 id 比 uint16 大一个数量级。</details>

### GPU 版 PyTorch 安装（简述）

#### 背景 / 动机
- 环境里原来是 `2.11.0+cpu`，`torch.cuda.is_available()` 为 False，训练只能走 CPU，太慢。
- 机器：RTX 3060 Laptop 6GB，够跑 TinyStories 配置（batch 32、context 256）。

#### 做法
- 用 `uv` 加 PyTorch cu128 索引；国内 `download.pytorch.org` 很慢时，换阿里云镜像，并设 `format = "flat"`（镜像是扁平 whl 目录，不是标准 pep503 索引）：

  ```toml
  [[tool.uv.index]]
  name = "pytorch-cu128"
  url = "https://mirrors.aliyun.com/pytorch-wheels/cu128/"
  format = "flat"
  explicit = true
  ```

- `UV_HTTP_TIMEOUT` 调大，避免大 whl 下到一半超时。装好后 `cuda.is_available()` 为 True。

### 训练实测结果

#### 过程
- 先 smoke（冒烟测试，短跑确认能通），再全量：laptop 3060，`batch_size=32`，`max_iters=t_c=40000`，`t_w=1000`，`lr_max=1e-3`。
- 中途有 `kernel cache` 目录建失败的警告，不影响数值，可忽略。

#### 数字
| 步数 | train loss（约） | val loss |
|------|------------------|----------|
| 0 | 9.28 | 9.28 |
| 1000 | 2.30 | 2.35 |
| 2000 | 1.90 | **1.95**（已低于 handout 目标 2.00） |
| 7000 | 1.68 | 1.63 |
| ~21000–22000 | ~1.4 | ~1.4 |

- train ≈ val，没有过拟合迹象（TinyStories 数据量远大于模型能背下的量）。
- 手动停在约 21000 步附近；生成时 `load_checkpoint` 读到 **iteration = 22001**。

#### 回忆题
1. **为什么第 0 步 loss 大约是 9.2？**
   <details><summary>答案</summary>随机初始化时对词表近似均匀，交叉熵 ≈ ln(vocab_size)=ln(10000)≈9.21。</details>
2. **train loss 和 val loss 一直接近说明什么？**
   <details><summary>答案</summary>没有明显过拟合；模型容量相对数据还很小。</details>

### §6 文本生成（`cs336_basics/generate.py`）

#### 背景 / 动机
- 训练好的 LM 输出的是每个位置、每个词表项的 **logits**（未归一化分数）。要写成故事，需要把「最后一个位置的 logits」变成下一个 token id，接到 prompt 后面，循环直到遇到 `<|endoftext|>` 或达到 `max_new_tokens`。
- **贪心（greedy）**：每次取 `argmax`，同一 prompt 永远同一续写，还容易死循环重复。
- **采样（sampling）**：softmax 变成概率后用 `torch.multinomial` 按概率抽；再配合 **temperature（温度）** 和 **top-p / nucleus sampling（核采样）** 控制「有多敢胡说八道」。

#### 公式
- Softmax：\(\mathrm{softmax}(z)_i = e^{z_i}/\sum_j e^{z_j}\)。
- Temperature：先做 \(z \leftarrow z / \tau\) 再 softmax。
  - \(\tau \to 0\)：分布更尖，接近贪心。
  - \(\tau \to \infty\)：分布更平，接近均匀乱抽。
- Top-p：把概率从大到小排序，累加到累计质量达到 p 为止，其余置 0，再归一化后抽样。保留条件用 \((\mathrm{cumsum} - \mathrm{sorted\_probs}) < p\)，这样**第一次让累计和超过 p 的那个 token 仍保留**。

#### 输入 / 输出 / 形状
- `generate(model, tokenizer, prompt, max_new_tokens, temperature, top_p=None, device=None) -> str`
- 每步：`window = ids[-context_length:]` → `x` 形状 `(1, S)` → `logits = model(x)[0, -1, :]` 形状 `(vocab_size,)` → 抽出标量 id，`ids.append`。
- 加载权重：先建 `TransformerLM` + 占位 `AdamW`，再 `load_checkpoint`（它要 optimizer 壳子）；推理包在 `@torch.no_grad()` 里。

#### 实现步骤
1. `ids = tokenizer.encode(prompt)`（实例方法，不是 `Tokenizer.encode`）。
2. 循环最多 `max_new_tokens` 次：截窗口 → 前向 → 取最后位置 logits。
3. `temperature <= 0`：贪心 `argmax`；否则 `logits /= temperature`，softmax。
4. 若开启 top-p：`sort` → `cumsum` → `mask = (cumsum - sorted_probs) < top_p` → 置零 → 归一化 → `multinomial` → 用 `indices[...]` 映回真正 token id。
5. id == 0（eot）则停；否则 append。循环结束后 `tokenizer.decode(ids)`。

#### 小数值例子
- logits `[2, 1, 0]`，\(\tau=1\)：softmax ≈ `[0.67, 0.24, 0.09]`。
- \(\tau=0.1\)：≈ `[0.99995, 0.00005, …]`，几乎总是抽第一个。
- \(\tau=10\)：≈ `[0.37, 0.33, 0.30]`，接近均分。
- 概率 `[0.5, 0.3, 0.15, 0.05]`，p=0.9：累计 0.5→0.8→0.95，候选前 3 个；`cumsum - sorted` 为 `[0, 0.5, 0.8, 0.95]`，mask 在前三个为 True。

#### 常见错误
1. **`Tokenizer.encode(prompt)`**：那是类，没有实例方法；要用传入的 `tokenizer.encode`。
2. **`return` 写在 `for` 里面**：第一次循环就返回，只生成一个 token。
3. **`next_id` 还是 0 维 tensor**：`ids.append` 和比较会出问题，要 `int(...)`。
4. **没有 top-p 的分支写成 `argmax`**：`temperature>0` 但没开 top-p 时仍应 `multinomial(softmax(...))`。
5. **主程序传 `top_k=`，函数参数是 `top_p`**：参数名对不上。
6. **以为 `load_checkpoint` 返回 dict**：它返回的是 **int**（iteration）。
7. **prompt 末尾多一个空格**：BPE 会留下单独的空格 token，模型常接着采到带数字感的 merge（如 `8`、`8ic`），看起来像「decode 多了个数字」，其实是模型真生成的——prompt 尽量落在完整词边界。

#### 值得注意的点
1. 空 prompt 用 `<|endoftext|>` 开头时，模型会按训练分布开一篇新故事。
2. 贪心版常从 Lily/Sue 等高频套路开写；采样版名字和情节会变。
3. 「The little girl was」信息少，续写更容易短或绕。

#### 回忆题
1. **temperature 很小 / 很大时，分布分别怎样？**
   <details><summary>答案</summary>很小：更尖，接近贪心；很大：更平，接近均匀随机。</details>
2. **top-p 的 mask 为什么写成 `(cumsum - sorted_probs) < p`？**
   <details><summary>答案</summary>这是「加入当前 token 之前」的累计和；第一次跨过 p 的那个 token 仍保留。</details>
3. **没有 top-p 时，为什么还要先 softmax 再 multinomial？不能直接在词表上均匀随机吗？**
   <details><summary>答案</summary>multinomial 要的是非负且和为 1 的概率；均匀随机会丢掉模型学到的分布，logits 大的 token 应该更容易抽中。</details>
4. **prompt 末尾多空格，输出里冒出 `8` 是 decode bug 吗？**
   <details><summary>答案</summary>不是。空格被单独编码后，模型按训练分布采到了数字相关 token。</details>

#### 下一步
- 可选：学习率扫描（`lr_max` 取 3e-4 / 1e-3 / 3e-3 对比曲线与最终 val loss）。
- 可选：把剩下的步数跑完，看 val loss 是否还能从 ~1.4 再降。
- 若 handout 还有 OpenWebText 实验，可接着做；准备 A2 前先看职业分析再决定优先级。
- 日志与代码注释仍大多未提交；需要时再单独 commit，本轮不提交、不 push。
