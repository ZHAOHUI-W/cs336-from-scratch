# Week 1 日志（中文）

英文版见 [week-01.en.md](./week-01.en.md)。

## Day 1 — Map A1 (2026-09-16)

> 本日笔记原为英文规划，详见同目录 `week-01.en.md` 对应章节。

## Day — BPE `train_bpe` (2026-09-21)

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

## Day — Tokenizer encode/decode（2026-09-22）

#### 做了什么
- 实现 `cs336_basics/tokenizer.py` 的 `Tokenizer`：`__init__`、`encode`、`decode`、`encode_iterable`。
- 在 `tests/adapters.py` 的 `get_tokenizer` 中直接返回 `Tokenizer(vocab, merges, special_tokens)`。
- Windows 上为 `tests/test_tokenizer.py` 做了 `resource` 可选导入（无 `RLIMIT_AS` 时跳过硬内存上限）。
- 补齐 GPT-2 fixture（`gpt2_vocab.json` / `gpt2_merges.txt`），用于与 tiktoken 对齐的测例。

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

#### 验收
- `tests/test_tokenizer.py` 中 roundtrip 与 tiktoken 对齐相关用例已通过。

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

#### 下一步
- 若追 &lt;2 min hint：分段计时 + 视情况再优 merge / worker 数。
- Handout 下一题：`train_bpe_expts_owt`（vocab 32k），或进入 Transformer。


## Day — Linear 与 Embedding（2026-10-12 ~ 2026-10-23）

#### 做了什么
- §3.3.2 Linear（无 bias）：`cs336_basics/linear.py` + `adapters.run_linear`。
- §3.3.3 Embedding：`cs336_basics/embedding.py` + `adapters.run_embedding`。
- 修测试收集报错：`pretokenization_example.py` 的演示代码放进 `if __name__ == "__main__":`，避免 pytest 导入时执行 `open(...)`。

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