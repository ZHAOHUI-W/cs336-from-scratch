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

## Day — RMSNorm 与 SwiGLU（2026-10-23 ~ 2026-10-24）

#### 做了什么
- §3.4.1 RMSNorm：`cs336_basics/rmsnorm.py` + `adapters.run_rmsnorm`。
- §3.4.2 SwiGLU 前馈网络：`cs336_basics/swiglu.py`（模块级 `silu` 函数 + `SwiGlu` 模块）+ `adapters.run_silu` / `adapters.run_swiglu`。
- 小清理：删掉 `embedding.py` 里没用到的 `math` / `einsum` import。

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

#### 下一步
- §3.4.3 RoPE（旋转位置编码）。

## Day — RoPE 旋转位置编码（2026-10-24 ~ 2026-10-26）

#### 做了什么
- §3.4.3 RoPE：`cs336_basics/rope.py`（`RoPE(theta, d_k, max_seq_len, device)`）+ `adapters.run_rope`。

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

#### 下一步
- §3.4.4 softmax（减最大值保证数值稳定），然后 scaled dot-product attention。
## Day — softmax、缩放点积注意力与多头自注意力（2026-10-26）

#### 做了什么
- §3.4.4 softmax：`cs336_basics/nn_utils.py` 的 `softmax(x, dim)` 和 `adapters.run_softmax`。
- §3.4.4 缩放点积注意力（SDPA，scaled dot-product attention）：`nn_utils.scaled_dot_product_attention(Q, K, V, mask)` 和 `adapters.run_scaled_dot_product_attention`。
- §3.4.5 因果多头自注意力（MHA，multi-head self-attention）：`cs336_basics/attention.py` 的 `MultiHeadSelfAttention`，以及 `adapters.run_multihead_self_attention`。
- 带 RoPE 的 MHA：构造时可选传入 `theta` / `max_seq_len`，对应 `adapters.run_multihead_self_attention_with_rope`。

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

#### 下一步
- §3.5 Transformer block（pre-norm：`y = x + MHA(RMSNorm(x))`，再 `z = y + FFN(RMSNorm(y))`）。

## Day — Transformer block 与 TransformerLM（2026-10-26 ~ 2026-10-27）

#### 做了什么
- §3.5 Transformer block：`cs336_basics/transformer.py` 的 `TransformerBlock`（pre-norm），以及 `adapters.run_transformer_block`。
- §3.5 TransformerLM：`cs336_basics/transformer_lm.py` 的 `TransformerLM`，以及 `adapters.run_transformer_lm`。

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

#### 下一步
- §4 交叉熵（cross-entropy），然后是 AdamW、学习率余弦调度、梯度裁剪。
## Day — Resource accounting：transformer_accounting（2026-10-27）

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

#### 下一步
- §4.1 交叉熵（cross-entropy）：已经答完概念题，接下来实现 `run_cross_entropy`。