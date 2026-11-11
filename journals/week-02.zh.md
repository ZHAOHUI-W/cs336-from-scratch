# Week 2 日志（中文）

English version: [week-02.en.md](./week-02.en.md)。本周开始 Assignment 2（Systems），代码在 `assignments/a2-systems/`。

## Day — §2.1.4 Nsight Systems 性能分析（nsys_profile）（2026-11-11）

#### 做了什么
- 在 Windows 上用 Nsight Systems 2026.6.1 给 `cs336_systems/benchmark.py` 采集了三种模式：只前向（`--options 0`）、前向+反向（`--options 1`）、完整训练一步（`--options 2`，含梯度裁剪和 AdamW）。
- 用 `nsys stats --filter-nvtx` 按 NVTX 范围统计 kernel，核对了每步 GEMM 次数：前向 109、反向 218、合计 327，和手算完全一致。
- 发现并解释了一个坑：反向的 kernel 由 autograd 的另一个线程启动，GUI 里 `single_backward` 的 GPU 投影只有约 1 µs。
- 在 A1 的 `nn_utils.py` 里给注意力内部加了 NVTX 范围（`attn_scores`、`attn_mask`、`attn_softmax`、`attn_final_matual`），确认 softmax 虽然 FLOPs 少得多，却比 QKᵀ 更耗时。
- 模型都是 small（d_model 768、12 层、12 头、d_head 64），B=1、S=256，RTX 3060 Laptop 6 GB，FP32。

### 背景 / 动机
- 用 Python 计时（§2.1.3 的 `timeit`）只能拿到一步的总时间，看不出时间花在哪个 kernel 上。profiler 能把一步拆成每个 CUDA kernel 的耗时、次数和占比。
- **FLOPs 和 FLOP/s 是两回事**：FLOPs 是浮点运算的**次数**（工作量）；FLOP/s 是硬件每秒能做多少次（速率）。矩阵乘法 $(m\times k)(k\times n)$ 约 $2mkn$ FLOPs（每个输出元素 k 次乘、k 次加）。
- **受算力限制（compute-bound）vs 受显存带宽限制（memory-bound）**：
  - 算术强度（arithmetic intensity）= FLOPs / 读写的字节数。
  - roofline 下界：$t \ge \max\left(\dfrac{\text{FLOPs}}{\text{峰值 FLOP/s}},\ \dfrac{\text{bytes}}{\text{显存带宽}}\right)$。
  - GEMM 同一块数据在片上被复用很多次，强度高，受算力限制；逐元素运算每个元素只做 1–2 次运算却要整块读写，强度低，受带宽限制。

### 关键概念

#### NVTX
- 在 CPU 端给一段代码打一个有名字的范围：
  ```python
  import torch.cuda.nvtx as nvtx
  with nvtx.range("single_forward"):
      logits = model(x)
  ```
- 不会做 GPU 同步，几乎没有开销；不在 nsys 下运行时等于没有。
- GUI 里每个范围显示两行：CPU 端的范围，以及它投影到 GPU 上的那段（这段时间内**同一线程**启动的 kernel）。
- `benchmark.py` 里的嵌套：`warmup_*` / `measurement_*` 包住整个循环 → 里面每步一个 `*_step_i` → 再里面是并列的 `single_forward`、`single_backward`、`single_grad_clip`、`optimizer_step`。
- 在 A1 里用不需要配置环境：NVTX 是 PyTorch 自带的。A2 通过 editable 安装引用 A1，改了 A1 代码直接生效，不用重装。

#### GEMM 和 kernel 名字
- GEMM = General Matrix Multiply，$C = \alpha AB + \beta C$。
- `ampere_sgemm_128x64_tn`：`ampere` 是架构；`s` 是 single（FP32）；`128x64` 是每个线程块负责的输出块大小；`tn` 表示 A 转置、B 不转置。
- 反向需要不同的转置布局，所以会出现 `128x32_nn`、`128x64_nt` 等变体。
- kernel 名字看不出是哪一次矩阵乘法，要靠 NVTX 范围区分。

#### 整份报告的百分比 vs 按范围过滤
- `nsys stats` 不加过滤时，百分比包含初始化和 warmup。例如 `erfinv` kernel 来自 `trunc_normal_` 参数初始化，跟训练步无关。
- 写答案要按 NVTX 范围过滤，比如 `--filter-nvtx=measurement_step_0`。

### 环境与命令（Windows）
- `.venv` 里的 python 在 nsys 下会崩溃 → 改用 uv 的 base python + 启动脚本 `C:\Users\zhaohui\AppData\Local\nsys_a2\venv_launch.py`。脚本里用 `site.addsitedir(...)` 加载 venv 的 site-packages，这样 editable 安装的 `.pth` 文件会被处理；只设 `PYTHONPATH` 不会处理 `.pth`，导入 `cs336_basics` 会失败。
- 启动脚本不要放在 TEMP：会被清理，结果 nsys 照样返回 0，却只生成一个约 64 KB 的空报告。所以每次都要检查 benchmark 日志里 `EXIT=0`，并用 `nsys stats` 确认里面真的有 kernel。
- GUI 数字乱码：Windows 区域设置里的 native digits（注册表 `sNativeDigits`）导致，改回 0123456789 即可。
- 采集模板：
  ```powershell
  nsys profile -t cuda,nvtx -o <out> --force-overwrite true $py $L cs336_systems.benchmark --model-size small --batch-size 1 --context-length 256 --w 2 --n 2 --options 0 --device cuda
  ```
  其中 `$py` 是 uv base python，`$L` 是启动脚本路径。
- 统计模板：
  ```powershell
  nsys stats --force-export=true --report cuda_gpu_kern_sum --filter-nvtx=<range> <rep>
  ```
  `Total Time` 是该范围内所有 kernel 耗时之和；`Instances` 是次数。

### 结果一：只前向（`--options 0`）

#### 输入 / 输出 / 形状
- 输入 token `(B, S) = (1, 256)`；每层 Q/K/V 形状 `(B, H, S, d_head) = (1, 12, 256, 64)`；注意力分数 `(1, 12, 256, 256)`；输出 logits `(1, 256, vocab)`。

#### 每步 GEMM 次数 = 109（已核实）
- 每层 9 次：
  - Q、K、V、O 四个投影：4 次；
  - QKᵀ 和 P·V：2 次；
  - SwiGLU 的 W1、W3、W2：3 次。
- 12 层 × 9 = 108，再加 LM head 1 次 = **109**。
- RMSNorm 不是矩阵乘法（逐元素 + 归约）；embedding 也不是（查表）。
- `measurement_forward_step_0` = `single_forward` 的 GPU 投影 ≈ **54.258 ms**。
- 排第一的 kernel：`ampere_sgemm_128x64_tn`，约占整份报告 39.7%。

### 结果二：前向 + 反向（`--options 1`）

#### 反向 GEMM 次数 = 218（已核实）
- 线性层 $y = xW^\top$ 反向要算两个梯度：$dx = dy\cdot W$，$dW = dy^\top\cdot x$ → 每个线性层 2 次 GEMM。
- 线性层个数：每层 7 个（Q、K、V、O、W1、W3、W2）× 12 + LM head = 85 → 170 次。
- 注意力里两个矩阵乘法各 2 个梯度：$S = QK^\top$ 要 dQ、dK；$O = PV$ 要 dP、dV → 每层 4 次 × 12 = 48 次。
  - softmax 本身不是矩阵乘法，它的反向是逐元素/归约。
- 反向 170 + 48 = **218**；每步合计 109 + 218 = **327**。
- 实测：主线程 109 个 GEMM + 另一个线程 218 个；每步 kernel 共 2070 个（主线程 727 + 另一个线程 1343）。

#### 坑：反向 kernel 在另一个线程上
- 现象：GUI 里 `single_backward` 的 GPU 投影只有 960 ns，`single_forward` 几乎占满整步。
- 原因：PyTorch 的 `loss.backward()` 由 autograd 工作线程启动 kernel。GUI 的 NVTX GPU 投影只算**同一线程**在范围内启动的 kernel，主线程在 backward 期间只启动了 1 个 `FillFunctor` 清零 kernel（约 1 µs）。
- CPU 端 `single_backward` 实际持续约 57 ms，期间另一个线程启动了 1343 个 kernel。
- `nsys stats --filter-nvtx` 是按**时间重叠**过滤的，会把其它线程的 kernel 也算进去，所以统计数字是完整的（327 就是这样数出来的）。
- forward 和 backward 之间约 1 ms 的空隙是计算 loss 的约 10 个小 kernel（max、exp、sum、log、mean、index）。
- 核查报告：`C:\Users\zhaohui\AppData\Local\nsys_a2\small_fb_check.nsys-rep`。
- 排除过的另一种可能：`loss.backward()` 确实在 `single_backward` 范围里（代码没问题）。

#### kernel 排名（整份报告）
- 第一变成 `vectorized_elementwise_kernel` 18.1%，第二 `elementwise_kernel` 17.7%。
- GEMM 拆成 `128x32_nn` 14.3%、`128x64_tn` 13.3%、`128x64_nt` 12.7%，合计约 40.3%：矩阵乘法总占比仍最大，但单个 kernel 排不到第一。
- 矩阵乘法占比比只前向时低：反向给每个逐元素运算都加了梯度 kernel（SiLU、RMSNorm、softmax、残差的反向），还有梯度累加（`param.grad += …`、残差分支梯度相加）。

#### 为什么逐元素 kernel FLOPs 很少却占约 30% 时间
- 每个 kernel 都要把整个张量读一遍、写一遍，每个元素只做 1–2 次运算 → 算术强度极低，受显存带宽限制；kernel 数量多，还有启动开销。
- **融合（fusion）**：FLOPs 不变，省掉中间结果写回显存再读出来的流量，以及多次 kernel 启动。

### 结果三：注意力内部拆分（`--options 0`，NVTX 加在 A1）

#### 每层一次调用的 FLOPs（B=1、H=12、S=256、d_head=64）
- QKᵀ：$2\cdot B\cdot H\cdot S^2\cdot d_{head} = 2\cdot1\cdot12\cdot256^2\cdot64 = 100{,}663{,}296$。
- softmax：$\approx 5\cdot B\cdot H\cdot S^2 = 3{,}932{,}160$（假设每个元素约 5 次运算：max、减、exp、求和、除）。
- P·V：同 QKᵀ，$100{,}663{,}296$。
- 比例 25.6 : 1 : 25.6。

#### 实测（GUI 里前 3 层的 GPU 投影）

| 范围 | 第 1 层 | 第 2 层 | 第 3 层 |
|---|---|---|---|
| `attn_scores`（QKᵀ 和 /√d_k） | 91.9 µs | 99.3 µs | 318.4 µs |
| `attn_mask` | 119.3 µs | 128.3 µs | 242.2 µs |
| `attn_softmax` | 137.2 µs | 187.4 µs | 620.1 µs |
| `attn_final_matual`（P·V） | 44.2 µs | 44.4 µs | 44.6 µs |

- softmax 的 FLOPs 只有 QKᵀ 的 1/25.6，耗时却是它的约 1.5–2 倍 → 受带宽限制。它要多遍读写：求 max 一遍、exp 一遍、求和一遍、除法一遍，拆成多个 kernel 时中间结果每次都进出显存。
- mask 也很贵：几乎不计算，只是把整个 S×S 矩阵读一遍写一遍。
- `attn_scores` 比 P·V 慢一倍多，可两者 FLOPs 相同：因为范围里还包着 `/ sqrt(d_k)` 这次逐元素读写。
- 各层波动大，写答案用 `nsys stats --filter-nvtx=attn_softmax` 的 Total Time ÷ Instances 更稳。
- 往后：把 scale、mask、softmax 合进一个 kernel，只读一次写一次；FlashAttention 更进一步，分块在片上算，S×S 矩阵根本不写回显存。

### 结果四：完整训练一步（`--options 2`）

#### 时间拆分（`measurement_step_0` = 253.352 ms）
| 范围 | GPU 投影 |
|---|---|
| `single_forward` | 57.144 ms |
| `single_backward` | 992 ns（线程投影问题，不是真实值） |
| `single_grad_clip` | 28.642 ms |
| `optimizer_step` | 65.185 ms |

- 反向 ≈ 253 − 57 − 28.6 − 65 ≈ 102 ms，约前向的 1.8 倍。
- 按 `measurement_step_0` 过滤：`AUnaryFunctor` 18.3%、`CUDAFunctor_add` 15.3%，之后才是 GEMM：`128x32_nn` 7.4%、`128x64_tn` 6.4%、`128x64_nt` 6.3% ……；图里能看到的 GEMM 行合计约 28.7%，比 forward+backward 的约 40% 更低。
- 优化器 kernel 是从名字推断的（**尚未**用 `--filter-nvtx=optimizer_step` 核实）：`pow_tensor_scalar`（g²）、`sqrt`（√v）、`add`、`mul`。
- 梯度裁剪 28.6 ms 对 small 模型偏慢，推测是逐参数循环产生大量小 kernel，可能还有 `.item()` 同步（**假设，未核实**）。

#### AdamW 的读写量
- 读 4 个张量：参数 θ、梯度 g、一阶矩 m、二阶矩 v；写 3 个：θ、m、v。全都和参数一样大。
- 所以 AdamW 耗时只取决于参数量，和 batch、序列长度无关，且受带宽限制。
- Python 里逐参数循环，每个参数启动好几个逐元素 kernel；PyTorch 的 `foreach` / `fused` 实现会把它们合并成少数几个 kernel。

#### 反向 ≈ 2× 前向，优化器不按前向比例
- 反向 FLOPs ≈ 前向的 2 倍（每个矩阵乘法反向算两个梯度），所以 fwd+bwd ≈ 3× 前向。
- 优化器不能这样算：前向/反向随 batch × 序列长度增长，AdamW 只随参数量增长。这里 B=1、S=256，前向很小，所以优化器（65 ms）显得特别大；batch 或序列变大后，它的占比会下降。

### 实现步骤（复现流程）
1. 在 `benchmark.py` 用 NVTX 包好 warmup / measurement / 每步 / 各阶段（已完成）。
2. 用上面的模板采集，三种 `--options` 各一份报告。
3. 检查 benchmark 日志 `EXIT=0`，并确认报告里有 kernel（不是 64 KB 空报告）。
4. `nsys stats --filter-nvtx=<step 范围>` 统计 kernel 和 GEMM 次数，和手算对照。
5. 需要细分时，在 A1 的模块里加 NVTX 范围再采集，一行代码一个范围，不要两个运算挤在同一行。

### 常见错误（今天我自己犯的）
1. **把 RMSNorm 和 embedding 算成矩阵乘法**：RMSNorm 是逐元素 + 归约，embedding 是查表。
2. **数前向 GEMM 时漏了 O 投影、QKᵀ / P·V 和 SwiGLU 的三个矩阵**：正确是每层 9 次 + LM head = 109。
3. **数反向 GEMM 时漏了 LM head**：线性层是 7×12+1 = 85 个，不是 84。
4. **说 fwd+bwd 时第一名 kernel 没变**：截图里第一已经是 `vectorized_elementwise_kernel`（18.1%）；GEMM 拆成了三种布局，只是加起来仍最大。
5. **AdamW 读写漏了权重本身**：要读 θ、g、m、v 四个，写 θ、m、v 三个。
6. **NVTX 导入写成 `from torch.cuda.nvtx import nvtx`**：这个名字不存在，ImportError；应写 `import torch.cuda.nvtx as nvtx`。
7. **NVTX 范围包错了运算**：
   - softmax 和 P·V 写在同一行 `einsum(softmax(scores), V, ...)`，被一起包进了 `attn_softmax`；要拆成两行。
   - `attn_final_matmul` 包在了 `self.output_proj(out)`（W_O 投影）上，不是 P·V。
   - `masked_fill` 也用了 `attn_scores` 这个名字，按名字过滤时 mask 的 kernel 会混进 QKᵀ；改成 `attn_mask`。
8. **以为 FLOPs 是速率**：FLOPs 是浮点运算次数；FLOP/s 才是速率。
9. **以为 A1 要单独配置 NVTX 环境**：不需要，PyTorch 自带；editable 安装会直接用上 A1 的改动。

### 值得注意的点
1. GUI 的 NVTX GPU 投影只看同一线程；`nsys stats --filter-nvtx` 看时间重叠。反向在 autograd 线程上，两者结果会不同。
2. nsys 返回 0 不代表采集成功，一定要看 benchmark 的 EXIT 和报告里的 kernel。
3. 整份报告的百分比混入了初始化（如 `erfinv`）和 warmup，写答案要按范围过滤。
4. 范围名字有拼写错误：代码里是 `attn_final_matual`，过滤时要用这个名字（或者把代码改成 `matmul`）。
5. B=1、S=256 时 GPU 吃不满，逐元素和启动开销被放大；结论在更大的 batch/序列下比例会变。

### 回忆题
1. **算术强度是什么？为什么 GEMM 受算力限制、逐元素运算受带宽限制？**
   <details><summary>答案</summary>FLOPs / 读写字节数。GEMM 同一块数据在片上被复用很多次，强度高；逐元素每个元素 1–2 次运算却要整块读写，强度低。</details>
2. **small 模型前向每步多少次 GEMM？怎么来的？**
   <details><summary>答案</summary>109 = 12 × (Q、K、V、O 4 + QKᵀ、PV 2 + W1、W3、W2 3) + LM head 1。</details>
3. **反向每步多少次 GEMM？**
   <details><summary>答案</summary>218 = 85 个线性层 × 2（dx、dW）+ 12 层 × 4（dQ、dK、dP、dV）。合计每步 327。</details>
4. **`ampere_sgemm_128x64_tn` 每部分是什么意思？**
   <details><summary>答案</summary>Ampere 架构，s = FP32，每个线程块算 128×64 的输出块，A 转置、B 不转置。</details>
5. **为什么 GUI 里 `single_backward` 只有约 1 µs？`nsys stats` 的数字可信吗？**
   <details><summary>答案</summary>反向 kernel 由 autograd 线程启动，GUI 投影只算同一线程；主线程只启动了一个清零 kernel。stats 按时间重叠过滤，包含其它线程，数字完整。</details>
6. **为什么加上反向后矩阵乘法占比下降？**
   <details><summary>答案</summary>反向给每个逐元素运算都加了梯度 kernel，还有梯度累加，这些都受带宽限制，摊薄了 GEMM 的占比。</details>
7. **softmax FLOPs 只有 QKᵀ 的 1/25.6，为什么反而更慢？怎么优化？**
   <details><summary>答案</summary>它受带宽限制，要多遍读写 S×S 矩阵。融合 scale、mask、softmax 成一个 kernel；FlashAttention 分块在片上算，不把 S×S 写回显存。</details>
8. **AdamW 每个参数要读写哪些张量？耗时和什么有关？**
   <details><summary>答案</summary>读 θ、g、m、v，写 θ、m、v；只和参数量有关，和 batch、序列长度无关。</details>
9. **反向 ≈ 几倍前向？为什么不能说加上优化器就是 3 倍？**
   <details><summary>答案</summary>反向约 2 倍（实测约 1.8 倍），fwd+bwd 约 3 倍前向。优化器只随参数量增长，不随 batch × 序列长度增长，和前向不成比例。</details>
10. **nsys 返回 0 但报告只有约 64 KB，可能是什么原因？**
    <details><summary>答案</summary>启动脚本放在 TEMP 被清掉，python 没跑起来；要看 benchmark 日志 EXIT=0 和 stats 里有没有 kernel。</details>

#### 下一步 / TODO
- handout 要求两个模型尺寸 × 三个上下文长度（>128，最大取 6 GB 能放下的），还没跑。
- 题 (a)：nsys 测的前向时间和 Python 计时的对比，还没写。
- 用 `--filter-nvtx=optimizer_step` 核实优化器的 kernel 构成；顺便核实梯度裁剪慢的原因。
- 之后进入 §2.1.5 混合精度。
- 日志未提交；本轮不 commit、不 push。
