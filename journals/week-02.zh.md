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

## Day — §2.1.5 混合精度（mixed_precision）（2026-11-11）

#### 做了什么
- 做了 `mixed_precision_accumulation`：把 0.01 加 1000 次，比较 fp32 累加器、fp16 累加器，以及用 fp32 累加器去加 fp16 的 0.01。
- 用 ToyModel 在 FP16 autocast 下核对了题 (a)：参数、各层输出、loss、梯度各自的 dtype。
- 分析了题 (b)：LayerNorm 里哪几步在 FP16 下会溢出或丢精度，以及换成 BF16 之后为什么均值和方差仍要在 fp32 里算。
- 给 `benchmark.py` 加了 `--mixed-precision`，修了两个坑：`type=bool` 导致开关失效，以及 `torch.autocast` 误用了 `args.device` 全局变量。
- 本地 RTX 3060 Laptop 上测了 small、只前向、B=4、S=256、n=30 的 FP32 vs BF16。
- 云上 A100 SXM4 80GB（PyTorch 2.11.0+cu128）跑了 small–xl 的前向+反向（`--options 1`），B=4、S=512、warmup 5、n=30，FP32 和 BF16 共 8 组，没有 OOM。

### 背景 / 动机
- 混合精度把计算和一部分中间结果从 FP32 降到 FP16 或 BF16，目的有两个：
  - **速度**：现代 GPU 的 Tensor Core 对 FP16 / BF16 的 GEMM 吞吐更高。
  - **显存**：激活变成低精度后，反向要存的中间结果大约减半。
- 低精度会带来两类问题，不要混为一谈：
  - **范围（range）**：指数位不够，数大会溢出成 Inf，数小会下溢成 0。FP16 只有 5 位指数，这是范围问题，不是“丢精度”。
  - **精度（precision）**：尾数位不够，累加时小数被吞掉，相近的数无法区分。
- 实际训练里的做法是：**参数和梯度仍用 FP32 更新**，只在适合的算子上用低精度算；FP16 训练还要配合 loss scaling，防止小梯度下溢。

### 关键概念

#### 浮点格式
- 浮点数都是符号位 + 指数 + 尾数。规格化数还藏了一个前导 1（hidden leading 1），值是 $(-1)^s \cdot 1.m \cdot 2^{(e-\text{bias})}$。
- **FP32**：1 + 8 + 23。
- **FP16**：1 + 5 + 10，bias = 15。
  - 指数码 31（全 1）留给 Inf / NaN，0 留给零和次正规数（subnormal，没有隐藏的前导 1）。
  - 最大可用指数码是 30，无偏指数 $30-15=15$；尾数 10 位全 1 时有效数字是 $2-2^{-10}$。所以最大值 $2^{15}\times(2-2^{-10})=65504$。
  - 最小规格化数：$2^{-14}\approx 6.1\times 10^{-5}$（指数码 1，无偏指数 $1-15=-14$）。
  - 例子：12 写成二进制是 $1.1_2\times 2^3$，无偏指数 3，存储的指数是 $3+15=18$，尾数是隐藏的 1 后面那部分 `1000000000`。
- **BF16**：1 + 8 + 7。指数位和 FP32 一样是 8 位，所以**范围和 FP32 相同**；尾数只有 7 位，比 FP16 的 10 位还少，精度更差。

#### Loss scaling
- FP16 的最小规格化数大约是 $6.1\times 10^{-5}$，很多梯度比这还小，直接用 FP16 存会变成 0（下溢）。
- 做法：把 loss 乘上一个缩放因子 $S$，反向得到的梯度也放大 $S$ 倍，小梯度就能落在 FP16 的范围内。
- 交给梯度裁剪和优化器之前，要先 **unscale**：梯度乘 $1/S$，回到真实尺度。顺序必须是 **先 unscale，再 clip，再 step**。否则 clip 的阈值作用在被放大了 $S$ 倍的梯度上，等于把裁剪阈值缩小了 $S$ 倍。
- **动态缩放**（`GradScaler` 一类）：某一步梯度里出现 Inf / NaN，就跳过这次更新并把 $S$ 减半；连续很多步都正常，再把 $S$ 加倍。
- BF16 的范围和 FP32 一样，小梯度一般不会下溢，**通常不需要 loss scaling**。

#### Autocast
- Autocast 是**按算子**选精度，不是把整网转成低精度：
  - `matmul` / `linear` / `einsum` → bf16 或 fp16；
  - `exp` / `log` / `pow` / `sum` / `softmax` / `log_softmax` / `norm` / `layer_norm` → fp32；
  - 其它算子跟随输入的 dtype。
- 手写的 `cross_entropy` 可能混用不同精度；最稳妥的是算 loss 之前先 `logits.float()`。
- **参数和梯度始终是 fp32**。autocast 会额外缓存一份权重的 bf16 副本，给低精度的 matmul / linear 用。
- 只有前向和 loss 放进 `with ctx`；`backward`、梯度裁剪、`optimizer.step` 都在外面。

### 结果一：累加实验（`mixed_precision_accumulation`）
- 把 0.01 加 1000 次，理想结果是 10。
- fp32 累加器加 fp32 的 0.01：**10.0001**（0.01 在 fp32 里也不能精确表示，会有一点误差）。
- fp16 累加器：**9.9531**。累加和变大之后，fp16 的尾数分辨不出这么小的增量，后面加上去的 0.01 被吞掉一部分。
- fp32 累加器去加 fp16 的 0.01：两种写法都是 **10.0021**。误差来自 0.01 存进 fp16 时变成约 $0.0100021$；再乘 1000 就是约 10.0021。**fp16 → fp32 的转换本身是精确的**（有限的 FP16 值都是 FP32 的子集）。
- 教训：低精度可以用来算，**累加要在高精度里做**。

### 结果二：题 (a) ToyModel 和题 (b) LayerNorm

#### 题 (a)：FP16 autocast 下的 dtype
- 参数：fp32。
- `fc1` 输出：fp16（linear 被降到低精度）。
- LayerNorm 输出：fp32（`layer_norm` 在 fp32 列表里）。
- logits：fp16（又一次 linear）。
- loss：fp32。
- 梯度：fp32。

#### 题 (b)：LayerNorm 里对精度敏感的几步
- **平方和求和**：输入稍大，平方就会超过 FP16 的 65504，这是**范围**问题，会溢出。
- **均值和方差的累加**：尾数位不够会丢精度（和上面的 0.01 × 1000 是同一类问题）。
- **除以 $\sqrt{\mathrm{var}+\varepsilon}$**：方差很小时，FP16 会下溢，或者 $\varepsilon$ 在低精度里被舍入，这也偏**范围**。
- 换成 BF16 之后，指数位和 FP32 一样，平方溢出、方差下溢这些范围问题都没了。但 BF16 尾数只有 7 位，比 FP16 还少，**累加丢精度反而更严重**。所以均值和方差仍要在 fp32 里算。这也是 PyTorch autocast 在 BF16 下仍把 `layer_norm` 放在 fp32 列表里的原因。

### 实现：`benchmark.py` 的 `--mixed-precision`
- 函数参数默认 `mixed_precision=False`。命令行用 `action="store_true"`，不要写 `type=bool`。
- 之前的 bug：用了 `type=bool`，再加上 `action="store_true"` 还设了 `default=True`，结果开关等于一直开着。`type=bool` 本身也不对：`bool("False")` 是 `True`。
- 上下文：
  ```python
  from contextlib import nullcontext

  ctx = (
      torch.autocast(device_type=device.type, dtype=torch.bfloat16)
      if mixed_precision
      else nullcontext()
  )
  with ctx:
      logits = model(x)
      loss = ...
  # backward / clip / step 在 ctx 外面
  ```
- 早先写成了 `torch.autocast(device_type=args.device, ...)`：一是函数里没有 `args`，import 时会 `NameError`；二是 `device_type` 只接受 `"cuda"` / `"cpu"` 这类类型名，`"cuda:1"` 是非法的。正确写法是 `device.type`。

### 结果三：本地 RTX 3060 Laptop（只前向）
- 配置：small，B=4，S=256，只前向，n=30。
- FP32 **0.0677 s**，BF16 **0.0573 s**，加速比 **1.18×**。
- n=10 时曾经得到 1.36×，不可信：那次 FP32 的标准差高达 0.015，波动把加速比抬高了。n=30 之后数据才稳。
- **Amdahl**：假设 GEMM 约占总时间的 60%，其余 40% 不加速，总时间比是 $0.4 + 0.6/k$。实测总时间比 $0.0573/0.0677\approx 0.847$，反推 GEMM 本身只快了约 **1.34×**。
- 加速不明显的原因：矩阵小，SM 吃不饱；kernel 启动开销和逐元素算子占比高，这部分 bf16 帮不上忙；autocast 还会多出把权重 / 激活 cast 成 bf16 的 kernel。
- 另外，据我所知，GeForce 上 BF16 Tensor Core（用 fp32 累加）的吞吐是减半的——**未在这张 3060 上核实**。
- 理论 2× 只适用于受带宽限制、读写字节减半的情况；这里既不是纯带宽瓶颈，矩阵也还不够大。

### 结果四：题 (c) A100 SXM4 80GB（前向+反向）
- 配置：PyTorch 2.11.0+cu128，`--options 1`，B=4，S=512，warmup 5，n=30。代码是作业里的 `benchmark.py` 和 A1 的 `cs336_basics`，没有改。8 组都跑完，没有 OOM。各组标准差都在约 3 ms 以内。

| 尺寸 | FP32 (s) | BF16 (s) | 加速比 | 显存 FP32 / BF16 (GiB) |
|---|---:|---:|---:|---:|
| small | 0.1418 | 0.0624 | 2.27× | 4.76 / 3.81 |
| medium | 0.4381 | 0.1390 | 3.15× | 12.22 / 10.08 |
| large | 0.9409 | 0.2452 | 3.84× | 25.02 / 20.80 |
| xl | 2.8083 | 0.5219 | 5.38× | 53.57 / 52.29 |

- **加速比随模型增大而上升**（2.27× → 5.38×）：small 慢的原因不是显存没跑满，而是**算力没吃满**。矩阵小，SM 喂不饱，kernel 启动和逐元素算子占比高。模型越大，GEMM 的占比和规模都变大，Tensor Core 越接近峰值，cast 开销被摊薄。
- **显存节省主要来自激活**：linear 的输出变成 bf16，反向要存的中间结果大约减半。参数和梯度仍是 fp32，这部分不变；autocast 还多存一份权重的 bf16 副本。看 xl：参数加梯度约 25 GiB，bf16 权重副本再加约 6 GiB，基本抵掉了激活省下的量，总共只省了 1.3 GiB。small 的激活占比高，所以省得更明显（约 20%）。
- 写答案可以概括成：**BF16 加速比随模型增大而上升，从 small 的 2.3× 到 xl 的 5.4×；显存节省主要来自激活，参数与梯度仍为 fp32。**

### 实现步骤（复现流程）
1. 先弄清 FP16 / BF16 的范围和精度，以及 FP16 为什么需要 loss scaling（先 unscale 再 clip）。
2. 跑 `mixed_precision_accumulation`，确认累加必须在 fp32 里做。
3. 用 ToyModel 在 autocast 下打印各层输出、参数、loss、梯度的 dtype（题 a）。
4. 逐项分析 LayerNorm 的平方 / 求和、均值 / 方差、除以 $\sqrt{\mathrm{var}+\varepsilon}$，分清哪些是范围问题、哪些是精度问题；BF16 下去掉范围问题后，均值和方差仍留在 fp32（题 b）。
5. 给 `benchmark.py` 加 `--mixed-precision`：`action="store_true"`，默认 False；用 `device.type` 建 `torch.autocast`，关掉时用 `nullcontext()`；只把前向和 loss 放进 `with ctx`。
6. 本地先用小模型、n=30 以上对比 FP32 / BF16，看标准差，避免 n 太少把波动当成加速。
7. 同一张足够大的卡上跑 small–xl 的前向+反向，记录时间和预留显存，再解释加速比和显存变化。

### 常见错误（今天我自己犯的）
1. **把 FP16 只有 5 位指数说成会丢精度**：5 位指数影响的是范围（溢出 / 下溢），丢精度是尾数位不够。
2. **以为 fp16 → fp32 的转换会丢精度**：这个转换本身是精确的；误差来自数值当初存进 fp16 的时候。
3. **以为 autocast 下梯度也是 bf16**：参数和梯度都还是 fp32。
4. **以为混合精度省显存是因为累加**：省的是激活（以及反向要存的中间结果）；累加反而应该在 fp32 里做，不会因此省显存。
5. **看到逐元素 kernel 没用 Tensor Core，就说没有发生类型转换**：cast 是会发生的，而且会多出额外的 kernel。
6. **一开始把本地加速不明显归咎于显存没跑满**：真正的原因是算力没吃满（矩阵小、启动开销、cast kernel）。
7. **`--mixed-precision` 写成 `type=bool`，还设了 `default=True`**：开关等于一直开着；`bool("False")` 也是 `True`。
8. **`torch.autocast` 用了 `args.device` 全局变量**：函数里没有 `args` 会 NameError；`"cuda:1"` 也不能当 `device_type`，要用 `device.type`。

### 值得注意的点
1. Autocast 是按算子选精度，不是整网转成 bf16 / fp16。
2. 参数和梯度始终是 fp32；autocast 会额外缓存一份 bf16 权重副本，所以混合精度并不总是按“全部减半”来省显存。
3. BF16 通常不用 loss scaling，但尾数只有 7 位，均值、方差、loss 这类累加更要留在 fp32。
4. fp16 → fp32 转换本身精确；看到误差时，先查这个数是不是在更早的时候就已经被舍入进 fp16 了。
5. 加速比随模型增大而上升，是因为 GEMM 占比和规模变大、Tensor Core 更接近峰值，不是因为显存跑满了。
6. 理论 2× 只适用于受带宽限制、流量减半的情况。小矩阵加上启动开销和 cast kernel，实际加速会小得多。
7. 据我所知，GeForce 上 BF16 Tensor Core（fp32 累加）吞吐会减半；这一点未在这张 3060 上核实。
8. 计时至少要看标准差：n=10 的 1.36× 就是被 FP32 的高波动抬上去的。

### 回忆题
1. **为什么 FP16 的最大值是 65504？**
   <details><summary>答案</summary>5 位指数里全 1（31）留给 Inf / NaN，最大可用指数码是 30，无偏指数 $30-15=15$；尾数 10 位全 1 时有效数字是 $2-2^{-10}$。所以最大值 $2^{15}\times(2-2^{-10})=65504$。</details>
2. **为什么 BF16 通常不需要 loss scaling？**
   <details><summary>答案</summary>BF16 的指数位和 FP32 一样是 8 位，范围相同，小梯度一般不会下溢成 0。Loss scaling 主要是为了救 FP16 的范围。</details>
3. **unscale 和 clip 谁先谁后？为什么？**
   <details><summary>答案</summary>先 unscale（乘 $1/S$）再 clip，再交给优化器。否则 clip 的阈值作用在被放大了 $S$ 倍的梯度上，等于把裁剪阈值缩小了 $S$ 倍。</details>
4. **为什么累加要在 fp32 里做？0.01 加 1000 次说明了什么？**
   <details><summary>答案</summary>fp16 累加器得到 9.9531，已经偏了；fp32 累加 fp16 的 0.01 得到 10.0021，误差来自 0.01 存进 fp16 时变成约 0.0100021，fp16 → fp32 转换本身是精确的。教训是累加要用高精度。</details>
5. **autocast 下 LayerNorm 的输出是什么 dtype？BF16 已经没有范围问题了，为什么均值和方差还要在 fp32 里算？**
   <details><summary>答案</summary>输出是 fp32，`layer_norm` 在 autocast 的 fp32 列表里。BF16 指数位和 FP32 一样，溢出 / 下溢没了，但尾数只有 7 位，累加丢精度比 FP16 更严重，所以均值和方差仍留在 fp32。</details>
6. **为什么 xl 用 BF16 几乎没省显存？**
   <details><summary>答案</summary>省的主要是激活。参数和梯度仍是 fp32（xl 约 25 GiB），autocast 还多存一份 bf16 权重副本（约 6 GiB），把激活省下的量基本抵掉了，总共只省了 1.3 GiB。</details>
7. **为什么加速比随模型增大而上升？**
   <details><summary>答案</summary>小模型矩阵小，SM 吃不饱，kernel 启动和逐元素算子占比高，bf16 帮不上忙。模型越大，GEMM 的占比和规模都变大，Tensor Core 越接近峰值，cast 开销被摊薄，所以从 small 的 2.3× 涨到 xl 的 5.4×。</details>
8. **按 Amdahl，GEMM 占 60%、总时间比约 0.847，GEMM 本身加速了多少？**
   <details><summary>答案</summary>$0.4 + 0.6/k = 0.847$ → $k \approx 1.34\times$。也就是说总加速 1.18× 时，GEMM 大约只快了 1.34 倍。</details>
9. **autocast 下参数和梯度是什么精度？权重还会发生什么？**
   <details><summary>答案</summary>都是 fp32。autocast 会缓存一份 bf16 权重副本，给低精度的 matmul / linear 用。</details>
10. **为什么 12 存在 FP16 里时，存储的指数是 18？**
    <details><summary>答案</summary>$12 = 1.1_2\times 2^3$，无偏指数 3，加上 bias 15，存储指数 $= 18$。</details>

#### 下一步 / TODO
- §2.1.5 到这里完成。§2.1.6 显存分析已开始记录（见后文 2026-11-16）；题 (f) 尚未完成。
- 日志未提交；本轮不 commit、不 push。

## Day — §2.1.6 显存分析（memory_profiling）（2026-11-16）

#### 做了什么
- 学习并实践作业 §2.1.6 的显存观察方法：PyTorch memory snapshot（`memory_viz`）和 Nsight Systems 的 CUDA 显存跟踪 + PyTorch NVTX。
- 把 11/16 本地 small 补采、用户自行生成的 `--memory-profile` 快照，以及此前 xl 前向快照里已经核实过的观察记下来。
- **本节没有完成题 (f)。** 没有单 block 的 saved-for-backward 总量、没有 top 5 算子占比、没有反向梯度占用的最终数值；S=2048 未跑；本轮未租新卡。下面记录的是方法、已核实证据和明确的不足。

### 背景 / 动机
- 计时只能回答「时间花在哪」；要解释「显存为什么涨、哪些张量一直活到反向」，需要把分配事件和算子 / 模块范围对齐。
- 工具分两路，不能混用：
  - PyTorch `_record_memory_history` / `_dump_snapshot` 看的是 **caching allocator 眼里的 block**：`alloc`、`free_requested`、`free_completed`，以及 Python 申请栈。`memory_viz` 用地址代次显示，例如 `bd2b500000_0`。
  - Nsight Systems 的 `--cuda-memory-usage` 看的是 **CUDA 层的显存使用 / 分配事件**。默认的 Memory 轨道常常是 memcpy / memset，那是传输和清零，不是占用曲线。
- 作业 (f) 还要求使用 **PyTorch 提供的 NVTX**（例如 `autograd-shapes-nvtx`、`functions-trace`），才能把单个 TransformerBlock 的前向保存量和反向梯度变化对上。

### 公式 / 规则
- 一块 FP32 激活的字节数是 $B\cdot S\cdot D\cdot 4$。换算成 MiB 时除以 $1024^2$，不是 $1000^2$。
- 本地 small、B=1、S=256、$d_{ff}=3072$ 时，$(B,S,d_{ff})$ 的 FP32 块恰好是

  $$
  1\times 256\times 3072\times 4 = 3{,}145{,}728\ \text{B} = 3\ \text{MiB}。
  $$

  注意力分数若为 $(B,H,S,S)=(1,12,256,256)$、FP32，同样是 3 MiB；**大小相同不等于来源相同**。
- 此前 xl、B=4、S=128、$d_{ff}=10240$ 时，$(B,S,d_{ff})$ 的 FP32 块是 20 MiB；形状为 $2560\times 10240$ 的 FP32 权重矩阵是 100 MiB。这只是对上尺寸的线索，不是来源证明。
- `memory_viz` 的地址代次（官方 JS `block_version` / `version_space`）：每个地址从 0 起；`alloc` 和 `free_requested` 只读当前代次、不递增；`free_completed` 返回当前代次后再加 1。界面名是 `b{addr.toString(16)}_{version}`。`d2b500000_0` 就是地址 `0xd2b500000` 的第 0 代，不能把 `_0` 想成「整份快照的第一个事件」。
- Caching allocator 会先向 CUDA 要大段 reserved 内存，再在池里切给张量。CUDA 层往往只能看到 pool 的 alloc，不一定能看到每个 tensor 的 alloc / free。**不能把 GPU Memory Usage 曲线的涨跌直接当成 tensor allocated，也不能当成 saved-for-backward。**

### 输入 / 输出 / 形状
- 本题观察对象是训练一步里的激活、权重和梯度占用，而不是新的模型接口。
- 已跑并核过的本地配置：small，B=1，S=256，FP32（未开 `--mixed-precision`），`--w 2 --n 2 --options 2`（完整训练步：前向 + 反向 + clip + optimizer）。这是方法演示，**不是 xl 的正式 (f) 结果**。
- xl、S=128 的四组 PyTorch pickle 只用于 (e) 一类观察；它们不是 Nsight 报告，不能和 nsys 混用。
- S=2048 的配置本轮没有跑。

### 实现或实验步骤（自然语言）
1. 先用 PyTorch 快照看「这块是谁申请的」：在稳定阶段打开 `_record_memory_history`，导出 pickle，用 [memory_viz](https://pytorch.org/memory_viz) 看 Active Memory Timeline。要追权重从哪来，记录必须在参数被搬到 GPU 之前开始；要看稳定运行，则在预热之后开始。两种目的不要混。
2. 选中色块时，分清「单次分配大小」和「分配后的总占用」。Detail 太低时，蓝色隆起可能是合并显示的小张量。调用栈要滚到自己的 `.py` 行号。
3. 若块在开始记录之前就已存在，工具可能标成 ghost block：仍占用、但没有申请栈，不代表泄漏。
4. 要用 Nsight 做 (f)，采集命令里需要 CUDA 显存跟踪，以及实际支持的 PyTorch 算子 / 模块 NVTX。只写 `-t cuda,nvtx` 不够。本机 `nsys profile --help`（2026.6.1.218）确认有 `--cuda-memory-usage`（默认 false）和 `--pytorch`（含 `autograd-shapes-nvtx`、`functions-trace` 等，默认为 none）。
5. 导出 SQLite 后，先数 `CUDA_GPU_MEMORY_USAGE_EVENTS` 的 Allocation / Deallocation，再看 NVTX 名字；不要把 reserved 统计直接写成每张量 allocated。
6. 本地补采必须用新文件名，不要 `--force-overwrite` 盖掉旧报告。

### 已核实的观察（只写有证据的）

#### 题 (e)：xl 前向快照里的大块
- 一个 **20 MiB** 分配，调用栈能对上：`swiglu.py:52` → `linear.py:45` → `einsum` / `bmm`。大小与 $(B,S,d_{ff})=(4,128,10240)$ 的 FP32 输出一致。这是前馈里的**中间激活**，不是注意力分数矩阵，也不是权重本身。
- 另有 **100 MiB** 的无调用栈分配（ghost block）。大小刚好等于 FP32 的 $2560\times 10240$ 权重，只能作为线索，**这份快照不能确认来源**。因此还不能把 20 MiB 写成「全局最大分配」。

#### 旧的 `small_full_process.nsys-rep` 为什么不够做 (f)
- 报告头和嵌入命令都表明：Nsight Systems 2026.6.1.218，主机 `LAPTOP-VA5CFNA3`，GPU 为 RTX 3060 Laptop，模型是 **small**，命令只有 `-t cuda,nvtx`，**没有** `--cuda-memory-usage`，也没有 `--pytorch`。
- GUI 里 Memory 轨道是 memcpy / memset，与「未开显存占用追踪」一致。这份报告不能替代 xl 的 (f) 数据。

#### 本地 small 补采（方法演示，2026-11-16 17:29 香港时区）
- 完整命令（未使用 `--force-overwrite`）：

  `nsys.exe profile -t cuda,nvtx --cuda-memory-usage=true --pytorch=autograd-shapes-nvtx,functions-trace -o C:\Users\zhaohui\AppData\Local\nsys_a2\small_mem_f_20261116_172922`，后面是既有的 `venv_launch.py cs336_systems.benchmark --model-size small --batch-size 1 --context-length 256 --w 2 --n 2 --options 2 --device cuda`。
- 报告：`C:\Users\zhaohui\AppData\Local\nsys_a2\small_mem_f_20261116_172922.nsys-rep`（以及同名 `.sqlite`）。nsys 与 benchmark 的退出码都是 0。非致命警告：无管理员权限时 CPU sampling / context switch 被关掉。
- SQLite 实证：`CUDA_GPU_MEMORY_USAGE_EVENTS` 共 735 条，**全部是 Allocation，Deallocation 为 0**。因此这条占用曲线不能当成逐张量的申请 / 释放，更不能直接读成 saved-for-backward。
- NVTX 里已经有 `aten::*`、autograd 引擎范围，以及 `TransformerLM.layers.0` … `layers.11` 等模块名。用户可以在 GUI 里自己读图；**这里不代写 top 5，也不把这次 small 采集标成 (f) 完成。**

#### 用户自行跑的 `--memory-profile` 快照（同一套 small 配置）
- 真正的 pickle（protocol 4）已用禁止 `find_class` / `persistent_load` 的受限 Unpickler 解析；只有基本容器，没有可执行全局对象。
- Windows 上关于 C++ stack 不支持和临时 kernel 缓存的警告是非致命的，不改变下面的事件结论。
- 目标块 `d2b500000_0`（即 `bd2b500000_0`）是 3,145,728 字节：
  - `alloc` 索引 **930**，香港时区 2026-11-16 17:48:28.764；栈为 `swiglu.py:23:silu` → `swiglu.py:52` → `transformer.py:47` → `transformer_lm.py:50` → `benchmark.py:216`。
  - `free_requested` 索引 **3659**、`free_completed` 索引 **3660**，同一时刻 17:48:28.824。**释放栈是空的**，所以 GUI 里看不到释放位置，不必再截图。
  - 生命周期约 59,800 µs。快照的 `device_traces` 里没有 NVTX，只能用注解和时间推断边界：第一次 `zero_grad` 结束之后、第一次 `optimizer.step` 开始之前；索引 1691 已经进入 `benchmark.py:220` 的 `backward`，而该块当时仍存活；释放之后下一条有栈的事件仍在这次 backward 里。因此可以推断：**前向申请，越过前向边界，在反向过程中释放**。不能指认层号，也不能指认具体反向算子，更不能与 nsys 范围做精确对齐。
- 同样写着 `swiglu.py:23:silu`、同为 3 MiB 的 alloc 共 48 次（12 层 × 2 个测量步 × 每层连续 2 次）。与 930 成对的是 929（另一块地址）。没有 `swiglu.py` 源码，不能断定第 23 行是两个内部算子，还是输入 / 输出各一次。
- 另有 3 MiB 的 softmax 分配，大小与 $(1,12,256,256)$ FP32 相符，这只是栈和尺寸上的候选，**未确认为全局最大分配**。

### 数值例子
- 20 MiB（xl，B=4，S=128，$d_{ff}=10240$，FP32 激活）和 3 MiB（local small，B=1，S=256，$d_{ff}=3072$，FP32 激活）都是用 $B\cdot S\cdot D\cdot 4 / 1024^2$ 对上的。
- `d2b500000_0` 的事件号是 930 / 3659 / 3660，不是「图上随便点到的一块」。
- 2.1.5 日志里已经记过 A100 上 xl 前向+反向的预留显存（FP32 53.57 GiB、BF16 52.29 GiB）。那是混合精度计时实验的预留值，**不要把它当成这次 (f) 的峰值或 saved-for-backward 结果**；此处不另造峰值表。

### 常见错误 / 注意点
1. 把 memcpy / memset 轨道当成显存占用曲线。
2. 把 CUDA Memory Usage 的涨跌当成 PyTorch 每个 tensor 的 allocated / freed。
3. 把 20 MiB 的已确认激活写成「最大分配」，却忽略 100 MiB 的无栈块。
4. 把 local small 的方法演示写成 xl 的正式 (f) 结论。
5. 把 ghost block 当成泄漏，或把 `_0` 理解成整份 trace 的第一条事件。
6. 快照没有 NVTX 时，假装已经和 nsys 范围精确对齐，或在释放栈为空时指定层号 / 算子。
7. 用 hook 次数或事件条数代替字节数（下一节还会再强调）。

### 回忆题
1. **为什么只开 `-t cuda,nvtx` 往往不够做题 (f)？**
   <details><summary>答案</summary>这条命令会追踪 CUDA API / kernel 和程序自己打的 NVTX，但不会默认打开 `--cuda-memory-usage`（文档和本机 help 都写默认是 false），也不会自动加上 PyTorch 算子级 NVTX（`--pytorch` 默认是 none）。没有显存占用事件和模块 / 算子范围，就无法按单个 TransformerBlock 统计 saved-for-backward。</details>
2. **`CUDA_GPU_MEMORY_USAGE_EVENTS` 里只有 Allocation、没有 Deallocation，能直接读 saved-for-backward 吗？**
   <details><summary>答案</summary>不能。这更像是 caching allocator 向 CUDA 预留了块、短期没有 `cudaFree` 回去。曲线反映的是保留量变化，不是每张激活被保存或释放的字节数。</details>
3. **`d2b500000_0` 的 `_0` 是什么意思？**
   <details><summary>答案</summary>它是该地址的第 0 代：`alloc` / `free_requested` / `free_completed` 在第一次 `free_completed` 递增之前都用代次 0。同一地址下次再分配会变成 `_1`。不是「整份快照的第一个事件」。</details>
4. **为什么 3 MiB 既可能来自 SwiGLU，也可能来自注意力分数？**
   <details><summary>答案</summary>两种形状在这组 small 配置下恰好都是 $1\times 256\times 3072\times 4$ 或 $1\times 12\times 256\times 256\times 4$。必须看调用栈，不能只看大小。</details>
5. **本题（f）现在完成了吗？**
   <details><summary>答案</summary>没有。方法、限制和若干已核实观察已经记下；top 5、单 block 总量、反向梯度占用、S=2048 和 xl 正式采集都还没有。</details>

#### 下一步 / TODO
- 不要把整个 §2.1.6 或题 (f) 标成完成。
- 若继续做 (f)，需要能同时看到显存占用和 PyTorch 模块 / 算子范围的报告（现有 small 报告只是方法演示），并自己在 GUI 里读单个 block；S=2048 与 xl 正式结果仍缺。
- 日志未提交；本轮不 commit、不 push。

## Day — §3.1 Saved tensors 与 §3.1.1 fusion（2026-11-17）

#### 做了什么
- 学习 §3.1：autograd 为反向保存哪些张量（residuals / saved tensors），以及保存通常是引用而非复制。
- 用作业目录里的 `cs336_systems/rmsnorm_test.py` 做 §3.1.1：`torch.compile` 加上 `saved_tensors_hooks`，对比编译前后打印到的 Saving / Loading 记录。命令是 `uv run python .\cs336_systems\rmsnorm_test.py`。
- 静态审阅过该脚本（未把「编译成功」写成已在 Windows CUDA / inductor 上验证）。随后贴出了编译与未编译的 hook 输出；2026-11-17 再次只读核对了当时的脚本内容。本轮没有用当前脚本再跑实验。
- 已口头回答 activation checkpointing 的概念题：反向进入 checkpoint 区域时，先用保存的输入重跑该区域所需的前向、恢复内部激活，再算梯度。§3.2 仍只是概念预告，还没有做实验。

### 背景 / 动机
- 反向并不是「每个前向算子都把所有中间量再存一份」。有的算子根本不需要输入值，有的只需要其中一侧，有的可以在反向时用别的量重建。
- 看清 RMSNorm 实际保存了什么，才能理解：编译路径上 saved tensors 为什么可能变少，以及少掉的那一项是不是真的少占了一块独立显存。hook 记录只说明保存行为，不能说明生成了多少 kernel，也不能说明发生了融合。
- 这也是后面 activation checkpointing（用重算换「少保存」）的铺垫。§3.2 的实验还没开始。

### 公式
- 对 $y=x^2$，反向是 $2x\cdot \mathrm{d}y$。实现上通常保存 $x$，不必须再存一份 $2x$。
- 对 $y=x+2$，反向只是把上游梯度原样传回 $x$，**不需要 $x$ 的值**。
- 对 $z=ab$：如果 $a$、$b$ 都需要梯度，两边都要保存；如果只有 $a$ 需要梯度，保存的是 $b$（因为 $\partial z/\partial a=b$）。这一点曾经答反，已经纠正。
- 线性 $y=xW$：求 $W$ 的梯度需要 $x$。
- 记 RMSNorm 中 $m=\mathrm{mean}(x^2)$（最后一维），$r=(m+\varepsilon)^{-1/2}$，$v=x r$，$y=w v$。
- $r$ 对 $m$ 的导数是

  $$
  \frac{\mathrm{d}r}{\mathrm{d}m}=-\frac12(m+\varepsilon)^{-3/2}=-\frac12 r^3。
  $$

- `mean` 的反向需要的是输出形状、缩减维度和元素个数，不是均值 $u$ 本身的值。
- 把某个参数的梯度冻结（`requires_grad=False`）并不等于「前向里就不必保存它的值」：别的参数算梯度时仍可能要用到它，例如 $y=wv$ 在 $v$ 需要梯度时要保存 $w$。
- `torch.compile` 的大致路径是：先捕获计算图，再联合优化前向和反向（包括保存什么、什么改为重算），然后由所选 backend 生成代码或调用算子库。逐元素算子更容易被融在一起；矩阵乘法仍常走专用库。并非所有算子都能融合，这受 backend、设备、依赖、算子支持和成本限制。数据相关的 Python 控制流或外部副作用可能造成 graph break。
- 当前脚本写明的 backend 是 `aot_eager`：它走 AOT Autograd，前后向图和 saved tensors 可以与纯 eager 不同，但图仍以 eager 方式执行。这**不等于** Inductor / Triton 已经生成了 fusion kernel。不能根据 hook 输出声称「已经融出了 kernel」，也不能声称「已经测到 CUDA 峰值下降」。

### 输入 / 输出 / 形状
- 输入 $x$ 的形状是 $(4,512,2560)$，FP32，约 $4\times 512\times 2560\times 4=20{,}971{,}520$ 字节 $=20$ MiB。
- $r$ 的形状是 $(4,512,1)$，FP32，共 $8192$ 字节 $=8$ KiB。
- $w$ 的形状是 $(2560,)$，FP32，共 $10240$ 字节 $=10$ KiB。
- $v=x r$ 的形状与 $x$ 相同，也是 20 MiB。
- 输出 $y=w v$，形状与 $x$ 相同。

### 实现或实验步骤（自然语言）
1. 在 forward 和 backward 外面打开 `torch.autograd.graph.saved_tensors_hooks`；`pack_hook` / `unpack_hook` 只观察并原样返回张量。观察函数应标记为 compile disable，**不要把 RMSNorm 的 forward 整段禁用掉**。这和 autocast 不是同一件事：autocast 按算子选精度，`torch.compile` 返回一个可调用的优化对象，通常在**第一次调用**时才真正编译。
2. 先看未编译图里出现了哪些 Saving / Loading，记下形状和 `grad_fn`，再与 $x$、$r$、$v$、$w$ 对应。
3. 再看编译路径上的记录，确认少了哪一项。若少掉的是 $v$，可用「引用 + storage 去重 + 可用 $x$、$r$ 重建 $v$」把少保留的约 20 MiB 解释为**该图上独立 storage 的推导**，而不是把 6 条记录变成 3 条理解成显存减半，更不是 CUDA 峰值实测。
4. 如果要对比 eager 与 compiled，两次实验应分开，避免在同一组参数上连续两次 backward 把梯度加在一起。

### 脚本审阅与当前代码（静态，未再跑实验）
- A1 的 `from cs336_basics.rmsnorm import RMSNorm` 仍被注释。脚本里用的是本地定义、手册风格的 `rsqrt` 实现，不是 A1 的 `sqrt` 再除法、并升到 float32 的版本。所以若 hook 输出与手册 rsqrt 版本一致，这是预期现象，不能当成「A1 实现已经被 compile 成这样」。
- 最初的版本在默认 CPU 上创建了 `ln_compiled = torch.compile(ln)`，但实际调用的是 `ln(x)`，编译对象没用上。这一点在审阅时已经指出。
- 2026-11-17 再读当时的 `rmsnorm_test.py`：`x` 和 `RMSNorm` 已经指定 `device="cuda"`；`rmsnorm_fwd` 用 `torch.compile(..., backend="aot_eager")` 得到 `rms_fwd_compiled`；模块的 `forward` 实际调用的是 `rms_fwd_compiled`。`torch.compile(ln)` 那一行改成了注释。也就是说，**当前脚本确实在调用编译后的函数**，设备也已改成 CUDA，写明的 backend 是 `aot_eager`。
- 这只能记录「脚本现在这样写」，**不能证明** Windows 上默认的 inductor / Triton CUDA 编译可用。`aot_eager` 的 AOT Autograd 前/反向图及 saved tensors 变化，不等于已经生成 Inductor / Triton fusion kernel。
- 贴出的「编译后」hook 输出（见下节）与当前 `aot_eager` 脚本是否为同一次运行，本轮没有再跑实验来核对。因此把那组输出和当前文件分开记录：前者是这次贴出的保存行为；后者是只读看到的脚本写法。不把前者臆断成「当前 aot_eager 已融出 kernel」或「已测到 CUDA 峰值减少」。

### 数值例子（贴出的 hook 输出）
未编译时是 6 次 Saving、6 次 Loading，顺序为：
1. $x$：`[4, 512, 2560]`，FP32，`grad_fn=None`
2. $r$：`[4, 512, 1]`，`RsqrtBackward`
3. $r$ 再出现一次
4. $x$ 再出现一次
5. $v$：`[4, 512, 2560]`，`MulBackward`
6. $w$：`[2560]`，`grad_fn=None`

这**不是**三块大张量、合计 60 MiB 的独立内存。保存通常是引用，同一块 storage 被多个算子记下时要去重：$x$ 的 20 MiB 和 $v$ 的 20 MiB 是两块，共 40 MiB；$r$ 8 KiB，$w$ 10 KiB。$x$ 和 $r$ 各出现两次，不加倍计算。

贴出的编译后记录与手册一致，只长期保留 $x$、$w$、$r$ 三项（Saving / Loading 各三次，形状分别是 `[4, 512, 2560]`、`[2560]`、`[4, 512, 1]`，`grad_fn` 均为 None），不再长期保留 $v$。因此「大约少保留 20 MiB」是按**该计算图上独立 storage** 做的推导（去掉 $v$ 的 20 MiB），**不是 CUDA 峰值实测**，也不能当作已生成 fusion kernel 的证据。hooks 只证明保存行为。这个理解是对的：反向若需要 $v$，可以用 $x$ 和 $r$ 重建，再参与 $w$ 的梯度。

### 常见错误 / 注意点
1. 看见 6 条记录变 3 条，就说显存减半。记录条数不是字节数。
2. 把多次保存同一 $x$ 加成多份 20 MiB；删除 Python 变量名也不等于释放那块 storage。
3. 把 $z=ab$ 的保存规则记反：只有 $a$ 需要梯度时，保存的是 $b$。
4. 以为 `requires_grad=False` 的权重就不必出现在 saved tensors 里。
5. 用 `@torch.compile.disable` 把整段 forward 关掉，只留下 hooks——那样比较的就不是编译后的计算图。
6. 把本次 `backend="aot_eager"` 的脚本，或这次贴出的 hook 输出，写成「已经验证 Windows CUDA inductor / Triton fusion kernel」，或写成「已经实测 CUDA 峰值下降」。`aot_eager` 以 eager 执行 AOT Autograd 图，不等于 fusion kernel。
7. activation checkpointing 仍只是概念预告，§3.2 还没有做实验。概念上已经答对：反向处理 checkpoint 区域时，先用保存的**该区域输入**重跑所需前向、恢复内部激活，再计算梯度；不必先重算整个模型；也不会因此消去参数、梯度或 optimizer 状态。
8. 并非所有算子都能被 `compile` 融合；数据相关的 Python 控制流或外部副作用可能 graph break。hooks 不能用来数 kernel 或证明融合。

### 回忆题
1. **$y=x^2$ 和 $y=x+2$，反向分别需要保存什么？**
   <details><summary>答案</summary>$y=x^2$ 通常保存 $x$，不必须保存 $2x$。$y=x+2$ 不需要 $x$ 的值，把上游梯度传回即可。</details>
2. **$z=ab$ 在「只有 $a$ 需要梯度」时保存谁？**
   <details><summary>答案</summary>保存 $b$。因为 $\partial z/\partial a=b$。</details>
3. **未编译的 6 条记录为什么不是 60 MiB？「少保留约 20 MiB」的依据是什么，不是什么？**
   <details><summary>答案</summary>$x$ 和 $v$ 各 20 MiB，去重后的大块是 40 MiB，再加上 $r$ 的 8 KiB 和 $w$ 的 10 KiB。$x$、$r$ 的重复记录是同一块 storage 的引用。编译路径上若不再长期保留 $v$，大约少 20 MiB；这是按该图独立 storage 做的推导，不是实测 CUDA 峰值，也不能证明已经生成了 fusion kernel。</details>
4. **`torch.compile` 和 autocast 有什么不同？hooks 上的 disable 应该禁用谁？**
   <details><summary>答案</summary>`torch.compile` 作用在函数或模块上，返回可调用的优化对象，通常第一次调用时编译。autocast 是按算子选择精度。disable 只应排除观察用的 hook 逻辑，不应把要研究的 forward 关掉。</details>
5. **为什么可以用 $x$ 和 $r$ 重构 $v$ 来算 $w$ 的梯度？**
   <details><summary>答案</summary>$v=x r$，$y=w v$。需要 $v$ 时不必长期保存它，用仍然保存的 $x$ 和 $r$ 相乘即可重建。</details>
6. **`torch.compile` 怎样自动处理这些算子？是不是所有算子都能融合？hook 输出能证明融合了 kernel 吗？**
   <details><summary>答案</summary>它先捕获计算图，再联合优化前向和反向（包括保存与重算），然后由 backend 生成代码或调用算子库。逐元素算子较易融合，矩阵乘法仍常走专用库。并非所有算子都能融合，受 backend、设备、依赖、支持和成本限制；数据相关的 Python 控制流或外部副作用可能 graph break。hooks 只证明保存行为，不能证明 kernel 数量或已经融合。当前脚本的 `aot_eager` 以 eager 执行 AOT Autograd 图，更不能据此声称已生成 Inductor / Triton fusion kernel。</details>
7. **一个 block 如果只保留输入，反向时要怎么做？需要先重算整个模型吗？会不会去掉参数和 optimizer 状态？**
   <details><summary>答案</summary>进入该 checkpoint 区域时，先用保存的输入重跑这一段所需的前向，把内部激活恢复出来，再计算梯度。不必先重算整个模型；参数、梯度和 optimizer 状态也不会因此被消掉。这是概念理解，§3.2 尚未做实验。</details>

#### 下一步 / TODO
- §3.1 / §3.1.1 的对照已经记入日志；checkpoint 的概念题已答对。§3.2 activation checkpointing 实验尚未开始。
- 日志未提交；本轮不 commit、不 push。

## Day — §3.2 Activation checkpointing（2026-11-17 下午）

#### 做了什么
- 在本机 CUDA、small 配置上写 `cs336_systems/checkpoint_test.py`：先做单个 `TransformerBlock` 的普通前向反向，再与 `torch.utils.checkpoint`（`use_reentrant=False`）对照。
- 曾经给 block 加上 `torch.compile(..., fullgraph=True, backend="aot_eager")`。捕获失败，原因是 A1 `cs336_basics/nn_utils.py` 第 43 行附近把 `nvtx.range` 当成上下文管理器使用，和 fullgraph 捕获冲突。随后取消 compile，在 eager 下运行。`aot_eager` 不是 Inductor / Triton 融合内核，这一点已在同日上午的 §3.1.1 记过。
- 用 `saved_tensors_hooks` 看普通路径和 checkpoint 路径为反向保存了多少；再在同权重、同输入、不 `step` / 不 clip 的条件下对照输出和梯度；再测完整前后向的 `torch.cuda.max_memory_allocated`。
- 讨论了正式题 gradient_checkpointing (a) 的嵌套前缀思路； (b) 的 xl 与相邻区域实测没有做。§4.1 即将开始，实验还没有开始。

### 背景 / 动机
- checkpoint 用「反向时重算该区域所需前向」换「少长期保存内部激活」。hooks 看到的是为反向保存的张量，不是 CUDA 峰值，也不是去重后的独立 storage。
- 单层 block 的完整前后向峰值里，仍然有参数、梯度、输入，以及反向重算时重新出现的中间激活，所以少保存不等于峰值一定下降。多层时，若不再同时保留每一层的内部激活，峰值差异才更容易出现。

### 公式
- 本次 small 输入的元素占用是

  $$
  4\times 256\times 768\times 4=3{,}145{,}728\ \text{字节}=3\ \text{MiB}。
  $$

- 正式 (a) 的候选做法：把前向切成嵌套前缀区域 $1..N,\ 1..(N-1),\ \ldots,\ 1$，并且这些区域共享同一块输入 $x_0$ 的存储，而不是每层各存一份独立输入。
- 在题目忽略 checkpoint bookkeeping、并且单层残差激活占主导的抽象下，这种共享 $x_0$ 的嵌套前缀给出峰值激活 $O(1)$、计算量 $O(N^2)$。这说的是激活峰值的渐近，**不是**参数、梯度或 optimizer 状态的总显存变成 $O(1)$。Python 里嵌套管理这些区域的额外成本，也不能普遍当成可以忽略。
- 中途曾把复杂度类推成 $O(D)$。那种计数只适用于「每一层都有自己独立输入」的切分，不适用于共享 $x_0$ 的方案。
- 另外讨论过均匀分段：记一层激活约 $A$、一段边界约 $R$，则 $M(k)\approx (N/k)A+kR$，启发式 $k^*\approx\sqrt{NA/R}$，具体取哪个合法整数要以实测为准。二分递归的典型量级是峰值激活 $O(R+A\log N)$、计算 $O(N\log N)$。Revolve 只作为概念扩展，没有实现。

### 输入 / 输出 / 形状
- 配置是 small：$B=4$，$S=256$，$d_{\mathrm{model}}=768$，$d_{\mathrm{ff}}=3072$，`num_heads=12`，FP32，CUDA。$768/12=64$，可以整除。
- 单层输入 $x$ 的形状是 $(4,256,768)$，是隐藏向量，不是 token id。
- 用来反向的标量是 block 输出的求和，不是词表交叉熵。

### 实现思路（记录这次实际写法，不是代写）
- 设备与 dtype 和输入对齐后，普通路径调用编译已取消的 eager `block`；checkpoint 路径使用 `checkpoint(..., use_reentrant=False)`。
- 正确性对照由自己完成：同一套权重、同一个 $x$，不做 `optimizer.step`、不做 clip；输出用 `detach` 后再 clone 另存，输入梯度和参数梯度也 clone；清参数梯度的同时还清 $x$.grad（只调用 optimizer 的 `zero_grad` 不会清掉 $x$.grad）；参数梯度为 `None` 时占位；再用 `zip` 逐个参数走 `if` / `elif` / `else`，与 `allclose` 比较输出、$x$ 梯度和全部参数梯度。用户报告这些比较都是 True。这是用户自己的运行结果；本轮只做静态记录，没有复跑，也不断言这是正式的全测试。
- 测峰值时的演进：去掉 hooks 和对照用的副本，避免额外引用抬高或干扰峰值；测量前清理梯度，先 synchronize 再 `reset_peak_memory_stats`，完整前向加反向之后再 synchronize，然后读 `max_memory_allocated`。也曾提醒要预热。2026-11-17 18:07（Hong Kong Time）审阅时的脚本还缺少预热和反向之后的同步；那之后的修改没有再复核，因此不宣称测量流程已经完善。
- 当前源码里的 `four_blocks` 是**同一个** `TransformerBlock` 实例连续调用四次，不是四个彼此独立、各有一份权重的模块。用户说明四层峰值对照时，是把四层当作一个整体包进 checkpoint，而不是逐 block 对照，也不是比较相邻更小或更大的区域（后两种没有做）。文件里另有 `four_blocks_checkpoint`，把整个 `four_blocks` 放进 `checkpoint(..., use_reentrant=False)`；按现稿，这个函数里连续写了两次这样的调用。当前测量段激活的是普通 `four_blocks`。用户报告的四层峰值没有在本轮复跑，不能自动当成对当前激活路径的复现。

### 数值例子（用户报告，本轮未复跑）
- hooks 逐条累加（不是 storage 去重，也不是 CUDA 峰值）：普通单个 block $189.47$ MiB；`use_reentrant=False` 的 checkpoint 为 $3$ MiB，与只长期保留那 $3$ MiB 输入相一致。保存记录的总和、去重后的独立 storage、以及 `max_memory_allocated`，三者不是同一件事。
- 单个 block 的 CUDA 峰值（用户报告）：普通 $202.72$ MiB，checkpoint $203.59$ MiB。这不能证明峰值下降。两次测量是否在独立进程里完成，当时没有确认。
- 四层（用户报告）：普通 $545.73$ MiB，整体 checkpoint $352.18$ MiB，相差 $193.55$ MiB，约 $35.5\%$。时间开销没有测。

### 常见错误 / 注意点
1. 把 hooks 的累计字节直接当成 CUDA 峰值，或当成已经去重的独立 storage。
2. 看见单层 checkpoint 的 hooks 从约 $189$ MiB 降到 $3$ MiB，就推断完整前后向峰值也会按同样幅度下降。单层在反向重算时，内部激活会重新出现在峰值窗口里。
3. 只调用 optimizer 的 `zero_grad`，以为 $x$.grad 也被清掉；同一份 $x$ 再 backward，输入梯度会累加。
4. 把共享 $x_0$ 的嵌套前缀，说成「每层独立输入」那种 $O(D)$；或把抽象的 $O(1)$ 峰值激活说成整机显存（含参数、梯度、optimizer）都是 $O(1)$。
5. 把正式 (b) 或 (a) 的代码提交写成已经完成。xl、$S=2048$ 以及相邻区域的实测都还没有做；(a) 只讨论了思路，草图还没有交。
6. 在尚未复核预热和前后同步的情况下，宣称峰值测量已经完善。

### 复习（直答）
**hooks 里普通路径约 $189.47$ MiB、checkpoint 约 $3$ MiB，和后来的 CUDA 峰值，为什么不是同一件事？**
<details><summary>直答</summary>hooks 累加的是「为反向保存」的记录，默认按条相加，没有做 storage 去重，更不是分配器给出的峰值。checkpoint 把长期保存收成该区域的输入（这里是 $3$ MiB），所以 hooks 会小很多。完整前后向的 `max_memory_allocated` 还要包含参数、梯度、输入，以及反向重算时再次出现的中间量，所以不能用 hooks 的降幅去预言峰值降幅。</details>

**为什么单层的两个峰值可能几乎一样，而四层整体 checkpoint 才更有机会看到差距？**
<details><summary>直答</summary>单层在 backward 时要把这一层的前向重算出来，峰值窗口里仍可能出现和普通路径同量级的中间激活。多层时，普通路径往往会同时留下各层内部激活；若把四层当作一个整体 checkpoint，长期保存的是这块的入口，内部在反向时再重算，峰值才更可能下降。即使用户报告四层有差距，也只说明这一次整体包装下的现象，不能代替逐 block 或相邻区域的对照（那些还没有做）。</details>

**共享 $x_0$ 的嵌套前缀，抽象复杂度是什么？它为什么既不是总显存 $O(1)$，也不是每层独立输入的 $O(D)$？**
<details><summary>直答</summary>在忽略 bookkeeping、单层残差占主导的题目抽象下，峰值激活是 $O(1)$，计算是 $O(N^2)$，因为各前缀共享同一块 $x_0$，不必每层再存一份输入。参数、梯度和 optimizer 仍在，所以总显存不是 $O(1)$。$O(D)$ 那种算法是在「每一层都有自己的独立输入」时数出来的，不能搬到共享 $x_0$ 的方案上。Python 嵌套管理的成本也不能普遍忽略。</details>

**做输出和梯度对照时，要守住哪些条件？**
<details><summary>直答</summary>两版必须是同一输入、同一套参数；不要在对照中间 `step` 或 clip，否则权重已经变了。输出要另存一份脱离计算图的拷贝，梯度和参数梯度也要拷贝后再清。清参数梯度时还要清 $x$.grad。`None` 要单独处理，再逐个参数比较。这些是对照能否成立的条件；用户报告结果为 True，但本轮没有复跑。</details>

**正式 (a)、(b) 现在分别做到哪一步？**
<details><summary>直答</summary>(a) 的嵌套前缀、共享 $x_0$，以及上述抽象复杂度已经讨论过，代码草图还没有交，不能当成作业已经提交。(b) 要求的 xl、$B=4$、$S=2048$ 以及相邻更小或更大区域的实测都还没有做，不能勾成完成。时间开销也还没有测。§4.1 只是即将开始，实验尚未开始。</details>

#### 下一步 / TODO
- 正式 gradient_checkpointing (a)：思路已讨论，代码草图未交。
- 正式 (b)：xl、$B=4$、$S=2048$ 与相邻区域实测未做；时间开销未测。
- 峰值测量流程不宣称已经完善；§2.1.6 (f) 仍未完成。
- §4.1 即将开始，实验还没有开始。
