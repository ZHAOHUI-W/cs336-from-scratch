# Week 2 journal (English)

Chinese version: [week-02.zh.md](./week-02.zh.md). This week starts Assignment 2 (Systems); code lives in `assignments/a2-systems/`.

## Day — §2.1.4 Nsight Systems profiling (nsys_profile) (2026-11-11)

#### What I did
- Profiled `cs336_systems/benchmark.py` with Nsight Systems 2026.6.1 on Windows in three modes: forward only (`--options 0`), forward+backward (`--options 1`), and a full training step (`--options 2`, with grad clipping and AdamW).
- Used `nsys stats --filter-nvtx` to count kernels per NVTX range and checked GEMMs per step: forward 109, backward 218, total 327, exactly matching the hand count.
- Found and explained a pitfall: backward kernels are launched from a separate autograd thread, so the GUI GPU projection of `single_backward` is only ~1 µs.
- Added NVTX ranges inside attention in A1 `nn_utils.py` (`attn_scores`, `attn_mask`, `attn_softmax`, `attn_final_matual`) and confirmed softmax takes longer than QKᵀ despite far fewer FLOPs.
- All runs: small model (d_model 768, 12 layers, 12 heads, d_head 64), B=1, S=256, RTX 3060 Laptop 6 GB, FP32.

### Background / motivation
- Python timing (§2.1.3 `timeit`) only gives the total per step; it can't tell which kernel the time goes to. A profiler breaks a step into per-CUDA-kernel time, count, and share.
- **FLOPs vs FLOP/s**: FLOPs is the **count** of floating-point operations (work); FLOP/s is the hardware's rate. A matmul $(m\times k)(k\times n)$ is about $2mkn$ FLOPs (k multiplies and k adds per output element).
- **Compute-bound vs memory-bandwidth-bound**:
  - Arithmetic intensity = FLOPs / bytes read and written.
  - Roofline lower bound: $t \ge \max\left(\dfrac{\text{FLOPs}}{\text{peak FLOP/s}},\ \dfrac{\text{bytes}}{\text{memory bandwidth}}\right)$.
  - A GEMM reuses each tile on-chip many times (high intensity, compute-bound); an elementwise op does 1–2 ops per element but reads and writes the whole tensor (low intensity, bandwidth-bound).

### Key concepts

#### NVTX
- Named CPU-side ranges around code:
  ```python
  import torch.cuda.nvtx as nvtx
  with nvtx.range("single_forward"):
      logits = model(x)
  ```
- No GPU sync, near-zero overhead; effectively nothing when not running under nsys.
- The GUI shows two rows per range: the CPU range, and its projection onto the GPU (kernels launched **by the same thread** inside that range).
- Nesting in `benchmark.py`: `warmup_*` / `measurement_*` wrap the loops → one `*_step_i` per step → sibling ranges `single_forward`, `single_backward`, `single_grad_clip`, `optimizer_step`.
- No environment setup needed in A1: NVTX ships with PyTorch. A2 uses A1 via an editable install, so A1 edits take effect without reinstalling.

#### GEMM and kernel names
- GEMM = General Matrix Multiply, $C = \alpha AB + \beta C$.
- `ampere_sgemm_128x64_tn`: `ampere` = architecture; `s` = single (FP32); `128x64` = output tile per thread block; `tn` = A transposed, B not.
- Backward needs other transpose layouts, so `128x32_nn`, `128x64_nt` etc. appear.
- The kernel name doesn't say which matmul it is; use NVTX ranges to tell them apart.

#### Whole-report percentages vs range filtering
- Without a filter, `nsys stats` percentages include init and warmup, e.g. the `erfinv` kernel comes from `trunc_normal_` parameter init.
- For write-ups, filter by an NVTX range, e.g. `--filter-nvtx=measurement_step_0`.

### Environment and commands (Windows)
- `.venv` python crashes under nsys → use uv's base python plus the launcher `C:\Users\zhaohui\AppData\Local\nsys_a2\venv_launch.py`, which calls `site.addsitedir(...)` on the venv site-packages so the editable `.pth` files are processed. `PYTHONPATH` alone does not process `.pth`, so `import cs336_basics` fails.
- Don't keep the launcher in TEMP: it gets cleaned, nsys still returns 0 but produces an empty ~64 KB report. Always check `EXIT=0` in the benchmark log and confirm kernels with `nsys stats`.
- Garbled GUI digits: caused by Windows regional native digits (registry `sNativeDigits`); set back to 0123456789.
- Profile template:
  ```powershell
  nsys profile -t cuda,nvtx -o <out> --force-overwrite true $py $L cs336_systems.benchmark --model-size small --batch-size 1 --context-length 256 --w 2 --n 2 --options 0 --device cuda
  ```
  `$py` = uv base python, `$L` = launcher path.
- Stats template:
  ```powershell
  nsys stats --force-export=true --report cuda_gpu_kern_sum --filter-nvtx=<range> <rep>
  ```
  `Total Time` = sum of kernel time in the range; `Instances` = count.

### Result 1: forward only (`--options 0`)

#### Inputs / outputs / shapes
- Tokens `(B, S) = (1, 256)`; per-layer Q/K/V `(B, H, S, d_head) = (1, 12, 256, 64)`; attention scores `(1, 12, 256, 256)`; logits `(1, 256, vocab)`.

#### GEMMs per step = 109 (verified)
- 9 per layer: Q, K, V, O projections (4); QKᵀ and P·V (2); SwiGLU W1, W3, W2 (3).
- 12 × 9 = 108, plus LM head 1 = **109**.
- RMSNorm is not a matmul (elementwise + reduction); embedding is not either (table lookup).
- `measurement_forward_step_0` = `single_forward` GPU projection ≈ **54.258 ms**.
- Top kernel: `ampere_sgemm_128x64_tn`, ~39.7% of the whole report.

### Result 2: forward + backward (`--options 1`)

#### Backward GEMMs = 218 (verified)
- Linear $y = xW^\top$ needs two grads: $dx = dy\cdot W$, $dW = dy^\top\cdot x$ → 2 GEMMs per linear.
- Linears: 7 per layer (Q, K, V, O, W1, W3, W2) × 12 + LM head = 85 → 170.
- Attention: $S = QK^\top$ needs dQ, dK; $O = PV$ needs dP, dV → 4 per layer × 12 = 48. Softmax itself is not a matmul.
- Backward 170 + 48 = **218**; per step 109 + 218 = **327**.
- Measured: 109 GEMMs on the main thread + 218 on another thread; 2070 kernels per step (727 main + 1343 other).

#### Pitfall: backward kernels on another thread
- Symptom: in the GUI, `single_backward` GPU projection is 960 ns and `single_forward` fills almost the whole step.
- Cause: `loss.backward()` launches kernels from the autograd worker thread. The GUI NVTX projection only counts kernels from the **same thread**; during backward the main thread launched one `FillFunctor` zero kernel (~1 µs).
- CPU-side `single_backward` actually lasts ~57 ms, during which the other thread launched 1343 kernels.
- `nsys stats --filter-nvtx` filters by **time overlap**, including other threads, so the stats counts are complete (that's how 327 was counted).
- The ~1 ms gap between forward and backward is ~10 small loss kernels (max, exp, sum, log, mean, index).
- Check report: `C:\Users\zhaohui\AppData\Local\nsys_a2\small_fb_check.nsys-rep`.
- Ruled out: `loss.backward()` is inside the `single_backward` range (code is fine).

#### Kernel ranking (whole report)
- Top is now `vectorized_elementwise_kernel` 18.1%, then `elementwise_kernel` 17.7%.
- GEMMs split into `128x32_nn` 14.3%, `128x64_tn` 13.3%, `128x64_nt` 12.7%, ≈40.3% combined: matmul is still the largest total, but no single GEMM kernel is first.
- Matmul share drops vs forward-only: backward adds a grad kernel for each elementwise op (SiLU, RMSNorm, softmax, residual backward) plus gradient accumulation (`param.grad += …`, summing residual-branch grads).

#### Why elementwise kernels take ~30% of time with few FLOPs
- Each reads and writes the whole tensor with 1–2 ops per element → very low arithmetic intensity, bandwidth-bound; many kernels also add launch overhead.
- **Fusion** keeps FLOPs the same but saves the DRAM write/read of intermediates and the extra launches.

### Result 3: attention breakdown (`--options 0`, NVTX in A1)

#### FLOPs per layer call (B=1, H=12, S=256, d_head=64)
- QKᵀ: $2\cdot B\cdot H\cdot S^2\cdot d_{head} = 100{,}663{,}296$.
- softmax: $\approx 5\cdot B\cdot H\cdot S^2 = 3{,}932{,}160$ (assumes ~5 ops per element: max, subtract, exp, sum, divide).
- P·V: same as QKᵀ, $100{,}663{,}296$. Ratio 25.6 : 1 : 25.6.

#### Measured (GPU projection, first 3 layers in GUI)

| Range | Layer 1 | Layer 2 | Layer 3 |
|---|---|---|---|
| `attn_scores` (QKᵀ and /√d_k) | 91.9 µs | 99.3 µs | 318.4 µs |
| `attn_mask` | 119.3 µs | 128.3 µs | 242.2 µs |
| `attn_softmax` | 137.2 µs | 187.4 µs | 620.1 µs |
| `attn_final_matual` (P·V) | 44.2 µs | 44.4 µs | 44.6 µs |

- Softmax has 1/25.6 of QKᵀ's FLOPs but ~1.5–2× its time → bandwidth-bound. It makes several passes (max, exp, sum, divide), and split across kernels each intermediate goes through DRAM.
- Mask is expensive too: almost no compute, just reads and writes the whole S×S matrix.
- `attn_scores` is >2× slower than P·V at equal FLOPs because the range also includes the elementwise `/ sqrt(d_k)`.
- Per-layer numbers vary; for the write-up use `nsys stats --filter-nvtx=attn_softmax` Total Time ÷ Instances.
- Next: fuse scale, mask, softmax into one kernel (one read, one write); FlashAttention goes further, tiling on-chip so S×S is never written to DRAM.

### Result 4: full training step (`--options 2`)

#### Time split (`measurement_step_0` = 253.352 ms)
| Range | GPU projection |
|---|---|
| `single_forward` | 57.144 ms |
| `single_backward` | 992 ns (thread projection issue, not real) |
| `single_grad_clip` | 28.642 ms |
| `optimizer_step` | 65.185 ms |

- Backward ≈ 253 − 57 − 28.6 − 65 ≈ 102 ms ≈ 1.8× forward.
- Filtered to `measurement_step_0`: `AUnaryFunctor` 18.3%, `CUDAFunctor_add` 15.3%, then GEMMs `128x32_nn` 7.4%, `128x64_tn` 6.4%, `128x64_nt` 6.3% …; visible GEMM rows sum to ≈28.7%, below the ~40% of fwd+bwd.
- Optimizer kernels inferred from names (**not yet** verified with `--filter-nvtx=optimizer_step`): `pow_tensor_scalar` (g²), `sqrt` (√v), `add`, `mul`.
- Grad clip at 28.6 ms is slow for small; hypothesis: per-parameter loops of small kernels, possibly a `.item()` sync (**unverified**).

#### AdamW traffic
- Reads 4 tensors (θ, g, m, v), writes 3 (θ, m, v), all parameter-sized.
- So AdamW time depends only on parameter count, not batch or sequence length, and is bandwidth-bound.
- A Python per-parameter loop launches several elementwise kernels per parameter; PyTorch `foreach` / `fused` implementations merge them into a few kernels.

#### Backward ≈ 2× forward; optimizer doesn't scale with forward
- Backward FLOPs ≈ 2× forward (two grads per matmul), so fwd+bwd ≈ 3× forward.
- The optimizer doesn't follow that: forward/backward scale with batch × sequence, AdamW only with parameter count. At B=1, S=256 forward is small, so the optimizer (65 ms) looks large; its share shrinks with bigger batch/sequence.

### Steps (reproduce)
1. NVTX-wrap warmup / measurement / each step / each phase in `benchmark.py` (done).
2. Profile with the template, one report per `--options`.
3. Check `EXIT=0` in the benchmark log and that the report has kernels (not a 64 KB empty one).
4. `nsys stats --filter-nvtx=<step range>` for kernels and GEMM counts; compare with the hand count.
5. For finer breakdown, add NVTX ranges in A1 modules and re-profile: one op per line per range.

### Common mistakes (mine today)
1. **Counted RMSNorm and embedding as matmuls**: RMSNorm is elementwise + reduction, embedding is a table lookup.
2. **Forward GEMM count missed the O projection, QKᵀ / P·V, and SwiGLU's three matrices**: correct is 9 per layer + LM head = 109.
3. **Backward count forgot the LM head**: 7×12+1 = 85 linears, not 84.
4. **Said the top kernel was unchanged in fwd+bwd**: the screenshot showed `vectorized_elementwise_kernel` (18.1%) first; GEMMs split into three layouts, only their sum is largest.
5. **AdamW traffic forgot the weights themselves**: reads θ, g, m, v; writes θ, m, v.
6. **NVTX import as `from torch.cuda.nvtx import nvtx`**: that name doesn't exist (ImportError); use `import torch.cuda.nvtx as nvtx`.
7. **NVTX ranges around the wrong ops**:
   - softmax and P·V on one line `einsum(softmax(scores), V, ...)`, both inside `attn_softmax`; split into two lines.
   - `attn_final_matmul` wrapped `self.output_proj(out)` (W_O projection), not P·V.
   - `masked_fill` reused the name `attn_scores`, mixing mask kernels into QKᵀ when filtering; renamed `attn_mask`.
8. **Thought FLOPs is a rate**: FLOPs = count of floating-point ops; FLOP/s is the rate.
9. **Thought A1 needed NVTX environment setup**: no, it's built into PyTorch; the editable install picks up A1 changes.

### Noteworthy points
1. GUI NVTX GPU projection = same thread only; `nsys stats --filter-nvtx` = time overlap. With backward on the autograd thread, they differ.
2. nsys returning 0 doesn't mean success; check the benchmark EXIT and kernels in the report.
3. Whole-report percentages include init (e.g. `erfinv`) and warmup; filter by range for answers.
4. Range name typo: the code says `attn_final_matual`; filter by that name (or rename to `matmul`).
5. At B=1, S=256 the GPU isn't saturated, which inflates elementwise and launch overhead; the shares will shift at larger batch/sequence.

### Recall questions
1. **What is arithmetic intensity? Why is a GEMM compute-bound and an elementwise op bandwidth-bound?**
   <details><summary>Answer</summary>FLOPs / bytes moved. GEMM reuses tiles on-chip many times (high intensity); elementwise does 1–2 ops per element while reading/writing the whole tensor (low intensity).</details>
2. **How many GEMMs per forward step for small, and why?**
   <details><summary>Answer</summary>109 = 12 × (Q, K, V, O 4 + QKᵀ, PV 2 + W1, W3, W2 3) + LM head 1.</details>
3. **How many backward GEMMs per step?**
   <details><summary>Answer</summary>218 = 85 linears × 2 (dx, dW) + 12 layers × 4 (dQ, dK, dP, dV). Total 327 per step.</details>
4. **What does each part of `ampere_sgemm_128x64_tn` mean?**
   <details><summary>Answer</summary>Ampere arch, s = FP32, 128×64 output tile per thread block, A transposed, B not.</details>
5. **Why does `single_backward` show ~1 µs in the GUI? Are the `nsys stats` numbers trustworthy?**
   <details><summary>Answer</summary>Backward kernels come from the autograd thread; GUI projection only counts the same thread, and the main thread launched one zero-fill kernel. Stats filter by time overlap across threads, so they're complete.</details>
6. **Why does the matmul share drop once backward is added?**
   <details><summary>Answer</summary>Backward adds a grad kernel per elementwise op plus gradient accumulation, all bandwidth-bound, diluting the GEMM share.</details>
7. **Softmax has 1/25.6 of QKᵀ's FLOPs; why is it slower, and how to fix it?**
   <details><summary>Answer</summary>Bandwidth-bound, multiple passes over S×S. Fuse scale+mask+softmax into one kernel; FlashAttention tiles on-chip and never writes S×S to DRAM.</details>
8. **Which tensors does AdamW read/write per parameter, and what does its time depend on?**
   <details><summary>Answer</summary>Reads θ, g, m, v; writes θ, m, v. Depends only on parameter count, not batch or sequence length.</details>
9. **Backward ≈ how many × forward? Why isn't "plus optimizer = 3×" right?**
   <details><summary>Answer</summary>~2× (measured ~1.8×), so fwd+bwd ≈ 3× forward. The optimizer scales with parameter count, not batch × sequence, so it isn't proportional to forward.</details>
10. **nsys returns 0 but the report is ~64 KB. What happened?**
    <details><summary>Answer</summary>Likely the launcher in TEMP was cleaned and python never ran; check benchmark log EXIT=0 and that stats show kernels.</details>

#### Next / TODO
- Handout requires two model sizes × three context lengths (>128, up to the largest that fits in 6 GB); not done yet.
- Part (a): compare nsys forward time with Python timing; not written yet.
- Verify the optimizer kernel breakdown with `--filter-nvtx=optimizer_step`; also check why grad clip is slow.
- Then §2.1.5 mixed precision.
- Journal not committed; no commit or push this round.

## Day — §2.1.5 Mixed Precision (mixed_precision) (2026-11-11)

#### What I did
- Ran `mixed_precision_accumulation`: added 0.01 a thousand times, comparing an fp32 accumulator, an fp16 accumulator, and an fp32 accumulator that adds fp16 0.01.
- Checked part (a) with a ToyModel under FP16 autocast: dtypes of parameters, each layer's output, the loss, and the gradients.
- Analyzed part (b): which LayerNorm steps overflow or lose precision in FP16, and why mean and variance should still be computed in fp32 after switching to BF16.
- Added `--mixed-precision` to `benchmark.py` and fixed two bugs: `type=bool` made the flag useless, and `torch.autocast` incorrectly used the `args.device` global.
- On the local RTX 3060 Laptop, timed small, forward-only, B=4, S=256, n=30, FP32 vs BF16.
- On a cloud A100 SXM4 80GB (PyTorch 2.11.0+cu128), ran forward+backward (`--options 1`) for small–xl, B=4, S=512, warmup 5, n=30, eight runs in FP32 and BF16, no OOM.

### Background / motivation
- Mixed precision drops compute and some intermediates from FP32 to FP16 or BF16 for two reasons:
  - **Speed**: modern GPU Tensor Cores have higher GEMM throughput in FP16 / BF16.
  - **Memory**: once activations are lower precision, the intermediates saved for backward are about half the size.
- Lower precision causes two distinct problems; don't mix them up:
  - **Range**: too few exponent bits, so large values overflow to Inf and small ones underflow to 0. FP16's 5 exponent bits are a range issue, not "loss of precision."
  - **Precision**: too few mantissa bits, so small increments are swallowed during accumulation and nearby values become indistinguishable.
- In real training, **parameters and gradients are still updated in FP32**; only suitable ops run in low precision. FP16 training also needs loss scaling so small gradients don't underflow.

### Key concepts

#### Floating-point formats
- A float is sign + exponent + mantissa. Normalized numbers also have a hidden leading 1; the value is $(-1)^s \cdot 1.m \cdot 2^{(e-\text{bias})}$.
- **FP32**: 1 + 8 + 23.
- **FP16**: 1 + 5 + 10, bias = 15.
  - Exponent code 31 (all ones) is reserved for Inf / NaN; 0 is for zero and subnormals (no hidden leading 1).
  - The largest usable exponent code is 30, unbiased exponent $30-15=15$; with a 10-bit all-ones mantissa the significand is $2-2^{-10}$. So the max is $2^{15}\times(2-2^{-10})=65504$.
  - Smallest normal: $2^{-14}\approx 6.1\times 10^{-5}$ (exponent code 1, unbiased $1-15=-14$).
  - Example: 12 in binary is $1.1_2\times 2^3$, unbiased exponent 3, stored exponent $3+15=18$, mantissa the bits after the hidden 1: `1000000000`.
- **BF16**: 1 + 8 + 7. Same 8 exponent bits as FP32, so **the same range as FP32**; only 7 mantissa bits, even fewer than FP16's 10, so less precision.

#### Loss scaling
- FP16's smallest normal is about $6.1\times 10^{-5}$; many gradients are smaller and become 0 (underflow) if stored in FP16.
- Fix: multiply the loss by a scale $S$ so backward gradients are also scaled by $S$ and small values stay inside the FP16 range.
- Before clipping and the optimizer, **unscale** by $1/S$ to restore the true scale. The order must be **unscale, then clip, then step**. Otherwise the clip threshold is applied to gradients already multiplied by $S$, which is equivalent to shrinking the threshold by $S$.
- **Dynamic scaling** (e.g. `GradScaler`): if a step's gradients contain Inf / NaN, skip the update and halve $S$; after many good steps, double $S$.
- BF16 has the same range as FP32, so small gradients generally don't underflow and **loss scaling is usually unnecessary**.

#### Autocast
- Autocast picks precision **per op**, rather than casting the whole network:
  - `matmul` / `linear` / `einsum` → bf16 or fp16;
  - `exp` / `log` / `pow` / `sum` / `softmax` / `log_softmax` / `norm` / `layer_norm` → fp32;
  - other ops follow the input dtype.
- A hand-written `cross_entropy` can mix dtypes; the safest fix is `logits.float()` before the loss.
- **Parameters and gradients stay fp32**. Autocast also caches a bf16 copy of the weights for the low-precision matmul / linear.
- Only the forward pass and the loss go inside `with ctx`; `backward`, grad clipping, and `optimizer.step` stay outside.

### Result 1: accumulation (`mixed_precision_accumulation`)
- Adding 0.01 a thousand times should give 10.
- fp32 accumulator + fp32 0.01: **10.0001** (0.01 is not exact in fp32 either, so there is a little rounding).
- fp16 accumulator: **9.9531**. Once the running sum gets large, fp16's mantissa can no longer resolve such a small increment, so later additions of 0.01 are partly swallowed.
- fp32 accumulator adding fp16 0.01: **10.0021** in both variants. The error comes from storing 0.01 in fp16 as ≈ $0.0100021$; times 1000 that is ≈ 10.0021. **The fp16 → fp32 conversion itself is exact** (every finite FP16 value is a subset of FP32).
- Lesson: low precision is fine for the arithmetic; **accumulate in high precision**.

### Result 2: part (a) ToyModel and part (b) LayerNorm

#### Part (a): dtypes under FP16 autocast
- Parameters: fp32.
- `fc1` output: fp16 (linear is lowered).
- LayerNorm output: fp32 (`layer_norm` is on the fp32 list).
- Logits: fp16 (another linear).
- Loss: fp32.
- Gradients: fp32.

#### Part (b): LayerNorm steps that are precision-sensitive
- **Squaring and summing**: a moderately large input, squared, can exceed FP16's 65504. This is a **range** problem (overflow).
- **Mean and variance accumulation**: too few mantissa bits lose precision (the same class of bug as 0.01 × 1000 above).
- **Dividing by $\sqrt{\mathrm{var}+\varepsilon}$**: a tiny variance can underflow in FP16, or $\varepsilon$ can round in low precision — also mostly a **range** issue.
- Switching to BF16 removes those range problems, because it has the same exponent bits as FP32. But BF16 has only 7 mantissa bits, fewer than FP16, so **accumulation precision is actually worse**. Mean and variance should still be computed in fp32. That is why PyTorch autocast keeps `layer_norm` on the fp32 list even under BF16.

### Implementation: `--mixed-precision` in `benchmark.py`
- Function default `mixed_precision=False`. On the CLI use `action="store_true"`; do not use `type=bool`.
- Earlier bug: `type=bool` plus `action="store_true"` with `default=True` meant the flag was effectively always on. `type=bool` is wrong on its own too: `bool("False")` is `True`.
- Context:
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
  # backward / clip / step stay outside ctx
  ```
- An earlier version used `torch.autocast(device_type=args.device, ...)`: the function has no `args`, so import raised `NameError`; and `device_type` only accepts type names like `"cuda"` / `"cpu"`, not `"cuda:1"`. The correct argument is `device.type`.

### Result 3: local RTX 3060 Laptop (forward only)
- Setup: small, B=4, S=256, forward only, n=30.
- FP32 **0.0677 s**, BF16 **0.0573 s**, speedup **1.18×**.
- n=10 once gave a misleading 1.36×: that run's FP32 std was 0.015, and the noise inflated the ratio. The n=30 numbers are stable.
- **Amdahl**: if GEMM is ~60% of the time and the other 40% is unaccelerated, the time ratio is $0.4 + 0.6/k$. The measured ratio $0.0573/0.0677\approx 0.847$ implies GEMM itself was only about **1.34×** faster.
- Why the speedup is modest: matrices are small, so SMs are underfed; launch overhead and elementwise ops are a large share, and bf16 does not help those; autocast also inserts kernels that cast weights / activations to bf16.
- Also, as far as I know, on GeForce a BF16 Tensor Core with fp32 accumulate has halved throughput — **not verified on this 3060**.
- A theoretical 2× applies only to memory-bound work whose traffic is halved; this run is not a pure bandwidth bound, and the matrices are not large enough either.

### Result 4: part (c) A100 SXM4 80GB (forward+backward)
- Setup: PyTorch 2.11.0+cu128, `--options 1`, B=4, S=512, warmup 5, n=30. Code was the assignment `benchmark.py` plus A1 `cs336_basics`, unchanged. All eight runs finished with no OOM. Std was within about 3 ms on every run.

| Size | FP32 (s) | BF16 (s) | Speedup | Mem FP32 / BF16 (GiB) |
|---|---:|---:|---:|---:|
| small | 0.1418 | 0.0624 | 2.27× | 4.76 / 3.81 |
| medium | 0.4381 | 0.1390 | 3.15× | 12.22 / 10.08 |
| large | 0.9409 | 0.2452 | 3.84× | 25.02 / 20.80 |
| xl | 2.8083 | 0.5219 | 5.38× | 53.57 / 52.29 |

- **Speedup grows with model size** (2.27× → 5.38×): small is slow not because VRAM isn't full, but because **compute is underutilized**. Matrices are small, SMs are underfed, and launch plus elementwise work is a large share. As the model grows, GEMM's share and size both increase, Tensor Cores get closer to peak, and cast overhead is amortized.
- **Memory savings come mainly from activations**: linear outputs become bf16, so backward intermediates are about half the size. Parameters and gradients stay fp32, unchanged; autocast also stores an extra bf16 copy of the weights. At xl, parameters plus gradients are ~25 GiB and the bf16 weight copies add ~6 GiB, which nearly cancels the activation savings, so the net is only 1.3 GiB. Small has a higher activation share, so it saves more (~20%).
- A short write-up line: **BF16 speedup rises with model size, from 2.3× at small to 5.4× at xl; memory savings come mainly from activations, while parameters and gradients stay fp32.**

### Steps (reproduce)
1. Learn FP16 / BF16 range vs precision, and why FP16 needs loss scaling (unscale before clip).
2. Run `mixed_precision_accumulation` and confirm accumulation must be in fp32.
3. Under autocast, print dtypes of each ToyModel output, the parameters, the loss, and the gradients (part a).
4. Walk through LayerNorm's square / sum, mean / variance, and divide by $\sqrt{\mathrm{var}+\varepsilon}$; separate range issues from precision issues. After BF16 removes the range problems, keep mean and variance in fp32 (part b).
5. Add `--mixed-precision` to `benchmark.py`: `action="store_true"`, default False; build `torch.autocast` with `device.type`, and `nullcontext()` when it is off; put only forward and loss inside `with ctx`.
6. First compare FP32 / BF16 locally on a small model with n≥30, and look at the std so a short noisy run is not mistaken for speedup.
7. On one GPU large enough for xl, run forward+backward for small–xl, record time and reserved memory, then explain the speedup and memory changes.

### Common mistakes (mine today)
1. **Said FP16's 5 exponent bits lose precision**: those bits control range (overflow / underflow); precision loss comes from too few mantissa bits.
2. **Thought the fp16 → fp32 conversion loses precision**: the conversion itself is exact; the error was introduced when the value was first stored in fp16.
3. **Guessed that gradients are bf16 under autocast**: parameters and gradients stay fp32.
4. **Thought mixed precision saves memory via accumulation**: the savings are in activations (and backward intermediates); accumulation should stay in fp32 and does not save memory that way.
5. **Said that elementwise kernels not using Tensor Cores meant no type conversion happened**: casts do happen, and they add extra kernels.
6. **At first attributed the small local speedup to VRAM not being full**: the real issue was underutilized compute (small matrices, launch overhead, cast kernels).
7. **Wrote `--mixed-precision` as `type=bool` with `default=True`**: the flag was effectively always on; `bool("False")` is also `True`.
8. **Passed `args.device` into `torch.autocast`**: no `args` in the function (NameError), and `"cuda:1"` is not a valid `device_type`; use `device.type`.

### Noteworthy points
1. Autocast selects precision per op; it does not cast the whole network to bf16 / fp16.
2. Parameters and gradients stay fp32; autocast also caches a bf16 weight copy, so mixed precision does not always cut memory in half.
3. BF16 usually needs no loss scaling, but with only 7 mantissa bits, reductions such as mean, variance, and loss should stay in fp32.
4. The fp16 → fp32 conversion is exact; when you see error, check whether the value was already rounded into fp16 earlier.
5. Speedup grows with model size because GEMM share and size grow and Tensor Cores get closer to peak — not because VRAM is full.
6. A theoretical 2× only applies to memory-bound work whose traffic is halved. Small matrices plus launch overhead and cast kernels cut the realized speedup a lot.
7. As far as I know, on GeForce a BF16 Tensor Core with fp32 accumulate has halved throughput; this was not verified on this 3060.
8. Always look at the std: the n=10 1.36× was inflated by high FP32 variance.

### Recall questions
1. **Why is the FP16 maximum 65504?**
   <details><summary>Answer</summary>The all-ones 5-bit exponent (31) is reserved for Inf / NaN, so the largest usable code is 30, unbiased exponent $30-15=15$; a 10-bit all-ones mantissa gives significand $2-2^{-10}$. Max = $2^{15}\times(2-2^{-10})=65504$.</details>
2. **Why does BF16 usually not need loss scaling?**
   <details><summary>Answer</summary>BF16 has the same 8 exponent bits as FP32, so the same range, and small gradients generally don't underflow to 0. Loss scaling is mainly to rescue FP16's range.</details>
3. **In which order should unscale and clip run, and why?**
   <details><summary>Answer</summary>Unscale (multiply by $1/S$) first, then clip, then the optimizer. Otherwise the clip threshold is applied to gradients already scaled by $S$, which shrinks the threshold by a factor of $S$.</details>
4. **Why accumulate in fp32? What does adding 0.01 a thousand times show?**
   <details><summary>Answer</summary>An fp16 accumulator gave 9.9531, already off; an fp32 accumulator adding fp16 0.01 gave 10.0021, because 0.01 stored in fp16 is ≈ 0.0100021, and the fp16 → fp32 conversion itself is exact. Lesson: accumulate in high precision.</details>
5. **What dtype is the LayerNorm output under autocast? BF16 no longer has range issues, so why still compute mean and variance in fp32?**
   <details><summary>Answer</summary>fp32; `layer_norm` is on autocast's fp32 list. BF16 matches FP32's exponent bits, so overflow / underflow go away, but it has only 7 mantissa bits, so accumulation precision is worse than FP16. Mean and variance therefore stay in fp32.</details>
6. **Why does BF16 save so little memory at xl?**
   <details><summary>Answer</summary>The savings are mainly in activations. Parameters and gradients stay fp32 (~25 GiB at xl), and autocast also stores a bf16 weight copy (~6 GiB), which nearly cancels the activation savings, leaving only 1.3 GiB net.</details>
7. **Why does the speedup grow with model size?**
   <details><summary>Answer</summary>On a small model the matrices are small, SMs are underfed, and launch plus elementwise work is a large share that bf16 doesn't help. As the model grows, GEMM share and size increase, Tensor Cores approach peak, and cast overhead is amortized, so the speedup rises from 2.3× at small to 5.4× at xl.</details>
8. **By Amdahl, GEMM is 60% of the time and the total time ratio is ≈ 0.847. How much faster was GEMM itself?**
   <details><summary>Answer</summary>$0.4 + 0.6/k = 0.847$ → $k \approx 1.34\times$. So a 1.18× end-to-end speedup means GEMM was only about 1.34× faster.</details>
9. **Under autocast, what precision are parameters and gradients? What happens to the weights?**
   <details><summary>Answer</summary>Both stay fp32. Autocast caches a bf16 copy of the weights for the low-precision matmul / linear.</details>
10. **Why is 12 stored in FP16 with exponent 18?**
    <details><summary>Answer</summary>$12 = 1.1_2\times 2^3$, unbiased exponent 3, plus bias 15, so the stored exponent is $18$.</details>

#### Next / TODO
- §2.1.5 is done. §2.1.6 memory profiling is now started in the 2026-11-16 section below; part (f) is not done.
- Journal not committed; no commit or push this round.

## Day — §2.1.6 Memory profiling (memory_profiling) (2026-11-16)

#### What I did
- Studied the two memory-observation methods in assignment §2.1.6: PyTorch memory snapshots (`memory_viz`) and Nsight Systems CUDA memory tracking plus PyTorch NVTX.
- Recorded the 16 Nov local small recapture, the `--memory-profile` snapshot I generated, and the xl-forward snapshot facts already checked.
- **Part (f) is not done.** There is still no single-block saved-for-backward total, no top-5 operator shares, and no final backward-gradient size; S=2048 was not run; no new GPU was rented in this round. What follows is the method, the evidence already checked, and the limits.

### Background / motivation
- Timing answers “where the time went.” Explaining “why memory rose, and which tensors stayed alive into backward” needs allocation events aligned with operator or module ranges.
- The two tools are not interchangeable:
  - PyTorch `_record_memory_history` / `_dump_snapshot` see **blocks in the caching allocator**: `alloc`, `free_requested`, `free_completed`, plus the Python allocation stack. `memory_viz` labels generations such as `bd2b500000_0`.
  - Nsight Systems `--cuda-memory-usage` sees **CUDA-level memory usage / allocation events**. The default Memory track is often memcpy / memset, i.e. transfers and fills, not a usage curve.
- Part (f) also asks for **PyTorch-provided NVTX** (for example `autograd-shapes-nvtx` and `functions-trace`) so a single TransformerBlock’s forward saved tensors and backward gradient change can be lined up.

### Formulas / rules
- Bytes of one FP32 activation are $B\cdot S\cdot D\cdot 4$. Convert to MiB by dividing by $1024^2$, not $1000^2$.
- For local small, B=1, S=256, $d_{ff}=3072$, an FP32 $(B,S,d_{ff})$ block is

  $$
  1\times 256\times 3072\times 4 = 3{,}145{,}728\ \text{B} = 3\ \text{MiB}.
  $$

  Attention scores with shape $(B,H,S,S)=(1,12,256,256)$ in FP32 are also 3 MiB; **the same size does not mean the same source**.
- Earlier, at xl, B=4, S=128, $d_{ff}=10240$, an FP32 $(B,S,d_{ff})$ block is 20 MiB; an FP32 $2560\times 10240$ weight matrix is 100 MiB. Those are size clues, not a proof of origin.
- `memory_viz` address generations (official JS `block_version` / `version_space`): each address starts at 0; `alloc` and `free_requested` read the current generation without incrementing; `free_completed` returns it and then increments. The display name is `b{addr.toString(16)}_{version}`. So `d2b500000_0` is generation 0 of address `0xd2b500000`, not “the first event in the whole snapshot.”
- The caching allocator reserves large CUDA segments and carves tensors from the pool. CUDA may only see those pool allocations, not every tensor alloc / free. **Do not treat rises and falls on the GPU Memory Usage curve as tensor allocated, or as saved-for-backward.**

### Inputs / outputs / shapes
- The object of study is activation, weight, and gradient memory during a training step, not a new model API.
- Local configuration already run and checked: small, B=1, S=256, FP32 (no `--mixed-precision`), `--w 2 --n 2 --options 2` (full step: forward + backward + clip + optimizer). This is a method demo, **not the official xl result for (f)**.
- The four xl S=128 PyTorch pickles are for observations such as (e). They are not Nsight reports and must not be mixed with nsys.
- S=2048 was not run in this round.

### Steps (in prose)
1. Use a PyTorch snapshot to see who allocated a block: start `_record_memory_history`, dump a pickle, and open the Active Memory Timeline in [memory_viz](https://pytorch.org/memory_viz). To catch parameter moves to the GPU, start recording before that happens; to see steady-state behavior, start after warmup. Do not mix the two goals.
2. When a block is selected, distinguish “size of this allocation” from “total usage after it.” With Detail set too low, a blue bulge may be many small tensors merged for display. Scroll the stack to your own `.py` line numbers.
3. A block that existed before recording started may appear as a ghost block: still live, but with no alloc stack. That is not by itself a leak.
4. For (f) in Nsight, the capture command needs CUDA memory tracking and actually supported PyTorch operator / module NVTX. `-t cuda,nvtx` alone is not enough. Local `nsys profile --help` (2026.6.1.218) confirms `--cuda-memory-usage` (default false) and `--pytorch` (including `autograd-shapes-nvtx` and `functions-trace`; default none).
5. After exporting SQLite, count Allocation vs Deallocation in `CUDA_GPU_MEMORY_USAGE_EVENTS`, then look at NVTX names. Do not write reserved-memory totals as per-tensor allocated.
6. Local recaptures must use a new filename; do not `--force-overwrite` an old report.

### Observations already checked (evidence only)

#### Part (e): large blocks in the xl forward snapshot
- One **20 MiB** allocation has a stack: `swiglu.py:52` → `linear.py:45` → `einsum` / `bmm`. The size matches an FP32 output of shape $(B,S,d_{ff})=(4,128,10240)$. This is a feed-forward **intermediate activation**, not the attention score matrix and not a weight.
- There is also a **100 MiB** allocation with no stack (ghost block). The size matches an FP32 $2560\times 10240$ weight, which is only a clue; **this snapshot cannot confirm the source**. So 20 MiB still cannot be written down as “the largest allocation.”

#### Why the older `small_full_process.nsys-rep` is not enough for (f)
- The report header and embedded command show Nsight Systems 2026.6.1.218 on `LAPTOP-VA5CFNA3`, an RTX 3060 Laptop, model **small**, and only `-t cuda,nvtx`. There is **no** `--cuda-memory-usage` and no `--pytorch`.
- In the GUI, the Memory track is memcpy / memset, which is consistent with usage tracking being off. That report cannot stand in for xl (f) data.

#### Local small recapture (method demo, 2026-11-16 17:29 Asia/Hong_Kong)
- Full command (no `--force-overwrite`):

  `nsys.exe profile -t cuda,nvtx --cuda-memory-usage=true --pytorch=autograd-shapes-nvtx,functions-trace -o C:\Users\zhaohui\AppData\Local\nsys_a2\small_mem_f_20261116_172922`, then the existing `venv_launch.py cs336_systems.benchmark --model-size small --batch-size 1 --context-length 256 --w 2 --n 2 --options 2 --device cuda`.
- Report path: `C:\Users\zhaohui\AppData\Local\nsys_a2\small_mem_f_20261116_172922.nsys-rep` (and the same-stem `.sqlite`). Both nsys and the benchmark exited 0. Non-fatal warning: CPU sampling / context-switch tracing was disabled without admin rights.
- SQLite evidence: `CUDA_GPU_MEMORY_USAGE_EVENTS` has 735 rows, **all Allocation, zero Deallocation**. That usage curve is not per-tensor alloc / free, and it is not saved-for-backward.
- NVTX already includes `aten::*`, autograd engine ranges, and module names such as `TransformerLM.layers.0` … `layers.11`. The GUI can be read later; **this log does not write a top-5 answer, and it does not mark this small capture as (f) complete.**

#### The `--memory-profile` snapshot I generated (same small configuration)
- The real pickle (protocol 4) was parsed with a restricted Unpickler that forbids `find_class` and `persistent_load`; it contained only basic containers, not executable globals.
- Windows warnings about unsupported C++ stacks and a temporary kernel cache were non-fatal and do not change the event conclusions below.
- The target block `d2b500000_0` (that is, `bd2b500000_0`) is 3,145,728 bytes:
  - `alloc` index **930**, 2026-11-16 17:48:28.764 Asia/Hong_Kong; stack `swiglu.py:23:silu` → `swiglu.py:52` → `transformer.py:47` → `transformer_lm.py:50` → `benchmark.py:216`.
  - `free_requested` index **3659** and `free_completed` index **3660**, same timestamp 17:48:28.824. **The free stacks are empty**, which is why the GUI cannot show a release site; more screenshots will not help.
  - Lifetime ≈ 59,800 µs. The snapshot `device_traces` have no NVTX, so the forward / backward boundary is inferred from annotations and time only: after the first `zero_grad` ended and before the first `optimizer.step` started; by index 1691 the program was already in `backward` at `benchmark.py:220` while this block was still live; the next framed event after the free is still inside that backward. So it is fair to infer: **allocated in forward, kept across the forward boundary, freed during backward**. The layer index, the exact backward operator, and a precise nsys-range alignment cannot be claimed.
- There are 48 allocs of 3 MiB with the same `swiglu.py:23:silu` stack (12 layers × 2 measured steps × 2 consecutive allocs per layer). The pair of index 930 is index 929 (a different address). Without the `swiglu.py` source, it is not possible to say whether line 23 is two internal operators or one input and one output.
- A separate 3 MiB softmax allocation matches $(1,12,256,256)$ FP32 in size. That is only a size-and-stack candidate, **not confirmed as the global maximum**.

### Numerical examples
- 20 MiB (xl, B=4, S=128, $d_{ff}=10240$, FP32 activation) and 3 MiB (local small, B=1, S=256, $d_{ff}=3072$, FP32 activation) both come from $B\cdot S\cdot D\cdot 4 / 1024^2$.
- The event indices for `d2b500000_0` are 930 / 3659 / 3660, not “whatever block was clicked in the plot.”
- The §2.1.5 journal already records reserved memory for xl forward+backward on an A100 (53.57 GiB FP32, 52.29 GiB BF16). Those are reserved-memory numbers from the mixed-precision timing experiment. **Do not treat them as the (f) peak or as saved-for-backward.** No new peak table is invented here.

### Common mistakes / notes
1. Treating the memcpy / memset track as a memory-usage curve.
2. Treating GPU Memory Usage deltas as each PyTorch tensor’s allocated / freed bytes.
3. Calling the confirmed 20 MiB activation “the largest allocation” while ignoring the 100 MiB stackless block.
4. Writing a local-small method demo as the official xl answer to (f).
5. Calling a ghost block a leak, or reading `_0` as the first event in the whole trace.
6. Pretending a snapshot without NVTX is precisely aligned to nsys ranges, or naming a layer / operator when the free stack is empty.
7. Using hook counts or event counts as if they were bytes (this comes up again in the next section).

### Recall questions
1. **Why is `-t cuda,nvtx` alone usually not enough for part (f)?**
   <details><summary>Answer</summary>That command traces CUDA APIs / kernels and whatever NVTX the program emits, but it does not enable `--cuda-memory-usage` by default (both the docs and local help say the default is false), and it does not add PyTorch operator NVTX (`--pytorch` defaults to none). Without usage events and module / operator ranges, you cannot attribute saved-for-backward memory to one TransformerBlock.</details>
2. **If `CUDA_GPU_MEMORY_USAGE_EVENTS` has only Allocation rows and no Deallocation, can you read saved-for-backward from it?**
   <details><summary>Answer</summary>No. That pattern is consistent with the caching allocator reserving CUDA segments and not `cudaFree`-ing them in the short run. The curve is reserved memory, not bytes of each saved or freed activation.</details>
3. **What does the `_0` in `d2b500000_0` mean?**
   <details><summary>Answer</summary>It is generation 0 of that address: `alloc` / `free_requested` / `free_completed` all use 0 until the first `free_completed` increments it. The next allocation at the same address becomes `_1`. It is not “the first event in the snapshot.”</details>
4. **Why can 3 MiB come from either SwiGLU or attention scores?**
   <details><summary>Answer</summary>In this small configuration both $1\times 256\times 3072\times 4$ and $1\times 12\times 256\times 256\times 4$ equal 3 MiB. You have to read the stack, not just the size.</details>
5. **Is part (f) done?**
   <details><summary>Answer</summary>No. The method, the limits, and some checked observations are logged; the top 5, the single-block total, the backward gradient size, S=2048, and an official xl capture are still missing.</details>

#### Next / TODO
- Do not mark all of §2.1.6, or part (f), complete.
- To continue (f), a report that actually has usage tracking and PyTorch module / operator ranges is needed (the existing small report is only a method demo), and the single-block read has to be done in the GUI; S=2048 and official xl results are still missing.
- Journal not committed; no commit or push this round.

## Day — §3.1 Saved tensors and §3.1.1 fusion (2026-11-17)

#### What I did
- Studied §3.1: which tensors autograd saves for backward (residuals / saved tensors), and that saves are usually references, not copies.
- Ran §3.1.1 with `cs336_systems/rmsnorm_test.py`: `torch.compile` plus `saved_tensors_hooks`, comparing Saving / Loading records before and after compile. The command was `uv run python .\cs336_systems\rmsnorm_test.py`.
- Reviewed the script statically (without writing “compile succeeded” as a verified Windows CUDA / inductor result). I later pasted compiled and eager hook output; the script was checked again read-only on 2026-11-17. This round did not re-run the experiment with the current script.
- The conceptual checkpoint question has been answered: when backward enters a checkpointed region, first rerun the needed forward from the saved input to restore internal activations, then compute gradients. §3.2 is still only a conceptual preview; the experiment has not been run.

### Background / motivation
- Backward does not mean “every forward operator stores every intermediate again.” Some operators need no input values, some need only one side, and some can rebuild a value from others during backward.
- Seeing what RMSNorm actually saves is what makes a change in saved tensors on a compiled path intelligible, and what decides whether the dropped item really was an independent chunk of memory. Hook records speak only to save behavior; they do not show how many kernels were generated, and they do not prove fusion.
- This also foreshadows activation checkpointing (recompute in exchange for keeping less). The §3.2 experiment has not started.

### Formulas
- For $y=x^2$, backward is $2x\cdot \mathrm{d}y$. Implementations usually save $x$; they need not also save $2x$.
- For $y=x+2$, backward just routes the incoming gradient back to $x$ and **does not need the value of $x$**.
- For $z=ab$: if both $a$ and $b$ need gradients, both are saved; if only $a$ needs a gradient, what is saved is $b$ (because $\partial z/\partial a=b$). I had this backwards and then corrected it.
- For a linear layer $y=xW$, the gradient w.r.t. $W$ needs $x$.
- In RMSNorm write $m=\mathrm{mean}(x^2)$ (last dimension), $r=(m+\varepsilon)^{-1/2}$, $v=x r$, $y=w v$.
- The derivative of $r$ w.r.t. $m$ is

  $$
  \frac{\mathrm{d}r}{\mathrm{d}m}=-\frac12(m+\varepsilon)^{-3/2}=-\frac12 r^3.
  $$

- The backward of `mean` needs the output shape, the reduced dimension, and the element count, not the value of the mean $u$ itself.
- Freezing a parameter’s gradient (`requires_grad=False`) does not mean “its value need not be saved in forward”: another parameter’s gradient may still use it, e.g. $y=w v$ must save $w$ if $v$ needs a gradient.
- Broadly, `torch.compile` captures a graph, jointly optimizes forward and backward (including what to save vs recompute), and then the chosen backend generates code or calls operator libraries. Elementwise operators are easier to fuse; matrix multiplies often still go through specialized libraries. Not every operator can be fused: that depends on the backend, device, dependencies, operator support, and cost. Data-dependent Python control flow or external side effects can cause a graph break.
- The backend written in the current script is `aot_eager`: it uses AOT Autograd, so the forward/backward graphs and saved tensors can differ from pure eager, but the graph is still executed eagerly. That is **not** the same thing as Inductor / Triton having generated fusion kernels. Hook output is not a basis for claiming that fusion kernels were produced, or that a CUDA peak reduction was measured.

### Inputs / outputs / shapes
- Input $x$ has shape $(4,512,2560)$, FP32, about $4\times 512\times 2560\times 4=20{,}971{,}520$ bytes $=20$ MiB.
- $r$ has shape $(4,512,1)$, FP32, $8192$ bytes $=8$ KiB.
- $w$ has shape $(2560,)$, FP32, $10240$ bytes $=10$ KiB.
- $v=x r$ has the same shape as $x$, also 20 MiB.
- Output $y=w v$ has the same shape as $x$.

### Steps (in prose)
1. Open `torch.autograd.graph.saved_tensors_hooks` around both forward and backward; `pack_hook` / `unpack_hook` should only observe and return the tensor. Mark the observers compile-disable; **do not disable the RMSNorm forward itself**. This is not the same as autocast: autocast picks dtypes per operator, while `torch.compile` returns a callable optimized object and usually compiles on the **first call**.
2. In the eager graph, note which Saving / Loading records appear, with shapes and `grad_fn`, and map them to $x$, $r$, $v$, and $w$.
3. On the compiled path, see which item is gone. If $v$ is no longer kept long-term, explain the missing ~20 MiB as a **derivation from independent storage on that graph** (references + storage dedup + $v$ can be rebuilt from $x$ and $r$), not by reading “6 records became 3” as a halved memory footprint, and not as a measured CUDA peak.
4. If comparing eager and compiled, isolate the two runs so two backwards on the same parameters do not accumulate gradients.

### Script review and current code (static; the experiment was not re-run here)
- `from cs336_basics.rmsnorm import RMSNorm` is still commented out. The script uses a local, handout-style `rsqrt` implementation, not the A1 `sqrt`-then-divide version with a float32 upcast. If hook output matches the handout rsqrt version, that is expected, and is not evidence that the A1 implementation was compiled into that form.
- The first version created `ln_compiled = torch.compile(ln)` on default CPU, but called `ln(x)`, so the compiled object was unused. That was already pointed out in review.
- Re-reading `rmsnorm_test.py` on 2026-11-17: `x` and `RMSNorm` now set `device="cuda"`; `rmsnorm_fwd` is wrapped with `torch.compile(..., backend="aot_eager")` as `rms_fwd_compiled`; the module `forward` actually calls `rms_fwd_compiled`. The `torch.compile(ln)` line is commented out. So **the current script does invoke the compiled function**, the device has been changed to CUDA, and the backend written in the file is `aot_eager`.
- That only records how the script is written now. It **does not prove** that default inductor / Triton CUDA compile works on Windows. A change in AOT Autograd forward/backward graphs and saved tensors under `aot_eager` is not the same as Inductor / Triton fusion kernels having been generated.
- The “compiled” hook output pasted below was not re-run against this exact `aot_eager` file state in this round. The pasted records and the current script are therefore kept distinct: the former is user-provided save behavior; the latter is what a read-only inspection of the script shows. The paste is not treated as proof that the current `aot_eager` path emitted fusion kernels, or that a CUDA peak reduction was measured.

### Numerical examples (hook output I provided)
The eager run printed 6 Saving and 6 Loading records, in this order:
1. $x$: `[4, 512, 2560]`, FP32, `grad_fn=None`
2. $r$: `[4, 512, 1]`, `RsqrtBackward`
3. $r$ again
4. $x$ again
5. $v$: `[4, 512, 2560]`, `MulBackward`
6. $w$: `[2560]`, `grad_fn=None`

This is **not** three large tensors and 60 MiB of independent memory. Saves are usually references; when several operators record the same storage, deduplicate: $x$ is 20 MiB and $v$ is 20 MiB, together 40 MiB, plus $r$ at 8 KiB and $w$ at 10 KiB. The repeated $x$ and $r$ records do not double the bytes.

The compiled records I pasted match the handout: only $x$, $w$, and $r$ are kept long-term (three Saving and three Loading records, shapes `[4, 512, 2560]`, `[2560]`, `[4, 512, 1]`, all with `grad_fn=None`). $v$ is no longer retained long-term. The “about 20 MiB less retained” figure is a derivation from **independent storage on that graph** (dropping the 20 MiB $v$ tensor); **it is not a measured CUDA peak**, and it is not evidence that fusion kernels were generated. Hooks only attest to save behavior. The interpretation is correct: if backward needs $v$, it can be rebuilt from $x$ and $r$ and then used in the gradient for $w$.

### Common mistakes / notes
1. Seeing 6 records become 3 and calling that “memory halved.” Record counts are not bytes.
2. Adding every save of $x$ as another 20 MiB; deleting a Python name also does not free that storage.
3. Reversing the $z=ab$ rule: if only $a$ needs a gradient, what is saved is $b$.
4. Assuming a `requires_grad=False` weight never appears in saved tensors.
5. Using `@torch.compile.disable` on the whole forward and leaving only the hooks—that no longer measures the compiled graph.
6. Treating this `backend="aot_eager"` script, or the pasted hook output, as “Windows CUDA inductor / Triton fusion has been verified,” or as “a CUDA peak reduction has been measured.” `aot_eager` executes the AOT Autograd graph eagerly; that is not a fusion kernel.
7. Activation checkpointing is still only a conceptual preview; the §3.2 experiment has not been run. The concept was answered correctly: when backward hits a checkpointed region, first rerun the needed forward from the **saved input of that region** to restore internal activations, then compute gradients. The whole model need not be recomputed first, and parameters, gradients, and optimizer state are not removed.
8. Not every operator can be fused by `compile`; data-dependent Python control flow or external side effects can graph-break. Hooks cannot be used to count kernels or to prove fusion.

### Recall questions
1. **For $y=x^2$ and $y=x+2$, what does backward need to save?**
   <details><summary>Answer</summary>$y=x^2$ usually saves $x$, not necessarily $2x$. $y=x+2$ does not need the value of $x$; the incoming gradient is passed through.</details>
2. **For $z=ab$, when only $a$ needs a gradient, what is saved?**
   <details><summary>Answer</summary>$b$, because $\partial z/\partial a=b$.</details>
3. **Why are the eager 6 records not 60 MiB? What is the “about 20 MiB less retained” figure based on, and what is it not?**
   <details><summary>Answer</summary>$x$ and $v$ are 20 MiB each, so the large unique blocks are 40 MiB, plus 8 KiB for $r$ and 10 KiB for $w$. Repeated $x$ and $r$ records are references to the same storage. If the compiled path no longer keeps $v$ long-term, about 20 MiB less is retained; that is a derivation from independent storage on that graph, not a measured CUDA peak, and not proof that fusion kernels were generated.</details>
4. **How does `torch.compile` differ from autocast? What should hook `disable` exclude?**
   <details><summary>Answer</summary>`torch.compile` wraps a function or module and returns a callable optimized object, usually compiled on first call. Autocast selects dtypes per operator. `disable` should exclude only the observation hooks, not the forward you are studying.</details>
5. **Why can $v$ be reconstructed from $x$ and $r$ when computing the gradient of $w$?**
   <details><summary>Answer</summary>$v=x r$ and $y=w v$. If $v$ is needed, it need not be kept long-term; multiplying the still-saved $x$ and $r$ rebuilds it.</details>
6. **How does `torch.compile` automatically handle these operators? Can every operator be fused? Do hook records prove that fusion kernels were generated?**
   <details><summary>Answer</summary>It captures a graph, jointly optimizes forward and backward (including save vs recompute), and then the backend generates code or calls operator libraries. Elementwise ops fuse more easily; matmuls often still use specialized libraries. Not all operators can be fused: that depends on backend, device, dependencies, support, and cost. Data-dependent Python control flow or external side effects can graph-break. Hooks only prove save behavior, not kernel counts or fusion. The current script’s `aot_eager` backend executes the AOT Autograd graph eagerly, so it is even less a basis for claiming Inductor / Triton fusion kernels.</details>
7. **If a block keeps only its input, what does backward do? Must the whole model be recomputed first? Are parameters and optimizer state removed?**
   <details><summary>Answer</summary>When backward enters that checkpointed region, first rerun the needed forward from the saved input to restore internal activations, then compute gradients. The whole model need not be recomputed first, and parameters, gradients, and optimizer state are not removed. This is conceptual; the §3.2 experiment has not been run.</details>

#### Next / TODO
- The §3.1 / §3.1.1 comparison is now in the journal; the conceptual checkpoint question has been answered correctly. The §3.2 activation checkpointing experiment has not started.
- Journal not committed; no commit or push this round.

## Day — §3.2 Activation checkpointing (afternoon of 2026-11-17)

#### What I did
- Wrote `cs336_systems/checkpoint_test.py` on local CUDA with the small config: first an ordinary forward and backward of a single `TransformerBlock`, then a comparison with `torch.utils.checkpoint` (`use_reentrant=False`).
- I had wrapped the block in `torch.compile(..., fullgraph=True, backend="aot_eager")`. Capture failed because A1 `cs336_basics/nn_utils.py` around line 43 uses `nvtx.range` as a context manager, which broke fullgraph capture. Compile was then removed, and the runs used eager. That `aot_eager` is not Inductor / Triton fusion was already recorded in §3.1.1 the same morning.
- Used `saved_tensors_hooks` to see how much the ordinary and checkpoint paths save for backward; then compared outputs and gradients with the same weights and the same input, without `step` or clip; then measured `torch.cuda.max_memory_allocated` over a full forward and backward.
- Discussed the nested-prefix approach for the official gradient_checkpointing part (a). The xl and neighboring-region measurements for (b) were not run. §4.1 is about to start; its experiments have not started.

### Background / motivation
- Checkpointing trades a recompute of the needed forward, when backward enters a region, for keeping fewer internal activations long-term. Hooks report tensors saved for backward, not a CUDA peak, and not deduplicated independent storage.
- A single block’s peak over a full forward and backward still includes parameters, gradients, the input, and intermediates that reappear when the region is recomputed, so “less is saved” does not mean the peak must fall. With several layers, the difference is easier to see if the internals of every layer are not all kept at once.

### Formulas
- The small input occupies

  $$
  4\times 256\times 768\times 4=3{,}145{,}728\ \text{bytes}=3\ \text{MiB}.
  $$

- The candidate for official (a) cuts the forward into nested prefixes $1..N,\ 1..(N-1),\ \ldots,\ 1$, and those regions share one stored input $x_0$, rather than keeping a separate input for every layer.
- Under the problem’s abstraction that ignores checkpoint bookkeeping and treats a single layer’s residual activation as dominant, this shared-$x_0$ nesting gives $O(1)$ peak activation and $O(N^2)$ compute. That is an asymptotic statement about activation peaks, **not** a claim that parameters, gradients, or optimizer state become $O(1)$ in total memory. The extra cost of managing nested regions in Python also cannot be dismissed in general.
- Along the way I had analogized the cost as $O(D)$. That count applies only when each layer has its own independent input, not to the shared-$x_0$ scheme.
- Uniform segmentation was also discussed: if a layer activation is about $A$ and a segment boundary about $R$, then $M(k)\approx (N/k)A+kR$, with a heuristic $k^*\approx\sqrt{NA/R}$; which legal integer to use should be measured. Binary recursion is typically $O(R+A\log N)$ peak activation and $O(N\log N)$ compute. Revolve was only a conceptual extension and was not implemented.

### Inputs / outputs / shapes
- The config is small: $B=4$, $S=256$, $d_{\mathrm{model}}=768$, $d_{\mathrm{ff}}=3072$, `num_heads=12`, FP32, CUDA. $768/12=64$, which divides evenly.
- The single-block input $x$ has shape $(4,256,768)$: hidden vectors, not token ids.
- The scalar used for backward is the sum of the block output, not a vocabulary cross-entropy.

### How it was implemented (a record of this write-up, not a substitute solution)
- After aligning device and dtype with the input, the ordinary path called the eager `block` (compile already removed). The checkpoint path used `checkpoint(..., use_reentrant=False)`.
- The correctness comparison was done by me: the same weights and the same $x$, no `optimizer.step` and no clip; the output was `detach`ed and then cloned; input and parameter gradients were cloned; parameter gradients were cleared and $x$.grad was cleared as well (optimizer `zero_grad` alone does not clear $x$.grad); `None` gradients were stored as placeholders; then each parameter was compared with `zip` and `if` / `elif` / `else` plus `allclose` on the output, $x$.grad, and every parameter gradient. I reported that all of those comparisons were True. That is my own run; this log is a static record, the comparisons were not re-run here, and they are not claimed as an official full test suite.
- The peak-measurement path evolved as follows: hooks and the comparison copies were removed so extra references would not inflate or disturb the peak; gradients were cleared before the measurement; `synchronize` ran before `reset_peak_memory_stats`; after the full forward and backward, `synchronize` ran again and then `max_memory_allocated` was read. Warmup was also recommended. The script as reviewed at 18:07 on 2026-11-17 (Hong Kong Time) still lacked warmup and a post-backward sync. Later edits were not reviewed again, so the measurement procedure is not claimed to be complete.
- In the current source, `four_blocks` calls the **same** `TransformerBlock` instance four times in a row; it is not four modules with independent weights. I described the four-layer peak comparison as wrapping all four layers as one checkpoint region, not a per-block comparison and not a comparison of adjacent smaller or larger regions (those were not done). The file also defines `four_blocks_checkpoint`, which places the whole `four_blocks` into `checkpoint(..., use_reentrant=False)`; as currently written, that function makes this call twice in sequence. The measurement section currently activates ordinary `four_blocks`. The four-layer peaks I reported were not re-run in this logging pass, and are not treated as a reproduction of the currently active path.

### Numerical examples (reported by me; not re-run in this pass)
- Hook running totals (not storage-deduplicated, and not a CUDA peak): $189.47$ MiB for an ordinary single block; $3$ MiB for non-reentrant checkpoint, consistent with keeping only that $3$ MiB input long-term. The sum of hook records, deduplicated independent storage, and `max_memory_allocated` are not the same quantity.
- Single-block CUDA peaks (reported): $202.72$ MiB ordinary vs $203.59$ MiB checkpoint. That does not prove a peak reduction. Whether the two measurements were in independent processes was not confirmed at the time.
- Four layers (reported): $545.73$ MiB ordinary vs $352.18$ MiB for a whole-region checkpoint, a difference of $193.55$ MiB, about $35.5\%$. Time cost was not measured.

### Common mistakes / notes
1. Treating the hook running total as a CUDA peak, or as already-deduplicated independent storage.
2. Seeing hooks drop from about $189$ MiB to $3$ MiB on a single block and concluding that the full forward/backward peak must drop by the same factor. During backward recompute, internal activations can reenter the peak window.
3. Calling only the optimizer’s `zero_grad` and assuming $x$.grad was cleared; a second backward on the same $x$ then accumulates the input gradient.
4. Describing shared-$x_0$ nested prefixes as the $O(D)$ scheme in which every layer has its own input; or reading $O(1)$ peak activation as “all memory, including parameters, gradients, and the optimizer, is $O(1)$.”
5. Marking official (b), or the (a) code submission, as done. The xl, $S=2048$, and neighboring-region measurements have not been run; (a) was discussed, but no sketch was submitted.
6. Claiming that peak measurement is already solid while warmup and the sync-before-read discipline have not been re-reviewed.

### Review (direct answers)
**Why are the hook totals of about $189.47$ MiB vs $3$ MiB not the same thing as the later CUDA peaks?**
<details><summary>Answer</summary>The hook total adds records of what is saved for backward, usually without storage dedup, and it is not the allocator peak. Checkpointing shrinks long-term saves to the region input ($3$ MiB here), so the hook number drops a lot. `max_memory_allocated` over a full forward and backward still includes parameters, gradients, the input, and intermediates that reappear during recompute, so a hook drop does not forecast a peak drop of the same size.</details>

**Why might the two single-block peaks be almost the same, while a four-layer whole-region checkpoint has more room to show a gap?**
<details><summary>Answer</summary>On a single block, backward recomputes that block’s forward, so the peak window can still hold intermediates comparable to the ordinary path. With several layers, the ordinary path often keeps internals from every layer at once; if four layers are checkpointed as one region, what is kept long-term is the region entrance, and the internals are recomputed in backward, so the peak is more likely to fall. Even if a four-layer gap was reported, that is only this whole-region wrapping; it does not replace a per-block or neighboring-region comparison, which were not done.</details>

**What is the abstract complexity of nested prefixes that share $x_0$? Why is it neither $O(1)$ total memory nor the $O(D)$ of per-layer independent inputs?**
<details><summary>Answer</summary>Under the problem’s abstraction (ignore bookkeeping; a single layer’s residual dominates), peak activation is $O(1)$ and compute is $O(N^2)$, because the prefixes share one $x_0$ and do not store a fresh input per layer. Parameters, gradients, and optimizer state remain, so total memory is not $O(1)$. The $O(D)$ count comes from giving every layer its own input and does not carry over to shared $x_0$. The cost of managing the nesting in Python also cannot be ignored in general.</details>

**What has to hold for an output-and-gradient comparison to be valid?**
<details><summary>Answer</summary>Both versions must use the same input and the same parameters; do not `step` or clip in between, or the weights have already changed. Keep a detached copy of the output, and copies of the input and parameter gradients, before clearing. Clearing parameter gradients is not enough: $x$.grad must be cleared too. Handle `None` explicitly, then compare parameter by parameter. Those are the conditions; I reported True, but the comparisons were not re-run for this log.</details>

**How far have official (a) and (b) actually been taken?**
<details><summary>Answer</summary>The nested prefixes, shared $x_0$, and the complexity above have been discussed for (a); no code sketch has been turned in, so it is not an assignment submission. The xl, $B=4$, $S=2048$ measurements and the neighboring smaller or larger regions required by (b) have not been run, so (b) is not complete. Time cost has not been measured either. §4.1 is only about to start; its experiments have not started.</details>

#### Next / TODO
- Official gradient_checkpointing (a): the approach has been discussed; no code sketch has been submitted.
- Official (b): xl, $B=4$, $S=2048$, and neighboring-region measurements have not been run; time cost has not been measured.
- The peak-measurement procedure is not claimed to be complete; §2.1.6 (f) is still unfinished.
- §4.1 is about to start; its experiments have not started.
