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
