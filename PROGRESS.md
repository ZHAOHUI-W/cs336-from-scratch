# Progress tracker

Update every Sunday. Mark: `[ ]` todo · `[~]` in progress · `[x]` done

## Goals
- [ ] Architecture mindset: can explain trade-offs without notes
- [ ] Engineering design: modules have contracts + tests + ADRs
- [ ] Complete A1–A5 in 6 weeks

## Week 1 — Tokenization, resources, A1 foundations
- [x] Day 1 map: deliverables + adapters contract (`journals/week-01.md`)
- [x] A1 starter at `assignments/a1-basics/` (run `scripts/fetch_large_assets.sh` before pytest)
- [ ] Read Lec 01 (tokenization / BPE)
- [ ] Read Lec 02 (memory, FLOPs, training loop primitives)
- [ ] ADR: BPE data structures (pair counts / updates)
- [ ] BPE encode/decode round-trip
- [ ] ADR: AdamW state + decoupled weight decay
- [ ] AdamW matches `torch.optim.AdamW` on a toy step
- [x] Journal `journals/week-01.md` (Day 1 map logged)

## Week 2 — Transformer + train (finish A1)
- [ ] Read Lec 03–04 (arch; MoE skim-only)
- [ ] ADR: Transformer block contract (norm placement, RoPE site, mask)
- [ ] RMSNorm + SwiGLU shape tests
- [ ] MHA + causal mask + RoPE
- [ ] Single-batch overfit
- [ ] Cloud: TinyStories (or handout) train
- [ ] A1 archived / submitted locally
- [ ] Journal week-02

## Week 3 — Systems (A2)
- [ ] Read Lec 05–08
- [ ] Profile Transformer block; write bottleneck note
- [ ] ADR: FlashAttention tiling / online softmax
- [ ] Triton FA2 forward matches reference
- [ ] Triton FA2 backward matches autograd
- [ ] FSDP multi-GPU smoke train
- [ ] Journal week-03

## Week 4 — Scaling (A3)
- [ ] Read Lec 09–12
- [ ] ADR: IsoFLOP experiment design (valid loss measurement)
- [ ] Sweep + U-curves
- [ ] Fit $N_{\\mathrm{opt}}(C)$, $D_{\\mathrm{opt}}(C)$
- [ ] Predict target budget
- [ ] Journal week-04

## Week 5 — Data (A4)
- [ ] Read Lec 13–14
- [ ] ADR: streaming pipeline stages + failure modes
- [ ] WARC → text on sample
- [ ] Filters + language ID
- [ ] MinHash + LSH (not $O(N^2)$)
- [ ] Journal week-05

## Week 6 — Alignment (A5)
- [ ] Read Lec 15–17
- [ ] ADR: SFT vs preference objective; β meaning
- [ ] SFT beats base on math eval
- [ ] DPO/GRPO: init loss sanity check
- [ ] Final eval + write-up
- [ ] Journal week-06
- [ ] Retro: top 5 design lessons in `docs/design/999-retro.md`

## A2 systems (logged in `journals/week-02.*.md`)

Status as of 2026-11-17. This does not rewrite the six-week checklist above, and it does **not** mark all of §2.1.6 or part (f) complete.

- [x] §2.1.4 Nsight Systems timing on local small (2026-11-11)
- [x] §2.1.5 mixed precision, including the local RTX 3060 and A100 tables already in the week-02 journals (2026-11-11)
- [~] §2.1.6 memory profiling (2026-11-16): method, limits, xl-forward (e) observations, local small nsys recapture, and the `--memory-profile` snapshot events are recorded. Part (f) top-5 / single-block totals / backward-gradient size are **not** done; S=2048 was not run; no new GPU rental this round; local small is a method demo, not official xl (f)
- [~] §3.1 saved tensors and §3.1.1 compile path (2026-11-17): `rmsnorm_test.py` hook comparison is recorded. The current script uses `torch.compile(..., backend="aot_eager")` and does call the compiled function on CUDA; AOT Autograd graph / saved-tensor changes are **not** Inductor/Triton fusion kernels, and hook output is not a measured CUDA peak. The conceptual checkpoint question was answered correctly (recompute the needed forward from the saved region input, then backward; not the whole model; parameters / gradients / optimizer remain). The later experiment is in §3.2
- [~] §3.2 activation checkpointing (afternoon of 2026-11-17): local small (`B=4`, `S=256`, `d_model=768`, `d_ff=3072`, 12 heads, FP32, CUDA) ordinary vs `use_reentrant=False` checkpoint is recorded from the user’s runs (hooks 189.47 vs 3 MiB; reported single-block peaks 202.72 vs 203.59 MiB, which do not prove a peak drop; reported four-layer whole-region peaks 545.73 vs 352.18 MiB). Current `four_blocks` reuses one `TransformerBlock` four times, not four independent modules. Compile was dropped after an NVTX fullgraph failure. Official (a) was discussed (no sketch submitted). Official (b) xl / `S=2048` / neighboring regions has **not** been run. Time cost was not measured. Peak-measurement hygiene is not claimed complete
- [ ] §4.1 has not started experimentally
- This round commits the week-02 journals, this progress section, and related A2 study scripts (`benchmark.py`, `rmsnorm_test.py`, `checkpoint_test.py`, `mixed_precision_accumulation.py`). The large memory-snapshot pickle is not included. §2.1.6 part (f) remains incomplete
