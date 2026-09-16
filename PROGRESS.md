# Progress tracker

Update every Sunday. Mark: `[ ]` todo · `[~]` in progress · `[x]` done

## Goals
- [ ] Architecture mindset: can explain trade-offs without notes
- [ ] Engineering design: modules have contracts + tests + ADRs
- [ ] Complete A1–A5 in 6 weeks

## Week 1 — Tokenization, resources, A1 foundations
- [x] A1 starter at `assignments/a1-basics/` (run `scripts/fetch_large_assets.sh` before pytest)
- [ ] Read Lec 01 (tokenization / BPE)
- [ ] Read Lec 02 (memory, FLOPs, training loop primitives)
- [ ] ADR: BPE data structures (pair counts / updates)
- [ ] BPE encode/decode round-trip
- [ ] ADR: AdamW state + decoupled weight decay
- [ ] AdamW matches `torch.optim.AdamW` on a toy step
- [ ] Journal `journals/week-01.md`

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
- [ ] Fit \(N_\mathrm{opt}(C)\), \(D_\mathrm{opt}(C)\)
- [ ] Predict target budget
- [ ] Journal week-04

## Week 5 — Data (A4)
- [ ] Read Lec 13–14
- [ ] ADR: streaming pipeline stages + failure modes
- [ ] WARC → text on sample
- [ ] Filters + language ID
- [ ] MinHash + LSH (not \(O(N^2)\))
- [ ] Journal week-05

## Week 6 — Alignment (A5)
- [ ] Read Lec 15–17
- [ ] ADR: SFT vs preference objective; β meaning
- [ ] SFT beats base on math eval
- [ ] DPO/GRPO: init loss sanity check
- [ ] Final eval + write-up
- [ ] Journal week-06
- [ ] Retro: top 5 design lessons in `docs/design/999-retro.md`
