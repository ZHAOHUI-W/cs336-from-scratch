# CS336 from Scratch — Architecture & Engineering Lab

Self-study of [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/), optimized for **architecture mindset** and **engineering design ability**, not just “tests pass.”

**Constraints**
- Notes-first: [cs336_notes.pdf](https://qihongruan.github.io/cs336/cs336_notes.pdf) (no lecture videos)
- Finish assignments in **~6 weeks**
- Train / benchmark on **cloud GPU**; local RTX 3060 only for light checks

**Primary goals**
1. Architecture mindset — name trade-offs (memory vs compute, interfaces, failure modes) before coding.
2. Engineering design — modules with clear contracts, incremental tests, and reversible decisions.
3. Honest progress — weekly journals + design decision records (ADRs) in this repo.

## How this repo tracks progress

| Path | Purpose |
| --- | --- |
| [`PROGRESS.md`](PROGRESS.md) | Checklist for the 6-week plan (update every Sunday) |
| [`docs/design/`](docs/design/) | Architecture Decision Records — *why* a design, not only *what* |
| [`journals/`](journals/) | Weekly reflection: bugs, trade-offs, what you’d redesign |
| [`assignments/`](assignments/) | Your implementation work for A1–A5 (keep private) |

### Definition of done (every component)
A piece is “done” only when **all** of these exist:
1. **Interface** — inputs/outputs/invariants written in the ADR or module docstring
2. **Test** — at least one unit or property check that would catch the last bug you hit
3. **Design note** — short ADR: options considered, choice, cost
4. **Progress tick** — `PROGRESS.md` updated

## 6-week plan (compressed)

| Week | Notes | Build | Architecture focus |
| --- | --- | --- | --- |
| 1 | Lec 01–02 | A1: BPE + AdamW | Data structures for BPE; optimizer state layout (~16 B/param) |
| 2 | Lec 03–04 | A1: Transformer + train | Block contract (pre-norm, RoPE, SwiGLU); single-batch overfit |
| 3 | Lec 05–08 | A2: profile → Triton FA2 → FSDP | Memory hierarchy; kernel tiling; sharding boundaries |
| 4 | Lec 09–12 | A3: IsoFLOP + fit | Experiment design; $C \approx 6ND$; measurement validity |
| 5 | Lec 13–14 | A4: extract → filter → MinHash/LSH | Streaming pipelines; approximate algorithms at scale |
| 6 | Lec 15–17 | A5: SFT → DPO/GRPO | Policy vs reference; loss as a design object |

**Cloud GPU rule:** correctness on CPU first; cloud only for real trains and A2 benchmarks.

## Honor / self-study note
This repo is **private** by default. Treat assignment implementations as your own work product. Use tutors/LLMs for concepts and debugging strategy, not as a solution paste.

## Weekly rhythm
1. Read linked notes (Key Concepts + assignment sections)
2. Write a stub ADR *before* coding the hard part
3. Implement + test on CPU
4. Cloud run if needed
5. Journal + update `PROGRESS.md`
