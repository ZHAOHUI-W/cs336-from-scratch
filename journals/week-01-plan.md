# Week 1 day-by-day — A1 foundations

**Goal:** BPE + AdamW solid enough to start the Transformer next week, with ADRs (architecture muscle), not only green-ish tests.

**Daily rhythm (every day)**
1. Notes pass (Key Concepts first)
2. One design decision written (ADR stub ok)
3. One implement + test block
4. 5-line journal note (bug or trade-off)

**Setup (once, Day 0 or Day 1 morning)**
```sh
git clone https://github.com/ZHAOHUI-W/cs336-from-scratch.git
cd cs336-from-scratch/assignments/a1-basics
chmod +x scripts/fetch_large_assets.sh && ./scripts/fetch_large_assets.sh
uv sync
uv run pytest   # expect NotImplementedError — good
```

Mark in root `PROGRESS.md` as you go.

---

## Day 1 — Map the assignment + loss intuition
- [ ] Skim A1 handout PDF (from fetch) — list deliverables only (tokenizer, model, AdamW, train)
- [ ] Skim `tests/adapters.py` — list the `run_*` entry points (what you must expose)
- [ ] Notes: Lec 01 TL;DR + “what is a tokenizer” (encode/decode, vocab)
- [ ] Recap next-token CE: loss = `-log p(correct)`; uniform baseline `log V`
- [ ] Journal: what “interface” means for `adapters.py` in one sentence

**Done when:** you can explain aloud what A1 builds end-to-end without looking.

## Day 2 — AdamW design (do optimizer before the hard BPE train)
- [ ] Notes: Lec 02 — parameters / grads / optimizer state memory (~16 bytes/param with AdamW fp32)
- [ ] **ADR `001-adamw-state-layout.md`** before coding: moments `m,v`, bias correction, *decoupled* weight decay vs L2-in-loss
- [ ] Implement AdamW (`torch.optim.Optimizer` subclass)
- [ ] Toy check: one step matches `torch.optim.AdamW` on a tiny tensor (write a tiny script or use course tests once wired)
- [ ] Wire the AdamW-related adapter(s); run the optimizer tests you can

**Done when:** ADR accepted + optimizer test(s) not stuck on your `step()`.

## Day 3 — BPE encode/decode (given merges)
- [ ] Notes: Lec 01 BPE section — bytes → merges → tokens; GPT-2 pretokenize idea
- [ ] **ADR `002-bpe-encode-decode.md`**: deterministic merge order (earliest merge / leftmost); special tokens
- [ ] Implement encode/decode assuming vocab + merges are given
- [ ] Round-trip tests on fixtures / course tokenizer tests you can reach
- [ ] Journal: one silent bug you almost shipped (mask, order, bytes vs str)

**Done when:** string → ids → string round-trips on sample fixtures.

## Day 4 — BPE train (the systems/design day)
- [ ] Re-read naive vs efficient pair counting (why rescan-every-merge dies)
- [ ] **ADR `003-bpe-pair-index.md`**: data structure for pair counts + updates after a merge (options → pick → cost)
- [ ] Implement `train` on a tiny corpus first; hand-trace 3 merges
- [ ] Scale to course train-BPE tests / sample corpus
- [ ] CPU only — no cloud needed

**Done when:** train-BPE tests pass or you’re blocked only on a named edge case written in the journal.

## Day 5 — Wire adapters + resource napkin math
- [ ] Finish wiring tokenizer + optimizer adapters; `uv run pytest` on those modules
- [ ] Notes: Lec 02 FLOPs — remember `≈ 6 N D` training FLOPs rule of thumb
- [ ] Napkin: for your intended TinyStories run, rough peak memory (params+grads+opt+activations)
- [ ] Short ADR or journal addendum: what you’ll run on **cloud** vs **CPU** next week
- [ ] Update `PROGRESS.md` Week 1 boxes

**Done when:** pytest failures are only in model/train pieces you haven’t built yet.

## Day 6 — Buffer / deepen (pick based on gaps)
- [ ] If BPE train still weak → only that
- [ ] If AdamW numerical mismatch → bias correction / decay order
- [ ] Optional: skim Lec 03 headings only (pre-norm, RMSNorm, SwiGLU, RoPE) — **no full implement yet**
- [ ] Fill `journals/week-01.md` (takeaway + next week’s risk)

**Done when:** Week 1 journal has a one-sentence architecture takeaway.

## Day 7 — Review + commit progress
- [ ] Re-read your 3 ADRs; tighten “Consequences” sections
- [ ] Commit/push ADRs + journal + `PROGRESS.md` ticks (no huge data)
- [ ] Write Week 2 preview: Transformer block contract (pre-norm, causal mask, RoPE site)
- [ ] Rest or light cloud account setup (Modal/etc.) — no long train yet

**Done when:** GitHub shows ADRs + journal; you know Monday’s first coding target.

---

## Architecture bar (this week)
A piece counts as done only if: **interface** + **test** + **ADR** + **PROGRESS tick**.

## Explicit non-goals this week
- Full Transformer / attention
- Cloud training runs (setup ok; long train = Week 2)
- MoE / systems / Triton
