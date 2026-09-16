# Changelog

All changes we make to the assignment code or PDF will be documented in this file.

## [26.0.3] 2026-04-07
- handout: Adjust leaderboard time limit from 90 to 45 minutes

## [26.0.2] 2026-04-06
- handout: Fix SGD example bug
- code: Add CLAUDE.md

## [26.0.1] 2026-04-05
- handout: Fix a few typos, broken links
- code: Fix some references to H100s in favor of B200
- code: Fixture for AdamW updated to cover bugfix from 26.0.0 (should not have caused an issue previously regardless)

## [26.0.0] 2026-03-30
- handout: Latex -> Typst (!)
- handout: Fix tens of typos
- handout: Assume B200s
- handout: Clarify some aspects of tokenization
- handout: Modernize architecture for accounting
- handout: Fix AdamW disparity to paper (order of weight-decay vs gradient update)
- handout: Revert column/row major changes
- code: Many tests fixed, made more specific
- code: Tensor typing revised is adapters and tests



## [1.0.6] 2025-08-28
- handout: Fix bug in RoPE formulation
- code: Fix typing for adapters
- code: Relock dependencies
- code: Minor reformatting


## [1.0.5] 2025-04-15
- code: Add submission script, fix typos
- code: Use `uv_build` for the package build system
- handout: Fix RoPE indexing
- code: Fix test for truncated LM inputs
- code: Ruff reformatting and linting
- code: Simplify snapshot testing
- code: Make everything compatible with `ty` typing

## [1.0.4] - 2025-04-08
### Added
- handout: add guidance on parallelizing pretokenization and provide starter code for chunking
- handout: add guidance on removing special tokens before pretokenization (you should split on them!)

### Changed
- handout: fix command for compiling model on MPS backend

## [1.0.3] - 2025-04-07
### Added
- code: Test for removing special tokens when training BPE

### Changed
- handout: Fixed RoPE off-by-one error
- code: Fix for Intel Macs, support 3.11 and PyTorch 2.2.2 when on MacOS x86_64

## [1.0.2] - 2025-04-03
### Added
- code: Missing tests for Linear and Embedding

### Changed
- handout: Fix RMSNorm interface
- handout: Add hints in RMSNorm and SwiGLU specifications to improve numerical stability
- handout: Clarify list of model hyperparameters listed in `adamwAccounting`.

## [1.0.1] - 2025-04-02
### Changed
- handout: Fix expected number of non-embedding parameters for model with recommended TinyStories hyperparameters (section 7.2).
- handout: replace `<|endofsequence|>` with `<|endoftext|>` in the `decoding` problem.
- code: fix `test_train_bpe` tiebreaking bug.

## [1.0.0] - 2025-04-01
Initial release.
