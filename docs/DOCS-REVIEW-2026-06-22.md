# Documentation Review — 2026-06-22

> Purpose: reconcile the project documentation after the external-review response and the subsequent local checks.

## What was reviewed

- `docs/PROGRESS-2026-06-21.md`
- `docs/EXTERNAL-REVIEW-RESPONSE.md`
- `docs/TODO-EXTERNAL-REVIEW.md`
- `docs/DATA_STORE.md`
- `docs/2026-06-21-three-way-comparison-memo.md`
- `docs/DATA_QC_RUBRIC.md`
- `README.md`
- Current git state and test result
- `sparkq` status for the SFT v1 job

## Verified current state

- Current branch: `main`.
- Test suite: `52 passed`.
- `qf-lib-harness` submodule remains modified/dirty from the parent repo view; this appears to be the only current git-status item outside documentation edits.
- SFT v1 job exists and is running as `sparkq` job `73c3` / tag `tr1-sft-v1`.
- No local `data/sft_adapter_v1/` artifact has been verified yet.

## Documentation gaps found and fixed

### 1. Multi-modal snapshot status was stale in the comparison memo

**Problem:** `docs/2026-06-21-three-way-comparison-memo.md` still said the multi-modal data had landed but was “not yet joined into the snapshot.”

**Fix:** Updated the caveat to distinguish:

- PIT-safe multi-modal snapshot join is implemented.
- The reported LLM metrics are still price+technical-only because the prompt-only LLM has not yet been re-run with `multimodal=` enabled.

### 2. P2.2 status in `TODO-EXTERNAL-REVIEW.md` was stale

**Problem:** The TODO doc described P2.2 as feasible only after snapshot joining, implying the join was still not done.

**Fix:** Updated P2.2 to say it is feasible now and still open; remaining work is running the prompt-only LLM with `multimodal=` enabled and comparing against the price+technical baseline.

### 3. SFT v1 wording overclaimed completion proximity

**Problem:** `EXTERNAL-REVIEW-RESPONSE.md` used phrases like “training now” / “finishing now,” which are operationally time-sensitive and can become stale.

**Fix:** Reworded to:

- SFT v1 training job has been launched.
- Adapter materialization, re-probe, and backtest remain required.

### 4. Progress log lacked current SFT v1 operational handle

**Problem:** `PROGRESS-2026-06-21.md` described SFT v1 as the next fix but did not record the current job handle.

**Fix:** Added the latest checked operational status:

- `sparkq` job `73c3`
- tag `tr1-sft-v1`
- no verified local adapter artifact yet

## Open project/documentation items after this pass

1. **SFT v1 completion follow-up**
   - Check `sparkq status 73c3` until terminal.
   - If successful, document output location and copy/verify adapter under `data/sft_adapter_v1/` or the chosen canonical path.
   - Re-probe against base and SFT v0.
   - Backtest if probe shows non-degenerate class distribution.

2. **P2.2 multi-modal prompt-only run**
   - Run prompt-only Qwen3-4B with `LLMProvider(multimodal=...)`.
   - Compare against price+technical-only 12/14-ticker baselines.
   - Document parse-rate, CR, Sharpe, MDD, and correlation changes.

3. **Teacher distillation / GRPO planning**
   - Keep gated on SFT v1 and/or multi-modal prompt-only findings.
   - Do not start GRPO until the decision-token collapse is addressed.

4. **Submodule cleanliness**
   - Inspect `qf-lib-harness` dirty state before declaring the repo fully clean.
   - Decide whether the submodule dirtiness is intentional, ignored, or needs a reset/commit inside the submodule.

## Bottom line

The external-review response is broadly accurate, but the documentation needed four precision updates: multi-modal join status, P2.2 feasibility, SFT v1 operational status, and the current SFT v1 job handle. Those updates have now been made.
