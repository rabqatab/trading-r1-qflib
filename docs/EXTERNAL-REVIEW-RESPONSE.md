# Response to External Review — Trading-R1 × qf-lib

**Date:** 2026-06-22 · **Branch:** `checkpoint/2026-06-21-llm-sft-data` (pushed to
`origin`) · **Tests:** 23 → **52 passing** · **Reviewer triage:**
[`TODO-EXTERNAL-REVIEW.md`](TODO-EXTERNAL-REVIEW.md)

Thank you for the triage — the recommended ordering (checkpoint → fundamentals →
multi-modal snapshot → parse guardrail → SFT-v0 eval) was exactly right and we
followed it end to end. Every item is done; this document is the detailed,
evidence-backed account, including one **important negative result** (SFT v0 is
degenerate) that we think is the most useful thing the harness produced.

---

## Executive summary

- All of **Priority 0 and Priority 1** are complete, plus **P2.1 (SFT-v0 eval)**.
- Work is on one branch, **10 grouped commits**, each a logical unit; **52 tests
  pass** (was 23). Every change was test-driven (RED → GREEN).
- **Headline:** the trained **SFT v0 collapsed to 100 % HOLD** (all-cash, CR 0 %)
  — it does *not* beat the prompt-only baseline. The comparison harness caught a
  trained model being *worse* than the untrained one, quantitatively. Root cause
  identified (full-sequence loss + class imbalance); the fix (**SFT v1**:
  completion-only loss + class balancing) is training now.
- Three delivered-data point-in-time defects were found and fixed (each
  unit-tested), so the multi-modal data is now leak-safe and join-ready.

---

## Per-item resolution

### P0 — checkpoint & hygiene ✅
Branched off `main`; sorted the working tree into **10 grouped commits** (LLM
tooling / SFT / PIT fixes / multi-modal / guardrail / docs). Session handoff
artifacts (`chat_files/`, `docs/context/`, `docs/file_notes/`, `tools/`) and
generated outputs (`compare_lab/output*/`, `compare_lab/sft/data/`) are
`.gitignore`-d — kept local, not committed. The frozen `qf-lib-harness` submodule
pointer is deliberately untouched. Validation: `uv run python -m pytest -q` → 52.

### P1.1 — normalize fundamentals revenue ✅  (commit `11c9c5d`)
`fundamentals.parquet` carried two XBRL revenue tags — `Revenues` (total) and
`RevenueFromContractWithCustomerExcludingAssessedTax` (ASC 606). 81 filing-periods
carried **both**. `compare_lab/fundamentals_pit.py` collapses them to one canonical
`Revenue` line per `(ticker, period_end, filing_date, fiscal_period)`, **preferring
`Revenues`** — empirically `Revenues ≥ contract` in **all 81** conflicts (it is the
superset: insurance investment income, energy other-income, etc.). Adds
`concept_normalized`; non-revenue concepts untouched; PIT on `filing_date`. Output
`fundamentals_pit.parquet` (2,156 → 2,075 rows; 312 Revenue rows). Tests:
`test_fundamentals_pit.py` (4).

### P1.2 — multi-modal PIT snapshot join ✅  (commit `0a6367f`)
`compare_lab/multimodal_context.py` (`MultiModalStore`) loads the `*_pit.parquet`
files and exposes per-`(ticker, as_of)` accessors that filter strictly on each
modality's own timestamp; `render_sections()` emits a compact text block.
`MarketSnapshotBuilder(ctx, multimodal=…)` appends it (opt-in; price-only stays the
default, so the existing LLM/SFT paths are unchanged). The one invariant —
**no row later than `as_of` may enter a snapshot** — is tested per modality
(`test_multimodal_context.py`, 8) plus the wiring (`test_snapshot.py`). ETFs
(SPY/QQQ) degrade gracefully (company sections render "none"). Timestamps used:
news `published_at`, fundamentals `filing_date`, analyst `gradedate`, insider
`start_date`, macro `release_date`.

### P1.3 — LLM parse-rate guardrail ✅  (commit `b81d6e2`)
`parse_decision_status()` returns `(decision, parsed)`; `LLMProvider` accumulates a
`parse_stats` dict (`total`, `no_tag`, `no_tag_rate`) and **warns past a
configurable 20 %** (the prompt-only run's real rate was 8.2 %); `run_comparison`
prints it. Fallback-to-HOLD stays explicit. (`LLMProvider` also gained an opt-in
`multimodal=` arg, wiring P1.2 into the prompt for a future multi-modal run.)
Tests: `test_providers_llm.py` (+2).

### P2.1 — serve & evaluate SFT v0 ✅  (commit `4ff9c0a`) — **negative result**
Served the SFT-v0 LoRA (`data/sft_adapter_v0/`) with vLLM `--enable-lora`. A
**cache-isolation gotcha** had to be fixed first: the response cache is keyed by
snapshot hash only (model-agnostic), so a second model would silently reuse the
base replies — added a `VLLM_CACHE_DIR` env override. Then probed SFT-v0 vs base on
a stride-sample of cached prompts (`tests/probe_sft_vs_base.py`):

| | base (prompt-only) | SFT v0 |
|---|---|---|
| 60-probe decision mix | StrongBuy 32 / Hold 17 / Sell 6 / Buy 3 / StrongSell 2 | **Hold 60** |

**SFT v0 emits HOLD for 100 % of inputs** → holds nothing → all-cash →
**CR 0 % / MDD 0 %**. The full 14-ticker comparison:

| Strategy (OOS 2024-01→2026-04, 14-ticker) | Cumulative | Sharpe | Max DD |
|---|---|---|---|
| Equal-weight | +143 % | 1.04 | 32.3 % |
| 12-1 Momentum | +52 % | 0.70 | 19.6 % |
| Prompt-only LLM (Qwen3-4B) | +36 % | 0.55 | 16.4 % |
| **SFT v0 (degenerate)** | **0 %** | n/a | **0 %** |

We did **not** spend ~4.5 h on the full SFT backtest once the 60-probe showed a
100 % collapse — an all-HOLD signal trivially backtests to a flat cash line, which
we confirmed directly with an all-zero weight matrix.

---

## Why SFT v0 collapsed (and how v1 fixes it)

The v0 LoRA reported a healthy-looking eval token-accuracy (~80 %), yet produced a
constant HOLD. The two are consistent once you see the cause:

1. **Full-sequence loss.** v0 trained on the *entire* chat sequence — a ~1–2 k-token
   price/technical prompt plus an ~80-token completion. The loss (and the 80 %
   accuracy) was dominated by the easy-to-predict prompt tokens; the gradient on
   the single decision token was negligible, so the model defaulted it to the
   majority class.
2. **Class imbalance.** The volatility labels are bullish-skewed (HOLD ≈ 37 %), so
   "always HOLD" is the lowest-risk constant prediction.

This is exactly the failure mode the paper avoids by using teacher-distilled,
evidence-grounded rationales and a GRPO **decision** reward rather than templated
text. Our **SFT v1** (training now) applies the two cheap structural fixes first:

- **Completion-only loss** (`assistant_only_loss` / prompt masking) — grade only
  the assistant turn. *Smoke evidence it bites:* loss rose 1.2 → 3.2 and token-acc
  fell 80 % → 61 %, i.e. the model is now scored on the decision, not the prompt.
- **Class balancing** (`build_dataset --balance`) — down-sample dominant classes
  (HOLD 37 % → ~24 %; cap 601/class, 2,541 examples) to remove the constant-HOLD
  prior.

If v1 still collapses, that is strong evidence the templated rationale itself is
insufficient and teacher distillation (Qwen3-32B) is required — which is the next
planned step regardless.

---

## Data-quality fixes (delivered store had 3 PIT defects)

The multi-modal pull (`data/qflib_data_store/`, [`DATA_STORE.md`](DATA_STORE.md))
carried real publish/filing timestamps, but three defects would have leaked or
degraded the backtest — all found, fixed, and unit-tested:

| Defect | Fix → output | Test |
|---|---|---|
| macro `release_date == date` (FRED lag missing → **leak**) | `macro_pit.py` → `macro_pit.parquet` (monthly → conservative day of M+1; daily → next business day; 0 leaks) | `test_macro_pit.py` |
| insider `transaction` empty for all 1,332 rows | `insider_pit.py` → `sentiment_insider_pit.parquet` (parse type from text; only 15 open-market Purchases → BUY) | `test_insider_pit.py` |
| fundamentals two revenue tags | `fundamentals_pit.py` (above) | `test_fundamentals_pit.py` |

Guiding rule throughout: in a backtest, a release date that is slightly **late** is
safe (conservative); slightly **early** is a leak. Each fix is biased to never be
early, and a unit test pins that property so a future change can't regress it.

---

## How to verify

```bash
git checkout checkpoint/2026-06-21-llm-sft-data
uv sync
uv run python -m pytest -q                 # 52 passed
# data fixes (need data/qflib_data_store/ present):
uv run python -m compare_lab.macro_pit
uv run python -m compare_lab.insider_pit
uv run python -m compare_lab.fundamentals_pit
# multi-modal snapshot (PIT-filtered) for one (ticker, date):
uv run python -c "import pandas as pd; from compare_lab.multimodal_context import MultiModalStore; print(MultiModalStore().render_sections('NVDA', pd.Timestamp('2024-06-28')))"
```

Full session narrative: [`PROGRESS-2026-06-21.md`](PROGRESS-2026-06-21.md);
comparison analysis: [`2026-06-21-three-way-comparison-memo.md`](2026-06-21-three-way-comparison-memo.md).

---

## Open items / next

1. **SFT v1** — finishing now; will re-probe (collapse fixed?) and backtest its row.
2. **P2.2** — re-run the prompt-only LLM with full multi-modal snapshots
   (`LLMProvider(multimodal=…)`), to see whether richer input moves the untrained
   signal before training.
3. **Teacher distillation → GRPO** — replace templated rationales with
   Qwen3-32B-distilled, evidence-grounded ones; add the structure/evidence/decision
   rewards. Gated on v1 + multi-modal being in place (your guidance, which we kept).

One open question for you: for the SFT decision target, do you prefer we keep the
**forward-return** volatility label (our choice, `shift(-τ)`; future info only in
the supervised answer, never in the model input) or also report a **trailing**
(momentum) variant for comparison? The labeler supports both behind a flag.
