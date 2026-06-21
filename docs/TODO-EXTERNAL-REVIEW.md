# TODO — External Review Triage

> Source: external review from the QF-Lib-LLM Telegram discussion.  
> Workspace: `/home/alphabridge/Study/tradingR1_qflib`  
> Status checked against current repo state on 2026-06-21.  
> Baseline health: `36 passed`; repo has many uncommitted/untracked files.

> **RESOLUTION (2026-06-22):** P0, P1.1, P1.2, P1.3, P2.1 all **done** and merged
> to `main` (grouped commits; tests **52 passed**). Data QC tooling from PR #1
> (`validate_data.py`, [`DATA_QC_RUBRIC.md`](DATA_QC_RUBRIC.md)) reconciled in —
> all hard gates pass, weighted 98.6.
> Headline: **SFT v0 evaluated → degenerate (100% HOLD → all-cash, CR 0%)** — a
> clean negative result; next is SFT v1 (balanced data + completion-only loss +
> distillation) and/or P2.2 (multi-modal prompt-only run). **Detailed reviewer
> response:** [`EXTERNAL-REVIEW-RESPONSE.md`](EXTERNAL-REVIEW-RESPONSE.md).
> Per-item details below and in [`PROGRESS-2026-06-21.md`](PROGRESS-2026-06-21.md).

## Current baseline

### Completed / landed

- [x] Prompt-only Qwen3-4B LLM row added to qf-lib comparison.
- [x] 12-ticker comparison results produced and documented.
- [x] 14-ticker SPY/QQQ robustness comparison produced in `compare_lab/output_14/`.
- [x] Multi-modal data store received under `data/qflib_data_store/`.
- [x] Macro PIT leak fixed via `compare_lab/macro_pit.py` → `macro_pit.parquet`.
- [x] Insider transaction type recovered via `compare_lab/insider_pit.py` → `sentiment_insider_pit.parquet`.
- [x] SFT v0 trained for Qwen3-4B LoRA; adapter stored in `data/sft_adapter_v0/`.
- [x] Documentation updated: `docs/DATA_STORE.md`, `docs/PROGRESS-2026-06-21.md`, `docs/2026-06-21-three-way-comparison-memo.md`.
- [x] Tests pass: `36 passed`.

### Current repository caution

The repo is **not clean**. Before large follow-on work, create a coherent checkpoint commit or at least inspect and group changes carefully.

Current notable status:

```text
M README.md
M docs/DATA_REQUIREMENTS.md
m qf-lib-harness
?? chat_files/
?? compare_lab/analyze_results.py
?? compare_lab/build_memo_report.py
?? compare_lab/insider_pit.py
?? compare_lab/labeling.py
?? compare_lab/macro_pit.py
?? compare_lab/output_14/
?? compare_lab/sft/
?? compare_lab/tests/smoke_llm_endpoint.py
?? compare_lab/tests/test_insider_pit.py
?? compare_lab/tests/test_labeling.py
?? compare_lab/tests/test_macro_pit.py
?? compare_lab/tests/verify_cache_provenance.py
?? docs/2026-06-21-three-way-comparison-memo.md
?? docs/DATA_STORE.md
?? docs/PROGRESS-2026-06-21.md
?? docs/context/
?? docs/file_notes/
?? tools/
```

---

## Priority 0 — checkpoint and hygiene

### P0.1 Create a clean checkpoint commit

**Why:** The current directory contains many completed deliverables plus generated/chat artifacts. New implementation work should not mix with this large uncommitted state.

**Actions:**

- [ ] Review `git diff` for tracked changes.
- [ ] Review untracked files and separate code/docs from generated handoff artifacts.
- [ ] Decide whether to commit `chat_files/`, `docs/context/`, and `docs/file_notes/` or keep them local only.
- [ ] Commit reproducible code/docs/tests first.
- [ ] Avoid committing large or private data files unless the repo policy explicitly allows it.

**Likely commit groups:**

1. `feat: add prompt-only llm comparison and reporting`
2. `feat: add sft v0 data and training pipeline`
3. `fix: add point-in-time macro and insider data repairs`
4. `docs: record progress and data-store status`

**Validation:**

```bash
cd /home/alphabridge/Study/tradingR1_qflib
.venv/bin/python -m pytest -q
```

Expected: `36 passed`.

---

## Priority 1 — feasible implementation tasks

### P1.1 Normalize fundamentals revenue XBRL tags

**Status:** Feasible now.  
**Reason:** `data/qflib_data_store/fundamentals.parquet` exists and `docs/DATA_STORE.md` identifies the issue.

**Problem:** The fundamentals file has two revenue concepts:

- `Revenues`
- `RevenueFromContractWithCustomerExcludingAssessedTax`

The pipeline needs one normalized revenue line per ticker/filing period.

**Suggested files:**

- Create: `compare_lab/fundamentals_pit.py`
- Create: `compare_lab/tests/test_fundamentals_pit.py`
- Output data: `data/qflib_data_store/fundamentals_pit.parquet`
- Update docs: `docs/DATA_STORE.md`, `docs/DATA_REQUIREMENTS.md`, `docs/PROGRESS-2026-06-21.md`

**Acceptance criteria:**

- [ ] A canonical `concept_normalized` or equivalent field exists.
- [ ] Revenue rows collapse to a single normalized concept per ticker/period/filing where possible.
- [ ] Non-revenue concepts remain unchanged.
- [ ] PIT semantics use `filing_date`, never `period_end`.
- [ ] Unit tests cover both revenue tag variants and conflict handling.

**Validation:**

```bash
.venv/bin/python -m pytest compare_lab/tests/test_fundamentals_pit.py -q
.venv/bin/python -m pytest -q
```

---

### P1.2 Extend `snapshot.py` to join all modalities by PIT timestamp

**Status:** Feasible after P1.1.  
**Reason:** All required data files are present; macro and insider PIT defects have already been fixed.

**Current limitation:** `compare_lab/snapshot.py` builds price + technical snapshots only.

**Required joins per `(ticker, as_of)`:**

- [ ] News: `published_at <= as_of`; bucket into paper windows `t−3..t`, `t−10..t−4`, `t−30..t−11`.
- [ ] Fundamentals: latest rows with `filing_date <= as_of`; use normalized revenue file from P1.1.
- [ ] Analyst sentiment: `gradedate <= as_of`.
- [ ] Insider sentiment: `start_date <= as_of`; prefer `sentiment_insider_pit.parquet`.
- [ ] Macro: `release_date <= as_of`; **must use `macro_pit.parquet`, not raw `macro.parquet`**.

**Suggested files:**

- Modify: `compare_lab/snapshot.py`
- Modify or create tests under: `compare_lab/tests/test_snapshot.py`
- Possibly create helper module: `compare_lab/multimodal_context.py`

**Acceptance criteria:**

- [ ] Snapshot output includes deterministic multi-modal sections.
- [ ] No future-dated rows can enter a snapshot.
- [ ] Missing modalities degrade gracefully, especially for SPY/QQQ where fundamentals/news/insider/analyst are expected to be absent.
- [ ] Tests explicitly prove PIT filtering for each modality.

**Validation:**

```bash
.venv/bin/python -m pytest compare_lab/tests/test_snapshot.py -q
.venv/bin/python -m pytest -q
```

---

### P1.3 Add parse-rate guardrail for LLM responses

**Status:** Feasible now.  
**Reason:** Existing prompt-only result has an 8.2% `NO_TAG` rate, documented as a quality leak.

**Likely files:**

- Modify: `compare_lab/llm_client.py`
- Modify: `compare_lab/providers/llm.py`
- Add/modify tests: `compare_lab/tests/test_llm_client.py`, `compare_lab/tests/test_providers_llm.py`

**Acceptance criteria:**

- [ ] LLM response parsing produces an explicit parse status.
- [ ] `NO_TAG` / invalid class rate is counted and surfaced in outputs or logs.
- [ ] Optional threshold can fail or warn a run if parse quality is too poor.
- [ ] Existing behavior of fallback-to-HOLD remains explicit and documented.

**Validation:**

```bash
.venv/bin/python -m pytest compare_lab/tests/test_llm_client.py compare_lab/tests/test_providers_llm.py -q
.venv/bin/python -m pytest -q
```

---

## Priority 2 — evaluation tasks

### P2.1 Serve SFT v0 adapter and run comparison

**Status:** Feasible, but operationally heavier.  
**Reason:** `data/sft_adapter_v0/` exists and SFT training completed successfully. It still needs vLLM+LoRA serving and a new qf-lib comparison row.

**Inputs:**

- Base model: `Qwen3-4B-Instruct-2507` / served as `Qwen/Qwen3-4B`.
- Adapter: `data/sft_adapter_v0/`.

**Actions:**

- [ ] Start vLLM with LoRA adapter enabled.
- [ ] Smoke-test one or more snapshot prompts.
- [ ] Run `compare_lab.run_comparison --llm` against the LoRA endpoint.
- [ ] Compare SFT-v0 row against prompt-only 12-ticker and/or 14-ticker rows.
- [ ] Update memo and progress docs.

**Acceptance criteria:**

- [ ] SFT-v0 emits parseable `[[[CLASS]]]` outputs.
- [ ] Backtest completes and writes comparison artifacts.
- [ ] Result is documented honestly, especially if SFT worsens any metric.

**Validation:**

```bash
.venv/bin/python -m pytest -q
# plus a live endpoint smoke test before the full run
```

---

### P2.2 Re-run prompt-only LLM with full multi-modal snapshots

**Status:** Feasible after P1.2.  
**Reason:** The full multi-modal data exists but is not yet joined into snapshots.

**Goal:** Determine whether adding news/fundamentals/sentiment/macro changes the prompt-only LLM signal quality before training.

**Acceptance criteria:**

- [ ] Multi-modal prompt-only run completes.
- [ ] Metrics are compared against price+technical-only prompt baseline.
- [ ] Parse-rate and drawdown/correlation are documented.

---

## Priority 3 — later research tasks

### P3.1 Teacher distillation

**Status:** Not first.  
**Dependencies:** P1.2 and preferably P2.1/P2.2.

**Goal:** Replace templated SFT rationales with teacher-generated rationales, likely Qwen3-32B or comparable model.

**Risks:** Cost, latency, teacher quality, evidence grounding.

---

### P3.2 GRPO

**Status:** Later.  
**Dependencies:** Multi-modal snapshots, stronger SFT data, reliable reward definitions.

**Reward components:**

- Decision reward from deterministic volatility label.
- Evidence/grounding reward once quotable multi-modal evidence is included.
- Format reward for parseable final `[[[CLASS]]]`.

---

## External-review conclusion

The external review is broadly correct: the proposed tasks are possible in this directory. The safest order is:

1. ✅ **Checkpoint current completed work.** — branch + 8 grouped commits.
2. ✅ **Normalize fundamentals revenue.** — `fundamentals_pit.py` → `fundamentals_pit.parquet`.
3. ✅ **Implement multi-modal PIT snapshot join.** — `multimodal_context.py`, opt-in in `snapshot.py`.
4. ✅ **Add parse-rate guardrail.** — `parse_decision_status` + `LLMProvider.parse_stats`.
5. ✅ **Evaluate SFT v0 adapter.** — **degenerate (all-HOLD)**; documented in the memo.
6. 🔜 **SFT v1** (fix the v0 collapse), then teacher distillation / GRPO.

Do not start GRPO or teacher distillation before multi-modal snapshot integration and SFT-v0 evaluation are complete. *(Both now complete.)*
