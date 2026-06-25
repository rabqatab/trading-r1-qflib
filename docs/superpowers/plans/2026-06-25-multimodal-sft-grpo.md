# Multimodal SFT→GRPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrain the Trading-R1 student SFT→GRPO on **multimodal** snapshots (price+technical + news/fundamentals/sentiment/macro) on a leak-safe 2024-train / 2025-H1-eval split, and measure whether the extra modalities close the gap to the paper.

**Architecture:** Reuse the existing price-only pipeline (`snapshot.py` already supports `multimodal=`, `multimodal_context.py` does the PIT joins). Add `--multimodal` + window flags to the dataset builders and an eval-window switch to `run_comparison`. SFT uses the proven v1 templated recipe; GRPO carries a parse guardrail fix. Train on DGX Spark GB10 node 2 (mirror the existing `run_node2.sh` recipe), eval by serving the LoRA on node 1 + the parallel backtest.

**Tech Stack:** Python 3.11 (`uv run`), TRL 1.6 + PEFT (LoRA), vLLM (serve), qf-lib (backtest), pandas, DGX Spark GB10, sparkq.

## Global Constraints

- Run Python via `uv run python ...` (never `.venv/bin/python` directly).
- No Claude/AI attribution in commits; author stays `rabqatab <minhan.nick.cho@gmail.com>`.
- Commit to `main`, then `git merge --ff-only` into `feat/sft-distill-grpo` and push both (established session pattern).
- Leak rule: train ⊂ 2024-01-01..2024-12-31, eval ⊂ 2025-01-01..2025-06-30, no overlap.
- Universe = 12 equities (no SPY/QQQ): NVDA MSFT AAPL META AMZN TSLA BRK-B JPM LLY JNJ XOM CVX.
- GB10 training: NVIDIA PyTorch container, `NVIDIA_DISABLE_REQUIRE=1`, run `train.py` as a **script** not `-m`; node 2 launch via `run_node2.sh` + rsync (node 2 can't read alphabridge's NFS home).
- Per-model eval cache isolation: `VLLM_CACHE_DIR` is required (cache key is model-agnostic).

---

### Task 1: Multimodal snapshot smoke gate

Verify the multimodal integration actually renders on the real store for the new
window/universe **before** building anything on it. Catches the suspected
`fundamentals` column mismatch (code reads `filing_date`/`form`; the parquet may
only have `period_end`/`fiscal_period`) and confirms empty modalities degrade to
"none" instead of crashing.

**Files:**
- Test: `compare_lab/tests/test_multimodal_snapshot.py`
- Possibly modify: `compare_lab/multimodal_context.py` (only if a column mismatch is found)

**Interfaces:**
- Consumes: `MultiModalStore()` (from `compare_lab.multimodal_context`), `MarketSnapshotBuilder(ctx, multimodal=store)` (from `compare_lab.snapshot`).
- Produces: confidence that `builder.build(ticker, as_of)` returns a string containing the four modality section headers without raising.

- [ ] **Step 1: Write the failing test**

```python
# compare_lab/tests/test_multimodal_snapshot.py
import pandas as pd
from alpha_lab.core import load_context
from compare_lab.config import UNIVERSE_MM            # added in Task 2; until then hard-code ("NVDA",)
from compare_lab.multimodal_context import MultiModalStore
from compare_lab.run_comparison import _available_universe
from compare_lab.snapshot import MarketSnapshotBuilder

def test_multimodal_snapshot_renders_2024():
    uni = _available_universe(UNIVERSE_MM)
    ctx = load_context(universe=uni)
    b = MarketSnapshotBuilder(ctx, multimodal=MultiModalStore())
    snap = b.build("NVDA", pd.Timestamp("2024-06-03"))
    for header in ("=== NEWS", "=== FUNDAMENTALS", "=== SENTIMENT", "=== MACRO"):
        assert header in snap, f"missing {header}"
    assert "Indicators (latest)" in snap            # price+technical still present
```

- [ ] **Step 2: Run it, expect failure**

Run: `uv run python -m pytest compare_lab/tests/test_multimodal_snapshot.py -x -q`
Expected: FAIL — either `ImportError: UNIVERSE_MM` (do Task 2 first or hard-code the tuple) or a `KeyError`/`AttributeError` from `render_sections` (the real bug we're hunting).

- [ ] **Step 3: Fix `render_sections` if it raises**

If `fundamentals`/`render_sections` KeyErrors on `filing_date`/`form`, read the
actual parquet columns (`uv run python -c "import pandas as pd; print(pd.read_parquet('data/qflib_data_store/fundamentals_pit.parquet').columns.tolist())"`)
and map to what exists (e.g. `filing_date`→`period_end`, drop `form` or substitute
`fiscal_period`). Guard every modality loop so a missing column or empty frame
renders `  none` rather than raising.

- [ ] **Step 4: Run it, expect pass**

Run: `uv run python -m pytest compare_lab/tests/test_multimodal_snapshot.py -x -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add compare_lab/tests/test_multimodal_snapshot.py compare_lab/multimodal_context.py
git commit -m "test: multimodal snapshot renders on the 2024 store (smoke gate)"
```

---

### Task 2: Experiment config constants

**Files:**
- Modify: `compare_lab/config.py`
- Test: `compare_lab/tests/test_config.py`

**Interfaces:**
- Produces: `UNIVERSE_MM: tuple[str,...]` (12 equities), `MM_TRAIN_START`, `MM_TRAIN_END`, `MM_OOS_START`, `MM_OOS_END` (datetimes).

- [ ] **Step 1: Write the failing test**

```python
# append to compare_lab/tests/test_config.py
from datetime import datetime
from compare_lab.config import UNIVERSE_MM, MM_TRAIN_START, MM_TRAIN_END, MM_OOS_START, MM_OOS_END

def test_mm_constants():
    assert len(UNIVERSE_MM) == 12
    assert "SPY" not in UNIVERSE_MM and "QQQ" not in UNIVERSE_MM
    assert MM_TRAIN_START == datetime(2024, 1, 1) and MM_TRAIN_END == datetime(2024, 12, 31)
    assert MM_OOS_START == datetime(2025, 1, 1) and MM_OOS_END == datetime(2025, 7, 1)
    assert MM_TRAIN_END < MM_OOS_START          # leak-safe ordering
```

- [ ] **Step 2: Run, expect FAIL** (`ImportError`).
Run: `uv run python -m pytest compare_lab/tests/test_config.py::test_mm_constants -q`

- [ ] **Step 3: Add the constants to `compare_lab/config.py`**

```python
# --- multimodal gap-closing cycle (spec 2026-06-25) ---
UNIVERSE_MM: tuple[str, ...] = tuple(t for t in UNIVERSE if t not in ("SPY", "QQQ"))
MM_TRAIN_START = datetime(2024, 1, 1)
MM_TRAIN_END = datetime(2024, 12, 31)
MM_OOS_START = datetime(2025, 1, 1)
MM_OOS_END = datetime(2025, 7, 1)
```

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Commit**

```bash
git add compare_lab/config.py compare_lab/tests/test_config.py
git commit -m "feat: multimodal experiment window + 12-equity universe constants"
```

---

### Task 3: Decision-reward parse guardrail

Last cycle, 10 % of GRPO outputs echoed the template menu
`[[[STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL]]]` (the strict parser already rejects it,
scoring the −1.5 no-tag penalty — but −1.5 is *milder* than the worst wrong call
(−2.25), so RL had no strong reason to avoid it). Make an invalid/echoed decision
the **harshest** outcome so GRPO learns to always emit one valid class.

**Files:**
- Modify: `compare_lab/grpo/rewards.py`
- Test: `compare_lab/tests/test_grpo_rewards.py`

**Interfaces:**
- Consumes: `parse_last_decision`, `DECISION_MATRIX` (existing).
- Produces: `decision_reward(text, label, lam=1.0)` returning `INVALID_DECISION_PENALTY * lam` (= −2.5) when no single valid class is parsed.

- [ ] **Step 1: Write the failing test**

```python
# append to compare_lab/tests/test_grpo_rewards.py
from compare_lab.grpo.rewards import decision_reward, INVALID_DECISION_PENALTY

def test_template_echo_is_harshest():
    echo = "## CONCLUSION\n[[[STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL]]]"
    # echo must score worse than the worst *valid* wrong call (SB vs SS = -2.25)
    assert decision_reward(echo, "BUY") == INVALID_DECISION_PENALTY
    assert INVALID_DECISION_PENALTY < -2.25
    # a real (wrong) call still scores per the matrix, not the penalty
    assert decision_reward("[[[STRONG_BUY]]]", "STRONG_SELL") == -2.25
```

- [ ] **Step 2: Run, expect FAIL** (`ImportError: INVALID_DECISION_PENALTY`).
Run: `uv run python -m pytest compare_lab/tests/test_grpo_rewards.py::test_template_echo_is_harshest -q`

- [ ] **Step 3: Edit `decision_reward` in `compare_lab/grpo/rewards.py`**

```python
INVALID_DECISION_PENALTY = -2.5   # harsher than any valid wrong call (matrix min -2.25)

def decision_reward(text: str, label: str, lam: float = 1.0) -> float:
    d = parse_last_decision(text)
    if d is None or label not in DECISION_MATRIX:
        return INVALID_DECISION_PENALTY * lam
    return DECISION_MATRIX[d][label] * lam
```

- [ ] **Step 4: Run the full reward suite, expect PASS.**
Run: `uv run python -m pytest compare_lab/tests/test_grpo_rewards.py -q`

- [ ] **Step 5: Commit**

```bash
git add compare_lab/grpo/rewards.py compare_lab/tests/test_grpo_rewards.py
git commit -m "feat: harsher invalid-decision penalty (-2.5) kills the template-echo mode"
```

---

### Task 4: `--multimodal` + window flags in SFT dataset builder

**Files:**
- Modify: `compare_lab/sft/build_dataset.py`
- Test: `compare_lab/tests/test_sft_build_dataset.py`

**Interfaces:**
- Consumes: `UNIVERSE_MM`, `MM_TRAIN_START/END` (Task 2); `MultiModalStore` (Task 1).
- Produces: CLI `--multimodal`, `--universe-mm`, `--train-start`/`--train-end`; when `--multimodal`, snapshots include the four modality sections.

- [ ] **Step 1: Write the failing test** (builds 1 record, asserts the prompt carries a modality section)

```python
# compare_lab/tests/test_sft_build_dataset.py
import json, subprocess, sys, pathlib

def test_multimodal_flag_injects_sections(tmp_path):
    out = tmp_path / "data"
    subprocess.run([sys.executable, "-m", "compare_lab.sft.build_dataset",
                    "--out", str(out), "--multimodal", "--every", "60", "--limit", "5"],
                   check=True)
    rows = [json.loads(l) for l in (out / "train.jsonl").read_text().splitlines()]
    assert rows, "no records"
    assert any("=== NEWS" in r["messages"][0]["content"] for r in rows)
```

- [ ] **Step 2: Run, expect FAIL** (`--multimodal`/`--limit` unknown).
Run: `uv run python -m pytest compare_lab/tests/test_sft_build_dataset.py -q`

- [ ] **Step 3: Modify `compare_lab/sft/build_dataset.py`**

Add args and switch the builder + universe + window:

```python
# imports
from compare_lab.config import UNIVERSE, UNIVERSE_MM, MM_TRAIN_START, MM_TRAIN_END
from compare_lab.multimodal_context import MultiModalStore

# in main(), after parsing:
ap.add_argument("--multimodal", action="store_true")
ap.add_argument("--limit", type=int, default=0, help="cap records (smoke)")
ap.add_argument("--train-start"); ap.add_argument("--train-end")
# ...
universe = _available_universe(UNIVERSE_MM if args.multimodal else UNIVERSE)
ctx = load_context(universe=universe)
mm = MultiModalStore() if args.multimodal else None
builder = MarketSnapshotBuilder(ctx, multimodal=mm)
start = pd.Timestamp(args.train_start or (MM_TRAIN_START if args.multimodal else TRAIN_START))
end = pd.Timestamp(args.train_end or (MM_TRAIN_END if args.multimodal else TRAIN_END))
# replace the TRAIN_START/END window filter with `start`/`end`
# after appending a record, honour --limit:
#     if args.limit and len(records) >= args.limit: break  (break both loops)
```

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Commit**

```bash
git add compare_lab/sft/build_dataset.py compare_lab/tests/test_sft_build_dataset.py
git commit -m "feat: --multimodal + window flags in SFT dataset builder"
```

---

### Task 5: `--multimodal` + window flags in GRPO dataset builder

Mirror Task 4 in `compare_lab/grpo/build_dataset.py` (it stores `{"prompt":[...],"label":...}`).

**Files:**
- Modify: `compare_lab/grpo/build_dataset.py`
- Test: `compare_lab/tests/test_grpo_build_dataset.py`

**Interfaces:**
- Consumes: `UNIVERSE_MM`, `MM_TRAIN_START/END`, `MultiModalStore`, `_balanced_examples` (existing, takes `ctx, universe, n` — pass the MM universe; it reads `ctx.adj_close` so the MM store is only needed for the snapshot text).
- Produces: same CLI flags as Task 4; prompts carry modality sections.

- [ ] **Step 1: Write the failing test**

```python
# compare_lab/tests/test_grpo_build_dataset.py
import json, subprocess, sys

def test_grpo_multimodal_sections(tmp_path):
    out = tmp_path / "data"
    subprocess.run([sys.executable, "-m", "compare_lab.grpo.build_dataset",
                    "--out", str(out), "--multimodal", "--n", "10"], check=True)
    rows = [json.loads(l) for l in (out / "train.jsonl").read_text().splitlines()]
    assert any("=== NEWS" in r["prompt"][0]["content"] for r in rows)
    assert all(r["label"] in {"STRONG_SELL","SELL","HOLD","BUY","STRONG_BUY"} for r in rows)
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Modify `compare_lab/grpo/build_dataset.py`** — add `--multimodal`, use `UNIVERSE_MM`, `MultiModalStore()`, and the 2024 window inside `_balanced_examples` (it already filters `TRAIN_START..TRAIN_END` in `distill._balanced_examples`; add a window-override path or re-implement the triple selection over `MM_TRAIN_START..MM_TRAIN_END` locally).
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit**

```bash
git add compare_lab/grpo/build_dataset.py compare_lab/tests/test_grpo_build_dataset.py
git commit -m "feat: --multimodal + 2024 window in GRPO dataset builder"
```

---

### Task 6: Eval-window flags + multimodal switch in `run_comparison`

**Files:**
- Modify: `compare_lab/run_comparison.py`
- Test: `compare_lab/tests/test_run_comparison_args.py`

**Interfaces:**
- Consumes: `MM_OOS_START/END`, `UNIVERSE_MM`, `LLMProvider(VLLMClient(), multimodal=...)`, `MultiModalStore`.
- Produces: CLI `--oos-start`, `--oos-end`, `--multimodal`, `--universe-mm`; the OOS slice + rebal dates honour the passed window; the LLM provider gets a `MultiModalStore` when `--multimodal`.

- [ ] **Step 1: Write the failing test** (arg parsing + window override; no GPU)

```python
# compare_lab/tests/test_run_comparison_args.py
import pandas as pd
from compare_lab.run_comparison import _rebal_dates
def test_rebal_dates_respect_window():
    idx = pd.bdate_range("2024-06-01", "2025-12-31")
    d = _rebal_dates(idx, start="2025-01-01", end="2025-07-01")
    assert d.min() >= pd.Timestamp("2025-01-01") and d.max() < pd.Timestamp("2025-07-01")
```

- [ ] **Step 2: Run, expect FAIL** (`_rebal_dates` takes no `start/end`).
- [ ] **Step 3: Modify `run_comparison.py`** — give `_rebal_dates(idx, start=OOS_START, end=OOS_END)` optional params; add `--oos-start/--oos-end/--multimodal/--universe-mm`; pick `UNIVERSE_MM` and pass `LLMProvider(VLLMClient(), multimodal=MultiModalStore())` when `--multimodal`; filter the OOS returns slice by the passed window.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit**

```bash
git add compare_lab/run_comparison.py compare_lab/tests/test_run_comparison_args.py
git commit -m "feat: --oos-start/--oos-end + --multimodal in run_comparison"
```

---

### Task 7: SFT/GRPO train knobs (context length, temperature, epochs)

**Files:**
- Modify: `compare_lab/sft/train.py` (add `--max-length`, default 2048), `compare_lab/grpo/train.py` (add `--temperature` for generation, ensure `--epochs` default usable >1).

**Interfaces:**
- Produces: `sft/train.py --max-length 8192`; `grpo/train.py --temperature 1.0 --epochs 2`.

- [ ] **Step 1:** In `sft/train.py`, replace `MAX_SEQ = 2048` usage with `ap.add_argument("--max-length", type=int, default=2048)` and `max_length=args.max_length` in `SFTConfig`.
- [ ] **Step 2:** In `grpo/train.py`, add `ap.add_argument("--temperature", type=float, default=1.0)` and pass `temperature=args.temperature` into `GRPOConfig` (TRL 1.6 generation temperature).
- [ ] **Step 3:** Smoke-import both (no GPU): `uv run python -c "import ast; ast.parse(open('compare_lab/sft/train.py').read()); ast.parse(open('compare_lab/grpo/train.py').read()); print('ok')"`
- [ ] **Step 4: Commit**

```bash
git add compare_lab/sft/train.py compare_lab/grpo/train.py
git commit -m "feat: --max-length (SFT) + --temperature (GRPO) train knobs"
```

---

### Task 8: Build the multimodal datasets (node 1)

**Files:** outputs `compare_lab/sft/data_mm/`, `compare_lab/grpo/data_mm/` (gitignored under `data*/`).

- [ ] **Step 1: Build SFT data**

Run: `uv run python -m compare_lab.sft.build_dataset --multimodal --balance --out compare_lab/sft/data_mm`
Expected: prints `universe=12 total=... train=... val=...` and a label distribution across all 5 classes.

- [ ] **Step 2: Sanity-check token length** (multimodal prompts are long)

Run: `uv run python -c "import json; rows=[json.loads(l) for l in open('compare_lab/sft/data_mm/train.jsonl')]; import statistics as s; L=[len(r['messages'][0]['content']) for r in rows]; print('chars: mean',int(s.mean(L)),'max',max(L))"`
Expected: mean a few-k to ~30k chars. If max ≫ 32k chars (~8k tokens), note it — Task 9 truncates via `--max-length`.

- [ ] **Step 3: Build GRPO data**

Run: `uv run python -m compare_lab.grpo.build_dataset --multimodal --n 300 --out compare_lab/grpo/data_mm`
Expected: `built ... prompts ... 5-class balanced`.

- [ ] **Step 4: Commit** (a one-line note; the data dirs are gitignored)

```bash
git commit --allow-empty -m "chore: built multimodal SFT + GRPO datasets (data_mm/, gitignored)"
```

---

### Task 9: Train multimodal SFT v3 on GB10 (node 2)

Mirror the proven `run_node2.sh` recipe. SFT first (GRPO needs its adapter).

- [ ] **Step 1: rsync code + data + adapter dir target to node 2**

```bash
SSH='ssh -i ~/.ssh/id_ed25519'
rsync -az -e "$SSH" compare_lab/sft/train.py compare_lab/sft/data_mm nvidia@192.168.200.13:/home/nvidia/tr1_sft_mm/
```

- [ ] **Step 2: Smoke (3 steps) via sparkq** — reuse the SFT container recipe with `--smoke --max-length 8192`. Gate: exit 0, adapter saved. (See `compare_lab/sft/README.md` for the exact docker line; adapt paths to `/home/nvidia/tr1_sft_mm`.)
- [ ] **Step 3: Full run** — same command without `--smoke`, `--completion-only --balance`, `--out /work/out`. `sparkq wait` on it.
- [ ] **Step 4: Pull the adapter back**

```bash
rsync -az -e "$SSH" --exclude 'checkpoint-*' nvidia@192.168.200.13:/home/nvidia/tr1_sft_mm/out/ data/sft_adapter_mm_v3/
```

- [ ] **Step 5: Commit** (`git commit --allow-empty -m "chore: trained multimodal SFT v3 (data/sft_adapter_mm_v3)"`)

---

### Task 10: Train multimodal GRPO on GB10 (node 2)

- [ ] **Step 1: rsync** `compare_lab/grpo/{train.py,rewards.py,run_node2.sh,data_mm}` + `data/sft_adapter_mm_v3/` (as `adapter_v1`) to `nvidia@192.168.200.13:/home/nvidia/tr1_grpo_mm/`.
- [ ] **Step 2: Smoke** via sparkq with `--smoke --temperature 1.0`. Gate: exit 0; check the sparkq log for `rewards/reward_decision` variance and whether `reward_structure/evidence` are now non-zero (multimodal outputs may be richer).
- [ ] **Step 3: Full run** — `bash /work/run_node2.sh --temperature 1.0 --epochs 2` (W&B off unless the key was rotated; reward metrics print to the sparkq log). `sparkq wait`.
- [ ] **Step 4: Pull adapter** → `data/sft_adapter_mm_grpo/`.
- [ ] **Step 5: Commit** (`--allow-empty` note).

---

### Task 11: Eval — ablation + trained models (2025-H1 window)

Serve each LoRA on node 1 (mirror the `tr1-vllm-grpo` serve job), backtest with the
2025-H1 window + per-model cache. All four LLM rows run on the **same** window/universe.

- [ ] **Step 1: Prompt-only multimodal OFF** (price-only control)

Run: `VLLM_MODEL=Qwen/Qwen3-4B VLLM_CACHE_DIR=compare_lab/.cache_mm_off uv run python -m compare_lab.run_comparison --llm --oos-start 2025-01-01 --oos-end 2025-07-01 --universe-mm --out compare_lab/output_mm_off`
(serve the base model — no adapter — or reuse a base serve job)

- [ ] **Step 2: Prompt-only multimodal ON** (criterion 1 ablation)

Run: same as Step 1 + `--multimodal`, `VLLM_CACHE_DIR=compare_lab/.cache_mm_on`, `--out compare_lab/output_mm_on`.

- [ ] **Step 3: Multimodal SFT v3** — serve `data/sft_adapter_mm_v3` as `sft-mm-v3`; backtest `--multimodal --universe-mm` window 2025-H1, `VLLM_CACHE_DIR=.cache_mm_sft`, `--out output_mm_sft`.
- [ ] **Step 4: Multimodal SFT→GRPO** — serve `data/sft_adapter_mm_grpo` as `mm-grpo`; same backtest, `.cache_mm_grpo`, `--out output_mm_grpo`.
- [ ] **Step 5: Collect** the four `comparison.csv` + NO_TAG parse rates into one table. Cancel the serve job to free GPU.

---

### Task 12: Document results + verdict

**Files:** `docs/2026-06-21-three-way-comparison-memo.md` (new multimodal section), `README.md` (#1 row + Roadmap), `docs/PROGRESS-2026-06-21.md`.

- [ ] **Step 1:** Add a "Multimodal cycle (2025-H1)" section to the memo: the ablation table (OFF vs ON) answering criterion 1, the trained-model rows answering criterion 2, NO_TAG (did the guardrail kill the echo?), and an honest verdict vs the price-only lineage and the paper trend.
- [ ] **Step 2:** Update README #1 row + Roadmap and PROGRESS with the outcome (numbers if positive, honest negative if not).
- [ ] **Step 3: Commit + push** (main, ff-only feat, push both).

---

## Self-Review

**Spec coverage:** goal/criteria → Tasks 11–12 (ablation + trained rows + parse). Leak-safe split → Task 2 constants + Tasks 8–11 windows. Universe 12 → Task 2. SFT multimodal v1-recipe → Tasks 4, 9. Context 8k → Task 7, 8(step 2), 9. GRPO + parse guardrail → Tasks 3, 5, 10. Eval/ablation → Task 11. Risks (column mismatch, graceful empty) → Task 1. All spec sections map to a task.

**Placeholders:** Tasks 1–3, 7 carry exact code; Tasks 4–6 give the precise edit + test; Tasks 8–12 are operational (commands + gates), code-free by nature (training/eval/doc). No "TBD"/"handle edge cases".

**Type consistency:** `UNIVERSE_MM`, `MM_TRAIN_START/END`, `MM_OOS_START/END` (Task 2) used verbatim in 4/5/6/8/11. `INVALID_DECISION_PENALTY` (Task 3) used in its own test. `MultiModalStore()` / `MarketSnapshotBuilder(ctx, multimodal=)` consistent across 1/4/5/6. `_rebal_dates(idx, start, end)` signature defined in Task 6 and used there.
