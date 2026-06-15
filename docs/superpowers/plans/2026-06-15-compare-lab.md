# compare_lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a common qf-lib evaluation substrate where equal-weight, momentum (#3), and a prompt-only open-source LLM (#2) each emit a target-weight matrix, get backtested under identical look-ahead-free rules, and are compared on CR/SR/HR/MDD.

**Architecture:** New `compare_lab/` package in the project root repo. It reuses the `qf-lib-harness` submodule's `alpha_lab.core.load_context(universe=...)` data loaders (read-only) and copies the qf-lib backtest bridge pattern from `alpha_lab/pipeline.py` (the harness core is FROZEN — never import its internals as a dependency, copy the small bridge). All signal sources implement one `SignalProvider.weights(...)` interface returning `DataFrame[date × ticker]`.

**Tech Stack:** Python 3.11, uv, qf-lib (pinned fork @9ba5a0f), pandas, numpy, stockstats, plotly, openai client, pytest.

**Commit policy:** Plain commit messages. NEVER add a Claude co-authorship / "Generated with Claude Code" trailer (project rule).

---

## File Structure

All paths relative to repo root `/home/alphabridge/tradingR1_qflib/`.

| File | Responsibility |
|---|---|
| `pyproject.toml` | Root uv project: deps (qf-lib pin, stockstats, plotly, openai, pytest) |
| `compare_lab/__init__.py` | Package marker; imports `_paths` for submodule bootstrap |
| `compare_lab/_paths.py` | Insert `qf-lib-harness/` into `sys.path` so `import alpha_lab` resolves |
| `compare_lab/config.py` | Universe, OOS dates, rebal freq, rf, MAX_POSITIONS, cache dir |
| `compare_lab/metrics.py` | `cumulative_return`, `sharpe_ratio`, `hit_rate`, `max_drawdown` (pure) |
| `compare_lab/snapshot.py` | `MarketSnapshotBuilder` — per-ticker, as_of, price+technical → text |
| `compare_lab/providers/base.py` | `SignalProvider` ABC + `normalize_weights` helper |
| `compare_lab/providers/equal_weight.py` | `EqualWeightProvider` |
| `compare_lab/providers/momentum.py` | `MomentumProvider` (12-1, top-N) |
| `compare_lab/providers/llm.py` | `LLMProvider` (snapshot→client→5-class→weight) |
| `compare_lab/llm_client.py` | `VLLMClient` — OpenAI-compatible + disk cache + injectable transport |
| `compare_lab/backtest.py` | `run_backtest(weights, ctx)` → daily simple returns (qf-lib) |
| `compare_lab/report.py` | `build_report(results)` → comparison table + plotly HTML |
| `compare_lab/run_comparison.py` | CLI orchestrator |
| `compare_lab/tests/` | unit + integration tests |

---

## Task 1: Root uv project + package skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `compare_lab/__init__.py`
- Create: `compare_lab/_paths.py`
- Create: `compare_lab/tests/__init__.py`
- Test: `compare_lab/tests/test_paths.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "tradingr1-compare-lab"
version = "0.1.0"
description = "Common qf-lib evaluation substrate comparing factor and LLM trading signals."
requires-python = ">=3.11,<3.13"
dependencies = [
    "qf-lib",
    "pandas>=2.0",
    "numpy>=1.26",
    "pyarrow>=15.0.0",
    "stockstats>=0.6.2",
    "plotly>=6.0.0",
    "openai>=1.40.0",
]

[dependency-groups]
dev = ["pytest>=8.0"]

[tool.uv.sources]
qf-lib = { git = "https://github.com/ico1036/qf-lib.git", rev = "9ba5a0f5dcc3c3f06f8488f2b8cc7fe12afa15d8" }

[tool.uv]
override-dependencies = ["WeasyPrint>=60.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["compare_lab"]

[tool.pytest.ini_options]
testpaths = ["compare_lab/tests"]
```

- [ ] **Step 2: Write `compare_lab/_paths.py`**

```python
"""Bootstrap: make the qf-lib-harness submodule's `alpha_lab` importable.

compare_lab lives in the project root repo; alpha_lab lives in the
qf-lib-harness submodule. We reuse alpha_lab.core's data loaders (read-only)
without vendoring or installing the harness as a package.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HARNESS = _REPO_ROOT / "qf-lib-harness"


def ensure_harness_on_path() -> Path:
    """Insert the submodule root on sys.path so `import alpha_lab` works.
    Returns the harness path. Idempotent."""
    p = str(_HARNESS)
    if p not in sys.path:
        sys.path.insert(0, p)
    return _HARNESS
```

- [ ] **Step 3: Write `compare_lab/__init__.py`**

```python
"""compare_lab — common qf-lib evaluation substrate."""
from compare_lab._paths import ensure_harness_on_path

ensure_harness_on_path()
```

- [ ] **Step 4: Write `compare_lab/tests/__init__.py`** (empty file)

```python
```

- [ ] **Step 5: Write the failing test `compare_lab/tests/test_paths.py`**

```python
import compare_lab  # noqa: F401  (triggers path bootstrap)


def test_alpha_lab_importable():
    import alpha_lab.core as core
    assert hasattr(core, "load_context")
    assert hasattr(core, "select_universe")


def test_prices_path_points_at_submodule_data():
    import alpha_lab.core as core
    assert core.PRICES_PATH.name == "prices.parquet"
    assert "qf-lib-harness" in str(core.PRICES_PATH)
```

- [ ] **Step 6: Sync env and run test**

Run: `uv sync && uv run pytest compare_lab/tests/test_paths.py -v`
Expected: 2 passed. (First `uv sync` builds qf-lib from the git pin — may take a few minutes.)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock compare_lab/
git commit -m "compare_lab: scaffold root uv project + alpha_lab path bootstrap"
```

---

## Task 2: config.py

**Files:**
- Create: `compare_lab/config.py`
- Test: `compare_lab/tests/test_config.py`

- [ ] **Step 1: Write the failing test `compare_lab/tests/test_config.py`**

```python
from datetime import datetime

import compare_lab  # noqa: F401
from compare_lab import config


def test_universe_has_paper_tickers_and_etfs():
    for t in ["NVDA", "AAPL", "BRK-B", "SPY", "QQQ"]:
        assert t in config.UNIVERSE
    assert len(config.UNIVERSE) == 14


def test_oos_window_ordered():
    assert config.OOS_START < config.OOS_END
    assert config.OOS_START == datetime(2024, 1, 2)


def test_constants():
    assert config.RF_ANNUAL == 0.04
    assert config.REBAL_FREQ == "W-FRI"
    assert config.MAX_POSITIONS >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest compare_lab/tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'compare_lab.config'`

- [ ] **Step 3: Write `compare_lab/config.py`**

```python
"""Locked evaluation constants for the comparison substrate (spec §5)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

# Paper Table S3 universe (12 equities + 2 ETFs). BRK.B -> BRK-B (Yahoo style).
UNIVERSE: tuple[str, ...] = (
    "NVDA", "MSFT", "AAPL", "META", "AMZN", "TSLA",
    "BRK-B", "JPM", "LLY", "JNJ", "XOM", "CVX",
    "SPY", "QQQ",
)

# Out-of-sample window. No training here, so the whole span is OOS.
OOS_START = datetime(2024, 1, 2)
OOS_END = datetime(2026, 4, 1)

# Paper's short holdout slice, reported alongside the full window.
PAPER_SLICE_START = datetime(2024, 6, 1)
PAPER_SLICE_END = datetime(2024, 9, 1)

REBAL_FREQ = "W-FRI"      # weekly rebalance anchor
RF_ANNUAL = 0.04          # risk-free rate, paper §7.1 (US10Y)
ANNUALIZATION = 252.0
TOP_N_MOMENTUM = 5        # held names for the momentum baseline
MAX_POSITIONS = 8         # LLM position budget → size = 1/MAX_POSITIONS

CACHE_DIR = Path(__file__).resolve().parent / ".cache"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest compare_lab/tests/test_config.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add compare_lab/config.py compare_lab/tests/test_config.py
git commit -m "compare_lab: add evaluation config constants"
```

---

## Task 3: metrics.py (pure functions, TDD)

**Files:**
- Create: `compare_lab/metrics.py`
- Test: `compare_lab/tests/test_metrics.py`

- [ ] **Step 1: Write the failing test `compare_lab/tests/test_metrics.py`**

```python
import numpy as np
import pandas as pd

import compare_lab  # noqa: F401
from compare_lab import metrics


def _series(vals):
    idx = pd.date_range("2024-01-01", periods=len(vals), freq="D")
    return pd.Series(vals, index=idx)


def test_cumulative_return():
    r = _series([0.1, 0.0, -0.05])
    # (1.1)(1.0)(0.95) - 1 = 0.045
    assert metrics.cumulative_return(r) == pytest_approx(0.045)


def test_max_drawdown():
    # equity: 1.1, 1.1, 1.045 ; peak 1.1 ; dd = 1 - 1.045/1.1 = 0.05
    r = _series([0.1, 0.0, -0.05])
    assert metrics.max_drawdown(r) == pytest_approx(0.05)


def test_hit_rate():
    r = _series([0.01, -0.02, 0.03, 0.0])
    # positive periods: 2 of 4 -> 0.5  (0.0 is not > 0)
    assert metrics.hit_rate(r) == pytest_approx(0.5)


def test_sharpe_needs_30_obs():
    r = _series([0.001] * 10)
    assert np.isnan(metrics.sharpe_ratio(r))


def test_sharpe_value():
    rng = np.random.default_rng(0)
    r = _series(rng.normal(0.001, 0.01, 252))
    excess = r - metrics.RF_DAILY
    expected = excess.mean() / excess.std(ddof=0) * np.sqrt(252)
    assert metrics.sharpe_ratio(r) == pytest_approx(expected)


# tiny local approx helper to avoid extra imports
def pytest_approx(x, tol=1e-9):
    class _A:
        def __eq__(self, other):
            return abs(other - x) <= tol
    return _A()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest compare_lab/tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'compare_lab.metrics'`

- [ ] **Step 3: Write `compare_lab/metrics.py`**

```python
"""Performance metrics on a daily simple-return series (spec §4.5, paper §7.1).

All functions are pure (numpy/pandas only) and model-independent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from compare_lab.config import ANNUALIZATION, RF_ANNUAL

RF_DAILY = RF_ANNUAL / ANNUALIZATION
_MIN_OBS = 30


def cumulative_return(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.empty:
        return float("nan")
    return float((1.0 + r).prod() - 1.0)


def sharpe_ratio(returns: pd.Series, rf_daily: float = RF_DAILY) -> float:
    r = returns.dropna()
    if len(r) < _MIN_OBS:
        return float("nan")
    excess = r - rf_daily
    s = excess.std(ddof=0)
    if not np.isfinite(s) or s < 1e-12:
        return float("nan")
    return float(excess.mean() / s * np.sqrt(ANNUALIZATION))


def hit_rate(returns: pd.Series) -> float:
    """Fraction of positive-return periods (long-only proxy for paper HR)."""
    r = returns.dropna()
    if r.empty:
        return float("nan")
    return float((r > 0).mean())


def max_drawdown(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.empty:
        return float("nan")
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    dd = 1.0 - equity / peak
    return float(dd.max())


def all_metrics(returns: pd.Series) -> dict[str, float]:
    return {
        "CR": cumulative_return(returns),
        "SR": sharpe_ratio(returns),
        "HR": hit_rate(returns),
        "MDD": max_drawdown(returns),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest compare_lab/tests/test_metrics.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add compare_lab/metrics.py compare_lab/tests/test_metrics.py
git commit -m "compare_lab: add CR/SR/HR/MDD metrics"
```

---

## Task 4: snapshot.py — MarketSnapshotBuilder

**Files:**
- Create: `compare_lab/snapshot.py`
- Test: `compare_lab/tests/test_snapshot.py`

- [ ] **Step 1: Write the failing test `compare_lab/tests/test_snapshot.py`**

```python
import numpy as np
import pandas as pd

import compare_lab  # noqa: F401
from compare_lab.snapshot import MarketSnapshotBuilder


def _ctx_like():
    """Minimal fake ctx: 400 business days for one ticker 'AAA'."""
    idx = pd.bdate_range("2023-01-02", periods=400)
    base = pd.Series(np.linspace(100, 200, 400), index=idx)
    df = pd.DataFrame({"AAA": base})
    vol = pd.DataFrame({"AAA": np.full(400, 1_000_000.0)}, index=idx)

    class Ctx:
        adj_close = df
        open = df * 0.99
        high = df * 1.01
        low = df * 0.98
        volume = vol
        dollar_volume = df * vol
        universe = ("AAA",)
    return Ctx()


def test_snapshot_excludes_future_bars():
    ctx = _ctx_like()
    as_of = pd.Timestamp("2024-01-15")
    b = MarketSnapshotBuilder(ctx)
    text = b.build("AAA", as_of)
    # No date strictly after as_of may appear in the serialized window.
    future = ctx.adj_close.index[ctx.adj_close.index > as_of]
    for d in future[:5]:
        assert d.strftime("%Y-%m-%d") not in text


def test_snapshot_is_deterministic_and_hashable():
    ctx = _ctx_like()
    as_of = pd.Timestamp("2024-01-15")
    b = MarketSnapshotBuilder(ctx)
    t1 = b.build("AAA", as_of)
    t2 = b.build("AAA", as_of)
    assert t1 == t2
    assert len(b.snapshot_hash("AAA", as_of)) == 12


def test_snapshot_mentions_ticker_and_indicators():
    ctx = _ctx_like()
    text = MarketSnapshotBuilder(ctx).build("AAA", pd.Timestamp("2024-01-15"))
    assert "AAA" in text
    assert "rsi" in text.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest compare_lab/tests/test_snapshot.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'compare_lab.snapshot'`

- [ ] **Step 3: Write `compare_lab/snapshot.py`**

```python
"""MarketSnapshotBuilder — per-ticker, as-of, price + technical indicators.

Strictly causal: bars with date > as_of_date are physically removed before
indicators are computed. Output is a compact text block for an LLM prompt.
MVP modalities = price + technical only (spec §1.3, §4.1).
"""
from __future__ import annotations

from hashlib import sha1

import pandas as pd
from stockstats import StockDataFrame

# Technical indicators (subset of paper Table S2) computed via stockstats.
_INDICATORS = [
    "close_50_sma", "close_200_sma", "close_10_ema",
    "macd", "macds", "macdh",
    "rsi_14", "kdjk", "cci", "roc",
    "atr", "boll", "boll_ub", "boll_lb",
    "dx", "mfi",
]
_OUTPUT_WINDOW = 15   # trailing bars shown in the snapshot
_LOOKBACK_BARS = 520  # ~2y of bars fed to indicators (for 200 SMA etc.)


def _abbrev(x: float) -> str:
    if pd.isna(x):
        return "na"
    ax = abs(x)
    if ax >= 1e9:
        return f"{x/1e9:.2f}b"
    if ax >= 1e6:
        return f"{x/1e6:.2f}m"
    if ax >= 1e3:
        return f"{x/1e3:.2f}k"
    return f"{x:.2f}"


class MarketSnapshotBuilder:
    def __init__(self, ctx):
        self._ctx = ctx

    def _window(self, ticker: str, as_of) -> pd.DataFrame:
        as_of = pd.Timestamp(as_of)
        c = self._ctx
        cols = {
            "open": c.open[ticker], "high": c.high[ticker],
            "low": c.low[ticker], "close": c.adj_close[ticker],
            "volume": c.volume[ticker],
        }
        df = pd.DataFrame(cols).dropna()
        df = df[df.index <= as_of]          # strict causality
        return df.tail(_LOOKBACK_BARS)

    def build(self, ticker: str, as_of) -> str:
        df = self._window(ticker, as_of)
        if df.empty:
            return f"Ticker {ticker} as of {pd.Timestamp(as_of).date()}: no data."
        sdf = StockDataFrame.retype(df.copy())
        for ind in _INDICATORS:
            _ = sdf[ind]                    # triggers computation
        recent = sdf.tail(_OUTPUT_WINDOW)
        lines = [
            f"Ticker: {ticker}",
            f"As of: {pd.Timestamp(as_of).date()} "
            f"(showing last {len(recent)} trading days)",
            "",
            "Date | Open High Low Close Volume",
        ]
        for d, row in recent.iterrows():
            lines.append(
                f"{d.strftime('%Y-%m-%d')} | "
                f"{_abbrev(row['open'])} {_abbrev(row['high'])} "
                f"{_abbrev(row['low'])} {_abbrev(row['close'])} "
                f"{_abbrev(row['volume'])}"
            )
        last = recent.iloc[-1]
        lines.append("")
        lines.append("Indicators (latest):")
        for ind in _INDICATORS:
            lines.append(f"  {ind}: {_abbrev(last.get(ind, float('nan')))}")
        return "\n".join(lines)

    def snapshot_hash(self, ticker: str, as_of) -> str:
        text = self.build(ticker, as_of)
        return sha1(text.encode()).hexdigest()[:12]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest compare_lab/tests/test_snapshot.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add compare_lab/snapshot.py compare_lab/tests/test_snapshot.py
git commit -m "compare_lab: add MarketSnapshotBuilder (price+technical, as-of safe)"
```

---

## Task 5: providers/base.py + equal_weight.py

**Files:**
- Create: `compare_lab/providers/__init__.py`
- Create: `compare_lab/providers/base.py`
- Create: `compare_lab/providers/equal_weight.py`
- Test: `compare_lab/tests/test_providers_equal.py`

- [ ] **Step 1: Write `compare_lab/providers/__init__.py`** (empty file)

```python
```

- [ ] **Step 2: Write the failing test `compare_lab/tests/test_providers_equal.py`**

```python
import pandas as pd

import compare_lab  # noqa: F401
from compare_lab.providers.base import normalize_weights
from compare_lab.providers.equal_weight import EqualWeightProvider


def _ctx():
    idx = pd.bdate_range("2024-01-01", periods=20)
    px = pd.DataFrame({"A": range(20), "B": range(20), "C": range(20)},
                      index=idx, dtype=float) + 1.0

    class Ctx:
        adj_close = px
        universe = ("A", "B", "C")
    return Ctx()


def test_normalize_rows_sum_to_one():
    df = pd.DataFrame({"A": [1.0, 0.0], "B": [1.0, 0.0], "C": [2.0, 0.0]})
    out = normalize_weights(df)
    assert out.iloc[0].sum() == 1.0
    assert out.iloc[1].sum() == 0.0  # all-zero row stays all-zero (cash)


def test_equal_weight_provider():
    ctx = _ctx()
    rebal = pd.bdate_range("2024-01-01", periods=20)[::5]
    w = EqualWeightProvider().weights(ctx, rebal)
    assert list(w.columns) == ["A", "B", "C"]
    assert (abs(w.sum(axis=1) - 1.0) < 1e-9).all()
    assert (abs(w["A"] - 1 / 3) < 1e-9).all()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest compare_lab/tests/test_providers_equal.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'compare_lab.providers.base'`

- [ ] **Step 4: Write `compare_lab/providers/base.py`**

```python
"""SignalProvider contract: produce a target-weight matrix (spec §4.2)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


def normalize_weights(weights: pd.DataFrame) -> pd.DataFrame:
    """Each row scaled to sum 1; all-zero rows stay zero (full cash)."""
    w = weights.clip(lower=0.0).fillna(0.0)
    row_sums = w.sum(axis=1)
    safe = row_sums.replace(0.0, 1.0)
    return w.div(safe, axis=0)


class SignalProvider(ABC):
    name: str = "base"

    @abstractmethod
    def weights(self, ctx, rebal_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Return DataFrame[index=rebal_dates, columns=ctx.universe], rows >=0
        summing to 1 (or 0 for full cash)."""
        raise NotImplementedError
```

- [ ] **Step 5: Write `compare_lab/providers/equal_weight.py`**

```python
"""EqualWeightProvider — hold every universe name equally (market baseline)."""
from __future__ import annotations

import pandas as pd

from compare_lab.providers.base import SignalProvider, normalize_weights


class EqualWeightProvider(SignalProvider):
    name = "equal_weight"

    def weights(self, ctx, rebal_dates: pd.DatetimeIndex) -> pd.DataFrame:
        cols = list(ctx.universe)
        w = pd.DataFrame(1.0, index=rebal_dates, columns=cols)
        return normalize_weights(w)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest compare_lab/tests/test_providers_equal.py -v`
Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add compare_lab/providers/ compare_lab/tests/test_providers_equal.py
git commit -m "compare_lab: add SignalProvider base + EqualWeightProvider"
```

---

## Task 6: providers/momentum.py

**Files:**
- Create: `compare_lab/providers/momentum.py`
- Test: `compare_lab/tests/test_providers_momentum.py`

- [ ] **Step 1: Write the failing test `compare_lab/tests/test_providers_momentum.py`**

```python
import numpy as np
import pandas as pd

import compare_lab  # noqa: F401
from compare_lab.providers.momentum import MomentumProvider


def _ctx_trend():
    """5 tickers; A>B>C>D>E by trailing return so ranking is deterministic."""
    idx = pd.bdate_range("2023-01-02", periods=400)
    px = {}
    for i, t in enumerate(["A", "B", "C", "D", "E"]):
        slope = (5 - i) * 0.5
        px[t] = pd.Series(100 + slope * np.arange(400), index=idx)
    df = pd.DataFrame(px)

    class Ctx:
        adj_close = df
        universe = ("A", "B", "C", "D", "E")
    return Ctx()


def test_momentum_picks_top_n_equal_weight():
    ctx = _ctx_trend()
    rebal = ctx.adj_close.index[300::20]
    w = MomentumProvider(top_n=2).weights(ctx, rebal)
    last = w.iloc[-1]
    held = last[last > 0].index.tolist()
    assert held == ["A", "B"]              # strongest two
    assert abs(last["A"] - 0.5) < 1e-9
    assert abs(last.sum() - 1.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest compare_lab/tests/test_providers_momentum.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'compare_lab.providers.momentum'`

- [ ] **Step 3: Write `compare_lab/providers/momentum.py`**

```python
"""MomentumProvider — 12-1 momentum, top-N equal weight (#3 as weights)."""
from __future__ import annotations

import pandas as pd

from compare_lab.config import TOP_N_MOMENTUM
from compare_lab.providers.base import SignalProvider, normalize_weights

_LOOKBACK = 252
_SKIP = 21


class MomentumProvider(SignalProvider):
    name = "momentum_12_1"

    def __init__(self, top_n: int = TOP_N_MOMENTUM):
        self.top_n = top_n

    def weights(self, ctx, rebal_dates: pd.DatetimeIndex) -> pd.DataFrame:
        px = ctx.adj_close[list(ctx.universe)]
        score = px.pct_change(_LOOKBACK).shift(_SKIP)        # PIT-safe 12-1
        # value as of the most recent bar strictly before each rebal date
        score = score.reindex(score.index.union(rebal_dates)).ffill()
        score = score.loc[rebal_dates]
        ranks = score.rank(axis=1, ascending=False, method="first")
        membership = (ranks <= self.top_n) & score.notna()
        w = membership.astype(float)
        return normalize_weights(w)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest compare_lab/tests/test_providers_momentum.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add compare_lab/providers/momentum.py compare_lab/tests/test_providers_momentum.py
git commit -m "compare_lab: add MomentumProvider (12-1 top-N)"
```

---

## Task 7: llm_client.py (cache + injectable transport)

**Files:**
- Create: `compare_lab/llm_client.py`
- Test: `compare_lab/tests/test_llm_client.py`

- [ ] **Step 1: Write the failing test `compare_lab/tests/test_llm_client.py`**

```python
import compare_lab  # noqa: F401
from compare_lab.llm_client import VLLMClient


def test_client_caches_by_key(tmp_path):
    calls = {"n": 0}

    def fake_transport(prompt: str) -> str:
        calls["n"] += 1
        return f"reply::{prompt}"

    c = VLLMClient(transport=fake_transport, cache_dir=tmp_path)
    r1 = c.complete("hello", key="k1")
    r2 = c.complete("hello", key="k1")   # served from cache
    assert r1 == r2 == "reply::hello"
    assert calls["n"] == 1               # transport called once only


def test_client_distinct_keys(tmp_path):
    def fake_transport(prompt: str) -> str:
        return prompt.upper()

    c = VLLMClient(transport=fake_transport, cache_dir=tmp_path)
    assert c.complete("a", key="ka") == "A"
    assert c.complete("b", key="kb") == "B"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest compare_lab/tests/test_llm_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'compare_lab.llm_client'`

- [ ] **Step 3: Write `compare_lab/llm_client.py`**

```python
"""VLLMClient — OpenAI-compatible chat client with a disk cache.

`transport` is injectable so tests run without a server. The default transport
calls a vLLM OpenAI-compatible endpoint (DGX Spark, single-node,
--enforce-eager BF16; served via sparkq). Cache key = caller-supplied snapshot
hash, so identical inputs always return identical outputs (reproducibility).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from compare_lab.config import CACHE_DIR


def _default_transport(base_url: str, model: str) -> Callable[[str], str]:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="EMPTY")

    def _call(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2048,
        )
        return resp.choices[0].message.content or ""

    return _call


class VLLMClient:
    def __init__(
        self,
        transport: Callable[[str], str] | None = None,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen3-4B",
        cache_dir: Path = CACHE_DIR,
    ):
        self._transport = transport or _default_transport(base_url, model)
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def complete(self, prompt: str, key: str) -> str:
        p = self._path(key)
        if p.exists():
            return json.loads(p.read_text())["response"]
        response = self._transport(prompt)
        p.write_text(json.dumps({"prompt": prompt, "response": response}))
        return response
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest compare_lab/tests/test_llm_client.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add compare_lab/llm_client.py compare_lab/tests/test_llm_client.py
git commit -m "compare_lab: add VLLMClient with disk cache + injectable transport"
```

---

## Task 8: providers/llm.py (5-class parse + mapping)

**Files:**
- Create: `compare_lab/providers/llm.py`
- Test: `compare_lab/tests/test_providers_llm.py`

- [ ] **Step 1: Write the failing test `compare_lab/tests/test_providers_llm.py`**

```python
import numpy as np
import pandas as pd

import compare_lab  # noqa: F401
from compare_lab.llm_client import VLLMClient
from compare_lab.providers.llm import LLMProvider, parse_decision


def test_parse_decision_variants():
    assert parse_decision("blah\n[[[STRONG_BUY]]]") == "STRONG_BUY"
    assert parse_decision("text [[[buy]]] more") == "BUY"
    assert parse_decision("no decision here") == "HOLD"   # default


def _ctx():
    idx = pd.bdate_range("2023-01-02", periods=400)
    px = pd.DataFrame(
        {t: np.linspace(100, 150, 400) for t in ("A", "B")}, index=idx)
    vol = pd.DataFrame({t: np.full(400, 1e6) for t in ("A", "B")}, index=idx)

    class Ctx:
        adj_close = px
        open = px * 0.99
        high = px * 1.01
        low = px * 0.98
        volume = vol
        dollar_volume = px * vol
        universe = ("A", "B")
    return Ctx()


def test_llm_provider_maps_classes_to_weights(tmp_path):
    # A -> BUY (held), B -> SELL (flat)
    def fake_transport(prompt: str) -> str:
        return "[[[BUY]]]" if "Ticker: A" in prompt else "[[[SELL]]]"

    client = VLLMClient(transport=fake_transport, cache_dir=tmp_path)
    ctx = _ctx()
    rebal = ctx.adj_close.index[300::20]
    w = LLMProvider(client, max_positions=8).weights(ctx, rebal)
    last = w.iloc[-1]
    assert last["A"] > 0
    assert last["B"] == 0.0
    assert abs(last["A"] - 1 / 8) < 1e-9   # size = 1/max_positions
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest compare_lab/tests/test_providers_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'compare_lab.providers.llm'`

- [ ] **Step 3: Write `compare_lab/providers/llm.py`**

```python
"""LLMProvider (#2) — prompt-only LLM signal mapped to target weights.

For each (rebal date, ticker): build a snapshot, ask the LLM for a 5-class
decision, map {STRONG_BUY, BUY} -> held (else flat). MVP is long-only with a
fixed position budget (size = 1/max_positions); held > budget is capped by
rank-order of class strength. (spec §4.2, §4.3)
"""
from __future__ import annotations

import re

import pandas as pd

from compare_lab.config import MAX_POSITIONS
from compare_lab.llm_client import VLLMClient
from compare_lab.providers.base import SignalProvider
from compare_lab.snapshot import MarketSnapshotBuilder

_CLASSES = ("STRONG_SELL", "SELL", "HOLD", "BUY", "STRONG_BUY")
_HELD = {"BUY", "STRONG_BUY"}
_STRENGTH = {"STRONG_BUY": 2, "BUY": 1}
_PATTERN = re.compile(r"\[\[\[\s*(STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL)\s*\]\]\]",
                      re.IGNORECASE)

_PROMPT_HEADER = (
    "You are a disciplined equity analyst. Based only on the price and "
    "technical data below, decide a 5-class trading signal for the next week. "
    "End your reply with exactly one line: [[[STRONG_BUY|BUY|HOLD|SELL|"
    "STRONG_SELL]]].\n\n"
)


def parse_decision(text: str) -> str:
    matches = _PATTERN.findall(text or "")
    if not matches:
        return "HOLD"
    return matches[-1].upper()


class LLMProvider(SignalProvider):
    name = "llm_prompt_only"

    def __init__(self, client: VLLMClient, max_positions: int = MAX_POSITIONS):
        self._client = client
        self._builder = None
        self.max_positions = max_positions

    def weights(self, ctx, rebal_dates: pd.DatetimeIndex) -> pd.DataFrame:
        self._builder = MarketSnapshotBuilder(ctx)
        cols = list(ctx.universe)
        w = pd.DataFrame(0.0, index=rebal_dates, columns=cols)
        size = 1.0 / self.max_positions
        for d in rebal_dates:
            strengths: dict[str, int] = {}
            for t in cols:
                snap = self._builder.build(t, d)
                key = self._builder.snapshot_hash(t, d)
                reply = self._client.complete(_PROMPT_HEADER + snap, key=key)
                decision = parse_decision(reply)
                if decision in _HELD:
                    strengths[t] = _STRENGTH[decision]
            # cap to budget by class strength (ties: stable order)
            held = sorted(strengths, key=lambda t: (-strengths[t], cols.index(t)))
            for t in held[: self.max_positions]:
                w.at[d, t] = size
        return w
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest compare_lab/tests/test_providers_llm.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add compare_lab/providers/llm.py compare_lab/tests/test_providers_llm.py
git commit -m "compare_lab: add LLMProvider (5-class parse + weight mapping)"
```

---

## Task 9: backtest.py (qf-lib bridge)

**Files:**
- Create: `compare_lab/backtest.py`
- Test: `compare_lab/tests/test_backtest.py`

This copies the bridge pattern from `qf-lib-harness/alpha_lab/pipeline.py`
(`_run_qf_backtest`, `_build_alpha_model`) — the harness is FROZEN, so we copy
rather than import. A weight matrix becomes a boolean membership matrix
(weight > 0 → LONG) and qf-lib sizes each LONG at `1/max_positions`.

- [ ] **Step 1: Write the failing test `compare_lab/tests/test_backtest.py`**

```python
import pandas as pd

import compare_lab  # noqa: F401
from compare_lab.backtest import weights_to_membership


def test_weights_to_membership():
    w = pd.DataFrame(
        {"A": [0.5, 0.0], "B": [0.5, 0.0], "C": [0.0, 0.0]},
        index=pd.to_datetime(["2024-01-05", "2024-01-12"]),
    )
    m = weights_to_membership(w)
    assert m.loc["2024-01-05", "A"]
    assert not m.loc["2024-01-05", "C"]
    assert not m.loc["2024-01-12"].any()   # all cash
```

> The full qf-lib backtest is exercised by the integration smoke in Task 10
> (it needs the real data provider + prices.parquet). This unit test covers the
> pure membership conversion.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest compare_lab/tests/test_backtest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'compare_lab.backtest'`

- [ ] **Step 3: Write `compare_lab/backtest.py`**

```python
"""qf-lib backtest bridge for compare_lab.

Adapted (copied, not imported) from qf-lib-harness/alpha_lab/pipeline.py — the
harness core is FROZEN. Converts a daily-reindexed membership matrix into a
qf-lib AlphaModel (LONG for held names, OUT otherwise) and runs one
BacktestTradingSession, returning the daily EOD simple-return series.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

import compare_lab  # noqa: F401  (path bootstrap)
from alpha_lab.core import DATA_START, OS_END
from compare_lab.config import MAX_POSITIONS


def weights_to_membership(weights: pd.DataFrame) -> pd.DataFrame:
    """Boolean membership: True where target weight > 0."""
    return weights.fillna(0.0) > 0.0


def _daily_membership(membership: pd.DataFrame, daily_index: pd.DatetimeIndex
                      ) -> pd.DataFrame:
    """Stair-step rebal-anchored membership onto every trading day."""
    m = membership.astype("int8").reindex(
        membership.index.union(daily_index)).ffill()
    m = m.reindex(daily_index).fillna(0).astype(bool)
    return m


def _build_alpha_model(membership_daily: pd.DataFrame, data_provider):
    from qf_lib.backtesting.alpha_model.alpha_model import AlphaModel
    from qf_lib.backtesting.alpha_model.exposure_enum import Exposure

    class PrecomputedMembershipAlphaModel(AlphaModel):
        def __init__(self, dp):
            super().__init__(risk_estimation_factor=1.0, data_provider=dp)
            self._m = membership_daily

        def calculate_exposure(self, ticker, current_exposure, current_time,
                               frequency):
            ts = pd.Timestamp(current_time).normalize()
            idx = self._m.index
            prior = idx[idx < ts]
            if len(prior) == 0:
                return Exposure.OUT
            row = self._m.loc[prior[-1]]
            tstr = ticker.ticker if hasattr(ticker, "ticker") else str(ticker)
            try:
                return Exposure.LONG if bool(row.at[tstr]) else Exposure.OUT
            except KeyError:
                return Exposure.OUT

        def __hash__(self):
            return hash(("PrecomputedMembershipAlphaModel", id(self._m)))

    return PrecomputedMembershipAlphaModel(data_provider)


def run_backtest(weights: pd.DataFrame, ctx, max_positions: int = MAX_POSITIONS
                 ) -> pd.Series:
    """Run the qf-lib backtest for a target-weight matrix; return daily simple
    returns over [DATA_START, OS_END)."""
    here = (Path(__file__).resolve().parents[1] / "qf-lib-harness")
    os.environ.setdefault("QF_STARTING_DIRECTORY", str(here))

    from alpha_lab import _weasyprint_stub  # noqa: F401  (must precede qf_lib)
    import matplotlib
    matplotlib.use("Agg")

    from qf_lib.backtesting.events.time_event.regular_time_event.calculate_and_place_orders_event import (  # noqa: E501
        CalculateAndPlaceOrdersRegularEvent,
    )
    from qf_lib.backtesting.execution_handler.commission_models.ib_commission_model import IBCommissionModel  # noqa: E501
    from qf_lib.backtesting.position_sizer.fixed_portfolio_percentage_position_sizer import (  # noqa: E501
        FixedPortfolioPercentagePositionSizer,
    )
    from qf_lib.backtesting.strategies.alpha_model_strategy import AlphaModelStrategy  # noqa: E501
    from qf_lib.backtesting.trading_session.backtest_trading_session_builder import BacktestTradingSessionBuilder  # noqa: E501
    from qf_lib.common.enums.frequency import Frequency
    from qf_lib.common.tickers.tickers import YFinanceTicker
    from qf_lib.documents_utils.document_exporting.pdf_exporter import PDFExporter  # noqa: E501
    from qf_lib.documents_utils.excel.excel_exporter import ExcelExporter
    from qf_lib.settings import Settings

    from alpha_lab.core import PRICES_PATH
    from alpha_lab.parquet_data_provider import build_data_provider

    data_provider = build_data_provider(
        PRICES_PATH, start_date=DATA_START, end_date=OS_END,
        tickers_subset=ctx.universe,
    )
    settings = Settings(str(here / "config_files" / "settings.json"),
                        str(here / "config_files" / "secret_settings.json"))
    sb = BacktestTradingSessionBuilder(settings, PDFExporter(settings),
                                       ExcelExporter(settings))
    sb.set_data_provider(data_provider)
    sb.set_backtest_name("compare_lab")
    sb.set_position_sizer(FixedPortfolioPercentagePositionSizer,
                          fixed_percentage=1.0 / max_positions)
    sb.set_commission_model(IBCommissionModel)
    sb.set_frequency(Frequency.DAILY)

    from alpha_lab.core import IS_START
    ts = sb.build(IS_START, OS_END)

    daily_index = ctx.adj_close.index
    membership_daily = _daily_membership(weights_to_membership(weights),
                                         daily_index)
    model = _build_alpha_model(membership_daily, ts.data_provider)
    model_tickers = [YFinanceTicker(t) for t in ctx.universe]
    ts.use_data_preloading(model_tickers)

    strategy = AlphaModelStrategy(ts, {model: model_tickers},
                                  use_stop_losses=False)
    CalculateAndPlaceOrdersRegularEvent.set_daily_default_trigger_time()
    CalculateAndPlaceOrdersRegularEvent.exclude_weekends()
    strategy.subscribe(CalculateAndPlaceOrdersRegularEvent)

    ts.start_trading()
    return ts.portfolio.portfolio_eod_series().to_simple_returns()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest compare_lab/tests/test_backtest.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add compare_lab/backtest.py compare_lab/tests/test_backtest.py
git commit -m "compare_lab: add qf-lib backtest bridge (weights -> daily returns)"
```

---

## Task 10: report.py + run_comparison.py + end-to-end smoke

**Files:**
- Create: `compare_lab/report.py`
- Create: `compare_lab/run_comparison.py`
- Test: `compare_lab/tests/test_report.py`
- Test: `compare_lab/tests/test_end_to_end.py`

- [ ] **Step 1: Write the failing test `compare_lab/tests/test_report.py`**

```python
import pandas as pd

import compare_lab  # noqa: F401
from compare_lab.report import build_table


def test_build_table_has_one_row_per_provider():
    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    results = {
        "equal_weight": pd.Series(0.001, index=idx),
        "momentum_12_1": pd.Series(0.002, index=idx),
    }
    table = build_table(results)
    assert set(table["provider"]) == {"equal_weight", "momentum_12_1"}
    for col in ("CR", "SR", "HR", "MDD"):
        assert col in table.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest compare_lab/tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'compare_lab.report'`

- [ ] **Step 3: Write `compare_lab/report.py`**

```python
"""Comparison report: metrics table + plotly equity-curve HTML (spec §4.6)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from compare_lab.metrics import all_metrics


def build_table(results: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, returns in results.items():
        m = all_metrics(returns)
        rows.append({"provider": name, **m})
    return pd.DataFrame(rows)


def build_html(results: dict[str, pd.Series], out_path: Path) -> Path:
    import plotly.graph_objects as go

    fig = go.Figure()
    for name, returns in results.items():
        equity = (1.0 + returns.dropna()).cumprod()
        fig.add_trace(go.Scatter(x=equity.index, y=equity.values, name=name))
    fig.update_layout(title="compare_lab — equity curves",
                      xaxis_title="date", yaxis_title="growth of 1")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))
    return out_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest compare_lab/tests/test_report.py -v`
Expected: 1 passed.

- [ ] **Step 5: Write `compare_lab/run_comparison.py`**

```python
"""CLI: run every provider through the qf-lib backtest and report metrics.

Usage:
    uv run python -m compare_lab.run_comparison [--llm] [--out DIR]

Without --llm only the equal-weight and momentum baselines run (no server
needed). With --llm the prompt-only LLM provider also runs (requires a vLLM
endpoint reachable at config/base_url, served on DGX Spark via sparkq).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import compare_lab  # noqa: F401
from alpha_lab.core import load_context
from compare_lab.backtest import run_backtest
from compare_lab.config import OOS_START, OOS_END, REBAL_FREQ, UNIVERSE
from compare_lab.providers.equal_weight import EqualWeightProvider
from compare_lab.providers.momentum import MomentumProvider
from compare_lab.report import build_html, build_table


def _rebal_dates(daily_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    anchors = pd.date_range(OOS_START, OOS_END, freq=REBAL_FREQ)
    snapped = [daily_index[daily_index <= a][-1] for a in anchors
               if (daily_index <= a).any()]
    return pd.DatetimeIndex(sorted(set(snapped)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="include prompt-only LLM")
    ap.add_argument("--out", default="compare_lab/output")
    args = ap.parse_args()

    ctx = load_context(universe=UNIVERSE)
    rebal = _rebal_dates(ctx.adj_close.index)

    providers = [EqualWeightProvider(), MomentumProvider()]
    if args.llm:
        from compare_lab.llm_client import VLLMClient
        from compare_lab.providers.llm import LLMProvider
        providers.append(LLMProvider(VLLMClient()))

    results: dict[str, pd.Series] = {}
    for p in providers:
        print(f"[compare_lab] {p.name}: computing weights ...")
        w = p.weights(ctx, rebal)
        print(f"[compare_lab] {p.name}: backtesting ...")
        returns = run_backtest(w, ctx)
        oos = returns[(returns.index >= pd.Timestamp(OOS_START))
                      & (returns.index < pd.Timestamp(OOS_END))]
        results[p.name] = oos

    table = build_table(results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "comparison.csv", index=False)
    build_html(results, out / "equity.html")
    print(table.to_string(index=False))
    print(f"[compare_lab] wrote {out/'comparison.csv'} and {out/'equity.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Write the end-to-end smoke `compare_lab/tests/test_end_to_end.py`**

```python
"""End-to-end smoke: 2 tickers, short window, baselines only (no LLM).

Requires the qf-lib-harness submodule data (data/prices.parquet). Skipped if
the parquet is absent.
"""
import pandas as pd
import pytest

import compare_lab  # noqa: F401
from alpha_lab.core import PRICES_PATH, load_context
from compare_lab.backtest import run_backtest
from compare_lab.providers.equal_weight import EqualWeightProvider


@pytest.mark.skipif(not PRICES_PATH.exists(), reason="prices.parquet absent")
def test_equal_weight_end_to_end():
    ctx = load_context(universe=("AAPL", "MSFT"))
    daily = ctx.adj_close.index
    rebal = daily[(daily >= pd.Timestamp("2024-01-01"))
                  & (daily < pd.Timestamp("2024-04-01"))][::5]
    w = EqualWeightProvider().weights(ctx, rebal)
    returns = run_backtest(w, ctx, max_positions=2)
    assert isinstance(returns, pd.Series)
    assert returns.notna().sum() > 20
```

- [ ] **Step 7: Run report + smoke tests**

Run: `uv run pytest compare_lab/tests/test_report.py compare_lab/tests/test_end_to_end.py -v`
Expected: 2 passed (smoke may take ~1 min; it runs a real qf-lib backtest).

- [ ] **Step 8: Run the baseline comparison (no LLM)**

Run: `uv run python -m compare_lab.run_comparison --out compare_lab/output`
Expected: a printed table with `equal_weight` and `momentum_12_1` rows and
CR/SR/HR/MDD columns; files `compare_lab/output/comparison.csv` and
`equity.html` written.

- [ ] **Step 9: Commit**

```bash
git add compare_lab/report.py compare_lab/run_comparison.py compare_lab/tests/test_report.py compare_lab/tests/test_end_to_end.py
git commit -m "compare_lab: add report + run_comparison CLI + end-to-end smoke"
```

---

## Task 11: Serve the LLM and run the 3-way comparison (operational)

**Files:** none (operational; uses the `sparkq` and `dgx-spark-gpu` skills).

This task is not unit-tested — it serves a model and runs inference. Do it once
the baselines pass.

- [ ] **Step 1: Pick the model**

Choose the latest ~4B-class instruct/reasoning model available at build time;
fall back to `Qwen/Qwen3-4B` if the newest fails the GB10 serving check. Set it
as `VLLMClient(model=...)` (and `base_url`).

- [ ] **Step 2: Serve via sparkq (DGX Spark single-node)**

Submit a vLLM server job with `--enforce-eager` and BF16 (NOT FP8, NOT Ray) per
the `dgx-spark-gpu` skill. Confirm the OpenAI-compatible endpoint answers:

Run (adapt host/port to the sparkq allocation):
`curl -s http://<host>:8000/v1/models`
Expected: JSON listing the served model id.

- [ ] **Step 3: Run the 3-way comparison**

Run: `uv run python -m compare_lab.run_comparison --llm --out compare_lab/output`
Expected: table with `equal_weight`, `momentum_12_1`, `llm_prompt_only` rows.
First run populates the response cache (~14 tickers × ~120 weeks); reruns are
instant and reproducible.

- [ ] **Step 4: Commit the results snapshot**

```bash
git add compare_lab/output/comparison.csv
git commit -m "compare_lab: baseline vs momentum vs prompt-only LLM results"
```

> Note: `compare_lab/output/` equity.html and `.cache/` are gitignored
> (`output/` and `compare_lab/.cache/` rules); only the small `comparison.csv`
> is committed as the result of record.

---

## Self-Review

**Spec coverage:**
- §3 architecture / weight-matrix representation → Tasks 5–8 (providers), 9 (backtest). ✓
- §4.1 MarketSnapshotBuilder → Task 4. ✓
- §4.2 SignalProvider + 3 impls → Tasks 5, 6, 8. ✓
- §4.3 llm_client + cache → Task 7. ✓
- §4.4 qf-lib bridge → Task 9. ✓
- §4.5 metrics CR/SR/HR/MDD → Task 3. ✓
- §4.6 report + CLI → Task 10. ✓
- §5 eval config (universe/period/rebal/rf) → Task 2 + CLI `_rebal_dates`. ✓
- §6 look-ahead control → Task 4 (snapshot as-of test), momentum `.shift(21)` PIT (Task 6), backtest prior-anchor membership (Task 9). ✓
- §7 error handling → `parse_decision` default HOLD (Task 8), normalize_weights all-zero→cash (Task 5), metrics <30 obs→NaN (Task 3). ✓
- §8 tests → each task is TDD; end-to-end smoke Task 10. ✓
- §9 validation (relative comparison) → Task 10/11 output table. ✓
- §11 build order → Tasks follow config→metrics→snapshot→providers→llm→backtest→report. ✓

**Placeholder scan:** No TBD/TODO; every code step has full code; commands have expected output. ✓

**Type consistency:** `SignalProvider.weights(ctx, rebal_dates)` signature identical across base/equal/momentum/llm. `VLLMClient.complete(prompt, key)` used consistently in Tasks 7–8. `run_backtest(weights, ctx, max_positions)` used in Tasks 9–10. `all_metrics`/`build_table`/`build_html` names consistent. ✓
