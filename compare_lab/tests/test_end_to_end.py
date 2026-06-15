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
