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
