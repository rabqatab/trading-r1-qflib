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
