"""Tests for the volatility-adjusted 5-class labeler (Algorithm S1).

The thresholds are quantiles {0.03, 0.15, 0.53, 0.85} of the weighted signal
itself, so applying them back to the same series must reproduce Table 2's
proportions (3 / 12 / 38 / 32 / 15 %). That makes the distribution a strong,
deterministic check that the quantile->class wiring is correct (not inverted).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from compare_lab.labeling import CLASSES, make_labels


def _price_series(n: int = 3000, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0005, 0.02, n)          # mild upward drift
    prices = 100.0 * np.exp(np.cumsum(steps))
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.Series(prices, index=idx)


def test_distribution_matches_table2():
    labels = make_labels(_price_series(), forward=True).dropna()
    frac = labels.value_counts(normalize=True)
    target = {"STRONG_SELL": 0.03, "SELL": 0.12, "HOLD": 0.38,
              "BUY": 0.32, "STRONG_BUY": 0.15}
    for cls, want in target.items():
        assert abs(frac.get(cls, 0.0) - want) < 0.03, (cls, frac.get(cls, 0.0))


def test_forward_tail_is_undefined():
    # forward uses shift(-tau): the last max(horizon)=15 rows have no future EMA
    labels = make_labels(_price_series(), forward=True)
    assert labels.iloc[-15:].isna().all()


def test_trailing_has_label_on_last_day():
    # trailing uses shift(+tau): the most recent day IS labelable (past data only)
    fwd = make_labels(_price_series(), forward=True)
    trail = make_labels(_price_series(), forward=False)
    assert pd.isna(fwd.iloc[-1])
    assert not pd.isna(trail.iloc[-1])


def test_only_valid_classes_emitted():
    labels = make_labels(_price_series(), forward=True).dropna()
    assert set(labels.unique()).issubset(set(CLASSES))


def test_deterministic():
    a = make_labels(_price_series(), forward=True)
    b = make_labels(_price_series(), forward=True)
    pd.testing.assert_series_equal(a, b)
