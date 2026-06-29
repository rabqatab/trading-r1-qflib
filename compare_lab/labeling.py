"""Volatility-adjusted deterministic 5-class labeler (paper Appendix S2, Algorithm S1).

The label is the *ground truth* for SFT targets and the RL decision reward — it
needs no model. For each trading day we build a Sharpe-like signal from EMA
returns over horizons {3,7,15}, blend them, and cut the blended signal at its own
quantiles {0.03,0.15,0.53,0.85} into Strong Sell … Strong Buy. Because the cuts
are quantiles of the signal itself, the class mix reproduces the paper's bullish-
skewed Table 2 (3 / 12 / 38 / 32 / 15 %).

forward vs trailing (paper §3.3 caveat): Algorithm S1 literally writes
`EMA.shift(tau)` which is *trailing* (momentum). The intent of a label — "the
right action at t, judged by what price does next" — needs *forward* returns,
`EMA.shift(-tau)`. We default to forward. This puts future info only in the
training *answer* (a supervised label), never in the model's input, so it does
NOT create backtest look-ahead: the backtest feeds the model inputs up to t only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Ordered Strong Sell -> Strong Buy (index aligns with the 4 quantile cuts).
CLASSES: tuple[str, ...] = ("STRONG_SELL", "SELL", "HOLD", "BUY", "STRONG_BUY")


def make_signal(
    prices: pd.Series,
    *,
    horizons: tuple[int, ...] = (3, 7, 15),
    weights: tuple[float, ...] = (0.3, 0.5, 0.2),
    ema_span: int = 3,
    vol_window: int = 20,
    forward: bool = True,
) -> pd.Series:
    """The raw vol-adjusted (Sharpe-like) signal the label is cut from — the
    per-day *performance proxy value* (NaN where undefined)."""
    ema = prices.ewm(span=ema_span).mean()
    weighted = pd.Series(0.0, index=prices.index)
    for tau, w in zip(horizons, weights):
        if forward:                       # (EMA_{t+tau} - EMA_t) / EMA_t
            r = (ema.shift(-tau) - ema) / ema
        else:                             # trailing: (EMA_t - EMA_{t-tau}) / EMA_{t-tau}
            r = (ema - ema.shift(tau)) / ema.shift(tau)
        v = r.rolling(vol_window).std()
        weighted = weighted + w * (r / v)  # NaN in any horizon -> NaN blend
    return weighted


def make_labels(
    prices: pd.Series,
    *,
    horizons: tuple[int, ...] = (3, 7, 15),
    weights: tuple[float, ...] = (0.3, 0.5, 0.2),
    quantiles: tuple[float, ...] = (0.03, 0.15, 0.53, 0.85),
    ema_span: int = 3,
    vol_window: int = 20,
    forward: bool = True,
) -> pd.Series:
    """Map a price series to a per-day 5-class label (NaN where undefined)."""
    weighted = make_signal(prices, horizons=horizons, weights=weights,
                           ema_span=ema_span, vol_window=vol_window, forward=forward)

    valid = weighted.dropna()
    thr = [valid.quantile(q) for q in quantiles]

    def classify(x: float):
        if pd.isna(x):
            return np.nan
        if x >= thr[3]:
            return "STRONG_BUY"
        if x >= thr[2]:
            return "BUY"
        if x >= thr[1]:
            return "HOLD"
        if x >= thr[0]:
            return "SELL"
        return "STRONG_SELL"

    return weighted.map(classify)
