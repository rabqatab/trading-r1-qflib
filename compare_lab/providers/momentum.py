"""MomentumProvider - 12-1 momentum, top-N equal weight (#3 as weights)."""
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
