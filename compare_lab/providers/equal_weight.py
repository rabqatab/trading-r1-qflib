"""EqualWeightProvider - hold every universe name equally (market baseline)."""
from __future__ import annotations

import pandas as pd

from compare_lab.providers.base import SignalProvider, normalize_weights


class EqualWeightProvider(SignalProvider):
    name = "equal_weight"

    def weights(self, ctx, rebal_dates: pd.DatetimeIndex) -> pd.DataFrame:
        cols = list(ctx.universe)
        w = pd.DataFrame(1.0, index=rebal_dates, columns=cols)
        return normalize_weights(w)
