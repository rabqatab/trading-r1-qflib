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
