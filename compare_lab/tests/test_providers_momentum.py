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
