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
