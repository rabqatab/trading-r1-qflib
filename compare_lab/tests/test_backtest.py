import pandas as pd

import compare_lab  # noqa: F401
from compare_lab.backtest import weights_to_membership


def test_weights_to_membership():
    w = pd.DataFrame(
        {"A": [0.5, 0.0], "B": [0.5, 0.0], "C": [0.0, 0.0]},
        index=pd.to_datetime(["2024-01-05", "2024-01-12"]),
    )
    m = weights_to_membership(w)
    assert m.loc["2024-01-05", "A"]
    assert not m.loc["2024-01-05", "C"]
    assert not m.loc["2024-01-12"].any()   # all cash
