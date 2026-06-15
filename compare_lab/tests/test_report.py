import pandas as pd

import compare_lab  # noqa: F401
from compare_lab.report import build_table


def test_build_table_has_one_row_per_provider():
    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    results = {
        "equal_weight": pd.Series(0.001, index=idx),
        "momentum_12_1": pd.Series(0.002, index=idx),
    }
    table = build_table(results)
    assert set(table["provider"]) == {"equal_weight", "momentum_12_1"}
    for col in ("CR", "SR", "HR", "MDD"):
        assert col in table.columns
