"""_rebal_dates honours an explicit OOS window (multimodal cycle evals 2025-H1)."""
import pandas as pd

from compare_lab.run_comparison import _rebal_dates


def test_rebal_dates_respect_window():
    idx = pd.bdate_range("2024-06-01", "2025-12-31")
    d = _rebal_dates(idx, start="2025-01-01", end="2025-07-01")
    assert d.min() >= pd.Timestamp("2025-01-01")
    assert d.max() < pd.Timestamp("2025-07-01")
