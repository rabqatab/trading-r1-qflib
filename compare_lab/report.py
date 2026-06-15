"""Comparison report: metrics table + plotly equity-curve HTML (spec §4.6)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from compare_lab.metrics import all_metrics


def build_table(results: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, returns in results.items():
        m = all_metrics(returns)
        rows.append({"provider": name, **m})
    return pd.DataFrame(rows)


def build_html(results: dict[str, pd.Series], out_path: Path) -> Path:
    import plotly.graph_objects as go

    fig = go.Figure()
    for name, returns in results.items():
        equity = (1.0 + returns.dropna()).cumprod()
        fig.add_trace(go.Scatter(x=equity.index, y=equity.values, name=name))
    fig.update_layout(title="compare_lab - equity curves",
                      xaxis_title="date", yaxis_title="growth of 1")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))
    return out_path
