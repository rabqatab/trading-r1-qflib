"""Oracle / pipeline sanity check: use the LABEL itself as the signal.

The label is forward-looking (15-day Sharpe-like) by construction, so following it
is "trading with look-ahead" — it MUST print money if the weight allocation and the
backtest are correct. If even the oracle fails, the bug is in the sizing/backtest,
not the model. Needs no inference (labels + prices only), so it also works on the
pre-2024 training window where no model decisions were ever cached.

    uv run python -m compare_lab.oracle_check                 # 2018-2023 (in-sample) + 2024-2026 (OOS)
"""
from __future__ import annotations

import pandas as pd

import compare_lab  # noqa: F401
from alpha_lab.core import load_context
from compare_lab.config import UNIVERSE
from compare_lab.labeling import make_labels
from compare_lab.metrics import all_metrics
from compare_lab.run_comparison import _available_universe
from compare_lab.sizing_compare import SCHEMES, TIER, _returns, _weight_matrix

WINDOWS = {"in-sample 2018-2023": ("2018-01-01", "2023-12-29"),
           "OOS 2024-2026": ("2024-01-02", "2026-03-31")}


def _label_score_matrix(ctx, tickers, start, end) -> pd.DataFrame:
    """[rebal_date x ticker] of label tier scores; weekly (every 5th trading day)."""
    lab = {t: make_labels(ctx.adj_close[t].dropna(), forward=True) for t in tickers}
    idx = ctx.adj_close.loc[start:end].index[::5]
    S = pd.DataFrame(index=idx, columns=tickers, dtype=float)
    for t in tickers:
        S[t] = lab[t].reindex(idx).map(lambda v: TIER.get(v))
    return S.dropna(how="all")


def run(start, end) -> pd.DataFrame:
    universe = list(_available_universe(UNIVERSE))
    ctx = load_context(universe=universe)
    universe = [t for t in universe if t in ctx.adj_close.columns]
    R = ctx.adj_close[universe].pct_change()
    S = _label_score_matrix(ctx, universe, start, end)
    rows = {}
    for name, fn in SCHEMES.items():
        r = _returns(_weight_matrix(S, R, fn), R, start=pd.Timestamp(start))
        m = all_metrics(r)
        yrs = len(r) / 252.0
        cagr = (1 + m["CR"]) ** (1 / yrs) - 1 if yrs > 0 and m["CR"] > -1 else float("nan")
        rows[name] = {"CAGR%": cagr * 100, "SR": m["SR"], "MDD%": m["MDD"] * 100,
                      "CR%": m["CR"] * 100}
    return pd.DataFrame(rows).T


def main() -> int:
    for label, (s, e) in WINDOWS.items():
        print(f"\n=== ORACLE (label-as-signal, look-ahead) — {label} ===")
        print(run(s, e).round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
