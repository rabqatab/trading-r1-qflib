"""1:1 comparison vs Trading-R1 paper (Table 3/4): per-ticker, single-name,
long/flat backtest on the paper's exact window (2024-06-01..08-31), built from
ALREADY-CACHED model decisions (no new inference).

Mapping matches our LLMProvider: {BUY, STRONG_BUY} -> fully held, else flat;
long-only; next-bar execution (decision at close d applies from d+1).

    uv run python -m compare_lab.compare_paper sftv1 grpo v1reggrpo_full
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

import compare_lab  # noqa: F401
from alpha_lab.core import load_context
from compare_lab.cache_io import read_decisions, resolve
from compare_lab.metrics import all_metrics

WIN_START, WIN_END = pd.Timestamp("2024-06-01"), pd.Timestamp("2024-08-31")
SEED_FROM = pd.Timestamp("2024-05-15")          # seed initial position from pre-window
PAPER = {  # paper Table 3/4 (Trading-R1 flagship): ticker -> (CR%, SR, HR%, MDD%)
    "NVDA": (8.08, 1.88, 70.0, 3.80), "AAPL": (5.82, 1.80, 63.6, 3.68),
    "MSFT": (2.38, 0.87, 60.4, 1.90), "AMZN": (5.39, 1.72, 63.0, 3.20),
    "SPY": (3.34, 1.60, 64.0, 1.52),
}
HELD = {"BUY", "STRONG_BUY"}


def _ticker_returns(px: pd.Series, dec: pd.DataFrame) -> pd.Series:
    """Daily strategy returns for one ticker over the paper window."""
    ret = px.pct_change()
    days = ret[(ret.index >= WIN_START) & (ret.index <= WIN_END)].index
    dec = dec.sort_values("date")
    # position effective the day AFTER each decision; hold until the next one
    held = (dec.assign(pos=dec.pred.isin(HELD).astype(float))
               .set_index("date")["pos"])
    out = []
    for t in days:
        eff = held[held.index < t]          # last decision strictly before today
        out.append((eff.iloc[-1] if len(eff) else 0.0) * ret.loc[t])
    return pd.Series(out, index=days)


def run(cache_dir) -> pd.DataFrame:
    dec = read_decisions(cache_dir)
    dec = dec[dec.date >= SEED_FROM]
    ctx = load_context(universe=sorted(dec.ticker.unique()))
    rows = []
    for tk in PAPER:
        if tk not in ctx.adj_close.columns:
            continue
        r = _ticker_returns(ctx.adj_close[tk].dropna(), dec[dec.ticker == tk])
        m = all_metrics(r)
        rows.append({"ticker": tk, "CR%": m["CR"] * 100, "SR": m["SR"],
                     "HR%": m["HR"] * 100, "MDD%": m["MDD"] * 100,
                     "p_SR": PAPER[tk][1], "p_MDD%": PAPER[tk][3]})
    return pd.DataFrame(rows).set_index("ticker")


def main() -> int:
    for c in sys.argv[1:]:
        p = resolve(c)
        df = run(p)
        print(f"\n=== {p.name}  (paper window 2024-06-01..08-31, per-ticker long/flat) ===")
        print(df.round(2).to_string())
        print(f"  our mean SR={df.SR.mean():.2f}  vs paper mean SR={df.p_SR.mean():.2f}"
              f"   |  our mean MDD={df['MDD%'].mean():.1f}%  vs paper {df['p_MDD%'].mean():.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
