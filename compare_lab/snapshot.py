"""MarketSnapshotBuilder - per-ticker, as-of, price + technical indicators.

Strictly causal: bars with date > as_of_date are physically removed before
indicators are computed. Output is a compact text block for an LLM prompt.
MVP modalities = price + technical only (spec §1.3, §4.1).

Note: stockstats does not support a bare 'roc' key; the correct form is
'close_10_roc' (10-period rate of change on close). This is used in place of
the generic 'roc' listed in the spec.
"""
from __future__ import annotations

from hashlib import sha1

import os
import pandas as pd
from stockstats import StockDataFrame

# Technical indicators (subset of paper Table S2) computed via stockstats.
# 'roc' is not a valid stockstats key; 'close_10_roc' (10-period ROC) is used.
_INDICATORS = [
    "close_50_sma", "close_200_sma", "close_10_ema",
    "macd", "macds", "macdh",
    "rsi_14", "kdjk", "cci", "close_10_roc",
    "atr", "boll", "boll_ub", "boll_lb",
    "dx", "mfi",
]
_OUTPUT_WINDOW = 15   # trailing bars shown in the snapshot
_LOOKBACK_BARS = 520  # ~2y of bars fed to indicators (for 200 SMA etc.)


def _abbrev(x: float) -> str:
    if pd.isna(x):
        return "na"
    ax = abs(x)
    if ax >= 1e9:
        return f"{x/1e9:.2f}b"
    if ax >= 1e6:
        return f"{x/1e6:.2f}m"
    if ax >= 1e3:
        return f"{x/1e3:.2f}k"
    return f"{x:.2f}"


class MarketSnapshotBuilder:
    def __init__(self, ctx, multimodal=None):
        self._ctx = ctx
        # optional MultiModalStore: when set, build() appends PIT-filtered
        # news/fundamentals/sentiment/macro sections (paper-parity input).
        self._multimodal = multimodal
        # MM_RICH=1 → dump every headline in a 60d window (paper-faithful raw text,
        # ~3-8k tok) instead of the bucketed ≤50 view.
        self._mm_rich = os.environ.get("MM_RICH") == "1"

    def _window(self, ticker: str, as_of) -> pd.DataFrame:
        as_of = pd.Timestamp(as_of)
        c = self._ctx
        cols = {
            "open": c.open[ticker], "high": c.high[ticker],
            "low": c.low[ticker], "close": c.adj_close[ticker],
            "volume": c.volume[ticker],
        }
        df = pd.DataFrame(cols).dropna()
        df = df[df.index <= as_of]          # strict causality
        return df.tail(_LOOKBACK_BARS)

    def build(self, ticker: str, as_of) -> str:
        df = self._window(ticker, as_of)
        if df.empty:
            return f"Ticker {ticker} as of {pd.Timestamp(as_of).date()}: no data."
        sdf = StockDataFrame.retype(df.copy())
        for ind in _INDICATORS:
            _ = sdf[ind]                    # triggers computation
        recent = sdf.tail(_OUTPUT_WINDOW)
        lines = [
            f"Ticker: {ticker}",
            f"As of: {pd.Timestamp(as_of).date()} "
            f"(showing last {len(recent)} trading days)",
            "",
            "Date | Open High Low Close Volume",
        ]
        for d, row in recent.iterrows():
            lines.append(
                f"{d.strftime('%Y-%m-%d')} | "
                f"{_abbrev(row['open'])} {_abbrev(row['high'])} "
                f"{_abbrev(row['low'])} {_abbrev(row['close'])} "
                f"{_abbrev(row['volume'])}"
            )
        last = recent.iloc[-1]
        lines.append("")
        lines.append("Indicators (latest):")
        for ind in _INDICATORS:
            lines.append(f"  {ind}: {_abbrev(last.get(ind, float('nan')))}")
        if self._multimodal is not None:
            lines.append("")
            lines.append(self._multimodal.render_sections(
                ticker, as_of, rich=self._mm_rich))
        return "\n".join(lines)

    def snapshot_hash(self, ticker: str, as_of) -> str:
        text = self.build(ticker, as_of)
        return sha1(text.encode()).hexdigest()[:12]
