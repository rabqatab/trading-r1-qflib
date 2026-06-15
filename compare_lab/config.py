"""Locked evaluation constants for the comparison substrate (spec §5)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

# Paper Table S3 universe (12 equities + 2 ETFs). BRK.B -> BRK-B (Yahoo style).
UNIVERSE: tuple[str, ...] = (
    "NVDA", "MSFT", "AAPL", "META", "AMZN", "TSLA",
    "BRK-B", "JPM", "LLY", "JNJ", "XOM", "CVX",
    "SPY", "QQQ",
)

# Out-of-sample window. No training here, so the whole span is OOS.
OOS_START = datetime(2024, 1, 2)
OOS_END = datetime(2026, 4, 1)

# Paper's short holdout slice, reported alongside the full window.
PAPER_SLICE_START = datetime(2024, 6, 1)
PAPER_SLICE_END = datetime(2024, 9, 1)

REBAL_FREQ = "W-FRI"      # weekly rebalance anchor
RF_ANNUAL = 0.04          # risk-free rate, paper §7.1 (US10Y)
ANNUALIZATION = 252.0
TOP_N_MOMENTUM = 5        # held names for the momentum baseline
MAX_POSITIONS = 8         # LLM position budget -> size = 1/MAX_POSITIONS

CACHE_DIR = Path(__file__).resolve().parent / ".cache"
