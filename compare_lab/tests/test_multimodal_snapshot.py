"""Smoke gate: the multimodal store renders into a snapshot on the 2024 window
(news/fundamentals/sentiment/macro PIT-joined) without crashing, and price+
technical is still present. Guards the SFT/GRPO multimodal builders downstream."""
import compare_lab  # noqa: F401  (sets up the alpha_lab import path)
import pandas as pd
from alpha_lab.core import load_context
from compare_lab.config import UNIVERSE_MM
from compare_lab.multimodal_context import MultiModalStore
from compare_lab.run_comparison import _available_universe
from compare_lab.snapshot import MarketSnapshotBuilder


def test_multimodal_snapshot_renders_2024():
    uni = _available_universe(UNIVERSE_MM)
    ctx = load_context(universe=uni)
    b = MarketSnapshotBuilder(ctx, multimodal=MultiModalStore())
    snap = b.build("NVDA", pd.Timestamp("2024-06-03"))
    for header in ("=== NEWS", "=== FUNDAMENTALS", "=== SENTIMENT", "=== MACRO"):
        assert header in snap, f"missing {header}"
    assert "Indicators (latest)" in snap          # price+technical still present
