import numpy as np
import pandas as pd

import compare_lab  # noqa: F401
from compare_lab.snapshot import MarketSnapshotBuilder


def _ctx_like():
    """Minimal fake ctx: 400 business days for one ticker 'AAA'."""
    idx = pd.bdate_range("2023-01-02", periods=400)
    base = pd.Series(np.linspace(100, 200, 400), index=idx)
    df = pd.DataFrame({"AAA": base})
    vol = pd.DataFrame({"AAA": np.full(400, 1_000_000.0)}, index=idx)

    class Ctx:
        adj_close = df
        open = df * 0.99
        high = df * 1.01
        low = df * 0.98
        volume = vol
        dollar_volume = df * vol
        universe = ("AAA",)
    return Ctx()


def test_snapshot_excludes_future_bars():
    ctx = _ctx_like()
    as_of = pd.Timestamp("2024-01-15")
    b = MarketSnapshotBuilder(ctx)
    text = b.build("AAA", as_of)
    # No date strictly after as_of may appear in the serialized window.
    future = ctx.adj_close.index[ctx.adj_close.index > as_of]
    for d in future[:5]:
        assert d.strftime("%Y-%m-%d") not in text


def test_snapshot_is_deterministic_and_hashable():
    ctx = _ctx_like()
    as_of = pd.Timestamp("2024-01-15")
    b = MarketSnapshotBuilder(ctx)
    t1 = b.build("AAA", as_of)
    t2 = b.build("AAA", as_of)
    assert t1 == t2
    assert len(b.snapshot_hash("AAA", as_of)) == 12


def test_snapshot_mentions_ticker_and_indicators():
    ctx = _ctx_like()
    text = MarketSnapshotBuilder(ctx).build("AAA", pd.Timestamp("2024-01-15"))
    assert "AAA" in text
    assert "rsi" in text.lower()


def test_price_only_has_no_modality_sections():
    snap = MarketSnapshotBuilder(_ctx_like()).build("AAA", pd.Timestamp("2024-06-28"))
    assert "=== NEWS" not in snap and "=== MACRO" not in snap


def test_multimodal_appends_sections_and_changes_hash():
    from compare_lab.multimodal_context import MultiModalStore
    ctx = _ctx_like()
    as_of = pd.Timestamp("2024-06-28")
    mm = MarketSnapshotBuilder(ctx, multimodal=MultiModalStore())
    snap = mm.build("AAA", as_of)
    assert "Indicators (latest):" in snap                 # price block preserved
    assert "=== NEWS (last 30d) ===" in snap
    assert "=== MACRO (latest) ===" in snap               # macro present (ticker-agnostic)
    assert mm.snapshot_hash("AAA", as_of) != \
        MarketSnapshotBuilder(ctx).snapshot_hash("AAA", as_of)
