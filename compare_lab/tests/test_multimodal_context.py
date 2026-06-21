"""PIT correctness tests for the multi-modal store.

The one invariant that matters: no row with a publish/filing timestamp later
than `as_of` may ever enter a snapshot. Each accessor is tested for that, plus
graceful degradation when a ticker has no rows for a modality (e.g. ETFs).
"""
from __future__ import annotations

import pandas as pd

from compare_lab.multimodal_context import MultiModalStore

AS_OF = pd.Timestamp("2024-06-28")


def _store():
    return MultiModalStore()


def test_news_never_future():
    df = _store().news("NVDA", AS_OF)
    assert not df.empty
    assert (df["published_at"] <= AS_OF).all()


def test_news_lookback_window():
    df = _store().news("NVDA", AS_OF, lookback_days=30)
    assert (df["published_at"] >= AS_OF - pd.Timedelta(days=30)).all()
    assert (df["published_at"] <= AS_OF).all()


def test_fundamentals_latest_filed_no_future():
    df = _store().fundamentals("AAPL", AS_OF)
    assert not df.empty
    assert (df["filing_date"] <= AS_OF).all()
    # one row per normalized concept (the latest filed as of AS_OF)
    assert df["concept_normalized"].is_unique
    assert "Revenue" in set(df["concept_normalized"])


def test_macro_uses_pit_release_date_no_future():
    df = _store().macro(AS_OF)
    assert not df.empty
    assert (df["release_date"] <= AS_OF).all()
    assert df["series"].is_unique          # latest value per series


def test_insider_has_direction_and_no_future():
    df = _store().insider("NVDA", AS_OF, lookback_days=180)
    if not df.empty:
        assert (df["start_date"] <= AS_OF).all()
        assert set(df["direction"].unique()).issubset({"BUY", "SELL", "NEUTRAL"})


def test_analyst_no_future():
    df = _store().analyst("NVDA", AS_OF, lookback_days=90)
    if not df.empty:
        assert (df["gradedate"] <= AS_OF).all()


def test_etf_modalities_degrade_gracefully():
    s = _store()
    # SPY has no company news / fundamentals / insider / analyst
    assert s.news("SPY", AS_OF).empty
    assert s.fundamentals("SPY", AS_OF).empty
    assert s.insider("SPY", AS_OF).empty


def test_render_sections_is_deterministic_and_nonempty():
    s = _store()
    a = s.render_sections("NVDA", AS_OF)
    b = s.render_sections("NVDA", AS_OF)
    assert a == b and isinstance(a, str) and len(a) > 0
    # ETF still renders (sections just say "none") without raising
    assert isinstance(s.render_sections("SPY", AS_OF), str)
