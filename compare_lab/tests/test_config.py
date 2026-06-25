from datetime import datetime

import compare_lab  # noqa: F401
from compare_lab import config


def test_universe_has_paper_tickers_and_etfs():
    for t in ["NVDA", "AAPL", "BRK-B", "SPY", "QQQ"]:
        assert t in config.UNIVERSE
    assert len(config.UNIVERSE) == 14


def test_oos_window_ordered():
    assert config.OOS_START < config.OOS_END
    assert config.OOS_START == datetime(2024, 1, 2)


def test_constants():
    assert config.RF_ANNUAL == 0.04
    assert config.REBAL_FREQ == "W-FRI"
    assert config.MAX_POSITIONS >= 1


def test_mm_constants():
    assert len(config.UNIVERSE_MM) == 12
    assert "SPY" not in config.UNIVERSE_MM and "QQQ" not in config.UNIVERSE_MM
    assert config.MM_TRAIN_START == datetime(2024, 1, 1)
    assert config.MM_TRAIN_END == datetime(2024, 12, 31)
    assert config.MM_OOS_START == datetime(2025, 1, 1)
    assert config.MM_OOS_END == datetime(2025, 7, 1)
    assert config.MM_TRAIN_END < config.MM_OOS_START      # leak-safe ordering
