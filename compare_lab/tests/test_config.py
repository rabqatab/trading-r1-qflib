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
