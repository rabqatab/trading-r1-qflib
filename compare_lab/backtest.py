"""qf-lib backtest bridge for compare_lab.

Adapted (copied, not imported) from qf-lib-harness/alpha_lab/pipeline.py - the
harness core is FROZEN. Converts a daily-reindexed membership matrix into a
qf-lib AlphaModel (LONG for held names, OUT otherwise) and runs one
BacktestTradingSession, returning the daily EOD simple-return series.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

import compare_lab  # noqa: F401  (path bootstrap)
from alpha_lab.core import DATA_START, OS_END
from compare_lab.config import MAX_POSITIONS


def weights_to_membership(weights: pd.DataFrame) -> pd.DataFrame:
    """Boolean membership: True where target weight > 0."""
    return weights.fillna(0.0) > 0.0


def _daily_membership(membership: pd.DataFrame, daily_index: pd.DatetimeIndex
                      ) -> pd.DataFrame:
    """Stair-step rebal-anchored membership onto every trading day."""
    m = membership.astype("int8").reindex(
        membership.index.union(daily_index)).ffill()
    m = m.reindex(daily_index).fillna(0).astype(bool)
    return m


def _build_alpha_model(membership_daily: pd.DataFrame, data_provider):
    from qf_lib.backtesting.alpha_model.alpha_model import AlphaModel
    from qf_lib.backtesting.alpha_model.exposure_enum import Exposure

    class PrecomputedMembershipAlphaModel(AlphaModel):
        def __init__(self, dp):
            super().__init__(risk_estimation_factor=1.0, data_provider=dp)
            self._m = membership_daily

        def calculate_exposure(self, ticker, current_exposure, current_time,
                               frequency):
            ts = pd.Timestamp(current_time).normalize()
            idx = self._m.index
            prior = idx[idx < ts]
            if len(prior) == 0:
                return Exposure.OUT
            row = self._m.loc[prior[-1]]
            tstr = ticker.ticker if hasattr(ticker, "ticker") else str(ticker)
            try:
                return Exposure.LONG if bool(row.at[tstr]) else Exposure.OUT
            except KeyError:
                return Exposure.OUT

        def __hash__(self):
            return hash(("PrecomputedMembershipAlphaModel", id(self._m)))

    return PrecomputedMembershipAlphaModel(data_provider)


def run_backtest(weights: pd.DataFrame, ctx, max_positions: int = MAX_POSITIONS
                 ) -> pd.Series:
    """Run the qf-lib backtest for a target-weight matrix; return daily simple
    returns over [DATA_START, OS_END)."""
    here = (Path(__file__).resolve().parents[1] / "qf-lib-harness")
    os.environ.setdefault("QF_STARTING_DIRECTORY", str(here))

    from alpha_lab import _weasyprint_stub  # noqa: F401  (must precede qf_lib)
    import matplotlib
    matplotlib.use("Agg")

    from qf_lib.backtesting.events.time_event.regular_time_event.calculate_and_place_orders_event import (  # noqa: E501
        CalculateAndPlaceOrdersRegularEvent,
    )
    from qf_lib.backtesting.execution_handler.commission_models.ib_commission_model import IBCommissionModel  # noqa: E501
    from qf_lib.backtesting.position_sizer.fixed_portfolio_percentage_position_sizer import (  # noqa: E501
        FixedPortfolioPercentagePositionSizer,
    )
    from qf_lib.backtesting.strategies.alpha_model_strategy import AlphaModelStrategy  # noqa: E501
    from qf_lib.backtesting.trading_session.backtest_trading_session_builder import BacktestTradingSessionBuilder  # noqa: E501
    from qf_lib.common.enums.frequency import Frequency
    from qf_lib.common.tickers.tickers import YFinanceTicker
    from qf_lib.documents_utils.document_exporting.pdf_exporter import PDFExporter  # noqa: E501
    from qf_lib.documents_utils.excel.excel_exporter import ExcelExporter
    from qf_lib.settings import Settings

    from alpha_lab.core import PRICES_PATH
    from alpha_lab.parquet_data_provider import build_data_provider

    data_provider = build_data_provider(
        PRICES_PATH, start_date=DATA_START, end_date=OS_END,
        tickers_subset=ctx.universe,
    )
    settings = Settings(str(here / "config_files" / "settings.json"),
                        str(here / "config_files" / "secret_settings.json"))
    sb = BacktestTradingSessionBuilder(settings, PDFExporter(settings),
                                       ExcelExporter(settings))
    sb.set_data_provider(data_provider)
    sb.set_backtest_name("compare_lab")
    sb.set_position_sizer(FixedPortfolioPercentagePositionSizer,
                          fixed_percentage=1.0 / max_positions)
    sb.set_commission_model(IBCommissionModel)
    sb.set_frequency(Frequency.DAILY)

    from alpha_lab.core import IS_START
    ts = sb.build(IS_START, OS_END)

    daily_index = ctx.adj_close.index
    membership_daily = _daily_membership(weights_to_membership(weights),
                                         daily_index)
    model = _build_alpha_model(membership_daily, ts.data_provider)
    model_tickers = [YFinanceTicker(t) for t in ctx.universe]
    ts.use_data_preloading(model_tickers)

    strategy = AlphaModelStrategy(ts, {model: model_tickers},
                                  use_stop_losses=False)
    CalculateAndPlaceOrdersRegularEvent.set_daily_default_trigger_time()
    CalculateAndPlaceOrdersRegularEvent.exclude_weekends()
    strategy.subscribe(CalculateAndPlaceOrdersRegularEvent)

    ts.start_trading()
    return ts.portfolio.portfolio_eod_series().to_simple_returns()
