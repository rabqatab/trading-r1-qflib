"""Tests for the macro release-date PIT correction.

The delivered macro.parquet has release_date == date (a leak: FRED publishes
with a lag). The fix must make release_date never earlier than the true
publication, i.e. monthly series move into the following month and daily series
move to the next business day. Conservative (slightly late) is fine; early is a
leak and must never happen.
"""
from __future__ import annotations

import pandas as pd

from compare_lab.macro_pit import MONTHLY, correct_release_dates


def _df(rows):
    return pd.DataFrame(rows)


def test_monthly_series_release_moves_into_a_later_month():
    df = _df([
        {"series": "CPIAUCSL", "date": pd.Timestamp("2022-01-01"),
         "value": 1.0, "release_date": pd.Timestamp("2022-01-01")},
        {"series": "UNRATE", "date": pd.Timestamp("2022-01-01"),
         "value": 1.0, "release_date": pd.Timestamp("2022-01-01")},
        {"series": "FEDFUNDS", "date": pd.Timestamp("2022-01-01"),
         "value": 1.0, "release_date": pd.Timestamp("2022-01-01")},
    ])
    out = correct_release_dates(df)
    for _, r in out.iterrows():
        assert r["release_date"] > r["date"]
        # reference month is January -> release must be Feb or later
        assert (r["release_date"].year, r["release_date"].month) >= (2022, 2)


def test_daily_series_release_is_next_business_day():
    # 2022-01-07 is a Friday -> next business day is Monday 2022-01-10
    df = _df([{"series": "DGS10", "date": pd.Timestamp("2022-01-07"),
               "value": 1.6, "release_date": pd.Timestamp("2022-01-07")}])
    out = correct_release_dates(df)
    rel = out.iloc[0]["release_date"]
    assert rel == pd.Timestamp("2022-01-10")
    assert rel.weekday() < 5


def test_never_earlier_than_reference():
    df = _df([{"series": s, "date": pd.Timestamp("2023-06-01"),
               "value": 1.0, "release_date": pd.Timestamp("2023-06-01")}
              for s in ["CPIAUCSL", "DGS2", "VIXCLS", "FEDFUNDS"]])
    out = correct_release_dates(df)
    assert (out["release_date"] >= out["date"]).all()


def test_real_file_has_no_leak():
    m = pd.read_parquet("data/qflib_data_store/macro.parquet")
    out = correct_release_dates(m)
    assert (out["release_date"] >= out["date"]).all()
    # every monthly row must now publish at least into the next month
    mon = out[out["series"].isin(MONTHLY)]
    assert (mon["release_date"] - mon["date"]).dt.days.min() >= 28
    # daily rows shift by 1-3 calendar days (next business day)
    day = out[~out["series"].isin(MONTHLY)]
    gap = (day["release_date"] - day["date"]).dt.days
    assert gap.min() >= 1 and gap.max() <= 3
