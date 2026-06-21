"""Fix the macro point-in-time leak: derive a leak-safe `release_date`.

The delivered `macro.parquet` sets `release_date == date`, but FRED publishes
with a lag — monthly series (CPI, unemployment, fed funds) for reference month M
are released early in month M+1, and daily series (Treasury yields, VIX, FX) are
the EOD value, available the next business day. Filtering "macro known as of t"
on the delivered column would admit figures before they were published.

We rebuild `release_date` with a CONSERVATIVE rule — never earlier than the true
publication (slightly late is safe; early leaks). The monthly target days
(FEDFUNDS 3rd, UNRATE 8th, CPI 15th of M+1) sit at or just after the real
release schedule, absorbing year-to-year drift. The exact fix is FRED/ALFRED
vintage dates (needs an API key); this is the no-key, leak-safe approximation.

    uv run python -m compare_lab.macro_pit          # writes macro_pit.parquet
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Monthly FRED series (reference date is the month start; published in M+1).
MONTHLY: tuple[str, ...] = ("CPIAUCSL", "UNRATE", "FEDFUNDS")
# Conservative day-of-(M+1) each monthly series is treated as published.
_MONTHLY_DAY = {"FEDFUNDS": 3, "UNRATE": 8, "CPIAUCSL": 15}

_STORE = Path(__file__).resolve().parents[1] / "data" / "qflib_data_store"


def _roll_to_business_day(s: pd.Series) -> pd.Series:
    """Push Saturdays/Sundays forward to the following Monday (vectorized)."""
    add = s.dt.weekday.map({5: 2, 6: 1}).fillna(0).astype("int64")
    return s + pd.to_timedelta(add, unit="D")


def correct_release_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with a leak-safe `release_date` column."""
    out = df.copy()
    date = pd.to_datetime(out["date"])
    is_monthly = out["series"].isin(MONTHLY)

    # daily series: value for day d is known the next business day
    daily_rel = _roll_to_business_day(date + pd.Timedelta(days=1))

    # monthly series: M+1 at the series-specific conservative day, off weekends
    target = out["series"].map(_MONTHLY_DAY).fillna(1).astype("int64")
    base = date + pd.DateOffset(months=1)            # still day 1 of M+1
    monthly_rel = base + pd.to_timedelta(target - 1, unit="D")
    monthly_rel = _roll_to_business_day(monthly_rel)

    out["release_date"] = monthly_rel.where(is_monthly, daily_rel)
    return out


def main() -> int:
    src = _STORE / "macro.parquet"
    df = pd.read_parquet(src)
    fixed = correct_release_dates(df)
    leaks = int((fixed["release_date"] < fixed["date"]).sum())
    assert leaks == 0, f"{leaks} rows still leak"
    out = _STORE / "macro_pit.parquet"
    fixed.to_parquet(out, index=False)
    delta = (fixed["release_date"] - df["release_date"]).dt.days
    print(f"rows={len(fixed)}  leaks={leaks}  "
          f"shift days: monthly~{delta[df['series'].isin(MONTHLY)].median():.0f}, "
          f"daily~{delta[~df['series'].isin(MONTHLY)].median():.0f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
