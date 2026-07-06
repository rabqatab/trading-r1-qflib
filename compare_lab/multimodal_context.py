"""Point-in-time multi-modal context for snapshots (news/fundamentals/sentiment/macro).

Loads the data-store parquets (the leak-fixed `*_pit.parquet` variants where they
exist) and exposes per-`(ticker, as_of)` accessors that filter strictly on each
modality's own publish/filing timestamp — so no future-dated row can ever enter a
snapshot. `render_sections` turns them into a compact text block to append to the
price+technical snapshot (`snapshot.py`), giving the LLM the paper's full input.

Timestamps used (all `<= as_of`): news `published_at`, fundamentals `filing_date`,
analyst `gradedate`, insider `start_date`, macro `release_date`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_STORE = Path(__file__).resolve().parents[1] / "data" / "qflib_data_store"


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


class MultiModalStore:
    def __init__(self, store_dir: Path = _STORE):
        d = Path(store_dir)

        def _pref(pit: str, raw: str) -> pd.DataFrame:
            p = d / pit
            return pd.read_parquet(p if p.exists() else d / raw)

        self._news = pd.read_parquet(d / "news.parquet")
        self._fund = _pref("fundamentals_pit.parquet", "fundamentals.parquet")
        self._analyst = pd.read_parquet(d / "sentiment_analyst.parquet")
        self._insider = _pref("sentiment_insider_pit.parquet", "sentiment_insider.parquet")
        self._macro = _pref("macro_pit.parquet", "macro.parquet")
        if "concept_normalized" not in self._fund.columns:
            self._fund = self._fund.assign(concept_normalized=self._fund["concept"])

    # ---- per-modality PIT accessors (all rows guaranteed <= as_of) ----------

    def news(self, ticker: str, as_of, lookback_days: int = 30) -> pd.DataFrame:
        as_of = pd.Timestamp(as_of)
        lo = as_of - pd.Timedelta(days=lookback_days)
        df = self._news[(self._news["ticker"] == ticker)
                        & (self._news["published_at"] <= as_of)
                        & (self._news["published_at"] >= lo)]
        return df.sort_values("published_at", ascending=False).reset_index(drop=True)

    def fundamentals(self, ticker: str, as_of) -> pd.DataFrame:
        as_of = pd.Timestamp(as_of)
        df = self._fund[(self._fund["ticker"] == ticker)
                        & (self._fund["filing_date"] <= as_of)]
        if df.empty:
            return df.reset_index(drop=True)
        df = df.sort_values("filing_date").drop_duplicates(
            subset=["concept_normalized"], keep="last")
        return df.sort_values("concept_normalized").reset_index(drop=True)

    def analyst(self, ticker: str, as_of, lookback_days: int = 90) -> pd.DataFrame:
        as_of = pd.Timestamp(as_of)
        lo = as_of - pd.Timedelta(days=lookback_days)
        df = self._analyst[(self._analyst["ticker"] == ticker)
                           & (self._analyst["gradedate"] <= as_of)
                           & (self._analyst["gradedate"] >= lo)]
        return df.sort_values("gradedate", ascending=False).reset_index(drop=True)

    def insider(self, ticker: str, as_of, lookback_days: int = 90) -> pd.DataFrame:
        as_of = pd.Timestamp(as_of)
        lo = as_of - pd.Timedelta(days=lookback_days)
        df = self._insider[(self._insider["ticker"] == ticker)
                           & (self._insider["start_date"] <= as_of)
                           & (self._insider["start_date"] >= lo)]
        return df.sort_values("start_date", ascending=False).reset_index(drop=True)

    def macro(self, as_of) -> pd.DataFrame:
        as_of = pd.Timestamp(as_of)
        df = self._macro[self._macro["release_date"] <= as_of]
        if df.empty:
            return df.reset_index(drop=True)
        df = df.sort_values("date").drop_duplicates(subset=["series"], keep="last")
        return df.sort_values("series").reset_index(drop=True)

    # ---- rendering ----------------------------------------------------------

    # paper Table S1: 3 time buckets, capped per bucket, for a spread-out view.
    _NEWS_BUCKETS = (("≤3d", 0, 3, 10), ("4-10d", 4, 10, 20), ("11-30d", 11, 30, 20))

    def render_sections(self, ticker: str, as_of, *, max_events: int = 12,
                        rich: bool = False, with_summary: bool = False) -> str:
        as_of = pd.Timestamp(as_of)
        out: list[str] = []

        # rich = paper-faithful "dump the raw text": every headline in a 60d window,
        # uncapped. Default = the bucketed ≤50 view.
        # with_summary = append each article's Finnhub `summary` (Track B, #5): the only
        # gap-closing lever that could raise I(X;Y) over headline-only text.
        def _line(r):
            base = f"  {r['published_at'].date()} | {r['headline']} ({r['source']})"
            s = str(r.get("summary", "") or "").strip() if with_summary else ""
            return base + (f"\n      {s}" if s else "")

        lookback = 60 if rich else 30
        news = self.news(ticker, as_of, lookback_days=lookback)
        out.append(f"=== NEWS (last {lookback}d) ===")
        if news.empty:
            out.append("  none")
        elif rich:
            for _, r in news.sort_values("published_at", ascending=False).iterrows():
                out.append(_line(r))
        else:
            age = (as_of - news["published_at"]).dt.days
            for tag, lo, hi, cap in self._NEWS_BUCKETS:
                b = news[(age >= lo) & (age <= hi)].head(cap)
                if b.empty:
                    continue
                out.append(f"  [{tag}]")
                for _, r in b.iterrows():
                    out.append(_line(r))

        fund = self.fundamentals(ticker, as_of)
        out.append("=== FUNDAMENTALS (latest filed) ===")
        if fund.empty:
            out.append("  none")
        else:
            for _, r in fund.iterrows():
                out.append(f"  {r['concept_normalized']}: {_abbrev(r['value'])} "
                           f"({r['form']}, filed {r['filing_date'].date()})")

        an = self.analyst(ticker, as_of)
        ins = self.insider(ticker, as_of)
        out.append("=== SENTIMENT ===")
        if an.empty and ins.empty:
            out.append("  none")
        for _, r in an.head(max_events).iterrows():
            frm = r["fromgrade"] or "?"
            out.append(f"  analyst {r['gradedate'].date()}: {r['firm']} "
                       f"{frm}->{r['tograde']} ({r['action']})")
        for _, r in ins.head(max_events).iterrows():
            out.append(f"  insider {r['start_date'].date()}: {r.get('txn_type','?')} "
                       f"{_abbrev(r['shares'])} sh [{r.get('direction','?')}]")

        macro = self.macro(as_of)
        out.append("=== MACRO (latest) ===")
        if macro.empty:
            out.append("  none")
        else:
            for _, r in macro.iterrows():
                out.append(f"  {r['series']}: {r['value']:.2f} "
                           f"(ref {r['date'].date()})")

        return "\n".join(out)
