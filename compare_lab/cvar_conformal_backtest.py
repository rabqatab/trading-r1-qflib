"""Roadmap B, realized: adversarially-robust online CVaR control (Chen-Shen-Deng-Lei,
arXiv:2606.00320 — Rockafellar-Uryasev conformal inference) applied to the 3-signal combo
portfolio, on a multi-year daily backtest (2017-07 .. 2026-05).

Two questions at once:
  1. Multi-year OOS: does the combo signal (mom10 + analyst-revision + PEAD rank mean,
     raw IC 0.096 on 2025-H1) survive 9 years of daily cross-sections?
  2. Risk control: does the paper's online controller keep realized CVaR_beta at target
     alpha through 2018Q4 / 2020 COVID / 2022 bear / 2025Q1 without recalibration,
     vs static full exposure and trailing fractional-Kelly (MacLean-Thorp-Ziemba)?

Strategy = daily-rebalanced dollar-neutral long-short: long top-quintile combo, short
bottom-quintile, r_LS = (mean_top - mean_bot)/2 (gross 1). lambda_t in [0,1] = fraction
in the strategy, rest cash (rf=0). Loss R_t = -lambda_t * r_LS_t.

Controller (paper Sec. 2, ponytail-faithful):
  inner  c_t   : AdaGrad projected subgradient on the RU threshold,
                 g_t = 1 - 1/(1-beta) * 1{R_t > c_t},  eta_t ∝ 1/sqrt(sum g^2)
  outer  lam_t : lam <- clip(lam - gamma*(l_RU_t - alpha), 0, 1),
                 l_RU_t = c_t + (R_t - c_t)_+ / (1-beta)

Honesty caveats printed with results: today's-top-150 universe => survivorship bias in
the ALPHA numbers for early years (the CVaR-CONTROL result is unaffected — it holds for
whatever loss stream you feed it); costs unhaircut; close-to-close same-close execution.

    uv run python -m compare_lab.cvar_conformal_backtest
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import compare_lab  # noqa: F401

_PRICES = Path("data/qflib_data_store_top150/prices_top150.parquet")
_RECS = Path("data/finnhub_recs")
_EARN = Path("data/finnhub_earnings_full")

BETA = 0.85          # CVaR level (paper's portfolio experiment)
ALPHA = 0.005        # target daily CVaR_0.85 of portfolio loss (0.5 %)
GAMMA = 0.05         # outer step size
START = "2017-07-01"


def build_signals() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    px = pd.read_parquet(_PRICES, columns=["date", "ticker", "Close"])
    px["date"] = pd.to_datetime(px["date"])
    piv = px.pivot(index="date", columns="ticker", values="Close").sort_index()
    dates = piv.index

    mom = piv / piv.shift(10) - 1

    # analyst-revision: daily-ffilled consensus score, 3-month (63 trading day) change
    rev = pd.DataFrame(index=dates, columns=piv.columns, dtype=float)
    for t in piv.columns:
        p = _RECS / f"{t}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        if not d:
            continue
        df = pd.DataFrame(d)
        df["period"] = pd.to_datetime(df["period"])
        tot = df.strongBuy + df.buy + df.hold + df.sell + df.strongSell
        s = ((df.strongBuy * 2 + df.buy - df.sell - df.strongSell * 2)
             / tot.replace(0, np.nan))
        s.index = df["period"]
        s = s.sort_index()
        s = s[~s.index.duplicated(keep="last")]
        daily = s.reindex(dates, method="ffill")
        rev[t] = daily - daily.shift(63)

    # PEAD: latest EPS surprise%, announced <= d, expires after 90 calendar days
    pead = pd.DataFrame(index=dates, columns=piv.columns, dtype=float)
    for t in piv.columns:
        p = _EARN / f"{t}.json"
        if not p.exists():
            continue
        rows = [(pd.Timestamp(a["date"]),
                 (a["epsActual"] - a["epsEstimate"]) / abs(a["epsEstimate"]))
                for a in json.loads(p.read_text())
                if a.get("epsActual") is not None and a.get("epsEstimate")]
        if not rows:
            continue
        ev = pd.Series(dict(rows)).sort_index()
        ev = ev[~ev.index.duplicated(keep="last")]
        daily = ev.reindex(dates, method="ffill")
        last = pd.Series(ev.index, index=ev.index).reindex(dates, method="ffill")
        daily[(dates - last) > pd.Timedelta(days=90)] = np.nan
        pead[t] = daily

    ranks = {"mom": mom.rank(axis=1), "rev": rev.rank(axis=1), "pead": pead.rank(axis=1)}
    combo = pd.concat(ranks.values()).groupby(level=0).mean()  # NaN-tolerant mean of ranks
    return piv, {"combo": combo, "mom": mom, "rev": rev, "pead": pead}


def ls_returns(piv: pd.DataFrame, sig: pd.DataFrame) -> pd.Series:
    fwd1 = piv.shift(-1) / piv - 1
    q_hi = sig.quantile(0.8, axis=1)
    q_lo = sig.quantile(0.2, axis=1)
    top = sig.ge(q_hi, axis=0)
    bot = sig.le(q_lo, axis=0)
    r = (fwd1.where(top).mean(axis=1) - fwd1.where(bot).mean(axis=1)) / 2
    return r.loc[START:].dropna()


def ru_conformal(losses: np.ndarray, beta=BETA, alpha=ALPHA, gamma=GAMMA):
    """Online RU-conformal CVaR controller. losses = per-unit-exposure loss stream
    (-r_LS); returns (lam_path, realized loss path lam*loss)."""
    rmin, rmax = np.quantile(losses[:60], 0.01), np.quantile(losses[:60], 0.99)
    c, lam, q = 0.0, 1.0, 0.0
    scale = max(rmax - rmin, 1e-6)
    lams, real = np.empty(len(losses)), np.empty(len(losses))
    for i, lraw in enumerate(losses):
        R = lam * lraw
        lams[i], real[i] = lam, R
        l_ru = c + max(R - c, 0.0) / (1 - beta)
        g = 1.0 - (1.0 / (1 - beta)) * (R > c)
        q += g * g
        c = np.clip(c - scale / (2 * np.sqrt(q)) * g, rmin, rmax)
        lam = np.clip(lam - gamma * (l_ru - alpha) / scale, 0.0, 1.0)
    return lams, real


def frac_kelly(r: pd.Series, frac=0.5, win=126) -> np.ndarray:
    mu = r.rolling(win).mean().shift(1)
    var = r.rolling(win).var().shift(1)
    return np.clip(frac * mu / var, 0.0, 1.0).fillna(1.0).to_numpy()


def stats(r: np.ndarray, name: str, beta=BETA) -> str:
    ann = r.mean() * 252
    vol = r.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else np.nan
    eq = np.cumprod(1 + r)
    mdd = (1 - eq / np.maximum.accumulate(eq)).max()
    loss = -r
    var = np.quantile(loss, beta)
    cvar = loss[loss >= var].mean()
    return (f"{name:<22}{ann:>+8.2%}{vol:>8.2%}{sharpe:>7.2f}{mdd:>8.2%}"
            f"{cvar:>+9.3%}")


def main() -> int:
    piv, sigs = build_signals()
    r_ls = ls_returns(piv, sigs["combo"])
    print(f"backtest {r_ls.index[0].date()} .. {r_ls.index[-1].date()}  ({len(r_ls)} days)\n")

    # 1) multi-year IC: daily cross-sectional Spearman(signal, raw 7d fwd) by year
    fwd7 = (piv.shift(-7) / piv - 1).loc[r_ls.index]
    print("multi-year daily cross-sectional IC vs RAW 7d fwd return "
          "(survivorship-biased universe — see caveats):")
    print(f"{'year':<6}" + "".join(f"{k:>8}" for k in ("mom", "rev", "pead", "combo")))
    for yr, idx in fwd7.groupby(fwd7.index.year).groups.items():
        row = f"{yr:<6}"
        for k in ("mom", "rev", "pead", "combo"):
            s = sigs[k].loc[idx]
            ics = s.rank(axis=1).corrwith(fwd7.loc[idx].rank(axis=1), axis=1)
            row += f"{ics.mean():>+8.3f}"
        print(row)

    # 2) exposure arms on the combo LS stream
    r = r_ls.to_numpy()
    lam_ru, _ = ru_conformal(-r)
    lam_fk = frac_kelly(r_ls)
    print(f"\nexposure arms on combo LS (beta={BETA}, target daily CVaR alpha={ALPHA:.3%}):")
    print(f"{'arm':<22}{'ann.ret':>8}{'vol':>8}{'Sharpe':>7}{'maxDD':>8}{'CVaR.85':>9}")
    print("-" * 62)
    print(stats(r, "static lam=1"))
    print(stats(0.5 * r, "static lam=0.5"))
    print(stats(lam_fk * r, "fractional-Kelly 0.5"))
    print(stats(lam_ru * r, "RU-conformal CVaR"))

    # 3) did control hold in the stress windows?
    lam_s = pd.Series(lam_ru, index=r_ls.index)
    print("\nmean lambda_t (RU) in stress windows vs calm:")
    for name, sl in [("2018Q4", slice("2018-10", "2018-12")),
                     ("2020 COVID", slice("2020-02", "2020-04")),
                     ("2022 bear", slice("2022-01", "2022-12")),
                     ("2025Q1", slice("2025-01", "2025-03")),
                     ("full period", slice(None))]:
        seg, rseg = lam_s.loc[sl], r_ls.loc[sl]
        loss = -(lam_s.loc[sl].to_numpy() * rseg.to_numpy())
        var = np.quantile(loss, BETA)
        print(f"  {name:<12} mean lam {seg.mean():.2f} | realized CVaR.85 "
              f"{loss[loss >= var].mean():+.3%} (target {ALPHA:.3%})")

    print("\ncaveats: today's-top-150 universe (survivorship inflates early-year alpha; "
          "CVaR-control claim unaffected), zero costs, same-close execution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
