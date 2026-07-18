"""Generality test for ru_conformal() (arXiv:2606.00320 online RU CVaR exposure
controller): is the "loss-stream-agnostic risk pinning" claim real?

Five qualitatively different daily portfolio streams, all built from parquets we
already have (2017-07 .. 2026-05):
  1. long-only equal-weight top-150            ("our universe market")
  2. long-only equal-weight PIT S&P universe   (honest market proxy, ~470 names/day)
  3. mom10-only quintile LS                    (single-signal alpha stream)
  4. NVDA buy-and-hold                         (pathological single asset)
  5. 2x levered long-only top-150              (fatter tails stress case)

Each stream is run at two targets (alpha = 0.5% and 1.0% daily CVaR_0.85) and
compared with the uncontrolled static lam=1 baseline: realized CVaR full period
and in 2020H1 / 2022, ann ret / Sharpe / maxDD, mean lambda overall and in the
stress windows, plus a per-year pin check for stream 2 at alpha=1.0%.

    uv run python -m compare_lab.cvar_control_generality
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from compare_lab.cvar_conformal_backtest import (
    BETA, START, build_signals, ls_returns, ru_conformal,
)
from compare_lab.pit_bounding_backtest import membership_mask

_YF = Path("data/yf_prices_sp500/prices.parquet")
ALPHAS = (0.005, 0.010)
WINDOWS = {"2020H1": slice("2020-01", "2020-06"), "2022": slice("2022-01", "2022-12")}


def tail_cvar(r: np.ndarray, beta: float = BETA) -> float:
    loss = -np.asarray(r, dtype=float)
    var = np.quantile(loss, beta)
    return loss[loss >= var].mean()


def perf(r: np.ndarray) -> tuple[float, float, float]:
    ann, vol = r.mean() * 252, r.std() * np.sqrt(252)
    eq = np.cumprod(1 + r)
    mdd = (1 - eq / np.maximum.accumulate(eq)).max()
    return ann, ann / vol if vol > 0 else np.nan, mdd


def build_streams() -> tuple[dict[str, pd.Series], pd.Series]:
    piv150, sigs = build_signals()
    ret150 = piv150.pct_change(fill_method=None)

    # 1. long-only EW top-150
    s1 = ret150.mean(axis=1).loc[START:].dropna()

    # 2. long-only EW PIT S&P universe
    yf = pd.read_parquet(_YF)
    yf["date"] = pd.to_datetime(yf["date"])
    piv_x = yf.pivot(index="date", columns="ticker", values="Close").sort_index()
    piv_x = piv_x.reindex(piv150.index)
    piv_x = piv_x.drop(columns=piv_x.columns.intersection(piv150.columns))
    piv_full = pd.concat([piv150, piv_x], axis=1).sort_index(axis=1)
    elig = membership_mask(piv_full.index, piv_full.columns) & piv_full.notna()
    s2 = (piv_full.pct_change(fill_method=None).where(elig)
          .mean(axis=1)).loc[START:].dropna()

    # 3. mom10-only quintile LS
    s3 = ls_returns(piv150, sigs["mom"])

    # 4. NVDA buy-and-hold
    s4 = ret150["NVDA"].loc[START:].dropna()

    # 5. 2x levered long-only top-150
    s5 = (2.0 * s1).rename("lev2x")

    # combo LS gross (known-result reproduction, cheap: signals already built)
    combo = ls_returns(piv150, sigs["combo"])

    return {
        "1 EW top-150 long-only": s1,
        "2 EW PIT S&P long-only": s2,
        "3 mom10 quintile LS": s3,
        "4 NVDA buy-and-hold": s4,
        "5 EW top-150 2x lever": s5,
    }, combo


def run_arm(r: pd.Series, alpha: float) -> dict:
    lam, _ = ru_conformal(-r.to_numpy(), alpha=alpha)
    lam_s = pd.Series(lam, index=r.index)
    ctrl = lam_s * r
    out = {
        "cvar_full": tail_cvar(ctrl.to_numpy()),
        "lam_full": lam.mean(),
        "lam_at_lo": (lam <= 1e-9).mean(),
        "lam_at_hi": (lam >= 1 - 1e-9).mean(),
        "perf": perf(ctrl.to_numpy()),
        "worst_day": float(-ctrl.min()),
        "ctrl": ctrl,
        "lam_s": lam_s,
    }
    for w, sl in WINDOWS.items():
        out[f"cvar_{w}"] = tail_cvar(ctrl.loc[sl].to_numpy())
        out[f"lam_{w}"] = lam_s.loc[sl].mean()
    return out


def main() -> int:
    streams, combo = build_streams()

    # --- sanity asserts -------------------------------------------------
    vol1 = streams["1 EW top-150 long-only"].std() * np.sqrt(252)
    assert 0.10 < vol1 < 0.35, f"stream 1 ann vol {vol1:.1%} outside plausible band"
    lam_c, _ = ru_conformal(-combo.to_numpy())          # alpha=0.5% default
    cvar_c = tail_cvar((lam_c * combo.to_numpy()))
    assert 0.0040 < cvar_c < 0.0055, \
        f"combo-stream reproduction failed: realized CVaR {cvar_c:.4%} != ~0.487%"
    print(f"sanity: stream-1 ann vol {vol1:.1%}; combo LS + RU (alpha=0.5%) "
          f"realized CVaR {cvar_c:.3%} (known 0.487%)  [OK]\n")

    # --- generality table ----------------------------------------------
    hdr = (f"{'stream':<24}{'alpha':>7}{'CVaRfull':>9}{'CVaR20H1':>9}{'CVaR22':>8}"
           f"{'annret':>8}{'Sharpe':>7}{'maxDD':>7}{'lam':>6}{'lam20':>6}{'lam22':>6}"
           f"{'@0':>5}{'@1':>5}")
    print(hdr)
    print("-" * len(hdr))
    for name, r in streams.items():
        stat = r.to_numpy()
        ann0, sh0, dd0 = perf(stat)
        print(f"{name:<24}{'static':>7}"
              f"{tail_cvar(stat):>9.3%}"
              f"{tail_cvar(r.loc[WINDOWS['2020H1']].to_numpy()):>9.3%}"
              f"{tail_cvar(r.loc[WINDOWS['2022']].to_numpy()):>8.3%}"
              f"{ann0:>+8.1%}{sh0:>7.2f}{dd0:>7.1%}{1.0:>6.2f}{'':>6}{'':>6}")
        for alpha in ALPHAS:
            a = run_arm(r, alpha)
            ann, sh, dd = a["perf"]
            print(f"{'':<24}{alpha:>7.2%}"
                  f"{a['cvar_full']:>9.3%}{a['cvar_2020H1']:>9.3%}{a['cvar_2022']:>8.3%}"
                  f"{ann:>+8.1%}{sh:>7.2f}{dd:>7.1%}"
                  f"{a['lam_full']:>6.2f}{a['lam_2020H1']:>6.2f}{a['lam_2022']:>6.2f}"
                  f"{a['lam_at_lo']:>5.0%}{a['lam_at_hi']:>5.0%}")
        print()

    # --- per-year pin check: stream 2 at alpha=1.0% ---------------------
    r2 = streams["2 EW PIT S&P long-only"]
    a2 = run_arm(r2, 0.010)
    ctrl2, lam2 = a2["ctrl"], a2["lam_s"]
    print("per-year pin check — stream 2 (EW PIT S&P) at alpha=1.0%:")
    print(f"{'year':<6}{'ctrl CVaR':>10}{'static CVaR':>12}{'mean lam':>9}{'worst day':>10}")
    for yr, ii in ctrl2.groupby(ctrl2.index.year).groups.items():
        print(f"{yr:<6}{tail_cvar(ctrl2.loc[ii].to_numpy()):>10.3%}"
              f"{tail_cvar(r2.loc[ii].to_numpy()):>12.3%}"
              f"{lam2.loc[ii].mean():>9.2f}{-ctrl2.loc[ii].min():>10.2%}")

    print("\nnotes: CVaR_0.85 per window is the mean of the worst 15% of days in "
          "that window; static baseline lam=1; @0/@1 = share of days lambda pinned "
          "at the bounds; zero costs, same-close execution, survivorship in "
          "stream 1/3/4 alpha (irrelevant to the control claim).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
