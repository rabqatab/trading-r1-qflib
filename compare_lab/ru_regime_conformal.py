"""Regime-weighted RU-conformal CVaR control — does arXiv:2602.03903-style
regime weighting fix the 2020 burst overshoot of the vanilla online controller
(arXiv:2606.00320, compare_lab.cvar_conformal_backtest.ru_conformal)?

Paper mechanism (RWC, adapted): wrap a base quantile forecaster q_hat_t with an
additive conformal buffer c_hat_t = weighted quantile of past one-sided
conformity scores s_i = R_i - q_hat_i, weights
    w_i(t) = exp(-lam_dec*(t-i)) * exp(-||z_i - z_t||^2 / (2 h^2))
(time decay x Gaussian regime-similarity kernel), regime features
z_t = (RV21, MAR5) of the loss stream through t-1, standardized; quantile level
rho_t = min(1, beta*(1 + 1/W_t)); ESS safeguard n_eff < n_min -> drop kernel
(time-decay only). Bound U_t = q_hat_t + c_hat_t.

Honest adaptation to our ONLINE controller (deviations documented):
  * base forecaster = the vanilla AdaGrad RU-threshold c_t (its own subgradient
    update is left untouched, indicator vs its own c);
  * the regime-weighted buffer corrects the threshold used in the RU loss:
    c_eff = c_t + c_hat_t,  l_RU = c_eff + (R-c_eff)_+/(1-beta);
    outer lambda update identical to vanilla (same gamma, alpha, scale).
  * paper standardizes z on pre-validation stats -> we use expanding PAST-ONLY
    mean/std (no lookahead); paper's calibration set -> trailing window of
    online scores. Warm-up (< WARMUP finite scores) falls back to buffer 0,
    i.e. exact vanilla behavior.
Channel that should fix bursts: entering a high-vol regime re-weights the score
history toward past high-vol days -> c_eff (and hence l_RU) jumps BEFORE large
losses realize -> lambda cut proactively instead of at AdaGrad O(1/sqrt(T)) lag.

Streams: (i) combo LS gross, (ii) band 20/35 net @3bps + borrow,
(iii) PIT arm-C banded net (from pit_bounding_backtest construction).

    uv run python -m compare_lab.ru_regime_conformal
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from compare_lab.cvar_conformal_backtest import (
    ALPHA, BETA, GAMMA, START, build_signals, ls_returns, ru_conformal,
)
from compare_lab.cost_haircut_backtest import (
    BORROW_DAILY, band_weights, stat_row, turnover,
)

HEADLINE_BPS = 3
# Hyperparams: selected on a 27-point (window x half-life x h) grid on stream (i)
# (in-sample on the stress windows — disclosed), then cross-validated unchanged on
# streams (ii)/(iii). Weak localization (h=1, HL=120, w=500) is a near-null
# (2020H1 -0.02pp); strong (h=0.3) over-tightens calm windows to ~0.39%.
WINDOW = 250        # trailing calibration window m
HALF_LIFE = 60.0    # time-decay half-life (days) -> lam_dec = ln2/HL
BANDWIDTH = 0.5     # Gaussian kernel bandwidth h in standardized units
N_MIN = 15.0        # ESS floor -> fallback to time-decay-only weights
WARMUP = 60         # min finite scores before the buffer switches on

WINDOWS = [
    ("2018Q4", slice("2018-10", "2018-12")),
    ("2020H1", slice("2020-01", "2020-06")),
    ("2020 Feb-Apr", slice("2020-02", "2020-04")),
    ("2022", slice("2022-01", "2022-12")),
    ("2025Q1", slice("2025-01", "2025-03")),
    ("full", slice(None)),
]


# ------------------------------------------------------------ regime machinery
def regime_features(losses: np.ndarray) -> np.ndarray:
    """z_t = (21d realized vol, 5d mean abs return) of the RAW per-unit loss
    stream, computed through t-1, standardized by expanding past-only stats."""
    s = pd.Series(losses)
    z = pd.concat([s.rolling(21).std().shift(1),
                   s.abs().rolling(5).mean().shift(1)], axis=1)
    mu = z.expanding(30).mean().shift(1)
    sd = z.expanding(30).std().shift(1)
    return ((z - mu) / sd.replace(0, np.nan)).to_numpy()


def _weighted_quantile(scores: np.ndarray, w: np.ndarray, rho: float) -> float:
    order = np.argsort(scores)
    cw = np.cumsum(w[order])
    if cw[-1] <= 0:
        return 0.0
    cw /= cw[-1]
    k = min(int(np.searchsorted(cw, rho)), len(scores) - 1)
    return float(scores[order][k])


def ru_regime_conformal(losses: np.ndarray, beta=BETA, alpha=ALPHA, gamma=GAMMA,
                        window=WINDOW, half_life=HALF_LIFE, h=BANDWIDTH,
                        n_min=N_MIN, warmup=WARMUP):
    """Regime-weighted variant. Same interface/state as ru_conformal; only the
    threshold fed into l_RU is conformally corrected."""
    Z = regime_features(losses)
    rmin, rmax = np.quantile(losses[:60], 0.01), np.quantile(losses[:60], 0.99)
    scale = max(rmax - rmin, 1e-6)
    c, lam, q = 0.0, 1.0, 0.0
    decay = np.log(2.0) / half_life
    n = len(losses)
    lams, real, scores = np.empty(n), np.empty(n), np.empty(n)
    for i, lraw in enumerate(losses):
        R = lam * lraw
        lams[i], real[i] = lam, R
        buf = 0.0
        lo = max(0, i - window)
        if i - lo >= warmup and np.all(np.isfinite(Z[i])):
            zh = Z[lo:i]
            ok = np.all(np.isfinite(zh), axis=1)
            if ok.sum() >= warmup:
                ages = i - np.arange(lo, i)
                wt = np.exp(-decay * ages) * ok
                d2 = np.where(ok, ((zh - Z[i]) ** 2).sum(axis=1), np.inf)
                wk = wt * np.exp(-np.minimum(d2, 700.0) / (2 * h * h))
                W = wk.sum()
                sq = (wk ** 2).sum()
                ess = (W * W / sq) if sq > 0 else 0.0
                if ess < n_min:          # ESS safeguard: time-decay only
                    wk, W = wt, wt.sum()
                rho = min(1.0, beta * (1.0 + 1.0 / max(W, 1e-9)))
                buf = _weighted_quantile(scores[lo:i], wk, rho)
        c_eff = c + buf
        l_ru = c_eff + max(R - c_eff, 0.0) / (1 - beta)
        scores[i] = R - c                # conformity score vs BASE forecaster
        g = 1.0 - (1.0 / (1 - beta)) * (R > c)
        q += g * g
        c = np.clip(c - scale / (2 * np.sqrt(q)) * g, rmin, rmax)
        lam = np.clip(lam - gamma * (l_ru - alpha) / scale, 0.0, 1.0)
    return lams, real


# ------------------------------------------------------------ evaluation
def window_cvar(loss: pd.Series, beta=BETA) -> float:
    x = loss.to_numpy()
    var = np.quantile(x, beta)
    return float(x[x >= var].mean())


def evaluate(r: pd.Series, lam: np.ndarray) -> dict:
    ctrl = lam * r.to_numpy()
    st = stat_row(ctrl)
    lam_s = pd.Series(lam, index=r.index)
    loss_s = pd.Series(-ctrl, index=r.index)
    rows = {name: (window_cvar(loss_s.loc[sl]), lam_s.loc[sl].mean())
            for name, sl in WINDOWS}
    return {"sharpe": st["sharpe"], "mdd": st["mdd"], "windows": rows}


def report_stream(name: str, r: pd.Series, alpha=ALPHA) -> tuple[dict, dict]:
    losses = -r.to_numpy()
    lam_v, _ = ru_conformal(losses)
    lam_g, _ = ru_regime_conformal(losses)
    ev_v, ev_g = evaluate(r, lam_v), evaluate(r, lam_g)
    print(f"\n=== {name}  ({r.index[0].date()} .. {r.index[-1].date()}, "
          f"{len(r)} days; target CVaR.{int(BETA*100)} = {alpha:.3%}) ===")
    print(f"{'window':<14}{'vanilla CVaR':>14}{'regime CVaR':>14}"
          f"{'d(pp)':>8}{'lam_v':>7}{'lam_g':>7}")
    for w, _sl in WINDOWS:
        cv, lv = ev_v["windows"][w]
        cg, lg = ev_g["windows"][w]
        print(f"{w:<14}{cv:>13.3%}{cg:>13.3%}{(cg-cv)*100:>+8.3f}"
              f"{lv:>7.2f}{lg:>7.2f}")
    print(f"Sharpe  vanilla {ev_v['sharpe']:.2f}  regime {ev_g['sharpe']:.2f}  "
          f"(d {ev_g['sharpe']-ev_v['sharpe']:+.2f})   "
          f"maxDD  vanilla {ev_v['mdd']:.2%}  regime {ev_g['mdd']:.2%}")
    return ev_v, ev_g


# ------------------------------------------------------------ stream builders
def stream_banded_net(piv, combo, idx) -> pd.Series:
    pct = combo.rank(axis=1, pct=True)
    W = band_weights(pct)
    fwd1 = piv.shift(-1) / piv - 1
    g = (W * fwd1).sum(axis=1).loc[idx]
    to = turnover(W, piv).loc[idx]
    return g - to * HEADLINE_BPS * 1e-4 - BORROW_DAILY


def stream_pit_c() -> pd.Series | None:
    """PIT arm-C banded net stream, rebuilt via pit_bounding_backtest pieces."""
    try:
        from pathlib import Path
        from compare_lab.pit_bounding_backtest import (
            _YF, combo_from_raw, membership_mask, raw_signals,
        )
        piv150, _ = build_signals()
        yf = pd.read_parquet(_YF)
        yf["date"] = pd.to_datetime(yf["date"])
        piv_x = yf.pivot(index="date", columns="ticker", values="Close").sort_index()
        piv_x = piv_x.reindex(piv150.index)
        dup = piv_x.columns.intersection(piv150.columns)
        if len(dup):
            piv_x = piv_x.drop(columns=dup)
        piv_full = pd.concat([piv150, piv_x], axis=1).sort_index(axis=1)
        elig = membership_mask(piv_full.index, piv_full.columns) & piv_full.notna()
        combo_C = combo_from_raw(raw_signals(piv_full), elig)
        piv_C = piv_full.where(elig)
        idx = ls_returns(piv_C, combo_C).index
        pctC = combo_C.rank(axis=1, pct=True)
        W = band_weights(pctC)
        fwd1 = piv_C.shift(-1) / piv_C - 1
        g = (W * fwd1.fillna(0.0)).sum(axis=1).loc[idx]
        to = turnover(W, piv_full).loc[idx]
        return g - to * HEADLINE_BPS * 1e-4 - BORROW_DAILY
    except Exception as e:  # data missing etc. -> skip with note
        print(f"\n[PIT arm-C stream skipped: {type(e).__name__}: {e}]")
        return None


# ------------------------------------------------------------ main
def main() -> int:
    piv, sigs = build_signals()
    r_gross = ls_returns(piv, sigs["combo"])

    # sanity: vanilla path reproduces the known full-period realized CVaR 0.487%
    lam_v, _ = ru_conformal(-r_gross.to_numpy())
    loss_v = -(lam_v * r_gross.to_numpy())
    cv_full = window_cvar(pd.Series(loss_v, index=r_gross.index))
    assert abs(cv_full - 0.00487) < 1e-4, \
        f"vanilla full-period CVaR {cv_full:.4%} != known 0.487% (+-0.01pp)"
    print(f"[sanity OK] vanilla combo-gross full-period realized CVaR.85 "
          f"{cv_full:.3%} ~ 0.487%")
    print(f"regime variant hyperparams: window={WINDOW}, half-life={HALF_LIFE:.0f}d, "
          f"h={BANDWIDTH}, ESS floor={N_MIN:.0f}, warmup={WARMUP}")

    # (i) combo LS gross
    report_stream("stream (i): combo LS gross", r_gross)

    # (ii) band 20/35 net @3bps
    r_net = stream_banded_net(piv, sigs["combo"], r_gross.index)
    report_stream("stream (ii): band 20/35 net @3bps + borrow", r_net)

    # (iii) PIT arm-C banded net
    r_pit = stream_pit_c()
    if r_pit is not None:
        report_stream("stream (iii): PIT arm-C band 20/35 net @3bps", r_pit)

    # gamma x alpha mini-sweep, regime variant, stream (i)
    print("\n=== 3x3 gamma x alpha sweep, REGIME variant on stream (i) ===")
    print(f"{'gamma':>6}{'alpha':>8}{'fullCVaR':>10}{'ratio':>7}"
          f"{'2020H1':>9}{'Sharpe':>8}{'mean lam':>9}")
    losses = -r_gross.to_numpy()
    for g_ in (0.025, 0.05, 0.10):
        for a_ in (0.0025, 0.005, 0.010):
            lam, _ = ru_regime_conformal(losses, alpha=a_, gamma=g_)
            ls = pd.Series(-(lam * r_gross.to_numpy()), index=r_gross.index)
            full = window_cvar(ls)
            h1 = window_cvar(ls.loc["2020-01":"2020-06"])
            sh = stat_row(lam * r_gross.to_numpy())["sharpe"]
            print(f"{g_:>6.3f}{a_:>8.4f}{full:>9.3%}{full/a_:>7.2f}"
                  f"{h1:>8.3%}{sh:>8.2f}{lam.mean():>9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
