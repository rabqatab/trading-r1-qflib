# RU-conformal online CVaR control on the combo signal — roadmap B, realized (2026-07-17)

> Co-author-recommended paper: **"Adversarially Robust Control of Conditional Value-at-Risk via
> Rockafellar-Uryasev Conformal Inference"** (Chen, Shen, Deng, Lei — arXiv:2606.00320). An online,
> distribution-free CVaR controller: inner AdaGrad step on the RU threshold c_t, outer step
> `λ ← clip(λ − γ(ℓ^RU − α))` on exposure. Guarantee: empirical CVaR_β ≤ α + O(T^(−1/2)) with **no
> stationarity assumption** — i.e. exactly the regime-shift problem our 2025-Q1 window exposed.
> Code: `compare_lab/cvar_conformal_backtest.py` (also backfilled `data/finnhub_earnings_full/`,
> 2016→2026, 150 tickers).

This experiment answers two open items at once: the pending **multi-year OOS validation** of the
3-signal combo, and **roadmap B** (CVaR + fractional-Kelly + regime — "convert the honest ~0.1 IC
into drawdown-controlled performance"). Setup: daily-rebalanced dollar-neutral LS (top/bottom
combo quintile, gross 1), 2017-07 → 2026-05 (2223 days), λ_t ∈ [0,1] = strategy exposure.

## Result 1 — multi-year OOS: the combo does NOT generalize backward (honest, important)

Daily cross-sectional IC vs raw 7d forward return, by year:

| year | mom | rev | pead | **combo** |
|--|--:|--:|--:|--:|
| 2017 | −0.032 | +0.016 | −0.010 | −0.012 |
| 2018 | −0.028 | +0.011 | +0.020 | −0.004 |
| 2019 | −0.018 | −0.005 | −0.009 | −0.019 |
| 2020 | +0.005 | +0.005 | +0.033 | +0.021 |
| 2021 | +0.006 | +0.018 | +0.006 | +0.008 |
| 2022 | +0.003 | +0.017 | +0.000 | +0.014 |
| 2023 | +0.013 | −0.012 | −0.011 | +0.001 |
| 2024 | +0.001 | +0.014 | +0.057 | **+0.034** |
| 2025 | +0.001 | +0.029 | +0.040 | **+0.031** |
| 2026 | +0.072 | −0.008 | +0.022 | **+0.047** |

- **The 2025-H1 snapshot 0.096 is not a decade-stable number.** Decade mean daily IC ≈ +0.012;
  the signal is ~0/negative 2017–2019 and consistently positive only from 2020, strongest 2024–2026.
  (Daily all-150 cross-sections also measure differently from the sparse snapshot grid — 2025 here
  is +0.031 daily vs 0.096 on the eval grid — but the temporal pattern is the finding.)
- Survivorship caveat cuts the *optimistic* way: today's-top-150 universe should inflate early-year
  longs, yet 2017–2019 is still ~0 → the weakness there is real, not an artifact.
- Still: LS Sharpe 0.72 over the full decade at zero modelling — the combo monetizes, modestly.

## Result 2 — the paper's controller works exactly as advertised

Exposure arms on the same LS stream (β=0.85, target daily CVaR α=0.5%):

| arm | ann.ret | vol | Sharpe | maxDD | realized CVaR.85 |
|--|--:|--:|--:|--:|--:|
| static λ=1 | +7.43% | 10.28% | 0.72 | 24.66% | 0.899% |
| static λ=0.5 | +3.72% | 5.14% | 0.72 | 13.03% | 0.450% |
| fractional-Kelly 0.5 (126d) | +6.06% | 8.33% | 0.73 | 19.33% | 0.741% |
| **RU-conformal CVaR** | +5.14% | 5.61% | **0.92** | **9.21%** | **0.487%** |

- **Tail-risk targeting is essentially exact**: realized 0.487% vs target 0.500% over 9 years with
  zero recalibration. Sensitivity sweep (γ ∈ {0.02,0.05,0.1} × α ∈ {0.4,0.5,0.6%}): realized CVaR
  lands within ±0.01pp of target in **all 9 cells**, Sharpe 0.84–0.92 in all cells → not
  parameter luck; α is a working dial.
- **Beats fractional-Kelly on every risk metric** — Kelly sizes on mean/variance and never sees the
  tail; the RU controller targets the tail directly.
- **Sharpe 0.72 → 0.92 for free**: deleveraging in high-vol regimes is the vol-managed-portfolio
  effect (Moreira-Muir 2017) falling out of the CVaR constraint.
- **Implicit regime detection** (roadmap B's third component, subsumed): mean λ = 0.72 in 2018Q4,
  0.45 in 2020 COVID, 0.50 through 2022, **0.35 in 2025Q1** vs 0.62 overall. Stress-window realized
  CVaR: 2022 0.485%, 2025Q1 0.553% — held near target; 2020 COVID overshot (0.766%) as the
  O(T^(−1/2)) theory predicts for a burst faster than the adaptation rate.

## Verdict

- The paper is **directly applicable and adopted**: it is the online version of roadmap B and
  replaces the CVaR-optimizer + regime-switch pair with one ~30-line controller with a guarantee.
- The project's two-layer conclusion is now complete: **prediction is information-bound (~0.06 raw,
  and the combo's 0.096 was partly a 2024–2026 regime), but risk is controllable** — the same weak
  signal delivers Sharpe 0.92 / maxDD 9.2% instead of 0.72 / 24.7% purely through exposure control.
- Caveats: zero transaction costs (daily LS rebalance is cost-heavy — needed before any live
  claim), same-close execution, survivorship-biased universe for alpha (not for control).

Chain: [[2026-07-06-post-ceiling-roadmap]] · [[2026-07-06-analyst-revision-signal]] ·
[[2026-07-07-final-synthesis]] · memory [[ic-ceiling-is-smoothing-inflated]].
