---
title: "RU-conformal CVaR exposure controller — the portable product"
status: established
evidence: [2026-07-17-cvar-conformal-control.md, 2026-07-18-free-reprocessing-batch.md, 2026-07-18-remaining-items-results.md]
code: compare_lab/cvar_conformal_backtest.py::ru_conformal
updated: 2026-07-18
---

# CVaR exposure controller

**Claim.** The RU-conformal online controller (Chen-Shen-Deng-Lei, arXiv:2606.00320) pins
realized daily CVaR_β of ANY loss stream to a chosen target via exposure λ_t ∈ [0,1] alone —
distribution-free, no stationarity assumption, zero recalibration. This is the project's one
portable artifact.

**Mechanism** (~30 lines): inner AdaGrad-FTRL threshold c_t (subgradient
g = 1 − 1{R>c}/(1−β)); outer projected descent λ ← clip(λ − γ(ℓ_RU − α), 0, 1). Guarantee:
empirical CVaR ≤ α + O(T^(−1/2)).

**Validation (8 streams, fixed hyperparameters β=0.85, γ=0.05):** gross combo LS (0.487% vs
0.500% target, Sharpe 0.72→0.92, maxDD 24.7→9.2%), banded net-of-cost (0.469%, Sharpe 0.73),
dead PIT strategy (0.364%), long-only market, ~472-name PIT universe, mom-only LS, NVDA
buy-and-hold, 2×-levered — **10/10 stream×target pins within ~7% relative, year-by-year**
(2020: static 3.74% → 1.02%), Sharpe ≥ static in every cell, maxDD −3–10×. Robust across the
3×3 γ×α sweep. Dominates fractional-Kelly on every risk metric. Mean λ auto-drops in stress
regimes (regime switching for free).

**Spec'd limits:** (1) crash-burst lag — sudden crashes overshoot transiently (2020H1 up to +37%
relative on levered streams); slow bears pin exactly. The regime-weighted fix (arXiv:2602.03903)
passes only with in-sample-tuned brittle config — **vanilla stays**; promote only after true OOS
(tune pre-2020 → test post-2022). (2) tight target + high-vol stream ⇒ λ≈0.1 (a de-facto vol
targeter). (3) bounds the tail *mean*, not the max single day.

**How to reuse:** feed any daily loss stream to `ru_conformal()`; pick β and α in daily-loss
units; α is a reliable dial. CPU-only pandas. Generality harness:
`compare_lab/cvar_control_generality.py`.
