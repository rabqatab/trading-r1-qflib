# You Can't Model Your Way Past the Information Bound — But You Can Control the Tail

**An honest reimplementation of Trading-R1 (Xiao et al., arXiv:2509.11420), and what survived it.**
Working period 2026-06-21 → 2026-07-18. This document is the consolidated write-up of the full
arc; every claim links to a dated doc and a runnable script in `compare_lab/`.

## Abstract

We reimplemented the Trading-R1 pipeline (Qwen3-4B, SFT → GRPO, 5-class trading signals,
leak-safe backtests) on 150 US large caps and then subjected every layer of it to increasingly
honest evaluation. Three findings. **(1) The prediction ceiling is information-bound, not
model-bound**: the apparent IC ≈ 0.21 is a smoothing-inflated proxy artifact; against raw 7-day
returns the ceiling is ≈ 0.06, matching measured mutual information (0.034 nats) and the
Gu-Kelly-Xiu best-ML benchmark (ρ≈0.063). A 1-feature linear momentum model, a GBM, and our best
LLM (Opus-4.8-distilled, reject-sampled) all converge there; every modelling lever recommended by
the 2023–26 literature — rank losses, label-horizon tuning, text embeddings, curriculum-adjacent
ideas, RL variants — returned null. **(2) The residual alpha was universe selection**: new
structured information (analyst revisions +0.080, PEAD +0.068, 3-signal combo +0.096 on 2025-H1)
appeared to clear the ceiling, but the combo decays to +0.012 over the decade and to **+0.004 on a
point-in-time S&P universe**; the long-short Sharpe goes 0.72 → 0.37 (∩ PIT membership) → **−0.27**
(full PIT cross-section). The gap persists at 96–99% data coverage, so it is not a delisting
artifact: "today's top-150" is a look-ahead winner filter, and the signal only ranks within the
winner set. **(3) The risk layer is real and portable**: the RU-conformal online CVaR controller
(arXiv:2606.00320) pins realized daily CVaR to target on **8 qualitatively different loss streams**
with fixed hyperparameters — including net-of-cost streams (Sharpe 0.73 with the tail pinned at
0.469% vs a 0.5% target) and a negative-drift dead strategy — year-by-year through COVID and 2022,
never reducing Sharpe below the static baseline. The project's product is not a predictor; it is a
distribution-free exposure controller and a documented chain of measurement traps.

## 1. Setup

- **Task**: reimplement Trading-R1 — Qwen3-4B LoRA, SFT on distilled reasoning, GRPO with
  decision rewards, 5-class vol-adjusted labels, 5 input modalities, leak-safe qf-lib backtests
  (CR/Sharpe/hit-rate/MDD, three look-ahead gates). Universe: top-150 US large caps, prices
  2015→2026-05, ~weekly effective horizon.
- **Evaluation discipline** (in the end, the project's core contribution): every score reported
  against **raw 7-day forward returns** (not the training proxy), daily cross-sectional Spearman
  alongside pooled, multi-year OOS alongside single-window, PIT universe alongside current-universe,
  net-of-cost alongside gross.

## 2. Result I — the ceiling is information, not modelling

### 2.1 The proxy trap ([[2026-07-03-why-the-ceiling]])
The paper-style label (overlapping EMA-blended multi-horizon returns) inflates IC ≈ 4× via
overlapping-returns smoothing (Boudoukh-Richardson-Whitelaw). GBM's "ceiling" of 0.24 on the proxy
was under-extraction: 1-feature linear momentum reaches proxy 0.266. On raw returns the same
momentum collapses to **0.064** — and measured MI between our inputs and the raw target (KSG,
0.034 nats) puts the theoretical raw-IC ceiling at ≈ 0.06. The proxy is also gameable: distill
v3.1 "beat" base+GBM on proxy (0.228) while being ~0 raw — a bearish-collapse mirage.

### 2.2 Every modelling lever converges or fails (the extraction table)

| lever | raw 7d IC (2025-H1) | doc |
|--|--:|--|
| momentum, 1 linear feature (ref) | **+0.064** | — |
| regression GBM, expanding window | +0.063 | [[2026-07-18-remaining-items-results]] |
| reject-sampled blind SFT (best LLM) | +0.053 | [[2026-07-05-paper-coverage-gap]] |
| GBM (2024-trained, original) | +0.042 | — |
| rank-loss arms (h∈{3..15}) | +0.012..+0.038 | [[2026-07-18-remaining-items-results]] |
| distill v3.1 / v3 / template SFT | +0.025 / −0.007 / +0.001 | [[2026-07-07-final-synthesis]] |
| base LLM, prompt-only | +0.000 | — |
| news text (headline / +article summaries) | −0.010 / −0.010 | Track B |
| FinBERT embeddings + GBM head | +0.024 (< shuffled placebo +0.035; price-only +0.051) | [[2026-07-18-remaining-items-results]] |

When a 1-feature linear model, a GBM, and an Opus-4.8-distilled 4B land within noise of each
other — and real embeddings underperform noise embeddings — the bottleneck is the input, not the
model. This is the data-processing inequality made operational; RL cannot help (GRPO reweights
latent ability, arXiv:2510.15990), and our GRPO runs confirmed it. Un-learnability has a second
source: quantile-cut tail classes are percentile edges, not events — Opus 4.8 predicting blind
matches the label only 28–29%.

## 3. Result II — the descent of the alpha

New **structured** information did move raw IC — the only thing that ever did:
analyst-recommendation revision +0.080, PEAD +0.068, and their rank-mean combo with momentum
**+0.096** (2025-H1, z≈3). Then each honesty upgrade removed most of what remained:

| evaluation | combo IC | LS Sharpe (gross) |
|--|--:|--:|
| 2025-H1 snapshot | +0.096 | — |
| decade OOS 2017–2026 (2223 days) | +0.012 | 0.72 |
| ∩ PIT S&P membership (arm B) | — | 0.37 |
| **full PIT still-listed cross-section (~472 names/day)** | **+0.004** | **−0.27** |

Diagnosis ([[2026-07-18-pit-bounding-test]]): not delisting bias — the 81 unrecoverable delisted
names bound the effect at ±0.1 Sharpe (bootstrap), and the A-vs-C gap persists in 2023-26 at
96–99% coverage (1.09 vs 0.43). **"Today's top-150 by size" is itself a look-ahead winner filter**,
and the combo signal is mega-cap-specific on top: it ranks within winners, not within the honest
opportunity set. Follow-up attacks on the alpha all returned null on free data: estimate-revision
breadth (untestable — Finnhub estimates are snapshots), broker-event drift (all information in
day 0: +2.1%/−2.4%, t≈15–20; CAR(+1..+5) insignificant), co-coverage momentum (structural: even
the bottom-coverage tercile of PIT S&P members has an 88–98%-dense broker graph), Hou big→small
lead-lag (size gradient absent in an all-S&P universe; a residual intra-industry leader-follower
effect of IC ≈ +0.01, t≈3–4, is the only stable positive on the PIT universe — too small to build
on). ([[2026-07-18-free-reprocessing-batch]])

## 4. Result III — the risk layer works, survives costs, and generalizes

The RU-conformal online CVaR controller (Chen-Shen-Deng-Lei, arXiv:2606.00320; ~30 lines:
AdaGrad-FTRL threshold + projected λ-descent, distribution-free O(T^{-1/2}) guarantee, no
stationarity assumption) applied as exposure control λ_t ∈ [0,1]:

- **On the gross combo LS** ([[2026-07-17-cvar-conformal-control]]): realized CVaR_0.85 0.487% vs
  0.500% target over 9 years, zero recalibration; Sharpe 0.72→0.92 (vol-managed effect), maxDD
  24.7%→9.2%; dominates fractional-Kelly on every risk metric; robust across the full 3×3 γ×α
  sweep. Mean λ auto-drops to 0.35–0.50 in 2018Q4/2020/2022/2025Q1 — regime de-risking for free.
- **Net of costs** ([[2026-07-18-remaining-items-results]]): buy/hold banding 20/35 halves
  turnover (0.171→0.089/day) at zero gross cost — the only mitigation that works (EMA smoothing
  and staggered rebalancing lag a fast signal). At 3bps one-way + 35bps/yr borrow: net 0.63,
  **controlled 0.73 with CVaR still pinned (0.469%)**. Caveat: one day of execution lag halves the
  net alpha (0.63→0.31) — MOC execution is load-bearing.
- **Generality** ([[2026-07-18-free-reprocessing-batch]]): 8 streams — gross LS, banded net,
  dead PIT strategy, long-only market, ~472-name PIT universe, mom-only LS, NVDA buy-and-hold,
  2×-levered — × 2 targets: **10/10 full-period pins (within ~7% relative), year-by-year pins**
  (2020: static 3.74% → 1.02%), Sharpe ≥ static in every cell, maxDD −3–10×.
- **Spec'd limits**: (a) crash-burst lag — sudden crashes overshoot transiently (2020H1 up to +37%
  relative on levered streams) while slow bears pin exactly; the literature's regime-weighted fix
  (arXiv:2602.03903) passes the burst gate only with an in-sample-tuned brittle configuration, so
  vanilla remains the headline; (b) tight target + high-vol stream ⇒ λ≈0.1 (a de-facto vol
  targeter); (c) it bounds the tail *mean*, not the max single day.

## 5. Traps for the next person (the reusable part)

1. **Report raw-return IC, never a smoothed proxy.** Overlapping/EMA labels inflate ~4× and are
   gameable by degenerate strategies.
2. **Pooled IC ≠ cross-sectional skill.** The pooled-1000-point protocol mixes time-series and
   cross-sectional variation: momentum's 2025-H1 pooled IC is +0.064 while its daily
   cross-sectional mean is −0.011. Report daily CS IC alongside.
3. **Universe selection can dwarf delisting bias.** A current-constituent large-cap universe is a
   look-ahead winner filter; PIT membership alone (still survivor-conditioned) halved our Sharpe,
   the honest cross-section reversed its sign. Bootstrap the residual holes to bound what you
   can't fetch.
4. **Same-close execution is a claim, not a detail.** One day of lag halved the net alpha.
5. **Quantile-cut tail labels are un-learnable by construction** — even a frontier teacher blind-
   matches 28%. Event labels or ranks don't fix it (they winsorize, they don't add information).
6. **Snapshot APIs cannot test point-in-time hypotheses.** Probe for as-of dates before building.
7. **A single-window OOS "breakthrough" is a regime until proven otherwise** — ours (0.096)
   decayed 24× under two honesty upgrades.

## 6. Conclusion

The paper's mechanism reproduces; its performance does not survive honest evaluation, and the
reasons are measurable: the prediction problem on watched mega-caps at a weekly horizon carries
≈0.06 raw IC of extractable information, none of it durable across regimes or universes. Modelling
harder cannot fix that — only more information can (PIT estimate history, intraday data, or a
genuinely sparse-coverage universe). What survives everything is the risk layer: one unmodified
online controller that turns *any* loss stream — profitable, costly, or dead — into a
tail-pinned one. **Modelling the conditional mean was bounded; modelling the tail was not.**

## Reproducibility

All experiments are single scripts under `compare_lab/` (`uv run python -m compare_lab.<name>`),
pure pandas/sklearn on CPU: `cvar_conformal_backtest` (ceiling + controller),
`cost_haircut_backtest`, `pit_bounding_backtest`, `cvar_control_generality`,
`rank_target_sweep`, `encoder_ablation`, `broker_event_study`, `cocoverage_pit_ic`,
`leadlag_bigsmall_ic`, `ru_regime_conformal`. Data (gitignored, all free-tier or already-licensed
Finnhub): PIT constituents, 635-ticker broker events (185k), 485-ticker prices/volume, earnings,
recs, sector profiles, FinBERT embeddings. Chain of dated docs: [[2026-07-03-why-the-ceiling]] →
[[2026-07-06-post-ceiling-roadmap]] → [[2026-07-07-final-synthesis]] →
[[2026-07-17-cvar-conformal-control]] → [[2026-07-17-remaining-items-lit-sweep]] →
[[2026-07-18-remaining-items-results]] → [[2026-07-18-pit-bounding-test]] →
[[2026-07-18-free-reprocessing-batch]].
