---
title: "The 7 measurement traps (paid for; don't re-pay)"
status: established
updated: 2026-07-18
---

# Measurement traps

Each of these inflated a result in this project before being caught. Full context in the
[write-up §5](../2026-07-18-writeup-two-layer-result.md).

1. **Smoothed-proxy IC inflates ~4× and is gameable.** Overlapping/EMA-blended labels
   (Boudoukh-Richardson-Whitelaw). v3.1 "beat" base+GBM on proxy while ~0 raw. → Report
   raw-return IC always. ([information-ceiling](information-ceiling.md))
2. **Pooled IC ≠ cross-sectional skill.** Pooled-1000-pt protocol mixes time-series and
   cross-sectional variation: momentum 2025-H1 pooled +0.064 vs daily-CS mean −0.011. → Report
   daily CS IC alongside pooled.
3. **Universe selection can dwarf delisting bias.** Current-constituent large-cap universe =
   look-ahead winner filter; cost us Sharpe 0.72 → −0.27.
   ([universe-selection-artifact](universe-selection-artifact.md))
4. **Same-close execution is a claim, not a detail.** One day of execution lag halved the net
   alpha (0.63 → 0.31). ([transaction-costs](transaction-costs.md))
5. **Quantile-cut tail labels are un-learnable by construction.** Percentile edges are not
   events; a frontier teacher blind-matches only 28%. Rank labels don't fix it (they winsorize).
6. **Snapshot APIs cannot test point-in-time hypotheses.** Probe for as-of dates BEFORE building
   a backfill (Finnhub estimate endpoints burned this).
7. **A single-window OOS "breakthrough" is a regime until proven otherwise.** Our +0.096 decayed
   24× under two honesty upgrades.

Operational (environment, not methodology): detached background jobs launched from sandboxed
shells silently lose file writes — verify outputs on disk; long jobs via tracked runs or sparkq.
