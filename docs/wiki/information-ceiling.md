---
title: The prediction ceiling is information-bound at raw IC ≈ 0.06
status: established
evidence: [2026-07-03-why-the-ceiling.md, 2026-07-07-final-synthesis.md, 2026-07-18-remaining-items-results.md]
updated: 2026-07-18
---

# Information ceiling

**Claim.** On 150 US large caps at a ~7-day horizon with price + public data, the tradeable
(raw-return) IC ceiling is ≈ 0.06. It is a property of the *input* (I(X;Y), measured 0.034 nats
via KSG), not of any model — the data-processing inequality makes it unbeatable by modelling.

**Evidence, three independent legs:**
1. *Measurement*: MI between inputs and raw 7d returns → ceiling ≈ 0.06; matches Gu-Kelly-Xiu
   best-ML (ρ≈0.063) ([why-the-ceiling](../2026-07-03-why-the-ceiling.md)).
2. *Convergence*: a 1-feature linear momentum model (+0.064), a regression GBM (+0.063), and the
   best LLM (Opus-4.8-distilled reject-sampled SFT, +0.053) land within noise of each other.
3. *Exhaustion*: every 2023–26 literature lever returned null — rank losses, label-horizon
   tuning, text embeddings (real embeddings ≤ shuffled placebo), RL variants (GRPO reweights
   latent ability, arXiv:2510.15990) ([results](../2026-07-18-remaining-items-results.md)).

**Corollaries.** The apparent IC ≈ 0.21–0.24 was a smoothing-inflated proxy artifact
([measurement-traps](measurement-traps.md) #1). Quantile-cut tail labels are additionally
un-learnable by construction (Opus blind-match 28%). Even the 0.06 is partly regime/universe —
see [universe-selection-artifact](universe-selection-artifact.md).

**What would move it:** more information, not more model — PIT estimate history, intraday data,
sparse-coverage universes ([open-questions](open-questions.md)).
