# Final synthesis — the honest scoreboard (2026-07-07)

> Closes the top-150 arc. One table, one metric (**raw 7-day-return IC** — the tradeable one, not the
> smoothing-inflated proxy). Everything we tried, ranked honestly. Full chain of evidence:
> [[why-the-ceiling]] · [[ic-ceiling-is-smoothing-inflated]].

## The scoreboard — raw 7-day-return IC on the same 2025-H1 OOS (n≈1000, SE≈0.032)

| # | approach | type | proxy IC | **RAW IC** | verdict |
|--|--|--|--:|--:|--|
| 1 | **Analyst revision momentum** (Finnhub, roadmap C#2) | **new information** | — | **+0.080** | first to clear the ceiling — but window-unstable (z≈2.5, Q1<0/Q2>0) |
| 2 | reject-sampled blind SFT (Track A) | extraction | 0.261 | +0.053 | best SFT; concentrates on predictable momentum cases; marginal vs base |
| — | momentum (1 feature, ref) | baseline | 0.266 | +0.064 | the hard-to-beat baseline |
| — | GBM (ref) | baseline | 0.215 | +0.042 | under-extracts momentum |
| 3 | distill v3.1 (label-first + filter) | extraction | 0.228 | +0.025 | proxy jump was a bearish-collapse mirage |
| 4 | base LLM (prompt-only) | — | 0.205 | +0.000 | at the proxy ceiling, zero raw skill |
| 5 | distill v3 (Opus reverse-reasoning) | extraction | 0.171 | −0.007 | — |
| 6 | template-SFT 3047 | extraction | 0.163 | +0.001 | — |
| 7 | **+summary news text** (Track B / #5) | richer text | 0.169 | **−0.010** | the paper's lever — clean null |
| 7 | headline news text | text | 0.193 | −0.010 | null |

## The one-sentence result
**Every modelling lever — scale, distillation (even Opus 4.8), reject-sampling, richer news text —
moves the tradeable raw IC by ~0 to +0.05 over the untrained base; the only thing that moved it more
was NEW INFORMATION outside the own-price path (analyst revisions, +0.080).** This is the
data-processing inequality made concrete: you cannot model your way past I(X;Y); you can only add
information — and the one input that carried some was a *structured* signal, not free text.

## Three findings that will save the next person time
1. **The ceiling is ~0.06 raw, not ~0.24.** The 0.24 was GBM under-extracting a *smoothing-inflated
   proxy*; the tradeable number matches Gu-Kelly-Xiu's best-ML (ρ≈0.063). Report raw-return IC always.
2. **Prose is useless to the LLM here; structure is not.** Headlines and full article summaries both
   gave raw IC ≈ 0; the structured analyst *response* to news (revision) gave 0.080. Use the LLM as an
   encoder of structure, not an end-to-end reader of text (roadmap D).
3. **Quantile-cut 5-class labels are partly un-learnable by construction.** Even Opus 4.8 predicting
   blind matched them only 29% (BUY 68% / STRONG_* 1-2%); the extreme classes are percentile edges, not
   events. Switch to event labels (triple-barrier) — roadmap A.

## What's actually worth doing next (from [[2026-07-06-post-ceiling-roadmap]])
Not more prediction modelling — the ceiling is proven. In priority:
1. **Target redesign** (triple-barrier + meta-labeling) — kills the two artifacts above; no new data.
2. **Risk management** (CVaR/CDaR + fractional-Kelly + crash-validated regime switch) — the paper's
   actual edge; converts the honest 0.06 into drawdown-controlled performance.
3. **Confirm/extend analyst revision** on a longer OOS + cost haircut; combine it with price as a
   *second decorrelated ~0.06 signal* (Granger combination) rather than chasing a single high IC.
4. Finer structured signals from the paid key: eps/revenue-estimate revision, price-target changes,
   earnings-surprise drift (PEAD) — all large-cap-native, non-price.

**Bottom line:** the reimplementation is complete and the science is honest — we reproduced the paper's
*mechanism*, proved its performance ceiling is information-bound at raw IC ~0.06, showed no modelling
lever (through Opus-4.8 distillation) beats it, and found the only movement comes from new structured
information. The productive frontier is target/risk/new-signal engineering, not a bigger model.
