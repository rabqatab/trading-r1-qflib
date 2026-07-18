# Final synthesis — the honest scoreboard (2026-07-07)

> Closes the top-150 arc. One table, one metric (**raw 7-day-return IC** — the tradeable one, not the
> smoothing-inflated proxy). Everything we tried, ranked honestly. Full chain of evidence:
> [[why-the-ceiling]] · [[ic-ceiling-is-smoothing-inflated]].

## The scoreboard — raw 7-day-return IC on the same 2025-H1 OOS (n≈1000, SE≈0.032)

| # | approach | type | proxy IC | **RAW IC** | verdict |
|--|--|--|--:|--:|--|
| 0 | **3-signal combo: momentum+revision+PEAD** (2026-07-08) | **combination of new info** | 0.217 | **+0.096** | best honest number (z≈3.0); zero modelling — just a rank average of ~decorrelated signals |
| 1 | **Analyst revision momentum** (Finnhub, roadmap C#2) | **new information** | — | **+0.080** | first to clear the ceiling — window-unstable, but momentum shares the same Q1<0 regime |
| 1b | **PEAD** (earnings-surprise drift, Finnhub calendar) | new information | — | +0.068 | second to clear it; most stable of the three (Q1 +0.008) |
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
1. ✅ **Target redesign** — CLOSED 2026-07-18, negative: rank targets/horizon sweep don't beat
   regression or momentum (ranking = winsorization; triple-barrier was demoted by the lit sweep).
   ([[2026-07-18-remaining-items-results]])
2. ✅ **Risk management** — DONE 2026-07-17 via the RU-conformal online CVaR controller
   ([[2026-07-17-cvar-conformal-control]]): Sharpe 0.72→0.92, maxDD 24.7→9.2%, CVaR pinned to target.
3. ✅ **Confirm/extend analyst revision + combine** — DONE 2026-07-08/17: combo (mom+rev+PEAD) 0.096 on
   2025-H1; multi-year check shows it's partly a 2024-26 regime (decade daily IC ~+0.01, ~0 pre-2020).
4. ✅ Finer structured signals — CLOSED 2026-07-18: eps/revenue-estimate + price-target endpoints
   are snapshots (no PIT history → untestable); broker up/down fallback redundant (dilutes combo).
   Encoder ablation (roadmap D) also closed — clean null. ([[2026-07-18-remaining-items-results]])

**Bottom line:** the reimplementation is complete and the science is honest — we reproduced the paper's
*mechanism*, proved its performance ceiling is information-bound at raw IC ~0.06, showed no modelling
lever (through Opus-4.8 distillation) beats it, and found the only movement comes from new structured
information. The productive frontier is target/risk/new-signal engineering, not a bigger model.

## Post-scriptum (2026-07-17): roadmap B executed — risk IS controllable
[[2026-07-17-cvar-conformal-control]]: the RU-conformal online CVaR controller (arXiv:2606.00320)
applied to the combo LS over 2017–2026 (2223 days). Two updates to this scoreboard's context:
1. **Multi-year OOS**: the combo's 0.096 is partly a 2024–2026 regime — decade mean daily IC ≈
   +0.012, ~0 in 2017–2019, consistently positive only from 2020. The honest prediction story
   stands, with a narrower window than the H1-2025 snapshot implied.
2. **Risk control works where prediction plateaus**: same signal, exposure-only control → Sharpe
   0.72 → **0.92**, maxDD 24.7% → **9.2%**, realized CVaR pinned to target (0.487% vs 0.500%)
   across 9 years incl. COVID/2022/2025Q1, robust across a 3×3 γ/α sweep. Fractional-Kelly is
   dominated on every risk metric. The paper's promised edge — drawdown control, not IC — is real.
