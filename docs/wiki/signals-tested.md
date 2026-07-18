---
title: "Signals & levers ledger — everything tried, with verdicts"
status: ledger
updated: 2026-07-18
---

# Signals tested (the "don't retry" list)

Raw 7d IC unless noted. **Do not re-run anything marked closed without genuinely new data**
([open-questions](open-questions.md) lists what "new data" means per item).

## Extraction levers (all bounded by the [information-ceiling](information-ceiling.md))

| lever | best result | verdict | evidence |
|--|--|--|--|
| SFT ladder (v0→v2, template, distill v3/v3.1) | +0.025 (v3.1; proxy 0.228 was a mirage) | closed | [synthesis](../2026-07-07-final-synthesis.md) |
| Reject-sampled blind SFT (Track A) | **+0.053** (best LLM) | closed — still < momentum | [coverage-gap](../2026-07-05-paper-coverage-gap.md) |
| GRPO / graded rewards / RL variants | IC ~0.19 proxy; bull-window Sharpe is long-bias | closed | README findings |
| Rank-loss targets + label-horizon sweep | +0.038 < regression +0.063 | closed — ranking = winsorization | [results](../2026-07-18-remaining-items-results.md) |
| News text, end-to-end (headline / +summaries) | −0.010 / −0.010 | closed — clean null | Track B |
| FinBERT embeddings + GBM head (encoder route) | +0.024 < shuffled placebo +0.035 | closed — replicates published null | [results](../2026-07-18-remaining-items-results.md) |
| Triple-barrier / meta-labeling | not built | closed by lit — no replicated OOS evidence | [lit-sweep](../2026-07-17-remaining-items-lit-sweep.md) |

## New-information signals

| signal | snapshot IC | honest IC | verdict | evidence |
|--|--|--|--|--|
| Analyst rec-revision (63d) | +0.080 (2025-H1) | ~0 pre-2020 | regime + [universe artifact](universe-selection-artifact.md) | [banner'd doc](../2026-07-06-analyst-revision-signal.md) |
| PEAD (earnings surprise) | +0.068 | academic: dead in large caps since ~2006 | regime | [lit-sweep](../2026-07-17-remaining-items-lit-sweep.md) |
| 3-signal combo (mom+rev+PEAD) | +0.096 | **+0.004 on PIT universe** | closed — universe artifact | [PIT test](../2026-07-18-pit-bounding-test.md) |
| Estimate-revision breadth (eps/revenue) | — | untestable: Finnhub endpoints are snapshots, no as-of | blocked (needs I/B/E/S-class data) | [results](../2026-07-18-remaining-items-results.md) |
| Broker upgrade/downgrade, 63d factor | +0.003, corr 0.54 w/ rev | dilutes combo | closed | [results](../2026-07-18-remaining-items-results.md) |
| Broker events, day+1..+5 drift | CAR insignificant both legs; all info in day 0 | closed | [free batch](../2026-07-18-free-reprocessing-batch.md) |
| Co-coverage momentum (150 & PIT, incl. sparse tercile) | +0.003–0.004 | closed — graph 86–98% dense everywhere in S&P | [free batch](../2026-07-18-free-reprocessing-batch.md) |
| Supply-chain neighbor momentum | +0.009 (snapshot graph = look-ahead ceiling) | closed | [results](../2026-07-18-remaining-items-results.md) |
| Hou big→small lead-lag | size gradient absent | closed on S&P universes | [free batch](../2026-07-18-free-reprocessing-batch.md) |
| Intra-industry leader-follower (residualized) | **+0.008–0.014, t≈3–4, 10/10 yrs on PIT** | curiosity — the only stable positive on the honest universe; too small to build on | [free batch](../2026-07-18-free-reprocessing-batch.md) |

## Risk/portfolio levers (the ones that worked)

| lever | result | evidence |
|--|--|--|
| [CVaR exposure control](cvar-exposure-controller.md) | tail pinned, 8 streams | [doc](../2026-07-17-cvar-conformal-control.md) |
| Buy/hold banding 20/35 | turnover −48% at zero gross cost | [transaction-costs](transaction-costs.md) |
| Fractional-Kelly | dominated by CVaR control on every risk metric | [doc](../2026-07-17-cvar-conformal-control.md) |
| EMA smoothing / JT overlapping rebalance | lose — signal too fast | [transaction-costs](transaction-costs.md) |
| Regime-weighted conformal (arXiv:2602.03903) | passes burst gate only in-sample-tuned — not promoted | [free batch](../2026-07-18-free-reprocessing-batch.md) |
