---
title: "Transaction costs — the gate passed, with one dependency"
status: established
evidence: [2026-07-18-remaining-items-results.md]
code: compare_lab/cost_haircut_backtest.py
updated: 2026-07-18
---

# Transaction costs

**Claim.** The daily LS strategy survives realistic large-cap costs, but only via buy/hold
banding — and half the net alpha depends on same-close (MOC) execution.

**Cost model** (lit-grounded): 2–3 bps one-way base / 5 bps stress (megacap effective
half-spreads 1–2 bps; Frazzini-Israel-Moskowitz JF 2018) + 35 bps/yr borrow on the short book.
Drag ≈ 252 × daily one-way turnover × cost.

**Mitigation ranking (measured, matches Novy-Marx-Velikov RFS 2016):**

| arm | turnover/day | Sharpe @3bps |
|--|--:|--:|
| unbuffered daily quintile | 0.171 | 0.59 |
| **band 20/35 (enter top/bot 20%, exit at 35%)** | **0.089** | **0.63** |
| EMA-smoothed ranks (3/5/10d) | 0.113–0.073 | 0.55–0.34 |
| Jegadeesh-Titman K=5 overlap | 0.083 | 0.28 |

Banding halves turnover at zero gross cost. EMA/overlap LOSE here: the signal is fast
(mom10/PEAD) so lagging it costs more alpha than it saves in fees.

**Headline:** banded net @3bps + [CVaR control](cvar-exposure-controller.md) = Sharpe 0.73,
maxDD 8.8%, CVaR still pinned (0.469% vs 0.5%).

**The dependency:** shifting execution to close t+1 drops net Sharpe 0.63 → 0.31. MOC execution
is load-bearing; verifying real-world capturability needs intraday data
([open-questions](open-questions.md)). Note the alpha itself is a
[universe artifact](universe-selection-artifact.md) — these numbers are conditional on the
top-150 universe framing.
