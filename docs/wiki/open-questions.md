---
title: "Open questions — all gated, none free"
status: open
updated: 2026-07-18
---

# Open questions

Free-data work is exhausted ([signals-tested](signals-tested.md)). Everything below is gated on
a purchase or a scope change; none is blocked on analysis.

| # | question | gate | expectation |
|--|--|--|--|
| 1 | Does a *defensible* large-cap arm exist? (PIT top-150-by-trailing-cap rebuild) | **Norgate US Platinum** (~$346.50/6mo — delisted prices + PIT membership) | Low — PIT membership alone already halved the Sharpe (arm B = 0.37); honest use is publishing the exact haircut |
| 2 | Is the MOC-execution alpha capturable in practice? | intraday data | Moot unless #1 revives an alpha claim ([transaction-costs](transaction-costs.md)) |
| 3 | Do estimate-revision signals work with real PIT history? | I/B/E/S-class data (WRDS or premium estimate-history feed) | Lit says IC 0.02–0.04 in institutional universes — the one untested-not-refuted signal |
| 4 | Do link/attention signals work where coverage is sparse? | universe change to small/micro-caps (data + philosophy change) | Effectively a new project |
| 5 | Regime-weighted conformal promotion | true OOS protocol (tune pre-2020 → test post-2022); no purchase | Only worth doing if the burst overshoot matters for a live use of the [controller](cvar-exposure-controller.md) |

**Not open** (settled; don't reopen without new data): anything in the closed sections of
[signals-tested](signals-tested.md); the text lever (both routes null); triple-barrier;
modelling-side improvements generally ([information-ceiling](information-ceiling.md)).
