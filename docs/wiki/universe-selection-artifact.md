---
title: The residual decade alpha was universe selection, not signal
status: established
evidence: [2026-07-18-pit-bounding-test.md]
updated: 2026-07-18
---

# Universe-selection artifact

**Claim.** The combo signal's apparent alpha was created by evaluating on "today's top-150 by
size" — a look-ahead winner filter. On an honest point-in-time universe the alpha is gone.

**The descent** (each row is one honesty upgrade):

| evaluation | combo IC | LS Sharpe (gross) |
|--|--:|--:|
| 2025-H1 snapshot | +0.096 | — |
| decade OOS 2017–2026 | +0.012 | 0.72 |
| ∩ PIT S&P membership | — | 0.37 |
| full PIT cross-section (~472 names/day) | **+0.004** | **−0.27** |

**Why it is NOT a delisting artifact:** the gap persists in 2023–26 at 96–99% data coverage
(0.43 vs 1.09), and the 81 unrecoverable delisted names bound the effect at ±0.1 Sharpe
(bootstrap). Two distinct biases stack: PIT *membership* alone halves the Sharpe (classic
survivorship), and expanding to the honest opportunity set kills it (the signal only ranks
within the winner set). Dollar-neutrality does not cancel either.

**Consequences.** Any current-constituent large-cap backtest result in this repo (and elsewhere)
should be presumed inflated until PIT-checked. Norgate (~$58/mo) is needed only to *defend* a PIT
top-150-by-cap variant, not for this verdict. Evidence:
[PIT bounding test](../2026-07-18-pit-bounding-test.md). See also
[measurement-traps](measurement-traps.md) #3.
