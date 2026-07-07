# Analyst estimate-revision signal (Finnhub paid key) — the first thing that touched raw returns (2026-07-06)

> Roadmap C#2 ([`post-ceiling-roadmap`](2026-07-06-post-ceiling-roadmap.md)): the literature said the
> real value of the paid Finnhub key is **analyst revision momentum** (large-cap-native, an information
> event, NOT a function of the own-price path → can escape the DPI ceiling), not insider (a small-cap
> trap). We tested it. Code: `compare_lab/analyst_revision_ic.py`.

## Signal
Finnhub `/stock/recommendation` gives a monthly analyst consensus per ticker. Consensus score =
`(SB·2 + B − S − SS·2) / total`. Two features, strictly point-in-time (only months with
`period <= as_of`), on the SAME 2025-H1 OOS grid (n=995):
- **level** = latest consensus score
- **revision** = score(latest) − score(~3 months prior)  ← *revision momentum*

## Result — the first feature with raw-return IC > 0.06
| feature | IC vs make_signal proxy | **IC vs RAW 7-day return** |
|--|--:|--:|
| level | −0.049 | −0.062 |
| **revision** | +0.085 | **+0.080** |

For the first time in this project a feature clears the raw-return ceiling (~0.06; momentum 0.064,
GBM 0.042). And `corr(revision, make_signal) = 0.085` — nearly **independent** of the price signal, i.e.
genuinely new information — exactly what DPI predicts for a non-own-price input. (`level` is negative —
high absolute rating ≠ future return; only the *change* carries signal, consistent with the
revision-momentum literature.)

## Honesty checks — promising but window-unstable (do NOT overclaim)
| check | result | verdict |
|--|--|--|
| permutation null (300×) | p = 0.010 | ✅ not chance |
| **temporal stability** | 2025-Q1 IC **−0.052** (n=526) / Q2 IC **+0.165** (n=469) | ⚠️ **unstable — the whole 0.080 lives in Q2** |
| quintile monotonicity | Q0..Q4 raw ret +0.47/+0.29/**+2.59**/+1.16/+1.74 %; long-short Q4−Q0 +1.27 % | ⚠️ non-monotone (Q2 peaks, not Q4) |

**Honest read:** revision momentum is permutation-significant but **not temporally stable** on our
2025-H1 slice — it was *negative* in Q1 and strongly positive in Q2, so the headline 0.080 may be a
single-half-year regime effect rather than durable alpha. The quintile ranking is non-monotone. This is
the same lesson as the graded-reward bull-window mirage: one window's impressive number can be a regime
artifact. That said, it is **clearly better than every prior signal** (all ~0 on raw returns) — DPI's
prediction that a non-own-price input *can* carry information is borne out; the magnitude and stability
are the open questions.

## What this changes / next
- This is the project's **first evidence that the ceiling is movable — by new information, not modelling**
  — consistent with the whole [[ic-ceiling-is-smoothing-inflated]] framework (the ceiling is I(X;Y); add
  an input outside the own-price path and I(X;Y) rises).
- ⚠️ Not yet a tradeable alpha: needs (1) a **longer OOS** (multiple years/regimes) to confirm stability —
  our 2025-H1 window is short; (2) **transaction-cost + multiple-testing haircut** (Harvey-Liu-Zhu t>3,
  McLean-Pontiff 58% post-publication decay); (3) test as an **incremental GBM feature** over price (does
  it add on top, or is even this partly price-correlated at longer horizons?).
- Cheap follow-ups on the same paid key, per the roadmap: `eps-estimate`/`revenue-estimate` revision
  (finer than recommendation buckets), `price-target` changes, earnings-surprise drift (PEAD is another
  large-cap-native, non-price signal). Each is a candidate incremental input.
- **Framing:** even if revision stabilizes at raw-IC ~0.05–0.08, that is still near the EMH floor — the
  win is not a high IC but **a second, decorrelated ~0.06 signal** to *combine* with price (Granger
  forecast-combination / √K), which is the honest route to a higher *effective* IC (roadmap B/§5).

## Follow-up (2026-07-08): PEAD works too, and the combination delivers — raw IC 0.096

`compare_lab/signal_combine_ic.py`, Finnhub `/calendar/earnings` (announcement **date** field →
fully PIT-safe post-earnings drift). Same 2025-H1 OOS grid:

| signal | RAW 7d IC | n | Q1 / Q2 |
|--|--:|--:|--|
| momentum (10d) | +0.064 | 1000 | −0.077 / +0.099 |
| revision (3mo consensus) | +0.080 | 995 | −0.052 / +0.165 |
| **PEAD (latest EPS surprise ≤90d)** | **+0.068** | 933 | **+0.008 / +0.089** (most stable) |
| mom+rev rank-mean | +0.090 | 1000 | |
| **all-3 rank-mean** | **+0.096** | 1000 | −0.084 / +0.164 |

- **Cross-correlations ~0** (mom-rev 0.080, mom-pead 0.078, **rev-pead −0.015**) → genuinely
  decorrelated → the simple equal-weight rank average lifts raw IC **0.064 → 0.096 (+50 % over the
  best single signal), with zero modelling** — the Granger/√K route working as the roadmap predicted.
  The highest honest number in the project (z≈3.0).
- **Mitigating context for the instability caveat:** the Q1/Q2 split shows **momentum itself was
  negative in 2025-Q1** (−0.077) — Q1 was a regime where *every* signal failed, so revision's Q1<0 is
  a regime effect shared with the oldest anomaly in finance, not evidence the signal is fake. (Still
  one window; a multi-year OOS remains the real test, and costs/multiple-testing are unhaircut.)
- PEAD is the **second** non-own-price signal to clear the ~0.06 ceiling, with the best temporal
  stability of the three — consistent with its status as one of the most replicated anomalies.
