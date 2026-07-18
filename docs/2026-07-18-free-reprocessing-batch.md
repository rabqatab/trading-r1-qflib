# Free data-reprocessing batch — 5 parallel experiments (2026-07-18)

> Follow-up to [[2026-07-18-pit-bounding-test]]: everything still doable at zero cost, run in one
> parallel batch. **Score: 1 strong positive (controller portability), 1 marginal curiosity
> (industry leader-follower), 3 nulls (event drift, regime-conformal upgrade, co-coverage final).**
> The pattern holds: risk-side work keeps paying, alpha-side work keeps returning nulls.

## Scoreboard

| # | experiment | verdict | headline |
|--|--|--|--|
| 1 | Controller generality (5 new streams) | ✅ **strong** | 10/10 stream×target CVaR pins; Sharpe ≥ static in all cells; maxDD −3–10× |
| 2 | Broker-event drift (58k events) | ❌ null | all info in day 0 (+2.1%/−2.4%); CAR(+1..+5) insignificant both legs |
| 3 | Regime-weighted conformal (arXiv:2602.03903) | 🟡 not promoted | passes the gate only with in-sample-tuned, brittle config; **vanilla stays** |
| 4 | Big→small lead-lag (Hou 2007) on PIT | 🟡 curiosity | Hou size-gradient ABSENT; generic industry leader-follower resid IC ~+0.01, t≈3–4, 10/10 yrs |
| 5 | Co-coverage on the sparse PIT side | ❌ **final null** | bottom coverage tercile still 88–98% dense (~9 brokers/stock) — idea closed for S&P-type universes |

## 1. Controller generality — the portability claim is now proven ✅

`compare_lab/cvar_control_generality.py`. Five qualitatively different streams (EW top-150
long-only; EW PIT ~472-name honest market; mom10-only LS; NVDA buy-and-hold; 2×-levered EW) ×
two targets (α=0.5%/1.0% daily CVaR_0.85), fixed hyperparameters:

- **10/10 cells pinned** (0.467–0.495% vs 0.5%; 0.898–0.991% vs 1.0%; worst miss 6.6% relative,
  conservative side). **Per-year pin holds** on the PIT market stream: 2020 static 3.74% →
  controlled 1.02%; 2022 2.25% → 1.00%.
- Sharpe ≥ static in every cell (mom LS 0.38→0.59; NVDA maxDD 66%→7%). λ never pegs at 0; pegs at
  1 only when the target is loose (by design).
- Honest limits, now quantified: (a) **crash-burst lag** — 2020H1 overshoots on every stream
  (worst +37% relative on 2× lever); slow bears (2022) pin exactly; (b) tight target + high-vol
  stream ⇒ λ≈0.1 (a de-facto vol-targeting throttle); (c) CVaR ≠ max loss — single worst days
  reach 2.8–3.4% with the yearly pin intact.
- With the three earlier streams (gross/net/dead-PIT): **8 streams, one unmodified controller.**
  This is the project's portable product, now with a stated spec sheet.

## 2. Broker-event drift — null ❌

`compare_lab/broker_event_study.py`. 3,698 up / 3,306 down / 3,696 init events with prices
(2015→2026), AR vs EW-150, day-0 assignment handles the 19–20:00-ET after-close stamps.
- **Day 0 carries everything**: +2.10% / −2.43% (t≈15–20 clustered); notch≥2 doubles it (n≈70/leg).
- **Drift is null**: CAR(+1..+5) up +0.10% (t 1.1), down −0.20% (t −1.8). Joint spread +0.30%
  (t 2.11) is borderline but the naive overlay turns ~40% gross/day — ≈−5%/yr at 5bps, eating the
  +9.1% gross; recent-regime Sharpe 0.50 pre-cost ⇒ ~0 after cost.
- Decay: day-0 reaction GREW 2020-26 (markets react harder/faster); drift never was solid.
  Confirms: on large caps the broker event completes at announcement — nothing to harvest at t+1.

## 3. Regime-weighted conformal — technical pass, not promoted 🟡

`compare_lab/ru_regime_conformal.py` (paper mechanism: time-decay × regime-similarity weighted
conformal buffer on the RU threshold; adapted to our online controller, vanilla path reproduces
0.487% exactly).
- Tuned config passes the stated gate: 2020H1 CVaR 0.582→0.458 on the gross stream, full period
  −0.039pp, Sharpe +0.01/+0.04, maxDD −1.3pp; generalizes to the banded net stream (−0.075pp).
- **Why not promoted**: paper-faithful defaults ≈ null; the winning config was selected on a
  27-point grid INCLUDING the stress windows (in-sample), neighboring grid points swing 2020H1 by
  ±0.10pp, calm windows acquire a ~10% conservative bias, and the PIT stream slightly worsens.
- Standing: vanilla `ru_conformal()` remains the headline; the regime variant is an overlay
  candidate pending true OOS validation (tune pre-2020 → test post-2022).

## 4. Big→small lead-lag — Hou gradient absent; a small stable leftover 🟡

`compare_lab/leadlag_bigsmall_ic.py` (+ `fetch_finnhub_profiles.py` 635 sector profiles,
`fetch_yf_volume.py` 485-ticker volume; size proxy = 63d dollar volume; 42 coarse industries,
~29/day pass ≥5 members).
- Raw leader-past-return signals: null (t 1–2). **Residualized (⊥ mom10, own-5d): IC +0.008–0.014,
  t≈3–4, `diff` variant positive 10/10 years** — the only signal so far that is stably positive
  ON THE HONEST PIT UNIVERSE (where the combo sits at +0.004).
- **But the Hou-specific prediction fails**: big tercile ≥ small tercile — no attention gradient
  (expected: S&P has no true small caps). What survives is generic intra-industry leader-follower
  / weekly industry momentum: real (t 3–4), tiny (~0.01), 13% correlated with the old combo.
  A curiosity to note, not machinery to build on.

## 5. Co-coverage, sparse side — structurally closed ❌

`compare_lab/cocoverage_pit_ic.py` (+ extended `fetch_cocoverage_data.py`; backfill now 635/635
tickers, 185,075 events, 392 brokers; 12 empty files = true delistings).
- **The sparsity hypothesis is dead on arrival**: full-PIT graph density 86→97% (worse than the
  121/149 = 81% that killed the first test), and the bottom coverage tercile is just as dense
  (88–98%, median ~9 brokers) — one megabroker connects everything.
- ICs: full/bottom-tercile ≈ 0; directed (visible→neglected) +0.01 residualized but
  regime-concentrated (2021/2025) and inside noise. **Ali-Hirshleifer needs 1–3-analyst
  small/micro-caps; no S&P-type universe can reach that. Idea closed.**

## Standing after this batch

- **Risk layer**: the controller is a proven, spec'd, portable product (8 streams, 2 targets,
  year-by-year pins). Its one mechanistic weakness (burst lag) resists the literature's fix
  honestly attempted (§3) — that's a fair boundary, not a to-do.
- **Alpha layer**: two more nulls and one ~0.01 curiosity. Free-data alpha work on this universe
  is now exhausted in every direction the literature suggested.
- Data assets accumulated (all gitignored): 635-ticker upgrade/downgrade (185k events), 635 sector
  profiles, 485-ticker volume panel, supply-chain graph, FinBERT embeddings, PIT constituent
  history — sufficient to reproduce every table in this arc for a write-up.

Scripts: `cvar_control_generality.py`, `broker_event_study.py`, `ru_regime_conformal.py`,
`leadlag_bigsmall_ic.py`, `fetch_finnhub_profiles.py`, `fetch_yf_volume.py`,
`cocoverage_pit_ic.py`, `fetch_cocoverage_data.py` (extended).
