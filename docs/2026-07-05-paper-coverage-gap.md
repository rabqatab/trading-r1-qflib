# Trading-R1 paper coverage — what we applied vs not, and the two gap-closing tracks (2026-07-05)

> Re-examined the original paper ([Xiao et al., arXiv:2509.11420](https://arxiv.org/abs/2509.11420),
> summary in [`trading-r1-paper-summary.md`](trading-r1-paper-summary.md)) against our
> reimplementation, then launched the two gaps worth closing. Read alongside
> [`why-the-ceiling.md`](2026-07-03-why-the-ceiling.md) — the ceiling result decides which gaps matter.

## Coverage matrix

### ✅ Fully applied
- 5-class volatility labels (Alg S1): H{3,7,15}, w{.3,.5,.2}, q{.03,.15,.53,.85}, EMA3, vol20 — exact.
- 5 modalities (news / technical / fundamentals / sentiment / macro) + PIT; news time-buckets (Table S1).
- GRPO (TRL); **decision reward** (asymmetric 5×5 matrix) + a graded dense variant.
- Reverse-reasoning distillation (local single-teacher, Opus 4.8) — core idea.
- Leak-safe backtest eval (IC / Sharpe / MDD). structure/evidence/decision reward **functions** all built.

### 🟡 Partially applied
| paper | ours | gap |
|--|--|--|
| Reverse-reasoning: multi-model (GPT-4.1 factor-decompose → nano elaborate → stitch) + reject sampling | single Opus, self-consistency filter (v3.1) | no multi-model stitching; reject sampling → **Track A** |
| Technical indicators Table S2 (~28) | **16** | missing Ichimoku(4), 50 EMA, ATR(5), Z-score(75), PVO, ADX/ADXR, VWMA, full KDJ |
| Structure + Evidence rewards **driving training** | functions exist, **unused** in the top-150 arc | our terse targets score ~0 on them |

### ❌ Not applied
1. **3-stage easy-to-hard curriculum** (STRUCTURE→CLAIMS→DECISION, each SFT→RFT→Augment interleaved) — the paper's central training architecture. We did flat SFT→GRPO.
2. **Reject-sampling self-distillation augmentation** between stages → **Track A** addresses the reject-sampling piece.
3. **Input ensembling/augmentation** (~20 variants per day×ticker via random subset + order shuffle → 100K). We use 1 variant.
4. **Full §8 XML output format** (5–7 sections, opinion-quote-source bullets, tables) — abandoned after v2's §8 regression; we went terse.
5. **Article bodies / full-text prompts** (15–23k tokens) — we are **headline-only** → **Track B**.

## The verdict that matters: which gaps could move the *real* ceiling?

Per [`why-the-ceiling.md`](2026-07-03-why-the-ceiling.md) the tradeable ceiling is **I(X;Y) ≈ 0.06**
(raw return), model-invariant (DPI). So:

- **Indicators (Table S2 gap): a non-lever.** Indicators are deterministic functions of price → by
  DPI they add **zero** information over price. Filling Table S2 cannot raise I(X;Y).
- **Curriculum / staged rewards / §8 format / ensembling / multi-model distill: extraction & format
  only.** They can at best approach the *proxy* ceiling or improve robustness — v3.1 proved this
  live (proxy 0.228 but raw 0.025). None move I(X;Y).
- **Article bodies (#5): the ONLY unapplied lever that could raise I(X;Y)** — real text carries
  information headlines don't. (Though Merrill/Tan show LLMs under-extract text, and our MM headline
  test was null, so the expectation is modest.)

⚠️ **Bull-window caveat.** The paper's headline numbers (NVDA Sharpe 1.88) are on a **2024-06–08 bull
holdout** — structurally the same long-bias artifact our graded-reward audit exposed. Reproducing the
full method may reproduce impressive backtest numbers **without** real skill; v3.1's proxy mirage
(0.228 → raw 0.025) is the preview.

## Gap-closing tracks (launched 2026-07-05) — results PENDING

### Track A — negative / reject sampling (paper §4.3a Investment Thesis Distillation)
`compare_lab/sft/distill_blind.py`. Opus predicts **blind** (not shown the label), writes a terse
thesis, commits its OWN call; keep theses where Opus's independent call == the true label
(reject-sampling positives, label-first format), store mismatches as negatives (hard cases for a
future DPO/contrastive signal). Cleaner than v3.1's reverse-reasoning (no label leaks into the
rationale).
- **Interim finding (strong, ceiling-relevant): Opus-agrees-label ≈ 28 %** — even Opus 4.8 predicting
  blind matches the noisy vol-adjusted label only ~28 % (below the all-HOLD best-const ~48 %). ⇒ ~72 %
  of the label is noise even the best teacher can't reconstruct — a *teacher-accuracy* angle on the
  small I(X;Y). Side effect: reject sampling keeps only ~28 % → a small (~800-ex) corpus (collapse-risk).
- **RESULT (2026-07-07):** reject-sampled corpus (806 train, BUY/SELL-heavy, STRONG_* dropped) →
  student SFT. **proxy IC 0.261 (best of any SFT), RAW 7d IC 0.053 (also best SFT — base 0.000,
  v3.1 0.025, template ~0).** Reject sampling kept only Opus's confidently-correct cases — mostly
  clear-momentum BUY/SELL (class dist BUY 594 / SELL 370, HOLD/STRONG ~0) — so the student
  concentrated on the *predictable* directional cases and got closest to momentum (0.064) on raw
  returns. But 0.053 is z≈1.7 vs base (marginal), and still **below the analyst-revision signal
  (0.080, new information)** → confirms the DPI hierarchy: extraction improvement < new information.
  The dropped STRONG_*/HOLD classes are the quantile-cut boundaries that aren't predictable events
  (roadmap A / López de Prado).

### Track B — #5 article text via Finnhub `summary` (the only ceiling-relevant lever)
`compare_lab/fetch_finnhub_news.py`. Our shipped news is Google-News-RSS **headlines only**, and those
URLs are unresolvable `CBMi` redirect blobs (probe: 0/10 resolved), so we can't recover bodies from
them. Finnhub `/company-news` returns a **`summary`** (98 % non-empty, median 149 chars vs 67 for
headlines, up to ~1.5k) with real content — the paper's actual news source.
- **Note:** the provided Finnhub key is a **paid/premium** key (premium endpoints insider-sentiment /
  revenue-estimate / candles all return data), flagged for cost; approved to continue. Premium
  endpoints noted as a *future* enrichment lever.
- Re-crawled 150 tk × 2024-11…2025-07 → `news_top150_summ.parquet` (150k rows, summary 95 %); built a
  clean ablation (same Finnhub news, headline-only vs +summary, ~2× length, base-model prompt-only, same
  2025-H1 OOS, max_length 8192 so the +summary variant isn't truncated).
- **RESULT — headline (2026-07-07):** proxy IC 0.193, **RAW 7d IC −0.010** (n=930). The base model on
  Finnhub headlines replicates the earlier Google-RSS-headline null — **headline text carries ~0 raw
  signal.** (+summary variant re-running as `ec49` after two max-runtime kills — base is slow, no early
  termination; result pending.)
- **Contrast that matters:** even if +summary comes back null, we already have a *positive* text-adjacent
  result from the same paid key — **analyst revision (raw IC 0.080, [[analyst-revision-signal]])** — a
  *structured* signal, not free text. Tentative synthesis: the value in the news feed is not the prose
  (LLM under-extracts it) but the *structured analyst response* to it.
