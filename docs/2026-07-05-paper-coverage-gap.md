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
- **Pending:** student SFT (sparkq) on the reject-sampled label-first corpus → IC (proxy + raw) vs
  base 0.205 / v3.1 0.228→raw 0.025 / template 0.163.

### Track B — #5 article text via Finnhub `summary` (the only ceiling-relevant lever)
`compare_lab/fetch_finnhub_news.py`. Our shipped news is Google-News-RSS **headlines only**, and those
URLs are unresolvable `CBMi` redirect blobs (probe: 0/10 resolved), so we can't recover bodies from
them. Finnhub `/company-news` returns a **`summary`** (98 % non-empty, median 149 chars vs 67 for
headlines, up to ~1.5k) with real content — the paper's actual news source.
- **Note:** the provided Finnhub key is a **paid/premium** key (premium endpoints insider-sentiment /
  revenue-estimate / candles all return data), flagged for cost; approved to continue. Premium
  endpoints noted as a *future* enrichment lever.
- **Pending:** re-crawl 150 tk × 2024-11…2025-07 → `news_top150_summ.parquet` → render headline+summary
  → rebuild the eval prompts → **base-model prompt-only** IC (proxy + raw) on headline vs
  headline+summary, same 2025-H1 OOS. Decisive test of whether richer text raises I(X;Y). Expectation
  modest (LLM text under-extraction; prior headline null), but this is the first experiment that could
  genuinely *move* the ceiling rather than just approach it.
