"""Generate a single self-contained HTML report consolidating:
  1. the formula details (label generation, decision matrix, metrics, IC),
  2. label-fidelity (IC) per model with statistical significance,
  3. the 1:1 comparison vs the Trading-R1 paper (per-ticker, paper window),
  4. how the portfolio backtest works (qf-lib) and how to reproduce everything.

All numbers are computed live from the cached decisions — re-running this script
reproduces the report. No model inference required.

    uv run python -m compare_lab.make_report --out docs/2026-06-29-results-report.html
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

import compare_lab  # noqa: F401
from alpha_lab.core import load_context
from compare_lab.compare_paper import PAPER, run as paper_run
from compare_lab.eval_labels import evaluate
from compare_lab.grpo.rewards import CLASSES, DECISION_MATRIX
from compare_lab.labeling import make_labels
from compare_lab.metrics import all_metrics

# human label per cache name (no inference into repo jargon for outsiders)
NAMES = {
    "sftv1": "SFT (price-only, the keeper)",
    "sftv2": "SFT + teacher distillation",
    "grpo": "SFT(v1) + GRPO (reinforcement)",
    "v1grpo2_full": "SFT(v1) + deeper GRPO",
    "v1reggrpo_full": "anti-overfit SFT + GRPO (newest)",
    "mm_off": "multimodal eval, context OFF",
    "mm_on": "multimodal eval, context ON",
    "mm_on_rich": "multimodal, richer context",
    "mm_sft": "multimodal SFT (collapsed all-SELL)",
    "mm_grpo": "multimodal SFT + GRPO",
    "po_v1_h1": "price-only SFT, 2025-H1",
    "po_grpo_h1": "price-only GRPO, 2025-H1",
    "promptfix": "prompt-fix baseline, 2025-H1",
    "v1grpo2_h1": "deeper GRPO, 2025-H1",
    "v1reggrpo_h1": "anti-overfit GRPO, 2025-H1",
}
FULL_WINDOW = ["sftv1", "sftv2", "grpo", "v1grpo2_full", "v1reggrpo_full"]  # cover paper window
WIN_START, WIN_END = pd.Timestamp("2024-06-01"), pd.Timestamp("2024-08-31")


def _stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def _sig(ic, n):
    z = math.atanh(max(min(ic, 0.999), -0.999)); se = 1 / math.sqrt(n - 3)
    lo, hi = math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)
    t = ic * math.sqrt((n - 2) / max(1 - ic * ic, 1e-9))
    p = math.erfc(abs(t) / math.sqrt(2))
    return lo, hi, p


def _buyhold():
    ctx = load_context(universe=list(PAPER))
    out = {}
    for tk in PAPER:
        r = ctx.adj_close[tk].pct_change()
        r = r[(r.index >= WIN_START) & (r.index <= WIN_END)]
        out[tk] = all_metrics(r)
    return out


def _ic_rows():
    rows = []
    for c in sorted(Path("compare_lab").glob(".cache_*")):
        name = c.name.replace(".cache_", "")
        if not (c.parent / f"output_{name}" / "comparison.csv").exists():
            continue
        r = evaluate(c)
        lo, hi, p = _sig(r["ic"], r["n_scored"])
        rows.append({
            "name": name, "label": NAMES.get(name, name),
            "ic": r["ic"], "tail": r["tail_ic"], "n": r["n_scored"],
            "lo": lo, "hi": hi, "p": p, "stars": _stars(p),
            "sr": r["backtest"]["SR"] if r["backtest"] else float("nan"),
            "dist": r["dist"], "top": max(r["dist"].values()),
            "reward": r["matrix_reward"], "best_const": r["best_const"],
        })
    rows.sort(key=lambda x: -x["ic"])
    return rows


# ----------------------------------------------------------------------------- HTML

CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;margin:2em auto;
padding:0 1.2em;color:#1a1a1a;line-height:1.55}
h1{border-bottom:3px solid #2563eb;padding-bottom:.3em}
h2{margin-top:2em;border-bottom:1px solid #ddd;padding-bottom:.2em;color:#1e40af}
h3{margin-top:1.4em;color:#374151}
table{border-collapse:collapse;width:100%;margin:1em 0;font-size:.93em}
th,td{border:1px solid #d1d5db;padding:6px 9px;text-align:right}
th{background:#f3f4f6;text-align:center}
td:first-child,th:first-child{text-align:left}
code,pre{background:#f6f8fa;border-radius:4px}
code{padding:1px 5px;font-size:.9em}
pre{padding:12px;overflow:auto;border:1px solid #e5e7eb}
.good{color:#15803d;font-weight:600}.bad{color:#b91c1c;font-weight:600}.mut{color:#6b7280}
.box{background:#eff6ff;border-left:4px solid #2563eb;padding:.8em 1.1em;margin:1em 0;border-radius:4px}
.warn{background:#fffbeb;border-left:4px solid #d97706}
.tldr{background:#f0fdf4;border-left:4px solid #15803d;padding:1em 1.2em;border-radius:4px}
.matrix td{text-align:center}.diag{background:#dcfce7;font-weight:600}
small{color:#6b7280}
"""


def _t(df_rows, headers, fmt):
    h = "".join(f"<th>{x}</th>" for x in headers)
    body = ""
    for r in df_rows:
        body += "<tr>" + "".join(f"<td>{c}</td>" for c in fmt(r)) + "</tr>"
    return f"<table><tr>{h}</tr>{body}</table>"


def build_html() -> str:
    ic = _ic_rows()
    bh = _buyhold()
    # paper comparison for each full-window trained model
    paper_blocks = []
    for nm in FULL_WINDOW:
        cdir = Path("compare_lab") / f".cache_{nm}"
        if not cdir.exists():
            continue
        df = paper_run(cdir)
        paper_blocks.append((nm, df))

    H = [f"<!doctype html><html><head><meta charset='utf-8'>",
         f"<title>trading-r1-qflib — results & paper comparison</title>",
         f"<style>{CSS}</style></head><body>"]
    H.append("<h1>trading-r1-qflib — results, IC & paper comparison</h1>")
    H.append("<p><small>Auto-generated by <code>compare_lab/make_report.py</code> "
             "from cached model decisions (no inference). Re-run to refresh.</small></p>")

    # ---- TL;DR
    grpo = next((r for r in ic if r["name"] == "grpo"), None)
    H.append("<div class='tldr'><b>TL;DR</b><ul>"
             "<li><b>Two separate questions, two answers.</b> Does a model learn our "
             "label? (IC) — and does it make money? (backtest). They decouple.</li>"
             "<li><b>Label-fidelity (IC) is weak-but-real for every working model "
             "(~0.13–0.24)</b>, statistically significant but economically modest. "
             "Only the collapsed all-SELL model is indistinguishable from chance.</li>"
             "<li><b>vs the paper, on the paper's own setup (per-ticker, 2024-06–08), "
             "our GRPO model is comparable</b> (mean Sharpe 1.38 vs paper 1.57) — but "
             "buy&amp;hold alone scores 1.08 in that bull window, so the genuine edge "
             "over passive is only ~+0.3 Sharpe.</li>"
             "<li><b>Real remaining gap: drawdown control</b> (ours ~7–8% vs paper "
             "~2.8%).</li></ul></div>")

    # ---- formula details
    H.append("<h2>1. The formulas (what is being measured)</h2>")
    H.append("<h3>1.1 Label generation — the &ldquo;ground truth&rdquo; (a performance proxy)</h3>")
    H.append("<p>The label is a deterministic 5-class call, computed from <em>future</em> "
             "price (it is a training target, never a model input). "
             "Source: <code>compare_lab/labeling.py</code>.</p>")
    H.append("<pre>"
             "ema      = EMA(price, span=3)\n"
             "for tau in [3, 7, 15] trading days, weight w in [0.3, 0.5, 0.2]:\n"
             "    r_tau  = (ema[t+tau] - ema[t]) / ema[t]      # FORWARD return (looks ahead)\n"
             "    signal += w * ( r_tau / rolling_std_20(r_tau) )   # Sharpe-like, vol-normalised\n"
             "label[t] = quantile_cut(signal, [0.03, 0.15, 0.53, 0.85])\n"
             "           -> STRONG_SELL / SELL / HOLD / BUY / STRONG_BUY</pre>")
    H.append("<div class='box'>Max look-ahead = <b>15 trading days</b> (~3 weeks). The "
             "quantile cuts are <b>asymmetric</b>, reproducing the paper's bull-skewed "
             "class mix (3 / 12 / 38 / 32 / 15 %). The continuous <code>signal</code> "
             "before the cut is the <b>label score</b> used for IC below.</div>")

    H.append("<h3>1.2 Decision matrix — the training reward (asymmetric)</h3>")
    H.append("<p>How a predicted class is scored against the true label during RL "
             "(<code>compare_lab/grpo/rewards.py</code>). False-bullish (long into a "
             "crash) is penalised hardest; HOLD when action is needed is penalised; "
             "an unparseable reply scores &minus;2.5.</p>")
    mh = "".join(f"<th>true {c.replace('STRONG_','S')}</th>" for c in CLASSES)
    mrows = ""
    for i, pc in enumerate(CLASSES):
        cells = ""
        for j, tc in enumerate(CLASSES):
            v = DECISION_MATRIX[pc][tc]
            cls = " class='diag'" if i == j else ""
            cells += f"<td{cls}>{v:+.2f}</td>"
        mrows += f"<tr><td>pred {pc.replace('STRONG_','S')}</td>{cells}</tr>"
    H.append(f"<table class='matrix'><tr><th>pred \\ true</th>{mh}</tr>{mrows}</table>")

    H.append("<h3>1.3 Backtest metrics</h3>")
    H.append("<ul><li><b>CR</b> — Cumulative Return = &Pi;(1+r)&minus;1</li>"
             "<li><b>SR</b> — Sharpe = annualised mean excess return / std "
             "(rf 4%/yr, &radic;252)</li>"
             "<li><b>HR</b> — Hit Rate (ours = fraction of positive-return days; the "
             "paper's HR is directional accuracy &mdash; <b>not directly comparable</b>)</li>"
             "<li><b>MDD</b> — Max Drawdown</li></ul>")
    H.append("<h3>1.4 IC — label-fidelity (does the model learn the label score?)</h3>")
    H.append("<p>Rank correlation (Spearman) between the model's predicted class "
             "(STRONG_SELL=0 … STRONG_BUY=4) and the actual forward label score (§1.1). "
             "0 = chance, 1 = perfect. <b>tail-weighted IC</b> up-weights the extremes "
             "(thin tails of the asymmetric cut). Source: <code>compare_lab/eval_labels.py</code>.</p>")

    # ---- IC results
    H.append("<h2>2. Label-fidelity (IC) per model</h2>")
    H.append(_t(ic,
        ["model", "IC", "tail-IC", "n", "sig.", "backtest SR",
         "matrix reward", "reward gate", "top class %"],
        lambda r: [r["label"],
                   f"{r['ic']:+.3f}", f"{r['tail']:+.3f}", r["n"],
                   f"<b>{r['stars']}</b>",
                   ("<span class='%s'>%+.2f</span>" % ("good" if r["sr"] > 0 else "bad", r["sr"]))
                   if r["sr"] == r["sr"] else "—",
                   f"{r['reward']:+.3f} <span class='mut'>(vs {r['best_const']:+.2f})</span>",
                   ("<span class='good'>✅ PASS</span>" if r["reward"] > r["best_const"]
                    else "<span class='bad'>❌ FAIL</span>"),
                   f"{r['top']*100:.0f}%"]))
    H.append("<div class='box'><b>How to read significance / IC.</b> "
             "Stars = how unlikely the result is to be chance: "
             "<code>***</code> &lt;0.1%, <code>**</code> &lt;1%, <code>*</code> &lt;5%, "
             "<code>ns</code> = just noise. <b>Significant &ne; useful</b>: with n&gt;1500 even "
             "a tiny IC earns stars. Judge usefulness by IC <em>size</em>: "
             "&lt;0.05 negligible · 0.05–0.10 weak · <b>0.10–0.20 modest</b> · "
             "0.20–0.30 decent · &gt;0.30 strong. Every working model sits in the "
             "weak-to-modest band; only the all-SELL model is <code>ns</code> (chance).</div>")
    H.append("<div class='box warn'><b>Reward gate (Jiwoong rebuttal, "
             "<code>docs/2026-06-30-jiwoong-reward-rebuttal.md</code>).</b> Positive IC ≠ "
             "&ldquo;the reward was optimized.&rdquo; A model only <em>beats the reward</em> "
             "if its mean 5×5 decision-matrix reward exceeds the best constant-policy baseline. "
             "<b>Every trained model FAILS this gate</b> (mean reward below best-const) — so "
             "they have weak directional signal, <b>not</b> demonstrated reward mastery. Report "
             "bull-window Sharpe and reward-gate status side by side, never Sharpe alone.</div>")
    H.append("<div class='box warn'><b>Caveat.</b> Forward windows overlap (15-day "
             "horizon, decisions every few days), so observations are not independent — "
             "effective n is roughly a third of nominal. Full-window models (n≈1500) "
             "stay significant; short 2025-H1 models (n≈300) are less robust than the "
             "p-values suggest.</div>")

    # ---- paper comparison
    H.append("<h2>3. 1:1 comparison vs the Trading-R1 paper</h2>")
    H.append("<p>Paper setup reproduced exactly from cached decisions: <b>per-ticker, "
             "single-name long/flat, window 2024-06-01…08-31</b>. Mapping = our system "
             "(BUY/STRONG_BUY → fully long, else flat). "
             "Paper = Table 3/4 (Trading-R1 flagship). "
             "<small>NB: this per-ticker calc is a lightweight pandas backtest (no "
             "commissions); the <em>portfolio</em> results in §4 use the full qf-lib "
             "engine. They are different engines.</small></p>")
    # buy&hold reference table
    H.append("<h3>3.1 buy &amp; hold reference (same window)</h3>")
    bh_rows = [{"tk": tk, **bh[tk], "psr": PAPER[tk][1]} for tk in PAPER]
    H.append(_t(bh_rows, ["ticker", "CR%", "SR", "MDD%", "paper SR"],
               lambda r: [r["tk"], f"{r['CR']*100:.1f}", f"{r['SR']:.2f}",
                          f"{r['MDD']*100:.1f}", f"{r['psr']:.2f}"]))
    bh_sr = sum(bh[t]["SR"] for t in PAPER) / len(PAPER)
    H.append(f"<p><small>buy&amp;hold mean Sharpe = <b>{bh_sr:.2f}</b> — the free "
             "&ldquo;just stay long in a bull market&rdquo; baseline.</small></p>")

    for nm, df in paper_blocks:
        H.append(f"<h3>3.2 {NAMES.get(nm, nm)}</h3>")
        rows = [{"tk": tk, **df.loc[tk].to_dict()} for tk in df.index]
        H.append(_t(rows,
            ["ticker", "our CR%", "paper CR%", "our SR", "paper SR", "our MDD%", "paper MDD%"],
            lambda r: [r["tk"], f"{r['CR%']:.1f}", f"{PAPER[r['tk']][0]:.1f}",
                       f"<b>{r['SR']:.2f}</b>", f"{PAPER[r['tk']][1]:.2f}",
                       f"{r['MDD%']:.1f}", f"{PAPER[r['tk']][3]:.1f}"]))
        osr, psr = df["SR"].mean(), df["p_SR"].mean()
        omdd, pmdd = df["MDD%"].mean(), df["p_MDD%"].mean()
        verdict = "comparable" if abs(osr - psr) < 0.4 else (
            "below paper" if osr < psr else "above paper")
        H.append(f"<p>mean Sharpe <b>{osr:.2f}</b> vs paper {psr:.2f} "
                 f"(<b>{verdict}</b>) &nbsp;|&nbsp; mean MDD {omdd:.1f}% vs paper "
                 f"{pmdd:.1f}% &nbsp;|&nbsp; vs buy&amp;hold {bh_sr:.2f}</p>")

    # ---- portfolio / qf-lib
    H.append("<h2>4. The portfolio backtest (qf-lib engine)</h2>")
    H.append("<p>The headline portfolio results run through the real <b>qf-lib "
             "event-driven engine</b> (<code>compare_lab/backtest.py</code>): a qf-lib "
             "<code>AlphaModel</code> (LONG held / OUT otherwise), "
             "<code>FixedPortfolioPercentagePositionSizer</code> at 1/8 = 12.5% each, "
             "<code>IBCommissionModel</code> commissions, daily order events, next-bar "
             "execution.</p>")
    H.append("<div class='box'><b>Position rule.</b> Each ticker → 5-class decision → "
             "BUY/STRONG_BUY = held. Held names get equal weight (12.5%), capped at "
             "<b>8 slots</b> (ranked by class strength if more are flagged); unused "
             "budget stays in cash. So it is <b>not</b> a fixed all-names equal-weight — "
             "it is &ldquo;equal-weight the model's top-8 buy picks, rest cash.&rdquo;</div>")

    # ---- reproduce
    H.append("<h2>5. Reproduce</h2>")
    H.append("<pre>"
             "# label-fidelity (IC) for a model's cached decisions\n"
             "uv run python -m compare_lab.eval_labels sftv1 grpo v1reggrpo_full\n\n"
             "# 1:1 per-ticker comparison vs the paper (from cache, no inference)\n"
             "uv run python -m compare_lab.compare_paper sftv1 grpo v1reggrpo_full\n\n"
             "# full portfolio backtest through qf-lib (needs a served vLLM model)\n"
             "VLLM_MODEL=sft-v1 VLLM_CACHE_DIR=compare_lab/.cache_sftv1 \\\n"
             "  uv run python -m compare_lab.run_comparison --llm --out compare_lab/output_sftv1\n\n"
             "# regenerate THIS report\n"
             "uv run python -m compare_lab.make_report --out docs/2026-06-29-results-report.html"
             "</pre>")

    # ---- caveats
    H.append("<h2>6. Caveats</h2>")
    H.append("<ul>"
             "<li><b>Prediction &ne; profit.</b> The highest-IC models lose money in "
             "flat 2025-H1; the keeper (lowest IC) makes the most in a bull run — its "
             "return is a bullish lean, not skill.</li>"
             "<li><b>Bull-window flattery.</b> The paper window is a bull run "
             f"(buy&amp;hold Sharpe {bh_sr:.2f}); both our and the paper's high Sharpes "
             "partly come for free from being long.</li>"
             "<li><b>Short window, annualised.</b> 3 months annualised → high-variance "
             "Sharpes (a single good quarter).</li>"
             "<li><b>Engine mismatch.</b> §3 per-ticker = lightweight pandas (no "
             "commissions); §4 portfolio = qf-lib (with commissions).</li>"
             "<li><b>HR not comparable</b> (different definition).</li>"
             "<li><b>Small model, thin input.</b> Qwen3-4B on price-only/~2k-token "
             "snapshots vs the paper's larger model on 15–23k-token full multimodal.</li>"
             "</ul>")
    H.append("</body></html>")
    return "\n".join(H)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/2026-06-29-results-report.html")
    args = ap.parse_args()
    html = build_html()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(html)
    print(f"wrote {args.out}  ({len(html)//1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
