"""Build a single self-contained HTML report for the 3-way comparison (task B).

Combines the headline metrics, the differentiation (correlation) finding, the
LLM decision distribution, and an interactive equity-curve chart into one
shareable page. Reads the CSVs produced by analyze_results.py.

    uv run python -m compare_lab.build_memo_report --out compare_lab/output
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from compare_lab.metrics import all_metrics

LABELS = {
    "equal_weight": "Equal-weight (market)",
    "momentum_12_1": "12-1 Momentum (top-5)",
    "llm_prompt_only": "Prompt-only LLM (Qwen3-4B)",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="compare_lab/output")
    args = ap.parse_args()
    out = Path(args.out)

    rets = pd.read_csv(out / "oos_daily_returns.csv", index_col=0, parse_dates=True)
    equity = (1.0 + rets.fillna(0.0)).cumprod()

    # metrics table
    rows = []
    for col in rets.columns:
        m = all_metrics(rets[col])
        rows.append({
            "Strategy": LABELS.get(col, col),
            "Cumulative": f"{m['CR']*100:+.1f}%",
            "Sharpe": f"{m['SR']:.2f}",
            "Max DD": f"{m['MDD']*100:.1f}%",
        })
    table = pd.DataFrame(rows)
    corr = rets.corr()

    # equity chart
    fig = go.Figure()
    for col in equity.columns:
        fig.add_trace(go.Scatter(x=equity.index, y=equity[col],
                                 name=LABELS.get(col, col), mode="lines"))
    fig.update_layout(
        title="Growth of $1 — OOS 2024-01 → 2026-04 (12 large-caps, weekly rebal)",
        xaxis_title="date", yaxis_title="growth of $1",
        template="plotly_white", legend=dict(orientation="h", y=-0.2),
        height=460, margin=dict(l=50, r=30, t=60, b=40))
    chart = pio.to_html(fig, include_plotlyjs="cdn", full_html=False)

    def _tbl(df: pd.DataFrame, idx=False) -> str:
        return df.to_html(index=idx, border=0, classes="t", justify="center")

    corr_disp = corr.round(2)
    corr_disp.columns = [LABELS.get(c, c).split(" ")[0] for c in corr_disp.columns]
    corr_disp.index = corr_disp.columns

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Trading-R1 × qf-lib — 3-Way Comparison</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;
   margin:32px auto;padding:0 18px;color:#1a2233;line-height:1.5}}
 h1{{font-size:26px;margin:0 0 4px}} .sub{{color:#667085;margin:0 0 22px}}
 table.t{{border-collapse:collapse;width:100%;margin:10px 0 24px;font-size:15px}}
 table.t th,table.t td{{border-bottom:1px solid #e5e9f2;padding:8px 10px;text-align:center}}
 table.t th{{background:#f6f8fb;font-weight:700}}
 .card{{background:#f8fafc;border:1px solid #e5e9f2;border-radius:12px;padding:14px 18px;margin:18px 0}}
 .key{{border-left:4px solid #2563eb;padding-left:12px}}
 code{{background:#eef2ff;padding:1px 5px;border-radius:4px}}
 .muted{{color:#667085;font-size:13px}}
</style></head><body>
<h1>Trading-R1 × qf-lib — 3-Way Signal Comparison</h1>
<p class="sub">Prompt-only 4B LLM vs quant baselines on one look-ahead-safe
qf-lib backtest · 2026-06-21</p>

<h3>Headline (out-of-sample)</h3>
{_tbl(table)}

<div class="card key"><b>The finding that matters:</b> the LLM signal is
<b>differentiated</b>. The two quant baselines move near-identically
(corr 0.90); the LLM is materially less correlated to both (0.63–0.69) — a
diversifying return stream, not a noisy copy of momentum. It also carries the
<b>lowest drawdown</b> in the set, even though its raw return lags in this
strong bull market.</div>

<h3>Daily-return correlation</h3>
{_tbl(corr_disp, idx=True)}

<h3>Equity curves</h3>
{chart}

<div class="card"><b>Caveats.</b> Bull-market regime (equal-weight wins by
construction); 12 tickers (SPY/QQQ pending); price+technical modalities only;
8.2% of LLM replies emitted no parseable decision and were forced to HOLD.
Up-day rate is calendar-day based and weakly discriminative — read Cumulative /
Sharpe / Max-DD.</div>

<p class="muted">Model: <code>Qwen3-4B-Instruct-2507</code> (BF16, temp 0, vLLM
on DGX Spark). Decisions disk-cached by snapshot hash; 12/12 re-queries
reproduced cached outputs (provenance confirmed). Full write-up:
<code>docs/2026-06-21-three-way-comparison-memo.md</code>.</p>
</body></html>"""

    (out / "report.html").write_text(html)
    print(f"wrote {out / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
