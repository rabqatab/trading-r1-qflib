"""Roadmap D: LLM-as-ENCODER ablation — does FinBERT text help a tabular GBM?

Context: the end-to-end LLM gave text raw IC ~ -0.01. 2024-26 literature says the only
text wins come from embeddings + tabular head (+0.005-0.010 in large caps), and frozen
FinBERT + PCA is a published null (arXiv:2606.29290). This script tests exactly that,
on the rank_target_sweep protocol (daily cross-sectional Spearman rank-IC vs the RAW
7-trading-day forward return; canonical 2025-H1 OOS; reg GBM anchor +0.051 under this
protocol / +0.042 under gbm_ceiling's).

Arms (all HistGradientBoostingRegressor, same hyperparams as gbm_ceiling):
  1. price            : the 16 technical indicators (baseline).
  2. price+pca32      : + 32-dim PCA (fit on TRAIN only) of the pooled FinBERT
                        embedding — mean over news items in a trailing 7-calendar-day
                        window. Published-null replication arm.
  3. price+supervised : + 1-dim ridge prediction (full embedding -> TRAIN forward
                        return) + top-8 PLS components — the return-supervised
                        projection the literature says is the only thing that works.
  4. price+sentiment  : + scalar mean (P_pos - P_neg) from FinBERT's sentiment head,
                        same trailing 7-day window.

News store: data/qflib_data_store_top150/news_top150_summ.parquet (Finnhub headlines +
summaries, 2024-11-01 .. 2025-07-31 only) — so TRAIN news coverage is Nov-Dec 2024 and
the news-covered OOS window is 2025-H1 (stated in the output). Rows without news carry
NaN news features (HGB handles NaN natively); PCA / ridge / PLS are fit only on TRAIN
rows that have news.

Leak safety: only news dated <= t enters the feature at t (asserted by recomputing
sampled windows from the raw item table); forward-return labels are asserted to be
built from days t+1..t+7 only; train cutoff is label-complete before the eval window.

Two stages (the embed stage needs torch+transformers, not in the repo env):

    UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
        uv run --with torch --with transformers python -m compare_lab.encoder_ablation --embed
    uv run python -m compare_lab.encoder_ablation
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import compare_lab  # noqa: F401
from compare_lab.snapshot import _INDICATORS  # same 16 features as gbm_ceiling

_PRICES = Path("data/qflib_data_store_top150/prices_top150.parquet")
_NEWS = Path("data/qflib_data_store_top150/news_top150_summ.parquet")
_EMB = Path(os.environ.get("FINBERT_EMB_PATH",
                           "data/qflib_data_store_top150/finbert_embed_daily_top150.parquet"))
_FEAT_CACHE = Path(os.environ.get(
    "RANK_SWEEP_CACHE",
    "/tmp/claude-1000/-home-alphabridge-Study-tradingR1-qflib/"
    "f4ef47e5-faf4-4073-9339-e7be58e5a29b/scratchpad/rank_sweep_features.parquet"))

H = 7                 # label/eval horizon, trading days
WIN = 7               # trailing news window, calendar days
N_PCA, N_PLS = 32, 8
MIN_NAMES = 50        # min cross-section size for a daily IC (rank_target_sweep)
EVAL_LO, EVAL_HI = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-30")
_MODELS = ("ProsusAI/finbert", "yiyanghkust/finbert-tone")


# --------------------------------------------------------------------------- embed
def embed(batch_size: int) -> int:
    """Per-news-item FinBERT forward pass -> per (ticker, day) mean embedding +
    sentiment probs, saved as one parquet. CPU is fine (~150k items)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    news = pd.read_parquet(_NEWS, columns=["ticker", "date", "headline", "summary"])
    texts = (news["headline"].str.strip() + ". " + news["summary"].str.strip()).tolist()
    print(f"[embed] {len(texts)} news items, {news.ticker.nunique()} tickers, "
          f"{news.date.min().date()}..{news.date.max().date()}", flush=True)

    tok = mdl = name = None
    for name in _MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            mdl = AutoModelForSequenceClassification.from_pretrained(name)
            break
        except Exception as e:  # noqa: BLE001
            print(f"[embed] {name} unavailable ({e}); trying next", flush=True)
    assert mdl is not None, "no FinBERT variant downloadable"
    mdl.eval()
    torch.set_num_threads(os.cpu_count() or 8)
    lab = {v.lower(): k for k, v in mdl.config.id2label.items()}
    i_pos, i_neg = lab["positive"], lab["negative"]
    print(f"[embed] model={name} labels={mdl.config.id2label}", flush=True)

    # length-sorted batching: unsorted batches all pad to ~128 (one long item each);
    # sorting by token count cuts CPU cost by the padding ratio (measured 16 it/s -> see log)
    ids = tok(texts, truncation=True, max_length=128)["input_ids"]
    order = np.argsort([len(x) for x in ids], kind="stable")
    embs = np.empty((len(texts), mdl.config.hidden_size), dtype=np.float32)
    probs = np.empty((len(texts), mdl.config.num_labels), dtype=np.float32)
    t0, done = time.time(), 0
    with torch.inference_mode():
        for i in range(0, len(order), batch_size):
            sel = order[i:i + batch_size]
            enc = tok([texts[j] for j in sel], padding=True, truncation=True,
                      max_length=128, return_tensors="pt")
            out = mdl(**enc, output_hidden_states=True)
            hid = out.hidden_states[-1]                       # (B, L, 768)
            m = enc["attention_mask"].unsqueeze(-1).float()
            embs[sel] = ((hid * m).sum(1) / m.sum(1)).numpy()
            probs[sel] = torch.softmax(out.logits, -1).numpy()
            done += len(sel)
            if i and i % (batch_size * 50) == 0:
                r = (time.time() - t0) / done
                print(f"[embed] {done}/{len(texts)}  {1/r:.0f} items/s  "
                      f"ETA {(len(texts)-done)*r/60:.0f} min", flush=True)

    d = pd.DataFrame({"ticker": news["ticker"].values,
                      "date": pd.to_datetime(news["date"]).values,
                      "pos": probs[:, i_pos], "neg": probs[:, i_neg]})
    for j in range(embs.shape[1]):
        d[f"e{j}"] = embs[:, j]
    ecols = [f"e{j}" for j in range(embs.shape[1])]
    agg = d.groupby(["ticker", "date"], as_index=False).agg(
        {**{c: "mean" for c in ecols + ["pos", "neg"]}})
    agg["n"] = d.groupby(["ticker", "date"]).size().values
    agg.to_parquet(_EMB)
    print(f"[embed] wrote {len(agg)} (ticker,day) rows -> {_EMB} "
          f"({time.time()-t0:.0f}s total)", flush=True)
    return 0


# --------------------------------------------------------------------------- panel
def build_panel():
    """Price feature panel + raw 7d forward label (identical to rank_target_sweep)."""
    from stockstats import StockDataFrame

    px = pd.read_parquet(_PRICES)
    px["date"] = pd.to_datetime(px["date"])
    piv = px.pivot(index="date", columns="ticker", values="Close").sort_index()
    fwd7 = piv.shift(-H) / piv - 1

    # leak assert: label at t = compounded daily returns of t+1..t+7 only
    ret1 = piv.pct_change()
    col, i = piv.columns[0], 1500
    manual = float(np.prod(1.0 + ret1[col].iloc[i + 1:i + H + 1].values) - 1.0)
    assert np.isclose(fwd7[col].iloc[i], manual, atol=1e-12), "label leak check failed"

    if _FEAT_CACHE.exists():
        feats = pd.read_parquet(_FEAT_CACHE)
    else:
        parts = []
        for t, g in px.groupby("ticker"):
            df = (g.set_index("date")[["Open", "High", "Low", "Close", "Volume"]]
                   .rename(columns=str.lower).sort_index().dropna())
            if len(df) < 250:
                continue
            sdf = StockDataFrame.retype(df.copy())
            for ind in _INDICATORS:
                _ = sdf[ind]
            F = sdf[_INDICATORS].copy()
            F["ticker"] = t
            parts.append(F)
        feats = pd.concat(parts).reset_index().rename(columns={"index": "date"})
        _FEAT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(_FEAT_CACHE)

    feats = feats.set_index(["date", "ticker"]).sort_index()
    feats["y"] = fwd7.stack()
    return feats, piv, fwd7


def news_window_features(cal: pd.DatetimeIndex):
    """Trailing WIN-calendar-day news aggregates per (trading day, ticker):
    mean embedding (item-weighted), mean (pos-neg), item count."""
    agg = pd.read_parquet(_EMB)
    ecols = [c for c in agg.columns if c.startswith("e")]
    days = pd.date_range(agg["date"].min(), agg["date"].max(), freq="D")
    tdays = cal[(cal >= days[0]) & (cal <= days[-1])]

    out = {}
    for t, g in agg.groupby("ticker"):
        g = g.set_index("date").sort_index()
        n = g["n"].reindex(days).fillna(0.0)
        S = g[ecols + ["pos", "neg"]].mul(g["n"], axis=0).reindex(days).fillna(0.0)
        n7 = n.rolling(WIN, min_periods=1).sum()
        S7 = S.rolling(WIN, min_periods=1).sum()
        W = S7.div(n7.replace(0, np.nan), axis=0)             # NaN when no news in window
        W["sent7"] = W["pos"] - W["neg"]
        W["n7"] = n7
        out[t] = W.loc[W.index.isin(tdays), ecols + ["sent7", "n7"]]
    nf = pd.concat(out, names=["ticker", "date"]).swaplevel().sort_index()

    # --- self-check: embedding-date alignment is leak-free (news <= t only) ---
    rng = np.random.default_rng(0)
    have = nf[nf["n7"] > 0]
    for k in rng.choice(len(have), size=5, replace=False):
        d, t = have.index[k]
        raw = agg[(agg["ticker"] == t) & (agg["date"] > d - pd.Timedelta(days=WIN))
                  & (agg["date"] <= d)]
        assert len(raw) and (raw["date"] <= d).all(), "future news in window!"
        manual = np.average(raw[ecols].values, weights=raw["n"].values, axis=0)
        assert np.allclose(manual, have.iloc[k][ecols].values.astype(float),
                           rtol=1e-4, atol=1e-5), "window aggregation mismatch"
        msent = (np.average(raw["pos"], weights=raw["n"])
                 - np.average(raw["neg"], weights=raw["n"]))
        assert np.isclose(msent, have.iloc[k]["sent7"], atol=1e-5)
    print("[ablate] leak-free window self-check passed (5 sampled (ticker,day) cells)")
    return nf, ecols


# --------------------------------------------------------------------------- ablate
def _daily_ic(pred: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    ics = pred.rank(axis=1).corrwith(fwd.rank(axis=1), axis=1)
    n = (pred.notna() & fwd.notna()).sum(axis=1)
    return ics[n >= MIN_NAMES]


def _pooled_ic(pred: pd.Series, y: pd.Series) -> float:
    df = pd.concat([pred, y], axis=1, keys=["p", "y"]).dropna()
    return float(df["p"].rank().corr(df["y"].rank()))


def ablate() -> int:
    from scipy import stats
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge

    feats, piv, fwd7 = build_panel()
    cal = piv.index
    nf, ecols = news_window_features(cal)

    # label-complete train cutoff (rank_target_sweep convention)
    cut = cal[cal.searchsorted(EVAL_LO) - H - 1]
    assert cal[cal.searchsorted(cut) + H] < EVAL_LO, "train label overlaps eval"
    eval_days = cal[(cal >= EVAL_LO) & (cal <= EVAL_HI)]

    tr = feats.loc[(slice(None, cut), slice(None))].dropna(subset=["y"])
    te = feats.loc[(slice(eval_days[0], eval_days[-1]), slice(None))]
    tr_news = tr.join(nf, how="inner")
    tr_news = tr_news[tr_news["n7"] > 0]
    te_news = te.join(nf, how="left")

    # coverage
    cov_te = float((te_news["n7"] > 0).mean())
    news_lo, news_hi = nf.index.get_level_values("date").min(), \
        nf.index.get_level_values("date").max()
    print(f"[ablate] news store covers {news_lo.date()}..{news_hi.date()} -> TRAIN news "
          f"rows {len(tr_news)} of {len(tr)} total train rows "
          f"({tr_news.index.get_level_values('date').min().date()}.."
          f"{tr_news.index.get_level_values('date').max().date()}); pre-2024-11 train "
          f"rows carry NaN news features")
    print(f"[ablate] EVAL 2025-H1 grid: {len(te)} (ticker,day) cells, "
          f"{te.index.get_level_values('ticker').nunique()} tickers x "
          f"{len(eval_days)} days | news coverage {100*cov_te:.1f}% of cells | "
          f"median items in 7d window {te_news.loc[te_news.n7>0,'n7'].median():.0f}")

    # supervised + unsupervised reductions, fit on TRAIN news rows only
    Etr, ytr_n = tr_news[ecols].values, tr_news["y"].values
    pca = PCA(n_components=N_PCA, random_state=0).fit(Etr)
    # ridge alpha by TIME-split validation inside train (LOO-CV is over-optimistic on
    # overlapping 7d forward returns and picks the grid-boundary alpha)
    tr_dates = tr_news.index.get_level_values("date")
    d_split = tr_dates.sort_values()[int(0.7 * len(tr_dates))]
    m_fit = np.asarray(tr_dates <= d_split)
    best = (np.inf, None)
    for a in np.logspace(0, 8, 17):
        r = Ridge(alpha=a).fit(Etr[m_fit], ytr_n[m_fit])
        mse = float(np.mean((r.predict(Etr[~m_fit]) - ytr_n[~m_fit]) ** 2))
        best = min(best, (mse, a))
    ridge = Ridge(alpha=best[1]).fit(Etr, ytr_n)
    pls = PLSRegression(n_components=N_PLS).fit(Etr, ytr_n)
    print(f"[ablate] PCA32 var explained {pca.explained_variance_ratio_.sum():.2f} | "
          f"ridge alpha {best[1]:.0f} (time-split val, train R2 "
          f"{ridge.score(Etr, ytr_n):.4f})")

    # placebo control: embeddings re-assigned to random (ticker,day) rows of the panel —
    # destroys information, keeps dimensionality/marginals. arm2 ~ placebo => noise.
    rng = np.random.default_rng(1)
    nf_sh = nf.copy()
    rows = np.where((nf["n7"] > 0).values)[0]
    ecols_pos = [nf.columns.get_loc(c) for c in ecols]
    nf_sh.iloc[rows, ecols_pos] = nf[ecols].values[rng.permutation(rows)]
    tr_sh = tr.join(nf_sh, how="inner")
    tr_sh = tr_sh[tr_sh["n7"] > 0]
    pca_sh = PCA(n_components=N_PCA, random_state=0).fit(tr_sh[ecols].values)

    def pca_sh_cols(df):
        out = pd.DataFrame(np.nan, index=df.index,
                           columns=[f"pcs{j}" for j in range(N_PCA)])
        m = (df["n7"] > 0).values
        if m.any():
            out.loc[m, :] = pca_sh.transform(df.loc[m, ecols].values)
        return out

    def news_cols(df):
        """PCA / ridge / PLS / sentiment features for a joined frame (NaN w/o news)."""
        out = pd.DataFrame(index=df.index)
        m = (df["n7"] > 0).values if "n7" in df else np.zeros(len(df), bool)
        E = df.loc[m, ecols].values if m.any() else np.empty((0, len(ecols)))
        for j in range(N_PCA):
            out[f"pca{j}"] = np.nan
        out.loc[m, [f"pca{j}" for j in range(N_PCA)]] = pca.transform(E) if m.any() else 0
        out["ridge1"] = np.nan
        out.loc[m, "ridge1"] = ridge.predict(E) if m.any() else 0
        for j in range(N_PLS):
            out[f"pls{j}"] = np.nan
        out.loc[m, [f"pls{j}" for j in range(N_PLS)]] = pls.transform(E) if m.any() else 0
        out["sent7"] = df["sent7"] if "sent7" in df else np.nan
        return out

    tr_all = pd.concat([tr, news_cols(tr.join(nf, how="left")),
                        pca_sh_cols(tr.join(nf_sh, how="left"))], axis=1)
    te_all = pd.concat([te, news_cols(te_news),
                        pca_sh_cols(te.join(nf_sh, how="left"))], axis=1)

    P = list(_INDICATORS)
    arms = {
        "price":            P,
        "price+pca32":      P + [f"pca{j}" for j in range(N_PCA)],
        "price+supervised": P + ["ridge1"] + [f"pls{j}" for j in range(N_PLS)],
        "price+sentiment":  P + ["sent7"],
        "price+pca32-SHUF": P + [f"pcs{j}" for j in range(N_PCA)],
    }

    rows = []
    mom10 = piv / piv.shift(10) - 1
    dic = _daily_ic(mom10.loc[eval_days], fwd7.loc[eval_days])
    rows.append(("mom10 (model-free)", dic, _pooled_ic(
        mom10.loc[eval_days].stack(), fwd7.loc[eval_days].stack())))

    for arm, cols in arms.items():
        gbm = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                            max_depth=3, l2_regularization=1.0,
                                            random_state=0)
        gbm.fit(tr_all[cols], tr_all["y"])
        pred = pd.Series(gbm.predict(te_all[cols]), index=te_all.index)
        pmat = pred.unstack("ticker")
        dic = _daily_ic(pmat, fwd7.loc[pmat.index])
        rows.append((arm, dic, _pooled_ic(pred, te_all["y"])))
        print(f"[ablate] fitted {arm:<17} ({len(cols)} features)", flush=True)

    print(f"\n=== encoder ablation, 2025-H1 OOS (news-covered window), eval horizon "
          f"{H}d raw fwd return ===")
    print(f"{'arm':<20} {'daily rank-IC':>13} {'t':>6} {'days':>5} {'pooled IC':>10}")
    for arm, dic, pool in rows:
        t = stats.ttest_1samp(dic, 0.0).statistic if len(dic) > 2 else float("nan")
        print(f"{arm:<20} {dic.mean():>+13.4f} {t:>6.2f} {len(dic):>5d} {pool:>+10.4f}")
    print("\nanchor: rank_target_sweep reg_raw_h7 2025H1 = +0.0508 under this exact "
          "protocol (gbm_ceiling proxy-target protocol: +0.042).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", action="store_true", help="run the FinBERT embed stage")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    return embed(args.batch_size) if args.embed else ablate()


if __name__ == "__main__":
    raise SystemExit(main())
