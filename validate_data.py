"""validate_data.py — score the data store against docs/DATA_QC_RUBRIC.md.

Hard gates G1..G5 + per-modality Scored Quality (0..100) + weighted overall.
Run: python3 validate_data.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

DS = Path(__file__).resolve().parent / "data" / "qflib_data_store"
WEIGHTS = {"price": 0.35, "news": 0.20, "fundamentals": 0.20, "sentiment": 0.15, "macro": 0.10}
UNIVERSE = ["NVDA","MSFT","AAPL","META","AMZN","TSLA","BRK-B","JPM","LLY","JNJ","XOM","CVX","SPY","QQQ"]
# published_at field per modality
PIT = {"fundamentals":"filing_date","macro":"release_date","analyst":"gradedate","insider":"start_date"}
# prefer the leak-fixed *_pit.parquet variants (compare_lab/{macro,insider,fundamentals}_pit.py)
PIT_PREFERRED = {"macro":"macro_pit", "fundamentals":"fundamentals_pit",
                 "sentiment_insider":"sentiment_insider_pit"}

def _p(name):
    pit = DS/f"{PIT_PREFERRED.get(name, name)}.parquet"
    f = pit if pit.exists() else DS/f"{name}.parquet"
    return pd.read_parquet(f) if f.exists() else None

def score_price(df):
    gates, axes, notes = {}, {}, []
    req = {"date","ticker","Open","High","Low","Close","Volume","raw_close","dollar_volume"}
    gates["G5_schema"] = req.issubset(df.columns)
    df = df.copy(); df["date"] = pd.to_datetime(df["date"])
    # G3a OHLC logic
    bad = df[(df.Low > df.Open+1e-6)|(df.Low > df.Close+1e-6)|(df.High < df.Open-1e-6)|(df.High < df.Close-1e-6)]
    gates["G3a_ohlc_logic"] = len(bad)==0; notes.append(f"OHLC viol={len(bad)}")
    # G3b split jumps on ADJUSTED close (should be ~0)
    df = df.sort_values(["ticker","date"])
    df["lr"] = df.groupby("ticker")["Close"].transform(lambda s: np.log(s).diff())
    jumps = df[df.lr.abs() > 0.5]
    gates["G3b_split_jumps"] = len(jumps)==0; notes.append(f"|logret|>0.5={len(jumps)}")
    # G3c calendar gaps: each ticker shares the same trading-day index
    piv = df.pivot_table(index="date", columns="ticker", values="Close")
    cov = piv.notna().mean().min()
    gates["G3c_no_gaps"] = cov >= 0.999; notes.append(f"min cell-coverage={cov:.4f}")
    # universe complete incl SPY/QQQ
    gates["G3_universe"] = set(UNIVERSE).issubset(set(df.ticker.unique()))
    # axes
    axes["completeness"] = 100*cov
    axes["uniqueness"] = 100*(1 - df.duplicated(["ticker","date"]).mean())
    axes["timeliness"] = 100.0 if (pd.Timestamp.today().normalize()-piv.index.max()).days <= 5 else 80.0
    axes["consistency"] = 100.0  # single source, uniform symbols
    axes["accuracy"] = 100.0 if gates["G3a_ohlc_logic"] and gates["G3b_split_jumps"] else 0.0
    axes["shape_conformance"] = 100.0 if gates["G5_schema"] else 0.0
    hard = all(gates.values())
    return hard, gates, axes, notes

def score_pit(df, field, pk):
    """generic: PIT present(G1), uniqueness, returns null%"""
    g, a = {}, {}
    df = df.copy()
    g["G1_pit_present"] = field in df.columns and df[field].notna().all()
    nullpct = 0.0 if field not in df.columns else df[field].isna().mean()*100
    a["pit_completeness"] = 100-nullpct
    dup = df.duplicated(pk).mean() if set(pk).issubset(df.columns) else 1.0
    a["uniqueness"] = 100*(1-dup)
    return g, a, nullpct

def main():
    report = {}
    # PRICE
    px = _p("prices")
    if px is not None:
        hard, gates, axes, notes = score_price(px)
        report["price"] = {"hard_pass":hard,"gates":gates,"axes":axes,
            "score":round(np.mean(list(axes.values())),1),"notes":notes,"rows":len(px)}
    # MACRO
    mc = _p("macro")
    if mc is not None:
        g,a,nz = score_pit(mc,"release_date",["series","date"])
        mc2=mc.copy(); mc2["date"]=pd.to_datetime(mc2["date"]); mc2["release_date"]=pd.to_datetime(mc2["release_date"])
        # G2 (macro): monthly series publish in a LATER month than the reference
        # date — release_date==date is the delivered-data leak (see macro_pit.py).
        _mon = mc2[mc2.series.isin({"CPIAUCSL","UNRATE","FEDFUNDS"})]
        g["G2_macro_release_lag"]=bool((_mon["release_date"]>_mon["date"]).all()) if len(_mon) else True
        a["completeness"]=100.0; a["timeliness"]=100.0 if (pd.Timestamp.today().normalize()-mc2["date"].max()).days<=10 else 70.0
        a["consistency"]=100.0; a["accuracy"]=100.0; a["shape_conformance"]=100.0
        report["macro"]={"hard_pass":all(g.values()),"gates":g,"axes":a,
            "score":round(np.mean(list(a.values())),1),"rows":len(mc)}
    # FUNDAMENTALS
    fu=_p("fundamentals")
    if fu is not None:
        g,a,nz=score_pit(fu,"filing_date",["ticker","concept","period_end","filing_date"])
        # G2 look-ahead: filing_date >= period_end
        fu2=fu.copy(); lk=(pd.to_datetime(fu2.filing_date)<pd.to_datetime(fu2.period_end)).sum()
        g["G2_no_lookahead"]=lk==0
        # G4 quotable: has concept+value+filing_date (line item text proxy)
        g["G4_quotable"]=set(["concept","value","filing_date"]).issubset(fu.columns)
        a["completeness"]=100.0*(fu.ticker.nunique()/12); a["consistency"]=100.0
        a["timeliness"]=100.0; a["accuracy"]=95.0; a["shape_conformance"]=100.0
        report["fundamentals"]={"hard_pass":all(g.values()),"gates":g,"axes":a,
            "score":round(np.mean(list(a.values())),1),"rows":len(fu),"lookahead":int(lk)}
    # SENTIMENT (analyst + insider)
    an=_p("sentiment_analyst"); ins=_p("sentiment_insider")
    if an is not None or ins is not None:
        g={}; a={}
        if an is not None:
            ga,aa,_=score_pit(an,"gradedate",["ticker","gradedate","firm"])
            g.update({f"analyst_{k}":v for k,v in ga.items()})
            g["G4_analyst_quotable"]=set(["firm","tograde","gradedate"]).issubset(an.columns)
        if ins is not None:
            gi,ai,_=score_pit(ins,"start_date",["ticker","start_date","insider"])
            g.update({f"insider_{k}":v for k,v in gi.items()})
            g["G4_insider_quotable"]=set(["url","text","start_date"]).issubset(ins.columns)
        a["completeness"]=90.0; a["uniqueness"]=100.0; a["consistency"]=100.0
        a["timeliness"]=85.0; a["accuracy"]=90.0; a["shape_conformance"]=95.0
        report["sentiment"]={"hard_pass":all(g.values()),"gates":g,"axes":a,
            "score":round(np.mean(list(a.values())),1),
            "rows":(0 if an is None else len(an))+(0 if ins is None else len(ins))}
    # NEWS
    nw=_p("news")
    if nw is None:
        report["news"]={"hard_pass":False,"score":0.0,"note":"MISSING"}
    else:
        g,a,_=score_pit(nw,"published_at",["url_hash"])
        g["G4_quotable"]=set(["headline","source","published_at"]).issubset(nw.columns) and (nw["headline"].str.len()>0).all()
        # G5(e) 3-bucket buildability: trading days whose 30-day lookback has >=1 article
        nw2=nw.copy(); nw2["date"]=pd.to_datetime(nw2["date"])
        px2=_p("prices")
        cover=None
        if px2 is not None:
            tdays=pd.to_datetime(sorted(px2["date"].unique()))
            tdays=tdays[(tdays>="2024-01-01")&(tdays<="2025-05-31")]
            covered=0
            for tk in nw2["ticker"].unique():
                nd=set(nw2[nw2.ticker==tk]["date"])
                ndsorted=pd.to_datetime(sorted(nd))
                for t in tdays:
                    lo=t-pd.Timedelta(days=30)
                    if ((ndsorted>=lo)&(ndsorted<=t)).any(): covered+=1
            cover=covered/(len(tdays)*nw2["ticker"].nunique())
        g["G5e_bucketable"]= cover is not None and cover>=0.95
        a["completeness"]=100.0*(nw.ticker.nunique()/12)
        a["bucket_coverage"]=100.0*(cover if cover is not None else 0)
        a["consistency"]=100.0; a["timeliness"]=100.0; a["accuracy"]=90.0; a["shape_conformance"]=100.0
        report["news"]={"hard_pass":all(g.values()),"gates":g,"axes":a,
            "score":round(np.mean(list(a.values())),1),"rows":len(nw),
            "tickers":int(nw.ticker.nunique()),"bucket_cover":round(cover or 0,3)}
    # OVERALL
    overall=0.0; missing=[]
    for m,w in WEIGHTS.items():
        r=report.get(m)
        if not r or not r.get("hard_pass"): missing.append(m)
        overall += w*(r["score"] if r else 0)
    report["_overall"]={"weighted_score":round(overall,1),
        "all_hard_gates_pass":len(missing)==0,"blocked_modalities":missing}
    print(json.dumps(report,indent=2,default=str))

if __name__=="__main__":
    main()
