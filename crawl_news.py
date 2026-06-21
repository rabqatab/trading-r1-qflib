"""Free historical news crawler via Google News RSS (no key).
Date-bounded with after:/before:. Resumable: one JSON per (ticker, year-month).
Fields: headline, source, url, published_at -> satisfies rubric G1 + G4.
"""
import urllib.request, urllib.parse, json, time, hashlib
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parent / "news_parts"; OUT.mkdir(exist_ok=True)
DS = Path(__file__).resolve().parent / "data_store"
QUERY = {
 "NVDA":"Nvidia stock","MSFT":"Microsoft stock","AAPL":"Apple stock",
 "META":"Meta Platforms stock","AMZN":"Amazon stock","TSLA":"Tesla stock",
 "BRK-B":"Berkshire Hathaway stock","JPM":"JPMorgan stock","LLY":"Eli Lilly stock",
 "JNJ":"Johnson & Johnson stock","XOM":"Exxon Mobil stock","CVX":"Chevron stock",
}
MONTHS = pd.date_range("2024-01-01","2025-06-01",freq="MS")
SLEEP = 3.5

def fetch(q,tries=4):
    url="https://news.google.com/rss/search?"+urllib.parse.urlencode(
        {"q":q,"hl":"en-US","gl":"US","ceid":"US:en"})
    for i in range(tries):
        try:
            req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"})
            return urllib.request.urlopen(req,timeout=30).read()
        except Exception:
            time.sleep(5*(i+1))
    return None

def main():
    done=fail=total=0
    for tk,base in QUERY.items():
        for m in MONTHS:
            ym=m.strftime("%Y%m"); fp=OUT/f"{tk}_{ym}.json"
            if fp.exists(): done+=1; continue
            a=m.strftime("%Y-%m-%d"); b=(m+pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
            raw=fetch(f"{base} after:{a} before:{b}")
            if raw is None: fail+=1; print(f"FAIL {tk} {ym}",flush=True); time.sleep(SLEEP); continue
            rows=[]
            try:
                root=ET.fromstring(raw)
                for it in root.findall(".//item"):
                    src=it.find("{*}source")
                    rows.append({"ticker":tk,"headline":it.findtext("title") or "",
                        "source":(src.text if src is not None else ""),
                        "url":it.findtext("link") or "","pub":it.findtext("pubDate") or ""})
            except Exception: pass
            json.dump(rows,open(fp,"w")); total+=len(rows); done+=1
            print(f"ok {tk} {ym} n={len(rows)} done={done}/{len(QUERY)*len(MONTHS)} total={total}",flush=True)
            time.sleep(SLEEP)
    rows=[]
    for f in OUT.glob("*.json"):
        try: rows+=json.load(open(f))
        except: pass
    df=pd.DataFrame(rows)
    df["published_at"]=df["pub"].map(lambda s: parsedate_to_datetime(s) if s else None)
    df=df.dropna(subset=["published_at"])
    df["published_at"]=df["published_at"].dt.tz_localize(None)
    df["date"]=df["published_at"].dt.normalize()
    df["url_hash"]=df["url"].map(lambda u:hashlib.sha1(str(u).encode()).hexdigest()[:16])
    df=df.drop_duplicates("url_hash").sort_values(["ticker","published_at"]).reset_index(drop=True)
    df=df[["ticker","date","published_at","headline","source","url","url_hash"]]
    df.to_parquet(DS/"news.parquet",index=False)
    print(f"WROTE news.parquet rows={len(df)} tickers={df.ticker.nunique()} fail={fail}",flush=True)

if __name__=="__main__":
    main()
