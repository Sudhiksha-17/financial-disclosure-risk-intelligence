"""
Extract Missing Pairs
=====================
Re-extracts text files for all annotated pairs that don't have
saved extracted files locally. Covers:
- Original session pairs (VLO, CINF, FITB, etc.)
- v9 pairs (ABNB, HCA, NFLX, SNAP, TGT, THC, CRM, CVS, WBD, TWLO, BKNG)
Output: extracted_missing/ folder
"""

import requests
import re
import time
from pathlib import Path

HEADERS = {"User-Agent": "sudhiksha.research@gmail.com"}

TARGETS = [
    # Original session pairs
    {"ticker": "VLO",  "dimension": "credit_risk",       "years": [2021, 2022]},
    {"ticker": "CINF", "dimension": "regulatory_risk",   "years": [2020, 2021]},
    {"ticker": "CINF", "dimension": "regulatory_risk",   "years": [2021, 2022]},
    {"ticker": "FITB", "dimension": "regulatory_risk",   "years": [2022, 2023]},
    {"ticker": "TWLO", "dimension": "credit_risk",       "years": [2019, 2020]},
    {"ticker": "ZS",   "dimension": "credit_risk",       "years": [2020, 2021]},
    {"ticker": "FANG", "dimension": "operational_risk",  "years": [2019, 2020]},
    {"ticker": "FANG", "dimension": "operational_risk",  "years": [2021, 2022]},
    {"ticker": "FANG", "dimension": "operational_risk",  "years": [2022, 2023]},
    {"ticker": "HUBS", "dimension": "operational_risk",  "years": [2021, 2022]},
    {"ticker": "AIZ",  "dimension": "operational_risk",  "years": [2019, 2020]},
    {"ticker": "CINF", "dimension": "credit_risk",       "years": [2022, 2023]},
    {"ticker": "EOG",  "dimension": "credit_risk",       "years": [2019, 2020]},
    {"ticker": "LYFT", "dimension": "operational_risk",  "years": [2021, 2022]},
    {"ticker": "UBER", "dimension": "operational_risk",  "years": [2021, 2022]},
    {"ticker": "ZM",   "dimension": "operational_risk",  "years": [2021, 2022]},
    # v8 pairs not saved locally
    {"ticker": "ABNB", "dimension": "operational_risk",  "years": [2021, 2022]},
    # v9 pairs
    {"ticker": "CRM",  "dimension": "operational_risk",  "years": [2021, 2022]},
    {"ticker": "CVS",  "dimension": "operational_risk",  "years": [2021, 2022]},
    {"ticker": "HCA",  "dimension": "operational_risk",  "years": [2021, 2022]},
    {"ticker": "NFLX", "dimension": "operational_risk",  "years": [2021, 2022]},
    {"ticker": "SNAP", "dimension": "operational_risk",  "years": [2021, 2022]},
    {"ticker": "TGT",  "dimension": "operational_risk",  "years": [2021, 2022]},
    {"ticker": "THC",  "dimension": "operational_risk",  "years": [2021, 2022]},
    {"ticker": "TWLO", "dimension": "operational_risk",  "years": [2021, 2022]},
    {"ticker": "WBD",  "dimension": "operational_risk",  "years": [2021, 2022]},
    {"ticker": "BKNG", "dimension": "operational_risk",  "years": [2021, 2022]},
]

SECTION_HINTS = [
    "covid", "pandemic", "coronavirus", "public health",
    "business continuity", "operations", "disruption",
    "workforce", "labor", "supply chain", "credit", "regulatory"
]

KEYWORDS = [
    "COVID", "pandemic", "coronavirus", "disruption", "operations",
    "workforce", "supply chain", "public health", "credit", "regulatory",
    "risk", "adverse", "material", "impact"
]

def get_cik(ticker):
    r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=HEADERS)
    for entry in r.json().values():
        if entry["ticker"].upper() == ticker.upper():
            return str(entry["cik_str"]).zfill(10)
    return None

def get_filings(cik, years):
    r = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=HEADERS)
    data = r.json()["filings"]["recent"]
    results = {}
    for i, form in enumerate(data["form"]):
        if form != "10-K":
            continue
        date = data["filingDate"][i]
        filing_year = int(date[:4])
        filing_month = int(date[5:7])
        fy = filing_year - 1 if filing_month <= 4 else filing_year
        if fy in years and fy not in results:
            acc = data["accessionNumber"][i].replace("-", "")
            results[fy] = {
                "date": date,
                "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{data['primaryDocument'][i]}"
            }
    if len(results) < len(years):
        older = r.json()["filings"].get("files", [])
        for f in older:
            if len(results) >= len(years):
                break
            furl = "https://data.sec.gov/submissions/" + f["name"]
            try:
                r2 = requests.get(furl, headers=HEADERS)
                data2 = r2.json()
                for i, form in enumerate(data2["form"]):
                    if form != "10-K":
                        continue
                    date = data2["filingDate"][i]
                    fy_year = int(date[:4])
                    fy_month = int(date[5:7])
                    fy = fy_year - 1 if fy_month <= 4 else fy_year
                    if fy in years and fy not in results:
                        acc = data2["accessionNumber"][i].replace("-", "")
                        results[fy] = {
                            "date": date,
                            "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{data2['primaryDocument'][i]}"
                        }
                time.sleep(0.3)
            except:
                continue
    return results

def fetch_clean(url):
    r = requests.get(url, headers=HEADERS, timeout=90)
    text = r.text
    text = re.sub(r'<ix:[^>]+>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'</ix:[^>]+>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    for e, c in [('&amp;','&'),('&lt;','<'),('&gt;','>'),('&nbsp;',' '),
                 ('&#160;',' '),('&#8226;','•'),('&#8217;',"'"),
                 ('&#8220;','"'),('&#8221;','"')]:
        text = text.replace(e, c)
    text = re.sub(r'&#\d+;', ' ', text)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def get_item1a(text):
    end_patterns = [
        r"ITEM\s+1B[\.\s]", r"Item\s+1B[\.\s]",
        r"ITEM\s+2[\.\s]", r"Item\s+2[\.\s]",
        r"UNRESOLVED\s+STAFF\s+COMMENTS",
    ]
    start_pattern = r"ITEM\s+1A[\.\s]*RISK\s+FACTORS"
    matches = list(re.finditer(start_pattern, text, re.IGNORECASE))
    candidates = []
    for m in matches:
        s = m.end()
        e = len(text)
        for ep in end_patterns:
            em = re.search(ep, text[s:s+500000], re.IGNORECASE)
            if em and s + em.start() < e:
                e = s + em.start()
        section = text[s:e].strip()
        candidates.append((len(section), s, section))
    candidates.sort(reverse=True)
    for length, pos, section in candidates:
        if length > 3000:
            risk_words = ["risk", "adverse", "material", "impact", "could"]
            hits = sum(1 for w in risk_words if w.lower() in section[:2000].lower())
            if hits >= 2:
                return section
    rf_positions = [m.start() for m in re.finditer(r'risk\s+factors', text, re.IGNORECASE)]
    best = ""
    for pos in rf_positions[1:]:
        candidate = text[pos:pos+100000]
        if len(candidate) > len(best):
            best = candidate
    if len(best) > 3000:
        return best
    return text[:80000]

def get_relevant(item1a, section_hints, keywords, max_chars=8000):
    paras = [p.strip() for p in re.split(r'\n\s*\n', item1a) if len(p.strip()) > 30]
    best_start = 0
    best_score = 0
    for i, para in enumerate(paras):
        hint_score = sum(3 for h in section_hints if h.lower() in para.lower())
        kw_score = sum(1 for k in keywords if k.lower() in para.lower())
        score = hint_score + kw_score
        if len(para) < 200:
            score *= 2
        if score > best_score:
            best_score = score
            best_start = i
    start = max(0, best_start - 1)
    result_paras = []
    total = 0
    for j in range(start, len(paras)):
        p = paras[j]
        if total + len(p) > max_chars and total > 0:
            break
        result_paras.append(p)
        total += len(p)
    return "\n\n".join(result_paras)[:max_chars]

def main():
    out = Path("extracted_missing")
    out.mkdir(exist_ok=True)

    failed = []
    success = []
    skipped = []

    # Track already processed to avoid duplicate extractions
    processed = set()

    for t in TARGETS:
        ticker = t["ticker"]
        ye, yl = t["years"]
        dim = t["dimension"]
        key = f"{ticker}_{ye}_{yl}_{dim}"

        if key in processed:
            print(f"  Skipping duplicate: {key}")
            skipped.append(key)
            continue
        processed.add(key)

        pair_id = f"{ticker}_10-K_{ye}_{yl}"
        fname = out / f"{pair_id}_{dim}.txt"

        if fname.exists():
            print(f"  Already exists: {fname.name}")
            skipped.append(str(fname.name))
            continue

        print(f"\nProcessing {ticker} {ye}->{yl} ({dim})...")
        try:
            cik = get_cik(ticker)
            if not cik:
                print(f"  ERROR: CIK not found")
                failed.append(f"{ticker}: CIK not found")
                continue
            time.sleep(0.3)

            filings = get_filings(cik, [ye, yl])
            if ye not in filings or yl not in filings:
                missing = [y for y in [ye, yl] if y not in filings]
                print(f"  ERROR: missing filings for {missing}")
                failed.append(f"{ticker}: missing {missing}")
                continue

            texts = {}
            for year in [ye, yl]:
                f = filings[year]
                print(f"  Fetching {year} ({f['date']})...")
                time.sleep(0.6)
                raw = fetch_clean(f["url"])
                item1a = get_item1a(raw)
                relevant = get_relevant(item1a, SECTION_HINTS, KEYWORDS, 8000)
                texts[year] = relevant
                print(f"  Item1A: {len(item1a):,} -> relevant: {len(relevant):,}")

            content = f"""PAIR:      {pair_id}
DIMENSION: {dim}
TICKER:    {ticker} | {ye} -> {yl}

{"="*70}
EARLIER ({ye})
{"="*70}
{texts[ye]}

{"="*70}
LATER ({yl})
{"="*70}
{texts[yl]}

{"="*70}
ANNOTATE
{"="*70}
  direction  : 
  intensity  : 
  confident  : 
  reason     : 
"""
            fname.write_text(content, encoding="utf-8")
            print(f"  Saved: {fname.name}")
            success.append(ticker)

        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed.append(f"{ticker}: {e}")

        time.sleep(1)

    print(f"\n{'='*50}")
    print(f"Success ({len(success)}): {success}")
    print(f"Skipped ({len(skipped)}): {skipped}")
    print(f"Failed  ({len(failed)}): {failed}")
    print(f"Output: {out.resolve()}")

if __name__ == "__main__":
    main()