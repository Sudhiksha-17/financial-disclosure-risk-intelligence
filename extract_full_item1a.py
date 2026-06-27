"""
Extract Full Item1A
====================
Re-extracts only the 4 low-signal pairs with full Item1A text (no 8000-char truncation).
Also re-extracts any pair where added+removed < 20 (low diff signal threshold).

Saves to extracted_full/ folder.
Pairs to fix: EXPE, HLT, LYFT, VLO (and any others with low signal)
"""

import requests
import re
import time
from pathlib import Path

HEADERS = {"User-Agent": "sudhiksha.research@gmail.com"}

# Only the low-signal pairs that need full extraction
TARGETS = [
    {"ticker": "EXPE", "dimension": "operational_risk", "years": [2021, 2022]},
    {"ticker": "HLT",  "dimension": "operational_risk", "years": [2021, 2022]},
    {"ticker": "LYFT", "dimension": "operational_risk", "years": [2021, 2022]},
    {"ticker": "VLO",  "dimension": "credit_risk",      "years": [2021, 2022]},
    # Also re-extract these which had borderline signal
    {"ticker": "CINF", "dimension": "regulatory_risk",  "years": [2020, 2021]},
    {"ticker": "CINF", "dimension": "credit_risk",      "years": [2022, 2023]},
    {"ticker": "FITB", "dimension": "regulatory_risk",  "years": [2022, 2023]},
    {"ticker": "WBD",  "dimension": "operational_risk", "years": [2021, 2022]},
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

def get_item1a_full(text, max_chars=30000):
    """Get full Item1A section without keyword-based truncation."""
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
                # Return full section up to max_chars, no keyword truncation
                return section[:max_chars]
    return text[:max_chars]

def main():
    out = Path("extracted_full")
    out.mkdir(exist_ok=True)
    success, failed = [], []

    for t in TARGETS:
        ticker = t["ticker"]
        ye, yl = t["years"]
        dim = t["dimension"]
        pair_id = f"{ticker}_10-K_{ye}_{yl}"
        fname = out / f"{pair_id}_{dim}.txt"

        print(f"\nProcessing {ticker} {ye}->{yl} ({dim})...")
        try:
            cik = get_cik(ticker)
            if not cik:
                failed.append(f"{ticker}: CIK not found")
                continue
            time.sleep(0.3)

            filings = get_filings(cik, [ye, yl])
            if ye not in filings or yl not in filings:
                failed.append(f"{ticker}: missing filings")
                continue

            texts = {}
            for year in [ye, yl]:
                f = filings[year]
                print(f"  Fetching {year} ({f['date']})...")
                time.sleep(0.6)
                raw = fetch_clean(f["url"])
                # Use full Item1A, no keyword truncation
                full_text = get_item1a_full(raw, max_chars=30000)
                texts[year] = full_text
                print(f"  Full Item1A: {len(full_text):,} chars")

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
            traceback.print_exc()
            failed.append(f"{ticker}: {e}")
        time.sleep(1)

    print(f"\nSuccess ({len(success)}): {success}")
    print(f"Failed  ({len(failed)}): {failed}")

if __name__ == "__main__":
    main()