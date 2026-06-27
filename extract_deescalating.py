"""
EDGAR 10-K Risk Factor Extractor v3
Fixed Item 1A extraction for XBRL-heavy filings
"""

import requests
import re
import time
from pathlib import Path

HEADERS = {"User-Agent": "sudhiksha.research@gmail.com"}

TARGETS = [
    {"ticker": "DAL", "company": "Delta Air Lines", "dimension": "credit_risk",
     "years": [2021, 2022], "keywords": ["credit", "counterparty", "customer", "financial institution", "default", "liquidity"]},
    {"ticker": "UAL", "company": "United Airlines", "dimension": "credit_risk",
     "years": [2021, 2022], "keywords": ["credit", "counterparty", "customer", "financial institution", "default", "liquidity"]},
    {"ticker": "MCD", "company": "McDonald's", "dimension": "operational_risk",
     "years": [2021, 2022], "keywords": ["supply chain", "labor", "franchise", "operations", "disruption", "COVID"]},
    {"ticker": "SBUX", "company": "Starbucks", "dimension": "operational_risk",
     "years": [2021, 2022], "keywords": ["supply chain", "labor", "operations", "disruption", "COVID", "staffing"]},
]

def get_cik(ticker):
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers=HEADERS)
    data = r.json()
    for entry in data.values():
        if entry["ticker"].upper() == ticker.upper():
            cik = str(entry["cik_str"]).zfill(10)
            print(f"  Found {ticker}: CIK={cik}, Name={entry['title']}")
            return cik
    return None

def get_10k_filings(cik, years):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = requests.get(url, headers=HEADERS)
    data = r.json()

    filings = data.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    dates = filings.get("filingDate", [])
    accessions = filings.get("accessionNumber", [])
    primary_docs = filings.get("primaryDocument", [])

    results = {}
    for i, form in enumerate(forms):
        if form == "10-K":
            date = dates[i]
            month = int(date[5:7])
            year = int(date[:4])
            fiscal_year = year - 1 if month <= 4 else year
            if fiscal_year in years and fiscal_year not in results:
                acc_clean = accessions[i].replace("-", "")
                cik_int = int(cik)
                primary = primary_docs[i] if i < len(primary_docs) else ""
                results[fiscal_year] = {
                    "accession": accessions[i],
                    "acc_clean": acc_clean,
                    "filing_date": date,
                    "primary_doc": primary,
                    "base_url": f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/"
                }
    return results

def get_document_url(filing):
    if filing["primary_doc"]:
        return filing["base_url"] + filing["primary_doc"]
    return None

def strip_html_and_xbrl(raw_html):
    """
    Remove all HTML/XML tags including XBRL ix: tags,
    decode entities, collapse whitespace.
    """
    # Remove XBRL namespace tags like <ix:nonnumeric ...> </ix:nonnumeric>
    text = re.sub(r'<ix:[^>]+>', ' ', raw_html, flags=re.IGNORECASE)
    text = re.sub(r'</ix:[^>]+>', ' ', text, flags=re.IGNORECASE)
    # Remove all remaining HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Decode common HTML entities
    entities = {
        '&amp;': '&', '&lt;': '<', '&gt;': '>',
        '&nbsp;': ' ', '&quot;': '"', '&#8226;': '•',
        '&#8220;': '"', '&#8221;': '"', '&#8217;': "'",
        '&#160;': ' ',
    }
    for ent, char in entities.items():
        text = text.replace(ent, char)
    # Remove numeric entities
    text = re.sub(r'&#\d+;', ' ', text)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    # Collapse whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_item1a(clean_text):
    """
    Extract Item 1A section from cleaned plain text.
    Tries multiple boundary patterns.
    """
    patterns = [
        # Standard Item 1A to Item 1B
        (r"ITEM\s+1A[\.\s]+RISK\s+FACTORS(.*?)ITEM\s+1B", re.DOTALL | re.IGNORECASE),
        (r"Item\s+1A[\.\s]+Risk\s+Factors(.*?)Item\s+1B", re.DOTALL | re.IGNORECASE),
        # Item 1A to Item 2
        (r"ITEM\s+1A[\.\s]+RISK\s+FACTORS(.*?)ITEM\s+2[\.\s]", re.DOTALL | re.IGNORECASE),
        (r"Item\s+1A[\.\s]+Risk\s+Factors(.*?)Item\s+2[\.\s]", re.DOTALL | re.IGNORECASE),
        # Looser: just find "Risk Factors" header and take next 40k chars
        (r"RISK\s+FACTORS\s*\n(.*?)(?:ITEM\s+[12]B?[\.\s]|UNRESOLVED\s+STAFF)", re.DOTALL | re.IGNORECASE),
    ]

    for pattern, flags in patterns:
        match = re.search(pattern, clean_text, flags)
        if match:
            section = match.group(1).strip()
            if len(section) > 1000:
                print(f"    Item 1A found: {len(section)} chars")
                return section

    # Last resort: find the phrase and take a window
    idx = clean_text.lower().find("risk factors")
    if idx > 0:
        window = clean_text[idx:idx+60000]
        print(f"    Item 1A fallback window: {len(window)} chars")
        return window

    return clean_text[:50000]

def extract_relevant_subsection(item1a_text, keywords, max_chars=4000):
    """Extract paragraphs most relevant to dimension keywords"""
    paragraphs = re.split(r'\n\s*\n', item1a_text)
    paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 80]

    scored = []
    for i, para in enumerate(paragraphs):
        score = sum(1 for kw in keywords if kw.lower() in para.lower())
        if score > 0:
            scored.append((score, i, para))

    scored.sort(reverse=True)

    selected_indices = set()
    total = 0
    for score, idx, para in scored:
        if total >= max_chars:
            break
        for j in [idx - 1, idx, idx + 1]:
            if 0 <= j < len(paragraphs) and j not in selected_indices:
                selected_indices.add(j)
                total += len(paragraphs[j])

    selected = sorted(selected_indices)
    return "\n\n".join(paragraphs[j] for j in selected)

def process_target(target, output_dir):
    ticker = target["ticker"]
    print(f"\nProcessing {ticker}...")

    cik = get_cik(ticker)
    if not cik:
        print(f"  ERROR: CIK not found")
        return
    time.sleep(0.5)

    filings = get_10k_filings(cik, target["years"])
    print(f"  Found filings for years: {sorted(filings.keys())}")

    year_earlier, year_later = target["years"]
    texts = {}

    for year in [year_earlier, year_later]:
        if year not in filings:
            print(f"  ERROR: Missing {year} filing")
            return

        filing = filings[year]
        print(f"  Getting {year} filing (filed {filing['filing_date']})...")

        doc_url = get_document_url(filing)
        if not doc_url:
            print(f"  ERROR: No document URL")
            return

        print(f"  Fetching: {doc_url[:90]}...")
        time.sleep(0.5)

        try:
            r = requests.get(doc_url, headers=HEADERS, timeout=60)
            r.raise_for_status()
        except Exception as e:
            print(f"  ERROR: {e}")
            return

        clean = strip_html_and_xbrl(r.text)
        item1a = extract_item1a(clean)
        relevant = extract_relevant_subsection(item1a, target["keywords"])

        if len(relevant) < 200:
            print(f"  WARNING: Very little relevant text found ({len(relevant)} chars)")

        texts[year] = relevant
        print(f"  Final extract: {len(relevant)} chars for {year}")

    pair_id = f"{ticker}_10-K_{year_earlier}_{year_later}"
    content = f"""PAIR:      {pair_id}
DIMENSION: {target["dimension"]}
TICKER:    {ticker} | {year_earlier} -> {year_later}
{'='*70}
EARLIER ({year_earlier})
{'='*70}
{texts[year_earlier]}
{'='*70}
LATER ({year_later})
{'='*70}
{texts[year_later]}
{'='*70}
ANNOTATE
{'='*70}
  direction  : 
  intensity  : 
  confident  : 
  reason     : 
"""
    out_file = output_dir / f"{pair_id}_{target['dimension']}.txt"
    out_file.write_text(content, encoding="utf-8")
    print(f"  Saved: {out_file.name}")

def main():
    output_dir = Path("deescalating_candidates")
    output_dir.mkdir(exist_ok=True)

    for target in TARGETS:
        try:
            process_target(target, output_dir)
        except Exception as e:
            print(f"  FAILED {target['ticker']}: {e}")
        time.sleep(1)

    print(f"\nDone. Files in: {output_dir.resolve()}")

if __name__ == "__main__":
    main()