# Financial Disclosure Risk Intelligence System

> Detecting and validating risk language shifts in SEC regulatory filings  
> using LLM-based semantic analysis and quantitative financial validation.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Status](https://img.shields.io/badge/Status-Active%20Development-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This system analyzes consecutive SEC 10-K and 10-Q filing pairs across  
100+ companies to detect shifts in risk factor language, then validates  
those signals against post-filing market reactions.

It mirrors and automates a core workflow performed manually by credit  
analysts and compliance teams at financial institutions — identifying  
when a company's disclosed risk posture is changing before the market  
fully prices it in.

**Sister project:** [Systemic Risk Knowledge Graph](https://github.com/YOURUSERNAME/financial-risk-knowledge-graph)  
Risk shift signals produced here feed directly into the contagion model there.

---

## Key Results

| Metric | Value |
|--------|-------|
| Companies covered | 100+ across 4 sectors |
| Filing pairs analyzed | 500+ (2019–2024) |
| Cohen's kappa — liquidity risk | TBD |
| Cohen's kappa — credit risk | TBD |
| Panel OLS β — liquidity escalation | TBD |
| Retrieval precision@5 | TBD |
| Answer faithfulness (RAGAS) | TBD |

*Results updated as experiments complete*

---

## System Architecture
SEC EDGAR
│
▼
┌──────────────────────┐
│   Data Ingestion     │  sec-edgar-downloader
│   Item 1A Extraction │  BeautifulSoup, regex
└──────────┬───────────┘
│
▼
┌──────────────────────┐
│   Preprocessing      │  Consecutive pair construction
│   + Cleaning         │  Boilerplate removal, normalization
└──────────┬───────────┘
│
▼
┌──────────────────────┐
│   LLM Risk Detection │  Llama-3 8B, structured output
│   5 Risk Dimensions  │  Direction + Intensity + Justification
└──────────┬───────────┘
│
┌────┴────┐
▼         ▼
┌──────────┐  ┌─────────────────┐
│  Human   │  │   Financial     │
│ Annotated│  │   Validation    │
│  Ground  │  │   Panel OLS     │
│  Truth   │  │   Abnormal Ret  │
│  Kappa   │  │   Sector FE     │
└──────────┘  └─────────────────┘
│
▼
┌──────────────────────┐
│   Streamlit Demo     │  Interactive Q&A + Risk Timeline
└──────────────────────┘

---

## Methodology

### Data Pipeline
- **Source:** SEC EDGAR via `sec-edgar-downloader`
- **Scope:** Item 1A (Risk Factors) section only
- **Coverage:** 100 companies across banking, insurance, technology, energy
- **Period:** 2019 to 2024, covering pre-COVID, COVID shock, recovery, rate hike cycle
- **Pairs:** Consecutive same-quarter filing pairs to control for seasonal language patterns

### Risk Detection
- **Model:** Llama-3 8B with structured output prompting
- **Risk dimensions:** Liquidity, Credit, Operational, Market, Regulatory
- **Output per filing pair:**
  - Direction: escalating, stable, or de-escalating
  - Intensity score: 1 to 5
  - Justification: one sentence explanation per dimension

### Evaluation
- **Ground truth:** 50 manually annotated filing pairs across all 5 risk categories
- **Metric:** Cohen's kappa per category
- **Failure analysis:** Documented edge cases where model disagrees with annotation

### Financial Validation
- **Signal:** 30-day post-filing cumulative abnormal return using S&P 500 as benchmark
- **Model:** Panel OLS regression with sector fixed effects and year fixed effects
- **Hypothesis:** Liquidity risk escalation predicts negative post-filing abnormal returns
- **Upgrade path:** CDS spreads via WRDS when institutional access is confirmed

---

## Project Structure

financial-disclosure-risk-intelligence/
├── src/
│   ├── ingestion/         # SEC EDGAR data collection
│   ├── preprocessing/     # Text cleaning, pair construction
│   ├── modeling/          # LLM risk detection pipeline
│   ├── evaluation/        # Kappa scores, regression analysis
│   └── visualization/     # Streamlit app, Plotly charts
├── data/
│   ├── raw/               # Raw EDGAR filings (gitignored)
│   └── processed/         # Cleaned filing pairs (gitignored)
├── notebooks/             # Exploratory analysis
├── tests/                 # Unit tests
├── requirements.txt
├── .gitignore
└── README.md

---

## Quickstart

```bash
git clone https://github.com/YOURUSERNAME/financial-disclosure-risk-intelligence
cd financial-disclosure-risk-intelligence
pip install -r requirements.txt
cp .env.example .env        # Add your config
python src/ingestion/edgar_downloader.py
```

*Full usage instructions added as modules are completed*

---

## Limitations

- Single annotator ground truth introduces subjectivity; inter-rater reliability not yet established
- Yahoo Finance abnormal return proxies are less precise than CDS spreads pending WRDS access
- LLM structured outputs are non-deterministic across runs; temperature set to 0 for reproducibility
- Coverage limited to US-listed companies with English language filings

---

## Future Work

- CDS spread validation layer via WRDS institutional access
- Real-time EDGAR monitoring pipeline for live filing ingestion
- Integration with sister Knowledge Graph repo for contagion modeling
- Fine-tuning on a finance-specific human annotation dataset
- Multi-annotator ground truth with inter-rater reliability scoring

---

---

## Engineering Notes

This section documents real problems encountered during development
and the decisions made to address them. It exists because honest
documentation of engineering tradeoffs is more valuable for replication or for debugging in the future

---

### Problem 1: Raw Filing Size at Scale (May 2026)

**What happened:**
Initial pipeline downloaded full 10-K and 10-Q filings for 45
companies across 2019 to 2024 using `sec-edgar-downloader`. Total
raw download size reached 22.6GB, exceeding available local disk
space before extraction could complete.

**Root cause:**
Full EDGAR filings include financial statements, exhibits, legal
documents, and appendices in addition to the narrative sections.
We only need Item 1A (Risk Factors), which is typically 8KB to
50KB per filing. Downloading the full filing to extract 0.1% of
its content is wasteful at scale.

**Decision:**
Adopted a batch processing approach rather than redesigning the
ingestion architecture mid-stream. Process one sector at a time:
download sector, extract Item 1A, delete raw files, move to next
sector. This keeps peak disk usage under 8GB at any point while
preserving the existing pipeline code.

**Future improvement:**
Production version should use the SEC EDGAR Full Text Search API
to fetch only the Item 1A section directly, eliminating the raw
download step entirely. This is the approach used by institutional
financial data pipelines and would reduce total disk usage from
~50GB to ~200MB for the full 100-company dataset.

---

### Problem 2: sec-edgar-downloader v5.x API Change (May 2026)

**What happened:**
Initial code used `Downloader(company_name, email_address, save_path)`
constructor which threw `TypeError: unexpected keyword argument 'save_path'`
on first run.

**Root cause:**
`sec-edgar-downloader` v5.x changed the constructor signature,
removing the `save_path` parameter. Files are now saved to
`sec-edgar-filings/` in the current working directory by default.

**Fix:**
Updated constructor to `Downloader(company_name, email_address)`
and added `download_details=True` to the `dl.get()` call as
required by v5.x. Updated `OUTPUT_DIR` reference to match new
default save path.

**Lesson:**
Pin dependency versions in `requirements.txt` to avoid silent
breaking changes. Updated to `sec-edgar-downloader==5.1.0`.

---

### Problem 3: Windows PowerShell Compatibility (May 2026)

**What happened:**
Setup commands using Unix `touch` and `mkdir -p` failed on
Windows PowerShell with `CommandNotFoundException`.

**Fix:**
Replaced all Unix commands with PowerShell equivalents.
Used `New-Item -ItemType File -Force` instead of `touch` and
`New-Item -ItemType Directory -Force` instead of `mkdir -p`.
Added a PowerShell setup script `scripts/setup_windows.ps1`
for reproducibility.

---

### Problem 4: Extractor Picking Wrong File from EDGAR Download (May 2026)

**What happened:**
Item 1A extraction success rate was only 47% (287/600) on first run.
Investigation showed the extractor was selecting `full-submission.txt`
(85MB raw EDGAR wrapper) instead of `primary-document.html` (21MB
actual filing document) because it was taking the largest file.

**Root cause:**
`sec-edgar-downloader` v5.x saves two files per filing:
- `full-submission.txt`: raw EDGAR submission wrapper, not human readable
- `primary-document.html`: actual filing document with readable content

Our file selection logic took the largest file which was always
`full-submission.txt`.

**Fix:**
Updated file selection to explicitly prioritise `primary-document.html`
over all other files. Added HTML tag stripping to the file reader since
primary documents are HTML format. Added explicit skip rule for
`full-submission.txt`.

**Result:**
Extraction success rate improved from 47% to the next iteration.

---

### Problem 5: Item 1A Extractor Matching Table of Contents Instead of Content (May 2026)

**What happened:**
268/600 extractions succeeded after the HTML fix but large bank
10-K filings (JPM, BAC, GS, MS, WFC) still failed. Investigation
showed Item 1A text existed in the stripped document but was not
being extracted.

**Root cause:**
Large bank 10-K filings contain a table of contents where Item 1A
appears as a short entry like "Item 1A. Risk Factors. 7-28" followed
immediately by "Item 1B." Our extractor matched the TOC entry first,
then found Item 1B within 50 characters and extracted almost nothing.
The real Item 1A content section appeared thousands of characters later.

Additionally some extractions were false positives: CFG returning
629 words and ZION returning 399 words when a real Item 1A section
is always at least 3,000 to 5,000 words.

**Fix:**
Added TOC detection: if Item 1B appears within 400 characters of an
Item 1A match, classify it as a TOC entry and skip it. Increased
minimum extraction threshold from 500 characters to 1,000 words to
filter false positives. Fall back to last match position if all
positions appear to be TOC entries.

**Result:**
JPM, BAC, GS, COF, FITB, KEY, PNC, RF, STT, TFC, ZION all began
extracting correctly. Extraction success rate improved significantly.

---

### Problem 6: XBRL Primary Documents and Incorporation by Reference (May 2026)

**What happened:**
MS and C failed with zero Item 1A matches despite full-submission.txt
containing narrative text. WFC and USB failed because their risk
factors are incorporated by reference from separate annual report
documents not downloaded by this pipeline.

**Root cause — two distinct issues:**

Issue A: Companies like MS and C file their 10-K narrative inside
full-submission.txt while primary-document.html is an XBRL data
file created by Workiva. XBRL files start with `<?xml` and contain
`xmlns:` namespace declarations. Our extractor was trying to read
the XBRL file and finding no text content.

Issue B: Companies like WFC and USB use SEC incorporation by
reference. Their 10-K literally states "Risk Factors can be found
in the Annual Report on pages X to Y." The actual content is in a
separate annual report document filed alongside the 10-K but not
downloaded by sec-edgar-downloader as a standalone file.

**Fix for Issue A:**
Added `is_xbrl_file()` detection that checks for `<?xml` prefix
and `xmlns:` namespaces in the first 500 characters. When
primary-document.html is XBRL, fall back to full-submission.txt
and extract the first `<DOCUMENT>` section which contains the
main filing narrative.

**Fix for Issue B:**
No fix applied. Companies using incorporation by reference require
downloading and parsing a separate annual report document. These
filings are documented as known gaps in our dataset.
Affected companies: WFC, USB (banking sector).

**Impact:**
Approximately 15% of banking sector filings use incorporation by
reference and cannot be extracted without downloading additional
documents. Future work: implement annual report downloader for
these specific companies.

---
### Problem 7: 10-Q Filings with No Risk Factor Content (May 2026)

**What happened:**
Many companies succeed on 10-K extraction but fail on all 10-Q
extractions. ALLY, BOKF, WAL, ZION, STT all have complete 10-K
coverage but near-zero 10-Q extractions.

**Root cause:**
SEC rules only require disclosure of material changes in 10-Q
filings. Companies with no material changes write brief statements
like "there have been no material changes to risk factors since
our most recent Annual Report." This is under our 1,000-word
minimum.

**Decision:**
Accepted as expected behavior. 10-K filings are the primary data
source. No change in 10-Q is itself a valid signal.

**Impact:**
Consecutive pair construction primarily uses annual 10-K pairs
supplemented by 10-Q pairs where available. Annual pairs
naturally control for seasonal language patterns.

---

### Dataset Quality Notes

**Sector skew:** Technology companies have cleaner filing formats
and higher extraction rates than banking. Banking has systematic
gaps from incorporation by reference at WFC, USB, and BK.

**Temporal skew:** Dataset is weighted toward annual 10-K filings
due to 10-Q no-change policy. This is methodologically acceptable
since annual pairs are the primary unit of analysis.

**Survivorship bias:** All 100 companies are currently listed or
were listed for the majority of the 2019-2024 period. Companies
that went bankrupt or were fully acquired before 2019 are not
represented. SIVB (Silicon Valley Bank, collapsed March 2023)
is a notable absence that would have been analytically interesting
for liquidity risk escalation detection.

---

## Limitations

- Single annotator ground truth introduces subjectivity
- Yahoo Finance abnormal return proxies less precise than CDS
  spreads pending WRDS summer access confirmation
- LLM outputs non-deterministic across runs; temperature set
  to 0 for reproducibility
- Coverage limited to US-listed companies with English filings
- Incorporation by reference gaps affect approximately 15% of
  banking sector filings

---

## Future Work

- CDS spread validation via WRDS Markit when access confirmed
- Real-time EDGAR monitoring pipeline for live filing ingestion
- Integration with sister Knowledge Graph repo for contagion modeling
- Fine-tuning on finance-specific annotation dataset
- Multi-annotator ground truth with inter-rater reliability scoring
- SEC EDGAR Full Text Search API to eliminate raw download step
- SIVB and First Republic Bank as stress-period validation cases

---
**Note for contributors:**
This project was developed on Windows. All shell commands in
this README use PowerShell syntax. Unix/Mac equivalents are
standard bash commands.

## Author

**Sudhiksha Kandavel Rajan**  
MS Artificial Intelligence, Northeastern University  
[LinkedIn](https://www.linkedin.com/in/sudhiksha-kandavel-rajan-71b4651b5/) · [GitHub](https://github.com/Sudhiksha-17) · [HUX AI Paper](https://arxiv.org/abs/2407.19492)
