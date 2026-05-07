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

**Note for contributors:**
This project was developed on Windows. All shell commands in
this README use PowerShell syntax. Unix/Mac equivalents are
standard bash commands.

## Author

**Sudhiksha Kandavel Rajan**  
MS Artificial Intelligence, Northeastern University  
[LinkedIn](https://www.linkedin.com/in/sudhiksha-kandavel-rajan-71b4651b5/) · [GitHub](https://github.com/Sudhiksha-17) · [HUX AI Paper](https://arxiv.org/abs/2407.19492)
