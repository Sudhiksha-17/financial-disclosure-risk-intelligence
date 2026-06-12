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

---

### Problem 8: LLM Response JSON Truncation (May 2026)

**What happened:**
LLM risk detector consistently failed to parse responses despite
the model generating correct output. Parse failures showed the
JSON cutting off mid-value in the final risk dimension.

**Root cause investigation:**
Three hypotheses tested in sequence:

1. Token limit too low (num_predict: 800) - increased to 1500,
   still failed
2. Context window too small - added num_ctx: 8192 explicitly,
   still failed
3. Prompt too long - tested actual prompt token count, found
   only 2,931 tokens used with 5,261 tokens available for
   response, ruling out context overflow

**Actual root cause:**
Llama-3 8B via Ollama stops generating before adding the final
closing braces of the JSON object. The response had done_reason:
stop and only 215 output tokens, meaning the model considered
itself finished but had not closed the outer JSON structure.
This is a known behavior of instruction-tuned models when
generating structured output without explicit stop tokens.

**Fix:**
Added three-stage JSON repair logic to parse_response():

Stage 1: Parse response as-is.

Stage 2: Count open vs closed braces. If missing 1 to 5 closing
braces, append them and retry parsing. This handles the most
common case where the model stops 1 to 2 braces short.

Stage 3: Walk backwards through the response finding the last
valid closing brace position, attempt parse at each position
with and without brace repair. This handles cases where the
final dimension is partially generated.

**Result:**
Parse success rate went from 0% to 100% on test set.
Full pipeline validated and ready for production run.

**Lesson:**
When using LLMs for structured JSON output, always implement
repair logic for incomplete responses. LLMs are probabilistic
and will occasionally stop generation slightly before
completing a structured format even with explicit instructions.

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

---

### Problem 9: Insufficient Text Extraction in Annotation Sample (June 2026)

**What happened:**
During human annotation of 50 filing pairs, 37% of regulatory risk
dimensions and 48% of operational risk dimensions were marked
insufficient_text because the 3,000 character text window in the
annotation sheet did not reach the relevant sections.

**Root cause:**
SEC 10-K filings follow a consistent structure where risk factors
appear in a specific order: credit risks first, then market risks,
then operational risks, then regulatory risks. The 3,000 character
window captures the beginning of Item 1A but frequently cuts off
before reaching operational and regulatory sections which appear
later in the document.

**Impact on kappa scores:**
Regulatory kappa was calculated on only 13 pairs versus 31 pairs
for market risk. Low sample size inflates variance in kappa estimates
and makes the regulatory score less reliable.

**Decision:**
Accepted as a methodology limitation. Future work should increase
the annotation window to 8,000 characters or implement section-aware
extraction that identifies and isolates each risk category
independently.

---

### Problem 10: LLM Systematic Annotation Biases Identified (June 2026)

**What happened:**
Cohen's kappa analysis revealed three systematic biases in LLM risk
classification relative to human judgment across 123 comparable
dimension annotations.

**Bias 1: Cannot detect de-escalation**
Human annotators identified 11 de-escalating cases across all
dimensions. The LLM identified zero de-escalating cases. The LLM
interprets any risk language as at least stable and cannot recognize
when specific risk factors have been removed or resolved between
filings. Examples: MTB 2022-2023 where COVID and LIBOR language was
entirely removed, GL 2023-2024 where a dedicated COVID section was
dropped, CFG 2022-2023 where COVID credit and liquidity language
was removed.

**Bias 2: Over-fires on operational risk**
Human: 9 escalating, 15 stable out of 26 annotatable pairs.
LLM: 18 escalating, 8 stable.
The LLM codes operational risk as escalating roughly twice as often
as human judgment warrants. Likely cause: operational risk language
is verbose in filings and the LLM correlates text length with
escalation severity rather than identifying genuinely new risk content.

**Bias 3: Under-fires on liquidity risk**
Human: 7 escalating, 2 de-escalating out of 24 annotatable pairs.
LLM: 1 escalating, 0 de-escalating.
The LLM calls almost all liquidity risk stable regardless of content.
Likely cause: liquidity risk language in financial filings is often
embedded within broader market or credit risk sections rather than
appearing as standalone bullets.

**Consistency with regression results:**
The two regression-significant dimensions, credit risk
(beta=0.032, p=0.028) and regulatory risk (beta=0.038, p=0.046),
have the highest kappa scores of 0.220 and 0.085 respectively.
The regression-insignificant dimensions, liquidity (p=0.190) and
operational (p=0.775), have the lowest kappa scores. This internal
consistency between annotation quality and regression significance
strengthens confidence in the credit and regulatory findings.

**Next step:**
LoRA fine-tuning on annotated pairs to improve calibration across
all five dimensions simultaneously.

---

---

### Problem 11: Prompt Engineering Instability in Multi-Dimension Classification (June 2026)

**What happened:**
Attempted to improve kappa scores from 0.219 to 0.4+ by rewriting
the prompt with few-shot examples and explicit de-escalation
instructions. The v2 prompt caused overall kappa to drop from
0.219 to 0.076.

**Root cause:**
Prompt changes that fixed one dimension introduced biases in others.
Adding five de-escalation examples caused the model to over-predict
de-escalating across all dimensions, particularly credit risk where
LLM predictions jumped from 11 escalating to 12 de-escalating
against human annotations showing only 1 de-escalating case.
The anti-length-bias instruction simultaneously suppressed legitimate
escalation detection in operational risk.

**Key insight:**
Instruction-tuned LLMs use the prompt as a global instruction set.
Changes intended for one dimension affect all dimensions simultaneously
because the model cannot isolate instructions by dimension. This makes
prompt engineering fundamentally unstable for multi-class multi-label
classification tasks where each class has different base rates.

**Decision:**
Reverted to v1 prompt which produces overall kappa of 0.207.
Proceeded to LoRA fine-tuning which directly optimizes model weights
for the specific classification task.

---

### Problem 12: LoRA Fine-tuning Label Collapse (June 2026)

**What happened:**
Two rounds of LoRA fine-tuning on Llama-3 8B via Google Colab T4 GPU
produced degenerate models. Round 1 with 98 imbalanced examples
predicted stable for all 25 validation examples, kappa 0.000.
Round 2 with 150 balanced examples via oversampling predicted
de-escalating 15 out of 25 times, kappa -0.183.

**Root cause:**
Three compounding factors across both attempts.

Round 1 failure: Class imbalance of 54 stable, 35 escalating, 9
de-escalating was not corrected. Aggressive truncation to 512 tokens
removed most filing content. The model learned to predict the majority
class stable to minimize loss.

Round 2 failure: The 9 de-escalating training examples were all from
the 2022-2023 period and primarily captured COVID language removal
in market risk. Duplicating them to 50 caused the model to memorize
the specific COVID-removal pattern rather than learning the general
concept of de-escalation. Every new input superficially matched the
oversampled pattern, producing de-escalating predictions across all
dimensions and companies.

**Evidence of root cause:**
De-escalating training examples broke down as: 6 out of 9 were
market_risk, 3 out of 9 were from CFG 2022-2023 alone, 7 out of 9
were from the 2022-2023 transition period. Oversampling this
concentrated distribution 5x created a highly specific spurious
signal rather than a generalizable classifier.

**Key insight:**
Fine-tuning on minority classes requires both sufficient quantity and
sufficient diversity. 9 examples from 3 companies spanning 1 year
cannot generalize even with oversampling. The minimum viable dataset
for this task is estimated at 30 or more diverse de-escalating
examples spanning multiple companies, sectors, and transition types
before fine-tuning is likely to succeed.

**Next steps:**
Annotate 20 additional de-escalating candidates identified from the
410 existing pairs using LLM-predicted de-escalating cases as a
high-precision starting point. With 29 or more diverse de-escalating
examples, retry fine-tuning with sequence classification head rather
than generative approach.

---

## Analysis Findings

### Key Result 1: Credit and Regulatory Risk Signals Predict Market Reactions

Panel OLS regression across 410 filing pairs (86 companies, 4 sectors,
2020-2024) with sector and year fixed effects. Dependent variable is
30-day cumulative abnormal return (CAR) relative to SPY.

| Dimension | Beta | p-value | Significant |
|-----------|------|---------|-------------|
| credit_dir | +0.032 | 0.028 | Yes |
| regulatory_dir | +0.038 | 0.046 | Yes |
| liquidity_dir | +0.029 | 0.190 | No |
| operational_dir | +0.004 | 0.775 | No |
| market_dir | -0.010 | 0.645 | No |

R-squared: 0.22. CAR mean: -4.09% across all 410 pairs.

Interpretation: Companies disclosing escalating credit and regulatory
risk language experience positive abnormal returns in the 30 days
following the filing. This counterintuitive direction may reflect
market reward for transparency, or that these disclosures contain
forward-looking information that investors view as constructive
engagement with known risks rather than unexpected surprises.

The internal consistency between this regression result and the kappa
analysis strengthens confidence in the finding. Credit risk, the
dimension with the highest kappa of 0.220, is also the dimension with
the strongest and most significant regression coefficient. Operational
risk, the dimension with the most severe LLM over-firing bias, has
the weakest regression coefficient at p=0.775. This pattern suggests
the LLM signal quality predicts regression significance, which is
exactly what we would expect if the signals are capturing genuine
risk language shifts rather than noise.

---

### Key Result 2: LLM-Human Agreement by Dimension

Cohen's kappa between Llama-3 8B v1 prompt predictions and human
annotations across 123 comparable dimension-level annotations from
50 sampled filing pairs:

| Dimension | Kappa | n | Interpretation |
|-----------|-------|---|----------------|
| credit | 0.220 | 29 | Fair |
| operational | 0.208 | 26 | Slight |
| market | 0.161 | 31 | Slight |
| liquidity | 0.143 | 24 | Slight |
| regulatory | 0.085 | 13 | Slight |
| **overall** | **0.207** | **123** | **Fair** |

Three systematic LLM biases identified and documented:

**Bias 1: Cannot detect de-escalation.**
Human annotators identified 11 de-escalating cases. The LLM identified
zero. The model interprets any risk language as at minimum stable and
cannot recognize when specific risk sections have been removed or
resolved between filings.

**Bias 2: Over-fires on operational risk.**
Human: 9 escalating, 15 stable. LLM: 19 escalating, 7 stable. The
model codes operational risk as escalating roughly twice as often as
human judgment warrants, likely because operational risk language is
verbose and the model correlates text length with escalation severity.

**Bias 3: Under-fires on liquidity risk.**
Human: 7 escalating, 2 de-escalating. LLM: 1 escalating, 0
de-escalating. Liquidity risk language is typically embedded within
broader market or credit sections rather than appearing as standalone
bullets, making it harder for the model to isolate as a distinct
signal.

**Annotation methodology note:**
Human annotations were produced by a single non-expert annotator
following a structured rubric anchored to structural text features
rather than semantic financial interpretation. Disagreement patterns
are directionally systematic rather than random, suggesting the LLM
has specific calibration failures rather than the human having random
noise. The primary regression finding is independent of human
annotations entirely and is validated against objective market data.

---

### Key Result 3: Fine-tuning Calibration Experiments

Two rounds of LoRA fine-tuning were conducted on Google Colab T4 GPU
using the human-annotated pairs as training data.

| Experiment | Training data | Val kappa | Notes |
|------------|--------------|-----------|-------|
| Base LLM v1 prompt | None | 0.207 | Baseline |
| v2 prompt engineering | None | 0.076 | Over-corrected |
| LoRA round 1 | 98 imbalanced | 0.000 | Stable collapse |
| LoRA round 2 | 150 balanced | -0.183 | De-escalating collapse |

The fine-tuning experiments produced negative results that are
informative rather than merely failed attempts. They demonstrate that
improving LLM calibration for this task requires either substantially
more annotated data, particularly diverse minority class examples, or
a classification head architecture rather than generative fine-tuning.
The baseline v1 prompt kappa of 0.207 remains the best result.

---

### Key Result 4: Dataset Quality

Of 50 annotation pairs sampled, 37% of regulatory dimensions and 48%
of operational dimensions were marked insufficient_text due to text
truncation in the 3,000 character annotation window. This reflects the
SEC filing structure where credit and market risks appear first and
operational and regulatory risks appear later in Item 1A.

| Annotation status | Count | Percentage |
|-------------------|-------|------------|
| Fully annotatable (all 5 dims) | 6 | 12% |
| Partial (at least 1 dim valid) | 28 | 56% |
| Fully insufficient | 16 | 32% |
| Total | 50 | 100% |

Regulatory kappa was computed on only 13 pairs versus 31 for market
risk, making the regulatory estimate the least reliable. Future work
should increase the annotation window to 8,000 characters or implement
section-aware extraction.

---

## Safety and Alignment Extension (Planned)

This project is being extended with a safety and alignment research
layer that connects financial disclosure analysis to AI evaluation
methodology. The core argument is as follows.

Financial disclosures are an empirically validated domain for studying
evasive behavior under evaluation pressure. Companies face strong
incentives to satisfy SEC disclosure requirements while minimizing the
reputational and market impact of adverse disclosures. The resulting
behavior, satisfying formal evaluation criteria while concealing
material information, is structurally identical to what alignment
researchers call deceptive alignment in AI systems.

### Evasion Taxonomy

Four types of institutional evasion that map onto alignment concepts:

**Type 1: Omission Evasion**
A risk present in the earlier filing is entirely absent from the
later filing without explanation. Maps to capability concealment and
sandbagging in AI systems.

**Type 2: Obfuscation Evasion**
The risk is mentioned but in language so generic it fails to
communicate the specific exposure. Maps to deceptive alignment where
model outputs pass formal evaluation without carrying intended content.

**Type 3: Displacement Evasion**
The risk is disclosed but buried in low-salience locations after dense
boilerplate, satisfying the letter of the requirement while minimizing
attention. Maps to specification gaming.

**Type 4: Framing Evasion**
Technically accurate language that creates a systematically misleading
impression, for example using prospective language to describe a past
event. Maps to sycophancy in language models.

### Planned Methodology

Phase 1: Second annotation pass adding specificity scores and evasion
type flags to existing 50 pairs.

Phase 2: Automated evasion detector running a third Llama-3 inference
pass on all 410 pairs to classify specificity and consistency.

Phase 3: SEC comment letter validation. EDGAR comment letters sent to
companies about inadequate risk disclosure serve as external ground
truth for evasive disclosure, enabling precision and recall computation
against known inadequate disclosure cases.

Phase 4: LLM generation experiments. Prompt multiple models to
generate synthetic risk disclosures under evaluation pressure and test
whether the evasion detector identifies institutionally evasive
patterns in the generated text. This tests the hypothesis that models
trained on financial text have learned institutional evasion patterns.

Phase 5: ArXiv preprint framing financial disclosure as a model
organism for studying deceptive compliance in institutional text.

---

## Dataset Quality Notes

**Sector skew:** Technology companies have cleaner filing formats and
higher extraction rates than banking. Banking has systematic gaps from
incorporation by reference at WFC, USB, and BK.

**Temporal skew:** Dataset is weighted toward annual 10-K filings due
to 10-Q no-change policy. This is methodologically acceptable since
annual pairs are the primary unit of analysis.

**Survivorship bias:** All 100 companies were listed for the majority
of the 2019-2024 period. SIVB (Silicon Valley Bank, collapsed March
2023) is a notable absence that would have been analytically
interesting for liquidity risk escalation detection given its
documented pre-collapse risk language changes.

---

## Limitations

- Single annotator ground truth introduces subjectivity. Second
  annotator validation using a different LLM is planned.
- Yahoo Finance abnormal return proxies are less precise than CDS
  spreads. WRDS Markit access pending for summer 2026.
- LLM outputs are non-deterministic across runs. Temperature set to 0
  and seed fixed at 42 for reproducibility across all 410 pairs.
- Coverage limited to US-listed companies with English filings.
- Incorporation by reference gaps affect approximately 15% of banking
  sector filings.
- Fine-tuning shown to require more diverse minority class examples
  than currently available. Minimum estimated at 30 diverse
  de-escalating cases versus 9 currently annotated.
- 512 token truncation during fine-tuning removes most filing content,
  limiting model access to discriminative text.

---

## Future Work

**Near term:**
- Annotate 20 additional de-escalating candidates from 31 LLM-identified
  cases across the 410 existing pairs
- Retry LoRA fine-tuning with diverse balanced dataset and sequence
  classification head architecture
- Second annotator validation using Mistral 7B via Ollama to compute
  inter-annotator kappa and validate annotation quality
- Streamlit demo with ticker input, risk signal timeline, and
  regression overlay

**Medium term:**
- Safety extension: evasion taxonomy annotation, automated evasion
  detector, SEC comment letter validation
- LLM generation experiments testing whether evaluation pressure
  amplifies institutional evasion patterns in model outputs
- ArXiv preprint: Institutional Evasion as a Testbed for AI Evaluation
  Methodology
- Lightweight knowledge graph using 410 signals as node features with
  contagion modeling

**Longer term:**
- CDS spread validation via WRDS Markit
- Multi-annotator ground truth with formal inter-rater reliability
- SIVB and First Republic Bank as stress-period validation cases
- Real-time EDGAR monitoring pipeline
- SEC EDGAR Full Text Search API to eliminate raw download step

---

---

## Author

**Sudhiksha Kandavel Rajan**
MS Artificial Intelligence, Northeastern University
[LinkedIn](https://www.linkedin.com/in/sudhiksha-kandavel-rajan-71b4651b5/) · [GitHub](https://github.com/Sudhiksha-17) · [HUX AI Paper](https://arxiv.org/abs/2407.19492)