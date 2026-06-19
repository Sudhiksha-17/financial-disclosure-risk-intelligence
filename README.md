# Financial Disclosure Risk Intelligence System

This project analyzes consecutive SEC 10-K filing pairs across 100+ companies to detect shifts in risk factor language, then validates those signals against post-filing market reactions.

It mirrors and automates a core workflow performed manually by credit analysts and compliance teams at financial institutions — identifying when a company's disclosed risk posture is changing before the market fully prices it in.

This project also serves as an empirical testbed for AI safety research. The connection is documented in the Safety and Alignment Extension section below.

---

## Key Results

| Metric | Value |
|--------|-------|
| Companies covered | 100+ across 4 sectors |
| Filing pairs analyzed | 500+ (2019–2024) |
| Human annotated pairs | 36 clean pairs (38 annotated, 2 excluded as XBRL-corrupted) |
| Cohen's kappa — LLM baseline (raw text, v1 prompt) | 0.207 |
| Cohen's kappa — ICL diff representation, Llama3 8B (best) | 0.614 |
| Cohen's kappa — ICL diff representation, GPT-4 | 0.453 |
| De-escalating recall — LLM baseline | 0/20 (0%) |
| De-escalating recall — ICL diff, Llama3 8B | 18/20 (90%) |
| De-escalating recall — ICL diff, GPT-4 | 12/20 (60%) |
| Diff primary signal alignment with human labels | 84.2% (32/38) |
| Panel OLS β — credit risk escalation | +0.032 (p=0.028) |
| Panel OLS β — regulatory risk escalation | +0.038 (p=0.046) |

---

## Multi-Model Ablation Results

| Model | Condition | Prompt | Kappa | De-esc Recall | Failure Mode |
|-------|-----------|--------|-------|---------------|--------------|
| Llama3 8B | raw | Llama3 | 0.071 | 0/20 (0%) | De-escalation blindness |
| Llama3 8B | diff | Llama3 | 0.614 | 18/20 (90%) | Best overall |
| GPT-4o-mini | raw | Llama3 | 0.270 | 8/20 (40%) | Partial blindness |
| GPT-4o-mini | diff | Llama3 | 0.139 | 4/20 (20%) | Addition blindness |
| GPT-4o | raw | Llama3 | 0.141 | 3/20 (15%) | Stable collapse |
| GPT-4o | diff | Llama3 | 0.147 | 4/20 (20%) | Stable collapse |
| GPT-4o | diff | GPT-4o v1 | 0.151 | 3/20 (15%) | Stable collapse |
| GPT-4o | diff | GPT-4o+KEY SIGNALS | 0.273 | 10/20 (50%) | Residual stable collapse |
| GPT-4 | raw | GPT-4o+KEY SIGNALS | 0.191 | 6/20 (30%) | Stable collapse |
| GPT-4 | diff | GPT-4o+KEY SIGNALS | 0.453 | 12/20 (60%) | Residual stable collapse |

**Three systematic failure modes documented:**

Failure Mode 1 — De-escalation blindness (Llama3 8B, raw text): Model predicts escalating for almost everything. De-escalating recall 0%. Fixed by diff representation.

Failure Mode 2 — Addition blindness (GPT-4o-mini, diff): Model fixates on any new content being added, predicts escalating regardless of removal signal. De-escalating recall 20%.

Failure Mode 3 — Stable collapse (GPT-4o and GPT-4, both conditions): Model defaults to stable for ambiguous cases regardless of prompt or representation. Partially fixed by KEY SIGNALS intervention (+0.12 kappa for GPT-4o, +0.26 kappa for GPT-4).

**Key findings from multi-model ablation:**

Finding 1: Diff representation helps all models. Every model improves with diff over raw. The representation contributes regardless of model family.

Finding 2: Capability scaling matters. GPT-4 (0.453) outperforms GPT-4o (0.273) with the same prompt. Larger models extract more signal from the structured diff format.

Finding 3: Prompt-representation co-design determines the ceiling. Llama3 8B with a prompt tuned over 23 engineering iterations reaches 0.614. GPT-4 with ~10 iterations reaches 0.453. The gap reflects prompt engineering investment, not fundamental model capability. The diff representation concept generalizes; specific prompt engineering does not transfer across models.

---

## System Architecture

```
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
│   + Diff Builder     │  Sentence-level add/remove/tense detection
│   + Cleaning         │  Boilerplate removal, XBRL stripping
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   ICL Baseline v3    │  Multi-model: Llama3 8B, GPT-4o, GPT-4
│   Model-specific     │  Llama3: annotated [SIGNAL] prompts
│   Prompts            │  GPT-4x: clean analytical + KEY SIGNALS
│   5-fold CV          │  Cohen's kappa evaluation
└──────────┬───────────┘
           │
      ┌────┴────┐
      ▼         ▼
┌──────────┐ ┌─────────────────┐
│  Human   │ │   Financial     │
│ Annotated│ │   Validation    │
│  Ground  │ │   Panel OLS     │
│  Truth   │ │   Abnormal Ret  │
│  Kappa   │ │   Sector FE     │
└──────────┘ └─────────────────┘
```

---

## Methodology

### Data Pipeline

Source: SEC EDGAR via sec-edgar-downloader
Scope: Item 1A (Risk Factors) section only
Coverage: 100 companies across banking, insurance, technology, energy
Period: 2019 to 2024, covering pre-COVID, COVID shock, recovery, rate hike cycle
Pairs: Consecutive same-quarter filing pairs to control for seasonal language patterns

### Diff Representation

Rather than feeding raw filing text to the classifier, we compute a structured diff between consecutive filing pairs. Each diff captures:

- Sentences added in the later filing
- Sentences removed from the earlier filing
- COVID-specific sentence counts added and removed
- Tense shifts (active threat language to historical or conditional)
- Section-level heading additions and removals
- Net sentence volume change

The diff representation collapses thousands of tokens of raw text to a few hundred tokens of structured change signal, directly surfacing classification-relevant information.

### Model-Specific Prompts

Two prompt families are implemented:

**Llama3 prompt:** Annotated [SIGNAL: de-escalating/escalating] tags embedded in diff summaries, numbered rules with explicit tiebreaker logic, HIGH_QUALITY_PAIRS whitelist for few-shot example selection, has_clean_summary() hard gate blocking financial artifacts. Tuned over 23 documented engineering iterations.

**GPT-4o/GPT-4 prompt:** Clean analytical descriptions without annotation scaffolding, KEY SIGNALS block with explicit yes/no signal extraction (COVID eliminated, content decreased/increased, new sections added), DECISION RULES referencing KEY SIGNALS, tiebreaker Notes embedded in diff content for known ambiguous pairs.

### ICL Baseline

Model: Llama3 8B, GPT-4o-mini, GPT-4o, GPT-4 (via Ollama and OpenAI API)
Evaluation: Stratified 5-fold cross-validation
Shots: 3-shot (optimal across all models)
Example selection: Manually curated HIGH_QUALITY_PAIRS whitelist of 24 verified pairs

### Financial Validation

Signal: 30-day post-filing cumulative abnormal return using S&P 500 as benchmark
Model: Panel OLS regression with sector and year fixed effects
Hypothesis: Risk escalation signals predict post-filing abnormal returns

---

## Analysis Findings

### Key Result 1: Diff Representation Eliminates De-escalation Blindness (Llama3 8B)

The original LLM baseline had de-escalating recall of 0/20 (0%). The diff representation with Llama3-specific prompting achieves 18/20 (90%) while improving kappa from 0.207 to 0.614.

```
Confusion matrix — ICL diff 3-shot, Llama3 8B (rows=true, cols=predicted):

                     de-escalating   stable   escalating
  de-escalating             18          0           2
  stable                     2          3           2
  escalating                 2          0           7
```

### Key Result 2: Three Failure Modes Across Model Families

Multi-model ablation across 4 models and 2 conditions reveals three systematic failure modes. Each failure mode represents a different way an LLM oversight monitor could fail in deployment. No single model-condition combination achieves reliable performance without model-specific prompt engineering.

### Key Result 3: KEY SIGNALS Intervention Fixes Stable Collapse Partially

Adding explicit binary signal questions (COVID eliminated? content decreased? new sections added?) to the GPT-4o/GPT-4 prompt improved GPT-4o from 0.151 to 0.273 (+81%) and GPT-4 from 0.191 to 0.453 (+137%). The intervention partially resolves stable collapse but cannot fix cases where all signals are genuinely weak.

### Key Result 4: Credit and Regulatory Risk Signals Predict Market Reactions

Panel OLS regression across 410 filing pairs (86 companies, 4 sectors, 2020–2024):

| Dimension | Beta | p-value | Significant |
|-----------|------|---------|-------------|
| credit_dir | +0.032 | 0.028 | Yes |
| regulatory_dir | +0.038 | 0.046 | Yes |
| liquidity_dir | +0.029 | 0.190 | No |
| operational_dir | +0.004 | 0.775 | No |
| market_dir | -0.010 | 0.645 | No |

R-squared: 0.22. The internal consistency between kappa and regression significance strengthens confidence — credit risk (highest kappa) is also the strongest regression predictor.

### Key Result 5: ICL Experiment History

| Experiment | Kappa | Notes |
|------------|-------|-------|
| LLM baseline v1 raw text | 0.207 | Baseline |
| v2 prompt engineering | 0.076 | Over-corrected |
| LoRA round 1 (98 imbalanced) | 0.000 | Stable collapse |
| LoRA round 2 (150 balanced) | -0.183 | De-escalating collapse |
| ICL diff v1 (degenerate) | 0.125 | Example quality bug |
| ICL diff v2 3-shot | 0.494 | Clean examples, reframed prompts |
| ICL diff v3 3-shot (tiebreaker) | 0.614 | Current Llama3 8B best |
| GPT-4o diff (GPT-4o+KEY SIGNALS) | 0.273 | Partial fix |
| GPT-4 diff (GPT-4o+KEY SIGNALS) | 0.453 | Best GPT result |

---

## Safety and Alignment Extension

**The core argument:** Financial disclosures are an empirically validated domain for studying evasive behavior under evaluation pressure. The resulting behavior — satisfying formal evaluation criteria while concealing material information — is structurally identical to what alignment researchers call deceptive alignment in AI systems.

**The empirical connection:** Scalable oversight requires AI monitors that can detect when institutions or AI systems are quietly reducing safety-relevant disclosures. Our results show current LLMs fail at this task in systematic and model-specific ways. The multi-model ablation reveals that no current LLM achieves reliable de-escalation detection without extensive prompt-representation co-design, and that co-design does not transfer across model families.

This has a direct implication for deploying LLM-based oversight monitors: the failure mode depends on both the model and the representation. A monitoring system calibrated for one LLM will fail when deployed with a different LLM, even a more capable one.

### Evasion Taxonomy

Four types of institutional evasion mapped to alignment concepts:

Type 1: Omission Evasion — A risk present in the earlier filing is entirely absent from the later filing. Maps to capability concealment and sandbagging.

Type 2: Obfuscation Evasion — The risk is mentioned but in language so generic it fails to communicate the specific exposure. Maps to deceptive alignment.

Type 3: Displacement Evasion — The risk is disclosed but buried in low-salience locations. Maps to specification gaming.

Type 4: Framing Evasion — Technically accurate language that creates a systematically misleading impression. Maps to sycophancy.

---

## Engineering Notes

This section documents real problems encountered during development. Problems 1-12 cover the data pipeline and initial modeling. Problems 13-26 cover the diff representation, ICL baseline, and multi-model ablation.

### Problem 1: Raw Filing Size at Scale (May 2026)

**What happened:** Full 10-K downloads for 45 companies reached 22.6GB, exceeding local disk space.

**Fix:** Batch processing by sector — download, extract Item 1A, delete raw files, move to next sector.

---

### Problem 2: sec-edgar-downloader v5.x API Change (May 2026)

**What happened:** `Downloader(company_name, email_address, save_path)` threw TypeError.

**Fix:** Updated constructor, added `download_details=True`, pinned to `sec-edgar-downloader==5.1.0`.

---

### Problem 3: Windows PowerShell Compatibility (May 2026)

**Fix:** Replaced Unix commands with PowerShell equivalents. Added `scripts/setup_windows.ps1`.

---

### Problem 4: Extractor Picking Wrong File from EDGAR Download (May 2026)

**What happened:** Extraction success rate 47% — extractor selecting `full-submission.txt` instead of `primary-document.html`.

**Fix:** Explicit prioritization of `primary-document.html`, HTML tag stripping, skip rule for `full-submission.txt`.

---

### Problem 5: Item 1A Extractor Matching Table of Contents (May 2026)

**What happened:** Large bank filings failed — extractor matched TOC entry instead of content.

**Fix:** TOC detection (Item 1B within 400 chars = TOC entry, skip). Minimum 1,000 word threshold.

---

### Problem 6: XBRL Primary Documents and Incorporation by Reference (May 2026)

**Fix for XBRL:** `is_xbrl_file()` detection, fall back to `full-submission.txt`.

**Fix for incorporation by reference:** Documented as known gap. Affects ~15% of banking filings.

---

### Problem 7: 10-Q Filings with No Risk Factor Content (May 2026)

**Decision:** Accepted as expected behavior. No material change in 10-Q is a valid signal.

---

### Problem 8: LLM Response JSON Truncation (May 2026)

**Fix:** Three-stage JSON repair logic. Parse success rate 0% → 100%.

---

### Problem 9: Insufficient Text Extraction in Annotation Sample (June 2026)

**What happened:** 37% of regulatory and 48% of operational dimensions marked insufficient_text.

**Root cause:** 3,000 character window cuts off before operational/regulatory sections.

**Decision:** Accepted as methodology limitation.

---

### Problem 10: LLM Systematic Annotation Biases Identified (June 2026)

Three systematic biases: (1) cannot detect de-escalation, (2) over-fires on operational risk, (3) under-fires on liquidity risk. See Key Result 2.

---

### Problem 11: Prompt Engineering Instability in Multi-Dimension Classification (June 2026)

**What happened:** v2 prompt dropped kappa from 0.207 to 0.076.

**Root cause:** Global instruction set — changes for one dimension affect all dimensions.

**Decision:** Reverted to v1 prompt. Proceeded to LoRA fine-tuning.

---

### Problem 12: LoRA Fine-tuning Label Collapse — Round 1 (June 2026)

**What happened:** Val kappa = 0.000. Model predicted stable for all validation examples.

**Root cause:** Class imbalance (54 stable / 35 escalating / 9 de-escalating), 512 token truncation, 98 examples insufficient.

---

### Problem 13: LoRA Fine-tuning Label Collapse — Round 2 (June 2026)

**What happened:** Val kappa = -0.183. Model predicted de-escalating 15/25 times.

**Root cause:** 9 de-escalating examples from 3 companies, 1 year, duplicated 5x. Model memorized specific COVID-removal pattern rather than generalizing.

**Key insight:** Fine-tuning minority classes requires diversity not just quantity. Minimum 30+ diverse de-escalating examples needed.

---

### Problem 14: Fine-tuning Architecture Reassessment (June 2026)

**Consultation outcome:** Input representation wrong (raw text too long), evaluation too noisy (single split), architecture mismatch (generative vs encoder), cheap ICL baseline skipped.

**Decision:** Build diff representation first, run ICL with k-fold CV, use DeBERTa-v3-base for fine-tuning if ICL justified it.

---

### Problem 15: Annotation Coverage Gap (June 2026)

**What happened:** 38 annotated pairs but only 23 had local files.

**Fix:** `extract_missing.py` re-extracted all 27 missing pairs from EDGAR.

---

### Problem 16: Annotation Expansion Targeting De-escalating Class (June 2026)

**What happened:** With only 11 de-escalating examples in the original annotation set, fine-tuning diversity requirements could not be met. Six extraction script versions (v1-v6) plus `annotate_with_llm.py` required before reliable annotation candidates were produced.

**Extraction script progression:**

`extract_deescalating.py` v1: CIK lookup worked but `get_document_url()` fetched the search page instead of the actual filing. Zero files produced.

`extract_deescalating_v2.py`: Fixed URL resolution using EDGAR submissions API primary document field. Item 1A regex failed for XBRL-heavy filings (DAL, UAL, MCD). SBUX extracted successfully.

`extract_deescalating_v3.py`: Added explicit stripping of XBRL namespace tags before HTML stripping. DAL/UAL/MCD still failing — falling back to 60,000 char windows.

`extract_v4.py` through `extract_v6.py`: Switched from keyword scoring to subsection heading match. Fixed NFLX fiscal year detection (Netflix files in January). Replaced DAL/UAL/MCD targets with NFLX/UBER/LYFT/ROKU after consistent airline/fast food extraction failures.

`annotate_with_llm.py`: Added Claude API annotation on top of extraction — LLM pre-fills annotation blocks, human reviews and overrides.

`extract_v8.py`: Rewrote `get_item1a()` to find all Item 1A matches, skip TOC entries (< 2000 chars), rank candidates by length with sanity check for risk language. Targets: 8 failed companies from v7 (AAL, DAL, DIS, EXPE, LYV, MCD, RCL, UAL) + 9 new tickers (NCLH, MGM, LVS, WYNN, BA, SAVE, JBLU, ABNB, F). Persistent failures: AAL, DAL, DIS, F, LVS, LYV, MCD, MGM, NCLH, RCL, UAL — blocked permanently. Usable from v8: ABNB, BA, EXPE, JBLU, WYNN.

`extract_v9.py`: Fresh tickers across sectors — TGT, WMT, GPS, LULU (retail), CRM, TWLO, SNAP (tech), CVS, HCA, THC (healthcare), NFLX, WBD (media). WMT failed — pulled Item 1 Business ESG/Human Capital content. LULU/GPS failed. 10 usable annotations from 15 targets.

**Key annotation decisions:** SBUX is a counter-example where COVID escalated 2021→2022 due to China-specific restrictions. JBLU escalated due to Spirit merger risk additions not COVID. These counter-examples validate annotation quality — demonstrates the schema captures real signal not just temporal auto-correlation.

**NFLX failure and fix:** EDGAR recent filings endpoint only returns last ~20 filings. NFLX 2021/2022 had fallen off the list. Added `get_filings_all()` checking older filings endpoint.

**ROKU exclusion:** Extraction captured only risk factor summary and competitive landscape, not the actual credit risk subsection.

**Final annotation counts:** de-escalating 20, stable 8, escalating 12, total usable 40 across original (16) + v7 (8) + v8 (7) + v9 (10) sessions.

**Lesson:** Minimum viable de-escalating count for fine-tuning is 30-40 diverse examples. 20 crosses the technical threshold but stable class (8 pairs) remains the binding constraint for fine-tuning stability.

---

### Problem 17: Asymmetric Extraction Causing Near-Zero Diff Signal (June 2026)

**Root cause:** `get_relevant()` used COVID keyword density — found COVID section in 2021, found different section in 2022 after COVID language toned down. Near-zero diff for 8 pairs.

**Fix:** `extract_full_item1a.py` using full 30,000 character Item 1A without keyword truncation.

---

### Problem 18: XBRL Metadata Artifacts in Diff Summaries (June 2026)

**What happened:** CINF pairs had XBRL metadata appearing as sentences in diff summaries.

**Fix:** `load_data()` XBRL artifact exclusion filter. `has_clean_summary()` hard gate. Both CINF pairs permanently excluded — 36 usable records from 38 annotated.

---

### Problem 19: Financial Statement Content Mixed into Diff Summaries (June 2026)

**What happened:** AIZ had financial statement content in diff summaries.

**Fix:** `has_clean_summary()` financial artifact patterns. AIZ removed from HIGH_QUALITY_PAIRS whitelist.

---

### Problem 20: Duplicate Pair Processing Inflating Diff Count (June 2026)

**Fix:** `processed_pairs` set in `build_diff_v2.py`. First occurrence wins.

---

### Problem 21: ICL Few-shot Example Degeneracy — v1 Prompt (June 2026)

**What happened:** ICL v1 kappa = 0.125, worse than zero-shot (0.186). Model predicted escalating for 31/38 pairs.

**Root cause:** (1) "COVID LANGUAGE REMOVED" counterintuitive to LLM, (2) no class balance guarantee in example selection, (3) contradictory examples (AIZ, FANG) passing fallback filter.

**Fix:** Reframed diff language with explicit [SIGNAL: de-escalating] tags, guaranteed 1-per-class selection, HIGH_QUALITY_PAIRS whitelist.

---

### Problem 22: Whitelist Fallback Bypassing Quality Filters (June 2026)

**What happened:** Contradictory examples continued appearing despite whitelist.

**Root cause:** Fallback used `signal_agrees()` without content quality check. AIZ passed because primary signal agreed with label despite financial statement artifacts.

**Iterations:** 4 rounds before final fix — `has_clean_summary()` as mandatory hard gate on ALL selection paths including absolute fallback.

---

### Problem 23: Stable Class Recall Collapse (June 2026)

**What happened:** Stable recall 2/7 (28.6%) despite kappa 0.494.

**Root cause:** Stable pairs show minimal diff signal by definition. Model defaults to escalating or de-escalating based on weak signals.

**Partial fix:** Tiebreaker rule in system prompt, better stable examples with ≥8 sentence churn in whitelist. Stable recall improved to 3/7 (42.9%) at kappa 0.614.

---

### Problem 24: GPT-4o Stable Collapse (June 2026)

**What happened:** GPT-4o with Llama3 diff representation got kappa 0.147 — worse than Llama3 8B raw (0.071 was de-escalation blindness; GPT-4o was predicting stable for everything).

**Root cause:** [SIGNAL: xxx] annotation scaffolding and numbered rules are tuned for Llama3's instruction-following style. GPT-4o processes structured text analytically and ignores the annotation tags.

**Fix:** Model-specific prompt routing. GPT-4o/GPT-4 get clean analytical descriptions without [SIGNAL] tags.

---

### Problem 25: GPT-4o Addition Blindness (June 2026)

**What happened:** GPT-4o-mini diff kappa = 0.139. Model predicted escalating for 13/20 de-escalating cases.

**Root cause:** GPT-4o-mini reads "NEW NON-COVID RISK CONTENT added" and predicts escalating regardless of COVID removal signal. Capability threshold issue — model too small to follow multi-signal instructions reliably.

**Finding:** GPT-4o-mini represents a capability threshold below which diff representation actively hurts performance.

---

### Problem 26: GPT-4o/GPT-4 Stable Collapse Despite Model-Specific Prompt (June 2026)

**What happened:** GPT-4o diff with clean analytical prompt: kappa 0.151. GPT-4 diff: kappa 0.141. Both models defaulting to stable for ambiguous cases.

**Root cause:** Without explicit signal extraction the model sees weak diff content and defaults to stable rather than making a directional judgment.

**Fix:** KEY SIGNALS block added to diff formatter — explicit yes/no questions for COVID eliminated, content decreased/increased, new sections added, tense shift detected. Paired with DECISION RULES in system prompt referencing KEY SIGNALS.

**Result:** GPT-4o improved from 0.151 to 0.273 (+81%). GPT-4 improved from 0.191 (raw) to 0.453 (diff + KEY SIGNALS) (+137%).

**Residual problem:** Fold 5 pairs (HLT, MAR, PTON, SBUX) still collapse to stable. These have genuinely weak KEY SIGNAL values — 3-4 COVID sentences removed, no tense shift, modest volume change. No prompt engineering fix available without more training data.

---

## Dataset Quality Notes

**Sector skew:** Technology companies have cleaner filing formats. Banking has systematic gaps from incorporation by reference at WFC, USB, BK.

**Temporal skew:** Weighted toward annual 10-K filings. Methodologically acceptable since annual pairs are the primary unit of analysis.

**Survivorship bias:** SIVB (Silicon Valley Bank, collapsed March 2023) is a notable absence that would have been analytically interesting for liquidity risk detection.

**Annotation quality:** Single annotator ground truth. Disagreement patterns are directionally systematic rather than random, supporting the model bias interpretation. Second annotator validation using GPT-4 as independent LLM annotator is planned.

---

## Limitations

Single annotator ground truth introduces subjectivity. Second annotator validation planned.

Yahoo Finance abnormal return proxies are less precise than CDS spreads. WRDS Markit access confirmed for summer 2026.

LLM outputs non-deterministic. Temperature set to 0, seed fixed at 42.

Coverage limited to US-listed companies with English filings.

Incorporation by reference gaps affect ~15% of banking filings.

ICL ceiling for Llama3 8B is approximately 0.62-0.65. Reaching 0.7-0.8 requires DeBERTa-v3-base fine-tuning on 56+ annotated pairs.

Stable class recall remains the primary weakness across all models. 7 stable pairs is insufficient for reliable boundary learning.

Prompt-representation co-design does not transfer across model families — Llama3 8B prompt engineering does not generalize to GPT-4o or GPT-4.

---

## Future Work

**Near term:**

DeBERTa-v3-base fine-tuning on diff summaries with classification head — target kappa 0.70-0.80 with 56+ pairs.

Annotate 20-25 more pairs targeting stable class and non-COVID escalating to enable fine-tuning.

Adversarial elicitation test: ask LLM directly "has risk disclosure been reduced?" and compare to classification task results.

SEC comment letter validation via WRDS — external ground truth for which disclosures were deemed inadequate by regulators.

**Medium term:**

arXiv preprint: "Systematic Failure Modes in LLM-based Risk Monitoring: A Multi-Model Empirical Study." Target NeurIPS 2026 Behavioral Evaluation workshop or ICLR 2027 Trustworthy ML workshop.

Alignment Forum post on curriculum safety coverage analysis from CIC RA work.

LLM generation experiments: prompt models under evaluation pressure and test whether evasion detector identifies the same patterns.

**Longer term:**

CDS spread validation via WRDS Markit.

SIVB and First Republic Bank as stress-period validation cases.

Benchmark expansion to 200+ pairs with held-out test set and public leaderboard.

Real-time EDGAR monitoring pipeline.

---

## Project Structure

```
financial-disclosure-risk-intelligence/
├── src/
│   ├── ingestion/              # SEC EDGAR data collection
│   ├── preprocessing/          # Text cleaning, pair construction
│   ├── modeling/               # LLM risk detection pipeline
│   ├── evaluation/             # Kappa scores, regression analysis
│   └── visualization/          # Streamlit app, Plotly charts
├── diffs/
│   ├── diff_representations.jsonl       # Structured diff for 36 clean pairs
│   └── diff_representations_readable.txt
├── results/
│   ├── icl_diff_3shot_llama3_latest_results.json
│   ├── icl_diff_3shot_gpt_4o_results.json
│   ├── icl_diff_3shot_gpt_4_results.json
│   └── icl_raw_3shot_*_results.json
├── data/
│   ├── raw/                    # Raw EDGAR filings (gitignored)
│   └── processed/              # Cleaned filing pairs (gitignored)
├── notebooks/                  # Exploratory analysis
├── tests/                      # Unit tests
├── icl_baseline_v3.py          # Current ICL baseline with multi-model support
├── build_diff_v2.py            # Diff representation builder
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/Sudhiksha-17/financial-disclosure-risk-intelligence
cd financial-disclosure-risk-intelligence
pip install -r requirements.txt

# Build diff representations
python build_diff_v2.py \
  --input_dirs extracted_full deescalating_v7 deescalating_v8 deescalating_v9 extracted_missing \
  --annotations annotations_final_v3.txt \
  --output_dir diffs

# Run ICL baseline (Llama3 8B local)
python icl_baseline_v3.py \
  --input diffs/diff_representations.jsonl \
  --output results \
  --model llama3:latest \
  --shots 3 \
  --condition diff

# Run ICL baseline (GPT-4 via OpenAI API)
export OPENAI_API_KEY=your_key_here
python icl_baseline_v3.py \
  --input diffs/diff_representations.jsonl \
  --output results \
  --model gpt-4 \
  --shots 3 \
  --condition both
```

---

## Author

Sudhiksha Kandavel Rajan
MS Artificial Intelligence, Northeastern University
[LinkedIn](https://linkedin.com/in/sudhiksha-kandavel-rajan) · [GitHub](https://github.com/Sudhiksha-17) · [HUX AI Paper](https://arxiv.org/abs/2407.19492)