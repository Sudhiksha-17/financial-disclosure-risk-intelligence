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
| Human annotated pairs | 38 (operational risk focus) |
| Cohen's kappa — LLM baseline (raw text, v1 prompt) | 0.207 |
| Cohen's kappa — ICL diff representation (3-shot) | 0.494 |
| Cohen's kappa — ICL diff representation (6-shot) | 0.470 |
| De-escalating recall — LLM baseline | 0/11 (0%) |
| De-escalating recall — ICL diff representation | 20/20 (100%) |
| Diff primary signal alignment with human labels | 84.2% (32/38) |
| Panel OLS β — credit risk escalation | +0.032 (p=0.028) |
| Panel OLS β — regulatory risk escalation | +0.038 (p=0.046) |

The de-escalating recall improvement from 0% to 100% is the headline finding. The diff representation eliminates the de-escalation blind spot that exists in raw text prompting.

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
│   ICL Baseline       │  Llama-3 8B, diff representation
│   5-fold CV          │  Reframed diff summaries, whitelist examples
│   3-shot / 6-shot    │  Cohen's kappa evaluation
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

This collapses thousands of tokens of raw text to a few hundred tokens of structured change signal, directly surfacing the classification-relevant information.

The reframed diff summary makes the signal explicit for the LLM. Instead of "COVID LANGUAGE REMOVED (13 sentences)" it reads "COVID/PANDEMIC RISK LANGUAGE REDUCED: 13 sentences discussing active COVID threats were ELIMINATED from the later filing [SIGNAL: de-escalating]."

### ICL Baseline

Model: Llama-3 8B via Ollama
Evaluation: Stratified 5-fold cross-validation
Shots: 3-shot and 6-shot ablations
Example selection: Manually curated whitelist of 24 high-quality pairs with unambiguous signals
Quality filters: Hard exclusion of XBRL metadata artifacts and financial statement content from few-shot examples

### Evaluation

Ground truth: 38 manually annotated filing pairs (operational risk dimension)
Metric: Cohen's kappa
Failure analysis: Documented systematic biases and ICL prompt iteration history

### Financial Validation

Signal: 30-day post-filing cumulative abnormal return using S&P 500 as benchmark
Model: Panel OLS regression with sector and year fixed effects
Hypothesis: Risk escalation signals predict post-filing abnormal returns

---

## Analysis Findings

### Key Result 1: Diff Representation Eliminates De-escalation Blindness

The original LLM baseline (Llama-3 8B, v1 prompt, raw text) had de-escalating recall of 0/11 (0%). The model predicted either escalating or stable for every test case, systematically missing all de-escalating cases.

The diff representation with reframed prompting achieves de-escalating recall of 20/20 (100%) while improving overall kappa from 0.207 to 0.494. This is the core finding.

```
Confusion matrix — ICL diff 3-shot (rows=true, cols=predicted):

                     de-escalating   stable   escalating
  de-escalating             20          0           0
  stable                     2          2           3
  escalating                 5          0           4
```

Remaining errors break into two categories. The five escalating cases predicted as de-escalating are pairs where COVID language was reduced but substantial non-COVID risk was added simultaneously — a tiebreaker problem, not a blindspot. The stable class has poor recall (2/7) because by definition nothing changed, leaving minimal diff signal.

### Key Result 2: LLM Baseline Systematic Biases (Raw Text)

Cohen's kappa between Llama-3 8B v1 prompt predictions and human annotations across 123 comparable dimension-level annotations from 50 sampled filing pairs:

| Dimension | Kappa | n | Interpretation |
|-----------|-------|---|----------------|
| credit | 0.220 | 29 | Fair |
| operational | 0.208 | 26 | Slight |
| market | 0.161 | 31 | Slight |
| liquidity | 0.143 | 24 | Slight |
| regulatory | 0.085 | 13 | Slight |
| overall | 0.207 | 123 | Fair |

Three systematic biases documented:

Bias 1: Cannot detect de-escalation. Human annotators identified 11 de-escalating cases. The LLM identified zero.

Bias 2: Over-fires on operational risk. Human: 9 escalating, 15 stable. LLM: 19 escalating, 7 stable. The model codes operational risk as escalating roughly twice as often as human judgment warrants.

Bias 3: Under-fires on liquidity risk. Human: 7 escalating, 2 de-escalating. LLM: 1 escalating, 0 de-escalating. Liquidity risk language is often embedded within broader market or credit sections rather than appearing as standalone bullets.

### Key Result 3: Credit and Regulatory Risk Signals Predict Market Reactions

Panel OLS regression across 410 filing pairs (86 companies, 4 sectors, 2020–2024) with sector and year fixed effects. Dependent variable is 30-day cumulative abnormal return (CAR) relative to SPY.

| Dimension | Beta | p-value | Significant |
|-----------|------|---------|-------------|
| credit_dir | +0.032 | 0.028 | Yes |
| regulatory_dir | +0.038 | 0.046 | Yes |
| liquidity_dir | +0.029 | 0.190 | No |
| operational_dir | +0.004 | 0.775 | No |
| market_dir | -0.010 | 0.645 | No |

R-squared: 0.22. CAR mean: -4.09% across all 410 pairs.

The internal consistency between regression results and kappa analysis strengthens confidence in the finding. Credit risk, the dimension with the highest kappa (0.220), is also the dimension with the strongest regression coefficient. Operational risk, the dimension with the most severe LLM over-firing bias, has the weakest coefficient (p=0.775). This pattern is exactly what we would expect if signal quality predicts regression significance.

### Key Result 4: Fine-tuning Calibration Experiments

| Experiment | Training data | Val kappa | Notes |
|------------|---------------|-----------|-------|
| Base LLM v1 prompt | None | 0.207 | Baseline |
| v2 prompt engineering | None | 0.076 | Over-corrected |
| LoRA round 1 | 98 imbalanced | 0.000 | Stable collapse |
| LoRA round 2 | 150 balanced | -0.183 | De-escalating collapse |
| ICL diff 3-shot (v2 prompts) | None | 0.494 | Current best |
| ICL diff 6-shot (v2 prompts) | None | 0.470 | Slightly lower |

---

## Safety and Alignment Extension

This project is being extended with a safety and alignment research layer connecting financial disclosure analysis to AI evaluation methodology.

**The core argument:** Financial disclosures are an empirically validated domain for studying evasive behavior under evaluation pressure. Companies face strong incentives to satisfy SEC disclosure requirements while minimizing reputational and market impact. The resulting behavior — satisfying formal evaluation criteria while concealing material information — is structurally identical to what alignment researchers call deceptive alignment in AI systems.

**The empirical connection:** Scalable oversight requires AI monitors that can detect when institutions or AI systems are quietly reducing safety-relevant disclosures. Our results show current LLMs fail at this monitoring task in a systematic and asymmetric way: de-escalating cases (risk quietly removed) have 85% false negative rate at baseline, while escalating cases have roughly 40% false negative rate. This asymmetry is precisely the failure mode that a strategically evasive system would exploit.

The diff representation reduces the de-escalating false negative rate from 85% to 0%. This suggests that architectural choices in oversight tools — specifically whether monitors operate on raw text or structured change representations — have significant impact on their ability to detect quiet risk removal.

### Evasion Taxonomy

Four types of institutional evasion mapped to alignment concepts:

Type 1: Omission Evasion — A risk present in the earlier filing is entirely absent from the later filing without explanation. Maps to capability concealment and sandbagging.

Type 2: Obfuscation Evasion — The risk is mentioned but in language so generic it fails to communicate the specific exposure. Maps to deceptive alignment where outputs pass formal evaluation without carrying intended content.

Type 3: Displacement Evasion — The risk is disclosed but buried in low-salience locations, satisfying the letter of the requirement while minimizing attention. Maps to specification gaming.

Type 4: Framing Evasion — Technically accurate language that creates a systematically misleading impression. Maps to sycophancy in language models.

---

## Engineering Notes

This section documents real problems encountered during development and the decisions made to address them. It exists because honest documentation of engineering tradeoffs is more valuable for replication than polished post-hoc narratives.

---

### Problem 1: Raw Filing Size at Scale (May 2026)

**What happened:** Initial pipeline downloaded full 10-K and 10-Q filings for 45 companies across 2019 to 2024. Total raw download size reached 22.6GB, exceeding available local disk space before extraction could complete.

**Root cause:** Full EDGAR filings include financial statements, exhibits, legal documents, and appendices. We only need Item 1A which is typically 8KB to 50KB per filing.

**Decision:** Batch processing by sector — download, extract Item 1A, delete raw files, move to next sector. Keeps peak disk usage under 8GB.

**Lesson:** Production version should use the SEC EDGAR Full Text Search API to fetch only Item 1A directly.

---

### Problem 2: sec-edgar-downloader v5.x API Change (May 2026)

**What happened:** Initial code used `Downloader(company_name, email_address, save_path)` which threw TypeError on first run.

**Root cause:** v5.x removed the `save_path` parameter. Files now save to `sec-edgar-filings/` by default.

**Fix:** Updated constructor and added `download_details=True` to `dl.get()`. Pinned to `sec-edgar-downloader==5.1.0` in requirements.txt.

---

### Problem 3: Windows PowerShell Compatibility (May 2026)

**What happened:** Setup commands using Unix `touch` and `mkdir -p` failed on Windows PowerShell.

**Fix:** Replaced with PowerShell equivalents. Added `scripts/setup_windows.ps1`.

---

### Problem 4: Extractor Picking Wrong File from EDGAR Download (May 2026)

**What happened:** Item 1A extraction success rate was 47% on first run. Extractor was selecting `full-submission.txt` (85MB EDGAR wrapper) instead of `primary-document.html`.

**Root cause:** File selection logic took the largest file which was always `full-submission.txt`.

**Fix:** Updated to explicitly prioritize `primary-document.html`. Added HTML tag stripping. Added skip rule for `full-submission.txt`.

---

### Problem 5: Item 1A Extractor Matching Table of Contents (May 2026)

**What happened:** Large bank 10-K filings (JPM, BAC, GS, MS, WFC) still failed. Item 1A text existed but was not being extracted.

**Root cause:** Table of contents entries like "Item 1A. Risk Factors. 7-28" followed immediately by "Item 1B." triggered our extractor to match the TOC entry first.

**Fix:** Added TOC detection — if Item 1B appears within 400 characters of an Item 1A match, classify as TOC entry and skip. Increased minimum extraction threshold to 1,000 words.

---

### Problem 6: XBRL Primary Documents and Incorporation by Reference (May 2026)

**What happened:** MS and C failed with zero Item 1A matches. WFC and USB failed because risk factors are incorporated by reference from separate documents.

**Fix for XBRL (MS, C):** Added `is_xbrl_file()` detection. When primary-document.html is XBRL, fall back to `full-submission.txt` and extract the first DOCUMENT section.

**Fix for incorporation by reference (WFC, USB):** No fix applied. Documented as known gaps. Affects approximately 15% of banking sector filings.

---

### Problem 7: 10-Q Filings with No Risk Factor Content (May 2026)

**What happened:** Many companies succeed on 10-K extraction but fail on all 10-Q extractions.

**Root cause:** SEC rules only require disclosure of material changes in 10-Q. Companies with no material changes write brief statements under our 1,000-word minimum.

**Decision:** Accepted as expected behavior. No material change in 10-Q is itself a valid signal.

---

### Problem 8: LLM Response JSON Truncation (May 2026)

**What happened:** LLM risk detector consistently failed to parse responses despite the model generating correct output. JSON cutting off mid-value in the final risk dimension.

**Root cause:** Llama-3 8B via Ollama stops generating before adding the final closing braces. This is a known behavior of instruction-tuned models generating structured output without explicit stop tokens.

**Fix:** Three-stage JSON repair logic in `parse_response()`. Stage 1: parse as-is. Stage 2: count open vs closed braces, append missing closings. Stage 3: walk backwards finding last valid closing brace position.

**Result:** Parse success rate went from 0% to 100%.

---

### Problem 9: Insufficient Text Extraction in Annotation Sample (June 2026)

**What happened:** During human annotation of 50 filing pairs, 37% of regulatory dimensions and 48% of operational dimensions were marked insufficient_text.

**Root cause:** The 3,000 character annotation window captures the beginning of Item 1A but SEC filings place credit and market risks first. Operational and regulatory risks appear later and are cut off.

**Decision:** Accepted as a methodology limitation. Regulatory kappa was computed on only 13 pairs versus 31 for market risk.

---

### Problem 10: LLM Systematic Annotation Biases Identified (June 2026)

**What happened:** Cohen's kappa analysis revealed three systematic biases in LLM risk classification. Documented in Key Result 2 above.

**Next step identified:** LoRA fine-tuning on annotated pairs to improve calibration.

---

### Problem 11: Prompt Engineering Instability in Multi-Dimension Classification (June 2026)

**What happened:** Attempted to improve kappa from 0.207 to 0.4+ by rewriting the prompt with few-shot examples and explicit de-escalation instructions. The v2 prompt caused kappa to drop from 0.207 to 0.076.

**Root cause:** Instruction-tuned LLMs use the prompt as a global instruction set. Adding five de-escalation examples caused the model to over-predict de-escalating across all dimensions. The anti-length-bias instruction simultaneously suppressed legitimate escalation detection in operational risk. Changes intended for one dimension affect all dimensions simultaneously.

**Decision:** Reverted to v1 prompt. Proceeded to LoRA fine-tuning.

---

### Problem 12: LoRA Fine-tuning Label Collapse — Round 1 (June 2026)

**What happened:** LoRA fine-tuning on Llama-3 8B via Google Colab T4 GPU produced a degenerate model predicting "stable" for all 25 validation examples. Val kappa = 0.000.

**Setup:** 98 training examples, MAX_SEQ_LEN 512 (reduced from 4096 to 1024 to 512 due to successive OOM errors), LoRA rank 8 (reduced from 16), `processing_class` fix in SFTTrainer (replacing deprecated `tokenizer` argument), `eval_strategy` adjusted to avoid OOM.

**Root cause:** Three compounding factors. Class imbalance of 54 stable / 35 escalating / 9 de-escalating was not corrected. 512 token truncation removed most filing content. 98 examples is insufficient for a 3-class classification task with high input variability.

**Windows dependency conflicts encountered:** Unsloth requires torch 2.4.0+ but CUDA setup requires torch 2.3.1+cu121, making them incompatible. pyarrow version conflict between unsloth (requiring 24.0.0) and pandas (requiring 14.0.1) also present. Decision: move all fine-tuning to Google Colab T4.

**HuggingFace gated access:** Approved for both Meta-Llama-3-8B-Instruct and Meta-Llama-3-70B-Instruct.

---

### Problem 13: LoRA Fine-tuning Label Collapse — Round 2 (June 2026)

**What happened:** Second fine-tuning attempt with 150 balanced examples via oversampling produced a model predicting de-escalating 15 out of 25 times. Val kappa = -0.183.

**Root cause:** The 9 de-escalating training examples were all from the 2022-2023 period and primarily captured COVID language removal in market risk (6 out of 9 were market_risk, 3 out of 9 were from CFG 2022-2023 alone, 7 out of 9 were from the 2022-2023 transition period). Duplicating these 5x caused the model to memorize a specific COVID-removal pattern rather than learning the general concept of de-escalation. Every new input superficially matched the oversampled pattern.

**Key insight:** Fine-tuning on minority classes requires both sufficient quantity and sufficient diversity. 9 examples from 3 companies spanning 1 year cannot generalize even with oversampling. Minimum viable dataset estimated at 30+ diverse de-escalating examples spanning multiple companies, sectors, and transition types.

---

### Problem 14: Fine-tuning Architecture Reassessment (June 2026)

**What happened:** After two failed LoRA rounds, a structured consultation was conducted. The consultation reordered the priority stack completely.

**Original hypothesis:** Label collapse was primarily caused by 512-token truncation. Proposed fix was to upgrade to A100 GPU for 2048-token sequences.

**Consultation outcome — four issues identified:**

Issue 1: Input representation is wrong. Concatenating two full 10-K sections creates a comparison task that exceeds any practical sequence length. The fix is to compute a year-over-year diff and feed that instead, collapsing thousands of tokens to a few hundred.

Issue 2: Evaluation is too noisy to trust. Val kappa on a single split with 63 examples has very high variance. Conclusions drawn from single-split results are not reliable without stratified k-fold CV.

Issue 3: Architecture mismatch. Using a generative 8B decoder to emit a 3-class label is the wrong tool. An encoder-based classifier (DeBERTa-v3-base) with a classification head and class-weighted loss is better suited and more sample-efficient.

Issue 4: Cheap baseline was skipped. Few-shot ICL with annotated pairs as demonstrations, using diff inputs, likely outperforms fine-tuning at this data size and costs nothing in training.

**Decision:** Build diff representation first, run few-shot ICL baseline with k-fold CV, use encoder-based fine-tuning only if ICL results justify it.

---

### Problem 15: Annotation Coverage Gap — Original Session Pairs Have No Local Files (June 2026)

**What happened:** 38 unique annotated pairs but only 23 had local extracted text files. Original session pairs (VLO, CINF, FITB, FANG, etc.) were annotated from pasted content in chat and never saved locally.

**Root cause:** Early annotation sessions were conducted by pasting content directly into conversation. No script was run to save the files locally. Later extraction sessions (v7, v8, v9) saved files to named folders but original session pairs were missing entirely.

**Fix:** `extract_missing.py` re-extracted all 27 missing pairs using the EDGAR submissions API. Pairs saved to `extracted_missing/` folder.

---

### Problem 16: Annotation Expansion — Targeting De-escalating Class (June 2026)

**What happened:** With only 11 de-escalating examples in the original annotation set, fine-tuning diversity requirements could not be met. Multiple extraction script iterations were required before reliable annotation candidates were produced.

**Extraction script progression:**

`extract_deescalating.py` (v1): CIK lookup worked but `get_document_url()` fetched the search page instead of the actual filing. Zero files produced.

`extract_deescalating_v2.py`: Fixed URL resolution using EDGAR submissions API primary document field. URLs now correct but Item 1A regex failed for XBRL-heavy filings (DAL, UAL, MCD). SBUX extracted successfully.

`extract_deescalating_v3.py`: Added explicit stripping of XBRL `<ix:nonnumeric>` and `<ix:continuation>` namespace tags before HTML stripping. Item 1A regex now worked for most filings but DAL/UAL/MCD still falling back to 60,000 char windows (too large).

`extract_v4.py` through `extract_v6.py`: Switched from keyword scoring to subsection heading match for relevant content extraction. Fixed NFLX filing detection (Netflix files in January — fiscal year detection edge case). Replaced DAL/UAL/MCD/SBUX targets with NFLX/UBER/LYFT/ROKU after consistent extraction failures on airline and fast food filings.

`annotate_with_llm.py`: Added Claude API annotation on top of extraction — LLM pre-fills annotation blocks, human reviews and overrides. Used for UBER, LYFT, ZM, ROKU pairs.

**NFLX failure:** EDGAR recent filings endpoint only returns last ~20 filings. NFLX 2021 and 2022 filings had fallen off the recent list. Added `get_filings_all()` function checking the older filings endpoint. NFLX replaced with ZM (Zoom) for the annotation batch as ZM had clearer COVID normalization signal.

**ROKU insufficient_text:** Extraction captured only risk factor summary and competitive landscape intro, not the actual credit risk subsection. ROKU excluded from final annotation set.

**Final new de-escalating pairs annotated:** VLO 2021-2022 (credit_risk), AIZ 2019-2020 (operational_risk), LYFT 2021-2022 (operational_risk), UBER 2021-2022 (operational_risk), ZM 2021-2022 (operational_risk). De-escalating count increased from 11 to 16.

**Lesson:** Minimum viable de-escalating count for fine-tuning is 30-40 diverse examples. 16 is still insufficient. Additional annotation sessions (v7, v8, v9) were conducted separately targeting de-escalating cases post-COVID normalization.

---

### Problem 17: Asymmetric Extraction Causing Near-Zero Diff Signal (June 2026)

**What happened:** Eight pairs (EXPE, HLT, LYFT, VLO and others) showed only 1-7 added/removed sentences instead of the expected 27-49. The diff builder was producing near-empty representations for pairs that had clear de-escalating signals in manual annotation.

**Root cause:** `get_relevant()` scored paragraphs by COVID keyword density. For the 2021 filing it found the COVID risk section (high keyword density). For the 2022 filing with toned-down COVID language, it found a completely different section with higher keyword density. The two years were extracting from different parts of the document, producing a near-zero diff that did not reflect the actual year-over-year change.

**Fix:** `extract_full_item1a.py` re-extracted all 8 affected pairs using the full 30,000 character Item 1A without keyword truncation. Saved to `extracted_full/` folder.

**Result:** EXPE went from 3 added/removed sentences to 31. HLT went from 5 to 38. All 8 pairs now show meaningful diff signal consistent with their human annotations.

---

### Problem 18: XBRL Metadata Artifacts in Diff Summaries (June 2026)

**What happened:** Two CINF pairs (CINF_10-K_2020_2021 and CINF_10-K_2022_2023) had XBRL filing metadata appearing as sentences in their diff summaries: "cinf-20211231 false 2021 FY 0000020286 --12-31 P3Y P3Y P3Y..."

**Root cause:** CINF files its 10-K using XBRL inline format (iXBRL). The document contains metadata blocks that survive both HTML stripping and the XBRL tag removal pass because they appear as plain text within the document body rather than inside tags. These strings pass sentence length filters and appear as legitimate sentences in the diff.

**Fix:** Two-layer filter. First, `load_data()` excludes any record where the diff summary contains known XBRL artifact strings ("false 2021 fy", "p3y p3y", "--12-31", "0000020286"). Second, `has_clean_summary()` function blocks affected pairs from few-shot example selection at all fallback levels. Both CINF pairs permanently excluded from training and evaluation — 36 usable records from 38 annotated pairs.

---

### Problem 19: Financial Statement Content Mixed into Diff Summaries (June 2026)

**What happened:** AIZ and several other pairs had financial statement content appearing in their diff summaries: "$79.3 million tax benefit related to...", "MCPS will convert into shares of common stock on March 15, 2021."

**Root cause:** Financial statements appear later in Item 1A in some filings (particularly insurance sector filings that embed financial tables within the risk section). These sentences pass our boilerplate filter because they contain no table reference markers and exceed the minimum length threshold.

**Fix:** `has_clean_summary()` hard filter added with financial artifact patterns: "net income", "tax benefit", "shares of common stock", "mcps will convert", "the change in program structure", and similar patterns. AIZ removed from the HIGH_QUALITY_PAIRS whitelist after its financial artifact content was confirmed.

---

### Problem 20: Duplicate Pair Processing Inflating Diff Count (June 2026)

**What happened:** `build_diff.py` produced 50 diff representations instead of 38 unique pairs. Running with multiple `--input_dirs` (deescalating_v8 and extracted_missing both contained ABNB, BKNG, and other pairs) caused the same pair to be processed twice.

**Root cause:** No deduplication logic in the diff builder. First-come-first-served processing meant whichever folder was listed first determined the representation used, but both runs were saved to the output.

**Fix:** Added `processed_pairs` set to `build_diff_v2.py`. First occurrence of a pair_id wins, subsequent occurrences from other input directories are skipped with a logged warning.

---

### Problem 21: ICL Few-shot Example Degeneracy — v1 Prompt (June 2026)

**What happened:** First ICL baseline run (v1 prompts, raw diff summaries) produced kappa = 0.125, worse than zero-shot (0.186). Model predicted "escalating" for 31 out of 38 pairs — essentially a degenerate classifier.

**Root cause — three compounding factors:**

Factor 1: Diff summary language was counterintuitive. "COVID LANGUAGE REMOVED (13 sentences)" contains the word "removed" which an LLM interprets as something being taken away — implying a gap or missing disclosure — rather than recognizing removal as a positive de-escalation signal.

Factor 2: `select_examples()` did not guarantee class balance. In Fold 4 and Fold 5, training data was dominated by escalating examples. The model latched onto the majority pattern.

Factor 3: The secondary fallback in example selection was not gated on content quality. AIZ (financial statement artifacts in diff) and FANG_2022_2023 (COVID signal contradicts escalating label) both passed the `signal_agrees()` filter and appeared as few-shot examples, teaching the model wrong patterns.

**Fix:** Rewrote diff summary language to make semantics explicit. "COVID LANGUAGE REMOVED" became "COVID/PANDEMIC RISK LANGUAGE REDUCED: 13 sentences discussing active COVID threats were ELIMINATED from the later filing [SIGNAL: de-escalating]". Added guaranteed 1-per-class selection in `select_balanced()`. Added HIGH_QUALITY_PAIRS whitelist as primary example source.

---

### Problem 22: Whitelist Fallback Bypassing Quality Filters (June 2026)

**What happened:** Even after implementing the HIGH_QUALITY_PAIRS whitelist, contradictory examples continued appearing in dry runs. Example 3 kept showing "COVID LANGUAGE REDUCED" with an escalating label, or financial statement artifacts.

**Root cause:** The fallback logic in `select_balanced()` used `signal_agrees()` as the secondary filter when the whitelist was exhausted in a fold. `signal_agrees()` checked only whether the primary signal direction matched the label — it did not check content quality. AIZ passed because its primary signal agreed with its escalating label despite having financial statement content mixed in. FANG_2022_2023 was also incorrectly included in the whitelist initially.

**Iterations required:**

Round 1: Added whitelist, kept secondary fallback using `signal_agrees()`. AIZ still appeared via fallback.

Round 2: Removed FANG_2022_2023 from whitelist. AIZ still appeared via fallback.

Round 3: Removed AIZ from whitelist. AIZ still appeared via `signal_agrees()` fallback because its signal agreed with its label.

Round 4: Added `has_clean_summary()` as a mandatory hard gate on ALL selection paths including whitelist, signal-agrees fallback, and absolute fallback. FINANCIAL_ARTIFACTS list defined at module level and checked before any example is selected. This permanently blocked AIZ and any other pair with financial statement content from appearing as few-shot examples regardless of which fallback path was triggered.

**Final example quality verified across 5 consecutive dry runs before full experiment was run.**

---

### Problem 23: Stable Class Recall Collapse (June 2026)

**What happened:** After fixing the above problems and achieving kappa 0.494, stable class recall remained at 2/7 (28.6%). The model correctly identified all de-escalating cases but could not reliably detect stable pairs.

**Root cause:** Stable pairs by definition show minimal diff signal — nothing changed, so there is little content in the diff summary. The LLM defaults to predicting escalating or de-escalating based on whatever weak signal exists rather than recognizing the low-signal case as stable.

Additionally, the few-shot examples for stable class had only 6 added sentences with no COVID signal — not enough for the model to learn what a stable pair looks like.

**Current status:** Unresolved. Identified as the primary target for kappa improvement from 0.494 to 0.6+. Three proposed fixes: (1) add tiebreaker rule to prompt for cases where COVID reduced but non-COVID escalated; (2) use stable examples with higher sentence churn showing that volume change alone does not imply directional change; (3) run GPT-4 ablation to determine if the failure persists across model capabilities.

---

## Dataset Quality Notes

**Sector skew:** Technology companies have cleaner filing formats and higher extraction rates than banking. Banking has systematic gaps from incorporation by reference at WFC, USB, and BK.

**Temporal skew:** Dataset is weighted toward annual 10-K filings due to 10-Q no-change policy. This is methodologically acceptable since annual pairs are the primary unit of analysis.

**Survivorship bias:** All 100 companies were listed for the majority of the 2019-2024 period. SIVB (Silicon Valley Bank, collapsed March 2023) is a notable absence that would have been analytically interesting for liquidity risk escalation detection.

**Annotation quality:** Single annotator ground truth introduces subjectivity. Disagreement patterns between human and LLM annotations are directionally systematic rather than random, supporting the model bias interpretation over annotation noise. Second annotator validation using GPT-4 as independent annotator is planned to compute inter-annotator kappa.

---

## Limitations

Single annotator ground truth introduces subjectivity. Second annotator validation planned using GPT-4 as independent LLM annotator.

Yahoo Finance abnormal return proxies are less precise than CDS spreads. WRDS Markit access pending for summer 2026.

LLM outputs are non-deterministic across runs. Temperature set to 0 and seed fixed at 42 for reproducibility.

Coverage limited to US-listed companies with English filings.

Incorporation by reference gaps affect approximately 15% of banking sector filings.

ICL baseline uses Llama-3 8B only. Multi-model ablation (GPT-4, Llama-3 70B) needed to confirm that de-escalation blindness persists across model capability levels — this would elevate the finding from a quirk of one model to a systematic property of current LLMs.

Stable class recall remains low (28.6%). The tiebreaker case — COVID reduced but non-COVID escalated — is not yet handled.

---

## Future Work

**Near term:**

GPT-4 ablation on same 36 pairs and diff representation to test generalization of de-escalation blindness finding across model capability levels.

Add tiebreaker rule to ICL prompt: if COVID language reduced but 10+ new non-COVID sentences added, classify as escalating.

Annotate 10 more pairs targeting stable class and non-COVID escalating to improve CV stability (current std 0.329 reflects small dataset variance).

Run adversarial elicitation test: ask LLM directly "has risk disclosure been reduced?" and compare to classification task results. Characterizes the failure mode more precisely.

**Medium term:**

arXiv preprint: "De-escalation Blindness in LLM-based Risk Monitoring: A Capability Gap for Scalable Oversight." Target NeurIPS 2026 Behavioral Evaluation workshop or ICLR 2027 Trustworthy ML workshop.

SEC comment letter validation via WRDS — external ground truth for which disclosures were deemed inadequate by regulators.

Alignment Forum post on curriculum safety coverage analysis from CIC RA work (target: before MATS winter application).

**Longer term:**

CDS spread validation via WRDS Markit.

LLM generation experiments: prompt models to generate synthetic risk disclosures under evaluation pressure and test whether the evasion detector identifies the same patterns.

SIVB and First Republic Bank as stress-period validation cases for liquidity risk escalation detection.

Real-time EDGAR monitoring pipeline.

---

## Project Structure

```
financial-disclosure-risk-intelligence/
├── src/
│   ├── ingestion/          # SEC EDGAR data collection
│   ├── preprocessing/      # Text cleaning, pair construction
│   ├── modeling/           # LLM risk detection pipeline
│   ├── evaluation/         # Kappa scores, regression analysis
│   └── visualization/      # Streamlit app, Plotly charts
├── diffs/
│   ├── diff_representations.jsonl     # Structured diff for 36 clean pairs
│   └── diff_representations_readable.txt
├── results/
│   ├── icl_diff_3shot_results.json
│   └── icl_diff_6shot_results.json
├── data/
│   ├── raw/                # Raw EDGAR filings (gitignored)
│   └── processed/          # Cleaned filing pairs (gitignored)
├── notebooks/              # Exploratory analysis
├── tests/                  # Unit tests
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
python build_diff_v2.py --input_dirs extracted_full deescalating_v7 deescalating_v8 deescalating_v9 extracted_missing --annotations annotations_final_v3.txt --output_dir diffs

# Run ICL baseline
python icl_baseline_v2.py --input diffs/diff_representations.jsonl --output results --model llama3:latest --shots 3 6 --condition diff
```

---

## Author

Sudhiksha Kandavel Rajan
MS Artificial Intelligence, Northeastern University
[LinkedIn](https://linkedin.com/in/sudhiksha-kandavel-rajan) · [GitHub](https://github.com/Sudhiksha-17) · [HUX AI Paper](https://arxiv.org/abs/2407.19492)