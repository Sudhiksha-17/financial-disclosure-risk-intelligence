"""
Annotation Sheet Generator
===========================
Creates a CSV for manual annotation of 50 filing pairs.
Used to calculate Cohen's kappa against LLM outputs.

Labeling Protocol is printed to console and saved as a
separate reference file.

Usage:
    python create_annotation_sheet.py
"""

import json
import csv
import random
from pathlib import Path

random.seed(42)

SIGNALS_DIR = Path("data/processed/risk_signals")
PAIRS_DIR   = Path("data/processed/pairs")
OUTPUT_DIR  = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Labeling Protocol ─────────────────────────────────────────────────────────

PROTOCOL = """
ANNOTATION LABELING PROTOCOL
==============================
Financial Disclosure Risk Intelligence System
Human Annotation Baseline for Cohen's Kappa Calculation

OVERVIEW
--------
You are comparing two consecutive annual 10-K filings from the same
company. Your task is to assess how risk factor language changed
between the earlier and later filing across 5 dimensions.

Read both text snippets for each pair before annotating.
Use the full text columns (earlier_text_full, later_text_full)
for your judgment, not the truncated preview columns.


DIRECTION DEFINITIONS
----------------------

ESCALATING:
The later filing introduces new risk factors not present in the
earlier filing, expands existing ones with more specific language,
adds quantitative estimates of potential losses, uses stronger
warning language, or elevates previously minor risks to greater
prominence.

Examples of escalating language:
- New risk factor category introduced
- Specific dollar amounts of potential loss added
- Language shifts from "may" to "will" or "significant"
- New regulatory risk added following a change in law
- Liquidity risk expanded with specific covenant details

STABLE:
The language is substantively the same across both filings.
Minor wording changes, updated dates, renumbering, or trivial
edits do not constitute escalation or de-escalation.
The overall risk profile described is materially unchanged.

Examples of stable language:
- Same risk factors with minor rephrasing
- Updated year references only
- Reordering of existing risk factors
- Minor additions of one sentence or less

DE-ESCALATING:
The later filing removes risk factors present in the earlier
filing, reduces the specificity or severity of language, or
explicitly notes that a previously disclosed risk has diminished
or been resolved.

Examples of de-escalating language:
- Risk factor removed entirely
- Language shifts from "will" to "may"
- Specific loss estimates removed
- Previously prominent risk moved to minor footnote


INTENSITY RUBRIC (1 to 5)
--------------------------

1 = TRIVIAL
Essentially identical language. Only cosmetic edits such as
punctuation, dates, or minor rephrasing. No material difference
in risk disclosure.
Example: "We may experience liquidity constraints" becomes
"We may experience liquidity challenges"

2 = MINOR
Some new language added or removed but the overall risk profile
is similar. One small new sub-point or a slightly stronger
adjective. The reader would not materially change their view
of the company's risk based on this change.
Example: One new sentence added about a specific regulatory body.

3 = MODERATE
A new risk factor introduced, an existing one substantially
expanded with new specifics, or a previously minor risk elevated
to a more prominent position. The reader would notice this change
and it could affect their assessment.
Example: New section on cybersecurity risk added with two
paragraphs of detail not present in prior year.

4 = SIGNIFICANT
Multiple new risk factors, a materially different characterization
of an existing risk, or explicit new quantitative loss estimates
added. A credit analyst reading this would flag the change.
Example: Liquidity risk section doubles in length with new
specific scenarios and dollar amounts.

5 = MAJOR
Fundamental shift in how the company frames a risk category.
New systemic risk language, major new regulatory or credit
concerns introduced, or a previously undisclosed category of
risk now prominently featured with extensive new disclosure.
Example: Company adds entirely new section on going concern
risk or material uncertainty that was absent in prior year.


RISK DIMENSION DEFINITIONS
---------------------------

liquidity_risk:
Concerns about the company's ability to meet short-term
obligations, access to funding, cash and cash equivalents,
credit facilities, and working capital adequacy.

credit_risk:
Concerns about borrower default, counterparty risk, credit
quality of the portfolio, concentration risk, and credit
losses or provisions.

operational_risk:
Concerns about systems failures, process breakdowns, human
error, third-party vendor risk, cybersecurity, business
continuity, and key person dependencies.

market_risk:
Concerns about market volatility, interest rate sensitivity,
foreign exchange risk, commodity price risk, and general
macroeconomic conditions affecting the business.

regulatory_risk:
Concerns about changes in laws or regulations, compliance
requirements, examination by regulators, litigation risk,
and pending regulatory changes affecting the business.


ANNOTATION RULES
----------------

1. Read both text snippets fully before annotating any dimension.

2. Judge each dimension independently. A filing can have
   escalating credit risk and stable liquidity risk simultaneously.

3. When in doubt between two intensity levels, choose the lower one.
   Err toward conservative annotations.

4. If the text snippet is too short to judge a dimension, write
   "insufficient_text" in the direction column and 0 in intensity.

5. Add a brief note in the notes column for ANY pair where you
   chose escalating or de-escalating. One sentence is enough.
   Example: "New section on SVB contagion risk added in 2023 filing"

6. Do not let your knowledge of what actually happened to the
   company affect your annotation. Judge only the language change.

7. Length alone does not constitute escalation. A longer filing
   may simply be more verbose without materially changing risk.


VALID VALUES
------------
Direction columns: escalating, stable, de-escalating, insufficient_text
Intensity columns: 1, 2, 3, 4, 5, 0 (for insufficient_text cases)
"""

# ── Sheet generator ───────────────────────────────────────────────────────────

def create_annotation_sheet():

    # Print protocol
    print(PROTOCOL)

    # Save protocol as reference file
    protocol_path = OUTPUT_DIR / "annotation_protocol.txt"
    protocol_path.write_text(PROTOCOL, encoding="utf-8")
    print(f"Protocol saved: {protocol_path}")

    # Load signal files
    signal_files = list(SIGNALS_DIR.glob("*.json"))
    if not signal_files:
        print("No signal files found. Run risk_detector.py first.")
        return

    random.shuffle(signal_files)

    rows = []
    for f in signal_files:
        if len(rows) >= 50:
            break

        try:
            signal = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue

        pair_id  = signal.get("pair_id", "")
        ticker   = signal.get("ticker", "")
        year_e   = signal.get("year_earlier", "")
        year_l   = signal.get("year_later", "")
        signals  = signal.get("signals", {})

        pair_file = PAIRS_DIR / f"{pair_id}.json"
        if not pair_file.exists():
            continue

        try:
            pair = json.loads(pair_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        earlier_text = pair.get("earlier", {}).get("text", "")
        later_text   = pair.get("later", {}).get("text", "")

        if not earlier_text or not later_text:
            continue

        rows.append({
            # Metadata
            "pair_id":          pair_id,
            "ticker":           ticker,
            "year_earlier":     year_e,
            "year_later":       year_l,

            # LLM predictions (pre-filled, do not modify)
            "llm_liquidity_dir":    signals.get("liquidity_risk", {}).get("direction", ""),
            "llm_credit_dir":       signals.get("credit_risk", {}).get("direction", ""),
            "llm_operational_dir":  signals.get("operational_risk", {}).get("direction", ""),
            "llm_market_dir":       signals.get("market_risk", {}).get("direction", ""),
            "llm_regulatory_dir":   signals.get("regulatory_risk", {}).get("direction", ""),
            "llm_liquidity_int":    signals.get("liquidity_risk", {}).get("intensity", ""),
            "llm_credit_int":       signals.get("credit_risk", {}).get("intensity", ""),
            "llm_operational_int":  signals.get("operational_risk", {}).get("intensity", ""),
            "llm_market_int":       signals.get("market_risk", {}).get("intensity", ""),
            "llm_regulatory_int":   signals.get("regulatory_risk", {}).get("intensity", ""),

            # LLM justifications (for reference)
            "llm_liquidity_just":   signals.get("liquidity_risk", {}).get("justification", ""),
            "llm_credit_just":      signals.get("credit_risk", {}).get("justification", ""),
            "llm_operational_just": signals.get("operational_risk", {}).get("justification", ""),
            "llm_market_just":      signals.get("market_risk", {}).get("justification", ""),
            "llm_regulatory_just":  signals.get("regulatory_risk", {}).get("justification", ""),

            # Human annotation columns (fill these in)
            "human_liquidity_dir":    "",
            "human_credit_dir":       "",
            "human_operational_dir":  "",
            "human_market_dir":       "",
            "human_regulatory_dir":   "",
            "human_liquidity_int":    "",
            "human_credit_int":       "",
            "human_operational_int":  "",
            "human_market_int":       "",
            "human_regulatory_int":   "",

            # Notes column (required for any escalating/de-escalating annotation)
            "notes": "",

            # Text for reading (truncated preview)
            "earlier_text_preview": earlier_text[:800],
            "later_text_preview":   later_text[:800],

            # Full text for annotation judgment
            "earlier_text_full":    earlier_text[:3000],
            "later_text_full":      later_text[:3000],
        })

    if not rows:
        print("No valid pairs found.")
        return

    # Write CSV
    sheet_path = OUTPUT_DIR / "annotation_sheet.csv"
    with open(sheet_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nAnnotation sheet created: {sheet_path}")
    print(f"Pairs selected: {len(rows)}")
    print(f"\nColumn guide:")
    print(f"  llm_*       = LLM predictions, DO NOT modify")
    print(f"  human_*_dir = Fill with: escalating, stable, de-escalating")
    print(f"  human_*_int = Fill with: 1, 2, 3, 4, or 5")
    print(f"  notes       = Required for any non-stable annotation")
    print(f"\nRead annotation_protocol.txt before starting.")
    print(f"Annotate in Excel or Google Sheets.")
    print(f"Save as CSV when done.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    create_annotation_sheet()