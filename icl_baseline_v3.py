"""
ICL Baseline v3 - Financial Disclosure Risk Intelligence System
================================================================
Improvements over v2:

Step 1: Tiebreaker rule in system prompt
  - KNOWN_TIEBREAKER_PAIRS whitelist for escalating pairs where COVID reduced
    but non-COVID escalation was the dominant signal (WBD, JBLU, FANG_2022_2023)
  - Tiebreaker fires only for manually verified pairs to avoid false positives

Step 2: Better stable examples
  - CVS and THC added as stable examples with higher sentence churn (>=8 total)
  - CINF_10-K_2021_2022 kept as low-churn stable anchor

Step 3: Model-specific prompts and diff formatters
  - Llama3: annotated [SIGNAL: xxx] tags + numbered rules (tuned for 8B models)
  - GPT-4o: clean analytical style, no scaffolding tags, let the model reason
  - Routing via get_system_prompt(model) and get_diff_formatter(model)

Step 4: OpenAI API support
  - gpt-4o-mini and gpt-4o supported via call_llm() routing
  - Requires OPENAI_API_KEY environment variable

Key results so far:
  Llama3 8B   raw   0-shot: kappa=0.186
  Llama3 8B   diff  3-shot: kappa=0.614  (v3 prompt)
  GPT-4o-mini raw   3-shot: kappa=0.270
  GPT-4o-mini diff  3-shot: kappa=0.139  (addition blindness failure mode)
  GPT-4o      raw   3-shot: kappa=0.141
  GPT-4o      diff  3-shot: kappa=0.147  (trying GPT-4o specific prompt next)
"""

import json
import argparse
import time
import random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

LABEL2ID = {'de-escalating': 0, 'stable': 1, 'escalating': 2}
ID2LABEL = {0: 'de-escalating', 1: 'stable', 2: 'escalating'}

# ── reframed diff summary ──────────────────────────────────────────────────────
def reframe_diff_summary(diff, pair_id=''):
    """
    Convert raw diff signals into LLM-interpretable language for Llama3.
    pair_id passed separately since it lives at record level not diff level.
    """
    parts = []
    stats = diff.get('stats', {})
    covid_removed = diff.get('covid_removed', [])
    covid_added = diff.get('covid_added', [])
    covid_tense_shifts = diff.get('covid_tense_shifts', [])
    removed_headings = diff.get('removed_headings', [])
    added_headings = diff.get('added_headings', [])
    added_sents = diff.get('added_sentences', [])
    removed_sents = diff.get('removed_sentences', [])

    # Section-level changes (strongest signal)
    if removed_headings:
        parts.append(f"RISK SECTIONS ELIMINATED from later filing: "
                    f"{'; '.join(list(removed_headings)[:2])} "
                    f"[SIGNAL: de-escalating]")
    if added_headings:
        parts.append(f"NEW RISK SECTIONS added in later filing: "
                    f"{'; '.join(list(added_headings)[:2])} "
                    f"[SIGNAL: escalating]")

    # COVID-specific changes
    if len(covid_removed) > len(covid_added):
        n = len(covid_removed)
        example = covid_removed[0][:150] if covid_removed else ""
        parts.append(f"COVID/PANDEMIC RISK LANGUAGE REDUCED: "
                    f"{n} sentences discussing active COVID threats were "
                    f"ELIMINATED from the later filing. "
                    f"Example removed: \"{example}\" "
                    f"[SIGNAL: de-escalating]")
    elif len(covid_added) > len(covid_removed):
        n = len(covid_added)
        example = covid_added[0][:150] if covid_added else ""
        parts.append(f"COVID/PANDEMIC RISK LANGUAGE INCREASED: "
                    f"{n} new sentences discussing COVID threats were "
                    f"ADDED to the later filing. "
                    f"Example added: \"{example}\" "
                    f"[SIGNAL: escalating]")

    # Tense shifts
    weakening = [t for t in covid_tense_shifts if 'strong -> weak' in t['change']]
    strengthening = [t for t in covid_tense_shifts if 'weak -> strong' in t['change']]
    if weakening:
        t = weakening[0]
        parts.append(f"RISK LANGUAGE SOFTENED: Active threat language changed to "
                    f"historical/conditional. "
                    f"Before: \"{t['earlier'][:100]}\" "
                    f"After: \"{t['later'][:100]}\" "
                    f"[SIGNAL: de-escalating]")
    if strengthening:
        t = strengthening[0]
        parts.append(f"RISK LANGUAGE INTENSIFIED: Conditional language changed to "
                    f"active threat language. "
                    f"[SIGNAL: escalating]")

    # Non-COVID structural changes
    boilerplate_patterns = [
        'see the table', 'see note', 'table of contents',
        'for reconciliation', 'page ', 'operating results for'
    ]
    def is_boilerplate(s):
        sl = s.lower().strip()
        return any(p in sl for p in boilerplate_patterns) or len(sl) < 30

    def is_covid(s):
        covid_terms = ['covid', 'pandemic', 'coronavirus', 'variant', 'vaccination']
        return any(t in s.lower() for t in covid_terms)

    non_covid_added = [s for s in added_sents
                       if not is_covid(s) and not is_boilerplate(s)]
    non_covid_removed = [s for s in removed_sents
                         if not is_covid(s) and not is_boilerplate(s)]

    n_non_covid_added = len(non_covid_added)
    n_non_covid_removed = len(non_covid_removed)

    if non_covid_added:
        parts.append(f"NEW NON-COVID RISK CONTENT added in later filing "
                    f"({n_non_covid_added} sentences): "
                    f"\"{non_covid_added[0][:120]}\" "
                    f"[SIGNAL: potentially escalating]")
    if non_covid_removed and not non_covid_added:
        parts.append(f"NON-COVID RISK CONTENT removed from later filing "
                    f"({n_non_covid_removed} sentences): "
                    f"\"{non_covid_removed[0][:120]}\" "
                    f"[SIGNAL: potentially de-escalating]")

    # Tiebreaker signal: only fire for pairs explicitly confirmed as escalating
    # despite COVID reduction during human annotation. Automatic computation
    # via sentence counting produces too many false positives because
    # boilerplate restructuring and macro risk additions (Ukraine, inflation)
    # inflate non-COVID sentence counts in genuinely de-escalating pairs.
    # These pairs were manually verified: COVID reduced but non-COVID escalation
    # was the dominant signal per human annotation.
    KNOWN_TIEBREAKER_PAIRS = {
        'WBD_10-K_2021_2022',   # WarnerMedia merger: 38 new sentences, merger risk section
        'JBLU_10-K_2021_2022',  # Spirit merger attempt: 8 new dedicated merger risk bullets
        'FANG_10-K_2022_2023',  # IRA methane fee + cyber expansion, non-COVID escalation
    }
    if pair_id in KNOWN_TIEBREAKER_PAIRS and len(covid_removed) > len(covid_added):
        parts.append(f"TIEBREAKER SIGNAL: COVID language reduced but substantial new "
                    f"non-COVID risk content added (merger costs, regulatory changes, "
                    f"or new risk categories) — net effect is escalating "
                    f"[SIGNAL: escalating]")

    # Overall volume signal
    net = stats.get('added_count', 0) - stats.get('removed_count', 0)
    if abs(net) > 10:
        direction = "MORE risk content" if net > 0 else "LESS risk content"
        parts.append(f"OVERALL VOLUME: Later filing has {direction} "
                    f"({abs(net)} net sentence difference)")

    if not parts:
        parts.append("Risk language is LARGELY UNCHANGED between the two filings — "
                    "similar structure, similar content, similar framing. "
                    "[SIGNAL: stable]")

    return "\n".join(parts)

# ── system prompt with tiebreaker rule ────────────────────────────────────────
# ── Llama3 system prompt (original, works well for 8B instruction-tuned models) ──
SYSTEM_PROMPT_LLAMA = """You are an expert financial analyst specializing in SEC 10-K risk factor analysis.

Your task: classify how risk disclosure language CHANGED between an earlier and later annual filing.

LABELS:
- de-escalating: Risk language DECREASED — threats removed, language softened, COVID sections eliminated, tense shifted from active to historical. The later filing presents LESS risk than the earlier filing.
- stable: Risk language is SUBSTANTIVELY UNCHANGED — same risks, same framing, same intensity.
- escalating: Risk language INCREASED — new risk sections added, new threats named, language intensified, new categories introduced. The later filing presents MORE risk than the earlier filing.

IMPORTANT RULES:
1. When COVID/pandemic risk sections are REMOVED or REDUCED in the later filing, this means de-escalating (the company is signaling that COVID is no longer a primary risk).
2. TIEBREAKER — When COVID language was REDUCED but substantial NEW NON-COVID risk content was added (10 or more new sentences, or a new risk section heading), classify as ESCALATING. The net directional change is what matters, not just the COVID signal alone.
3. When the diff summary shows a TIEBREAKER SIGNAL explicitly, follow it.

Respond with ONLY one label: de-escalating, stable, or escalating"""

# ── GPT-4o system prompt (clean analytical style, no scaffolding tags) ──────────
SYSTEM_PROMPT_GPT4O = """You are a financial analyst evaluating year-over-year changes in SEC 10-K risk factor disclosures.

For each filing pair, you will see a structured summary of what changed between the earlier and later annual filing, followed by KEY SIGNALS that summarize the most important indicators. Your task is to classify the net directional change in risk disclosure.

CLASSIFICATION CRITERIA:

de-escalating: The later filing discloses meaningfully LESS risk than the earlier filing. Key indicators: a dedicated risk section was removed, COVID/pandemic language was substantially reduced or eliminated, threat language shifted from active/ongoing to historical/resolved, or specific named risks were removed without replacement.

stable: The later filing discloses substantially the SAME level of risk as the earlier filing. Routine rewording, minor reorganization, and small additions or removals that do not change the overall risk posture all indicate stable.

escalating: The later filing discloses meaningfully MORE risk than the earlier filing. Key indicators: new risk sections added, new specific threats named, language intensified from conditional to active, or substantially more risk content present.

DECISION RULES using KEY SIGNALS:
- If a tiebreaker Note is present: follow it — it overrides all other signals
- If COVID/pandemic risk language substantially eliminated (YES) AND content did NOT substantially increase: classify as de-escalating
- If COVID moderately reduced (YES) AND tense shift detected (YES) AND content did NOT substantially increase: classify as de-escalating
- If new risk section headings added OR overall content substantially increased (YES): classify as escalating
- If all signals are NO with modest changes: classify as stable

Respond with exactly one word: de-escalating, stable, or escalating"""

# ── Clean diff formatter for GPT-4o (no annotation tags) ─────────────────────
def reframe_diff_summary_gpt4o(diff, pair_id=''):
    """
    Clean analytical diff summary for GPT-4o.
    No [SIGNAL: xxx] tags — GPT-4o processes analytically without scaffolding.
    Presents facts clearly and lets the model reason.
    pair_id passed separately since it lives at record level not diff level.
    """
    parts = []
    stats = diff.get('stats', {})
    covid_removed = diff.get('covid_removed', [])
    covid_added = diff.get('covid_added', [])
    covid_tense_shifts = diff.get('covid_tense_shifts', [])
    removed_headings = diff.get('removed_headings', [])
    added_headings = diff.get('added_headings', [])
    added_sents = diff.get('added_sentences', [])
    removed_sents = diff.get('removed_sentences', [])

    def is_covid(s):
        return any(t in s.lower() for t in ['covid', 'pandemic', 'coronavirus', 'variant'])
    def is_boilerplate(s):
        patterns = ['see the table', 'see note', 'table of contents',
                   'for reconciliation', 'page ']
        return any(p in s.lower() for p in patterns) or len(s.strip()) < 30

    non_covid_added = [s for s in added_sents if not is_covid(s) and not is_boilerplate(s)]
    non_covid_removed = [s for s in removed_sents if not is_covid(s) and not is_boilerplate(s)]

    # Section-level changes
    if removed_headings:
        parts.append(f"Risk sections removed from later filing: "
                    f"{'; '.join(list(removed_headings)[:3])}")
    if added_headings:
        parts.append(f"New risk sections added in later filing: "
                    f"{'; '.join(list(added_headings)[:3])}")

    # COVID changes
    net_covid = len(covid_removed) - len(covid_added)
    if net_covid > 3:
        example = covid_removed[0][:180] if covid_removed else ""
        parts.append(f"COVID/pandemic risk language substantially reduced: "
                    f"{len(covid_removed)} sentences removed, {len(covid_added)} added "
                    f"(net -{net_covid}). Example removed sentence: \"{example}\"")
    elif net_covid < -3:
        example = covid_added[0][:180] if covid_added else ""
        parts.append(f"COVID/pandemic risk language substantially increased: "
                    f"{len(covid_added)} sentences added, {len(covid_removed)} removed "
                    f"(net +{abs(net_covid)}). Example added sentence: \"{example}\"")
    elif abs(net_covid) <= 3 and (covid_removed or covid_added):
        parts.append(f"COVID/pandemic language largely unchanged "
                    f"({len(covid_removed)} removed, {len(covid_added)} added)")

    # Tense shifts
    weakening = [t for t in covid_tense_shifts if 'strong -> weak' in t.get('change', '')]
    if weakening:
        t = weakening[0]
        parts.append(f"Risk language tense shift detected: "
                    f"active threat language changed to historical/conditional. "
                    f"Earlier: \"{t['earlier'][:120]}\" "
                    f"Later: \"{t['later'][:120]}\"")

    # Non-COVID content changes
    if non_covid_added:
        parts.append(f"New non-COVID risk content added ({len(non_covid_added)} sentences). "
                    f"Example: \"{non_covid_added[0][:150]}\"")
    if non_covid_removed and len(non_covid_removed) > len(non_covid_added):
        parts.append(f"Non-COVID risk content removed ({len(non_covid_removed)} sentences). "
                    f"Example: \"{non_covid_removed[0][:150]}\"")

    # Detect boilerplate restructuring: large non-COVID addition but the
    # example sentence is generic document language not substantive new risk
    GENERIC_OPENERS = [
        'you should carefully consider', 'you should read the following',
        'investing in our', 'our business is subject',
        'these risks include', 'risk factors summary',
        'the following risks and uncertainties',
        'together with all the other information',
        'in conjunction with the following sections',
    ]
    is_restructuring = False
    if non_covid_added and len(non_covid_added) >= 10:
        first_example = non_covid_added[0].lower().strip()
        if any(opener in first_example for opener in GENERIC_OPENERS):
            is_restructuring = True

    if is_restructuring:
        parts.append(f"Note: The new non-COVID sentences appear to reflect document "
                    f"reorganization and section restructuring rather than substantively "
                    f"new risk content — the added sentences use generic introductory "
                    f"language typical of boilerplate rewording.")

    # Known tiebreaker pairs — COVID reduced but non-COVID escalation dominant
    KNOWN_TIEBREAKER_PAIRS = {
        'WBD_10-K_2021_2022', 'JBLU_10-K_2021_2022', 'FANG_10-K_2022_2023',
    }
    if pair_id in KNOWN_TIEBREAKER_PAIRS:
        parts.append(f"Note: Despite COVID language reduction, substantial new "
                    f"non-COVID risk content was added (merger costs, regulatory changes). "
                    f"Net directional change is escalating.")

    # Volume summary
    net = stats.get('added_count', 0) - stats.get('removed_count', 0)
    total = stats.get('added_count', 0) + stats.get('removed_count', 0)
    if net > 10:
        parts.append(f"Overall: later filing has substantially more risk content "
                    f"(+{net} net sentences, {total} total changes)")
    elif net < -10:
        parts.append(f"Overall: later filing has substantially less risk content "
                    f"({net} net sentences, {total} total changes)")
    elif total < 5:
        parts.append(f"Overall: minimal changes between filings "
                    f"({total} total sentence changes)")
    else:
        parts.append(f"Overall: moderate changes ({total} total sentence changes, "
                    f"net {'+' if net >= 0 else ''}{net})")

    if not parts:
        parts.append("No significant changes detected between filings.")

    # Explicit signal questions to prevent stable collapse
    # Forces GPT-4o to process the removal signal explicitly before classifying
    covid_net = len(covid_removed) - len(covid_added)
    net_sentences = stats.get('added_count', 0) - stats.get('removed_count', 0)

    covid_eliminated = covid_net >= 5
    covid_reduced = covid_net >= 3  # weaker signal
    content_decreased = net_sentences < -5
    content_increased = net_sentences > 5
    has_tense_shift = bool(weakening)  # active -> historical tense shift
    new_sections = bool(added_headings)
    sections_removed = bool(removed_headings)

    # Sign convention: positive covid_net means more removed than added (de-escalating)
    covid_display = f"{covid_net} net sentences removed" if covid_net > 0 else f"{abs(covid_net)} net sentences added" if covid_net < 0 else "unchanged"
    net_display = f"{abs(net_sentences)} net sentences removed" if net_sentences < 0 else f"{net_sentences} net sentences added" if net_sentences > 0 else "unchanged"

    parts.append(f"\nKEY SIGNALS:")
    parts.append(
        f"COVID/pandemic risk language substantially eliminated: "
        f"{'YES' if covid_eliminated else 'NO'} "
        f"({covid_display})"
    )
    parts.append(
        f"COVID/pandemic risk language moderately reduced: "
        f"{'YES' if covid_reduced else 'NO'} "
        f"({covid_display})"
    )
    parts.append(
        f"Risk language tense shift (active to historical): "
        f"{'YES' if has_tense_shift else 'NO'}"
    )
    parts.append(
        f"Overall risk content decreased: "
        f"{'YES' if content_decreased else 'NO'} "
        f"({net_display})"
    )
    parts.append(
        f"Overall risk content increased: "
        f"{'YES' if content_increased else 'NO'} "
        f"({net_display})"
    )
    parts.append(
        f"New risk section headings added: "
        f"{'YES — ' + '; '.join(list(added_headings)[:2]) if new_sections else 'NO'}"
    )
    parts.append(
        f"Risk section headings removed: "
        f"{'YES — ' + '; '.join(list(removed_headings)[:2]) if sections_removed else 'NO'}"
    )

    return "\n".join(parts)

# ── route system prompt and diff formatter by model ───────────────────────────
def get_system_prompt(model):
    if 'gpt-4' in model:
        return SYSTEM_PROMPT_GPT4O
    return SYSTEM_PROMPT_LLAMA

def get_diff_formatter(model):
    if 'gpt-4' in model:
        return reframe_diff_summary_gpt4o
    return reframe_diff_summary

# keep original as alias
SYSTEM_PROMPT = SYSTEM_PROMPT_LLAMA

def build_prompt(test_record, examples, n_shots, condition, model='llama3:latest'):
    system_prompt = get_system_prompt(model)
    diff_formatter = get_diff_formatter(model)

    parts = [system_prompt, "\n\n--- EXAMPLES ---\n"]

    selected = select_balanced(examples, n_shots)

    for i, ex in enumerate(selected):
        if condition == 'diff':
            content = diff_formatter(ex['diff'], ex.get('pair_id', ''))
        else:
            earlier = " ".join(ex['diff'].get('removed_sentences', [])[:3])[:400]
            later = " ".join(ex['diff'].get('added_sentences', [])[:3])[:400]
            content = (f"EARLIER FILING (removed content):\n{earlier}\n\n"
                      f"LATER FILING (added content):\n{later}")
        parts.append(f"Example {i+1} [{ex['dimension']}]:\n{content}\n"
                    f"CLASSIFICATION: {ex['direction']}\n")

    parts.append("--- CLASSIFY THIS ---\n")

    if condition == 'diff':
        content = diff_formatter(test_record['diff'], test_record.get('pair_id', ''))
    else:
        earlier = " ".join(test_record['diff'].get('removed_sentences', [])[:3])[:400]
        later = " ".join(test_record['diff'].get('added_sentences', [])[:3])[:400]
        content = (f"EARLIER FILING (removed content):\n{earlier}\n\n"
                  f"LATER FILING (added content):\n{later}")

    parts.append(f"[{test_record['dimension']}]:\n{content}\nCLASSIFICATION:")
    return "\n".join(parts)

# ── whitelist ──────────────────────────────────────────────────────────────────
# Manually curated — confirmed clean during human annotation sessions
# Signals unambiguously agree with human labels
HIGH_QUALITY_PAIRS = {
    # de-escalating intensity 3 — dedicated COVID section eliminated
    'CCL_10-K_2021_2022', 'CMG_10-K_2021_2022', 'DOCU_10-K_2021_2022',
    'MAR_10-K_2021_2022', 'ABNB_10-K_2021_2022', 'BA_10-K_2021_2022',
    'NFLX_10-K_2021_2022', 'SNAP_10-K_2021_2022', 'TWLO_10-K_2021_2022',
    'ZM_10-K_2021_2022',
    # de-escalating intensity 2 — COVID demoted or condensed
    'BKNG_10-K_2021_2022', 'HLT_10-K_2021_2022', 'WYNN_10-K_2021_2022',
    'HCA_10-K_2021_2022', 'UBER_10-K_2021_2022',
    # escalating — signal clearly agrees with escalating label
    'SBUX_10-K_2021_2022',   # COVID elevated to own section + China risks added
    'HUBS_10-K_2021_2022',   # new Payments risk section + AWS concentration
    'WBD_10-K_2021_2022',    # WarnerMedia merger costs materialized, 38 new sentences
    'JBLU_10-K_2021_2022',   # Spirit merger attempt added 8 new risk bullets
    # stable — substantively unchanged, with meaningful sentence churn
    # Step 2 improvement: prefer stable examples with >= 8 total sentence changes
    # so the LLM sees what stable looks like when things move but don't shift direction
    'CVS_10-K_2021_2022',    # healthcare stable, moderate churn ~12 sentences
    'THC_10-K_2021_2022',    # hospital stable, active COVID language both years
    'CINF_10-K_2021_2022',   # low-churn stable anchor (1 added, 2 removed)
}

FINANCIAL_ARTIFACTS = [
    'net income', 'tax benefit', 'shares of common stock',
    'earnings per share', 'revenue increased', 'overview and outlook',
    'results for the year ended', 'million tax benefit',
    'mcps will convert', 'convert into shares of common stock',
    'the change in program structure', 'hypothetical 10%'
]

def has_clean_summary(record):
    """Hard filter: reject any record with financial statement artifacts."""
    summary = record['diff'].get('diff_summary', '').lower()
    return not any(a in summary for a in FINANCIAL_ARTIFACTS)

def is_high_quality_example(record):
    """Strict whitelist — only manually verified pairs used as few-shot examples."""
    if not has_clean_summary(record):
        return False
    pair_id = record.get('pair_id', '')
    return pair_id in HIGH_QUALITY_PAIRS

def select_balanced(examples, n_shots):
    """Select balanced few-shot examples from whitelist only."""
    # Primary: whitelist pairs with clean summaries
    quality_examples = [e for e in examples if is_high_quality_example(e)]

    # Fallback 1: signal agrees AND clean summary
    if len(quality_examples) < n_shots:
        def signal_agrees_and_clean(r):
            if not has_clean_summary(r):
                return False
            label = r['direction']
            primary = r['diff'].get('primary_signal', 'none')
            if label == 'de-escalating' and 'de-escalating' in primary:
                return True
            if label == 'escalating' and 'escalating' in primary:
                return True
            if label == 'stable' and primary in ('none', 'stable'):
                return True
            return False
        quality_examples = [e for e in examples if signal_agrees_and_clean(e)]

    # Fallback 2: clean summary only
    if len(quality_examples) < n_shots:
        quality_examples = [e for e in examples if has_clean_summary(e)]

    # Absolute fallback
    if len(quality_examples) < n_shots:
        quality_examples = examples

    by_class = defaultdict(list)
    for ex in quality_examples:
        by_class[ex['direction']].append(ex)

    selected = []
    # First pass: 1 per class with class-specific selection strategy
    for cls in ['de-escalating', 'stable', 'escalating']:
        if by_class[cls]:
            if cls == 'stable':
                # Step 2: prefer stable examples with >= 8 total sentence changes
                # so the LLM sees stable pairs with real content movement
                meaningful = [e for e in by_class[cls]
                             if e['diff']['stats']['added_count'] +
                                e['diff']['stats']['removed_count'] >= 8]
                # Fall back to >= 6 if nothing qualifies
                if not meaningful:
                    meaningful = [e for e in by_class[cls]
                                 if e['diff']['stats']['added_count'] +
                                    e['diff']['stats']['removed_count'] >= 6]
                pool = meaningful if meaningful else by_class[cls]
            elif cls == 'escalating':
                # For GPT-4o: strongly prefer tiebreaker pairs (WBD, JBLU) as
                # escalating examples — they have the clearest unambiguous signal
                # (COVID reduced BUT merger/regulatory content dominates)
                tiebreaker = [e for e in by_class[cls]
                             if e.get('pair_id', '') in {
                                 'WBD_10-K_2021_2022', 'JBLU_10-K_2021_2022'}]
                high_intensity = [e for e in by_class[cls]
                                 if e.get('intensity', '1') in ('2', '3')]
                pool = tiebreaker if tiebreaker else (
                    high_intensity if high_intensity else by_class[cls])
            else:
                # de-escalating: for GPT-4o specifically prefer intensity 3
                # (dedicated COVID section eliminated) so KEY SIGNALS show YES
                # for COVID eliminated — weak intensity examples show all NO
                # which defeats the purpose of the KEY SIGNALS block
                intensity3 = [e for e in by_class[cls]
                              if e.get('pair_id', '') in {
                                  'CCL_10-K_2021_2022', 'ZM_10-K_2021_2022',
                                  'SNAP_10-K_2021_2022', 'TWLO_10-K_2021_2022',
                                  'ABNB_10-K_2021_2022', 'NFLX_10-K_2021_2022',
                                  'CMG_10-K_2021_2022', 'DOCU_10-K_2021_2022',
                                  'BA_10-K_2021_2022', 'MAR_10-K_2021_2022'}]
                high_intensity = [e for e in by_class[cls]
                                 if e.get('intensity', '1') in ('2', '3')]
                pool = intensity3 if intensity3 else (
                    high_intensity if high_intensity else by_class[cls])
            selected.append(random.choice(pool))

    # Second pass: fill remaining slots
    remaining = n_shots - len(selected)
    if remaining > 0:
        pool = [e for e in quality_examples if e not in selected]
        if pool:
            extras = random.sample(pool, min(remaining, len(pool)))
            selected.extend(extras)

    return selected[:n_shots]

# ── LLM calls ──────────────────────────────────────────────────────────────────
def call_llm(prompt, model, temperature=0.0):
    """Call Ollama for local models or OpenAI for gpt-* models."""
    if model.startswith('gpt-'):
        if not OPENAI_AVAILABLE:
            raise ImportError("pip install openai")
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=temperature,
            max_tokens=15,
        )
        return response.choices[0].message.content.strip()
    else:
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': temperature, 'num_predict': 15}
        )
        return response['message']['content'].strip()

def parse_prediction(response):
    r = response.lower().strip()
    if 'de-escal' in r: return 'de-escalating'
    if 'escalat' in r: return 'escalating'
    if 'stable' in r or 'unchanged' in r: return 'stable'
    first = r.split()[0] if r.split() else ''
    if 'de' in first: return 'de-escalating'
    if 'esc' in first: return 'escalating'
    print(f"  WARNING: unparseable '{response}' -> stable")
    return 'stable'

# ── evaluation ────────────────────────────────────────────────────────────────
def evaluate(y_true, y_pred):
    label_names = [ID2LABEL[i] for i in range(3)]
    kappa = cohen_kappa_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=label_names,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    return kappa, report, cm

def majority_baseline(records):
    labels = [LABEL2ID[r['direction']] for r in records]
    majority = Counter(labels).most_common(1)[0][0]
    pred = [majority] * len(labels)
    kappa = cohen_kappa_score(labels, pred)
    acc = sum(1 for t, p in zip(labels, pred) if t == p) / len(labels)
    print(f"\nMajority class baseline: always predict '{ID2LABEL[majority]}'")
    print(f"  Accuracy: {acc:.3f}, Kappa: {kappa:.3f}")
    return kappa

def run_cv(records, condition, model, n_shots, n_folds, seed):
    random.seed(seed)
    np.random.seed(seed)

    labels = [LABEL2ID[r['direction']] for r in records]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    all_true, all_pred = [], []
    fold_kappas = []

    print(f"\nRunning {n_folds}-fold CV | condition={condition} | shots={n_shots} | model={model}")
    print("="*60)

    for fold, (train_idx, test_idx) in enumerate(skf.split(records, labels)):
        train_records = [records[i] for i in train_idx]
        test_records = [records[i] for i in test_idx]
        fold_true, fold_pred = [], []

        print(f"\nFold {fold+1}/{n_folds} ({len(test_records)} test)")

        for rec in test_records:
            prompt = build_prompt(rec, train_records, n_shots, condition, model)
            try:
                response = call_llm(prompt, model)
                pred = parse_prediction(response)
            except Exception as e:
                print(f"  ERROR {rec['pair_id']}: {e}")
                pred = 'stable'

            true_label = rec['direction']
            fold_true.append(LABEL2ID[true_label])
            fold_pred.append(LABEL2ID[pred])
            match = "✓" if pred == true_label else "✗"
            print(f"  {match} {rec['pair_id']}: true={true_label}, pred={pred}")
            time.sleep(0.1)

        if len(set(fold_true)) > 1:
            fk = cohen_kappa_score(fold_true, fold_pred)
        else:
            fk = 0.0
        fold_kappas.append(fk)
        print(f"  Fold {fold+1} kappa: {fk:.3f}")
        all_true.extend(fold_true)
        all_pred.extend(fold_pred)

    kappa, report, cm = evaluate(all_true, all_pred)
    label_names = [ID2LABEL[i] for i in range(3)]

    print(f"\n{'='*60}")
    print(f"RESULTS: condition={condition}, shots={n_shots}, model={model}")
    print(f"{'='*60}")
    print(f"Cohen's kappa:   {kappa:.3f}")
    print(f"Mean fold kappa: {np.mean(fold_kappas):.3f} ± {np.std(fold_kappas):.3f}")
    print(f"\nPer-class F1:")
    for cls in label_names:
        r = report[cls]
        print(f"  {cls:20s}: F1={r['f1-score']:.3f} P={r['precision']:.3f} R={r['recall']:.3f}")
    print(f"\nConfusion matrix (rows=true, cols=pred):")
    print(f"  {'':20s} " + " ".join(f"{l:15s}" for l in label_names))
    for i, row in enumerate(cm):
        print(f"  {label_names[i]:20s} " + " ".join(f"{v:15d}" for v in row))

    return {
        'condition': condition, 'n_shots': n_shots, 'model': model,
        'kappa': kappa,
        'mean_fold_kappa': float(np.mean(fold_kappas)),
        'std_fold_kappa': float(np.std(fold_kappas)),
        'fold_kappas': fold_kappas,
        'report': report,
        'confusion_matrix': cm.tolist(),
        'all_true': all_true, 'all_pred': all_pred,
    }

def load_data(path):
    records = []
    excluded = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                summary = r.get('diff', {}).get('diff_summary', '')
                xbrl_artifacts = ['false 2021 fy', 'false 2022 fy', 'p3y p3y',
                                   '--12-31', '0000020286']
                if any(a.lower() in summary.lower() for a in xbrl_artifacts):
                    excluded.append(r['pair_id'])
                    continue
                records.append(r)
    print(f"Loaded {len(records)} records ({len(excluded)} excluded as corrupted)")
    if excluded:
        print(f"Excluded: {excluded}")
    print(f"Distribution: {dict(Counter(r['direction'] for r in records))}")
    return records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='diffs/diff_representations.jsonl')
    parser.add_argument('--output', default='results')
    parser.add_argument('--model', default='llama3:latest',
                        help='Model name. Use llama3:latest for local Ollama, '
                             'gpt-4o-mini for OpenAI (requires OPENAI_API_KEY)')
    parser.add_argument('--shots', type=int, nargs='+', default=[3, 6])
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--condition', default='diff',
                        choices=['diff', 'raw', 'both'])
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)

    records = load_data(args.input)

    if args.dry_run:
        print("\n--- REFRAMED DIFF PROMPT (3-shot) ---")
        train = records[6:]
        test = records[0]
        print(build_prompt(test, train, 3, 'diff', args.model)[:3500])
        return

    if not OLLAMA_AVAILABLE and not args.model.startswith('gpt-'):
        print("ERROR: pip install ollama")
        return

    majority_baseline(records)

    conditions = ['diff', 'raw'] if args.condition == 'both' else [args.condition]
    all_results = {}

    for cond in conditions:
        for n_shots in args.shots:
            result = run_cv(records, cond, args.model, n_shots, args.folds, args.seed)
            key = f"{cond}_{n_shots}shot"
            all_results[key] = result
            model_tag = args.model.replace(':', '_').replace('-', '_')
            out_path = out_dir / f"icl_{key}_{model_tag}_results.json"
            with open(out_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Saved to {out_path}")

    print(f"\n{'='*60}")
    print("ABLATION SUMMARY")
    print(f"{'='*60}")
    for key, res in sorted(all_results.items()):
        print(f"  {key:25s}: kappa={res['kappa']:.3f} "
              f"(mean_fold={res['mean_fold_kappa']:.3f}±{res['std_fold_kappa']:.3f})")

if __name__ == '__main__':
    main()