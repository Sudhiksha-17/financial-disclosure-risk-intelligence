"""
Cohen's Kappa Calculator
=========================
Calculates Cohen's kappa between human annotations
and LLM predictions for each risk dimension.

Usage:
    python src/evaluation/calculate_kappa.py
"""

import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
from collections import Counter

SHEET_PATH  = Path("outputs/annotation_sheet.csv")
OUTPUT_DIR  = Path("outputs")

DIMENSIONS = {
    "liquidity":   ("human_liquidity_dir",   "llm_liquidity_dir"),
    "credit":      ("human_credit_dir",       "llm_credit_dir"),
    "operational": ("human_operational_dir",  "llm_operational_dir"),
    "market":      ("human_market_dir",       "llm_market_dir"),
    "regulatory":  ("human_regulatory_dir",   "llm_regulatory_dir"),
}

VALID_LABELS = ["escalating", "stable", "de-escalating"]


def run_kappa_analysis():
    df = pd.read_csv(SHEET_PATH)
    print("=" * 60)
    print("COHEN'S KAPPA ANALYSIS")
    print(f"Total rows: {len(df)}")
    print("=" * 60)

    results = {}

    for dim_name, (human_col, llm_col) in DIMENSIONS.items():
        # Filter to rows where human annotation is a valid label
        # excluding insufficient_text and empty
        mask = (
            df[human_col].isin(VALID_LABELS) &
            df[llm_col].isin(VALID_LABELS)
        )
        subset = df[mask]

        n = len(subset)
        if n < 5:
            print(f"\n{dim_name:15s}: insufficient data (n={n})")
            results[dim_name] = {
                "n": n,
                "kappa": None,
                "agreement_pct": None
            }
            continue

        human_labels = subset[human_col].tolist()
        llm_labels   = subset[llm_col].tolist()

        kappa = cohen_kappa_score(human_labels, llm_labels)
        agreement = sum(
            h == l for h, l in zip(human_labels, llm_labels)
        ) / n

        # Distribution of human labels
        human_dist = Counter(human_labels)
        llm_dist   = Counter(llm_labels)

        print(f"\n{dim_name.upper()} RISK (n={n})")
        print(f"  Kappa          : {kappa:.3f}")
        print(f"  Agreement      : {agreement:.1%}")
        print(f"  Human dist     : {dict(human_dist)}")
        print(f"  LLM dist       : {dict(llm_dist)}")

        # Kappa interpretation
        if kappa >= 0.80:
            interp = "Almost perfect"
        elif kappa >= 0.61:
            interp = "Substantial"
        elif kappa >= 0.41:
            interp = "Moderate"
        elif kappa >= 0.21:
            interp = "Fair"
        elif kappa >= 0.0:
            interp = "Slight"
        else:
            interp = "Poor (less than chance)"

        print(f"  Interpretation : {interp}")

        results[dim_name] = {
            "n":              n,
            "kappa":          round(float(kappa), 4),
            "agreement_pct":  round(float(agreement), 4),
            "interpretation": interp,
            "human_dist":     dict(human_dist),
            "llm_dist":       dict(llm_dist)
        }

    # Overall kappa across all dimensions
    all_human = []
    all_llm   = []

    for dim_name, (human_col, llm_col) in DIMENSIONS.items():
        mask = (
            df[human_col].isin(VALID_LABELS) &
            df[llm_col].isin(VALID_LABELS)
        )
        subset = df[mask]
        all_human.extend(subset[human_col].tolist())
        all_llm.extend(subset[llm_col].tolist())

    if len(all_human) >= 10:
        overall_kappa = cohen_kappa_score(all_human, all_llm)
        overall_agreement = sum(
            h == l for h, l in zip(all_human, all_llm)
        ) / len(all_human)

        print(f"\n{'='*60}")
        print(f"OVERALL (all dimensions combined, n={len(all_human)})")
        print(f"  Kappa     : {overall_kappa:.3f}")
        print(f"  Agreement : {overall_agreement:.1%}")
        print("=" * 60)

        results["overall"] = {
            "n":              len(all_human),
            "kappa":          round(float(overall_kappa), 4),
            "agreement_pct":  round(float(overall_agreement), 4)
        }

    # Save results
    output_path = OUTPUT_DIR / "kappa_results.json"
    output_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8"
    )
    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        print("Run: pip install scikit-learn")
        exit(1)

    run_kappa_analysis()