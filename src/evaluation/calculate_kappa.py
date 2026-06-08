"""
Cohen's Kappa Calculator v2
=============================
Calculates Cohen's kappa between human annotations
and CURRENT LLM predictions (from risk_signals folder)
for each risk dimension.

This version reads LLM predictions from the signal files
directly rather than from the annotation sheet columns,
ensuring it uses the latest v2 prompt predictions.

Usage:
    python src/evaluation/calculate_kappa.py
"""

import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
from collections import Counter

SHEET_PATH   = Path("outputs/annotation_sheet.xlsx")
SIGNALS_DIR  = Path("data/processed/risk_signals")
OUTPUT_DIR   = Path("outputs")

DIMENSIONS = {
    "liquidity":   "liquidity_risk",
    "credit":      "credit_risk",
    "operational": "operational_risk",
    "market":      "market_risk",
    "regulatory":  "regulatory_risk",
}

HUMAN_COLS = {
    "liquidity":   "human_liquidity_dir",
    "credit":      "human_credit_dir",
    "operational": "human_operational_dir",
    "market":      "human_market_dir",
    "regulatory":  "human_regulatory_dir",
}

VALID_LABELS = ["escalating", "stable", "de-escalating"]


def load_signal_predictions() -> dict:
    """
    Load latest LLM predictions from risk_signals folder.
    Returns dict: pair_id -> {dim_name -> direction}
    """
    predictions = {}
    for f in SIGNALS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            pair_id = data.get("pair_id", "")
            signals = data.get("signals", {})
            predictions[pair_id] = {
                dim: signals.get(dim, {}).get("direction", "")
                for dim in DIMENSIONS.values()
            }
        except Exception:
            continue
    return predictions


def run_kappa_analysis():
    # Load human annotations
    df = pd.read_excel(SHEET_PATH)
    print("=" * 60)
    print("COHEN'S KAPPA ANALYSIS v2")
    print("Using current LLM predictions from risk_signals folder")
    print(f"Total annotation rows: {len(df)}")
    print("=" * 60)

    # Load latest LLM predictions
    predictions = load_signal_predictions()
    print(f"Signal files loaded: {len(predictions)}")

    results = {}

    for dim_name, signal_key in DIMENSIONS.items():
        human_col = HUMAN_COLS[dim_name]

        human_labels = []
        llm_labels   = []

        for _, row in df.iterrows():
            pair_id   = str(row.get("pair_id", "")).strip()
            human_val = str(row.get(human_col, "")).strip()

            if human_val not in VALID_LABELS:
                continue

            # Get latest LLM prediction for this pair
            pair_signals = predictions.get(pair_id, {})
            # Try with _signals suffix too
            if not pair_signals:
                pair_signals = predictions.get(
                    f"{pair_id}_signals", {}
                )

            llm_val = pair_signals.get(signal_key, "")

            if llm_val not in VALID_LABELS:
                continue

            human_labels.append(human_val)
            llm_labels.append(llm_val)

        n = len(human_labels)
        if n < 5:
            print(f"\n{dim_name:15s}: insufficient data (n={n})")
            results[dim_name] = {"n": n, "kappa": None}
            continue

        kappa     = cohen_kappa_score(human_labels, llm_labels)
        agreement = sum(
            h == l for h, l in zip(human_labels, llm_labels)
        ) / n

        human_dist = Counter(human_labels)
        llm_dist   = Counter(llm_labels)

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
            interp = "Poor"

        print(f"\n{dim_name.upper()} RISK (n={n})")
        print(f"  Kappa          : {kappa:.3f}")
        print(f"  Agreement      : {agreement:.1%}")
        print(f"  Human dist     : {dict(human_dist)}")
        print(f"  LLM dist       : {dict(llm_dist)}")
        print(f"  Interpretation : {interp}")

        results[dim_name] = {
            "n":              n,
            "kappa":          round(float(kappa), 4),
            "agreement_pct":  round(float(agreement), 4),
            "interpretation": interp,
            "human_dist":     dict(human_dist),
            "llm_dist":       dict(llm_dist)
        }

    # Overall kappa
    all_human = []
    all_llm   = []

    for dim_name, signal_key in DIMENSIONS.items():
        human_col = HUMAN_COLS[dim_name]

        for _, row in df.iterrows():
            pair_id   = str(row.get("pair_id", "")).strip()
            human_val = str(row.get(human_col, "")).strip()

            if human_val not in VALID_LABELS:
                continue

            pair_signals = predictions.get(pair_id, {})
            if not pair_signals:
                pair_signals = predictions.get(
                    f"{pair_id}_signals", {}
                )

            llm_val = pair_signals.get(signal_key, "")

            if llm_val not in VALID_LABELS:
                continue

            all_human.append(human_val)
            all_llm.append(llm_val)

    if len(all_human) >= 10:
        overall_kappa = cohen_kappa_score(all_human, all_llm)
        overall_agreement = sum(
            h == l for h, l in zip(all_human, all_llm)
        ) / len(all_human)

        print(f"\n{'='*60}")
        print(f"OVERALL (n={len(all_human)})")
        print(f"  Kappa     : {overall_kappa:.3f}")
        print(f"  Agreement : {overall_agreement:.1%}")
        print("=" * 60)

        results["overall"] = {
            "n":             len(all_human),
            "kappa":         round(float(overall_kappa), 4),
            "agreement_pct": round(float(overall_agreement), 4)
        }

    output_path = OUTPUT_DIR / "kappa_results_v2.json"
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