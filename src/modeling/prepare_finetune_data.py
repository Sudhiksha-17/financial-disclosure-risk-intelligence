"""
Fine-tuning Data Preparation
==============================
Converts human annotations into instruction-response training
examples for LoRA fine-tuning.

Strategy: One training example per valid dimension per pair.
A pair with 3 annotatable dimensions generates 3 examples.
This maximizes use of partial annotations.

Each example asks the model to classify ONE dimension only,
which is a simpler and more learnable task than classifying
all five simultaneously.

Output:
    outputs/finetune_train.jsonl
    outputs/finetune_val.jsonl

Usage:
    python src/modeling/prepare_finetune_data.py
"""

import json
import random
import pandas as pd
from pathlib import Path
from collections import defaultdict

random.seed(42)

SHEET_PATH  = Path("outputs/annotation_sheet.xlsx")
PAIRS_DIR   = Path("data/processed/pairs")
OUTPUT_DIR  = Path("outputs")

DIMENSIONS = {
    "liquidity_risk": {
        "human_dir": "human_liquidity_dir",
        "human_int": "human_liquidity_int",
        "description": "Cash, funding, and liquidity concerns"
    },
    "credit_risk": {
        "human_dir": "human_credit_dir",
        "human_int": "human_credit_int",
        "description": "Borrower default, counterparty, and credit quality"
    },
    "operational_risk": {
        "human_dir": "human_operational_dir",
        "human_int": "human_operational_int",
        "description": "Systems, processes, people, and operational failures"
    },
    "market_risk": {
        "human_dir": "human_market_dir",
        "human_int": "human_market_int",
        "description": "Market volatility, interest rates, and price risks"
    },
    "regulatory_risk": {
        "human_dir": "human_regulatory_dir",
        "human_int": "human_regulatory_int",
        "description": "Regulatory changes, compliance, and legal risks"
    },
}

VALID_LABELS = ["escalating", "stable", "de-escalating"]
MAX_WORDS    = 1200


def truncate_text(text: str, max_words: int = MAX_WORDS) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    half = max_words // 2
    return " ".join(words[:half]) + " [...] " + " ".join(words[-half:])


def build_single_dim_prompt(
    ticker, year_earlier, year_later,
    text_e, text_l, dim_name, dim_desc
):
    """
    Build a prompt that asks the model to classify
    ONE dimension only. Simpler and more learnable
    than classifying all five simultaneously.
    """
    return f"""You are a financial risk analyst comparing two consecutive annual reports from {ticker}.

EARLIER FILING ({year_earlier} Risk Factors):
{text_e}

LATER FILING ({year_later} Risk Factors):
{text_l}

Analyze how the {dim_name.replace('_', ' ')} language changed between these two filings.
{dim_name.replace('_', ' ').title()}: {dim_desc}

Output EXACTLY this JSON with no additional text:

{{
  "{dim_name}": {{
    "direction": "escalating" or "stable" or "de-escalating",
    "intensity": 1 to 5 integer where 1=minimal change 5=major change,
    "justification": "max 15 words explaining the key change"
  }}
}}

Respond with ONLY the JSON object. No preamble, no explanation, no markdown."""


def build_single_dim_response(dim_name, direction, intensity):
    return json.dumps({
        dim_name: {
            "direction":     direction,
            "intensity":     intensity,
            "justification": f"Human annotated as {direction} intensity {intensity}"
        }
    }, indent=2)


def prepare_examples():
    df = pd.read_excel(SHEET_PATH)
    print(f"Annotation rows loaded: {len(df)}")

    examples = []

    for _, row in df.iterrows():
        pair_id = str(row.get("pair_id", "")).strip()

        # Load pair text once per row
        pair_file = PAIRS_DIR / f"{pair_id}.json"
        if not pair_file.exists():
            continue

        try:
            pair = json.loads(
                pair_file.read_text(encoding="utf-8")
            )
        except Exception:
            continue

        ticker       = pair.get("ticker", "")
        year_earlier = pair.get("year_earlier", "")
        year_later   = pair.get("year_later", "")
        text_earlier = pair.get("earlier", {}).get("text", "")
        text_later   = pair.get("later", {}).get("text", "")

        if not text_earlier or not text_later:
            continue

        text_e = truncate_text(text_earlier)
        text_l = truncate_text(text_later)

        # Create one example per valid dimension
        for dim_name, cols in DIMENSIONS.items():
            direction = str(row.get(cols["human_dir"], "")).strip()
            intensity = row.get(cols["human_int"], 1)

            if direction not in VALID_LABELS:
                continue

            try:
                intensity = int(intensity)
                if not 1 <= intensity <= 5:
                    intensity = 1
            except (ValueError, TypeError):
                intensity = 1

            prompt = build_single_dim_prompt(
                ticker, year_earlier, year_later,
                text_e, text_l, dim_name, cols["description"]
            )
            response = build_single_dim_response(
                dim_name, direction, intensity
            )

            examples.append({
                "pair_id":   pair_id,
                "ticker":    ticker,
                "dim":       dim_name,
                "direction": direction,
                "prompt":    prompt,
                "response":  response,
            })

    print(f"Valid examples: {len(examples)}")

    # Show distribution
    dir_counts = defaultdict(int)
    dim_counts = defaultdict(int)
    for ex in examples:
        dir_counts[ex["direction"]] += 1
        dim_counts[ex["dim"]] += 1

    print("\nDirection distribution:")
    for k, v in sorted(dir_counts.items()):
        print(f"  {k}: {v}")

    print("\nDimension distribution:")
    for k, v in sorted(dim_counts.items()):
        print(f"  {k}: {v}")

    return examples


def stratified_split(examples, val_ratio=0.2):
    """
    Split stratified by direction label to ensure
    both train and val have escalating, stable, and
    de-escalating examples.
    """
    by_direction = defaultdict(list)
    for ex in examples:
        by_direction[ex["direction"]].append(ex)

    train_examples = []
    val_examples   = []

    for direction, items in by_direction.items():
        random.shuffle(items)
        n_val = max(1, round(len(items) * val_ratio))
        val_examples.extend(items[:n_val])
        train_examples.extend(items[n_val:])

    random.shuffle(train_examples)
    random.shuffle(val_examples)

    print(f"\nTrain: {len(train_examples)} examples")
    print(f"Val:   {len(val_examples)} examples")

    # Show val direction distribution
    val_dirs = defaultdict(int)
    for ex in val_examples:
        val_dirs[ex["direction"]] += 1
    print(f"Val distribution: {dict(val_dirs)}")

    return train_examples, val_examples


def save_jsonl(examples, path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({
                "instruction": ex["prompt"],
                "output":      ex["response"],
                "pair_id":     ex["pair_id"],
                "dim":         ex["dim"],
                "direction":   ex["direction"]
            }) + "\n")
    print(f"Saved: {path} ({len(examples)} examples)")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    examples = prepare_examples()

    if len(examples) < 10:
        print(f"Too few examples: {len(examples)}")
        return

    train, val = stratified_split(examples, val_ratio=0.2)

    save_jsonl(train, OUTPUT_DIR / "finetune_train.jsonl")
    save_jsonl(val,   OUTPUT_DIR / "finetune_val.jsonl")

    print("\nData preparation complete.")
    print("Next: run src/modeling/finetune_lora.py")


if __name__ == "__main__":
    main()