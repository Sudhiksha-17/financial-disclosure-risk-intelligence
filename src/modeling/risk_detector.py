"""
LLM Risk Language Shift Detector
==================================
Analyzes consecutive SEC filing pairs to detect shifts in
risk factor language across 5 dimensions using Llama-3 8B
via Ollama.

For each filing pair, produces:
- Direction: escalating, stable, or de-escalating
- Intensity: 1 to 5
- Justification: one sentence explanation

Usage:
    python src/modeling/risk_detector.py
    python src/modeling/risk_detector.py --limit 10  # test run
"""

import json
import time
import logging
import argparse
import requests
from pathlib import Path
from datetime import datetime

# ── Logging ───────────────────────────────────────────────────────────────────

Path("outputs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("outputs/detection_log.txt")
    ]
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

PAIRS_DIR    = Path("data/processed/pairs")
RESULTS_DIR  = Path("data/processed/risk_signals")
OLLAMA_URL   = "http://localhost:11434/api/generate"
MODEL        = "llama3"

# Risk dimensions to analyze
RISK_DIMENSIONS = [
    "liquidity_risk",
    "credit_risk",
    "operational_risk",
    "market_risk",
    "regulatory_risk"
]

# Truncate texts to this many words to fit context window
# Llama-3 8B has 8k context, we use 2k per text to leave room for prompt
MAX_WORDS_PER_TEXT = 2000


# ── Text truncator ────────────────────────────────────────────────────────────

def truncate_text(text: str, max_words: int = MAX_WORDS_PER_TEXT) -> str:
    """
    Truncate text to max_words words.
    Takes the first half and last half to preserve
    both introduction and conclusion of risk section.
    """
    words = text.split()
    if len(words) <= max_words:
        return text

    half = max_words // 2
    first_half = " ".join(words[:half])
    last_half  = " ".join(words[-half:])
    return first_half + " [...] " + last_half


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(
    ticker: str,
    year_earlier: int,
    year_later: int,
    text_earlier: str,
    text_later: str
) -> str:
    """
    Build a structured prompt for risk language shift detection.
    Returns a prompt that instructs Llama-3 to output valid JSON.
    """
    text_e = truncate_text(text_earlier)
    text_l = truncate_text(text_later)

    prompt = f"""You are a financial risk analyst comparing two consecutive annual reports from {ticker}.

EARLIER FILING ({year_earlier} Risk Factors):
{text_e}

LATER FILING ({year_later} Risk Factors):
{text_l}

Analyze how the risk language changed between these two filings across exactly these 5 dimensions:
1. liquidity_risk: Cash, funding, and liquidity concerns
2. credit_risk: Borrower default, counterparty, and credit quality concerns
3. operational_risk: Systems, processes, people, and operational failures
4. market_risk: Market volatility, interest rates, and price risks
5. regulatory_risk: Regulatory changes, compliance, and legal risks

For each dimension output EXACTLY this JSON structure with no additional text:

{{
  "liquidity_risk": {{
    "direction": "escalating" or "stable" or "de-escalating",
    "intensity": 1 to 5 integer where 1=minimal change 5=major change,
    "justification": "one sentence explaining the key language change"
  }},
  "credit_risk": {{
    "direction": "escalating" or "stable" or "de-escalating",
    "intensity": 1 to 5 integer,
    "justification": "one sentence explaining the key language change"
  }},
  "operational_risk": {{
    "direction": "escalating" or "stable" or "de-escalating",
    "intensity": 1 to 5 integer,
    "justification": "one sentence explaining the key language change"
  }},
  "market_risk": {{
    "direction": "escalating" or "stable" or "de-escalating",
    "intensity": 1 to 5 integer,
    "justification": "one sentence explaining the key language change"
  }},
  "regulatory_risk": {{
    "direction": "escalating" or "stable" or "de-escalating",
    "intensity": 1 to 5 integer,
    "justification": "one sentence explaining the key language change"
  }}
}}

Respond with ONLY the JSON object. No preamble, no explanation, no markdown."""

    return prompt


# ── Ollama caller ─────────────────────────────────────────────────────────────

def call_ollama(prompt: str, retries: int = 3) -> str | None:
    """
    Call Ollama API with the given prompt.
    Returns the model response text or None on failure.
    """
    payload = {
        "model":  MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,       # Deterministic output
            "num_predict": 800,     # Max tokens for response
            "top_p": 1,
            "seed": 42
        }
    }

    for attempt in range(retries):
        try:
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json().get("response", "")

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt + 1}/{retries}")
            time.sleep(5)

        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama. Is it running?")
            logger.error("Start with: ollama serve")
            return None

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)

    return None


# ── Response parser ───────────────────────────────────────────────────────────

def parse_response(response_text: str) -> dict | None:
    """
    Parse the JSON response from Llama-3.
    Handles common formatting issues like markdown code blocks.
    """
    if not response_text:
        return None

    # Strip markdown code blocks if present
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text  = "\n".join(lines[1:-1])

    # Find JSON object in response
    start = text.find("{")
    end   = text.rfind("}") + 1

    if start == -1 or end == 0:
        return None

    json_str = text[start:end]

    try:
        parsed = json.loads(json_str)

        # Validate structure
        for dim in RISK_DIMENSIONS:
            if dim not in parsed:
                return None
            dim_data = parsed[dim]
            if not all(
                k in dim_data
                for k in ["direction", "intensity", "justification"]
            ):
                return None
            if dim_data["direction"] not in [
                "escalating", "stable", "de-escalating"
            ]:
                return None
            if not isinstance(dim_data["intensity"], int):
                dim_data["intensity"] = int(dim_data["intensity"])
            if not 1 <= dim_data["intensity"] <= 5:
                return None

        return parsed

    except (json.JSONDecodeError, ValueError, TypeError):
        return None


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_detection_pipeline(limit: int = None) -> None:
    """
    Run LLM risk detection on all filing pairs.
    Skips pairs already processed.

    Args:
        limit: If set, only process this many pairs (for testing)
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pair_files = list(PAIRS_DIR.glob("*.json"))

    if not pair_files:
        logger.error(f"No pairs found in {PAIRS_DIR}")
        logger.error("Run build_pairs.py first")
        return

    # Filter to only 10-K pairs for initial run
    # These are the highest quality and most methodologically important
    pair_files = [
        f for f in pair_files
        if "_10-K_" in f.name
    ]

    if limit:
        pair_files = pair_files[:limit]

    total     = len(pair_files)
    success   = 0
    failed    = 0
    skipped   = 0

    logger.info("=" * 60)
    logger.info("LLM Risk Language Shift Detector")
    logger.info(f"Model      : {MODEL} via Ollama")
    logger.info(f"Pairs      : {total} (10-K annual pairs)")
    logger.info(f"Output dir : {RESULTS_DIR}")
    if limit:
        logger.info(f"LIMIT      : {limit} pairs (test mode)")
    logger.info("=" * 60)

    # Verify Ollama is running
    try:
        requests.get("http://localhost:11434", timeout=5)
    except Exception:
        logger.error("Ollama is not running.")
        logger.error("Start it with: ollama serve")
        return

    start_time = datetime.now()

    for i, pair_file in enumerate(sorted(pair_files), 1):
        pair_id     = pair_file.stem
        output_path = RESULTS_DIR / f"{pair_id}_signals.json"

        # Skip if already processed
        if output_path.exists():
            skipped += 1
            continue

        # Load pair
        try:
            pair = json.loads(
                pair_file.read_text(encoding="utf-8")
            )
        except Exception as e:
            logger.warning(f"Could not load {pair_file.name}: {e}")
            failed += 1
            continue

        ticker       = pair.get("ticker", "")
        year_earlier = pair.get("year_earlier", "")
        year_later   = pair.get("year_later", "")
        text_earlier = pair.get("earlier", {}).get("text", "")
        text_later   = pair.get("later", {}).get("text", "")

        if not text_earlier or not text_later:
            logger.warning(f"Missing text: {pair_id}")
            failed += 1
            continue

        # Log progress
        elapsed = (datetime.now() - start_time).seconds
        rate    = success / max(elapsed, 1) * 60
        eta_min = (total - i) / max(rate, 0.1)

        logger.info(
            f"[{i}/{total}] {ticker} {year_earlier}->{year_later} "
            f"| {success} done | ETA ~{eta_min:.0f}min"
        )

        # Build prompt and call Ollama
        prompt   = build_prompt(
            ticker, year_earlier, year_later,
            text_earlier, text_later
        )
        response = call_ollama(prompt)

        if response is None:
            logger.warning(f"No response for {pair_id}")
            failed += 1
            continue

        # Parse response
        signals = parse_response(response)

        if signals is None:
            logger.warning(
                f"Could not parse response for {pair_id}: "
                f"{response[:200]}"
            )
            failed += 1
            continue

        # Save result
        result = {
            "pair_id":      pair_id,
            "ticker":       ticker,
            "filing_type":  pair.get("filing_type"),
            "year_earlier": year_earlier,
            "year_later":   year_later,
            "signals":      signals,
            "raw_response": response,
            "processed_at": datetime.now().isoformat()
        }

        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        success += 1

    # ── Summary ───────────────────────────────────────────────────────────────

    total_elapsed = datetime.now() - start_time

    logger.info("\n" + "=" * 60)
    logger.info("DETECTION COMPLETE")
    logger.info(f"Successful : {success}")
    logger.info(f"Failed     : {failed}")
    logger.info(f"Skipped    : {skipped} (already processed)")
    logger.info(f"Total time : {total_elapsed}")
    logger.info("=" * 60)

    # Save summary
    summary_path = Path("outputs/detection_summary.json")
    summary_path.write_text(
        json.dumps({
            "run_at":       datetime.now().isoformat(),
            "model":        MODEL,
            "success":      success,
            "failed":       failed,
            "skipped":      skipped,
            "total_time":   str(total_elapsed)
        }, indent=2),
        encoding="utf-8"
    )
    logger.info(f"Summary saved: {summary_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM Risk Language Shift Detector"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of pairs to process (for testing)"
    )
    args = parser.parse_args()

    run_detection_pipeline(limit=args.limit)