"""
Filing Pair Constructor
========================
Takes extracted Item 1A JSON files and constructs consecutive
same-quarter filing pairs for each company.

For each company we pair:
- 10-K 2019 with 10-K 2020 (year-over-year annual pairs)
- 10-K 2020 with 10-K 2021
- etc.

And for 10-Q where available:
- Q1 2019 with Q1 2020
- Q2 2019 with Q2 2020
- etc.

Same-quarter pairing controls for seasonal language patterns.

Output: one JSON file per pair with both texts and metadata.

Usage:
    python src/preprocessing/build_pairs.py
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ── Logging ───────────────────────────────────────────────────────────────────

Path("outputs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("outputs/pairs_log.txt")
    ]
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed/item1a")
PAIRS_DIR     = Path("data/processed/pairs")

# Minimum word count for both texts in a pair
MIN_WORD_COUNT = 1000


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_year_from_accession(accession: str) -> int | None:
    """
    Extract filing year from EDGAR accession number.
    Format: XXXXXXXXXX-YY-XXXXXX where YY is 2-digit year.
    Example: 0000019617-19-000054 -> 2019
    """
    try:
        parts = accession.split("-")
        if len(parts) >= 2:
            year_2digit = int(parts[1])
            return (
                2000 + year_2digit
                if year_2digit < 50
                else 1900 + year_2digit
            )
    except Exception:
        pass
    return None


def get_quarter(month: int) -> str:
    """Return quarter label for a given month number."""
    if month <= 3:
        return "Q1"
    elif month <= 6:
        return "Q2"
    elif month <= 9:
        return "Q3"
    else:
        return "Q4"


# ── Pair builder ──────────────────────────────────────────────────────────────

def build_pairs_for_company(
    ticker: str,
    filings: list[dict]
) -> list[dict]:
    """
    Build consecutive same-quarter filing pairs for a single company.

    Strategy:
    1. Separate filings by type (10-K vs 10-Q)
    2. For 10-K: sort by year, pair consecutive years
    3. For 10-Q: group by quarter, pair consecutive years
       within the same quarter group

    Returns list of pair dicts.
    """
    pairs = []

    annual_filings    = []
    quarterly_filings = []

    for filing in filings:
        filing_type = filing.get("filing_type", "")
        accession   = filing.get("filing_date", "")
        year        = extract_year_from_accession(accession)

        if year is None:
            continue

        if filing_type == "10-K":
            annual_filings.append({
                "year":   year,
                "filing": filing
            })

        elif filing_type == "10-Q":
            try:
                extracted = datetime.fromisoformat(
                    filing.get("extracted_at", "")
                )
                quarter = get_quarter(extracted.month)
            except Exception:
                continue

            quarterly_filings.append({
                "year":    year,
                "quarter": quarter,
                "filing":  filing
            })

    # ── Annual pairs (10-K) ───────────────────────────────────────────────────

    annual_sorted = sorted(annual_filings, key=lambda x: x["year"])

    for i in range(len(annual_sorted) - 1):
        earlier_item = annual_sorted[i]
        later_item   = annual_sorted[i + 1]

        earlier = earlier_item["filing"]
        later   = later_item["filing"]

        if (earlier.get("word_count", 0) < MIN_WORD_COUNT or
                later.get("word_count", 0) < MIN_WORD_COUNT):
            continue

        year_e = earlier_item["year"]
        year_l = later_item["year"]

        pairs.append({
            "pair_id":      f"{ticker}_10-K_{year_e}_{year_l}",
            "ticker":       ticker,
            "filing_type":  "10-K",
            "period":       "annual",
            "year_earlier": year_e,
            "year_later":   year_l,
            "earlier": {
                "filing_date": earlier.get("filing_date"),
                "word_count":  earlier.get("word_count"),
                "text":        earlier.get("text"),
                "source_file": earlier.get("source_file")
            },
            "later": {
                "filing_date": later.get("filing_date"),
                "word_count":  later.get("word_count"),
                "text":        later.get("text"),
                "source_file": later.get("source_file")
            },
            "created_at": datetime.now().isoformat()
        })

    # ── Quarterly pairs (10-Q) ────────────────────────────────────────────────

    by_quarter = defaultdict(list)
    for item in quarterly_filings:
        by_quarter[item["quarter"]].append(item)

    for quarter, items in by_quarter.items():
        items_sorted = sorted(items, key=lambda x: x["year"])

        for i in range(len(items_sorted) - 1):
            earlier_item = items_sorted[i]
            later_item   = items_sorted[i + 1]

            earlier = earlier_item["filing"]
            later   = later_item["filing"]

            if (earlier.get("word_count", 0) < MIN_WORD_COUNT or
                    later.get("word_count", 0) < MIN_WORD_COUNT):
                continue

            year_e = earlier_item["year"]
            year_l = later_item["year"]

            pairs.append({
                "pair_id":      f"{ticker}_10-Q_{quarter}_{year_e}_{year_l}",
                "ticker":       ticker,
                "filing_type":  "10-Q",
                "period":       quarter,
                "year_earlier": year_e,
                "year_later":   year_l,
                "earlier": {
                    "filing_date": earlier.get("filing_date"),
                    "word_count":  earlier.get("word_count"),
                    "text":        earlier.get("text"),
                    "source_file": earlier.get("source_file")
                },
                "later": {
                    "filing_date": later.get("filing_date"),
                    "word_count":  later.get("word_count"),
                    "text":        later.get("text"),
                    "source_file": later.get("source_file")
                },
                "created_at": datetime.now().isoformat()
            })

    return pairs


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pair_construction() -> None:
    """
    Load all extracted Item 1A JSONs, group by company,
    and construct consecutive same-quarter filing pairs.
    Skips pairs that already exist.
    """
    PAIRS_DIR.mkdir(parents=True, exist_ok=True)

    if not PROCESSED_DIR.exists():
        logger.error(f"Processed dir not found: {PROCESSED_DIR}")
        logger.error("Run extract_item1a.py first")
        return

    all_files = list(PROCESSED_DIR.glob("*.json"))

    logger.info("=" * 60)
    logger.info("Filing Pair Constructor")
    logger.info(f"Input files  : {len(all_files)}")
    logger.info(f"Output dir   : {PAIRS_DIR}")
    logger.info(f"Min words    : {MIN_WORD_COUNT}")
    logger.info("=" * 60)

    if not all_files:
        logger.error("No extracted filings found")
        return

    # Load and group all filings by ticker
    filings_by_ticker = defaultdict(list)

    for filepath in all_files:
        try:
            data = json.loads(
                filepath.read_text(encoding="utf-8")
            )
            ticker = data.get("ticker", "")
            if ticker:
                filings_by_ticker[ticker].append(data)
        except Exception as e:
            logger.warning(f"Could not load {filepath.name}: {e}")

    logger.info(f"Companies loaded: {len(filings_by_ticker)}")

    # Build pairs for each company
    total_pairs   = 0
    skip_count    = 0
    company_stats = []

    for ticker, filings in sorted(filings_by_ticker.items()):
        pairs = build_pairs_for_company(ticker, filings)

        company_new  = 0
        company_skip = 0

        for pair in pairs:
            output_path = PAIRS_DIR / f"{pair['pair_id']}.json"

            if output_path.exists():
                company_skip += 1
                skip_count   += 1
                continue

            output_path.write_text(
                json.dumps(pair, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            company_new += 1
            total_pairs += 1

        if company_new > 0 or company_skip > 0:
            logger.info(
                f"OK  {ticker:8s} | "
                f"{len(filings):3d} filings | "
                f"{company_new:3d} new pairs | "
                f"{company_skip:2d} skipped"
            )

        company_stats.append({
            "ticker":       ticker,
            "filing_count": len(filings),
            "new_pairs":    company_new,
            "skipped":      company_skip
        })

    # ── Summary ───────────────────────────────────────────────────────────────

    logger.info("\n" + "=" * 60)
    logger.info("PAIR CONSTRUCTION COMPLETE")
    logger.info(f"New pairs created  : {total_pairs}")
    logger.info(f"Pairs skipped      : {skip_count} (already exist)")
    logger.info(f"Companies processed: {len(filings_by_ticker)}")
    logger.info("=" * 60)

    # Save summary
    summary_path = Path("outputs/pairs_summary.json")
    summary_path.write_text(
        json.dumps({
            "run_at":        datetime.now().isoformat(),
            "total_new":     total_pairs,
            "total_skipped": skip_count,
            "company_count": len(filings_by_ticker),
            "companies":     company_stats
        }, indent=2),
        encoding="utf-8"
    )
    logger.info(f"Summary saved: {summary_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    start_time = datetime.now()
    run_pair_construction()
    elapsed = datetime.now() - start_time
    logger.info(f"Total elapsed time: {elapsed}")