"""
Item 1A Risk Factor Extractor
==============================
Extracts the Risk Factors section (Item 1A) from raw
SEC EDGAR 10-K and 10-Q filings.

Outputs one text file per filing into data/processed/item1a/

Usage:
    python src/preprocessing/extract_item1a.py
"""

import re
import json
import logging
from pathlib import Path
from datetime import datetime

# ── Logging ───────────────────────────────────────────────────────────────────

Path("outputs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("outputs/extraction_log.txt")
    ]
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

RAW_DIR       = Path("sec-edgar-filings")
PROCESSED_DIR = Path("data/processed/item1a")

# Patterns that mark the START of Item 1A
ITEM1A_START_PATTERNS = [
    r"item\s*1a[\.\s]*risk\s*factors",
    r"item\s*1a\s*[\.\-–—]\s*risk\s*factors",
    r"risk\s*factors\s*item\s*1a",
]

# Patterns that mark the END of Item 1A (start of next section)
ITEM1A_END_PATTERNS = [
    r"item\s*1b[\.\s]*unresolved\s*staff\s*comments",
    r"item\s*2[\.\s]*properties",
    r"item\s*2[\.\s]*description\s*of\s*properties",
    r"item\s*1b\s*[\.\-–—]",
    r"item\s*2\s*[\.\-–—]",
]

# Boilerplate phrases to remove
BOILERPLATE_PHRASES = [
    r"table\s*of\s*contents",
    r"edgar\s*filing",
    r"this\s*page\s*intentionally\s*left\s*blank",
    r"forward[\s\-]*looking\s*statements",
]


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean extracted Item 1A text.
    Removes excess whitespace, boilerplate, and formatting artifacts.
    """
    # Remove HTML artifacts if any slipped through
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove unicode artifacts common in EDGAR filings
    text = text.encode("ascii", "ignore").decode("ascii")

    # Remove boilerplate
    for pattern in BOILERPLATE_PHRASES:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ── Item 1A extraction ────────────────────────────────────────────────────────

def extract_item1a(filing_text: str) -> str | None:
    """
    Extract Item 1A text from a full filing document.
    Returns the extracted text or None if not found.
    """
    text_lower = filing_text.lower()

    # Find start position
    start_pos = None
    for pattern in ITEM1A_START_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            start_pos = match.start()
            break

    if start_pos is None:
        return None

    # Find end position (start of next section after Item 1A)
    end_pos = len(filing_text)
    search_text = text_lower[start_pos + 100:]  # skip past the header

    for pattern in ITEM1A_END_PATTERNS:
        match = re.search(pattern, search_text)
        if match:
            candidate_end = start_pos + 100 + match.start()
            if candidate_end < end_pos:
                end_pos = candidate_end

    extracted = filing_text[start_pos:end_pos]

    # Sanity check: Item 1A should be at least 500 characters
    if len(extracted) < 500:
        return None

    return clean_text(extracted)


# ── Filing reader ─────────────────────────────────────────────────────────────

def read_filing(filing_path: Path) -> str | None:
    """
    Read a filing file. Handles both .txt and .htm extensions.
    """
    try:
        # Try UTF-8 first, fall back to latin-1
        try:
            return filing_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return filing_path.read_text(encoding="latin-1")
    except Exception as e:
        logger.warning(f"Could not read {filing_path}: {e}")
        return None


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_extraction_pipeline() -> None:
    """
    Walk through all downloaded filings and extract Item 1A.
    Saves one JSON file per filing with metadata and extracted text.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_DIR.exists():
        logger.error(f"Raw filings directory not found: {RAW_DIR}")
        logger.error("Run edgar_downloader.py first")
        return

    logger.info("=" * 60)
    logger.info("Item 1A Extraction Pipeline")
    logger.info(f"Input  : {RAW_DIR}")
    logger.info(f"Output : {PROCESSED_DIR}")
    logger.info("=" * 60)

    success_count = 0
    fail_count    = 0
    skip_count    = 0
    results       = []

    # Walk through sec-edgar-filings/TICKER/FILING_TYPE/DATE/
    for ticker_dir in sorted(RAW_DIR.iterdir()):
        if not ticker_dir.is_dir():
            continue

        ticker = ticker_dir.name
        logger.info(f"\nProcessing: {ticker}")

        for filing_type_dir in ticker_dir.iterdir():
            if not filing_type_dir.is_dir():
                continue

            filing_type = filing_type_dir.name

            for filing_date_dir in sorted(filing_type_dir.iterdir()):
                if not filing_date_dir.is_dir():
                    continue

                filing_date = filing_date_dir.name

                # Check if already processed
                output_path = PROCESSED_DIR / f"{ticker}_{filing_type}_{filing_date}.json"
                if output_path.exists():
                    skip_count += 1
                    continue

                # Find the actual filing file
                filing_file = None
                for ext in [".txt", ".htm", ".html"]:
                    candidates = list(filing_date_dir.glob(f"*{ext}"))
                    if candidates:
                        # Take the largest file (most likely the full filing)
                        filing_file = max(candidates, key=lambda p: p.stat().st_size)
                        break

                if filing_file is None:
                    logger.warning(f"No filing file found in {filing_date_dir}")
                    fail_count += 1
                    continue

                # Read and extract
                filing_text = read_filing(filing_file)
                if filing_text is None:
                    fail_count += 1
                    continue

                item1a_text = extract_item1a(filing_text)

                if item1a_text is None:
                    logger.warning(f"Item 1A not found: {ticker} {filing_type} {filing_date}")
                    fail_count += 1
                    continue

                # Save as JSON with metadata
                record = {
                    "ticker":       ticker,
                    "filing_type":  filing_type,
                    "filing_date":  filing_date,
                    "char_count":   len(item1a_text),
                    "word_count":   len(item1a_text.split()),
                    "text":         item1a_text,
                    "source_file":  str(filing_file),
                    "extracted_at": datetime.now().isoformat()
                }

                output_path.write_text(
                    json.dumps(record, indent=2, ensure_ascii=False),
                    encoding="utf-8"
                )

                success_count += 1
                logger.info(
                    f"OK  {ticker:8s} {filing_type:5s} {filing_date} "
                    f"| {record['word_count']:,} words"
                )

                results.append({
                    "ticker":       ticker,
                    "filing_type":  filing_type,
                    "filing_date":  filing_date,
                    "word_count":   record["word_count"],
                    "status":       "success"
                })

    # ── Summary ───────────────────────────────────────────────────────────────

    logger.info("\n" + "=" * 60)
    logger.info("EXTRACTION COMPLETE")
    logger.info(f"Successful : {success_count}")
    logger.info(f"Failed     : {fail_count}")
    logger.info(f"Skipped    : {skip_count} (already processed)")
    logger.info("=" * 60)

    # Save extraction summary
    summary_path = Path("outputs/extraction_summary.json")
    summary_path.write_text(
        json.dumps({
            "run_at":        datetime.now().isoformat(),
            "success_count": success_count,
            "fail_count":    fail_count,
            "skip_count":    skip_count,
            "results":       results
        }, indent=2),
        encoding="utf-8"
    )
    logger.info(f"Summary saved: {summary_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    start_time = datetime.now()
    run_extraction_pipeline()
    elapsed = datetime.now() - start_time
    logger.info(f"Total elapsed time: {elapsed}")