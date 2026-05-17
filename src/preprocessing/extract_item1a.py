"""
Item 1A Risk Factor Extractor
==============================
Extracts the Risk Factors section (Item 1A) from raw
SEC EDGAR 10-K and 10-Q filings.

Handles multiple document structures:
1. Standard HTML: primary-document.html with readable content
2. XBRL primary: primary-document.html is XBRL data,
   narrative is in full-submission.txt
3. Incorporation by reference: risk factors are in a separate
   annual report document (documented as known gap)
4. TOC + Content: Item 1A appears in table of contents first

Version: v4 - XBRL detection, full-submission.txt fallback

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

# Minimum word count for a valid Item 1A extraction
# Real Item 1A sections are never shorter than this
MIN_WORD_COUNT = 1000

# Characters to look ahead when detecting TOC entries
# If Item 1B appears within this window it is a TOC entry
TOC_DETECTION_WINDOW = 400

# Patterns that mark the START of Item 1A
ITEM1A_START_PATTERNS = [
    r"item\s*1a[\.\s]*risk\s*factors",
    r"item\s*1a\s*[\.\-\u2013\u2014]\s*risk\s*factors",
    r"risk\s*factors\s*item\s*1a",
    r"item\s+1a\s*\n",
    r"item\s+1a\.",
]

# Patterns that mark the END of Item 1A
ITEM1A_END_PATTERNS = [
    r"item\s*1b[\.\s]*unresolved\s*staff\s*comments",
    r"item\s*1b\s*[\.\-\u2013\u2014]",
    r"item\s*2[\.\s]*properties",
    r"item\s*2[\.\s]*description\s*of\s*properties",
    r"item\s*2\s*[\.\-\u2013\u2014]",
]

# Pattern indicating a TOC entry
TOC_NEARBY_PATTERNS = r"item\s*1b|item\s*2[\.\s]"

# Boilerplate phrases to remove
BOILERPLATE_PHRASES = [
    r"table\s*of\s*contents",
    r"edgar\s*filing",
    r"this\s*page\s*intentionally\s*left\s*blank",
]


# ── XBRL detector ─────────────────────────────────────────────────────────────

def is_xbrl_file(filepath: Path) -> bool:
    """
    Detect if a file is an XBRL data file rather than
    a human-readable narrative document.

    XBRL files are characterized by:
    - Starting with <?xml declaration
    - Containing xmlns: namespace declarations
    - Being created by tools like Workiva/Wdesk

    These files contain structured financial data, not
    the narrative text we need for Item 1A extraction.
    """
    try:
        header = filepath.read_text(
            encoding="utf-8",
            errors="ignore"
        )[:500]

        return (
            header.strip().startswith("<?xml") or
            "xmlns:" in header[:200]
        )
    except Exception:
        return False


# ── Filing file selector ──────────────────────────────────────────────────────

def select_filing_file(filing_date_dir: Path) -> Path | None:
    """
    Select the best file to extract text from in a filing directory.

    Priority order:
    1. primary-document.html if NOT an XBRL file
    2. full-submission.txt as fallback when primary is XBRL
       (used by MS, C and similar companies)
    3. Any other .html file
    4. Any .htm file
    5. Any remaining .txt file

    Known limitation:
    Companies using incorporation by reference (WFC, USB) store
    their risk factors in separate annual report documents that
    are not downloaded by this pipeline. These are documented
    as known gaps.
    """

    # Priority 1: primary-document.html if not XBRL
    primary = filing_date_dir / "primary-document.html"
    if primary.exists():
        if not is_xbrl_file(primary):
            return primary
        else:
            logger.debug(
                f"primary-document.html is XBRL in "
                f"{filing_date_dir.name}, trying fallback"
            )

    # Priority 2: full-submission.txt fallback for XBRL primary
    # Contains full filing narrative for MS, C style filings
    full_sub = filing_date_dir / "full-submission.txt"
    if full_sub.exists():
        return full_sub

    # Priority 3: any other html file
    html_candidates = [
        f for f in filing_date_dir.glob("*.html")
        if f.name != "primary-document.html"
    ]
    if html_candidates:
        return max(html_candidates, key=lambda p: p.stat().st_size)

    # Priority 4: htm files
    htm_candidates = list(filing_date_dir.glob("*.htm"))
    if htm_candidates:
        return max(htm_candidates, key=lambda p: p.stat().st_size)

    # Priority 5: any remaining txt file
    txt_candidates = [
        f for f in filing_date_dir.glob("*.txt")
        if f.name != "full-submission.txt"
    ]
    if txt_candidates:
        return max(txt_candidates, key=lambda p: p.stat().st_size)

    return None


# ── HTML stripper ─────────────────────────────────────────────────────────────

def strip_html(raw: str) -> str:
    """
    Remove HTML tags and decode common HTML entities.
    """
    # Remove script and style blocks entirely
    raw = re.sub(
        r"<(script|style)[^>]*>.*?</(script|style)>",
        " ", raw, flags=re.DOTALL | re.IGNORECASE
    )

    # Remove all remaining HTML tags
    raw = re.sub(r"<[^>]+>", " ", raw)

    # Decode common HTML entities
    replacements = {
        "&amp;":  "&",
        "&lt;":   "<",
        "&gt;":   ">",
        "&nbsp;": " ",
        "&#160;": " ",
        "&quot;": '"',
        "&#8212;": "-",
        "&#8211;": "-",
        "&#8217;": "'",
        "&#8216;": "'",
        "&#8220;": '"',
        "&#8221;": '"',
        "&#8230;": "...",
        "&#147;":  '"',
        "&#148;":  '"',
        "&#32;":   " ",
    }
    for entity, replacement in replacements.items():
        raw = raw.replace(entity, replacement)

    return raw


# ── Text cleaner ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean extracted Item 1A text.
    Removes excess whitespace, boilerplate, and encoding artifacts.
    """
    # Remove non-ASCII characters
    text = text.encode("ascii", "ignore").decode("ascii")

    # Remove boilerplate phrases
    for pattern in BOILERPLATE_PHRASES:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ── Filing reader ─────────────────────────────────────────────────────────────

def read_filing(filing_path: Path) -> str | None:
    """
    Read a filing file and return plain text.

    Handles three document types:
    1. Standard HTML files: strip tags directly
    2. full-submission.txt: extract first <DOCUMENT> section
       which contains the main 10-K/10-Q narrative
    3. Plain text files: return as-is after encoding fix
    """
    try:
        try:
            raw = filing_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw = filing_path.read_text(encoding="latin-1")

        # For full-submission.txt, extract the first DOCUMENT
        # section which contains the main filing narrative.
        # The file concatenates multiple documents including
        # exhibits, so we only want the primary document.
        if filing_path.name == "full-submission.txt":
            doc_match = re.search(
                r"<DOCUMENT>(.*?)</DOCUMENT>",
                raw,
                flags=re.DOTALL
            )
            if doc_match:
                raw = doc_match.group(1)
            # Strip HTML from the extracted section
            raw = strip_html(raw)
            return raw

        # For HTML files strip tags
        if filing_path.suffix.lower() in [".html", ".htm"]:
            raw = strip_html(raw)
            return raw

        # For plain text check if it contains HTML and strip if so
        if "<html" in raw[:1000].lower() or "<HTML" in raw[:1000]:
            raw = strip_html(raw)

        return raw

    except Exception as e:
        logger.warning(f"Could not read {filing_path}: {e}")
        return None


# ── TOC detector ──────────────────────────────────────────────────────────────

def is_toc_entry(text_lower: str, position: int) -> bool:
    """
    Determine whether an Item 1A match is a table of contents
    entry rather than the actual section content.

    A TOC entry is characterized by Item 1B or Item 2 appearing
    within TOC_DETECTION_WINDOW characters of the match.
    """
    window = text_lower[
        position + 20: position + TOC_DETECTION_WINDOW
    ]
    return bool(re.search(TOC_NEARBY_PATTERNS, window))


# ── Item 1A extractor ─────────────────────────────────────────────────────────

def extract_item1a(filing_text: str) -> str | None:
    """
    Extract Item 1A text from a full filing document.

    Strategy:
    1. Find all positions matching Item 1A start patterns
    2. Skip positions that look like TOC entries
    3. Use the first non-TOC position as the real section start
    4. Find the end using Item 1B / Item 2 patterns
    5. Apply minimum word count filter to reject false positives

    Returns extracted text or None if extraction fails.
    """
    text_lower = filing_text.lower()

    # Find all Item 1A match positions
    start_positions = []
    for pattern in ITEM1A_START_PATTERNS:
        for match in re.finditer(pattern, text_lower):
            start_positions.append(match.start())

    if not start_positions:
        return None

    # Deduplicate and sort ascending
    start_positions = sorted(set(start_positions))

    # Find best start by skipping TOC entries
    best_start = None

    for pos in start_positions:
        if not is_toc_entry(text_lower, pos):
            best_start = pos
            break

    # If all positions looked like TOC entries fall back to last
    if best_start is None:
        best_start = start_positions[-1]

    # Find end of section
    end_pos = len(filing_text)
    search_text = text_lower[best_start + 100:]

    for pattern in ITEM1A_END_PATTERNS:
        match = re.search(pattern, search_text)
        if match:
            candidate_end = best_start + 100 + match.start()
            if candidate_end < end_pos:
                end_pos = candidate_end

    extracted = filing_text[best_start:end_pos]

    # Apply minimum word count filter
    word_count = len(extracted.split())
    if word_count < MIN_WORD_COUNT:
        return None

    return clean_text(extracted)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_extraction_pipeline() -> None:
    """
    Walk through all downloaded filings and extract Item 1A.
    Saves one JSON file per filing with metadata and extracted text.
    Skips filings already processed.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_DIR.exists():
        logger.error(f"Raw filings directory not found: {RAW_DIR}")
        logger.error("Run edgar_downloader.py first")
        return

    logger.info("=" * 60)
    logger.info("Item 1A Extraction Pipeline")
    logger.info(f"Input     : {RAW_DIR}")
    logger.info(f"Output    : {PROCESSED_DIR}")
    logger.info(f"Min words : {MIN_WORD_COUNT}")
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
        logger.info(f"Processing: {ticker}")

        for filing_type_dir in ticker_dir.iterdir():
            if not filing_type_dir.is_dir():
                continue

            filing_type = filing_type_dir.name

            for filing_date_dir in sorted(filing_type_dir.iterdir()):
                if not filing_date_dir.is_dir():
                    continue

                filing_date = filing_date_dir.name

                # Skip if already processed
                output_path = (
                    PROCESSED_DIR /
                    f"{ticker}_{filing_type}_{filing_date}.json"
                )
                if output_path.exists():
                    skip_count += 1
                    continue

                # Select best file
                filing_file = select_filing_file(filing_date_dir)
                if filing_file is None:
                    logger.warning(
                        f"No suitable file: "
                        f"{ticker} {filing_type} {filing_date}"
                    )
                    fail_count += 1
                    continue

                # Read file
                filing_text = read_filing(filing_file)
                if filing_text is None:
                    fail_count += 1
                    continue

                # Extract Item 1A
                item1a_text = extract_item1a(filing_text)

                if item1a_text is None:
                    logger.warning(
                        f"Item 1A not found: "
                        f"{ticker} {filing_type} {filing_date} "
                        f"[{filing_file.name}]"
                    )
                    fail_count += 1
                    continue

                # Save JSON with metadata
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
                    "ticker":      ticker,
                    "filing_type": filing_type,
                    "filing_date": filing_date,
                    "word_count":  record["word_count"],
                    "status":      "success"
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