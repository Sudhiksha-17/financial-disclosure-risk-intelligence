"""
SEC EDGAR Filing Downloader
============================
Downloads 10-K and 10-Q filings for target companies
across 4 sectors: banking, insurance, technology, energy.

Version: compatible with sec-edgar-downloader 5.x
Dataset: 100 companies, 25 per sector, 2019-2024

Usage:
    # Run all sectors
    python src/ingestion/edgar_downloader.py

    # Run a single sector
    python src/ingestion/edgar_downloader.py banking
    python src/ingestion/edgar_downloader.py insurance
    python src/ingestion/edgar_downloader.py technology
    python src/ingestion/edgar_downloader.py energy
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from sec_edgar_downloader import Downloader

# ── Logging setup ─────────────────────────────────────────────────────────────

Path("outputs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("outputs/download_log.txt")
    ]
)
logger = logging.getLogger(__name__)

# ── Company universe: 100 companies across 4 sectors ─────────────────────────

COMPANIES = {
    "banking": [
    "JPM",  "BAC",  "WFC",  "GS",   "MS",
    "C",    "USB",  "PNC",  "TFC",  "COF",
    "BK",   "STT",  "FITB", "RF",   "HBAN",
    "ALLY", "CFG",  "NTRS", "ZION", "BOKF",
    "FHN",  "IBOC", "WAL",  "KEY",  "MTB"
    ],
    "insurance": [
        "MET",  "PRU",  "AFL",  "TRV",  "ALL",
        "CB",   "AIG",  "HIG",  "LNC",  "UNM",
        "PGR",  "CNA",  "RLI",  "CINF", "AIZ",
        "GL",   "EQH",  "BHF",  "ERIE", "KMPR",
        "THG",  "WRB",  "RNR",  "ACGL", "MLK"
    ],
    "technology": [
        "AAPL", "MSFT", "GOOGL","META", "AMZN",
        "NVDA", "INTC", "IBM",  "ORCL", "CRM",
        "ADBE", "NOW",  "SNOW", "PLTR", "PANW",
        "CRWD", "ZS",   "OKTA", "DDOG", "NET",
        "TWLO", "HUBS", "ESTC", "CFLT", "MDB"
    ],
    "energy": [
        "XOM",  "CVX",  "COP",  "EOG",  "SLB",
        "PSX",  "VLO",  "MPC",  "OXY",  "HAL",
        "PXD",  "DVN",  "FANG", "MRO",  "APA",
        "HES",  "BKR",  "NOV",  "HP",   "WHD",
        "MTDR", "SM",   "CIVI", "CPE",  "PR"
    ]
}

# ── Config ────────────────────────────────────────────────────────────────────

START_DATE   = "2019-01-01"
END_DATE     = "2024-12-31"
FILING_TYPES = ["10-K", "10-Q"]

# SEC fair access policy: stay well below 10 requests per second
SLEEP_BETWEEN_TICKERS = 2.0  # seconds


# ── Core download function ────────────────────────────────────────────────────

def download_filings(
    dl: Downloader,
    ticker: str,
    sector: str
) -> dict:
    """
    Download 10-K and 10-Q filings for a single ticker.
    Returns a summary dict of results.
    """
    results = {
        "ticker":  ticker,
        "sector":  sector,
        "status":  {}
    }

    for filing_type in FILING_TYPES:
        try:
            dl.get(
                filing_type,
                ticker,
                after=START_DATE,
                before=END_DATE,
                download_details=True
            )
            results["status"][filing_type] = "success"
            logger.info(f"OK   {ticker:8s} {filing_type}")

        except Exception as e:
            results["status"][filing_type] = f"failed: {str(e)}"
            logger.warning(f"FAIL {ticker:8s} {filing_type} | {e}")

    return results


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_download_pipeline(companies: dict = None) -> None:
    """
    Download all filings for all companies in the given dict.
    Defaults to the full COMPANIES universe if none provided.
    Logs progress and writes a summary report on completion.
    """
    if companies is None:
        companies = COMPANIES

    logger.info("=" * 60)
    logger.info("SEC EDGAR Filing Download Pipeline")
    logger.info(f"Period    : {START_DATE} to {END_DATE}")
    logger.info(f"Sectors   : {list(companies.keys())}")
    logger.info(f"Tickers   : {sum(len(v) for v in companies.values())} total")
    logger.info(f"Save path : ./sec-edgar-filings/")
    logger.info("=" * 60)

    # Initialise downloader
    dl = Downloader(
        company_name="Sudhiksha Kandavel Rajan",
        email_address="kandavelrajan.s@northeastern.edu"
    )

    all_results = []
    total = sum(len(v) for v in companies.values())
    count = 0

    for sector, tickers in companies.items():
        logger.info(f"\n── Sector: {sector.upper()} ──")

        for ticker in tickers:
            count += 1
            logger.info(f"[{count}/{total}] Downloading {ticker}")

            result = download_filings(dl, ticker, sector)
            all_results.append(result)

            # Respect EDGAR rate limits
            time.sleep(SLEEP_BETWEEN_TICKERS)

    # ── Summary ───────────────────────────────────────────────────────────────

    success_count = sum(
        1 for r in all_results
        for status in r["status"].values()
        if status == "success"
    )
    fail_count = sum(
        1 for r in all_results
        for status in r["status"].values()
        if status != "success"
    )

    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info(f"Successful : {success_count}")
    logger.info(f"Failed     : {fail_count}")
    logger.info(f"Log saved  : outputs/download_log.txt")
    logger.info("=" * 60)

    # Write failed tickers to retry file
    failed = [
        f"{r['ticker']} ({r['sector']}): {r['status']}"
        for r in all_results
        if any(s != "success" for s in r["status"].values())
    ]

    if failed:
        retry_path = Path("outputs/failed_tickers.txt")
        retry_path.write_text("\n".join(failed))
        logger.info(f"Failed tickers saved to: {retry_path}")
        logger.info("Re-run with the same sector argument to retry")
    else:
        logger.info("All tickers downloaded successfully")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Optional: pass sector name as argument for batch processing
    # Example: python src/ingestion/edgar_downloader.py banking
    # Default: runs all sectors

    sector_filter = sys.argv[1] if len(sys.argv) > 1 else None

    if sector_filter and sector_filter not in COMPANIES:
        logger.error(f"Unknown sector: {sector_filter}")
        logger.error(f"Available sectors: {list(COMPANIES.keys())}")
        sys.exit(1)

    if sector_filter:
        logger.info(f"Running single sector: {sector_filter}")
        companies_to_run = {sector_filter: COMPANIES[sector_filter]}
    else:
        logger.info("Running all sectors")
        companies_to_run = COMPANIES

    start_time = datetime.now()
    run_download_pipeline(companies_to_run)
    elapsed = datetime.now() - start_time
    logger.info(f"Total elapsed time: {elapsed}")