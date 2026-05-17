"""
Financial Validation Layer
===========================
Validates LLM-derived risk signals against post-filing
stock market reactions using panel OLS regression.

Methodology:
- Calculate 30-day cumulative abnormal returns (CAR) for each
  filing using S&P 500 as market benchmark
- Merge risk signals with CAR data
- Run panel OLS regression with sector and year fixed effects
- Report coefficients, standard errors, and p-values

Usage:
    python src/evaluation/financial_validation.py
"""

import json
import logging
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.formula.api as smf
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────────────

Path("outputs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("outputs/validation_log.txt")
    ]
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

SIGNALS_DIR = Path("data/processed/risk_signals")
PAIRS_DIR   = Path("data/processed/pairs")
OUTPUT_DIR  = Path("outputs")

CAR_WINDOW  = 30

SECTOR_MAP = {
    "JPM": "banking", "BAC": "banking", "WFC": "banking",
    "GS": "banking",  "MS": "banking",  "C": "banking",
    "USB": "banking",  "PNC": "banking", "TFC": "banking",
    "COF": "banking",  "BK": "banking",  "STT": "banking",
    "FITB": "banking", "RF": "banking",  "HBAN": "banking",
    "ALLY": "banking", "CFG": "banking", "NTRS": "banking",
    "ZION": "banking", "BOKF": "banking","FHN": "banking",
    "IBOC": "banking", "WAL": "banking", "KEY": "banking",
    "MTB": "banking",
    "MET": "insurance", "PRU": "insurance", "AFL": "insurance",
    "TRV": "insurance", "ALL": "insurance", "CB": "insurance",
    "AIG": "insurance", "HIG": "insurance", "LNC": "insurance",
    "UNM": "insurance", "PGR": "insurance", "CNA": "insurance",
    "RLI": "insurance", "CINF": "insurance","AIZ": "insurance",
    "GL": "insurance",  "EQH": "insurance", "BHF": "insurance",
    "ERIE": "insurance","KMPR": "insurance","THG": "insurance",
    "WRB": "insurance", "RNR": "insurance", "ACGL": "insurance",
    "MKL": "insurance",
    "AAPL": "technology","MSFT": "technology","GOOGL": "technology",
    "META": "technology","AMZN": "technology","NVDA": "technology",
    "INTC": "technology","IBM": "technology", "ORCL": "technology",
    "CRM": "technology", "ADBE": "technology","NOW": "technology",
    "SNOW": "technology","PLTR": "technology","PANW": "technology",
    "CRWD": "technology","ZS": "technology",  "OKTA": "technology",
    "DDOG": "technology","NET": "technology", "TWLO": "technology",
    "HUBS": "technology","ESTC": "technology","MDB": "technology",
    "XOM": "energy", "CVX": "energy", "COP": "energy",
    "EOG": "energy", "SLB": "energy", "PSX": "energy",
    "VLO": "energy", "MPC": "energy", "OXY": "energy",
    "HAL": "energy", "DVN": "energy", "FANG": "energy",
    "APA": "energy", "BKR": "energy", "NOV": "energy",
    "HP": "energy",  "WHD": "energy", "MTDR": "energy",
    "SM": "energy",  "PR": "energy",
}


# ── Filing date extractor ─────────────────────────────────────────────────────

def get_filing_date(pair_id: str, pairs_dir: Path) -> datetime | None:
    """
    Get the actual filing date from the pair JSON.
    Uses the later filing date as the event date.
    """
    pair_file = pairs_dir / f"{pair_id}.json"
    if not pair_file.exists():
        return None

    try:
        pair = json.loads(pair_file.read_text(encoding="utf-8"))
        accession = pair.get("later", {}).get("filing_date", "")

        parts = accession.split("-")
        if len(parts) >= 2:
            year_2digit = int(parts[1])
            year = 2000 + year_2digit if year_2digit < 50 else 1900 + year_2digit
            return datetime(year, 3, 1)
    except Exception:
        pass

    return None


# ── Abnormal return calculator ────────────────────────────────────────────────

def calculate_car(
    ticker: str,
    event_date: datetime,
    window_days: int = CAR_WINDOW
) -> float | None:
    """
    Calculate Cumulative Abnormal Return (CAR) for a ticker
    over window_days after event_date.

    Abnormal return = Stock return - Market return (S&P 500)
    CAR = Sum of daily abnormal returns over window
    """
    try:
        start = event_date
        end   = event_date + timedelta(days=window_days + 10)

        stock_data  = yf.download(
            ticker, start=start, end=end,
            progress=False, auto_adjust=True
        )
        market_data = yf.download(
            "SPY", start=start, end=end,
            progress=False, auto_adjust=True
        )

        if stock_data.empty or market_data.empty:
            return None

        stock_returns  = stock_data["Close"].pct_change().dropna()
        market_returns = market_data["Close"].pct_change().dropna()

        common_dates = stock_returns.index.intersection(
            market_returns.index
        )

        if len(common_dates) < 10:
            return None

        common_dates = common_dates[:window_days]
        stock_ret    = stock_returns.loc[common_dates]
        market_ret   = market_returns.loc[common_dates]

        abnormal = stock_ret.values - market_ret.values
        car      = float(np.sum(abnormal))

        return car

    except Exception as e:
        logger.warning(f"Could not calculate CAR for {ticker}: {e}")
        return None


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_regression_dataset() -> pd.DataFrame:
    """
    Build panel dataset combining risk signals with
    post-filing abnormal returns.
    """
    signal_files = list(SIGNALS_DIR.glob("*.json"))
    logger.info(f"Loading {len(signal_files)} signal files")

    records   = []
    car_cache = {}

    for i, signal_file in enumerate(sorted(signal_files), 1):

        try:
            signal = json.loads(
                signal_file.read_text(encoding="utf-8")
            )
        except Exception as e:
            logger.warning(f"Could not load {signal_file.name}: {e}")
            continue

        ticker     = signal.get("ticker", "")
        year_later = signal.get("year_later", "")
        pair_id    = signal.get("pair_id", "")
        signals    = signal.get("signals", {})

        if not ticker or not year_later or not signals:
            continue

        sector = SECTOR_MAP.get(ticker, "unknown")

        filing_date = get_filing_date(pair_id, PAIRS_DIR)
        if filing_date is None:
            continue

        cache_key = f"{ticker}_{year_later}"
        if cache_key not in car_cache:
            logger.info(
                f"[{i}/{len(signal_files)}] "
                f"Downloading returns for {ticker} {year_later}"
            )
            car = calculate_car(ticker, filing_date)
            car_cache[cache_key] = car
        else:
            car = car_cache[cache_key]

        if car is None:
            continue

        def encode_direction(d: str) -> int:
            if d == "escalating":
                return 1
            elif d == "de-escalating":
                return -1
            return 0

        record = {
            "ticker":                ticker,
            "sector":                sector,
            "year":                  int(year_later),
            "car_30d":               car,
            "liquidity_dir":         encode_direction(
                signals.get("liquidity_risk", {}).get("direction", "stable")
            ),
            "credit_dir":            encode_direction(
                signals.get("credit_risk", {}).get("direction", "stable")
            ),
            "operational_dir":       encode_direction(
                signals.get("operational_risk", {}).get("direction", "stable")
            ),
            "market_dir":            encode_direction(
                signals.get("market_risk", {}).get("direction", "stable")
            ),
            "regulatory_dir":        encode_direction(
                signals.get("regulatory_risk", {}).get("direction", "stable")
            ),
            "liquidity_intensity":   int(signals.get(
                "liquidity_risk", {}
            ).get("intensity", 1)),
            "credit_intensity":      int(signals.get(
                "credit_risk", {}
            ).get("intensity", 1)),
            "operational_intensity": int(signals.get(
                "operational_risk", {}
            ).get("intensity", 1)),
            "market_intensity":      int(signals.get(
                "market_risk", {}
            ).get("intensity", 1)),
            "regulatory_intensity":  int(signals.get(
                "regulatory_risk", {}
            ).get("intensity", 1)),
        }

        records.append(record)

    df = pd.DataFrame(records)
    logger.info(f"Dataset built: {len(df)} observations")
    return df


# ── Panel OLS regression ──────────────────────────────────────────────────────

def run_panel_ols(df: pd.DataFrame) -> None:
    """
    Run panel OLS regression of CAR on risk signals.
    Reports results for both direction and intensity encoding.
    """
    if len(df) < 50:
        logger.error(
            f"Insufficient observations: {len(df)}. Need at least 50."
        )
        return

    logger.info("\n" + "=" * 60)
    logger.info("PANEL OLS REGRESSION RESULTS")
    logger.info(f"Observations: {len(df)}")
    logger.info(f"Sectors: {df['sector'].nunique()}")
    logger.info(f"Years: {sorted(df['year'].unique())}")
    logger.info("=" * 60)

    # ── Model 1: Direction encoding ───────────────────────────────────────────

    logger.info("\nModel 1: Risk Direction -> 30-Day CAR")
    logger.info("(escalating=1, stable=0, de-escalating=-1)")
    logger.info("With sector and year fixed effects")
    logger.info("-" * 40)

    formula_dir = (
        "car_30d ~ "
        "liquidity_dir + credit_dir + operational_dir + "
        "market_dir + regulatory_dir + "
        "C(sector) + C(year)"
    )

    results_dir = []
    try:
        model_dir = smf.ols(formula_dir, data=df).fit(cov_type="HC3")

        risk_vars = [
            "liquidity_dir", "credit_dir", "operational_dir",
            "market_dir", "regulatory_dir"
        ]

        for var in risk_vars:
            coef   = model_dir.params.get(var, None)
            pvalue = model_dir.pvalues.get(var, None)
            se     = model_dir.bse.get(var, None)

            if coef is not None:
                sig = "***" if pvalue < 0.01 else "** " if pvalue < 0.05 else "*  " if pvalue < 0.10 else "   "
                logger.info(
                    f"{var:25s} coef={coef:+.4f}  "
                    f"se={se:.4f}  p={pvalue:.3f}  {sig}"
                )
                results_dir.append({
                    "variable":    var,
                    "coef":        float(coef),
                    "se":          float(se),
                    "pvalue":      float(pvalue),
                    "significant": bool(pvalue < 0.05)
                })

        logger.info(f"\nR-squared: {model_dir.rsquared:.4f}")
        logger.info("*** p<0.01  ** p<0.05  * p<0.10")

    except Exception as e:
        logger.error(f"Model 1 failed: {e}")

    # ── Model 2: Intensity encoding ───────────────────────────────────────────

    logger.info("\nModel 2: Risk Intensity -> 30-Day CAR")
    logger.info("(intensity 1-5 scale)")
    logger.info("With sector and year fixed effects")
    logger.info("-" * 40)

    formula_int = (
        "car_30d ~ "
        "liquidity_intensity + credit_intensity + "
        "operational_intensity + market_intensity + "
        "regulatory_intensity + "
        "C(sector) + C(year)"
    )

    results_int = []
    try:
        model_int = smf.ols(formula_int, data=df).fit(cov_type="HC3")

        int_vars = [
            "liquidity_intensity", "credit_intensity",
            "operational_intensity", "market_intensity",
            "regulatory_intensity"
        ]

        for var in int_vars:
            coef   = model_int.params.get(var, None)
            pvalue = model_int.pvalues.get(var, None)
            se     = model_int.bse.get(var, None)

            if coef is not None:
                sig = "***" if pvalue < 0.01 else "** " if pvalue < 0.05 else "*  " if pvalue < 0.10 else "   "
                logger.info(
                    f"{var:30s} coef={coef:+.4f}  "
                    f"se={se:.4f}  p={pvalue:.3f}  {sig}"
                )
                results_int.append({
                    "variable":    var,
                    "coef":        float(coef),
                    "se":          float(se),
                    "pvalue":      float(pvalue),
                    "significant": bool(pvalue < 0.05)
                })

        logger.info(f"\nR-squared: {model_int.rsquared:.4f}")
        logger.info("*** p<0.01  ** p<0.05  * p<0.10")

    except Exception as e:
        logger.error(f"Model 2 failed: {e}")

    # ── Save results ──────────────────────────────────────────────────────────

    results_path = OUTPUT_DIR / "regression_results.json"
    results_path.write_text(
        json.dumps({
            "run_at":          datetime.now().isoformat(),
            "n_observations":  int(len(df)),
            "n_sectors":       int(df["sector"].nunique()),
            "years":           [int(y) for y in sorted(df["year"].unique())],
            "model_direction": results_dir,
            "model_intensity": results_int,
            "car_stats": {
                "mean": float(df["car_30d"].mean()),
                "std":  float(df["car_30d"].std()),
                "min":  float(df["car_30d"].min()),
                "max":  float(df["car_30d"].max()),
            }
        }, indent=2),
        encoding="utf-8"
    )
    logger.info(f"\nResults saved: {results_path}")

    df.to_csv(OUTPUT_DIR / "regression_dataset.csv", index=False)
    logger.info(f"Dataset saved: {OUTPUT_DIR / 'regression_dataset.csv'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_validation() -> None:
    logger.info("=" * 60)
    logger.info("Financial Validation Layer")
    logger.info(f"Signal files : {SIGNALS_DIR}")
    logger.info(f"CAR window   : {CAR_WINDOW} days")
    logger.info("=" * 60)

    try:
        import yfinance
        import statsmodels
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Run: pip install yfinance statsmodels")
        return

    df = build_regression_dataset()

    if df.empty:
        logger.error("Empty dataset, cannot run regression")
        return

    logger.info(f"\nDataset summary:")
    logger.info(f"  Observations : {len(df)}")
    logger.info(f"  Tickers      : {df['ticker'].nunique()}")
    logger.info(f"  Sectors      : {df['sector'].value_counts().to_dict()}")
    logger.info(f"  Years        : {sorted(df['year'].unique())}")
    logger.info(
        f"  CAR mean     : {df['car_30d'].mean():.4f} "
        f"({df['car_30d'].mean()*100:.2f}%)"
    )

    run_panel_ols(df)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_validation()