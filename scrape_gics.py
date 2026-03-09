"""scrape_gics.py - Scrape GICS/ICB classification data from Wikipedia index pages."""

import logging
import re
import sqlite3
import sys
import time
from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "data" / "gics_data.db"
SCRAPE_DATE = date.today().isoformat()
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; gics-scraper/1.0)"}

# Priority order for deduplication (lower index = higher priority)
PRIORITY_ORDER = ["sp500", "sp400", "sp600", "russell1000", "nasdaq100"]

# ---- Index configurations ----

INDEX_CONFIGS = {
    "sp500": {
        "url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "table_index": 0,
        "column_map": {
            "Symbol": "ticker",
            "Security": "company_name",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "sub_industry",
            "Headquarters Location": "headquarters_location",
            "Date added": "date_added",
            "CIK": "cik",
        },
        "classification_system": "GICS",
    },
    "sp400": {
        "url": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
        "table_index": 0,
        "column_map": {
            "Symbol": "ticker",
            "Security": "company_name",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "sub_industry",
            "Headquarters Location": "headquarters_location",
        },
        "classification_system": "GICS",
    },
    "sp600": {
        "url": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
        "table_index": 0,
        "column_map": {
            "Symbol": "ticker",
            "Security": "company_name",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "sub_industry",
            "Headquarters Location": "headquarters_location",
            "CIK": "cik",
        },
        "classification_system": "GICS",
    },
    "russell1000": {
        "url": "https://en.wikipedia.org/wiki/Russell_1000_Index",
        "table_match": {"min_rows": 500},
        "column_map": {
            "Symbol": "ticker",
            "Company": "company_name",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "sub_industry",
        },
        "classification_system": "GICS",
    },
    "nasdaq100": {
        "url": "https://en.wikipedia.org/wiki/Nasdaq-100",
        "table_match": {"min_rows": 80},
        "column_map": {
            "Ticker": "ticker",
            "Company": "company_name",
        },
        "icb_columns": True,
        "classification_system": "ICB",
    },
}

# Expected row count ranges for verification
EXPECTED_RANGES = {
    "sp500": (490, 520),
    "sp400": (390, 420),
    "sp600": (580, 620),
    "russell1000": (980, 1030),
    "nasdaq100": (95, 115),
}

# ---- Database ----


def init_db(db_path: Path) -> sqlite3.Connection:
    """Create database and tables if they don't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS index_components (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            index_name TEXT NOT NULL,
            ticker TEXT NOT NULL,
            company_name TEXT NOT NULL,
            classification_system TEXT NOT NULL,
            sector TEXT,
            sub_industry TEXT,
            headquarters_location TEXT,
            date_added TEXT,
            cik TEXT,
            scrape_date TEXT NOT NULL,
            UNIQUE(index_name, ticker, scrape_date)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS consolidated (
            ticker TEXT PRIMARY KEY,
            company_name TEXT NOT NULL,
            classification_system TEXT NOT NULL,
            sector TEXT,
            sub_industry TEXT,
            headquarters_location TEXT,
            indices TEXT NOT NULL,
            scrape_date TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


# ---- Scraping ----


def fetch_page(url: str) -> str:
    """Fetch a Wikipedia page with retries."""
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            log.warning("Attempt %d failed for %s: %s", attempt + 1, url, e)
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise
    raise RuntimeError("fetch failed")


def clean_column_name(col: str) -> str:
    """Strip footnote references like [14] and extra whitespace from column names."""
    return re.sub(r"\[.*?\]", "", str(col)).strip()


def parse_table(html: str, config: dict) -> pd.DataFrame:
    """Extract and normalize the target table from page HTML."""
    dfs = pd.read_html(StringIO(html))

    # Select the right table
    if "table_index" in config:
        df = dfs[config["table_index"]]
    elif "table_match" in config:
        min_rows = config["table_match"]["min_rows"]
        df = next((d for d in dfs if len(d) >= min_rows), None)
        if df is None:
            raise ValueError(
                f"No table found with >= {min_rows} rows "
                f"(tables have {[len(d) for d in dfs]} rows)"
            )
    else:
        raise ValueError("Config must have 'table_index' or 'table_match'")

    # Clean column names
    df.columns = [clean_column_name(c) for c in df.columns]

    # Handle ICB columns for Nasdaq-100
    if config.get("icb_columns"):
        for col in df.columns:
            col_lower = col.lower()
            if "icb" in col_lower or "industry" in col_lower:
                if "sub" in col_lower or "subsector" in col_lower:
                    df = df.rename(columns={col: "sub_industry"})
                else:
                    df = df.rename(columns={col: "sector"})

    # Apply column mapping
    col_map = config["column_map"]
    rename_map = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Keep only schema columns that exist
    schema_cols = [
        "ticker", "company_name", "sector", "sub_industry",
        "headquarters_location", "date_added", "cik",
    ]
    keep = [c for c in schema_cols if c in df.columns]
    df = df[keep]

    # Validate required columns
    if "ticker" not in df.columns:
        raise ValueError(f"Missing 'ticker' column. Available: {list(df.columns)}")
    if "company_name" not in df.columns:
        raise ValueError(f"Missing 'company_name' column. Available: {list(df.columns)}")

    # Clean ticker: strip whitespace, remove footnote markers
    df["ticker"] = (
        df["ticker"]
        .astype(str)
        .str.strip()
        .str.replace(r"\[.*?\]", "", regex=True)
        .str.strip()
    )

    # Drop rows with empty/NaN tickers
    df = df[df["ticker"].notna() & (df["ticker"] != "") & (df["ticker"] != "nan")]

    # Add classification system
    df["classification_system"] = config["classification_system"]

    return df


def store_raw(
    conn: sqlite3.Connection,
    index_name: str,
    df: pd.DataFrame,
    scrape_date: str,
) -> None:
    """Insert scraped data into index_components table."""
    df = df.copy()
    df["index_name"] = index_name
    df["scrape_date"] = scrape_date

    # Ensure all schema columns exist (fill missing with None)
    for col in ["sector", "sub_industry", "headquarters_location", "date_added", "cik"]:
        if col not in df.columns:
            df[col] = None

    insert_cols = [
        "index_name", "ticker", "company_name", "classification_system",
        "sector", "sub_industry", "headquarters_location", "date_added",
        "cik", "scrape_date",
    ]
    df[insert_cols].to_sql(
        "index_components",
        conn,
        if_exists="append",
        index=False,
        method="multi",
    )
    conn.commit()


# ---- ICB to GICS mapping ----

# Manual mapping for Nasdaq-100 tickers that only have ICB classifications.
# These are foreign-domiciled companies not in S&P or Russell indices.
ICB_TO_GICS_OVERRIDES = {
    "ARM":  ("Information Technology", "Semiconductors"),
    "ASML": ("Information Technology", "Semiconductor Materials & Equipment"),
    "CCEP": ("Consumer Staples", "Soft Drinks & Non-alcoholic Beverages"),
    "FER":  ("Industrials", "Construction & Engineering"),
    "MELI": ("Consumer Discretionary", "Broadline Retail"),
    "PDD":  ("Consumer Discretionary", "Broadline Retail"),
    "SHOP": ("Information Technology", "Application Software"),
    "TRI":  ("Industrials", "Research & Consulting Services"),
}


def apply_icb_to_gics(df: pd.DataFrame) -> pd.DataFrame:
    """Reclassify ICB-only tickers to GICS using manual overrides."""
    icb_mask = df["classification_system"] == "ICB"
    mapped = 0
    for idx in df[icb_mask].index:
        ticker = df.at[idx, "ticker"]
        if ticker in ICB_TO_GICS_OVERRIDES:
            sector, sub_industry = ICB_TO_GICS_OVERRIDES[ticker]
            df.at[idx, "sector"] = sector
            df.at[idx, "sub_industry"] = sub_industry
            df.at[idx, "classification_system"] = "GICS"
            mapped += 1
    if mapped:
        log.info("  Mapped %d ICB tickers to GICS", mapped)
    unmapped = icb_mask.sum() - mapped
    if unmapped:
        log.warning("  %d ICB tickers have no GICS override", unmapped)
    return df


# ---- Consolidation ----


def build_consolidated(conn: sqlite3.Connection, scrape_date: str) -> None:
    """Build deduplicated consolidated table from raw index_components."""
    df = pd.read_sql_query(
        "SELECT * FROM index_components WHERE scrape_date = ?",
        conn,
        params=(scrape_date,),
    )

    if df.empty:
        log.warning("No data found for scrape_date=%s", scrape_date)
        return

    # Assign priority
    priority_map = {name: i for i, name in enumerate(PRIORITY_ORDER)}
    df["priority"] = df["index_name"].map(priority_map).fillna(len(PRIORITY_ORDER))

    # Build indices membership map
    indices_map = (
        df.groupby("ticker")["index_name"]
        .apply(lambda x: ",".join(sorted(set(x))))
        .to_dict()
    )

    # Reclassify ICB-only tickers to GICS where possible
    df = apply_icb_to_gics(df)

    # Prefer GICS over ICB, then by priority
    df["is_gics"] = (df["classification_system"] == "GICS").astype(int)
    df = df.sort_values(
        ["ticker", "is_gics", "priority"], ascending=[True, False, True]
    )
    best = df.drop_duplicates(subset="ticker", keep="first").copy()
    best["indices"] = best["ticker"].map(indices_map)

    # Rebuild consolidated table
    conn.execute("DELETE FROM consolidated")
    best[
        [
            "ticker", "company_name", "classification_system", "sector",
            "sub_industry", "headquarters_location", "indices", "scrape_date",
        ]
    ].to_sql("consolidated", conn, if_exists="append", index=False)
    conn.commit()

    log.info("Consolidated: %d unique tickers", len(best))


# ---- Verification ----


def verify_results(conn: sqlite3.Connection, scrape_date: str) -> bool:
    """Sanity-check the scraped data."""
    ok = True

    for index_name, (lo, hi) in EXPECTED_RANGES.items():
        row = conn.execute(
            "SELECT COUNT(*) FROM index_components WHERE index_name=? AND scrape_date=?",
            (index_name, scrape_date),
        ).fetchone()
        count = row[0] if row else 0
        if lo <= count <= hi:
            log.info("  %s: %d rows OK", index_name, count)
        elif count == 0:
            log.warning("  %s: NO DATA (skipped or failed)", index_name)
            ok = False
        else:
            log.warning("  %s: %d rows (expected %d-%d)", index_name, count, lo, hi)
            ok = False

    total = conn.execute("SELECT COUNT(*) FROM consolidated").fetchone()[0]
    log.info("  Consolidated total: %d unique tickers", total)

    # Check well-known tickers
    for ticker in ["AAPL", "MSFT", "AMZN", "GOOGL", "JPM"]:
        row = conn.execute(
            "SELECT indices FROM consolidated WHERE ticker=?", (ticker,)
        ).fetchone()
        if row is None:
            log.warning("  Known ticker %s missing from consolidated!", ticker)
            ok = False

    # Check GICS sector count
    sector_count = conn.execute(
        "SELECT COUNT(DISTINCT sector) FROM consolidated WHERE classification_system='GICS'"
    ).fetchone()[0]
    log.info("  Distinct GICS sectors: %d", sector_count)
    if sector_count < 10:
        log.warning("  Expected ~11 GICS sectors, got %d", sector_count)

    return ok


# ---- Main ----


def main() -> None:
    log.info("Starting GICS scrape for %s", SCRAPE_DATE)
    conn = init_db(DB_PATH)

    succeeded = []
    failed = []

    for index_name, config in INDEX_CONFIGS.items():
        log.info("Scraping %s from %s", index_name, config["url"])
        try:
            html = fetch_page(config["url"])
            df = parse_table(html, config)
            store_raw(conn, index_name, df, SCRAPE_DATE)
            log.info("  %s: %d rows scraped", index_name, len(df))
            succeeded.append(index_name)
        except Exception:
            log.exception("  Failed to scrape %s", index_name)
            failed.append(index_name)

    log.info("Scraping complete: %d succeeded, %d failed", len(succeeded), len(failed))

    if succeeded:
        build_consolidated(conn, SCRAPE_DATE)

    log.info("Verification:")
    verify_results(conn, SCRAPE_DATE)

    conn.close()
    log.info("Done. Database at %s", DB_PATH)


if __name__ == "__main__":
    main()
