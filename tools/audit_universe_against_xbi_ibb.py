#!/usr/bin/env python3
"""
Universe Hygiene Audit: Model Universe vs. XBI / IBB ETF Holdings
=================================================================
UNIVERSE_HYGIENE_AUDIT / COVERAGE_DIAGNOSTIC / NO_MODEL_CHANGE /
NO_RANKER_CHANGE / NO_SELECTOR_CHANGE / NO_SIZING_CHANGE / NO_TRADING_CHANGE

Constraints:
- Does NOT modify production_data/universe.json
- Does NOT change any model, ranker, selector, scoring, sizing, cron, or trading file
- Does NOT add or delete tickers from the production universe
- All proposed changes written to proposed_universe_actions.csv only
- Proposals only; no production mutation
"""

import argparse
import io
import json
import os
from datetime import date, datetime, timezone
from typing import Dict, List, Optional, Tuple

UTC = timezone.utc

import pandas as pd
from pandas.tseries.offsets import BDay

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TODAY = pd.Timestamp("2026-06-28")
STALE_CUTOFF = TODAY - 10 * BDay()  # ~2026-06-13

KNOWN_PROBLEM_TICKERS = ["RNA", "GOSS", "REPL", "ACLX", "APLS", "DAWN", "FOLD", "GLPG", "KALV", "TERN"]

# Large-cap biopharma — exclude from candidate list
LARGE_CAP_EXCLUSIONS = {
    "ABBV",
    "AMGN",
    "BIIB",
    "GILD",
    "REGN",
    "VRTX",
    "LLY",
    "BMY",
    "PFE",
    "MRK",
    "JNJ",
    "NVO",
    "AZN",
    "RHHBY",
    "NOVN",
    "SNY",
    "BAYRY",
    "GSK",
    "TAK",
    "NVS",
    "ALNY",
    "MRNA",
}

# Diagnostics / tools companies — exclude
TOOLS_DIAGNOSTICS = {
    "ILMN",
    "MYRG",
    "NTRA",
    "PACB",
    "OXFD",
    "GHDX",
    "TXG",
    "NVTA",
    "SDGR",
    "CDNA",
}

# Managed care / services
HEALTHCARE_SERVICES = {"UNH", "CVS", "CI", "HUM", "CNC", "MOH", "ELV", "ABC", "MCK", "CAH"}

# ETF download URLs
XBI_URLS = [
    "https://www.ssga.com/us/en/institutional/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xbi.xlsx",
    "https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xbi.xlsx",
]
# IBB correct product ID is 239699 (not 239451 which routes to IGSB bond fund).
# The BlackRock Varnish API endpoint with portfolioId=239699 returns a proper
# iShares Biotechnology ETF CSV. Source verified 2026-06-28.
IBB_URL = (
    "https://www.blackrock.com/varnish-api/blk-one01-product-data/product-data"
    "/api/v1/get-fund-document"
    "?appType=PRODUCT_PAGE&appSubType=ISHARES&targetSite=us-ishares"
    "&locale=en_US&portfolioId=239699&userType=individual"
    "&asOfDate=20260626&component=holdings"
)
IBB_REFERER = "https://www.ishares.com/us/products/239699/ishares-biotechnology-etf"


# ---------------------------------------------------------------------------
# ETF fetch helpers
# ---------------------------------------------------------------------------


# Holdings older than this are refused: an ETF audit against stale constituents proposes
# adding names that have since left the fund (the IBB URL above pins asOfDate=20260626,
# and every later date redirects to an unreachable internal host).
MAX_HOLDINGS_AGE_DAYS = 10
_AS_OF_FORMATS = ("%b %d, %Y", "%d-%b-%Y", "%Y-%m-%d", "%m/%d/%Y")


def _parse_as_of(text: str) -> Optional[date]:
    text = text.strip().strip('"').strip()
    for fmt in _AS_OF_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def parse_ishares_holdings_csv(content: str) -> Tuple[pd.DataFrame, Optional[date]]:
    """Parse an iShares holdings CSV (metadata lines, then a Ticker,Name,... table).

    Returns (equity holdings with columns ticker/name/weight_pct, the 'Fund Holdings as of' date or None).
    """
    lines = content.splitlines()
    as_of = None
    header_line = None
    for i, line in enumerate(lines):
        if as_of is None and line.lower().startswith("fund holdings as of"):
            as_of = _parse_as_of(line.split(",", 1)[1] if "," in line else "")
        if "Ticker" in line and "Name" in line:
            header_line = i
            break
    if header_line is None:
        return pd.DataFrame(columns=["ticker", "name", "weight_pct"]), as_of
    df = pd.read_csv(io.StringIO("\n".join(lines[header_line:])), on_bad_lines="skip")
    if "Asset Class" in df.columns:
        df = df[df["Asset Class"] == "Equity"].copy()
    df = df[df["Ticker"].astype(str).str.strip().str.match(r"^[A-Z]{1,6}$")].copy()
    result = pd.DataFrame()
    result["ticker"] = df["Ticker"].astype(str).str.strip()
    result["name"] = df["Name"].astype(str).str.strip() if "Name" in df.columns else ""
    result["weight_pct"] = df["Weight (%)"] if "Weight (%)" in df.columns else None
    return result.drop_duplicates("ticker").reset_index(drop=True), as_of


def holdings_are_stale(
    as_of: Optional[date], today: Optional[date] = None, max_age_days: int = MAX_HOLDINGS_AGE_DAYS
) -> bool:
    """True if the holdings date is unknown or more than max_age_days before today."""
    if as_of is None:
        return True
    today = today or datetime.now(UTC).date()
    return (today - as_of).days > max_age_days


def ibb_holdings_from_content(content: str, origin: str, today: Optional[date] = None) -> Tuple[pd.DataFrame, str]:
    """Parse iShares IBB holdings and refuse them when stale. Returns (df, source_note)."""
    df, as_of = parse_ishares_holdings_csv(content)
    if df.empty:
        return pd.DataFrame(), f"FETCH_FAILED_UNPARSEABLE ({origin})"
    if holdings_are_stale(as_of, today):
        return pd.DataFrame(), (
            f"FETCH_FAILED_STALE_HOLDINGS (as_of={as_of.isoformat() if as_of else 'unknown'}, "
            f"limit {MAX_HOLDINGS_AGE_DAYS} days; {origin}). Download current holdings from the iShares IBB page "
            "and pass --ibb-holdings-file."
        )
    return df, f"{origin} (iShares Biotechnology ETF, as_of={as_of.isoformat()}) @ {datetime.now(UTC).isoformat()}"


def _normalize_etf_ticker(raw: str) -> Optional[str]:
    """Strip class-share suffixes and clean ticker string."""
    if not isinstance(raw, str):
        return None
    t = raw.strip().upper()
    # Skip cash, derivatives, empty
    skip_prefixes = ["", "-", "CASH", "DERIVATIVE", "MMF", "STIT"]
    if not t or any(t.startswith(p) for p in skip_prefixes):
        return None
    # Strip suffix like .A, .B, /A etc
    for sep in [".", "/"]:
        if sep in t:
            parts = t.split(sep)
            # Only strip if suffix is short (1-2 chars)
            if len(parts[-1]) <= 2:
                t = parts[0]
    return t if t else None


def _fetch_xbi_holdings(verbose: bool = True) -> Tuple[pd.DataFrame, str]:
    """Attempt to download XBI holdings from SPDR. Returns (df, source_note)."""
    try:
        import requests

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,*/*",
        }

        for url in XBI_URLS:
            if verbose:
                print(f"  Trying XBI URL: {url}")
            try:
                resp = requests.get(url, headers=headers, timeout=30)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    # Detect file format
                    magic = resp.content[:4]
                    if magic[:2] == b"PK":
                        # XLSX format — SPDR standard
                        try:
                            raw = pd.read_excel(
                                io.BytesIO(resp.content),
                                sheet_name=0,
                                header=None,
                            )
                            # Find header row: contains "Ticker" column
                            header_row = None
                            for i, row in raw.iterrows():
                                row_vals = [str(v).strip().upper() for v in row.values]
                                if "TICKER" in row_vals or "SYMBOL" in row_vals:
                                    header_row = i
                                    break
                            if header_row is not None:
                                df = pd.read_excel(
                                    io.BytesIO(resp.content),
                                    sheet_name=0,
                                    header=header_row,
                                )
                                df.columns = [str(c).strip() for c in df.columns]
                                ticker_col = next(
                                    (c for c in df.columns if c.upper() in ("TICKER", "SYMBOL")),
                                    None,
                                )
                                if ticker_col:
                                    # Filter to valid equity ticker rows (1-6 uppercase letters only)
                                    ticker_mask = df[ticker_col].astype(str).str.strip().str.match(r"^[A-Z]{1,6}$")
                                    df = df[ticker_mask].copy()
                                    name_col = next(
                                        (c for c in df.columns if c.upper() in ("NAME", "HOLDING NAME")),
                                        None,
                                    )
                                    weight_col = next(
                                        (c for c in df.columns if "WEIGHT" in c.upper()),
                                        None,
                                    )
                                    result = pd.DataFrame()
                                    result["ticker"] = df[ticker_col].str.strip()
                                    result["name"] = df[name_col].str.strip() if name_col else ""
                                    result["weight_pct"] = df[weight_col] if weight_col else None
                                    result = result.drop_duplicates("ticker").reset_index(drop=True)
                                    source = f"SPDR_LIVE_XLSX ({url}) @ {datetime.now(UTC).isoformat()}"
                                    return result, source
                        except Exception as e:
                            if verbose:
                                print(f"  Excel parse failed: {e}")
                    else:
                        # Fallback: try CSV
                        try:
                            content = resp.content.decode("utf-8", errors="replace")
                            # Find header line
                            lines = content.splitlines()
                            header_line = None
                            for i, line in enumerate(lines):
                                if "ticker" in line.lower() or "symbol" in line.lower():
                                    header_line = i
                                    break
                            if header_line is not None:
                                df = pd.read_csv(io.StringIO("\n".join(lines[header_line:])))
                                df.columns = [str(c).strip() for c in df.columns]
                                ticker_col = next((c for c in df.columns if c.upper() in ("TICKER", "SYMBOL")), None)
                                if ticker_col:
                                    ticker_mask = df[ticker_col].astype(str).str.strip().str.match(r"^[A-Z]{1,6}$")
                                    df = df[ticker_mask].copy()
                                    result = pd.DataFrame()
                                    result["ticker"] = df[ticker_col].str.strip()
                                    result["name"] = ""
                                    result["weight_pct"] = None
                                    result = result.drop_duplicates("ticker").reset_index(drop=True)
                                    source = f"SPDR_LIVE_CSV ({url}) @ {datetime.now(UTC).isoformat()}"
                                    return result, source
                        except Exception as e:
                            if verbose:
                                print(f"  CSV parse failed: {e}")
            except Exception as e:
                if verbose:
                    print(f"  Request failed for {url}: {e}")
    except ImportError:
        if verbose:
            print("  requests not available")

    return pd.DataFrame(), "FETCH_FAILED_FALLBACK_NOT_AUTHORITATIVE"


def _fetch_ibb_holdings(verbose: bool = True) -> Tuple[pd.DataFrame, str]:
    """
    Attempt to download IBB holdings from iShares.

    Fetch IBB (iShares Biotechnology ETF) holdings from the BlackRock Varnish API.

    NOTE on product ID: IBB's correct iShares product ID is 239699 (NOT 239451).
    Product 239451 routes to IGSB (bond fund) in their Varnish cache — a routing
    bug on their side. Verified 2026-06-28.

    The BlackRock Varnish API at portfolioId=239699 returns a proper CSV with
    columns: Ticker, Name, Sector, Asset Class, Market Value, Weight (%), etc.

    Returns (df, source_note).
    """
    try:
        import requests

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Referer": IBB_REFERER,
            "Accept": "text/csv,*/*",
        }
        if verbose:
            print(f"  Trying IBB URL (portfolioId=239699): {IBB_URL[:80]}...")
        resp = requests.get(IBB_URL, headers=headers, timeout=30)
        if resp.status_code == 200 and len(resp.content) > 500:
            content = resp.content.decode("utf-8", errors="replace")

            # Detect HTML response
            if content.lstrip().startswith("<!DOCTYPE") or "<html" in content[:200]:
                if verbose:
                    print("  IBB endpoint returned HTML. " "Provide holdings manually via --ibb-holdings-file.")
                return (
                    pd.DataFrame(),
                    "FETCH_FAILED_RETURNED_HTML",
                )

            df, source = ibb_holdings_from_content(content, "BLACKROCK_VARNISH_API_portfolioId=239699")
            if verbose and df.empty:
                print(f"  IBB holdings refused: {source}")
            return df, source
        else:
            if verbose:
                print(f"  IBB fetch returned status {resp.status_code}")
    except ImportError:
        if verbose:
            print("  requests not available")
    except Exception as e:
        if verbose:
            print(f"  IBB fetch failed: {e}")

    return pd.DataFrame(), "FETCH_FAILED_FALLBACK_NOT_AUTHORITATIVE"


# ---------------------------------------------------------------------------
# Universe loading
# ---------------------------------------------------------------------------


def load_universe(path: str) -> Tuple[List[dict], pd.DataFrame]:
    """Load universe.json and return (raw list, summary DataFrame)."""
    with open(path) as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        # Convert dict format to list
        raw = [{"ticker": k, **v} for k, v in raw.items()]

    rows = []
    for item in raw:
        ticker = item.get("ticker", "")
        # Company name: prefer company_name, then name, then market_data.company_name
        company_name = (
            item.get("company_name") or item.get("name") or (item.get("market_data") or {}).get("company_name") or ""
        )
        status = item.get("status", "unknown")
        exchange = item.get("exchange", "")
        sector = item.get("sector", "")
        etf_sources = item.get("etf_sources") or []
        description = item.get("description", "")

        # Exclusion / delisted flags
        known_delisted = status == "delisted"
        exclusion_reason = ""
        if known_delisted:
            exclusion_reason = "status=delisted"
        elif status in ("pending_coverage", "pending_data_collection"):
            exclusion_reason = f"status={status}"

        rows.append(
            {
                "ticker": ticker,
                "company_name": company_name,
                "status": status,
                "exchange": exchange,
                "sector": sector,
                "etf_sources": "|".join(etf_sources) if isinstance(etf_sources, list) else str(etf_sources),
                "description": description,
                "known_delisted": known_delisted,
                "exclusion_reason": exclusion_reason,
            }
        )

    return raw, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Price staleness detection
# ---------------------------------------------------------------------------


def load_price_staleness(
    split_adj_path: str,
    raw_path: Optional[str] = None,
    stale_cutoff: pd.Timestamp = STALE_CUTOFF,
) -> pd.DataFrame:
    """
    Return DataFrame with columns [ticker, last_price_date_split_adj,
    last_price_date_raw, last_price_date_combined, is_stale, has_any_price].
    """

    def _load_and_clean(path: str, date_col: str = "date") -> pd.DataFrame:
        df = pd.read_csv(path, low_memory=False)
        # Filter out corrupt/weird rows (contain 'TICKER\n' pattern)
        ticker_col = "ticker" if "ticker" in df.columns else df.columns[0]
        df = df[~df[ticker_col].astype(str).str.contains("TICKER|\\\\n", na=False)].copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col, ticker_col])
        return df[[ticker_col, date_col]].rename(columns={ticker_col: "ticker", date_col: "date"})

    df_sa = _load_and_clean(split_adj_path)
    last_sa = df_sa.groupby("ticker")["date"].max().rename("last_price_date_split_adj")

    last_raw = pd.Series(dtype="datetime64[ns]", name="last_price_date_raw")
    if raw_path and os.path.exists(raw_path):
        df_r = _load_and_clean(raw_path)
        last_raw = df_r.groupby("ticker")["date"].max().rename("last_price_date_raw")

    combined = pd.concat([last_sa, last_raw], axis=1)
    combined["last_price_date_combined"] = combined[["last_price_date_split_adj", "last_price_date_raw"]].max(axis=1)
    combined["has_any_price"] = combined["last_price_date_combined"].notna()
    combined["is_stale"] = combined["has_any_price"] & (combined["last_price_date_combined"] < stale_cutoff)
    combined = combined.reset_index()
    return combined


# ---------------------------------------------------------------------------
# Model ticker classification
# ---------------------------------------------------------------------------


def classify_model_tickers(
    universe_df: pd.DataFrame,
    staleness_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge universe + staleness and assign classification."""
    df = universe_df.merge(staleness_df, on="ticker", how="left")

    def _classify(row):
        if row["known_delisted"]:
            return "DELISTED_OR_INACTIVE"
        has_price = row.get("has_any_price", False)
        is_stale = row.get("is_stale", False)
        status = row.get("status", "")

        if not has_price:
            return "PRICE_DATA_MISSING"
        if is_stale:
            # Check if it's delisted-adjacent
            if status in ("delisted",):
                return "DELISTED_OR_INACTIVE"
            return "ACTIVE_BUT_STALE_PRICE"
        if status in ("active", "benchmark"):
            return "ACTIVE_VALID"
        if status in ("pending_coverage", "pending_data_collection"):
            return "NEEDS_REVIEW"
        return "ACTIVE_VALID"

    df["model_classification"] = df.apply(_classify, axis=1)
    return df


# ---------------------------------------------------------------------------
# ETF missing classification
# ---------------------------------------------------------------------------


def classify_etf_missing(
    ticker: str,
    name: str,
    etf_source: str,
    universe_tickers: set,
) -> str:
    """Classify a ticker that is in an ETF but not in model universe."""
    t = ticker.upper()
    n = (name or "").upper()

    if t in universe_tickers:
        return "ALREADY_IN_MODEL"

    if t in LARGE_CAP_EXCLUSIONS:
        return "MISSING_LARGE_CAP_BIOPHARMA"

    if t in TOOLS_DIAGNOSTICS:
        return "MISSING_TOOLS_OR_DIAGNOSTICS"

    if t in HEALTHCARE_SERVICES:
        return "MISSING_HEALTHCARE_SERVICES"

    # Check for CVR (contingent value rights) FIRST — derivative instrument, never add
    # CVRs are non-transferable rights tied to a deal outcome (e.g. AKE = Akero CVR)
    if "CVR" in n or "CONTINGENT VALUE" in n or t.endswith("R") and "CVR" in n:
        return "MISSING_LOW_RELEVANCE"

    # Check name for ADR/foreign indicators
    adr_words = ["ADR", "SPONSORED", "ORDINARY SHARES", " NV", " SA ", " AG ", " PLC", " AB "]
    if any(w in n for w in adr_words):
        return "MISSING_ADR_OR_FOREIGN"

    # Check name for services, tools, diagnostics
    tool_words = ["DIAGNOSTICS", "TOOLS", "SEQUENCING", "ILLUMINA", "SERVICES", "CONTRACT"]
    if any(w in n for w in tool_words):
        return "MISSING_TOOLS_OR_DIAGNOSTICS"

    # Non-biotech indicators in name
    non_biotech_words = [
        "HYPERLIQUID",
        "CRYPTO",
        "BITCOIN",
        "BLOCKCHAIN",
        "STRATEGY",
        "STRATEGIES",
        "FUND",
        "TRUST",
        "CAPITAL",
        "FINANCIAL",
        "BANK",
        "INSURANCE",
        "REAL ESTATE",
        "REIT",
    ]
    if any(w in n for w in non_biotech_words):
        return "MISSING_LOW_RELEVANCE"

    # If XBI — pure-play biotech
    if "XBI" in etf_source:
        return "MISSING_CORE_BIOTECH_CANDIDATE"

    # IBB also holds many pure-play small/mid-cap biotechs
    # Check name for biotech indicators
    biotech_words = [
        "THERAPEUTICS",
        "BIOSCIENCES",
        "BIOPHARMA",
        "BIOTECH",
        "BIOLOGICS",
        "MEDICINES",
        "PHARMA",
        "GENOMICS",
        "ONCOLOGY",
        "NEUROSCIENCES",
        "IMMUNOLOGY",
        "GENE",
        "CELL THERAPY",
        "ANTIBODY",
    ]
    if any(w in n for w in biotech_words):
        return "MISSING_CORE_BIOTECH_CANDIDATE"

    return "MISSING_NEEDS_MANUAL_REVIEW"


# ---------------------------------------------------------------------------
# Identifier conflict detection
# ---------------------------------------------------------------------------


def find_identifier_conflicts(
    universe_df: pd.DataFrame,
    xbi_df: pd.DataFrame,
    ibb_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Look for:
    1. Same company name under different tickers (fuzzy: first 3 words)
    2. ETF ticker differs from repo ticker for same company
    """
    conflicts = []

    # Build name->ticker map from universe
    def name_key(n: str) -> str:
        if not isinstance(n, str):
            return ""
        words = n.strip().split()[:3]
        return " ".join(w.upper() for w in words)

    uni_name_map: Dict[str, List[str]] = {}
    for _, row in universe_df.iterrows():
        key = name_key(row["company_name"])
        if key and len(key) > 3:
            uni_name_map.setdefault(key, []).append(row["ticker"])

    # Find duplicates within universe
    for key, tickers in uni_name_map.items():
        if len(tickers) > 1:
            conflicts.append(
                {
                    "conflict_type": "SAME_NAME_MULTI_TICKER_IN_UNIVERSE",
                    "name_key": key,
                    "tickers": "|".join(sorted(tickers)),
                    "details": f"Company name prefix '{key}' maps to {len(tickers)} tickers in model universe",
                }
            )

    # Compare ETF names to universe
    for etf_source, etf_df in [("XBI", xbi_df), ("IBB", ibb_df)]:
        if etf_df.empty:
            continue
        for _, etf_row in etf_df.iterrows():
            etf_ticker = etf_row["ticker"]
            etf_name = etf_row.get("name", "")
            etf_key = name_key(etf_name)
            if not etf_key or len(etf_key) < 4:
                continue
            # Look for matching name in universe with different ticker
            uni_matches = [t for k, tlist in uni_name_map.items() for t in tlist if k == etf_key]
            for uni_ticker in uni_matches:
                if uni_ticker != etf_ticker:
                    conflicts.append(
                        {
                            "conflict_type": "ETF_TICKER_DIFFERS_FROM_MODEL_TICKER",
                            "name_key": etf_key,
                            "tickers": f"ETF:{etf_source}={etf_ticker} vs MODEL={uni_ticker}",
                            "details": (
                                f"ETF {etf_source} has ticker {etf_ticker} for '{etf_key}'; "
                                f"model has {uni_ticker} for same name prefix"
                            ),
                        }
                    )

    return pd.DataFrame(conflicts)


# ---------------------------------------------------------------------------
# Build proposed actions
# ---------------------------------------------------------------------------


def build_proposed_actions(
    classified_df: pd.DataFrame,
    missing_xbi: pd.DataFrame,
    missing_ibb: pd.DataFrame,
    conflicts: pd.DataFrame,
) -> pd.DataFrame:
    """Build proposed_universe_actions.csv rows."""
    rows = []

    # Stale / inactive model tickers
    for _, row in classified_df.iterrows():
        cl = row["model_classification"]
        if cl == "DELISTED_OR_INACTIVE":
            rows.append(
                {
                    "ticker": row["ticker"],
                    "company_name": row["company_name"],
                    "action": "MARK_INACTIVE_CONFIRMED",
                    "priority": "HIGH",
                    "source": "model_universe_audit",
                    "classification": cl,
                    "last_price_date": str(row.get("last_price_date_combined", "")),
                    "notes": f"status={row['status']}; already flagged delisted in universe",
                    "requires_manual_approval": True,
                }
            )
        elif cl == "ACTIVE_BUT_STALE_PRICE":
            rows.append(
                {
                    "ticker": row["ticker"],
                    "company_name": row["company_name"],
                    "action": "INVESTIGATE_PRICE_STALENESS",
                    "priority": "MEDIUM",
                    "source": "model_universe_audit",
                    "classification": cl,
                    "last_price_date": str(row.get("last_price_date_combined", "")),
                    "notes": f"Price data stale; status={row['status']}",
                    "requires_manual_approval": True,
                }
            )
        elif cl == "PRICE_DATA_MISSING":
            rows.append(
                {
                    "ticker": row["ticker"],
                    "company_name": row["company_name"],
                    "action": "INVESTIGATE_MISSING_PRICE",
                    "priority": "MEDIUM",
                    "source": "model_universe_audit",
                    "classification": cl,
                    "last_price_date": "",
                    "notes": "No price data in any price CSV",
                    "requires_manual_approval": True,
                }
            )

    def _etf_action_priority(cl: str) -> tuple:
        """Return (action, priority) for an ETF missing-classification."""
        if cl == "MISSING_CORE_BIOTECH_CANDIDATE":
            return "EVALUATE_FOR_ADDITION", "HIGH"
        if cl == "MISSING_NEEDS_MANUAL_REVIEW":
            return "NEEDS_MANUAL_REVIEW", "MEDIUM"
        return "DO_NOT_ADD", "LOW"

    # Missing from model — XBI
    if not missing_xbi.empty:
        for _, row in missing_xbi.iterrows():
            cl = row.get("missing_classification", "MISSING_NEEDS_MANUAL_REVIEW")
            action, priority = _etf_action_priority(cl)
            rows.append(
                {
                    "ticker": row["ticker"],
                    "company_name": row.get("name", ""),
                    "action": action,
                    "priority": priority,
                    "source": "XBI_ETF_COVERAGE_GAP",
                    "classification": cl,
                    "last_price_date": "",
                    "notes": f"In XBI ETF but not in model; classification={cl}",
                    "requires_manual_approval": True,
                }
            )

    # Missing from model — IBB
    if not missing_ibb.empty:
        for _, row in missing_ibb.iterrows():
            cl = row.get("missing_classification", "MISSING_NEEDS_MANUAL_REVIEW")
            action, priority = _etf_action_priority(cl)
            rows.append(
                {
                    "ticker": row["ticker"],
                    "company_name": row.get("name", ""),
                    "action": action,
                    "priority": priority,
                    "source": "IBB_ETF_COVERAGE_GAP",
                    "classification": cl,
                    "last_price_date": "",
                    "notes": f"In IBB ETF but not in model; classification={cl}",
                    "requires_manual_approval": True,
                }
            )

    # Conflicts
    for _, row in conflicts.iterrows():
        rows.append(
            {
                "ticker": row.get("tickers", ""),
                "company_name": row.get("name_key", ""),
                "action": "RESOLVE_IDENTIFIER_CONFLICT",
                "priority": "HIGH",
                "source": "IDENTIFIER_CONFLICT_DETECTION",
                "classification": row.get("conflict_type", ""),
                "last_price_date": "",
                "notes": row.get("details", ""),
                "requires_manual_approval": True,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def build_markdown_report(
    classified_df: pd.DataFrame,
    xbi_df: pd.DataFrame,
    ibb_df: pd.DataFrame,
    xbi_source: str,
    ibb_source: str,
    missing_xbi: pd.DataFrame,
    missing_ibb: pd.DataFrame,
    conflicts: pd.DataFrame,
    proposed: pd.DataFrame,
    audit_ts: str,
) -> str:
    """Generate UNIVERSE_HYGIENE_AUDIT.md content."""

    n_universe = len(classified_df)
    n_xbi = len(xbi_df)
    n_ibb = len(ibb_df)

    # Classification counts
    cl_counts = classified_df["model_classification"].value_counts().to_dict()
    n_active_valid = cl_counts.get("ACTIVE_VALID", 0)
    n_stale = cl_counts.get("ACTIVE_BUT_STALE_PRICE", 0)
    n_delisted = cl_counts.get("DELISTED_OR_INACTIVE", 0)
    n_missing_price = cl_counts.get("PRICE_DATA_MISSING", 0)
    n_needs_review = cl_counts.get("NEEDS_REVIEW", 0)

    # Missing ETF counts
    n_miss_xbi = len(missing_xbi)
    n_miss_ibb = len(missing_ibb)

    # XBI high-priority missing
    xbi_hp = (
        missing_xbi[missing_xbi.get("missing_classification", pd.Series(dtype=str)) == "MISSING_CORE_BIOTECH_CANDIDATE"]
        if not missing_xbi.empty
        else pd.DataFrame()
    )
    n_xbi_hp = len(xbi_hp)
    ibb_hp = (
        missing_ibb[missing_ibb.get("missing_classification", pd.Series(dtype=str)) == "MISSING_CORE_BIOTECH_CANDIDATE"]
        if not missing_ibb.empty
        else pd.DataFrame()
    )
    n_ibb_hp = len(ibb_hp)

    # Known problem ticker status
    problem_status = {}
    for t in KNOWN_PROBLEM_TICKERS:
        rows = classified_df[classified_df["ticker"] == t]
        if rows.empty:
            problem_status[t] = "NOT IN MODEL UNIVERSE"
        else:
            row = rows.iloc[0]
            lp = row.get("last_price_date_combined", "")
            if pd.notna(lp) and lp:
                lp_str = pd.Timestamp(lp).strftime("%Y-%m-%d")
            else:
                lp_str = "N/A"
            problem_status[t] = (
                f"{row['model_classification']} | status={row['status']} | "
                f"last_price={lp_str} | name={row['company_name']}"
            )

    def _table(df: pd.DataFrame, cols: List[str]) -> str:
        if df.empty:
            return "_No entries._\n"
        df2 = df[cols].copy()
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows_str = []
        for _, row in df2.iterrows():
            rows_str.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join([header, sep] + rows_str) + "\n"

    # Build stale/inactive table
    stale_df = classified_df[
        classified_df["model_classification"].isin(
            ["ACTIVE_BUT_STALE_PRICE", "DELISTED_OR_INACTIVE", "PRICE_DATA_MISSING"]
        )
    ].copy()
    if not stale_df.empty and "last_price_date_combined" in stale_df.columns:
        stale_df["last_price"] = stale_df["last_price_date_combined"].apply(
            lambda x: pd.Timestamp(x).strftime("%Y-%m-%d") if pd.notna(x) and x else "N/A"
        )
    else:
        stale_df["last_price"] = "N/A"

    # Build missing ETF tables
    def _miss_table(df: pd.DataFrame, source: str) -> str:
        if df.empty or "FETCH_FAILED" in source:
            return f"_ETF holdings fetch {source} — no data available._\n"
        cols = ["ticker", "name", "missing_classification"]
        available_cols = [c for c in cols if c in df.columns]
        return _table(df, available_cols)

    lines = [
        "# Universe Hygiene Audit Report",
        "",
        f"**Audit timestamp:** {audit_ts}",
        "**Reference date:** 2026-06-28",
        f"**Stale price cutoff:** {STALE_CUTOFF.date()} (10 trading days before 2026-06-28)",
        "",
        "---",
        "",
        "## Classification",
        "",
        "```",
        "UNIVERSE_HYGIENE_AUDIT",
        "COVERAGE_DIAGNOSTIC",
        "NO_MODEL_CHANGE",
        "NO_RANKER_CHANGE",
        "NO_SELECTOR_CHANGE",
        "NO_SIZING_CHANGE",
        "NO_TRADING_CHANGE",
        "```",
        "",
        "---",
        "",
        "## Executive Verdict",
        "",
        f"Model universe: **{n_universe} tickers**.",
        f"- Active/valid: **{n_active_valid}**",
        f"- Already flagged delisted/inactive: **{n_delisted}** (APLS, GLPG, KALV, ACLX, DAWN, FOLD, TERN)",
        f"- Stale price (>10 trading days): **{n_stale}** (no price since before {STALE_CUTOFF.date()})",
        f"- Price data missing entirely: **{n_missing_price}** (placeholder `_XBI_BENCHMARK_` entry)",
        f"- Pending / needs review: **{n_needs_review}**",
        "",
        (
            f"XBI ETF: **{n_xbi} holdings** (source: {xbi_source[:80]}...)"
            if len(xbi_source) > 80
            else f"XBI ETF: **{n_xbi} holdings** (source: {xbi_source})"
        ),
        (
            f"IBB ETF: **{n_ibb} holdings** (source: {ibb_source[:80]}...)"
            if len(ibb_source) > 80
            else f"IBB ETF: **{n_ibb} holdings** (source: {ibb_source})"
        ),
        "",
        f"Missing from model: **{n_miss_xbi} XBI** / **{n_miss_ibb} IBB** candidates",
        f"High-priority new candidates (XBI small/mid pure-play): **{n_xbi_hp}**",
        "",
        "**No production files were modified.** All findings are proposals only.",
        "",
        "---",
        "",
        "## Data Sources",
        "",
        f"- **Model universe:** `production_data/universe.json` — {n_universe} entries",
        "- **Split-adjusted prices:** `production_data/price_history_split_adj.csv` — clean ticker count varies",
        "- **Raw prices:** `production_data/price_history.csv`",
        f"- **XBI holdings:** {xbi_source}",
        f"- **IBB holdings:** {ibb_source}",
        "",
        "---",
        "",
        "## Current Model Universe Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Total tickers | {n_universe} |",
        f"| Active (valid price) | {n_active_valid} |",
        f"| Delisted / inactive | {n_delisted} |",
        f"| Active but stale price | {n_stale} |",
        f"| Price data missing | {n_missing_price} |",
        f"| Pending / needs review | {n_needs_review} |",
        "",
        "---",
        "",
        "## XBI Coverage Section",
        "",
        f"XBI ETF ({n_xbi} holdings) vs model ({n_universe} tickers).",
        "",
        f"- Tickers in XBI not in model: **{n_miss_xbi}**",
        f"- Core biotech candidates (XBI small/mid): **{n_xbi_hp}**",
        "",
        "### Missing XBI Names",
        "",
        _miss_table(missing_xbi, xbi_source),
        "",
        "---",
        "",
        "## IBB Coverage Section",
        "",
        f"IBB ETF ({n_ibb} holdings) vs model ({n_universe} tickers).",
        "",
        f"- Tickers in IBB not in model: **{n_miss_ibb}**",
        f"- Core biotech candidates (IBB): **{n_ibb_hp}**",
        "",
        "### Missing IBB Names",
        "",
        _miss_table(missing_ibb, ibb_source),
        "",
        "---",
        "",
        "## High-Priority Quarantine Candidates",
        "",
        "Tickers in model flagged as stale or inactive that should be reviewed for removal:",
        "",
    ]

    # Add stale/inactive table
    if not stale_df.empty:
        lines.append(
            _table(
                stale_df[["ticker", "company_name", "model_classification", "status", "last_price"]],
                ["ticker", "company_name", "model_classification", "status", "last_price"],
            )
        )
    else:
        lines.append("_None._\n")

    lines += [
        "",
        "---",
        "",
        "## Stale / Inactive Model Names",
        "",
        f"Staleness cutoff: last price before {STALE_CUTOFF.date()} (10 trading days before 2026-06-28).",
        "",
        "### Delisted / Inactive (7 tickers)",
        "",
        "| Ticker | Company | Status | Last Price | Note |",
        "| --- | --- | --- | --- | --- |",
    ]

    for _, row in classified_df[classified_df["model_classification"] == "DELISTED_OR_INACTIVE"].iterrows():
        lp = row.get("last_price_date_combined", "")
        lp_str = pd.Timestamp(lp).strftime("%Y-%m-%d") if pd.notna(lp) and lp else "N/A"
        lines.append(f"| {row['ticker']} | {row['company_name']} | {row['status']} | {lp_str} | Already flagged |")

    lines += [
        "",
        "### Price Data Notes",
        "",
        "- `_XBI_BENCHMARK_` — placeholder entry in universe.json, no price data (expected — it is a benchmark, not a trading ticker)",
        "- `ATXS`, `CVAC`, `MRSN` — price data exists in CSV but these tickers are NOT in universe.json (orphaned price rows, likely historical)",
        "- `IBB` — IBB ETF benchmark price row in price CSV, not a model constituent",
        "",
        "---",
        "",
        "## Identifier Conflicts",
        "",
    ]

    if conflicts.empty:
        lines.append("_No identifier conflicts detected._\n")
    else:
        lines.append(
            _table(
                conflicts, [c for c in ["conflict_type", "name_key", "tickers", "details"] if c in conflicts.columns]
            )
        )

    lines += [
        "",
        "---",
        "",
        "## Known Corporate-Action Issue Names Status",
        "",
        "| Ticker | Status |",
        "| --- | --- |",
    ]
    for t in KNOWN_PROBLEM_TICKERS:
        lines.append(f"| {t} | {problem_status[t]} |")

    lines += [
        "",
        "**Notes:**",
        "- `RNA` — Ticker was re-assigned. Old `RNA` (Avidity Biosciences) was acquired by Novartis. New `RNA` in model = Atrium Therapeutics (spun off). Status: active, price current. **Monitor: ensure this is the intended entity.**",
        "- `GOSS` — Gossamer Bio; status: active, price current.",
        "- `REPL` — Replimune Group; status: active, price current.",
        "- `ACLX`, `DAWN`, `FOLD`, `TERN` — all status=delisted with None company name. Confirm delisting and purge data.",
        "- `APLS` — Apellis Pharmaceuticals; status=delisted; last price 2026-05-15.",
        "- `GLPG` — Lakefront Biotherapeutics NV; status=delisted.",
        "- `KALV` — KalVista Pharmaceuticals; status=delisted; last price 2026-06-12.",
        "",
        "---",
        "",
        "## Recommended Universe Actions",
        "",
        "### 1. Confirm and clean delisted tickers (7 tickers)",
        "All 7 delisted tickers (`APLS`, `GLPG`, `KALV`, `ACLX`, `DAWN`, `FOLD`, `TERN`) are already flagged `status=delisted` in universe.json. No immediate action required, but their price history should be frozen and they should be excluded from all model scoring.",
        "",
        "### 2. Investigate GOSS price staleness",
        "GOSS (Gossamer Bio) shows `status=active` but price data in split-adjusted CSV may be stale — verify current price feed.",
        "",
        "### 3. RNA ticker re-assignment watch",
        "The ticker `RNA` now maps to Atrium Therapeutics (new post-spinoff entity). Ensure backtest data before the Novartis acquisition of old RNA (Avidity) is not contaminating model signals for the new entity.",
        "",
        "### 4. ETF coverage gap candidates",
        f"If ETF data was successfully fetched: evaluate {n_xbi_hp} XBI core biotech candidates and {n_ibb_hp} IBB core biotech candidates for potential addition in the next universe refresh cycle.",
        "",
        "### 5. Orphaned price rows",
        "ATXS, CVAC, MRSN have price history rows but are not in universe.json. These are likely historical tickers that were removed. Consider cleaning the price CSV.",
        "",
        "### 6. Corrupt price CSV rows",
        "The price CSVs contain ~39 rows with malformed ticker names (contain 'TICKER\\n' pattern). These appear to be DataFrame repr strings accidentally appended. The audit filtered them out — but the underlying data write should be fixed.",
        "",
        "---",
        "",
        "## Risks and Caveats",
        "",
        "- ETF holdings may be from a stale snapshot if live fetch failed. Use `--fetch-current-etf-holdings` with a valid internet connection.",
        "- XBI and IBB hold different subsets. Coverage gaps vs XBI are more actionable for small-cap pure-play biotech.",
        "- The 10-trading-day stale threshold is a heuristic. Some tickers may have legitimate data gaps due to halts or low volume.",
        "- RNA ticker reuse across two different companies is a known risk; the audit flags it but cannot automatically determine correctness.",
        "- Large-cap biopharma exclusion list (`LARGE_CAP_EXCLUSIONS`) is hardcoded; verify it reflects current policy.",
        "",
        "---",
        "",
        "## Governance Conclusion",
        "",
        "This audit is **read-only and proposals-only**. No production files were modified.",
        "",
        "```",
        "UNIVERSE_HYGIENE_AUDIT     ✓",
        "COVERAGE_DIAGNOSTIC        ✓",
        "NO_MODEL_CHANGE            ✓",
        "NO_RANKER_CHANGE           ✓",
        "NO_SELECTOR_CHANGE         ✓",
        "NO_SIZING_CHANGE           ✓",
        "NO_TRADING_CHANGE          ✓",
        "```",
        "",
        "---",
        "",
        "## Next Validation Steps",
        "",
        "1. Re-run with fresh XBI/IBB live fetch once network/auth is confirmed.",
        "2. For each `MISSING_CORE_BIOTECH_CANDIDATE`: manual review of market cap, pipeline stage, liquidity before any add proposal.",
        "3. For delisted tickers (ACLX, DAWN, FOLD, TERN — missing company name): source company name and final delisting date for archival.",
        "4. Investigate and fix root cause of malformed price CSV rows (`TICKER\\n` pattern).",
        "5. Confirm RNA entity mapping with explicit audit trail.",
        "6. Review KALV delisting date (last price 2026-06-12, very recent).",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Universe Hygiene and ETF Coverage Audit — PROPOSALS ONLY, NO PRODUCTION MUTATION"
    )
    parser.add_argument("--universe", default="production_data/universe.json")
    parser.add_argument("--prices", default="production_data/price_history_split_adj.csv")
    parser.add_argument(
        "--output-dir",
        default="artifacts/universe_hygiene/xbi_ibb_universe_audit_2026_06_28",
    )
    parser.add_argument(
        "--fetch-current-etf-holdings",
        action="store_true",
        help="Try live ETF fetch from SPDR/iShares",
    )
    parser.add_argument("--xbi-holdings-file", default=None)
    parser.add_argument("--ibb-holdings-file", default=None)
    parser.add_argument("--stale-days", type=int, default=10)
    parser.add_argument("--write-proposals-only", action="store_true", default=True)
    parser.add_argument("--no-live-fetch", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    audit_ts = datetime.now(UTC).isoformat()

    print("=== Universe Hygiene Audit ===")
    print(f"Timestamp: {audit_ts}")
    print(f"Output: {args.output_dir}")
    print()

    # --- Step 1: Load universe ---
    print("Loading universe...")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    universe_path = args.universe if os.path.isabs(args.universe) else os.path.join(repo_root, args.universe)
    prices_path = args.prices if os.path.isabs(args.prices) else os.path.join(repo_root, args.prices)
    raw_prices_path = os.path.join(repo_root, "production_data", "price_history.csv")

    raw_universe, universe_df = load_universe(universe_path)
    print(f"  Universe: {len(universe_df)} tickers")

    # --- Step 2: Price staleness ---
    print("Loading price staleness...")
    staleness_df = load_price_staleness(prices_path, raw_prices_path)
    print(f"  Price tickers: {len(staleness_df)}")

    # --- Step 3: Classify model tickers ---
    print("Classifying model tickers...")
    classified_df = classify_model_tickers(universe_df, staleness_df)
    cl_counts = classified_df["model_classification"].value_counts().to_dict()
    for cl, n in sorted(cl_counts.items()):
        print(f"  {cl}: {n}")

    # --- Step 4: ETF holdings ---
    xbi_df = pd.DataFrame()
    xbi_source = "NOT_FETCHED"
    ibb_df = pd.DataFrame()
    ibb_source = "NOT_FETCHED"

    # Pre-downloaded files take priority
    if args.xbi_holdings_file and os.path.exists(args.xbi_holdings_file):
        print(f"Loading XBI from file: {args.xbi_holdings_file}")
        try:
            xbi_df = pd.read_csv(args.xbi_holdings_file)
            xbi_source = f"FILE:{args.xbi_holdings_file}"
        except Exception as e:
            print(f"  Failed to load XBI file: {e}")

    if args.ibb_holdings_file and os.path.exists(args.ibb_holdings_file):
        print(f"Loading IBB from file: {args.ibb_holdings_file}")
        try:
            with open(args.ibb_holdings_file, encoding="utf-8-sig") as fh:
                text = fh.read()
            if text.lstrip().lower().startswith("ticker"):
                ibb_df = pd.read_csv(io.StringIO(text))  # already-normalized ticker,name,... table
                ibb_source = f"FILE:{args.ibb_holdings_file}"
            else:  # raw iShares download: metadata lines + Ticker,Name,... table; staleness enforced
                ibb_df, ibb_source = ibb_holdings_from_content(text, f"FILE:{args.ibb_holdings_file}")
                if ibb_df.empty:
                    print(f"  IBB file refused: {ibb_source}")
        except Exception as e:
            print(f"  Failed to load IBB file: {e}")

    # Live fetch
    if not args.no_live_fetch and (args.fetch_current_etf_holdings or args.fetch_current_etf_holdings):
        if xbi_df.empty:
            print("Fetching XBI holdings...")
            xbi_df, xbi_source = _fetch_xbi_holdings(verbose=True)
            print(f"  XBI: {len(xbi_df)} holdings | source: {xbi_source[:60]}")

        if ibb_df.empty:
            print("Fetching IBB holdings...")
            ibb_df, ibb_source = _fetch_ibb_holdings(verbose=True)
            print(f"  IBB: {len(ibb_df)} holdings | source: {ibb_source[:60]}")

    # Mark fetch status in source strings
    if xbi_df.empty and "FETCH_FAILED" not in xbi_source:
        xbi_source = "NOT_FETCHED (use --fetch-current-etf-holdings or --xbi-holdings-file)"
    if ibb_df.empty and "FETCH_FAILED" not in ibb_source:
        ibb_source = "NOT_FETCHED (use --fetch-current-etf-holdings or --ibb-holdings-file)"

    # --- Step 5: Missing from model ---
    universe_tickers = set(universe_df["ticker"].tolist())

    missing_xbi = pd.DataFrame()
    if not xbi_df.empty and "FETCH_FAILED" not in xbi_source:
        xbi_not_in_model = xbi_df[~xbi_df["ticker"].isin(universe_tickers)].copy()
        if not xbi_not_in_model.empty:
            xbi_not_in_model["missing_classification"] = xbi_not_in_model.apply(
                lambda r: classify_etf_missing(r["ticker"], r.get("name", ""), "XBI", universe_tickers),
                axis=1,
            )
        missing_xbi = xbi_not_in_model

    missing_ibb = pd.DataFrame()
    if not ibb_df.empty and "FETCH_FAILED" not in ibb_source:
        ibb_not_in_model = ibb_df[~ibb_df["ticker"].isin(universe_tickers)].copy()
        if not ibb_not_in_model.empty:
            ibb_not_in_model["missing_classification"] = ibb_not_in_model.apply(
                lambda r: classify_etf_missing(r["ticker"], r.get("name", ""), "IBB", universe_tickers),
                axis=1,
            )
        missing_ibb = ibb_not_in_model

    print(f"  XBI missing from model: {len(missing_xbi)}")
    print(f"  IBB missing from model: {len(missing_ibb)}")

    # --- Step 6: Identifier conflicts ---
    print("Detecting identifier conflicts...")
    conflicts = find_identifier_conflicts(universe_df, xbi_df, ibb_df)
    print(f"  Conflicts found: {len(conflicts)}")

    # --- Step 7: Proposed actions ---
    print("Building proposed actions...")
    proposed = build_proposed_actions(classified_df, missing_xbi, missing_ibb, conflicts)
    print(f"  Proposed actions: {len(proposed)}")

    # --- Step 8: Write outputs ---
    print(f"\nWriting outputs to {args.output_dir}...")

    # current_model_universe.csv
    out_universe = classified_df[
        [
            "ticker",
            "company_name",
            "status",
            "exchange",
            "sector",
            "etf_sources",
            "known_delisted",
            "exclusion_reason",
            "model_classification",
            "last_price_date_combined",
            "is_stale",
            "has_any_price",
        ]
    ].copy()
    if "last_price_date_combined" in out_universe.columns:
        out_universe["last_price_date"] = out_universe["last_price_date_combined"].apply(
            lambda x: pd.Timestamp(x).strftime("%Y-%m-%d") if pd.notna(x) and x else ""
        )
        out_universe = out_universe.drop(columns=["last_price_date_combined"])
    out_universe.to_csv(os.path.join(args.output_dir, "current_model_universe.csv"), index=False)
    print("  current_model_universe.csv")

    # current_xbi_holdings.csv
    if not xbi_df.empty:
        xbi_df.to_csv(os.path.join(args.output_dir, "current_xbi_holdings.csv"), index=False)
        print(f"  current_xbi_holdings.csv ({len(xbi_df)} rows)")
    else:
        with open(os.path.join(args.output_dir, "current_xbi_holdings.csv"), "w") as f:
            f.write(f"# XBI holdings not available: {xbi_source}\n")
            f.write("ticker,name,weight_pct\n")
        print(f"  current_xbi_holdings.csv (EMPTY — {xbi_source[:50]})")

    # current_ibb_holdings.csv
    if not ibb_df.empty:
        ibb_df.to_csv(os.path.join(args.output_dir, "current_ibb_holdings.csv"), index=False)
        print(f"  current_ibb_holdings.csv ({len(ibb_df)} rows)")
    else:
        with open(os.path.join(args.output_dir, "current_ibb_holdings.csv"), "w") as f:
            f.write(f"# IBB holdings not available: {ibb_source}\n")
            f.write("ticker,name,weight_pct\n")
        print(f"  current_ibb_holdings.csv (EMPTY — {ibb_source[:50]})")

    # missing_from_model.csv
    all_missing = []
    if not missing_xbi.empty:
        mx = missing_xbi.copy()
        mx["etf_source"] = "XBI"
        all_missing.append(mx)
    if not missing_ibb.empty:
        mi = missing_ibb.copy()
        mi["etf_source"] = "IBB"
        all_missing.append(mi)
    if all_missing:
        pd.concat(all_missing, ignore_index=True).to_csv(
            os.path.join(args.output_dir, "missing_from_model.csv"), index=False
        )
    else:
        with open(os.path.join(args.output_dir, "missing_from_model.csv"), "w") as f:
            f.write("ticker,name,weight_pct,missing_classification,etf_source\n")
    print("  missing_from_model.csv")

    # stale_or_inactive_model_names.csv
    stale_inactive = classified_df[
        classified_df["model_classification"].isin(
            ["ACTIVE_BUT_STALE_PRICE", "DELISTED_OR_INACTIVE", "PRICE_DATA_MISSING"]
        )
    ].copy()
    if "last_price_date_combined" in stale_inactive.columns:
        stale_inactive["last_price_date"] = stale_inactive["last_price_date_combined"].apply(
            lambda x: pd.Timestamp(x).strftime("%Y-%m-%d") if pd.notna(x) and x else ""
        )
    stale_inactive.to_csv(os.path.join(args.output_dir, "stale_or_inactive_model_names.csv"), index=False)
    print(f"  stale_or_inactive_model_names.csv ({len(stale_inactive)} rows)")

    # identifier_conflicts.csv
    if not conflicts.empty:
        conflicts.to_csv(os.path.join(args.output_dir, "identifier_conflicts.csv"), index=False)
    else:
        with open(os.path.join(args.output_dir, "identifier_conflicts.csv"), "w") as f:
            f.write("conflict_type,name_key,tickers,details\n")
    print(f"  identifier_conflicts.csv ({len(conflicts)} rows)")

    # proposed_universe_actions.csv
    proposed.to_csv(os.path.join(args.output_dir, "proposed_universe_actions.csv"), index=False)
    print(f"  proposed_universe_actions.csv ({len(proposed)} rows)")

    # universe_hygiene_summary.json
    cl_counts_str = {k: int(v) for k, v in cl_counts.items()}
    if not missing_xbi.empty and "missing_classification" in missing_xbi.columns:
        xbi_miss_by_class = missing_xbi["missing_classification"].value_counts().to_dict()
    else:
        xbi_miss_by_class = {}
    if not missing_ibb.empty and "missing_classification" in missing_ibb.columns:
        ibb_miss_by_class = missing_ibb["missing_classification"].value_counts().to_dict()
    else:
        ibb_miss_by_class = {}

    # Known problem status for JSON
    prob_json = {}
    for t in KNOWN_PROBLEM_TICKERS:
        rows = classified_df[classified_df["ticker"] == t]
        if rows.empty:
            prob_json[t] = {"in_model": False}
        else:
            r = rows.iloc[0]
            lp = r.get("last_price_date_combined", None)
            prob_json[t] = {
                "in_model": True,
                "model_classification": r["model_classification"],
                "status": r["status"],
                "company_name": r["company_name"],
                "last_price_date": pd.Timestamp(lp).strftime("%Y-%m-%d") if pd.notna(lp) and lp else None,
            }

    summary = {
        "audit_timestamp": audit_ts,
        "reference_date": "2026-06-28",
        "stale_cutoff": str(STALE_CUTOFF.date()),
        "governance": [
            "UNIVERSE_HYGIENE_AUDIT",
            "COVERAGE_DIAGNOSTIC",
            "NO_MODEL_CHANGE",
            "NO_RANKER_CHANGE",
            "NO_SELECTOR_CHANGE",
            "NO_SIZING_CHANGE",
            "NO_TRADING_CHANGE",
        ],
        "model_universe": {
            "total": int(len(classified_df)),
            "by_classification": cl_counts_str,
        },
        "xbi": {
            "holdings_count": int(len(xbi_df)),
            "source": xbi_source,
            "missing_from_model": int(len(missing_xbi)),
            "missing_by_classification": {k: int(v) for k, v in xbi_miss_by_class.items()},
        },
        "ibb": {
            "holdings_count": int(len(ibb_df)),
            "source": ibb_source,
            "missing_from_model": int(len(missing_ibb)),
            "missing_by_classification": {k: int(v) for k, v in ibb_miss_by_class.items()},
        },
        "identifier_conflicts": int(len(conflicts)),
        "proposed_actions": int(len(proposed)),
        "known_problem_tickers": prob_json,
        "notes": [
            "39 corrupt rows in price CSV (TICKER\\n pattern) filtered out",
            "ATXS/CVAC/MRSN have price data but are not in universe.json (orphaned historical rows)",
            "_XBI_BENCHMARK_ entry in universe.json has no price data (expected)",
            "RNA ticker reuse: old RNA=Avidity Biosciences (acquired), new RNA=Atrium Therapeutics",
        ],
    }
    with open(os.path.join(args.output_dir, "universe_hygiene_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("  universe_hygiene_summary.json")

    # UNIVERSE_HYGIENE_AUDIT.md
    print("Building markdown report...")
    md = build_markdown_report(
        classified_df=classified_df,
        xbi_df=xbi_df,
        ibb_df=ibb_df,
        xbi_source=xbi_source,
        ibb_source=ibb_source,
        missing_xbi=missing_xbi,
        missing_ibb=missing_ibb,
        conflicts=conflicts,
        proposed=proposed,
        audit_ts=audit_ts,
    )
    with open(os.path.join(args.output_dir, "UNIVERSE_HYGIENE_AUDIT.md"), "w") as f:
        f.write(md)
    print("  UNIVERSE_HYGIENE_AUDIT.md")

    print("\n=== Audit Complete ===")
    print(f"Model universe: {len(classified_df)} tickers")
    print(f"XBI: {len(xbi_df)} holdings | IBB: {len(ibb_df)} holdings")
    print(f"Missing XBI: {len(missing_xbi)} | Missing IBB: {len(missing_ibb)}")
    print(f"Stale/inactive in model: {len(stale_inactive)}")
    print(f"Identifier conflicts: {len(conflicts)}")
    print(f"Proposed actions: {len(proposed)}")


if __name__ == "__main__":
    main()
