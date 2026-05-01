#!/usr/bin/env python3
"""Fetch Form 4 insider transaction data from SEC EDGAR.

Downloads and parses Form 4 XML filings for all tickers in the universe.
Uses EDGAR acceptance timestamp (not transaction date) as the PIT-safe signal date.

Output:
    data/form4/raw/{TICKER}.json         — raw parsed transactions per ticker
    data/form4/form4_panel.csv           — flat panel for research harness
    data/form4/fetch_state.json          — last-fetched metadata

Usage:
    python3 tools/fetch_form4_insider.py
    python3 tools/fetch_form4_insider.py --tickers RVMD KURA
    python3 tools/fetch_form4_insider.py --since 2020-01-01
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# EDGAR config
USER_AGENT = "Wake Robin Capital Management institutional.validation@wakerobincapital.com"
EFTS_BASE = "https://efts.sec.gov/LATEST/search-index"
SUBMISSIONS_BASE = "https://data.sec.gov/submissions"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
REQUEST_DELAY = 0.11  # ~9 req/s, under SEC 10/s limit
MAX_WORKERS = 8  # concurrent ticker fetches

# Global rate limiter: ensures total requests across all threads stay under SEC limit
_rate_lock = threading.Lock()
_last_request_time = 0.0
_MIN_REQUEST_INTERVAL = 0.11  # ~9 req/s max across all threads (under SEC 10/s limit)

RAW_DIR = PROJECT_ROOT / "data" / "form4" / "raw"
PANEL_CSV = PROJECT_ROOT / "data" / "form4" / "form4_panel.csv"
STATE_FILE = PROJECT_ROOT / "data" / "form4" / "fetch_state.json"


# ---------------------------------------------------------------------------
# Producer state bookkeeping (Spec 065 stable-snapshot gate)
# ---------------------------------------------------------------------------
#
# State file schema (post 2026-05-01 operational repair):
#   last_attempt        ISO timestamp — producer run started (any mode)
#   last_success        ISO timestamp — producer run completed without error
#   last_fetch          alias of last_success (backward compat)
#   last_new_filing     ISO timestamp — most recent successful run that found
#                       at least one new accession (None until first observed)
#   last_error          {"at": ISO, "msg": str} when most recent run errored;
#                       cleared on next successful run
#   last_panel_rebuild  ISO timestamp — panel CSV last rebuilt
#   panel_rows          int — n_rows in last panel rebuild
#   schema_fingerprint  12-char sha256 prefix of InsiderTransaction field list
#                       (criterion #2 — detect producer-side schema drift)
#   tickers_checked,
#   tickers_updated,
#   new_transactions,
#   failed_tickers,
#   since               from most recent fetch run
#
# Update policy:
#   - last_attempt is written at the START of every run (resilient to timeout)
#   - last_success is written ONLY at the end of an unwound, error-free run
#   - last_panel_rebuild is written at the END of a successful panel rebuild
#   - All writes preserve other state fields via _state_load() merge
#


def _state_load() -> Dict[str, Any]:
    """Load existing fetch_state.json, returning {} if missing/corrupt."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _state_write(updates: Dict[str, Any]) -> None:
    """Atomically merge updates into fetch_state.json (preserves other keys)."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state = _state_load()
    state.update(updates)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _schema_fingerprint() -> str:
    """12-char hash of InsiderTransaction dataclass field list. Changes when
    the producer-side schema changes (new key, removed key, renamed key).
    Used to detect producer-side schema drift per Spec 065 §1 #2."""
    import hashlib

    keys = sorted(InsiderTransaction.__dataclass_fields__.keys())
    return hashlib.sha256("|".join(keys).encode()).hexdigest()[:12]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class InsiderTransaction:
    """A single insider transaction from Form 4."""

    ticker: str
    cik: str  # issuer CIK
    filer_cik: str  # reporting person CIK
    filer_name: str
    filing_date: str  # EDGAR acceptance date (PIT-safe)
    transaction_date: str  # actual transaction date (NOT used for signal timing)
    form_type: str  # "4", "4/A"
    accession_number: str

    # Transaction details
    is_director: bool = False
    is_officer: bool = False
    officer_title: str = ""
    is_ten_pct_owner: bool = False

    # Securities
    security_title: str = ""
    transaction_code: str = ""  # P=purchase, S=sale, A=award, M=exercise, etc.
    shares: float = 0.0
    price_per_share: float = 0.0
    value: float = 0.0  # shares * price
    acquired_disposed: str = ""  # A=acquired, D=disposed
    shares_owned_after: float = 0.0
    direct_indirect: str = ""  # D=direct, I=indirect

    # Derived
    is_buy: bool = False
    is_sell: bool = False
    is_derivative: bool = False


# ---------------------------------------------------------------------------
# EDGAR API helpers
# ---------------------------------------------------------------------------


def _fetch_url(url: str) -> bytes:
    """Fetch URL with global rate limiting and SEC-compliant headers."""
    global _last_request_time
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    # Global rate limiter: serialize the timing check across threads
    with _rate_lock:
        now = time.monotonic()
        wait = _MIN_REQUEST_INTERVAL - (now - _last_request_time)
        if wait > 0:
            time.sleep(wait)
        _last_request_time = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 429:
            log.warning("Rate limited, sleeping 10s...")
            time.sleep(10)
            return _fetch_url(url)
        raise


def _fetch_json(url: str) -> Any:
    return json.loads(_fetch_url(url))


def get_form4_filings(cik: str, since: str = "2020-01-01") -> List[Dict[str, Any]]:
    """Get Form 4 filing metadata for a CIK from EDGAR submissions API."""
    cik_padded = cik.lstrip("0").zfill(10)
    url = f"{SUBMISSIONS_BASE}/CIK{cik_padded}.json"

    try:
        data = _fetch_json(url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            log.debug(f"CIK {cik} not found in EDGAR")
            return []
        raise

    filings = []
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    for i, form in enumerate(forms):
        if form not in ("4", "4/A"):
            continue
        filing_date = dates[i] if i < len(dates) else ""
        if filing_date < since:
            continue
        filings.append(
            {
                "form": form,
                "filingDate": filing_date,
                "accessionNumber": accessions[i] if i < len(accessions) else "",
                "primaryDocument": primary_docs[i] if i < len(primary_docs) else "",
            }
        )

    return filings


def parse_form4_xml(
    xml_bytes: bytes, ticker: str, cik: str, filing_date: str, accession: str
) -> List[InsiderTransaction]:
    """Parse a Form 4 XML document into transactions."""
    txns = []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        log.warning(f"XML parse error for {ticker} {accession}")
        return []

    # Namespace handling — Form 4 XML can have various namespace prefixes
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    # Reporting person info
    rp = root.find(f".//{ns}reportingOwner")
    filer_cik = ""
    filer_name = ""
    is_director = False
    is_officer = False
    officer_title = ""
    is_ten_pct = False

    if rp is not None:
        rid = rp.find(f"{ns}reportingOwnerId")
        if rid is not None:
            filer_cik = (rid.findtext(f"{ns}rptOwnerCik") or "").strip()
            filer_name = (rid.findtext(f"{ns}rptOwnerName") or "").strip()

        rel = rp.find(f"{ns}reportingOwnerRelationship")
        if rel is not None:
            is_director = (rel.findtext(f"{ns}isDirector") or "").strip().lower() in ("1", "true")
            is_officer = (rel.findtext(f"{ns}isOfficer") or "").strip().lower() in ("1", "true")
            officer_title = (rel.findtext(f"{ns}officerTitle") or "").strip()
            is_ten_pct = (rel.findtext(f"{ns}isTenPercentOwner") or "").strip().lower() in ("1", "true")

    # Non-derivative transactions
    for txn_el in root.findall(f".//{ns}nonDerivativeTransaction"):
        txn = _parse_transaction_element(
            txn_el,
            ns,
            ticker,
            cik,
            filer_cik,
            filer_name,
            filing_date,
            accession,
            is_director,
            is_officer,
            officer_title,
            is_ten_pct,
            is_derivative=False,
        )
        if txn:
            txns.append(txn)

    # Derivative transactions
    for txn_el in root.findall(f".//{ns}derivativeTransaction"):
        txn = _parse_transaction_element(
            txn_el,
            ns,
            ticker,
            cik,
            filer_cik,
            filer_name,
            filing_date,
            accession,
            is_director,
            is_officer,
            officer_title,
            is_ten_pct,
            is_derivative=True,
        )
        if txn:
            txns.append(txn)

    return txns


def _parse_transaction_element(
    el,
    ns: str,
    ticker: str,
    cik: str,
    filer_cik: str,
    filer_name: str,
    filing_date: str,
    accession: str,
    is_director: bool,
    is_officer: bool,
    officer_title: str,
    is_ten_pct: bool,
    is_derivative: bool,
) -> Optional[InsiderTransaction]:
    """Parse a single transaction element."""
    sec_title = (el.findtext(f".//{ns}securityTitle/{ns}value") or "").strip()

    # Transaction coding (contains transactionCode)
    coding = el.find(f"{ns}transactionCoding")
    code_el = ""
    if coding is not None:
        code_el = (coding.findtext(f"{ns}transactionCode") or "").strip()

    # Transaction amounts
    amounts = el.find(f"{ns}transactionAmounts")
    if amounts is None:
        return None

    shares_str = amounts.findtext(f"{ns}transactionShares/{ns}value") or "0"
    price_str = amounts.findtext(f"{ns}transactionPricePerShare/{ns}value") or "0"
    ad = (amounts.findtext(f"{ns}transactionAcquiredDisposedCode/{ns}value") or "").strip()

    # Transaction date
    txn_date = (el.findtext(f".//{ns}transactionDate/{ns}value") or "").strip()

    # Post-transaction ownership
    ownership = el.find(f"{ns}postTransactionAmounts")
    shares_after = 0.0
    if ownership is not None:
        shares_after_str = ownership.findtext(f"{ns}sharesOwnedFollowingTransaction/{ns}value") or "0"
        try:
            shares_after = float(shares_after_str)
        except ValueError:
            pass

    # Ownership nature
    own_nature = el.find(f"{ns}ownershipNature")
    di = ""
    if own_nature is not None:
        di = (own_nature.findtext(f"{ns}directOrIndirectOwnership/{ns}value") or "").strip()

    try:
        shares = float(shares_str)
    except ValueError:
        shares = 0.0
    try:
        price = float(price_str)
    except ValueError:
        price = 0.0

    value = shares * price
    is_buy = code_el in ("P",) and ad == "A"  # Open market purchase
    is_sell = code_el in ("S",) and ad == "D"  # Open market sale

    return InsiderTransaction(
        ticker=ticker,
        cik=cik,
        filer_cik=filer_cik,
        filer_name=filer_name,
        filing_date=filing_date,
        transaction_date=txn_date,
        form_type="4",
        accession_number=accession,
        is_director=is_director,
        is_officer=is_officer,
        officer_title=officer_title,
        is_ten_pct_owner=is_ten_pct,
        security_title=sec_title,
        transaction_code=code_el,
        shares=shares,
        price_per_share=price,
        value=value,
        acquired_disposed=ad,
        shares_owned_after=shares_after,
        direct_indirect=di,
        is_buy=is_buy,
        is_sell=is_sell,
        is_derivative=is_derivative,
    )


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

EXEC_TITLES = {
    "ceo",
    "cfo",
    "coo",
    "president",
    "chief executive",
    "chief financial",
    "chief operating",
    "chief medical",
    "chief scientific",
}


def is_executive(title: str) -> bool:
    """Check if officer title indicates C-suite/president."""
    t = title.lower()
    return any(ex in t for ex in EXEC_TITLES)


def compute_insider_features(
    txns: List[InsiderTransaction],
    as_of_date: str,
    windows: tuple[int, ...] = (30, 60, 90),
) -> Dict[str, Any]:
    """Compute insider signal features as of a date using filing_date (PIT-safe).

    Only considers transactions where filing_date <= as_of_date.
    """
    from datetime import datetime as dt

    as_of = dt.strptime(as_of_date, "%Y-%m-%d").date()
    features: Dict[str, Any] = {}

    for window in windows:
        cutoff = as_of - __import__("datetime").timedelta(days=window)
        window_txns = [
            t for t in txns if t.filing_date and cutoff <= dt.strptime(t.filing_date, "%Y-%m-%d").date() <= as_of
        ]

        buys = [t for t in window_txns if t.is_buy and not t.is_derivative]
        sells = [t for t in window_txns if t.is_sell and not t.is_derivative]

        buy_value = sum(t.value for t in buys)
        sell_value = sum(t.value for t in sells)
        buy_shares = sum(t.shares for t in buys)
        sell_shares = sum(t.shares for t in sells)

        # Unique buyers/sellers
        buyers = {t.filer_cik for t in buys}
        sellers = {t.filer_cik for t in sells}

        # Executive buys
        exec_buys = [t for t in buys if t.is_officer and is_executive(t.officer_title)]

        sfx = f"_{window}d"
        features[f"insider_net_buy_value{sfx}"] = buy_value - sell_value
        features[f"insider_buy_value{sfx}"] = buy_value
        features[f"insider_sell_value{sfx}"] = sell_value
        features[f"insider_net_buy_shares{sfx}"] = buy_shares - sell_shares
        features[f"insider_buy_count{sfx}"] = len(buys)
        features[f"insider_sell_count{sfx}"] = len(sells)
        features[f"insider_net_buyer_flag{sfx}"] = 1 if buy_value > sell_value else 0
        features[f"insider_buying_by_exec_flag{sfx}"] = 1 if exec_buys else 0
        features[f"insider_cluster_buy_flag{sfx}"] = 1 if len(buyers) >= 2 else 0
        features[f"insider_unique_buyers{sfx}"] = len(buyers)
        features[f"insider_unique_sellers{sfx}"] = len(sellers)
        features[f"insider_exec_buy_value{sfx}"] = sum(t.value for t in exec_buys)

    return features


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def _find_raw_xml_url(cik: str, acc_path: str) -> Optional[str]:
    """Find the raw ownership XML URL from the filing index."""
    cik_clean = cik.lstrip("0")
    index_url = f"{ARCHIVES_BASE}/{cik_clean}/{acc_path}/index.json"
    try:
        data = _fetch_json(index_url)
    except Exception:
        return None

    for item in data.get("directory", {}).get("item", []):
        name = item.get("name", "")
        # Raw XML files: ownership.xml, doc4.xml, primary_doc.xml, etc.
        if name.endswith(".xml") and "xsl" not in name.lower() and "R" not in name:
            return f"{ARCHIVES_BASE}/{cik_clean}/{acc_path}/{name}"
    return None


def fetch_ticker(
    ticker: str,
    cik: str,
    since: str,
    seen_accessions: Optional[set] = None,
) -> List[InsiderTransaction]:
    """Fetch and parse Form 4 filings for a ticker.

    If ``seen_accessions`` is provided, filings with those accession numbers are
    skipped before the expensive XML fetch. Returns only transactions belonging
    to accessions NOT in ``seen_accessions`` (incremental refresh).
    """
    filings = get_form4_filings(cik, since=since)
    if seen_accessions:
        total = len(filings)
        filings = [f for f in filings if f.get("accessionNumber") not in seen_accessions]
        log.info(
            f"  {ticker} (CIK {cik}): {total} Form 4 filings since {since}, "
            f"{len(filings)} new (skipped {total - len(filings)} already in raw)"
        )
    else:
        log.info(f"  {ticker} (CIK {cik}): {len(filings)} Form 4 filings since {since}")

    all_txns = []
    for f in filings:
        acc = f["accessionNumber"]
        acc_path = acc.replace("-", "")
        primary = f.get("primaryDocument", "")

        # Try direct raw XML first (strip XSL prefix if present)
        cik_clean = cik.lstrip("0")
        if "/" in primary:
            # e.g. "xslF345X06/ownership.xml" -> try "ownership.xml" directly
            base_name = primary.split("/")[-1]
            url = f"{ARCHIVES_BASE}/{cik_clean}/{acc_path}/{base_name}"
        elif primary:
            url = f"{ARCHIVES_BASE}/{cik_clean}/{acc_path}/{primary}"
        else:
            url = None

        xml_bytes = None
        if url:
            try:
                xml_bytes = _fetch_url(url)
                # Verify it's actual XML, not HTML
                if xml_bytes[:50].lstrip().startswith(b"<!") or b"<html" in xml_bytes[:200].lower():
                    xml_bytes = None  # HTML, need to find raw XML
            except Exception:
                xml_bytes = None

        # Fallback: look up filing index for raw XML
        if xml_bytes is None:
            raw_url = _find_raw_xml_url(cik, acc_path)
            if raw_url:
                try:
                    xml_bytes = _fetch_url(raw_url)
                except Exception as e:
                    log.warning(f"    Failed to fetch {acc}: {e}")
                    continue
            else:
                log.debug(f"    No raw XML found for {acc}")
                continue

        txns = parse_form4_xml(xml_bytes, ticker, cik, f["filingDate"], acc)
        all_txns.extend(txns)

    return all_txns


def _merge_transactions(
    existing: List[InsiderTransaction],
    newly_fetched: List[InsiderTransaction],
) -> List[InsiderTransaction]:
    """Merge newly-fetched transactions onto existing raw and sort deterministically.

    ``newly_fetched`` is assumed to come from accessions not present in
    ``existing`` (caller filtered upstream via ``seen_accessions``). Sort key:
    (filing_date, accession_number, transaction_date, shares, price_per_share).
    """
    merged = list(existing) + list(newly_fetched)
    merged.sort(
        key=lambda t: (
            t.filing_date or "",
            t.accession_number or "",
            t.transaction_date or "",
            t.shares,
            t.price_per_share,
        )
    )
    return merged


def _load_existing_raw(raw_file: Path) -> List[InsiderTransaction]:
    """Load a raw/{ticker}.json back into InsiderTransaction objects; [] if missing."""
    if not raw_file.exists():
        return []
    try:
        data = json.loads(raw_file.read_text())
    except json.JSONDecodeError:
        log.warning(f"  {raw_file.name}: raw JSON corrupt; treating as empty")
        return []
    return [InsiderTransaction(**t) for t in data]


def build_panel(raw_dir: Path, panel_path: Path) -> int:
    """Build flat CSV panel from raw JSON files."""
    import csv

    rows = []
    for jf in sorted(raw_dir.glob("*.json")):
        ticker = jf.stem
        data = json.loads(jf.read_text())
        txns = [InsiderTransaction(**t) for t in data]

        # Compute features at each unique filing date
        filing_dates = sorted({t.filing_date for t in txns if t.filing_date})
        for fd in filing_dates:
            feats = compute_insider_features(txns, fd)
            feats["ticker"] = ticker
            feats["as_of_date"] = fd
            rows.append(feats)

    if not rows:
        log.warning("No data for panel")
        return 0

    # Write CSV
    all_keys = sorted({k for r in rows for k in r.keys()})
    # Ensure ticker and as_of_date are first
    cols = ["ticker", "as_of_date"] + [k for k in all_keys if k not in ("ticker", "as_of_date")]
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    with open(panel_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["ticker"], x["as_of_date"])):
            w.writerow(r)

    log.info(f"Panel: {len(rows)} rows, {len(cols)} columns -> {panel_path}")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Fetch Form 4 insider data from SEC EDGAR")
    parser.add_argument("--tickers", nargs="*", help="Specific tickers (default: all universe)")
    parser.add_argument("--since", default="2020-01-01", help="Earliest filing date")
    parser.add_argument("--panel-only", action="store_true", help="Skip fetch, just rebuild panel")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Always record an attempt at start of run (resilient to timeout).
    # Spec 065 §1 #1 distinguishes ran-but-failed-midway vs never-ran.
    _state_write(
        {
            "last_attempt": _now_iso(),
            "schema_fingerprint": _schema_fingerprint(),
        }
    )

    if not args.panel_only:
        # Load universe
        universe = json.loads((PROJECT_ROOT / "production_data" / "universe.json").read_text())
        tickers = {}
        for entry in universe:
            t = entry["ticker"]
            cik = entry.get("cik") or entry.get("financial_data", {}).get("cik", "")
            if cik and (not args.tickers or t in args.tickers):
                tickers[t] = cik.lstrip("0")

        # Incremental mode is default: every ticker is checked against SEC for new
        # accessions; only those unseen in existing raw/{ticker}.json are fetched.
        # Pre-existing coverage is never wasted; daily refresh cost scales with
        # new-filing volume, not total history.
        to_fetch = {t: cik for t, cik in sorted(tickers.items())}
        log.info(
            f"Incremental refresh of Form 4 for {len(to_fetch)} tickers since {args.since} " f"({MAX_WORKERS} workers)"
        )

        fetched = 0
        failed = 0
        total_txns = 0  # total new transactions appended across all tickers
        updated_tickers = 0  # tickers that received at least one new accession
        _counter_lock = threading.Lock()

        def _fetch_one(ticker_cik):
            nonlocal fetched, failed, total_txns, updated_tickers
            ticker, cik = ticker_cik
            raw_file = RAW_DIR / f"{ticker}.json"
            try:
                existing = _load_existing_raw(raw_file)
                seen = {t.accession_number for t in existing if t.accession_number}
                new_txns = fetch_ticker(ticker, cik, args.since, seen_accessions=seen)
                if not new_txns:
                    with _counter_lock:
                        fetched += 1
                        if fetched % 25 == 0:
                            log.info(
                                f"  Progress: {fetched}/{len(to_fetch)} tickers, "
                                f"{total_txns} new txns, {updated_tickers} updated"
                            )
                    return ticker, 0, None
                merged = _merge_transactions(existing, new_txns)
                raw_file.write_text(json.dumps([asdict(t) for t in merged], indent=1))
                with _counter_lock:
                    fetched += 1
                    total_txns += len(new_txns)
                    updated_tickers += 1
                    if fetched % 25 == 0:
                        log.info(
                            f"  Progress: {fetched}/{len(to_fetch)} tickers, "
                            f"{total_txns} new txns, {updated_tickers} updated"
                        )
                return ticker, len(new_txns), None
            except Exception as e:
                with _counter_lock:
                    failed += 1
                log.error(f"  {ticker}: {e}")
                return ticker, 0, str(e)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(_fetch_one, (t, c)) for t, c in to_fetch.items()]
            for f in as_completed(futures):
                f.result()  # propagate exceptions

        log.info(
            f"Checked {fetched} tickers ({failed} failed), {updated_tickers} received "
            f"new accessions, {total_txns} new transactions appended"
        )

        # Save state — fetch run completed without unhandled exception.
        # last_attempt was already written at start; now mark success.
        now = _now_iso()
        updates: Dict[str, Any] = {
            "last_success": now,
            "last_fetch": now,  # backward-compat alias
            "tickers_checked": fetched,
            "tickers_updated": updated_tickers,
            "new_transactions": total_txns,
            "failed_tickers": failed,
            "since": args.since,
            "last_error": None,  # clear any prior error record
        }
        if total_txns > 0:
            updates["last_new_filing"] = now
        _state_write(updates)

    # Build panel — runs in BOTH modes (fetch and --panel-only).
    # Always record panel rebuild result so a stale panel can be detected.
    n = build_panel(RAW_DIR, PANEL_CSV)
    _state_write({"last_panel_rebuild": _now_iso(), "panel_rows": n})
    log.info(f"Done. Panel has {n} rows.")


if __name__ == "__main__":
    main()
