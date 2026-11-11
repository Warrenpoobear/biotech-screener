"""Ingest FDA Purple Book CSV → production_data/purple_book.json.

Downloads or parses the monthly Purple Book CSV export from
purplebooksearch.fda.gov/downloads.

Usage:
    python scripts/ingest_purple_book.py path/to/purplebook-data-download.csv
    python scripts/ingest_purple_book.py --download          # fetch latest
    python scripts/ingest_purple_book.py path/to/file.csv --ticker-map production_data/purple_book_ticker_map.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
PROD_DATA = REPO_ROOT / "production_data"
OUTPUT_PATH = PROD_DATA / "purple_book.json"
TICKER_MAP_PATH = PROD_DATA / "purple_book_ticker_map.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("ingest_purple_book")

# Column mapping: Purple Book CSV column → internal field
# Purple Book CSV columns vary slightly between releases; map flexibly
COLUMN_MAP = {
    # BLA number
    "BLA Number": "bla_number",
    "BLA": "bla_number",
    "BLA_Number": "bla_number",
    "bla_number": "bla_number",
    "License Number": "license_number",
    # Proprietary name
    "Proprietary Name": "product_name_proprietary",
    "Product (Proprietary Name)": "product_name_proprietary",
    "Trade Name": "product_name_proprietary",
    "proprietary_name": "product_name_proprietary",
    # Nonproprietary name
    "Proper Name": "product_name_nonproprietary",
    "Nonproprietary Name": "product_name_nonproprietary",
    "proper_name": "product_name_nonproprietary",
    "nonproprietary_name": "product_name_nonproprietary",
    # Applicant
    "Applicant": "applicant",
    "Sponsor": "applicant",
    "applicant": "applicant",
    # Date of licensure
    "Date of Licensure": "licensing_date",
    "Approval Date": "licensing_date",
    "License Date": "licensing_date",
    "date_of_licensure": "licensing_date",
    "Date of First Licensure": "first_licensure_date",
    # Product category / BLA type
    "Product Category": "product_category",
    "Category": "product_category",
    "product_category": "product_category",
    "BLA Type": "bla_type",
    # FDA renamed "BLA Type" to "License Type" (seen 2026-10); without this the
    # 351(k) biosimilar flag is silently empty and n_biosimilars drops to 0.
    "License Type": "bla_type",
    # Biosimilar — derived from BLA Type if not explicit
    "Biosimilar": "is_biosimilar",
    "Is Biosimilar": "is_biosimilar",
    "biosimilar": "is_biosimilar",
    # Interchangeable — derived from BLA Type if not explicit
    "Interchangeable": "is_interchangeable",
    "Is Interchangeable": "is_interchangeable",
    "interchangeable": "is_interchangeable",
    # Submission type (for interchangeability detection)
    "Submission Type": "submission_type",
    # Reference product
    "Reference Product BLA Number": "reference_product_bla",
    "Reference Product BLA": "reference_product_bla",
    "Ref Product BLA": "reference_product_bla",
    "reference_product_bla": "reference_product_bla",
    "Ref. Product Proprietary Name": "reference_product_name",
    "Reference Product Proprietary Name": "reference_product_name",
    "Reference Product": "reference_product_name",
    "reference_product_name": "reference_product_name",
    "Ref. Product Proper Name": "reference_product_proper_name",
    # Exclusivity
    "Reference Product Exclusivity Expiry": "exclusivity_expiry_date",
    "Exclusivity Expiry Date": "exclusivity_expiry_date",
    "Exclusivity End Date": "exclusivity_expiry_date",
    "Exclusivity Expiration Date": "exclusivity_expiry_date",
    "exclusivity_expiry_date": "exclusivity_expiry_date",
    "Ref. Product Exclusivity Exp. Date": "ref_exclusivity_expiry_date",
    "First Interchangeable Exclusivity Exp. Date": "interchangeable_exclusivity_date",
    "Orphan Exclusivity Exp. Date": "orphan_exclusivity_date",
    # Marketing status
    "Marketing Status": "marketing_status",
    "Status": "marketing_status",
    "marketing_status": "marketing_status",
    "Licensure": "licensure_status",
    # Strength / form
    "Strength": "strength",
    "strength": "strength",
    "Dosage Form": "dosage_form",
    "Route of Administration": "route",
    "dosage_form": "dosage_form",
    # Center
    "Center": "center",
    # Change type
    "N/R/U": "change_type",
    # Product presentation / number
    "Product Presentation": "product_presentation",
    "Product Number": "product_number",
    "Supplement Number": "supplement_number",
}


def _parse_date(v: str | None) -> str | None:
    if not v or v.strip() in ("", "-", "N/A", "n/a"):
        return None
    v = v.strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%d-%b-%Y", "%d-%b-%y", "%B %d, %Y", "%m/%d/%y"):
        try:
            parsed = datetime.strptime(v, fmt)
            # Handle 2-digit year: strptime maps 00-68 to 2000-2068, 69-99 to 1969-1999
            # For Purple Book, dates before 1990 are unlikely — but FDA biologics go back to 1980s
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue
    log.warning("Could not parse date: %s", v)
    return None


def _parse_bool(v: str | None) -> bool:
    if not v:
        return False
    return v.strip().lower() in ("true", "1", "yes", "y", "biosimilar", "interchangeable")


def _normalize_applicant(name: str) -> str:
    """Normalize applicant names for matching."""
    if not name:
        return ""
    # Remove common suffixes
    name = re.sub(
        r",?\s*(Inc\.?|LLC|Ltd\.?|Corp\.?|Co\.?|L\.?P\.?|plc|SA|AG|SE|GmbH|N\.?V\.?)$", "", name, flags=re.IGNORECASE
    )
    return name.strip()


DOWNLOADS_PAGE = "https://purplebooksearch.fda.gov/downloads"
_LINK_RE = re.compile(
    r"https?://[^\"'\s]*/PurpleBook/(\d{4})/purplebook-search-([A-Za-z]+)-data-download\.csv",
)
_MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        start=1,
    )
}


def find_latest_csv_url(page_html: str) -> Optional[tuple[str, int, int]]:
    """Return (url, year, month) of the newest monthly CSV linked on the page.

    FDA moved the files to accessdata.fda.gov and the month casing is not
    consistent (2026 has ``October`` but ``january``), so the URL is scraped
    from the downloads page instead of being built from the date.
    """
    best = None
    for m in _LINK_RE.finditer(page_html):
        month = _MONTHS.get(m.group(2).lower())
        if not month:
            continue
        key = (int(m.group(1)), month)
        if best is None or key > best[1:]:
            best = (m.group(0), *key)
    return best


def download_latest_csv(output_dir: Path) -> Optional[Path]:
    """Download the newest Purple Book CSV linked from the FDA downloads page."""
    import urllib.request

    try:
        req = urllib.request.Request(DOWNLOADS_PAGE, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            page = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        log.warning("Could not fetch %s: %s", DOWNLOADS_PAGE, e)
        return None
    found = find_latest_csv_url(page)
    if not found:
        log.warning("No Purple Book CSV links found on %s", DOWNLOADS_PAGE)
        return None
    url, y, m = found
    dest = output_dir / f"purple_book_{y}_{m:02d}.csv"
    try:
        log.info("Downloading: %s", url)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        tmp = dest.with_suffix(".csv.tmp")
        tmp.write_bytes(data)
        tmp.replace(dest)
        log.info("Downloaded: %s", dest)
        return dest
    except Exception as e:
        log.warning("Download failed for %s: %s", url, e)
        return None


def ingest_csv(csv_path: Path, as_of_date: str, ticker_map: dict[str, str] | None = None) -> dict:
    """Parse Purple Book CSV, normalize, return JSON-serializable dict."""
    records = []
    skipped = 0

    with open(csv_path, encoding="utf-8-sig") as f:
        # FDA Purple Book CSVs have header rows before the actual data header.
        # Scan for the row that contains "BLA Number" or "Applicant" to find the real header.
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if "BLA Number" in line or "Applicant" in line:
            # Check this looks like a header (has multiple known column names)
            if sum(1 for col in ["Applicant", "BLA", "Proprietary", "Proper Name"] if col in line) >= 2:
                header_idx = i
                break

    if header_idx is None:
        log.error("Could not find data header row in CSV")
        return {"deals": [], "as_of_date": as_of_date, "n_records": 0, "n_skipped": 0}

    log.info("Found data header at line %d (skipping %d preamble lines)", header_idx + 1, header_idx)

    import io

    data_text = "".join(lines[header_idx:])
    reader = csv.DictReader(io.StringIO(data_text))
    raw_cols = reader.fieldnames or []
    log.info("CSV columns: %s", raw_cols)

    col_map = {}
    for raw_col in raw_cols:
        if raw_col in COLUMN_MAP:
            col_map[raw_col] = COLUMN_MAP[raw_col]
        elif raw_col.strip() in COLUMN_MAP:
            col_map[raw_col] = COLUMN_MAP[raw_col.strip()]

    unmapped = [c for c in raw_cols if c not in col_map]
    if unmapped:
        log.info("Unmapped columns: %s", unmapped)

    for i, row in enumerate(reader):
        mapped = {}
        for raw_col, internal_col in col_map.items():
            mapped[internal_col] = row.get(raw_col, "").strip()

        # BLA Number may be in "BLA Number" or "License Number" column
        bla = mapped.get("bla_number", "").strip() or mapped.get("license_number", "").strip()
        if not bla:
            skipped += 1
            continue

        # Licensing date: try Approval Date, then Date of First Licensure
        licensing_date = _parse_date(mapped.get("licensing_date")) or _parse_date(mapped.get("first_licensure_date"))

        # PIT filter: exclude products licensed after as_of_date
        if licensing_date and licensing_date > as_of_date:
            skipped += 1
            continue

        applicant_raw = mapped.get("applicant", "")
        applicant_norm = _normalize_applicant(applicant_raw)

        # Try to resolve ticker
        resolved_ticker = None
        if ticker_map:
            resolved_ticker = ticker_map.get(applicant_raw) or ticker_map.get(applicant_norm)
            if not resolved_ticker:
                for map_name, map_ticker in ticker_map.items():
                    if map_name.lower() == applicant_norm.lower():
                        resolved_ticker = map_ticker
                        break

        # Derive biosimilar / interchangeable from BLA Type
        bla_type = mapped.get("bla_type", "").strip()
        is_biosimilar_explicit = _parse_bool(mapped.get("is_biosimilar"))
        is_interchangeable_explicit = _parse_bool(mapped.get("is_interchangeable"))

        # 351(k) = biosimilar pathway; 351(a) = original biologic
        is_biosimilar = is_biosimilar_explicit or "351(k)" in bla_type
        # Interchangeable: check explicit field, submission_type, or interchangeable exclusivity
        is_interchangeable = (
            is_interchangeable_explicit
            or "interchangeable" in mapped.get("submission_type", "").lower()
            or bool(mapped.get("interchangeable_exclusivity_date", "").strip())
        )

        # Reference product BLA: for 351(k) products, the reference is usually
        # indicated by Ref. Product fields. If not explicit, leave None.
        ref_bla = mapped.get("reference_product_bla", "").strip() or None
        ref_name = (
            mapped.get("reference_product_name", "").strip()
            or mapped.get("reference_product_proper_name", "").strip()
            or None
        )
        # If ref fields say "N/A", clear them
        if ref_name and ref_name.upper() == "N/A":
            ref_name = None

        # Exclusivity: use own exclusivity first, then reference product exclusivity
        exclusivity_date = _parse_date(mapped.get("exclusivity_expiry_date")) or _parse_date(
            mapped.get("ref_exclusivity_expiry_date")
        )
        orphan_exclusivity = _parse_date(mapped.get("orphan_exclusivity_date"))

        # Marketing status: use Marketing Status or Licensure
        marketing_status = (
            mapped.get("marketing_status", "").strip() or mapped.get("licensure_status", "").strip() or None
        )

        # Center (CDER vs CBER)
        center = mapped.get("center", "").strip() or None

        record = {
            "bla_number": bla,
            "product_name_proprietary": mapped.get("product_name_proprietary") or None,
            "product_name_nonproprietary": mapped.get("product_name_nonproprietary") or None,
            "applicant": applicant_raw or None,
            "applicant_normalized": applicant_norm or None,
            "resolved_ticker": resolved_ticker,
            "licensing_date": licensing_date,
            "product_category": bla_type or mapped.get("product_category") or None,
            "is_biosimilar": is_biosimilar,
            "is_interchangeable": is_interchangeable,
            "reference_product_bla": ref_bla,
            "reference_product_name": ref_name,
            "exclusivity_expiry_date": exclusivity_date,
            "orphan_exclusivity_date": orphan_exclusivity,
            "marketing_status": marketing_status,
            "center": center,
            "strength": mapped.get("strength") or None,
            "dosage_form": mapped.get("dosage_form") or None,
        }
        records.append(record)

    # Build reference product → biosimilar mapping
    ref_products: dict[str, dict] = {}
    for r in records:
        if not r["is_biosimilar"] and not r["is_interchangeable"]:
            bla = r["bla_number"]
            if bla not in ref_products:
                ref_products[bla] = {
                    "bla_number": bla,
                    "product_name": r["product_name_proprietary"],
                    "applicant": r["applicant"],
                    "resolved_ticker": r["resolved_ticker"],
                    "licensing_date": r["licensing_date"],
                    "exclusivity_expiry_date": r["exclusivity_expiry_date"],
                    "biosimilars": [],
                    "interchangeables": [],
                }

    for r in records:
        ref_bla = r.get("reference_product_bla")
        if ref_bla and ref_bla in ref_products:
            entry = {
                "bla_number": r["bla_number"],
                "product_name": r["product_name_proprietary"],
                "applicant": r["applicant"],
                "licensing_date": r["licensing_date"],
            }
            if r["is_interchangeable"]:
                ref_products[ref_bla]["interchangeables"].append(entry)
            elif r["is_biosimilar"]:
                ref_products[ref_bla]["biosimilars"].append(entry)

    # Stats
    n_biosimilar = sum(1 for r in records if r["is_biosimilar"])
    n_interchangeable = sum(1 for r in records if r["is_interchangeable"])
    n_reference = len(ref_products)
    n_with_exclusivity = sum(1 for r in ref_products.values() if r["exclusivity_expiry_date"])
    n_resolved = sum(1 for r in records if r["resolved_ticker"])
    unique_tickers = sorted(set(r["resolved_ticker"] for r in records if r["resolved_ticker"]))

    applicants = sorted(set(r["applicant_normalized"] for r in records if r["applicant_normalized"]))

    result = {
        "schema": "purple_book.v1",
        "as_of_date": as_of_date,
        "ingested_at": datetime.utcnow().isoformat() + "Z",
        "source_file": csv_path.name,
        "n_records": len(records),
        "n_skipped": skipped,
        "stats": {
            "n_reference_products": n_reference,
            "n_biosimilars": n_biosimilar,
            "n_interchangeables": n_interchangeable,
            "n_with_exclusivity": n_with_exclusivity,
            "n_resolved_to_ticker": n_resolved,
            "unique_tickers": unique_tickers,
            "n_unique_applicants": len(applicants),
        },
        "products": records,
        "reference_product_map": ref_products,
        "unresolved_applicants": [
            a for a in applicants if not any(r["resolved_ticker"] for r in records if r["applicant_normalized"] == a)
        ][:50],
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Ingest FDA Purple Book CSV")
    parser.add_argument("csv_path", nargs="?", type=Path, help="Path to Purple Book CSV")
    parser.add_argument("--download", action="store_true", help="Download latest CSV first")
    parser.add_argument("--as-of-date", default=date.today().isoformat())
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--ticker-map", type=Path, default=TICKER_MAP_PATH)
    args = parser.parse_args()

    # Load ticker map if it exists
    ticker_map = None
    if args.ticker_map.exists():
        with open(args.ticker_map, encoding="utf-8") as f:
            ticker_map = json.load(f)
        log.info("Loaded ticker map: %d entries", len(ticker_map))

    csv_path = args.csv_path
    if args.download or not csv_path:
        download_dir = PROD_DATA / "purple_book_downloads"
        download_dir.mkdir(parents=True, exist_ok=True)
        csv_path = download_latest_csv(download_dir)
        if not csv_path:
            log.error("Could not download or find Purple Book CSV")
            sys.exit(1)

    if not csv_path.exists():
        log.error("CSV not found: %s", csv_path)
        sys.exit(1)

    log.info("Ingesting %s with as_of_date=%s", csv_path, args.as_of_date)
    result = ingest_csv(csv_path, args.as_of_date, ticker_map)

    # Guard: a schema change that loses the license-type column yields 0
    # biosimilars with no error (225 -> 0 on 2026-11-03). Never write that.
    if result.get("n_records", 0) == 0 or result["stats"]["n_biosimilars"] == 0:
        log.error(
            "Refusing to write %s: n_records=%d n_biosimilars=%d (likely a CSV schema change)",
            args.output,
            result.get("n_records", 0),
            result.get("stats", {}).get("n_biosimilars", 0),
        )
        sys.exit(2)

    # Atomic replace: production_data files are hard-linked into data/staging,
    # so writing in place would also change the staged PIT inputs.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    tmp.replace(args.output)

    log.info(
        "Done: %d products, %d reference, %d biosimilars, %d interchangeables → %s",
        result["n_records"],
        result["stats"]["n_reference_products"],
        result["stats"]["n_biosimilars"],
        result["stats"]["n_interchangeables"],
        args.output,
    )
    if result["unresolved_applicants"]:
        log.info("Unresolved applicants (add to ticker map): %s", result["unresolved_applicants"][:10])


if __name__ == "__main__":
    main()
