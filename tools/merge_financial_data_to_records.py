"""
Merge tickers from financial_data.json into financial_records.json.

financial_data.json  — written by fetch_pending_biotech_data.py (stub enrichment)
financial_records.json — read by run_screen.py for survivability + runway scoring

Newly promoted stubs land in financial_data.json only. Without this merge they
show as financials_missing ineligible in the next screen run.

Usage:
    python3 tools/merge_financial_data_to_records.py [--dry-run]
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from common.atomic_io import atomic_write_text  # noqa: E402

FIN_DAT = REPO / "production_data" / "financial_data.json"
FIN_REC = REPO / "production_data" / "financial_records.json"

# Canonical field order for financial_records.json
RECORD_KEYS = [
    "ticker",
    "cik",
    "Assets",
    "Assets_date",
    "CurrentAssets",
    "CurrentAssets_date",
    "Liabilities",
    "Liabilities_date",
    "CurrentLiabilities",
    "CurrentLiabilities_date",
    "ShareholdersEquity",
    "ShareholdersEquity_date",
    "Cash",
    "Cash_date",
    "CashRestricted",
    "CashRestricted_date",
    "ShortTermInvestments",
    "ShortTermInvestments_date",
    "R&D",
    "R&D_date",
    "NetIncome",
    "NetIncome_date",
    "CFO",
    "CFO_date",
    "OperatingExpenses",
    "OperatingExpenses_date",
    "CashAndSecurities",
    "CashAndSecurities_date",
    "collected_at",
]


def load_json(path: Path) -> list:
    with open(path) as f:
        return json.load(f)


def main(dry_run: bool = False) -> int:
    fin_dat = load_json(FIN_DAT)
    fin_recs = load_json(FIN_REC)

    fin_dat_map = {r["ticker"]: r for r in fin_dat if "ticker" in r}
    fin_rec_tickers = {r["ticker"] for r in fin_recs if "ticker" in r}

    missing = sorted(set(fin_dat_map.keys()) - fin_rec_tickers)

    if not missing:
        print(
            f"merge_financial: no missing tickers — financial_records.json already complete ({len(fin_recs)} records)"
        )
        return 0

    added = []
    skipped = []
    for ticker in missing:
        src = fin_dat_map[ticker]
        cash = src.get("Cash") or src.get("cash_and_equivalents") or src.get("cash")
        if not cash or cash <= 0:
            skipped.append(ticker)
            continue
        rec = {k: src.get(k) for k in RECORD_KEYS}
        added.append(rec)

    print(f"merge_financial: {len(missing)} missing from financial_records.json")
    for r in added:
        print(f"  ADD  {r['ticker']:<6}  Cash={r['Cash']}  CFO={r['CFO']}")
    for t in skipped:
        print(f"  SKIP {t:<6}  no valid Cash — enrichment incomplete")

    if dry_run:
        print(f"[dry-run] would add {len(added)} records, skip {len(skipped)}")
        return 0

    fin_recs.extend(added)
    # Atomic replace: production_data files are hard-linked into staging dirs,
    # and an in-place write would mutate every linked copy.
    atomic_write_text(FIN_REC, json.dumps(fin_recs, indent=2, default=str))
    print(f"merge_financial: wrote financial_records.json ({len(fin_recs) - len(added)} → {len(fin_recs)} records)")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing")
    args = parser.parse_args()
    sys.exit(main(dry_run=args.dry_run))
