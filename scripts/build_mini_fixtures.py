#!/usr/bin/env python3
"""
Build mini fixture data for integration tests.

Subsets production data to 8 tickers + XBI benchmark and writes to
tests/fixtures/mini_data/. Run once; output is committed to the repo.

Usage:
    python3 scripts/build_mini_fixtures.py
"""

import csv
import io
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROD_DIR = PROJECT_ROOT / "production_data"
OUT_DIR = PROJECT_ROOT / "tests" / "fixtures" / "mini_data"

TICKERS = {"MRNA", "ACRS", "RVMD", "ALMS", "IBRX", "GILD", "BIIB", "BMRN"}
TICKERS_WITH_BENCH = TICKERS | {"XBI"}


def subset_json_array(src: Path, dst: Path, key: str = "ticker") -> int:
    """Filter a JSON array file to entries matching TICKERS."""
    data = json.loads(src.read_text())
    filtered = [rec for rec in data if rec.get(key) in TICKERS]
    dst.write_text(json.dumps(filtered, indent=2) + "\n")
    print(f"  {dst.name}: {len(data)} -> {len(filtered)} records")
    return len(filtered)


def subset_price_history(src: Path, dst: Path) -> int:
    """Filter price_history.csv to TICKERS + XBI."""
    with open(src, newline="") as fin:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        rows = [row for row in reader if row.get("ticker") in TICKERS_WITH_BENCH]

    with open(dst, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  {dst.name}: filtered to {len(rows)} rows ({len(TICKERS_WITH_BENCH)} tickers)")
    return len(rows)


def subset_momentum(src: Path, dst: Path) -> None:
    """Filter momentum_results.json to tickers present in our subset."""
    data = json.loads(src.read_text())

    # Filter summary lists
    summary = data.get("summary", {})
    for list_key in ("coordinated_buys", "coordinated_sells"):
        if list_key in summary:
            summary[list_key] = [t for t in summary[list_key] if t in TICKERS]

    # Filter rankings array
    if "rankings" in data:
        data["rankings"] = [r for r in data["rankings"] if r.get("ticker") in TICKERS]

    # Filter signals dict
    if "signals" in data:
        data["signals"] = {k: v for k, v in data["signals"].items() if k in TICKERS}

    # Update count
    data["tickers_analyzed"] = len(TICKERS)

    dst.write_text(json.dumps(data, indent=2) + "\n")
    print(f"  {dst.name}: filtered to {len(TICKERS)} tickers")


def main() -> None:
    print(f"Building mini fixtures in {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. universe.json
    print("\n[1/6] universe.json")
    n = subset_json_array(PROD_DIR / "universe.json", OUT_DIR / "universe.json")
    assert n == len(TICKERS), f"Expected {len(TICKERS)} universe entries, got {n}"

    # 2. financial_records.json
    print("[2/6] financial_records.json")
    n = subset_json_array(PROD_DIR / "financial_records.json", OUT_DIR / "financial_records.json")
    assert n == len(TICKERS), f"Expected {len(TICKERS)} financial records, got {n}"

    # 3. trial_records.json
    print("[3/6] trial_records.json")
    subset_json_array(PROD_DIR / "trial_records.json", OUT_DIR / "trial_records.json")

    # 4. market_data.json
    print("[4/6] market_data.json")
    n = subset_json_array(PROD_DIR / "market_data.json", OUT_DIR / "market_data.json")
    assert n == len(TICKERS), f"Expected {len(TICKERS)} market data entries, got {n}"

    # 5. price_history.csv
    print("[5/6] price_history.csv")
    subset_price_history(PROD_DIR / "price_history.csv", OUT_DIR / "price_history.csv")

    # 6. momentum_results.json
    print("[6/6] momentum_results.json")
    subset_momentum(PROJECT_ROOT / "momentum_results.json", OUT_DIR / "momentum_results.json")

    # Create ctgov_state placeholder
    (OUT_DIR / "ctgov_state").mkdir(exist_ok=True)
    (OUT_DIR / "ctgov_state" / ".gitkeep").touch()

    print(f"\nDone. Fixtures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
