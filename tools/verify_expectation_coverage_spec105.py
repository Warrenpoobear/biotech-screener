#!/usr/bin/env python3
"""Spec 105: standalone expectation-layer coverage verification.

Wraps production_qa_check feature coverage for host/CI use and writes
a dedicated artifact under artifacts/spec105/.

Usage:
    python3 tools/verify_expectation_coverage_spec105.py --as-of-date 2026-06-24
    python3 tools/verify_expectation_coverage_spec105.py --as-of-date 2026-06-24 --write
    python3 tools/verify_expectation_coverage_spec105.py --rankings-path data/snapshots/2026-06-24/rankings.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "artifacts" / "spec105"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def build_report(*, as_of_date: str, rankings_path: Path | None = None) -> dict[str, object]:
    from tools.production_qa_check import FEATURE_COVERAGE_REQUIREMENTS, check_feature_coverage

    if rankings_path is not None:
        import csv

        from tools.production_qa_check import _check

        if not rankings_path.is_file():
            check = _check("feature_coverage", False, "No rankings.csv")
        else:
            with open(rankings_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                cols = set(reader.fieldnames or [])
                rows = list(reader)
            n = len(rows)
            failures: list[str] = []
            per_field: list[str] = []
            for field, min_cov, required in FEATURE_COVERAGE_REQUIREMENTS:
                tag = "" if required else "*"
                if field not in cols:
                    per_field.append(f"{field}{tag}=MISSING")
                    if required:
                        failures.append(f"{field}: column missing")
                    continue
                non_empty = sum(1 for r in rows if r.get(field, "").strip() not in ("", "None", "nan", "NaN"))
                cov = non_empty / n if n else 0.0
                per_field.append(f"{field}{tag}={cov * 100:.1f}%")
                if required and cov < min_cov:
                    failures.append(f"{field}: {cov * 100:.1f}% < {min_cov * 100:.0f}%")
            detail = "; ".join(per_field)
            if failures:
                detail = f"{detail} | FAIL: {'; '.join(failures)}"
                check = _check("feature_coverage", False, detail)
            else:
                check = _check("feature_coverage", True, detail)
    else:
        check = check_feature_coverage(as_of_date)

    return {
        "schema": "spec105_expectation_coverage.v1",
        "as_of_date": as_of_date,
        "generated_at": f"{as_of_date}T00:00:00Z",
        "overall": "PASS" if check.get("status") == "PASS" else "FAIL",
        "check": check.get("check"),
        "detail": check.get("detail"),
        "requirements": [
            {"field": field, "min_coverage": min_cov, "required": required}
            for field, min_cov, required in FEATURE_COVERAGE_REQUIREMENTS
        ],
        "operator_note": "Live QA gate for Spec 105 — does not modify production scoring.",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Spec 105 expectation coverage verifier")
    ap.add_argument("--as-of-date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--rankings-path", help="Override rankings.csv path")
    ap.add_argument("--write", action="store_true", help="Write artifacts/spec105/{date}_coverage.json")
    ap.add_argument("--json", action="store_true", help="Print JSON only")
    args = ap.parse_args()

    ds = args.as_of_date
    rankings_path = Path(args.rankings_path) if args.rankings_path else None
    if rankings_path and not rankings_path.is_file():
        print(f"rankings file not found: {rankings_path}", file=sys.stderr)
        return 2

    report = build_report(as_of_date=ds, rankings_path=rankings_path)

    if args.write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / f"{ds}_coverage.json"
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not args.json:
            print(f"Wrote {out_path}")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Spec 105 coverage — {ds} — overall={report['overall']}")
        print(f"  {report.get('detail', '')}")

    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
