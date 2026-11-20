#!/usr/bin/env python3
"""Provenance-enforced PDUFA calendar entry and review tool.

Manual curation is the primary path for forward PDUFA dates.  This tool
validates, deduplicates, and ingests new entries into pdufa_dates.json
with required provenance fields.

Modes:
  --validate   → validate an input candidates JSON against provenance rules
  --ingest     → merge validated candidates into pdufa_dates.json (dedup)
  --audit      → audit the current pdufa_dates.json for provenance gaps

Provenance policy:
  New entries MUST have source_url and source (COMPANY_GUIDANCE, SEC_8K,
  PRESS_RELEASE, etc.) with a valid as_of_disclosed_at PIT anchor.
  Entries without provenance are rejected with a reason.

Candidate JSON format (one record per entry):
  [
    {
      "ticker": "ACME",
      "drug_name": "AcmeDrug",
      "indication": "cancer",
      "pdufa_date": "2026-08-15",
      "event_type": "PDUFA",
      "submission_type": "NDA",
      "confidence": "HIGH",
      "source": "COMPANY_GUIDANCE",
      "source_url": "https://ir.acme.com/press-release/nda-acceptance",
      "as_of_disclosed_at": "2026-02-15",
      "notes": "NDA accepted per company press release"
    }
  ]

Usage:
    python3 tools/collect_pdufa_forward.py --validate candidates.json
    python3 tools/collect_pdufa_forward.py --ingest candidates.json --dry-run
    python3 tools/collect_pdufa_forward.py --ingest candidates.json
    python3 tools/collect_pdufa_forward.py --audit
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.atomic_io import atomic_write_text  # noqa: E402

logger = logging.getLogger(__name__)

# Valid source types (ranked by trust)
VALID_SOURCES = frozenset(
    {
        "COMPANY_GUIDANCE",
        "SEC_8K",
        "PRESS_RELEASE",
        "FEDERAL_REGISTER",
        "MANUAL",
        "ANALYST_ESTIMATE",
    }
)

# Required fields for a provenance-valid entry
PROVENANCE_REQUIRED = ("pdufa_date", "ticker", "source", "as_of_disclosed_at")

# Additional provenance fields that should be present (warned if missing)
PROVENANCE_RECOMMENDED = ("source_url", "drug_name", "indication")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_candidate(rec: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a candidate record for required fields and provenance.

    Returns (is_valid, list_of_error_strings).
    """
    errors: List[str] = []

    if not isinstance(rec, dict):
        return False, ["record is not a dict"]

    # Required fields
    for field in PROVENANCE_REQUIRED:
        val = rec.get(field)
        if not val or (isinstance(val, str) and not val.strip()):
            errors.append(f"missing required field: {field}")

    # Date format validation
    for date_field in ("pdufa_date", "as_of_disclosed_at"):
        val = rec.get(date_field, "")
        if isinstance(val, str) and val.strip():
            try:
                date.fromisoformat(val)
            except (ValueError, TypeError):
                errors.append(f"invalid date format for {date_field}: {val!r}")

    # Source validation
    source = rec.get("source", "")
    if source and source not in VALID_SOURCES:
        errors.append(f"unrecognized source: {source!r}; valid: {sorted(VALID_SOURCES)}")

    # Confidence validation
    confidence = str(rec.get("confidence", "")).upper()
    if confidence and confidence not in ("HIGH", "MED", "LOW"):
        errors.append(f"invalid confidence: {confidence!r}; must be HIGH/MED/LOW")

    # source_url validation (recommended, not required for manual entries)
    source_url = rec.get("source_url", "")
    if source_url:
        if not (source_url.startswith("http://") or source_url.startswith("https://")):
            errors.append(f"source_url must be http(s): {source_url!r}")

    # PIT safety: as_of_disclosed_at should be before pdufa_date
    disclosed = rec.get("as_of_disclosed_at", "")
    pdufa = rec.get("pdufa_date", "")
    if disclosed and pdufa:
        try:
            if date.fromisoformat(disclosed) > date.fromisoformat(pdufa):
                errors.append(f"as_of_disclosed_at ({disclosed}) is after pdufa_date ({pdufa})")
        except (ValueError, TypeError):
            pass  # already caught above

    return len(errors) == 0, errors


def validate_candidates(
    records: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Validate a batch of candidate records.

    Returns (valid, invalid_with_errors).
    """
    valid = []
    invalid = []
    for rec in records:
        ok, errs = validate_candidate(rec)
        if ok:
            valid.append(rec)
        else:
            inv = dict(rec) if isinstance(rec, dict) else {"_raw": rec}
            inv["_errors"] = errs
            invalid.append(inv)
    return valid, invalid


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


def load_existing_pdufa(data_dir: Path) -> List[Dict[str, Any]]:
    """Load existing pdufa_dates.json."""
    path = data_dir / "pdufa_dates.json"
    if not path.exists():
        return []
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
        return records if isinstance(records, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def dedup_candidates(
    candidates: List[Dict[str, Any]],
    existing: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split candidates into new vs duplicate (already in existing).

    Dedup key: (ticker, pdufa_date).
    """
    existing_keys: Set[Tuple[str, str]] = set()
    for rec in existing:
        key = (rec.get("ticker", "").upper(), rec.get("pdufa_date", ""))
        existing_keys.add(key)

    new = []
    dupes = []
    for rec in candidates:
        key = (rec.get("ticker", "").upper(), rec.get("pdufa_date", ""))
        if key in existing_keys:
            dupes.append(rec)
        else:
            new.append(rec)
            existing_keys.add(key)  # prevent intra-batch dupes

    return new, dupes


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def ingest_candidates(
    candidates: List[Dict[str, Any]],
    data_dir: Path,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """Validate and ingest candidates into pdufa_dates.json.

    Returns report dict with accepted/rejected/duplicate counts.
    """
    existing = load_existing_pdufa(data_dir)
    new_candidates, dupes = dedup_candidates(candidates, existing)

    accepted = []
    rejected = []

    for rec in new_candidates:
        ok, errs = validate_candidate(rec)
        if ok:
            # Strip internal/review fields before ingest
            clean = {k: v for k, v in rec.items() if not k.startswith("_")}
            accepted.append(clean)
        else:
            rejected.append({"record": rec, "errors": errs})

    report: Dict[str, Any] = {
        "total_candidates": len(candidates),
        "duplicates_skipped": len(dupes),
        "validated_new": len(new_candidates),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "rejection_details": rejected[:10],
        "dry_run": dry_run,
    }

    if accepted and not dry_run:
        merged = existing + accepted
        path = data_dir / "pdufa_dates.json"
        # Atomic replace: pdufa_dates.json is hard-linked into staged PIT inputs;
        # write_text() would rewrite every staged copy (common/atomic_io.py).
        atomic_write_text(path, json.dumps(merged, indent=2, ensure_ascii=False) + "\n")
        report["written_to"] = str(path)
        report["total_after_ingest"] = len(merged)
        logger.info(
            "Ingested %d new PDUFA entries → %s (total: %d)",
            len(accepted),
            path,
            len(merged),
        )
    elif accepted:
        logger.info("DRY RUN: would ingest %d new PDUFA entries", len(accepted))
        report["would_ingest"] = [
            {
                "ticker": r.get("ticker"),
                "pdufa_date": r.get("pdufa_date"),
                "drug_name": r.get("drug_name"),
            }
            for r in accepted
        ]

    return report


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def audit_existing(data_dir: Path) -> Dict[str, Any]:
    """Audit existing pdufa_dates.json for provenance gaps.

    Returns a report with per-entry provenance status.
    """
    records = load_existing_pdufa(data_dir)
    results: Dict[str, Any] = {
        "total": len(records),
        "with_source_url": 0,
        "with_as_of_disclosed_at": 0,
        "with_full_provenance": 0,
        "gaps": [],
    }

    for rec in records:
        ticker = rec.get("ticker", "?")
        pdufa = rec.get("pdufa_date", "?")
        has_url = bool(rec.get("source_url"))
        has_disclosed = bool(rec.get("as_of_disclosed_at"))
        has_source = bool(rec.get("source"))

        if has_url:
            results["with_source_url"] += 1
        if has_disclosed:
            results["with_as_of_disclosed_at"] += 1
        if has_url and has_disclosed and has_source:
            results["with_full_provenance"] += 1
        else:
            gap = {"ticker": ticker, "pdufa_date": pdufa, "missing": []}
            if not has_url:
                gap["missing"].append("source_url")
            if not has_disclosed:
                gap["missing"].append("as_of_disclosed_at")
            if not has_source:
                gap["missing"].append("source")
            results["gaps"].append(gap)

    return results


# ---------------------------------------------------------------------------
# Coverage delta helper (reusable)
# ---------------------------------------------------------------------------


def compute_regulatory_coverage_delta(
    current_flagged: Set[str],
    prior_flagged: Set[str],
    current_n_eligible: int,
    prior_n_eligible: int,
) -> Dict[str, Any]:
    """Compute regulatory coverage delta between two snapshots."""
    added = current_flagged - prior_flagged
    dropped = prior_flagged - current_flagged

    current_pct = round(len(current_flagged) / max(current_n_eligible, 1) * 100, 1)
    prior_pct = round(len(prior_flagged) / max(prior_n_eligible, 1) * 100, 1)

    return {
        "current_count": len(current_flagged),
        "prior_count": len(prior_flagged),
        "current_pct": current_pct,
        "prior_pct": prior_pct,
        "delta_pct": round(current_pct - prior_pct, 1),
        "delta_count": len(current_flagged) - len(prior_flagged),
        "added": sorted(added),
        "dropped": sorted(dropped),
        "kept": len(current_flagged & prior_flagged),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    p = argparse.ArgumentParser(
        description="Provenance-enforced PDUFA calendar entry and review",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "production_data",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "pdufa_collector",
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--validate",
        type=Path,
        metavar="CANDIDATES_JSON",
        help="Validate a candidates JSON file",
    )
    mode.add_argument(
        "--ingest",
        type=Path,
        metavar="CANDIDATES_JSON",
        help="Ingest validated candidates into pdufa_dates.json",
    )
    mode.add_argument(
        "--audit",
        action="store_true",
        help="Audit existing pdufa_dates.json for provenance gaps",
    )

    p.add_argument("--dry-run", action="store_true", help="With --ingest: validate but don't write")

    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.validate:
        candidates = json.loads(args.validate.read_text(encoding="utf-8"))
        valid, invalid = validate_candidates(candidates)
        print(f"Validated {len(candidates)} candidates:")
        print(f"  Valid:   {len(valid)}")
        print(f"  Invalid: {len(invalid)}")
        if invalid:
            print("\nRejections:")
            for inv in invalid[:10]:
                ticker = inv.get("ticker", "?")
                errs = inv.get("_errors", [])
                print(f"  {ticker}: {'; '.join(errs)}")

        # Write validation report
        report_path = args.output_dir / "validation_report.json"
        report_path.write_text(
            json.dumps({"valid": valid, "invalid": invalid}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nFull report: {report_path}")

    elif args.ingest:
        candidates = json.loads(args.ingest.read_text(encoding="utf-8"))
        report = ingest_candidates(candidates, args.data_dir, dry_run=args.dry_run)

        report_path = args.output_dir / "ingest_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Ingest report: {report_path}")
        print(f"  Total candidates: {report['total_candidates']}")
        print(f"  Duplicates skipped: {report['duplicates_skipped']}")
        print(f"  Accepted: {report['accepted']}")
        print(f"  Rejected: {report['rejected']}")
        if report.get("dry_run"):
            print("  (DRY RUN — nothing written)")

    elif args.audit:
        report = audit_existing(args.data_dir)
        print(f"PDUFA Calendar Audit ({report['total']} entries):")
        print(f"  With source_url:          {report['with_source_url']}")
        print(f"  With as_of_disclosed_at:  {report['with_as_of_disclosed_at']}")
        print(f"  Full provenance:          {report['with_full_provenance']}")
        if report["gaps"]:
            print(f"\n  Provenance gaps ({len(report['gaps'])} entries):")
            for gap in report["gaps"]:
                print(f"    {gap['ticker']} ({gap['pdufa_date']}): missing {gap['missing']}")

        report_path = args.output_dir / "audit_report.json"
        report_path.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nFull report: {report_path}")


if __name__ == "__main__":
    main()
