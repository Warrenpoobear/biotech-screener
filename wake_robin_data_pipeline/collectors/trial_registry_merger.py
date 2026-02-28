"""
trial_registry_merger.py - Merge CTgov + EUCTR + CTIS + ISRCTN into unified PIT-filtered snapshot.

Deduplicates by cross-registry ID matching (NCT IDs, EudraCT numbers, protocol codes).
Produces a single merged trial list with stable sort by (ticker, primary_id).
"""

import json
import logging
import os
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA = "merged_trials.v1"


def merge_trial_registries(
    ctgov_records: List[dict],
    euctr_records: List[dict],
    ctis_records: List[dict],
    isrctn_records: List[dict],
    as_of_date: date,
) -> List[dict]:
    """Merge all registry records into a unified, deduplicated, PIT-filtered list.

    Deduplication strategy:
    1. Build ID index: map every known ID (primary + secondary) -> record
    2. If two records share any ID -> merge (prefer CTgov as base)
    3. PIT filter: only records with last_update_posted <= as_of_date

    Returns merged list sorted by (ticker, primary_id).
    """
    # Tag CTgov records with registry field if missing
    for rec in ctgov_records:
        if "registry" not in rec:
            rec["registry"] = "ctgov"
        if "primary_id" not in rec:
            rec["primary_id"] = rec.get("nct_id", "")
        if "secondary_ids" not in rec:
            rec["secondary_ids"] = []

    all_records = ctgov_records + euctr_records + ctis_records + isrctn_records

    # Build clusters of records sharing IDs
    merged = _deduplicate(all_records)

    # PIT filter
    cutoff = as_of_date.isoformat()
    pit_filtered = []
    for rec in merged:
        lup = (rec.get("last_update_posted") or "")[:10]
        if not lup or lup <= cutoff:
            pit_filtered.append(rec)

    # Stable sort
    pit_filtered.sort(key=lambda r: (r.get("ticker", ""), r.get("primary_id", "")))

    logger.info(
        "Merged %d total -> %d deduplicated -> %d PIT-filtered (cutoff=%s)",
        len(all_records), len(merged), len(pit_filtered), cutoff,
    )
    return pit_filtered


def _deduplicate(records: List[dict]) -> List[dict]:
    """Deduplicate records by cross-registry ID matching.

    Returns list of unique (possibly merged) records.
    """
    # Build ID -> record index
    id_to_cluster: Dict[str, int] = {}
    clusters: Dict[int, dict] = {}
    next_cluster = 0

    for rec in records:
        all_ids = _extract_all_ids(rec)

        # Find existing clusters that share any ID
        matched_clusters = set()
        for rid in all_ids:
            if rid in id_to_cluster:
                matched_clusters.add(id_to_cluster[rid])

        if not matched_clusters:
            # New cluster
            cluster_id = next_cluster
            next_cluster += 1
            clusters[cluster_id] = rec
            for rid in all_ids:
                id_to_cluster[rid] = cluster_id
        else:
            # Merge into first matched cluster
            target_id = min(matched_clusters)
            base = clusters[target_id]
            clusters[target_id] = _merge_records(base, rec)

            # Remap all IDs from other clusters
            for other_id in matched_clusters:
                if other_id != target_id:
                    if other_id in clusters:
                        # Merge other cluster into target too
                        clusters[target_id] = _merge_records(
                            clusters[target_id], clusters[other_id]
                        )
                        del clusters[other_id]

            # Update ID index
            merged_ids = _extract_all_ids(clusters[target_id])
            for rid in all_ids | merged_ids:
                id_to_cluster[rid] = target_id

    return list(clusters.values())


def _extract_all_ids(rec: dict) -> set:
    """Extract all identifiable IDs from a record."""
    ids = set()

    primary = (rec.get("primary_id") or rec.get("nct_id") or "").strip()
    if primary:
        ids.add(primary.upper())

    for sid in rec.get("secondary_ids", []):
        sid = str(sid).strip()
        if sid:
            # Strip protocol: prefix for matching
            clean = sid.upper()
            ids.add(clean)
            if clean.startswith("PROTOCOL:"):
                ids.add(clean[9:].strip())

    # Also extract NCT IDs from secondary_ids
    nct = (rec.get("nct_id") or "").strip()
    if nct:
        ids.add(nct.upper())

    return ids


def _merge_records(base: dict, overlay: dict) -> dict:
    """Merge two records for the same trial.

    Prefer CTgov record as base. Fill in missing fields from overlay.
    Combine secondary_ids from both.
    """
    # Prefer CTgov as canonical base
    if overlay.get("registry") == "ctgov" and base.get("registry") != "ctgov":
        base, overlay = overlay, base

    merged = dict(base)

    # Fill missing fields from overlay
    for key in ("title", "phase", "status", "study_type", "start_date",
                "primary_completion_date", "completion_date", "results_posted_date",
                "first_posted", "last_update_posted"):
        base_val = merged.get(key)
        overlay_val = overlay.get(key)
        if (not base_val or base_val == "NA" or base_val == "UNKNOWN") and overlay_val:
            merged[key] = overlay_val

    # Merge sponsor info
    base_sponsor = merged.get("sponsor", {})
    overlay_sponsor = overlay.get("sponsor", {})
    if isinstance(base_sponsor, dict) and isinstance(overlay_sponsor, dict):
        if not base_sponsor.get("name") and overlay_sponsor.get("name"):
            base_sponsor["name"] = overlay_sponsor["name"]
        if not base_sponsor.get("country") and overlay_sponsor.get("country"):
            base_sponsor["country"] = overlay_sponsor["country"]
        merged["sponsor"] = base_sponsor

    # Union secondary IDs
    existing_ids = set(merged.get("secondary_ids", []))
    overlay_primary = overlay.get("primary_id", "")
    if overlay_primary and overlay_primary != merged.get("primary_id"):
        existing_ids.add(overlay_primary)
    for sid in overlay.get("secondary_ids", []):
        existing_ids.add(sid)
    merged["secondary_ids"] = sorted(existing_ids)

    # Union conditions
    existing_conds = set(merged.get("conditions", []))
    for c in overlay.get("conditions", []):
        existing_conds.add(c)
    merged["conditions"] = sorted(existing_conds)

    # Union countries
    existing_countries = set(merged.get("countries", []))
    for c in overlay.get("countries", []):
        existing_countries.add(c)
    merged["countries"] = sorted(existing_countries)

    return merged


def write_merged_cache(
    records: List[dict],
    as_of_date: date,
    out_dir: Path,
) -> Path:
    """Atomic write merged records to cache directory.

    Writes:
        out_dir/trial_records_{as_of_date}.json
        out_dir/trial_records_{as_of_date}.meta.json

    Returns path to main cache file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    live_path = out_dir / f"trial_records_{as_of_date.isoformat()}.json"
    meta_path = out_dir / f"trial_records_{as_of_date.isoformat()}.meta.json"

    # Count per-registry
    registry_counts: Dict[str, int] = {}
    for rec in records:
        reg = rec.get("registry", "unknown")
        registry_counts[reg] = registry_counts.get(reg, 0) + 1

    # Atomic write
    fd, tmp_path = tempfile.mkstemp(dir=str(out_dir), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(records, f, indent=2)
        os.replace(tmp_path, str(live_path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    # Write meta sidecar
    meta = {
        "schema": _SCHEMA,
        "as_of_date": as_of_date.isoformat(),
        "total_records": len(records),
        "registry_counts": registry_counts,
        "dedup_stats": {
            "input_total": sum(registry_counts.values()),
            "output_total": len(records),
        },
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info(
        "Merged trials: %d records -> %s (by registry: %s)",
        len(records), live_path, registry_counts,
    )
    return live_path
