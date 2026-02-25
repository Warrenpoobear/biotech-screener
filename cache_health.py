"""Cache health sentinel for SEC 8-K and CTGov upstream caches.

Detects cache outages and refresh anomalies by comparing current event
counts against prior snapshot baselines.  Produces a stable-schema
health record persisted as ``cache_health.json`` in each snapshot.

Status hierarchy: ok < warning < bad  (worst wins for overall_status).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Thresholds (defensive defaults; intentionally generous)
# ---------------------------------------------------------------------------
SEC8K_MIN_COUNT = 0            # BAD if strictly zero (complete outage)
SEC8K_MIN_RATIO_VS_PRIOR = 0.30   # WARNING if ratio < 0.30 (sharp collapse)

CTGOV_WARN_RATIO_LOW = 0.80   # WARNING band: [0.80, 1.25]
CTGOV_WARN_RATIO_HIGH = 1.25
CTGOV_BAD_RATIO_LOW = 0.60    # BAD band: outside [0.60, 1.50]
CTGOV_BAD_RATIO_HIGH = 1.50

_STATUS_RANK = {"ok": 0, "warning": 1, "bad": 2}


def _worst(*statuses: str) -> str:
    return max(statuses, key=lambda s: _STATUS_RANK.get(s, 0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_cache_health(
    sec8k_count: int,
    ctgov_count: int,
    prior_sec8k_count: Optional[int] = None,
    prior_ctgov_count: Optional[int] = None,
    *,
    as_of_date: str = "",
) -> Dict[str, Any]:
    """Compute cache health from event counts and optional prior baselines.

    Pure function — no I/O.  Returns a dict matching the cache_health.v1 schema.
    """
    sec8k_status = "ok"
    sec8k_reason = ""
    sec8k_ratio: Optional[float] = None

    # --- SEC 8-K checks ---
    if sec8k_count == 0:
        sec8k_status = "bad"
        sec8k_reason = "SEC 8-K cache empty (0 events)"
    elif prior_sec8k_count is not None and prior_sec8k_count > 0:
        sec8k_ratio = round(sec8k_count / prior_sec8k_count, 4)
        if sec8k_ratio < SEC8K_MIN_RATIO_VS_PRIOR:
            sec8k_status = "warning"
            sec8k_reason = (
                f"SEC 8-K count collapsed: {sec8k_count} vs prior {prior_sec8k_count} "
                f"(ratio={sec8k_ratio:.2f} < {SEC8K_MIN_RATIO_VS_PRIOR})"
            )

    # --- CTGov checks ---
    ctgov_status = "ok"
    ctgov_reason = ""
    ctgov_ratio: Optional[float] = None

    if prior_ctgov_count is not None and prior_ctgov_count > 0:
        ctgov_ratio = round(ctgov_count / prior_ctgov_count, 4)
        if ctgov_ratio < CTGOV_BAD_RATIO_LOW or ctgov_ratio > CTGOV_BAD_RATIO_HIGH:
            ctgov_status = "bad"
            ctgov_reason = (
                f"CTGov count extreme shift: {ctgov_count} vs prior {prior_ctgov_count} "
                f"(ratio={ctgov_ratio:.2f} outside [{CTGOV_BAD_RATIO_LOW}, {CTGOV_BAD_RATIO_HIGH}])"
            )
        elif ctgov_ratio < CTGOV_WARN_RATIO_LOW or ctgov_ratio > CTGOV_WARN_RATIO_HIGH:
            ctgov_status = "warning"
            ctgov_reason = (
                f"CTGov count shifted: {ctgov_count} vs prior {prior_ctgov_count} "
                f"(ratio={ctgov_ratio:.2f} outside [{CTGOV_WARN_RATIO_LOW}, {CTGOV_WARN_RATIO_HIGH}])"
            )

    overall = _worst(sec8k_status, ctgov_status)

    return {
        "schema": "cache_health.v1",
        "as_of_date": as_of_date,
        "sec8k": {
            "count": sec8k_count,
            "status": sec8k_status,
            "reason": sec8k_reason,
        },
        "ctgov": {
            "count": ctgov_count,
            "status": ctgov_status,
            "reason": ctgov_reason,
        },
        "overall_status": overall,
        "degraded_run": overall != "ok",
        "thresholds": {
            "sec8k_min_count": SEC8K_MIN_COUNT,
            "sec8k_min_ratio_vs_prior": SEC8K_MIN_RATIO_VS_PRIOR,
            "ctgov_warn_ratio_low": CTGOV_WARN_RATIO_LOW,
            "ctgov_warn_ratio_high": CTGOV_WARN_RATIO_HIGH,
            "ctgov_bad_ratio_low": CTGOV_BAD_RATIO_LOW,
            "ctgov_bad_ratio_high": CTGOV_BAD_RATIO_HIGH,
        },
        "comparisons": {
            "sec8k_ratio_vs_prior": sec8k_ratio,
            "ctgov_ratio_vs_prior": ctgov_ratio,
        },
    }


def load_prior_cache_health(
    snapshot_dir,
    current_date: str,
    *,
    max_candidates: int = 10,
) -> Optional[Dict[str, Any]]:
    """Find most recent prior cache_health.json from snapshot directory.

    Walks backward through date-named subdirectories (max *max_candidates*).
    Returns parsed dict or None.
    """
    import json
    import re
    from pathlib import Path

    snapshot_dir = Path(snapshot_dir)
    if not snapshot_dir.exists():
        return None

    date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    all_dates = sorted(
        [d.name for d in snapshot_dir.iterdir() if d.is_dir() and date_re.match(d.name)],
        reverse=True,
    )
    candidates = [d for d in all_dates if d < current_date][:max_candidates]

    for date_str in candidates:
        health_path = snapshot_dir / date_str / "cache_health.json"
        if not health_path.exists():
            continue
        try:
            with open(health_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "sec8k" in data and "ctgov" in data:
                return data
        except (json.JSONDecodeError, OSError):
            continue
    return None
