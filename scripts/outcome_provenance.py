"""Outcome provenance — fingerprint price data + panels for audit trail.

Provides deterministic SHA256 hashes of price data and panel CSVs so that
walkforward and calibration artifacts can be traced back to the exact data
they were computed from.  Any outcome-altering data change (backfill,
correction, split adjustment) will produce a different hash, making the
change obvious and auditable.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from governance.hashing import hash_file

PRICE_CSV = PROJECT_ROOT / "production_data" / "price_history.csv"


def get_git_sha() -> str:
    """Short git SHA, graceful fallback."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(PROJECT_ROOT),
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def build_provenance(
    price_csv: Path = PRICE_CSV,
    panel_path: Optional[Path] = None,
    ruleset_id: str = "",
    explanation_registry_fingerprint: str = "",
) -> Dict[str, Any]:
    """Build provenance dict for embedding in artifact JSON.

    Args:
        price_csv: Path to the price history CSV.
        panel_path: Optional path to the walkforward panel CSV.
        ruleset_id: Ruleset ID string to include.
        explanation_registry_fingerprint: Registry fingerprint from
            decision_engine_codes.registry_fingerprint().

    Returns:
        Dict with keys: code_version, ruleset_id, price_history_sha256,
        panel_sha256, explanation_registry_fingerprint (None when not provided).
    """
    prov: Dict[str, Any] = {
        "code_version": get_git_sha(),
        "ruleset_id": ruleset_id,
        "explanation_registry_fingerprint": explanation_registry_fingerprint or None,
    }

    if price_csv.exists():
        prov["price_history_sha256"] = hash_file(price_csv)
    else:
        prov["price_history_sha256"] = None

    if panel_path is not None and panel_path.exists():
        prov["panel_sha256"] = hash_file(panel_path)
    else:
        prov["panel_sha256"] = None

    return prov
