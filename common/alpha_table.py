"""Shared alpha cohort table resolution.

Provides policy-based resolution of alpha cohort tables for both production
(``run_screen.py``) and PIT bundle (``run_screen_from_bundle.py``) pipelines.

The resolver enforces PIT-safety: for historical dates, a per-date OOS table
is used instead of the static table (which may be built from future data).
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional, Tuple

__all__ = [
    "check_artifact_marker",
    "resolve_alpha_table_path",
    "alpha_table_metadata",
    "EARLIEST_DAILY_TABLE_DATE",
]

# First available daily table date — any as_of_date after this should
# never silently fall back to static in PIT/bundle mode.
EARLIEST_DAILY_TABLE_DATE = "2020-02-14"


def check_artifact_marker(
    dated_path: Path,
) -> Tuple[bool, bool, Optional[dict]]:
    """Check if an artifact provenance marker exists and matches.

    Returns ``(marker_present, marker_valid, marker_data)`` where
    *marker_valid* means the marker SHA-256 matches the file on disk.
    """
    marker_path = dated_path.with_suffix(".artifact.json")
    if not marker_path.exists():
        return False, False, None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True, False, None
    expected_sha = marker.get("sha256", "")
    if not expected_sha or not dated_path.exists():
        return True, False, marker
    actual_sha = hashlib.sha256(dated_path.read_bytes()).hexdigest()
    return True, actual_sha == expected_sha, marker


def _find_nearest_earlier_table(
    daily_dir: Path,
    as_of_date: str,
) -> Optional[Tuple[Path, str]]:
    """Find the most recent per-date table strictly <= as_of_date.

    Scans ``daily_dir`` for files matching ``v1_YYYY-MM-DD.json`` and
    returns the closest earlier one.  Returns None if none found.
    """
    if not daily_dir.is_dir():
        return None
    best_date: Optional[str] = None
    best_path: Optional[Path] = None
    for p in daily_dir.glob("v1_*.json"):
        stem = p.stem  # e.g. "v1_2024-06-28"
        if stem.endswith(".artifact"):
            continue
        dt_str = stem[3:]  # strip "v1_"
        if len(dt_str) != 10:
            continue
        if dt_str <= as_of_date and (best_date is None or dt_str > best_date):
            best_date = dt_str
            best_path = p
    if best_path is not None:
        return best_path, best_date  # type: ignore[return-value]
    return None


def resolve_alpha_table_path(
    ruleset: Any,
    as_of_date: str,
    project_root: Path,
    logger: logging.Logger,
    *,
    allow_rebuild: bool = True,
) -> Tuple[Path, str]:
    """Resolve alpha cohort table path based on rebuild policy.

    Policies:
      - ``"never"``: use static ``alpha_cohort_table_path`` from ruleset
      - ``"if_missing"``: use dated file if it exists, nearest earlier if not,
        build OOS only if no dated tables exist at all
      - ``"daily"``: rebuild OOS table every run

    Parameters
    ----------
    ruleset : DecisionRuleset
        The active ruleset (provides policy, paths, train config).
    as_of_date : str
        Target date in YYYY-MM-DD format.
    project_root : Path
        Absolute path to the project root.
    logger : logging.Logger
        Logger instance for progress messages.
    allow_rebuild : bool
        If False, never build OOS in-process (useful for batch bundle runs
        where building per-date is expensive).  Falls back to nearest earlier
        or static instead.

    Returns
    -------
    (resolved_path, source_tag)
        *source_tag* is one of:
        ``"artifact"`` — CI-fetched table with valid provenance marker
        ``"preexisting_local"`` — dated file exists without marker
        ``"nearest_earlier:{date}"`` — closest earlier per-date table
        ``"rebuilt_in_run"`` — built OOS in this process
        ``"static_fallback"`` — fell back to the static table
        ``"static"`` — policy is "never" (always uses static)
    """
    policy = ruleset.alpha_table_rebuild_policy

    # --- Policy: never → always static ---
    if policy == "never":
        return _static_path(ruleset, project_root), "static"

    daily_dir = project_root / "production_data" / "alpha_cohort_tables" / "daily"
    dated_path = daily_dir / f"v1_{as_of_date}.json"

    # --- Exact dated file exists ---
    if dated_path.exists():
        if policy == "if_missing":
            marker_present, marker_valid, _ = check_artifact_marker(dated_path)
            source = "artifact" if (marker_present and marker_valid) else "preexisting_local"
            logger.info(
                "  Alpha table (%s): reusing %s [source=%s]",
                policy, dated_path, source,
            )
            return dated_path, source
        # policy == "daily" falls through to rebuild below

    # --- if_missing: try nearest earlier before rebuilding ---
    if policy == "if_missing" and not dated_path.exists():
        nearest = _find_nearest_earlier_table(daily_dir, as_of_date)
        if nearest is not None:
            path, near_date = nearest
            logger.info(
                "  Alpha table (if_missing): nearest earlier %s for %s",
                near_date, as_of_date,
            )
            return path, f"nearest_earlier:{near_date}"

    # --- Rebuild OOS table in-process ---
    if allow_rebuild:
        logger.info(
            "  Alpha table (%s): building OOS table for %s "
            "(mode=%s, horizon=%s)",
            policy, as_of_date,
            ruleset.alpha_train_mode, ruleset.alpha_train_horizon,
        )
        try:
            from scripts.build_alpha_cohort_table_oos import (
                build_oos_table,
                write_table_with_marker,
            )

            table = build_oos_table(
                as_of_date=as_of_date,
                train_mode=ruleset.alpha_train_mode,
                horizon=ruleset.alpha_train_horizon,
                min_train_dates=ruleset.alpha_train_min_train_dates,
                shrink_k=ruleset.alpha_cohort_shrink_k,
            )

            if table is not None:
                write_table_with_marker(
                    table, dated_path,
                    as_of_date=as_of_date, source="rebuilt_in_run",
                )
                logger.info("  Alpha table: wrote %s + marker", dated_path)
                return dated_path, "rebuilt_in_run"

            logger.warning(
                "  Alpha table rebuild returned None (insufficient data)"
            )
        except Exception as exc:
            logger.warning("  Alpha table rebuild failed: %s", exc)

    # --- Fallback: nearest earlier (for daily mode too) ---
    if not dated_path.exists():
        nearest = _find_nearest_earlier_table(daily_dir, as_of_date)
        if nearest is not None:
            path, near_date = nearest
            logger.info(
                "  Alpha table: falling back to nearest earlier %s for %s",
                near_date, as_of_date,
            )
            return path, f"nearest_earlier:{near_date}"

    # --- Last resort: static ---
    logger.warning(
        "  Alpha table: falling back to static table for %s", as_of_date,
    )
    return _static_path(ruleset, project_root), "static_fallback"


def _static_path(ruleset: Any, project_root: Path) -> Path:
    """Resolve the static alpha table path."""
    p = Path(ruleset.alpha_cohort_table_path)
    if not p.is_absolute():
        p = project_root / p
    return p


def alpha_table_metadata(
    as_of_date: str,
    resolved_path: Path,
    source: str,
) -> dict:
    """Build audit metadata for the resolved alpha table.

    Returns a dict with:
      - ``alpha_table_source``: source tag from resolver
      - ``alpha_table_path_resolved``: filename of resolved table
      - ``alpha_table_date_used``: the effective table date (from filename or source)
      - ``alpha_table_gap_days``: days between as_of_date and table date (0 = exact)
      - ``alpha_table_static_warning``: True if static was used when daily was expected
    """
    from datetime import date as _date

    # Extract effective date from source tag or filename
    table_date: Optional[str] = None
    if source.startswith("nearest_earlier:"):
        table_date = source.split(":", 1)[1]
    elif resolved_path.stem.startswith("v1_") and len(resolved_path.stem) == 13:
        # v1_YYYY-MM-DD → extract date
        table_date = resolved_path.stem[3:]
    elif source in ("artifact", "preexisting_local", "rebuilt_in_run"):
        # Exact match — table date == as_of_date
        table_date = as_of_date

    # Compute gap
    gap_days: Optional[int] = None
    if table_date is not None:
        try:
            gap_days = (_date.fromisoformat(as_of_date) - _date.fromisoformat(table_date)).days
        except ValueError:
            pass

    # Static warning: if we fell back to static for a date that should have
    # had a per-date table available
    static_warning = (
        "static" in source
        and as_of_date >= EARLIEST_DAILY_TABLE_DATE
    )

    return {
        "alpha_table_source": source,
        "alpha_table_path_resolved": resolved_path.name,
        "alpha_table_date_used": table_date,
        "alpha_table_gap_days": gap_days,
        "alpha_table_static_warning": static_warning,
    }
