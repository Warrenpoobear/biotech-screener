"""
Startup preflight checks for the biotech screener pipeline.

Validates environment variables, critical data files, and external
connectivity BEFORE the pipeline begins processing. Fail-fast design:
surface configuration problems at startup, not mid-pipeline.

Usage:
    from common.startup_preflight import run_preflight

    issues = run_preflight(data_dir=Path("production_data"), mode="strict")
    # issues.hard  → list of blocking errors (pipeline should abort)
    # issues.soft  → list of warnings (pipeline can continue)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class PreflightResult:
    """Aggregated preflight check results."""

    hard: List[str] = field(default_factory=list)
    soft: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.hard) == 0

    def summary(self) -> str:
        lines = []
        if self.hard:
            lines.append(f"HARD FAIL ({len(self.hard)}):")
            for h in self.hard:
                lines.append(f"  ✗ {h}")
        if self.soft:
            lines.append(f"WARNINGS ({len(self.soft)}):")
            for s in self.soft:
                lines.append(f"  ! {s}")
        if not lines:
            lines.append("All preflight checks passed.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Environment variable checks
# ---------------------------------------------------------------------------

# (var_name, required_for_feature, is_hard_requirement)
_ENV_SCHEMA = [
    ("AS_OF_DATE", "pipeline date override", False),
    ("SEC_USER_AGENT", "SEC EDGAR API compliance (prevents rate-limiting; SEC policy)", False),
    ("FRED_API_KEY", "macro data collection (Fed funds, yield curve)", False),
    ("MD_AUTH_TOKEN", "Morningstar Direct integration", False),
    ("XAI_API_KEY", "Herald event classification (xAI Grok)", False),
    ("TT_SECRET", "Tastytrade options data", False),
    ("TT_REFRESH", "Tastytrade options data", False),
    ("NCBI_API_KEY", "PubMed literature enrichment (10 req/s vs 3 req/s anonymous)", False),
    ("SLACK_WEBHOOK_URL", "Slack alerting", False),
    ("PIPELINE_ALERT_WEBHOOK", "pipeline failure alerts", False),
]


def _check_env_vars(result: PreflightResult) -> None:
    """Check that expected environment variables are set."""
    for var_name, feature, is_hard in _ENV_SCHEMA:
        val = os.environ.get(var_name, "")
        if not val.strip():
            msg = f"ENV missing: {var_name} (needed for {feature})"
            if is_hard:
                result.hard.append(msg)
            else:
                result.soft.append(msg)


# ---------------------------------------------------------------------------
# Data file checks
# ---------------------------------------------------------------------------

# (relative_path, description, is_hard_requirement)
_REQUIRED_FILES = [
    ("universe.json", "ticker universe", True),
    ("financial_records.json", "financial health data", True),
]

_OPTIONAL_FILES = [
    ("trial_records.json", "clinical trial records"),
    ("catalyst_events.json", "catalyst event timeline"),
    ("market_data.json", "market data (price, volume)"),
    ("adcom_outcomes.json", "ADCOM voting outcomes"),
    ("pdufa_dates.json", "PDUFA calendar"),
    ("holdings_history.json", "13F institutional holdings"),
]


def _check_data_files(data_dir: Path, result: PreflightResult) -> None:
    """Check that critical data files exist and are non-empty."""
    if not data_dir.exists():
        result.hard.append(f"data_dir does not exist: {data_dir}")
        return

    if not data_dir.is_dir():
        result.hard.append(f"data_dir is not a directory: {data_dir}")
        return

    for rel_path, desc, is_hard in _REQUIRED_FILES:
        fpath = data_dir / rel_path
        if not fpath.exists():
            result.hard.append(f"Required file missing: {rel_path} ({desc})")
        elif fpath.stat().st_size == 0:
            result.hard.append(f"Required file is empty: {rel_path} ({desc})")

    for rel_path, desc in _OPTIONAL_FILES:
        fpath = data_dir / rel_path
        if not fpath.exists():
            result.soft.append(f"Optional file missing: {rel_path} ({desc})")
        elif fpath.stat().st_size == 0:
            result.soft.append(f"Optional file is empty: {rel_path} ({desc})")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_preflight(
    data_dir: Path,
    check_env: bool = True,
    check_files: bool = True,
) -> PreflightResult:
    """Run all preflight checks and return results.

    Args:
        data_dir: Path to the data directory (production_data/).
        check_env: Whether to validate environment variables.
        check_files: Whether to validate data files.

    Returns:
        PreflightResult with hard (blocking) and soft (warning) issues.
    """
    result = PreflightResult()

    if check_env:
        _check_env_vars(result)

    if check_files:
        _check_data_files(data_dir, result)

    # Log summary
    if result.hard:
        logger.error("Preflight FAILED: %d hard errors", len(result.hard))
        for h in result.hard:
            logger.error("  HARD: %s", h)
    if result.soft:
        for s in result.soft:
            logger.warning("  WARN: %s", s)
    if result.ok:
        logger.info("Preflight passed (%d warnings)", len(result.soft))

    return result
