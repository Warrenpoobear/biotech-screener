#!/usr/bin/env python3
"""
run_screen.py - Deterministic Biotech Screening Orchestrator

Coordinates Modules 1-5 to produce weekly ranked investment opportunities.

DETERMINISM GUARANTEES:
- as_of_date is REQUIRED (no today() defaults)
- All modules receive explicit as_of_date
- PIT discipline enforced throughout
- Decimal-only arithmetic
- Stable ordering on all outputs
- Content-hash verification on inputs (if enabled)

ORCHESTRATION FEATURES:
- Dry-run mode: Validate inputs without running pipeline
- Checkpointing: Save/resume intermediate module outputs
- Audit trail: Parameter snapshots and content hashes

Usage:
    python run_screen.py --as-of-date 2024-12-15 --data-dir ./data --output results.json
    python run_screen.py --as-of-date 2024-12-15 --data-dir ./data --dry-run
    python run_screen.py --as-of-date 2024-12-15 --data-dir ./data --output results.json --checkpoint-dir ./checkpoints

Architecture:
    Module 1: Universe filtering
    Module 2: Financial health
    Module 3: Catalyst detection (NEW: Delta-based event detection)
    Module 4: Clinical development
    Module 5: Composite ranking
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import logging
import math
import os
import re
import shutil
import sys
import tarfile
import tempfile
from datetime import date, datetime, timedelta
from decimal import Decimal

# Load .env file (credentials, API keys) before any module reads os.environ
from dotenv import load_dotenv

load_dotenv()

# Configure logging with rotation support
from logging.handlers import RotatingFileHandler
from pathlib import Path
from statistics import median, quantiles
from typing import Any, Callable, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from archive_snapshot import get_git_info, sha256_file
from backtest.cost_model import CostSchedule, estimate_trade_cost
from common.adcom_vote_features import ADCOM_VOTE_COLUMNS, build_adcom_vote_lookup
from common.binary_quality_score import compute_binary_quality_score
from common.clinical_corroboration import evaluate_corroboration
from common.data_integration_contracts import (
    normalize_financial_field_alias,
    normalize_ticker_set,
    safe_numeric_check,
    validate_financial_records_schema,
    validate_market_data_schema,
    validate_trial_records_schema,
)

# Common utilities
from common.date_utils import normalize_date, to_date_object, to_date_string
from common.event_quality_features import (
    OPTIONS_QUALITY_COLUMNS,
    compute_clinical_91_180_quality,
    compute_event_quality_features,
    compute_options_quality_composite,
)
from common.hard_catalyst import is_hard_catalyst as _is_hard_catalyst
from common.integration_contracts import SchemaValidationError, validate_module_5_output, validate_pipeline_handoff
from common.options_diagnostics import OPTIONS_DIAGNOSTIC_COLUMNS, empty_diagnostics

# Production hardening utilities
from common.production_hardening import (  # Path security; File operations; Integrity; Timeouts; Logging; Validation; Resources; Constants
    DEFAULT_MODULE_EXECUTION_TIMEOUT,
    DEFAULT_PIPELINE_TIMEOUT,
    MAX_JSON_FILE_SIZE_MB,
    FileSizeError,
    IntegrityError,
    OperationTimeoutError,
    PathTraversalError,
    SymlinkError,
    compute_content_hash,
    json_serializer,
    load_with_integrity_check,
    operation_timeout,
    require_minimum_memory,
    safe_join_path,
    safe_mkdir,
    safe_read_json,
    safe_write_json,
    sanitize_for_logging,
    save_with_integrity,
    validate_checkpoint_path,
    validate_date_format,
    validate_file_size,
    validate_numeric_bounds,
    validate_path_within_base,
    verify_integrity,
)
from decision_engine import ACTIONABLE_COLUMNS, DEFAULT_RULESET, SORT_CONTRIB_KEYS
from decision_engine import VERSION as DE_VERSION
from decision_engine import (
    DecisionRuleset,
    _safe_float,
    compute_actionable_sort_key,
    compute_decision_fields,
    compute_sort_contribs,
    compute_target_weights,
    resolve_catalyst_priority,
)
from ranker_engine import compute_ranker_adjustments
from ranker_v2_pairwise import RankerV2Config, model_from_dict, score_snapshot
from selector_engine import BlockWeight, SelectorConfig, SignalSpec, compute_selector_scores, get_regime_modulation
from tools.check_market_snapshot_freshness import check_freshness as _check_snapshot_freshness
from tools.check_market_snapshot_freshness import emit_diagnostic as _emit_snapshot_diagnostic
from tools.compute_path_c_drawdown import compute_drawdown_vs_xbi

# Spec 050 → v1.14.0: coinvest-only selector (inst_delta_z removed)
# Validated on true PIT backtest: +2.34pp/mo net, t=2.60, 67 periods (bundle)
# v1.14.0 change: inst_delta_z zeroed out from selector (2026-05-04)
#   Reason: two-frame ALERT — ic_health_monitor mean_ic=-0.097 hit_rate=0.111
#   (36 dates 2026-02-19 to 2026-04-02) + calibration_evidence event-IC=-0.244
#   (75 postmortems). Comparator: coinvest_score_z mean_ic=+0.097 hit_rate=0.897
#   same window, IC cross-correlation rho=-0.33 — degradation isolated to inst_delta_z.
#   Governance: INST_DELTA_Z_SIGNAL_HEALTH_GOVERNANCE_REVIEW_2026_05_04.md
#   Ruleset: v1.14.0_coinvest_only_selector.json (id=8887576e; initial commit
#   produced phantom hash 622edb77, corrected by commit bd91b523d 2026-05-05).
A4_SELECTOR_CONFIG = SelectorConfig(
    block_weights=(
        BlockWeight("clinical", 0.00),
        BlockWeight("catalyst", 0.15),
        BlockWeight("survivability", 0.10),
        BlockWeight("institutional", 0.65),
        BlockWeight("market_structure", 0.10),
    ),
    institutional_signals=(
        SignalSpec("coinvest_score_z", 1.00),  # was 0.65; inst_delta_z weight redistributed
        SignalSpec("inst_delta_z", 0.00),  # was 0.35; ALERT degradation 2026-05-04
        SignalSpec(
            "coinvest_recency_state", 0.00, categorical=True, value_map=(("fresh", 1.0), ("stale", 0.3), ("", 0.0))
        ),
    ),
)

# Spec 051: Production pairwise ranker config (winning full-strength settings)
# Promoted 2026-04-05: 2-feature model (coinvest_score_z + financial_score)
# Walk-forward: spread +2.95%, IC +0.143 (t=2.98) — beats prior 5-feature on all metrics
# Scoring audit: dead features (inst_delta_z, catalyst_decay_w, binary_quality_score) added noise
# Rollback: swap ranker_v2_model.json with ranker_v2_model_5feat_rollback.json,
#           change feature_set back to "minimal"
PRODUCTION_RANKER_V2_CONFIG = RankerV2Config(
    feature_set="minimal_v2",
    cohort_top_n=60,
    require_catalyst_window=False,
    n_epochs=200,
    max_pairs_per_date=400,
    train_window=36,
)

# Module 3A specific imports
from event_detector import SimpleMarketCalendar
from event_ledger import REGULATORY_EVENT_TYPES, classify_catalyst_family

# Module imports
from module_1_universe import compute_module_1_universe
from module_2_financial_v2 import compute_module_2_financial
from module_3_catalyst import Module3Config, compute_module_3_catalyst
from module_4_clinical_dev import compute_module_4_clinical_dev
from module_5_composite_with_defensive import compute_module_5_composite_with_defensive
from module_5_scoring_v3 import summarize_dev_catalyst_guardrail

# Production validation
from production_validation import validate_screening_output

# Enhancement modules (optional)
try:
    from indication_mapper import IndicationMapper
    from pos_engine import ProbabilityOfSuccessEngine
    from regime_engine import RegimeDetectionEngine
    from short_interest_engine import ShortInterestSignalEngine

    HAS_ENHANCEMENTS = True
except ImportError as e:
    HAS_ENHANCEMENTS = False
    logger.info(f"Enhancement modules not available: {e}")

# Macro data collector for enhanced regime detection (optional)
try:
    from wake_robin_data_pipeline.collectors.macro_data_collector import MacroDataCollector

    HAS_MACRO_COLLECTOR = True
except ImportError as e:
    HAS_MACRO_COLLECTOR = False
    logger.info(f"Macro data collector not available: {e}")

# Accuracy enhancements adapter (optional)
try:
    from accuracy_enhancements_adapter import AccuracyEnhancementsAdapter

    HAS_ACCURACY_ENHANCEMENTS = True
except ImportError as e:
    HAS_ACCURACY_ENHANCEMENTS = False
    logger.info(f"Accuracy enhancements not available: {e}")

# Dilution risk engine (optional)
try:
    from dilution_risk_engine import DilutionRiskEngine

    HAS_DILUTION_RISK = True
except ImportError as e:
    HAS_DILUTION_RISK = False
    logger.info(f"Dilution risk engine not available: {e}")

# Timeline slippage engine (optional)
try:
    from timeline_slippage_engine import TimelineSlippageEngine

    HAS_TIMELINE_SLIPPAGE = True
except ImportError as e:
    HAS_TIMELINE_SLIPPAGE = False
    logger.info(f"Timeline slippage engine not available: {e}")

# FDA designation engine (optional)
try:
    from fda_designation_engine import FDADesignationEngine, generate_sample_designations

    HAS_FDA_DESIGNATIONS = True
except ImportError as e:
    HAS_FDA_DESIGNATIONS = False
    logger.info(f"FDA designation engine not available: {e}")

# Pipeline diversity engine (optional)
try:
    from pipeline_diversity_engine import PipelineDiversityEngine

    HAS_PIPELINE_DIVERSITY = True
except ImportError as e:
    HAS_PIPELINE_DIVERSITY = False
    logger.info(f"Pipeline diversity engine not available: {e}")

# Competitive intensity engine (optional)
try:
    from competitive_intensity_engine import CompetitiveIntensityEngine

    HAS_COMPETITIVE_INTENSITY = True
except ImportError as e:
    HAS_COMPETITIVE_INTENSITY = False
    logger.info(f"Competitive intensity engine not available: {e}")

# Partnership validation engine (optional)
try:
    from partnership_engine import PartnershipEngine

    HAS_PARTNERSHIP_ENGINE = True
except ImportError as e:
    HAS_PARTNERSHIP_ENGINE = False
    logger.info(f"Partnership engine not available: {e}")

# Cash burn trajectory engine (optional)
try:
    from cash_burn_engine import CashBurnEngine

    HAS_CASH_BURN_ENGINE = True
except ImportError as e:
    HAS_CASH_BURN_ENGINE = False
    logger.info(f"Cash burn engine not available: {e}")

# Phase transition momentum engine (optional)
try:
    from phase_momentum_engine import PhaseTransitionEngine

    HAS_PHASE_MOMENTUM_ENGINE = True
except ImportError as e:
    HAS_PHASE_MOMENTUM_ENGINE = False
    logger.info(f"Phase momentum engine not available: {e}")

# Morningstar quantitative signal engine (optional)
try:
    from morningstar_signal_engine import MorningstarSignalEngine

    HAS_MORNINGSTAR = True
except ImportError as e:
    HAS_MORNINGSTAR = False
    logger.info(f"Morningstar signal engine not available: {e}")

# Optional: Risk gates for audit trail
try:
    from risk_gates import compute_parameters_hash as risk_params_hash
    from risk_gates import get_parameters_snapshot as get_risk_params

    HAS_RISK_GATES = True
except ImportError as e:
    HAS_RISK_GATES = False
    logger.info(f"Risk gates not available: {e}")

try:
    from liquidity_scoring import compute_parameters_hash as liq_params_hash
    from liquidity_scoring import get_parameters_snapshot as get_liq_params

    HAS_LIQUIDITY_SCORING = True
except ImportError as e:
    HAS_LIQUIDITY_SCORING = False
    logger.info(f"Liquidity scoring not available: {e}")

# Ticker validation for fail-loud data quality
try:
    from src.validators.ticker_validator import validate_blacklist, validate_ticker_list

    HAS_TICKER_VALIDATION = True
except ImportError as e:
    HAS_TICKER_VALIDATION = False
    logger.warning(f"Ticker validation module not available - skipping validation: {e}")

# Baker overlay (optional)
try:
    from baker_overlay import compute_baker_overlay

    HAS_BAKER_OVERLAY = True
except ImportError as e:
    HAS_BAKER_OVERLAY = False
    logger.info(f"Baker overlay not available: {e}")

VERSION = "1.6.0"  # Bumped for governance-friendly CLI enhancements
DETERMINISTIC_TIMESTAMP_SUFFIX = "T00:00:00Z"
GOVERNANCE_SCHEMA_VERSION = "screen_output_governance.v1"
SCORE_AFFECTING_CSV_INPUTS = ("price_history.csv",)
GENERATED_JSON_OUTPUT_PREFIXES = (
    "catalyst_events_",
    "catalyst_events_vnext_",
    "diagnostics_",
    "run_log",
    "screen_",
)
SCHEMA_DECISION_PORTFOLIO = "decision_portfolio.v1"
SCHEMA_PORTFOLIO_POSITIONS = "portfolio_positions.v1"
SCHEMA_METADATA = "metadata.v1"

# =============================================================================
# STAGE TIMING HELPER
# =============================================================================

from contextlib import contextmanager as _contextmanager


@_contextmanager
def _timed(label: str):
    """Print wall-clock time for a named pipeline stage."""
    import time as _time

    _t0 = _time.time()
    yield
    print(f"[timing] {label}: {_time.time() - _t0:.2f}s", flush=True)


# =============================================================================
# RANK-DEPTH SHADOW COHORTS
# =============================================================================
# VALIDATION_INFRASTRUCTURE / RANK_DEPTH_SHADOW_TRACKING / NO_MODEL_CHANGE.
#
# Annotation-only layer that labels ranked names by depth-of-rank cohort and
# exports a Top-60 sidecar. It NEVER affects rank, score, eligibility, sizing,
# or production selection. The Top-30 remains the primary model basket; ranks
# 31-60 are monitored as shadow candidates only; Top-60 tests rank-depth
# robustness. Nothing here is tradable by default.

_RANK_DEPTH_TOP30_MAX = 30
_RANK_DEPTH_BAND_MAX = 60  # ranks 31..60 form the shadow reserve bench

# Sidecar column -> source field in rankings.csv (csv_rows). Computed/derived
# columns (as_of_date + cohort flags) are filled separately.
_RANK_DEPTH_SIDECAR_FIELDS = [
    ("ticker", "ticker"),
    ("company_name", "company_name"),
    ("actionable_rank", "actionable_rank"),
    ("final_score", "final_score"),
    ("market_cap_mm", "market_cap_mm"),
    ("close_price", "close_price"),
    ("catalyst_bucket", "catalyst_bucket"),
    ("catalyst_date", "next_catalyst_date"),
    ("primary_catalyst", "catalyst_event_type"),
    ("institutional_score", "inst_delta_z"),
    ("coinvest_score", "coinvest_score_z"),
    ("expectation_gap", "expectation_error_score"),
    ("ees_shadow_flag", "ees_v3_gate"),
    ("options_shadow_flag", "opt_use_for_judgment"),
    ("data_quality_flags", "risk_flags"),
]


def _rank_depth_safe_rank(rank):
    """Coerce an actionable_rank to int; non-numeric/blank -> large sentinel."""
    try:
        return int(float(rank))
    except (TypeError, ValueError):
        return 10**9


def _classify_rank_depth(rank):
    """Return (cohort, is_top30, is_rank31_60, is_top60) for one actionable_rank.

    Non-numeric / blank ranks (ineligible rows) fall into ``outside_top60``.
    """
    r = _rank_depth_safe_rank(rank)
    is_top30 = 1 <= r <= _RANK_DEPTH_TOP30_MAX
    is_band = _RANK_DEPTH_TOP30_MAX < r <= _RANK_DEPTH_BAND_MAX
    is_top60 = 1 <= r <= _RANK_DEPTH_BAND_MAX
    if is_top30:
        cohort = "top30"
    elif is_band:
        cohort = "rank31_60"
    else:
        cohort = "outside_top60"
    return (cohort, is_top30, is_band, is_top60)


def add_rank_depth_cohorts(df, rank_col="actionable_rank"):
    """Annotate a ranked dataframe with rank-depth shadow cohorts.

    VALIDATION_INFRASTRUCTURE / RANK_DEPTH_SHADOW_TRACKING / NO_MODEL_CHANGE.
    Adds ``rank_depth_cohort`` (top30 | rank31_60 | outside_top60) plus boolean
    ``is_top30`` / ``is_rank31_60`` / ``is_top60`` columns. Annotation only:
    does not affect rank, score, eligibility, sizing, or selection.
    """
    import pandas as pd

    out = df.copy()
    rank = pd.to_numeric(out[rank_col], errors="coerce")
    is_top30 = rank.between(1, _RANK_DEPTH_TOP30_MAX)
    is_band = rank.between(_RANK_DEPTH_TOP30_MAX + 1, _RANK_DEPTH_BAND_MAX)

    out["rank_depth_cohort"] = "outside_top60"
    out.loc[is_top30, "rank_depth_cohort"] = "top30"
    out.loc[is_band, "rank_depth_cohort"] = "rank31_60"
    out["is_top30"] = is_top30
    out["is_rank31_60"] = is_band
    out["is_top60"] = rank.between(1, _RANK_DEPTH_BAND_MAX)
    return out


def export_rank_depth_top60_sidecar(csv_rows, snap_path, as_of_date):
    """Write snapshots/<date>/rank_depth_top60.csv from the ranked rows.

    Shadow discovery + depth-validation artifact: who is populating ranks
    31-60, and why. Annotation only — derived from the already-written
    rankings.csv rows, never mutating them. Returns the output Path or None.
    """
    out_path = Path(snap_path) / "rank_depth_top60.csv"
    header = (
        ["as_of_date"]
        + [name for name, _ in _RANK_DEPTH_SIDECAR_FIELDS]
        + ["rank_depth_cohort", "is_top30", "is_rank31_60", "is_top60"]
    )
    out_rows = []
    for row in csv_rows:
        cohort, is_top30, is_band, is_top60 = _classify_rank_depth(row.get("actionable_rank"))
        if not is_top60:
            continue  # sidecar is Top-60 only
        rec = {"as_of_date": as_of_date}
        for name, src in _RANK_DEPTH_SIDECAR_FIELDS:
            rec[name] = row.get(src, "")
        rec["rank_depth_cohort"] = cohort
        rec["is_top30"] = is_top30
        rec["is_rank31_60"] = is_band
        rec["is_top60"] = is_top60
        out_rows.append(rec)

    out_rows.sort(key=lambda r: _rank_depth_safe_rank(r["actionable_rank"]))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for rec in out_rows:
            writer.writerow(rec)
    return out_path


# =============================================================================
# CATALYST WINDOW PRESETS AND DECAY FUNCTIONS
# =============================================================================

CATALYST_WINDOW_PRESETS = {
    "tight": (7, 30),
    "standard": (15, 45),
    "wide": (15, 90),
}

# =============================================================================
# COMPANY ARCHETYPE CLASSIFICATION
# =============================================================================

# Maps Yahoo Finance industry strings to company archetypes.
# Archetypes other than "drug_developer" are exempt from the clinical trial gate.
INDUSTRY_TO_ARCHETYPE = {
    "Biotechnology": "drug_developer",
    "Drug Manufacturers - Specialty & Generic": "commercial_pharma",
    "Drug Manufacturers - General": "commercial_pharma",
    "Diagnostics & Research": "platform_diagnostics",
    "Medical Devices": "platform_devices",
    "Medical Instruments & Supplies": "platform_devices",
    "Health Information Services": "platform_services",
    "Medical Care Facilities": "platform_services",
}

# Archetypes exempt from clinical trial gate
CLINICAL_GATE_EXEMPT_ARCHETYPES = frozenset(
    {
        "commercial_pharma",
        "commercial_biotech",
        "platform_diagnostics",
        "platform_devices",
        "platform_services",
    }
)


def _is_eligible(row: Dict[str, Any]) -> bool:
    """Robust eligibility check — handles '1', '1.0', 'true', 'yes'."""
    v = str(row.get("eligible", "")).strip().lower()
    return v in ("1", "1.0", "true", "yes")


def classify_company_archetype(
    ticker: str,
    industry: str,
    has_revenue: bool,
    revenue_scale_bucket: str,
) -> str:
    """
    Classify a company into an archetype based on industry and revenue signals.

    Drug developers get the standard clinical trial gate. All other archetypes
    (commercial pharma, revenue-generating biotech, platforms/diagnostics/devices)
    are exempt because their business model doesn't depend on proprietary trials.

    Args:
        ticker: Ticker symbol (for logging only)
        industry: Yahoo Finance industry string (e.g. "Biotechnology")
        has_revenue: Whether the company has meaningful revenue (from Module 2)
        revenue_scale_bucket: Revenue bucket from Module 2 ('pre_revenue', 'small', 'medium', 'large')

    Returns:
        Archetype string: one of drug_developer, commercial_pharma, commercial_biotech,
        platform_diagnostics, platform_devices, platform_services
    """
    archetype = INDUSTRY_TO_ARCHETYPE.get(industry, "drug_developer")

    # Override: biotech with meaningful revenue is commercial_biotech
    if archetype == "drug_developer" and has_revenue and revenue_scale_bucket in ("medium", "large"):
        archetype = "commercial_biotech"

    return archetype


# =============================================================================
# CLINICAL ACTIVITY FILTER
# =============================================================================

# Phase ordering for minimum phase filter (lower index = earlier phase)
PHASE_ORDER = {
    "unknown": 0,
    "preclinical": 0,
    "phase1": 1,
    "phase 1": 1,
    "phase1/phase2": 1,
    "phase 1/2": 1,
    "phase2": 2,
    "phase 2": 2,
    "phase2/phase3": 2,
    "phase 2/3": 2,
    "phase3": 3,
    "phase 3": 3,
    "approved": 4,
    "nda/bla": 4,
}


# =============================================================================
# DEVELOPMENT STAGE (display-only column, NOT a scoring input)
# =============================================================================

# Canonical enum for the rankings.csv development_stage column. Order is
# loosely early -> late but the field is categorical, not numeric.
DEVELOPMENT_STAGE_VALUES = (
    "preclinical",
    "phase_1",
    "phase_1_2",
    "phase_2",
    "phase_2_3",
    "phase_3",
    "nda_bla",
    "approved",
    "commercial",
    "unknown",
)

# Phase string -> development_stage enum. Covers Module 4 PHASE_SCORES values
# ("approved", "phase 3", "phase 2/3", "phase 2", "phase 1/2", "phase 1",
# "preclinical") plus the synonyms tracked in PHASE_ORDER.
_PHASE_TO_DEVELOPMENT_STAGE = {
    "preclinical": "preclinical",
    "phase 1": "phase_1",
    "phase1": "phase_1",
    "phase 1/2": "phase_1_2",
    "phase1/phase2": "phase_1_2",
    "phase 1/phase 2": "phase_1_2",
    "phase 2": "phase_2",
    "phase2": "phase_2",
    "phase 2/3": "phase_2_3",
    "phase2/phase3": "phase_2_3",
    "phase 2/phase 3": "phase_2_3",
    "phase 3": "phase_3",
    "phase3": "phase_3",
    "nda/bla": "nda_bla",
    "nda": "nda_bla",
    "bla": "nda_bla",
    "nda_bla": "nda_bla",
    "approved": "approved",
    # Numeric phase encoding: rankings.csv stores lead_program_phase as a
    # float code (0=preclinical, 1=phase 1, 2=phase 2, 3=phase 3, 4=approved).
    # Module 4 internally uses string values; the CSV-side encoding is
    # numeric. Both formats normalize to the same enum.
    "0": "preclinical",
    "0.0": "preclinical",
    "1": "phase_1",
    "1.0": "phase_1",
    "1.5": "phase_1_2",
    "2": "phase_2",
    "2.0": "phase_2",
    "2.5": "phase_2_3",
    "3": "phase_3",
    "3.0": "phase_3",
    "4": "approved",
    "4.0": "approved",
}


def _normalize_phase_to_development_stage(phase) -> str:
    """Normalize a free-form phase string into the development_stage enum."""
    if not phase:
        return "unknown"
    return _PHASE_TO_DEVELOPMENT_STAGE.get(str(phase).strip().lower(), "unknown")


# Display-only manual overrides for the development_stage column (Spec 068).
# Loaded once at import. Affects ONLY development_stage / development_stage_source;
# does NOT mutate archetype, tier_commercial, stage_bucket, lead_program_phase,
# scoring, ranker, selector, eligibility, EES, Event EV, Module 5 cohorts, or QA.
def _load_development_stage_overrides() -> Dict[str, str]:
    """Read production_data/development_stage_overrides.json -> {ticker: stage}.

    Returns empty dict on missing file, malformed JSON, or unknown stage values.
    Entries with stages outside DEVELOPMENT_STAGE_VALUES are silently dropped to
    prevent invalid values from leaking into rankings.csv.
    """
    p = Path(__file__).resolve().parent / "production_data" / "development_stage_overrides.json"
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    entries = raw.get("entries", {}) if isinstance(raw, dict) else {}
    if not isinstance(entries, dict):
        return {}
    out: Dict[str, str] = {}
    for ticker, payload in entries.items():
        if not isinstance(payload, dict):
            continue
        stage = payload.get("stage")
        if isinstance(stage, str) and stage in DEVELOPMENT_STAGE_VALUES:
            out[str(ticker).upper()] = stage
    return out


_DEVELOPMENT_STAGE_OVERRIDES: Dict[str, str] = _load_development_stage_overrides()


def _classify_cohort_churn_severity(churn_pct: float, threshold_pct: float = 10.0) -> str:
    """Severity bucket for churn-alert (info < threshold <= warn)."""
    if churn_pct >= threshold_pct:
        return "warn"
    return "info"


def _write_cohort_churn_alert(
    csv_rows: List[Dict[str, Any]],
    snap_path: Path,
    threshold_pct: float = 10.0,
    prior_snapshot_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """
    Compute today vs prior-snapshot v2 cohort delta and write a JSON alert
    to `snap_path/cohort_churn_alert.json`.

    Returns the alert dict (None if no prior snapshot found). Does not
    modify csv_rows or any scoring/decision-engine output.

    `prior_snapshot_dir` defaults to `snap_path.parent`, which is only correct
    when writing directly into the real snapshots root. Production writes into a
    throwaway staging dir, so the caller must pass the real root or no prior
    snapshot is ever found (churn permanently null at severity "info").
    """
    if snap_path is None:
        return None
    snapshots_root = Path(prior_snapshot_dir) if prior_snapshot_dir else snap_path.parent
    current_dir = snap_path.name
    _date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    try:
        candidates = sorted(d for d in os.listdir(snapshots_root) if _date_re.match(d) and d < current_dir)
    except OSError:
        candidates = []
    prior_dir = None
    for d in reversed(candidates):
        if (snapshots_root / d / "rankings.csv").exists():
            prior_dir = d
            break
    today_cohort = {r.get("ticker", "") for r in csv_rows if r.get("ranker_v2_score") or ""}
    today_cohort.discard("")

    if prior_dir is None:
        alert = {
            "as_of": current_dir,
            "prior_snapshot": None,
            "today_cohort_n": len(today_cohort),
            "prior_cohort_n": None,
            "churn_n": None,
            "churn_pct": None,
            "severity": "info",
            "names_left": [],
            "names_joined": [],
            "threshold_pct": threshold_pct,
            "note": "no prior snapshot available for comparison",
        }
    else:
        prior_path = snapshots_root / prior_dir / "rankings.csv"
        prior_cohort: set = set()
        try:
            with open(prior_path, "r", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r.get("ranker_v2_score") or "":
                        prior_cohort.add(r.get("ticker", ""))
            prior_cohort.discard("")
        except Exception:
            prior_cohort = set()
        names_left = sorted(prior_cohort - today_cohort)
        names_joined = sorted(today_cohort - prior_cohort)
        cohort_n = max(len(today_cohort), len(prior_cohort))
        churn_n = len(names_left) + len(names_joined)
        # Symmetric churn pct: count both directions on the larger cohort base.
        churn_pct = (max(len(names_left), len(names_joined)) / cohort_n * 100.0) if cohort_n else 0.0
        alert = {
            "as_of": current_dir,
            "prior_snapshot": prior_dir,
            "today_cohort_n": len(today_cohort),
            "prior_cohort_n": len(prior_cohort),
            "churn_n": churn_n,
            "churn_pct": round(churn_pct, 2),
            "severity": _classify_cohort_churn_severity(churn_pct, threshold_pct),
            "names_left": names_left,
            "names_joined": names_joined,
            "threshold_pct": threshold_pct,
            "note": (
                "ranker_v2 cohort_top_n=60; ~5% daily churn is typical. "
                ">=10% trips a warn — see artifacts/ranker_v2_cohort_audit_2026-04-26.md."
            ),
        }

    out_path = snap_path / "cohort_churn_alert.json"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(alert, f, indent=2, sort_keys=True)
    except OSError as exc:
        logger.warning(f"Could not write cohort_churn_alert.json: {exc}")
    return alert


def _annotate_cohort_membership_streaks(
    csv_rows: List[Dict[str, Any]],
    snap_path: Path,
    max_walkback_days: int = 30,
    prior_snapshot_dir: Optional[Path] = None,
) -> None:
    """
    Display-only annotation of `cohort_membership` ("in"/"out") and
    `cohort_membership_streak` (consecutive prior snapshots with same status)
    on each row in csv_rows.

    Walks back through dated snapshot dirs under `prior_snapshot_dir` (plain
    `YYYY-MM-DD` only — skips `__pre_…` and other suffixed dirs) and reads their
    rankings.csv files; membership = "in" iff ranker_v2_score is populated.
    Streak is capped by max_walkback_days. Pure-read on csv_rows except for the
    two new fields.

    `prior_snapshot_dir` defaults to `snap_path.parent`, which is only correct
    when writing directly into the real snapshots root. Production writes into a
    throwaway staging dir, so the caller must pass the real root or no history
    is ever found (streak pinned at 1 for every ticker).
    """
    if snap_path is None:
        for _r in csv_rows:
            _r["cohort_membership"] = "in" if _r.get("ranker_v2_score") else "out"
            _r["cohort_membership_streak"] = 1
        return

    snapshots_root = Path(prior_snapshot_dir) if prior_snapshot_dir else snap_path.parent
    current_dir = snap_path.name

    # Only consider plain YYYY-MM-DD sibling dirs strictly before today's.
    _date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    try:
        candidates = [d for d in os.listdir(snapshots_root) if _date_re.match(d) and d < current_dir]
    except OSError:
        candidates = []
    candidates.sort()
    candidates = candidates[-max_walkback_days:]

    # Load prior membership: {date: {ticker: "in"|"out"}}
    prior: Dict[str, Dict[str, str]] = {}
    for d in candidates:
        path = snapshots_root / d / "rankings.csv"
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                membership = {}
                for r in csv.DictReader(f):
                    tk = r.get("ticker", "")
                    if tk:
                        membership[tk] = "in" if (r.get("ranker_v2_score") or "") else "out"
                prior[d] = membership
        except Exception:
            continue

    ordered_dates = [d for d in candidates if d in prior]

    for row in csv_rows:
        ticker = row.get("ticker", "")
        current_state = "in" if (row.get("ranker_v2_score") or "") else "out"
        row["cohort_membership"] = current_state
        streak = 1
        for d in reversed(ordered_dates):
            prev_state = prior.get(d, {}).get(ticker)
            if prev_state == current_state:
                streak += 1
            else:
                break
        row["cohort_membership_streak"] = streak


def _derive_development_stage(row: Dict[str, Any], m4_lead_phase) -> Tuple[str, str, str]:
    """
    Display-only derivation of (development_stage, source, lead_program_phase_raw).

    Precedence (per spec):
      0. development_stage_overrides.json    -> override / source="override" (Spec 068)
      1. archetype starts with "commercial_" -> "commercial" / source="archetype"
      2. tier_commercial non-empty           -> "commercial" / source="tier_commercial"
      3. Module 4 lead_phase populated       -> normalize / source="module_4_lead_phase"
      4. lead_program_phase populated        -> normalize / source="lead_program_phase"
      5. otherwise                           -> "unknown" / source="unknown"

    Override map (Spec 068, Lane 1) is display-only and consulted first. It does
    NOT mutate archetype, tier_commercial, lead_program_phase, scoring, ranker,
    selector, eligibility, EES, Event EV, Module 5 cohorts, or QA gates.

    Never reads composite_score / final_score / actionable_rank / clinical_score
    inputs; never writes anywhere except the returned tuple.
    """
    archetype = str(row.get("archetype") or "").strip()
    tier_commercial = str(row.get("tier_commercial") or "").strip()
    raw_m4 = str(m4_lead_phase or "").strip()
    raw_prog = str(row.get("lead_program_phase") or "").strip()

    ticker = str(row.get("ticker") or "").upper()
    if ticker in _DEVELOPMENT_STAGE_OVERRIDES:
        return (_DEVELOPMENT_STAGE_OVERRIDES[ticker], "override", raw_m4 or raw_prog)

    if archetype.startswith("commercial_"):
        return ("commercial", "archetype", raw_m4 or raw_prog)
    if tier_commercial:
        return ("commercial", "tier_commercial", raw_m4 or raw_prog)
    if raw_m4:
        return (
            _normalize_phase_to_development_stage(raw_m4),
            "module_4_lead_phase",
            raw_m4,
        )
    if raw_prog:
        return (
            _normalize_phase_to_development_stage(raw_prog),
            "lead_program_phase",
            raw_prog,
        )
    return ("unknown", "unknown", "")


def apply_clinical_activity_filter(
    m4_scores: List[Dict[str, Any]],
    min_trials: int = 5,
    min_phase: str = "phase1",
    archetypes: Dict[str, str] = None,
) -> tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Filter tickers based on clinical activity thresholds.

    Non-drug-developer archetypes (commercial pharma, platform companies, etc.)
    are exempt from exclusion since their business model doesn't depend on
    proprietary clinical trials.

    Args:
        m4_scores: Module 4 clinical scores list
        min_trials: Minimum number of trials required
        min_phase: Minimum lead phase required (preclinical, phase1, phase2, phase3, approved)
        archetypes: Dict mapping ticker -> archetype string. Tickers with archetypes
                    other than "drug_developer" are exempt from exclusion.

    Returns:
        Tuple of (excluded_tickers, exclusion_details, exemption_details)
    """
    min_phase_order = PHASE_ORDER.get(min_phase.lower(), 1)

    excluded = []
    exclusion_details = []
    exemption_details = []

    for score in m4_scores:
        ticker = score.get("ticker", "")
        trial_count = score.get("n_trials_unique", score.get("trial_count", 0))
        lead_phase = str(score.get("lead_phase", "preclinical")).lower()
        phase_order = PHASE_ORDER.get(lead_phase, 0)

        reasons = []

        # Check minimum trials
        if trial_count < min_trials:
            reasons.append(f"trials={trial_count}<{min_trials}")

        # Check minimum phase
        if phase_order < min_phase_order:
            reasons.append(f"phase={lead_phase}<{min_phase}")

        if reasons:
            archetype = archetypes.get(ticker, "drug_developer") if archetypes else "drug_developer"
            if archetype != "drug_developer":
                # Exempt: non-drug-developer archetype doesn't need clinical trials
                exemption_details.append(
                    {
                        "ticker": ticker,
                        "reason": "clinical_gate_exempt",
                        "archetype": archetype,
                        "trial_count": trial_count,
                        "lead_phase": lead_phase,
                    }
                )
            else:
                excluded.append(ticker)
                exclusion_details.append(
                    {
                        "ticker": ticker,
                        "reason": "clinical_activity_filter",
                        "details": ", ".join(reasons),
                        "trial_count": trial_count,
                        "lead_phase": lead_phase,
                    }
                )

    return excluded, exclusion_details, exemption_details


def parse_catalyst_window(window_str: str) -> tuple[int, int]:
    """
    Parse catalyst window string 'START-END' into tuple of ints.

    Args:
        window_str: String like '15-45'

    Returns:
        Tuple (start_days, end_days)

    Raises:
        ValueError: If format is invalid or values out of range
    """
    if not window_str or "-" not in window_str:
        raise ValueError(f"Invalid catalyst window format: '{window_str}'. Expected 'START-END' (e.g., '15-45')")

    parts = window_str.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid catalyst window format: '{window_str}'. Expected 'START-END' (e.g., '15-45')")

    try:
        start = int(parts[0])
        end = int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid catalyst window values: '{window_str}'. Both values must be integers.")

    if start < 0 or end < 0:
        raise ValueError(f"Catalyst window values must be non-negative: got {start}-{end}")
    if start >= end:
        raise ValueError(f"Catalyst window start must be less than end: got {start}-{end}")
    if end > 365:
        raise ValueError(f"Catalyst window end exceeds 365 days: got {end}")

    return (start, end)


def compute_catalyst_decay_weight(
    days_to_event: int,
    window_start: int,
    window_end: int,
    decay_mode: str,
    half_life_days: int = 30,
) -> float:
    """
    Compute decay weight for a catalyst event based on days until event.

    Args:
        days_to_event: Days from as_of_date to catalyst event
        window_start: Start of optimal window (days)
        window_end: End of optimal window (days)
        decay_mode: 'step', 'linear', or 'exp'
        half_life_days: Half-life for exponential decay (only used if decay_mode='exp')

    Returns:
        Weight in [0.0, 1.0] where 1.0 = fully within window
    """
    import math

    # Events in the past get zero weight
    if days_to_event < 0:
        return 0.0

    # Events within optimal window get full weight
    if window_start <= days_to_event <= window_end:
        return 1.0

    if decay_mode == "step":
        # Binary: full weight in window, zero outside
        return 0.0

    elif decay_mode == "linear":
        # Linear taper: 1.0 at window edge, 0.0 at 2x window distance
        if days_to_event < window_start:
            # Too soon - taper down as we get closer to 0
            if window_start == 0:
                return 1.0  # Edge case: window starts at 0
            return max(0.0, days_to_event / window_start)
        else:
            # Beyond window_end - taper down
            overshoot = days_to_event - window_end
            taper_range = window_end  # Taper over same distance as window
            return max(0.0, 1.0 - (overshoot / max(1, taper_range)))

    elif decay_mode == "exp":
        # Exponential decay with half-life
        hl = max(half_life_days, 1)  # guard against division by zero
        if days_to_event < window_start:
            # Too soon - exponential approach
            distance = window_start - days_to_event
            return math.exp(-0.693 * distance / hl)  # 0.693 = ln(2)
        else:
            # Beyond window_end - exponential decay
            distance = days_to_event - window_end
            return math.exp(-0.693 * distance / hl)

    else:
        # Unknown mode - fall back to step
        return 0.0 if days_to_event < window_start or days_to_event > window_end else 1.0


def _force_deterministic_generated_at(obj: Any, generated_at: str) -> None:
    """Recursively overwrite provenance.generated_at for byte-identical outputs."""
    if isinstance(obj, dict):
        prov = obj.get("provenance")
        if isinstance(prov, dict) and "generated_at" in prov:
            prov["generated_at"] = generated_at
        for v in obj.values():
            _force_deterministic_generated_at(v, generated_at)
    elif isinstance(obj, list):
        for item in obj:
            _force_deterministic_generated_at(item, generated_at)


def _deterministic_timestamp(as_of_date: str) -> str:
    """Return the deterministic timestamp used in reproducible artifacts."""
    return as_of_date + DETERMINISTIC_TIMESTAMP_SUFFIX


def validate_as_of_date_param(as_of_date: str) -> None:
    """
    Validate as_of_date format with strict security checks.

    Raises:
        ValueError: If date format is invalid

    Note:
        Does not compare to date.today() to maintain time-invariance.
        Lookahead protection is enforced via PIT filters in modules.
    """
    # SECURITY: Use hardened date validation to prevent injection
    try:
        validate_date_format(as_of_date, "as_of_date")
    except ValueError as e:
        raise ValueError(f"Invalid as_of_date format '{as_of_date}': must be YYYY-MM-DD") from e

    # Also validate via existing utility for compatibility
    try:
        normalize_date(as_of_date)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid as_of_date format '{as_of_date}': must be YYYY-MM-DD") from e

    # NOTE: Do not compare to date.today() here (wall-clock dependency breaks time-invariance).
    # Lookahead protection should be enforced via PIT filters and/or input snapshot dating.


def _live_mode_today_iso(today: Optional[date] = None) -> str:
    """Return today's ISO date for explicit live-mode guards only."""
    return (today or date.today()).isoformat()


def _require_current_as_of_date_for_live_modes(
    as_of_date: str,
    live_modes: Dict[str, bool],
    *,
    today: Optional[date] = None,
) -> None:
    """Fail closed when live data sources are requested for historical runs."""
    enabled = sorted(name for name, is_enabled in live_modes.items() if is_enabled)
    if not enabled:
        return

    today_iso = _live_mode_today_iso(today)
    if as_of_date != today_iso:
        joined = ", ".join(enabled)
        raise ValueError(
            f"{joined} only allowed when --as-of-date equals today ({today_iso}); "
            f"got {as_of_date}. Fetching live data for a historical date would violate PIT discipline."
        )


def _ctgov_fallback_pit_cutoff(as_of_date: str) -> str:
    """Standard PIT cutoff for unfiltered CTGov fallback records."""
    return (date.fromisoformat(as_of_date) - timedelta(days=1)).isoformat()


def _filter_ctgov_fallback_records_for_pit(
    trial_records: List[Dict[str, Any]],
    as_of_date: str,
) -> Tuple[List[Dict[str, Any]], str]:
    """Filter unversioned CTGov fallback records to standard prior-day PIT availability."""
    pit_cutoff = _ctgov_fallback_pit_cutoff(as_of_date)
    filtered = [
        r
        for r in trial_records
        if (r.get("last_update_posted") or "9999")[:10] <= pit_cutoff
        and (r.get("first_posted") or "9999")[:10] <= pit_cutoff
    ]
    return filtered, pit_cutoff


def _compute_run_input_hashes(data_dir: Path) -> Dict[str, str]:
    """Hash primary run inputs, including score-affecting non-JSON inputs."""
    content_hashes: Dict[str, str] = {}
    for json_file in sorted(data_dir.glob("*.json")):
        if json_file.name.startswith(GENERATED_JSON_OUTPUT_PREFIXES):
            continue  # Skip generated sidecars - they're outputs, not inputs
        content_hashes[json_file.name] = hashlib.sha256(json_file.read_bytes()).hexdigest()[:16]

    for csv_name in SCORE_AFFECTING_CSV_INPUTS:
        csv_file = data_dir / csv_name
        if csv_file.exists():
            content_hashes[csv_file.name] = hashlib.sha256(csv_file.read_bytes()).hexdigest()[:16]

    return dict(sorted(content_hashes.items()))


def _sha256_prefixed(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_governance_envelope(
    *,
    as_of_date: str,
    run_metadata: Dict[str, Any],
) -> Dict[str, str]:
    """Build top-level deterministic governance metadata for primary output."""
    parameter_keys = (
        "version",
        "enhancements_enabled",
        "catalyst_window",
        "catalyst_decay",
        "catalyst_half_life_days",
        "enable_smart_money",
        "smart_money_source",
        "min_component_coverage",
        "min_component_coverage_mode",
        "defensive_multiplier_enabled",
        "defensive_config",
        "position_sizing_enabled",
        "pit_mode",
        "catalyst_mode",
    )
    parameter_payload = {key: run_metadata.get(key) for key in parameter_keys}
    parameters_hash = _sha256_prefixed(parameter_payload)
    run_id_payload = {
        "as_of_date": as_of_date,
        "input_hashes": run_metadata.get("input_hashes", {}),
        "parameters_hash": parameters_hash,
        "schema_version": GOVERNANCE_SCHEMA_VERSION,
        "score_version": VERSION,
    }
    return {
        "run_id": hashlib.sha256(
            json.dumps(run_id_payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()[:16],
        "score_version": VERSION,
        "schema_version": GOVERNANCE_SCHEMA_VERSION,
        "parameters_hash": parameters_hash,
        "pit_cutoff": _ctgov_fallback_pit_cutoff(as_of_date),
        "as_of_date": as_of_date,
    }


def _maybe_refresh_price_history_live(
    *,
    as_of_date: str,
    data_dir: Path,
    enabled: bool,
    today: Optional[date] = None,
    refresh_func: Optional[Callable[[List[str]], int]] = None,
) -> bool:
    """Refresh price history only when explicitly requested for a current-date live run."""
    if not enabled:
        return False

    _require_current_as_of_date_for_live_modes(
        as_of_date,
        {"--refresh-price-history-live": True},
        today=today,
    )

    price_csv = data_dir / "price_history.csv"
    if not price_csv.exists():
        logger.info("[PRICE_REFRESH] price_history.csv not found; skipping live refresh")
        return False

    try:
        if refresh_func is None:
            from scripts.refresh_price_history import main as refresh_func

        logger.info("[PRICE_REFRESH] Refreshing price_history.csv in explicit live mode...")
        refresh_rc = refresh_func(["--price-csv", str(price_csv), "--days-back", "5"])
        if refresh_rc == 0:
            logger.info("[PRICE_REFRESH] Done")
        else:
            logger.warning("[PRICE_REFRESH] Non-zero return: %d", refresh_rc)
        return True
    except Exception as exc:
        logger.debug("Price refresh skipped: %s", exc)
        return False


def _atomic_write_text(path: Path, content: str) -> None:
    """Write text content atomically via temp-file-then-rename.

    Used for diagnostic sidecars where partial writes could confuse
    downstream consumers. Not needed for primary outputs (rankings.csv,
    metadata.json) which already have their own atomic write patterns.
    """
    import tempfile as _tf

    fd, tmp = _tf.mkstemp(dir=path.parent, prefix=".tmp_", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        Path(tmp).replace(path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_json_data(
    filepath: Path, description: str, max_size_mb: float = MAX_JSON_FILE_SIZE_MB, base_dir: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Load JSON data file with validation and security checks.

    Args:
        filepath: Path to JSON file
        description: Human-readable description for error messages
        max_size_mb: Maximum file size in MB (default: 100MB)
        base_dir: If provided, validate path is within this directory

    Returns:
        List of records

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
        FileSizeError: If file exceeds size limit
        PathTraversalError: If path escapes base_dir
        SymlinkError: If file is a symlink
    """
    # SECURITY: Validate file size before loading into memory
    try:
        validate_file_size(filepath, max_size_mb)
    except FileNotFoundError:
        raise FileNotFoundError(f"{description} file not found: {filepath}")

    # SECURITY: Check for symlinks (potential directory traversal)
    if filepath.is_symlink():
        raise SymlinkError(f"{description} file is a symbolic link (security risk): {filepath}")

    # SECURITY: Validate path is within expected directory
    if base_dir:
        try:
            validate_path_within_base(filepath, base_dir)
        except PathTraversalError as e:
            raise PathTraversalError(f"{description} path validation failed: {e}") from e

    # Load with timeout protection for large files
    try:
        with operation_timeout(60, f"Loading {description}"):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
    except UnicodeDecodeError as e:
        raise ValueError(f"{description} file is not valid UTF-8: {filepath}") from e

    if not isinstance(data, list):
        raise ValueError(f"{description} must be a JSON array, got {type(data)}")

    logger.debug(f"Loaded {description}: {len(data)} records from {filepath}")
    return data


try:
    import orjson as _orjson
except ImportError:
    _orjson = None  # type: ignore[assignment]


def _load_json_threadsafe(filepath: Path, description: str) -> List[Dict[str, Any]]:
    """JSON load safe for ThreadPoolExecutor workers.

    signal.alarm (used by operation_timeout) is main-thread-only on POSIX, so
    load_json_data cannot be called from worker threads. This variant preserves
    the size and symlink security checks but omits the SIGALRM timeout wrapper.
    Uses orjson when available (3–5× faster than stdlib for large files).
    """
    try:
        validate_file_size(filepath, MAX_JSON_FILE_SIZE_MB)
    except FileNotFoundError:
        raise FileNotFoundError(f"{description} file not found: {filepath}")
    if filepath.is_symlink():
        raise SymlinkError(f"{description} file is a symbolic link: {filepath}")
    if _orjson is not None:
        data = _orjson.loads(filepath.read_bytes())
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{description} must be a JSON array, got {type(data)}")
    return data


def _load_json_any_threadsafe(filepath: Path, description: str) -> Any:
    """Like _load_json_threadsafe but accepts a dict or list at the JSON root.

    Used for files such as morningstar_price_history.json whose root is an
    object rather than an array.
    """
    try:
        validate_file_size(filepath, MAX_JSON_FILE_SIZE_MB)
    except FileNotFoundError:
        raise FileNotFoundError(f"{description} file not found: {filepath}")
    if filepath.is_symlink():
        raise SymlinkError(f"{description} file is a symbolic link: {filepath}")
    if _orjson is not None:
        return _orjson.loads(filepath.read_bytes())
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_pit_json_worker(path: Path) -> dict:
    """Load a single PIT financial JSON file for use in ThreadPoolExecutor workers."""
    if _orjson is not None:
        return _orjson.loads(path.read_bytes())
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# =============================================================================
# HOLDINGS SCHEMA VALIDATION (fail-loud, forward compatible)
# =============================================================================
HOLDINGS_DETAILED_SCHEMA_NAMES = {"holdings_detailed", "holdings_history"}
HOLDINGS_DETAILED_SCHEMA_MIN_MAJOR = 1
# Legacy schema versions from extract_13f_history.py (version field only, no name)
HOLDINGS_LEGACY_VERSIONS = {"13f_holdings_snapshot_v1"}


def _schema_major(version_val) -> int:
    """Extract major version number from version string (e.g., '1.2.3' -> 1)."""
    if version_val is None:
        return 0
    s = str(version_val).strip()
    try:
        return int(s.split(".", 1)[0])
    except Exception as exc:
        logger.warning("Schema version parse failed for %r: %s", version_val, exc)
        return 0


def _validate_holdings_detailed_payload(schema: dict, tickers: dict) -> None:
    """
    Validate holdings_detailed.json payload shape and schema version.

    Accepts:
    - New format: {"name": "holdings_detailed", "version": "1.x"}
    - Legacy format: {"version": "13f_holdings_snapshot_v1"} (from extract_13f_history.py)

    Raises RuntimeError if validation fails (fail-loud to prevent silent fallback).
    """
    name = (schema or {}).get("name")
    version = (schema or {}).get("version")

    # Accept legacy format from extract_13f_history.py
    if name is None and version in HOLDINGS_LEGACY_VERSIONS:
        # Legacy schema is valid
        pass
    else:
        # New format: require name in allowlist and major version >= 1
        major = _schema_major(version)
        if name not in HOLDINGS_DETAILED_SCHEMA_NAMES or major < HOLDINGS_DETAILED_SCHEMA_MIN_MAJOR:
            raise RuntimeError(
                f"holdings_detailed.json schema mismatch: name={name!r}, version={version!r}. "
                f"Expected name in {sorted(HOLDINGS_DETAILED_SCHEMA_NAMES)} and major>={HOLDINGS_DETAILED_SCHEMA_MIN_MAJOR}, "
                f"or legacy version in {sorted(HOLDINGS_LEGACY_VERSIONS)}."
            )
    if not isinstance(tickers, dict) or not tickers:
        raise RuntimeError("holdings_detailed.json invalid: expected non-empty dict under top-level 'tickers'.")
    # Shallow shape check on first non-private ticker
    sample_key = next((k for k in tickers.keys() if not str(k).startswith("_")), None)
    if not sample_key:
        raise RuntimeError("holdings_detailed.json invalid: no ticker keys found.")
    sample = tickers.get(sample_key, {})
    h = (sample or {}).get("holdings", {})
    if not (isinstance(h, dict) and "current" in h and "prior" in h and "filings_metadata" in (sample or {})):
        raise RuntimeError(
            f"holdings_detailed.json invalid ticker shape for {sample_key!r}: "
            "expected holdings.current/prior and filings_metadata."
        )


def _load_manager_registry(data_dir: Path = None) -> Dict[str, Dict]:
    """
    Load manager registry and build CIK to manager info mapping.

    Returns dict: {cik: {"name": str, "tier": int}}
    """
    registry_paths = [
        Path("production_data/manager_registry.json"),
    ]
    if data_dir:
        registry_paths.insert(0, data_dir / "manager_registry.json")

    for registry_path in registry_paths:
        if registry_path.exists():
            try:
                with open(registry_path) as f:
                    registry = json.load(f)

                manager_map = {}
                # Elite Core managers are Tier 1
                for mgr in registry.get("elite_core", []):
                    cik = mgr.get("cik", "").zfill(10)
                    if cik:
                        manager_map[cik] = {"name": mgr.get("name", f"Manager_{cik[-4:]}"), "tier": 1}
                # Conditional managers are Tier 2
                for mgr in registry.get("conditional", []):
                    cik = mgr.get("cik", "").zfill(10)
                    if cik:
                        manager_map[cik] = {"name": mgr.get("name", f"Manager_{cik[-4:]}"), "tier": 2}
                return manager_map
            except Exception as e:
                logger.warning(f"Failed to load manager registry from {registry_path}: {e}")

    return {}


def _convert_holdings_to_coinvest(
    holdings_snapshots: Dict[str, Any],
    data_dir: Path = None,
    as_of_date: str = None,
    audit_out: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict]:
    """
    Convert holdings_snapshots.json format to coinvest_signals format for Module 5.

    Supports both detailed dict format and simplified list format.

    CONVICTION-WEIGHTED OVERLAP (Baker-style):
    For each Tier-1 holder, compute:
      - pos_w = clamp(sqrt(position_pct / 0.01), 0.5, 2.0)  # 1% ≈ 1.0x
      - chg_w = 1.25 (new/add), 1.0 (hold), 0.75 (trim), 0.5 (exit)
      - recency_w = clamp(1.5 - days_since_filing/180, 0.7, 1.5)
      - tier_w = 1.0 (T1), 0.6 (T2), 0.2 (unknown)

    conviction_overlap = Σ tier_w * pos_w * chg_w * recency_w

    holdings_snapshots format (detailed dict):
    {
        "TICKER": {
            "holdings": {
                "current": {"CIK": {"value_kusd": 123456, "shares": 50000}},
                "prior": {"CIK": {"value_kusd": 100000, "shares": 40000}}
            },
            "filings_metadata": {
                "CIK": {"total_value_kusd": 13800000, "filed_at": "2024-11-14T00:00:00"}
            }
        }
    }

    coinvest_signals format expected by Module 5:
    {
        "TICKER": {
            "coinvest_overlap_count": int,
            "coinvest_holders": [str, ...],
            "position_changes": {"name": "NEW"|"INCREASE"|"DECREASE"|"EXIT"|"HOLD"},
            "holder_tiers": {"name": tier_int},
            # NEW conviction fields (Baker-style):
            "conviction_overlap": float,
            "tier1_conviction_overlap": float,
            "max_tier1_position_pct": float,
            "days_since_latest_filing": int,
            "tier1_count": int,
        }
    }

    Position change types:
    - NEW: Manager in current but not in prior (new position initiated)
    - INCREASE: Manager increased position value by >10%
    - DECREASE: Manager decreased position value by >10%
    - EXIT: Manager in prior but not in current (position closed)
    - HOLD: Position relatively unchanged (±10%)
    """
    import math
    from datetime import datetime

    # Load manager registry for name resolution
    MANAGER_REGISTRY = _load_manager_registry(data_dir)
    if MANAGER_REGISTRY:
        logger.info(f"  Loaded {len(MANAGER_REGISTRY)} managers from registry for name resolution")

    # Threshold for detecting meaningful position changes (10%)
    CHANGE_THRESHOLD = 0.10

    # Conviction weight constants (Baker-style)
    TIER_WEIGHTS = {1: 1.0, 2: 0.6, 3: 0.2, 0: 0.2}  # T1 > T2 > unknown
    CHANGE_WEIGHTS = {
        "NEW": 1.25,
        "INCREASE": 1.25,
        "HOLD": 1.0,
        "DECREASE": 0.75,
        "EXIT": 0.5,
    }

    # Parse as_of_date for recency calculation (PIT-safe: no datetime.now() fallback)
    ref_date = None
    if as_of_date:
        try:
            ref_date = datetime.strptime(as_of_date, "%Y-%m-%d")
        except (ValueError, TypeError):
            logger.warning(f"  Invalid as_of_date '{as_of_date}' - recency_w will default to 1.0")

    coinvest_signals = {}

    # PIT audit counters (persisted into run_metadata via audit_out)
    skipped_future_filings = 0
    holder_count_pit_skips = 0  # future filings excluded from overlap/tier1 counts
    missing_total_value = 0
    filed_at_parse_failures = 0

    for ticker, ticker_data in holdings_snapshots.items():
        if not isinstance(ticker_data, dict):
            continue

        holdings = ticker_data.get("holdings", {})
        filings_metadata = ticker_data.get("filings_metadata", {})
        current = holdings.get("current", {})
        prior = holdings.get("prior", {})

        # Normalize ticker to uppercase for consistent lookups
        ticker = ticker.upper()

        if not current:
            continue

        # Count managers and identify tiers
        holder_names = []  # Current holders by name
        holder_tiers = {}  # Dict[name -> tier_int]
        position_changes = {}  # Dict[name -> change_type]

        # Conviction tracking
        conviction_overlap = 0.0
        tier1_conviction_overlap = 0.0
        max_tier1_position_pct = 0.0
        min_days_since_filing = 999
        tier1_count = 0

        # All CIKs that appear in either current or prior
        all_ciks = sorted(set(current.keys()) | set(prior.keys()))

        for cik in all_ciks:
            # Determine tier and resolve holder name from registry
            if cik in MANAGER_REGISTRY:
                holder_name = MANAGER_REGISTRY[cik]["name"]
                tier = MANAGER_REGISTRY[cik]["tier"]
            else:
                holder_name = f"Manager_{cik[-4:]}"
                tier = 2

            # Track current holders by name (not CIK).
            # PIT guard: only count holders whose filing was available as of ref_date.
            # Future filings (filed_at > ref_date) must not activate smart_money_score.
            # When filed_at is absent the holder is counted (AVAILABILITY_UNKNOWN --
            # matches conviction-path behaviour; flagged separately in the audit log).
            if cik in current:
                _hc_filed_at = None
                _hc_filed_str = (filings_metadata.get(cik, {}).get("filed_at", "") or "").strip()
                if _hc_filed_str:
                    try:
                        if _hc_filed_str.endswith("Z"):
                            _hc_filed_str = _hc_filed_str[:-1]
                        _hc_filed_at = datetime.fromisoformat(_hc_filed_str).replace(tzinfo=None)
                    except (ValueError, TypeError):
                        _hc_filed_at = None
                if ref_date is not None and _hc_filed_at is not None and _hc_filed_at > ref_date:
                    holder_count_pit_skips += 1
                else:
                    holder_names.append(holder_name)
                    if tier == 1:
                        tier1_count += 1

            # Store tier by holder name (int, not dict)
            holder_tiers[holder_name] = tier

            # Calculate position change
            current_val = current.get(cik, {}).get("value_kusd", 0) or 0
            prior_val = prior.get(cik, {}).get("value_kusd", 0) or 0

            if cik in current and cik not in prior:
                change_type = "NEW"
            elif cik not in current and cik in prior:
                change_type = "EXIT"
            elif prior_val > 0:
                change_pct = (current_val - prior_val) / prior_val
                if change_pct > CHANGE_THRESHOLD:
                    change_type = "INCREASE"
                elif change_pct < -CHANGE_THRESHOLD:
                    change_type = "DECREASE"
                else:
                    change_type = "HOLD"
            else:
                change_type = "NEW" if current_val > 0 else "HOLD"

            position_changes[holder_name] = change_type

            # =========================================================
            # CONVICTION CALCULATION (only for current holders)
            # PIT-safe: exclude future filings, no datetime.now()
            # =========================================================
            if cik not in current:
                continue  # Skip exited positions for conviction

            # Get manager's filing metadata
            manager_meta = filings_metadata.get(cik, {})

            # PIT CHECK: Parse filing date and exclude future filings
            filed_at_str = (manager_meta.get("filed_at", "") or "").strip()
            filed_at = None
            if filed_at_str:
                try:
                    # Normalize ISO format: handle "Z" suffix and timezone offsets
                    if filed_at_str.endswith("Z"):
                        filed_at_str = filed_at_str[:-1]
                    filed_at = datetime.fromisoformat(filed_at_str)
                    # Strip timezone to avoid aware/naive comparison errors
                    filed_at = filed_at.replace(tzinfo=None)
                except (ValueError, TypeError):
                    filed_at = None
                    filed_at_parse_failures += 1

            # PIT SAFETY: Skip holders with future filing dates
            if ref_date is not None and filed_at is not None:
                if filed_at > ref_date:
                    # Future filing - exclude from conviction (PIT violation)
                    skipped_future_filings += 1
                    continue

            # Get manager's total 13F value for position % calculation
            manager_total_kusd = manager_meta.get("total_value_kusd", 0) or 0
            if manager_total_kusd <= 0:
                missing_total_value += 1

            # pos_w: position size weight (sqrt scaling, 1% ≈ 1.0x)
            # position_pct is in PERCENT (e.g., 3.5 means 3.5% of manager's 13F)
            if manager_total_kusd > 0:
                position_pct = (current_val / manager_total_kusd) * 100  # In percent
                pos_w = max(0.5, min(2.0, math.sqrt(position_pct / 1.0)))
            else:
                position_pct = 0.0
                pos_w = 0.5  # Minimum weight if no total available

            # Track max Tier-1 position % (in percent, e.g., 5.2 means 5.2%)
            if tier == 1 and position_pct > max_tier1_position_pct:
                max_tier1_position_pct = position_pct

            # chg_w: change direction weight
            chg_w = CHANGE_WEIGHTS.get(change_type, 1.0)

            # recency_w: filing freshness weight (PIT-safe)
            if ref_date is not None and filed_at is not None:
                days_since = (ref_date - filed_at).days
                if days_since < min_days_since_filing:
                    min_days_since_filing = days_since
                recency_w = max(0.7, min(1.5, 1.5 - days_since / 180))
            else:
                # No ref_date or filing date: use neutral recency
                recency_w = 1.0

            # tier_w: tier weight
            tier_w = TIER_WEIGHTS.get(tier, 0.2)

            # Compute holder's conviction contribution
            holder_conviction = tier_w * pos_w * chg_w * recency_w
            conviction_overlap += holder_conviction

            # Track Tier-1 conviction separately
            if tier == 1:
                tier1_conviction_overlap += holder_conviction

        # Build signal dict with conviction fields
        coinvest_signals[ticker] = {
            # Existing fields (backward compatible)
            "coinvest_overlap_count": len(holder_names),
            "coinvest_holders": holder_names,
            "position_changes": position_changes,
            "holder_tiers": holder_tiers,
            # NEW: Conviction fields (Baker-style)
            "conviction_overlap": round(conviction_overlap, 4),
            "tier1_conviction_overlap": round(tier1_conviction_overlap, 4),
            "tier1_count": tier1_count,
            "max_tier1_position_pct": round(max_tier1_position_pct, 2),
            "days_since_latest_filing": min_days_since_filing if min_days_since_filing < 999 else None,
        }

    # Log PIT audit summary
    if (
        skipped_future_filings > 0
        or holder_count_pit_skips > 0
        or missing_total_value > 0
        or filed_at_parse_failures > 0
    ):
        logger.info(
            f"  PIT audit (conviction+counts): future_conviction_skips={skipped_future_filings}, "
            f"holder_count_pit_skips={holder_count_pit_skips}, "
            f"missing_total_value={missing_total_value}, filed_at_parse_failures={filed_at_parse_failures}"
        )

    # Persist audit counters for run_metadata
    if audit_out is not None:
        audit_out.update(
            {
                "as_of_date": as_of_date,
                "future_filing_skips": int(skipped_future_filings),
                "holder_count_pit_skips": int(holder_count_pit_skips),
                "missing_total_value": int(missing_total_value),
                "filed_at_parse_failures": int(filed_at_parse_failures),
            }
        )

    return coinvest_signals


def enrich_volatility_from_morningstar_prices(
    price_history_path: Path,
    market_data_by_ticker: Dict[str, Dict],
    min_trading_days: int = 60,
    preloaded_data: Optional[Dict] = None,
) -> int:
    """
    Compute annualized volatility from Morningstar daily close prices (HS377).

    Uses std(daily_log_returns) * sqrt(252) for each ticker with sufficient
    price history.  Injects ``volatility_252d`` into market_data_by_ticker
    so that Module 5's ``compute_volatility_adjustment`` receives real data.

    Args:
        price_history_path: Path to morningstar_price_history.json
        market_data_by_ticker: Dict to enrich (modified in place)
        min_trading_days: Minimum observations required (default 60)
        preloaded_data: Pre-parsed JSON dict (skips file I/O when provided)

    Returns:
        Number of tickers enriched
    """
    import json as _json
    import math

    if preloaded_data is not None:
        raw = preloaded_data
    else:
        try:
            with open(price_history_path, "r", encoding="utf-8") as fh:
                raw = _json.load(fh)
        except (OSError, ValueError):
            return 0

    records = raw.get("records", {})
    enriched = 0

    for ticker_key, ticker_data in records.items():
        if ticker_key == "metadata":
            continue

        ticker_upper = ticker_key.upper()
        if ticker_upper not in market_data_by_ticker:
            continue

        # Already has volatility_252d (e.g. from enrichment script)
        if market_data_by_ticker[ticker_upper].get("volatility_252d") is not None:
            enriched += 1
            continue

        # Extract HS377 daily close time-series
        hs377 = ticker_data.get("HS377")
        if not hs377:
            continue
        if isinstance(hs377, dict):
            ts = hs377.get("time_series", [])
        elif isinstance(hs377, list) and len(hs377) == 1 and isinstance(hs377[0], tuple):
            # Legacy tuple format: [("time_series", [...])]
            ts = hs377[0][1] if hs377[0][0] == "time_series" else []
        else:
            ts = []

        if len(ts) < min_trading_days + 1:
            continue

        # Parse close prices (sorted by date ascending in source)
        closes = []
        for point in ts:
            val = point.get("value") if isinstance(point, dict) else None
            if val is not None:
                try:
                    closes.append(float(val))
                except (ValueError, TypeError):
                    continue

        if len(closes) < min_trading_days + 1:
            continue

        # Daily log-returns
        returns = []
        for i in range(1, len(closes)):
            if closes[i - 1] > 0 and closes[i] > 0:
                returns.append(math.log(closes[i] / closes[i - 1]))

        if len(returns) < min_trading_days:
            continue

        # Annualize: std(daily_returns) * sqrt(252)
        mean_r = sum(returns) / len(returns)
        variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance)
        annualized_vol = std_dev * math.sqrt(252)

        market_data_by_ticker[ticker_upper]["volatility_252d"] = round(annualized_vol, 6)
        enriched += 1

    return enriched


# =============================================================================
# PRICE HISTORY ROW CACHE (production I/O optimization — behavior-preserving)
# =============================================================================
# price_history.csv (28MB+) was opened and fully re-parsed by every consumer
# (momentum, beta/alpha recompute, drawdown/beta-rsi hydration). This run-scoped
# cache parses it once and serves the EXACT csv.DictReader rows to all consumers,
# which each still rebuild their own filtered view — so outputs are byte-identical
# (proven by tests/test_golden_baseline.py). Keyed by (resolved path, mtime_ns,
# size); only the current file is retained so memory stays bounded. Read-only:
# callers must not mutate the returned row dicts.
_PRICE_HISTORY_ROW_CACHE: Dict[tuple, List[Dict[str, str]]] = {}


def _read_price_history_rows(price_history_path: Path) -> List[Dict[str, str]]:
    """Return cached csv.DictReader rows for price_history.csv (parsed once per run).

    Falls back to a direct uncached read if the file cannot be stat'd. The
    returned list MUST be treated as read-only by callers.
    """
    p = Path(price_history_path)
    try:
        st = p.stat()
        key = (str(p.resolve()), st.st_mtime_ns, st.st_size)
    except OSError:
        with open(p, "r", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    cached = _PRICE_HISTORY_ROW_CACHE.get(key)
    if cached is None:
        with open(p, "r", encoding="utf-8") as fh:
            cached = list(csv.DictReader(fh))
        _PRICE_HISTORY_ROW_CACHE.clear()  # retain only the current file → bounded memory
        _PRICE_HISTORY_ROW_CACHE[key] = cached
        logger.info(f"[price-cache] parsed price_history.csv once ({len(cached)} rows); reused downstream")
    else:
        logger.debug(f"[price-cache] reusing parsed price_history.csv ({len(cached)} rows)")
    return cached


def compute_momentum_from_price_history(
    price_history_path: Path, as_of_date: str, market_data_by_ticker: Dict[str, Dict], xbi_ticker: str = "XBI"
) -> int:
    """
    Compute momentum returns from price_history.csv and inject into market_data.

    Computes return_20d, return_60d, return_120d for each ticker and XBI benchmark.

    Args:
        price_history_path: Path to price_history.csv
        as_of_date: Analysis date (YYYY-MM-DD)
        market_data_by_ticker: Dict to inject returns into (modified in place)
        xbi_ticker: Benchmark ticker (default XBI)

    Returns:
        Number of tickers enriched with momentum data
    """
    import csv
    from datetime import datetime, timedelta

    if not price_history_path.exists():
        logger.warning(f"Price history file not found: {price_history_path}")
        return 0

    # Parse as_of_date
    try:
        ref_date = datetime.strptime(as_of_date, "%Y-%m-%d").date()
    except ValueError:
        logger.warning(f"Invalid as_of_date format: {as_of_date}")
        return 0

    # Load price history into dict: ticker -> [(date, close), ...]
    logger.info(f"Loading price history from {price_history_path}...")
    prices_by_ticker = {}

    with __import__("contextlib").nullcontext():
        reader = _read_price_history_rows(price_history_path)
        for row in reader:
            ticker = row.get("ticker", "").upper()
            date_str = row.get("date", "")
            close_str = row.get("close", "")

            if not ticker or not date_str or not close_str:
                continue

            try:
                row_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                close_price = float(close_str)
            except (ValueError, TypeError):
                continue

            if ticker not in prices_by_ticker:
                prices_by_ticker[ticker] = []
            prices_by_ticker[ticker].append((row_date, close_price))

    logger.info(f"Loaded price history for {len(prices_by_ticker)} tickers")

    # Staleness check: warn if latest price date is far before as_of_date
    if prices_by_ticker:
        _max_dates = [max(pts, key=lambda x: x[0])[0] for pts in prices_by_ticker.values() if pts]
        if _max_dates:
            _latest_price_date = max(_max_dates)
            _staleness_days = (ref_date - _latest_price_date).days
            if _staleness_days > 7:
                logger.warning(
                    f"  Price history stale: latest date {_latest_price_date} is "
                    f"{_staleness_days} days before as_of_date {as_of_date} — "
                    f"momentum returns may be invalid"
                )

    # Sort each ticker's prices by date (most recent first for easy lookups)
    for ticker in prices_by_ticker:
        prices_by_ticker[ticker].sort(key=lambda x: x[0], reverse=True)

    def get_return(ticker: str, days_back: int) -> Optional[float]:
        """Get return over specified trading days (approximate)."""
        if ticker not in prices_by_ticker:
            return None

        prices = prices_by_ticker[ticker]
        if len(prices) < 2:
            return None

        # Find most recent price on or before ref_date
        current_price = None
        current_date = None
        for dt, price in prices:
            if dt <= ref_date:
                current_price = price
                current_date = dt
                break

        if current_price is None:
            return None

        # Find price approximately days_back trading days ago
        # Trading days ≈ calendar days * 252/365
        calendar_days_back = int(days_back * 365 / 252)
        target_date = current_date - timedelta(days=calendar_days_back)

        past_price = None
        for dt, price in prices:
            if dt <= target_date:
                past_price = price
                break

        if past_price is None or past_price == 0:
            return None

        return (current_price / past_price) - 1.0

    # Compute XBI benchmark returns first
    xbi_return_20d = get_return(xbi_ticker, 20)
    xbi_return_60d = get_return(xbi_ticker, 60)
    xbi_return_120d = get_return(xbi_ticker, 120)

    if xbi_return_60d is not None:
        _fmt = lambda v: f"{v:.2%}" if v is not None else "N/A"
        logger.info(
            f"XBI benchmark returns: 20d={_fmt(xbi_return_20d)}, 60d={_fmt(xbi_return_60d)}, 120d={_fmt(xbi_return_120d)}"
        )
    else:
        logger.warning(f"Could not compute XBI benchmark returns - XBI not in price history")

    # Compute returns for each ticker in market_data
    enriched_count = 0
    for ticker in market_data_by_ticker:
        # Skip if already has return data (don't overwrite 13F momentum)
        if market_data_by_ticker[ticker].get("return_60d") is not None:
            continue

        ticker_upper = ticker.upper()
        return_20d = get_return(ticker_upper, 20)
        return_60d = get_return(ticker_upper, 60)
        return_120d = get_return(ticker_upper, 120)

        if return_60d is not None:
            market_data_by_ticker[ticker]["return_20d"] = return_20d
            market_data_by_ticker[ticker]["return_60d"] = return_60d
            market_data_by_ticker[ticker]["return_120d"] = return_120d
            market_data_by_ticker[ticker]["xbi_return_20d"] = xbi_return_20d
            market_data_by_ticker[ticker]["xbi_return_60d"] = xbi_return_60d
            market_data_by_ticker[ticker]["xbi_return_120d"] = xbi_return_120d
            market_data_by_ticker[ticker]["trading_days_available"] = len(prices_by_ticker.get(ticker_upper, []))
            market_data_by_ticker[ticker]["_momentum_source"] = "price_history"
            enriched_count += 1

    logger.info(f"Enriched {enriched_count} tickers with momentum data from price history")
    return enriched_count


def write_json_output(filepath: Path, data: Dict[str, Any], secure: bool = True) -> None:
    """
    Write JSON output with deterministic formatting and security.

    Args:
        filepath: Output path
        data: Data to serialize
        secure: Use secure file permissions (default True)
    """
    # SECURITY: Create parent directory with secure permissions
    safe_mkdir(filepath.parent, mode=0o700)

    # Custom encoder for dataclass objects
    class DataclassEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            if hasattr(obj, "__dataclass_fields__"):
                return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
            if isinstance(obj, Decimal):
                return str(obj)
            if isinstance(obj, date):
                return obj.isoformat()
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, set):
                return sorted(list(obj))
            return super().default(obj)

    # Serialize to string first
    json_content = (
        json.dumps(
            data,
            indent=2,
            sort_keys=True,  # Deterministic key ordering
            ensure_ascii=False,
            cls=DataclassEncoder,
        )
        + "\n"
    )  # Trailing newline for diff-friendliness

    if secure:
        # SECURITY: Write atomically with secure permissions
        import os
        import tempfile

        fd, tmp_path = tempfile.mkstemp(dir=filepath.parent, prefix=".tmp_output_", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json_content)
            os.chmod(tmp_path, 0o600)
            Path(tmp_path).replace(filepath)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    else:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json_content)


# =============================================================================
# COLUMN DEFINITIONS & CONSTANTS — delegated to run_screen_columns.py
# =============================================================================

from run_screen_columns import (  # noqa: E402
    PHASE2_DEFAULT_HEALTH_THRESHOLDS_PATH,
    PHASE2_DEFAULT_RULESET_PATH,
    PHASE2_DEFAULT_TIER_FILTER,
    PHASE2_DEFAULT_TOP_K,
    PHASE2_PINNED_RULESET_ID,
    PHASE2_PINNED_THRESHOLDS_ID,
    PHASE2_PORTFOLIO_COLUMNS,
    PORTFOLIO_POSITIONS_COLUMNS,
    POSITIONS_TOP_K,
    SNAPSHOT_COLUMNS,
)

# =============================================================================
# INPUT MANIFEST – dependency registry
# =============================================================================


class InputDependency:
    """Describes one input file the pipeline may consume."""

    __slots__ = ("key", "path_template", "required", "condition", "resolved_from", "load_site")

    def __init__(
        self, key: str, path_template: str, required: bool, condition: str, resolved_from: str, load_site: str
    ):
        self.key = key
        self.path_template = path_template
        self.required = required
        self.condition = condition
        self.resolved_from = resolved_from
        self.load_site = load_site


DEPENDENCY_REGISTRY: List[InputDependency] = [
    # --- Required core inputs ---
    InputDependency("universe", "universe.json", True, "", "data_dir", "run_screen.py:4171"),
    InputDependency("financial_records", "financial_records.json", True, "", "data_dir", "run_screen.py:4185"),
    InputDependency("trial_records", "trial_records.json", True, "", "data_dir", "run_screen.py:4196"),
    InputDependency("market_data", "market_data.json", True, "", "data_dir", "run_screen.py:4200"),
    # --- Optional data-dir files ---
    InputDependency(
        "coinvest_signals", "coinvest_signals.json", False, "enable_coinvest", "data_dir", "run_screen.py:4310"
    ),
    InputDependency(
        "holdings_detailed", "holdings_detailed.json", False, "enable_coinvest", "data_dir", "run_screen.py:4320"
    ),
    InputDependency("price_history", "price_history.csv", False, "", "data_dir", "run_screen.py:4250"),
    InputDependency(
        "morningstar_prices", "morningstar_price_history.json", False, "", "data_dir", "run_screen.py:4245"
    ),
    InputDependency(
        "market_snapshot", "market_snapshot.json", False, "enable_enhancements", "data_dir", "run_screen.py:4350"
    ),
    InputDependency(
        "short_interest", "short_interest.json", False, "enable_enhancements", "data_dir", "run_screen.py:4360"
    ),
    InputDependency("fda_designations", "fda_designations.json", False, "", "data_dir", "run_screen.py:5110"),
    InputDependency("partnerships", "partnerships.json", False, "", "data_dir", "run_screen.py:5120"),
    InputDependency("quarterly_burn", "quarterly_burn_history.json", False, "", "data_dir", "run_screen.py:5143"),
    InputDependency("pdufa_dates", "pdufa_dates.json", False, "", "data_dir", "run_screen.py:5130"),
    # --- Dynamic / cache paths (path_template="" → resolved at runtime) ---
    InputDependency("ctgov_cache", "", False, "pit_mode", "cache/ctgov", "run_screen.py:4209"),
    InputDependency("sec_8k_cache", "", False, "sec_8k", "cache/sec", "run_screen.py:4026"),
    InputDependency("fda_adcom_cache", "", False, "fda_adcom", "cache/fda", "run_screen.py:4030"),
    InputDependency("decision_ruleset", "", False, "decision_phase2", "production_data", "run_screen.py:6544"),
]

MANIFEST_VERSION = "v1"

MIN_BARS_FOR_ESTIMATE = 126  # trading bars (rows), half of 252-bar window; below this drawdown is unreliable

VALID_DRAWDOWN_MISSING_REASONS = frozenset({"", "no_price_series", "series_too_short"})

# ---------------------------------------------------------------------------
# Price outlier / split-artifact detection
# ---------------------------------------------------------------------------
SPLIT_JUMP_THRESHOLD = 3.0  # +300% day-over-day → probable reverse split artifact
SPLIT_DROP_THRESHOLD = -0.75  # -75% day-over-day → probable forward split artifact


def _filter_price_outliers(
    series: List[tuple],
    jump_threshold: float = SPLIT_JUMP_THRESHOLD,
    drop_threshold: float = SPLIT_DROP_THRESHOLD,
) -> tuple:
    """Remove unadjusted split artifacts from a sorted (date, close) series.

    When a single-day jump exceeding the thresholds is detected, *all* rows
    before the discontinuity are dropped — not just the immediate predecessor.
    This is necessary because yfinance occasionally fails to back-adjust the
    full history, leaving the entire pre-split regime at the old price level.

    Returns ``(filtered_series, warnings)`` where *filtered_series* keeps
    only rows from the most recent price regime, and *warnings* is a list of
    dicts describing each detected split for logging.
    """
    if len(series) < 2:
        return series, []
    warnings = []
    # Track the latest split point — we'll truncate everything before it
    latest_split_idx = -1
    for i in range(1, len(series)):
        prev_close = series[i - 1][1]
        curr_close = series[i][1]
        if prev_close <= 0:
            continue
        pct = (curr_close / prev_close) - 1.0
        if pct >= jump_threshold or pct <= drop_threshold:
            warnings.append(
                {
                    "idx": i,
                    "date": series[i][0].isoformat() if hasattr(series[i][0], "isoformat") else str(series[i][0]),
                    "prev_close": prev_close,
                    "curr_close": curr_close,
                    "pct_change": round(pct, 4),
                    "flag": "reverse_split" if pct > 0 else "forward_split",
                }
            )
            latest_split_idx = i
    if latest_split_idx < 0:
        return series, []
    # Keep only from the split point onward (the current price regime)
    filtered = series[latest_split_idx:]
    return filtered, warnings


def _validate_price_splits(
    prices_by_ticker: Dict[str, List[tuple]],
    jump_threshold: float = SPLIT_JUMP_THRESHOLD,
    drop_threshold: float = SPLIT_DROP_THRESHOLD,
) -> Dict[str, List[dict]]:
    """Check all tickers in a price dict for split artifacts.

    Returns ``{ticker: [warning_dicts]}`` for tickers with detected outliers.
    Useful as a pre-flight check before hydration.
    """
    all_warnings: Dict[str, List[dict]] = {}
    for ticker, series in prices_by_ticker.items():
        _, warnings = _filter_price_outliers(series, jump_threshold, drop_threshold)
        if warnings:
            all_warnings[ticker] = warnings
    return all_warnings


def _corp_action_split_adjust(
    ticker: str,
    series: List[Any],
    as_of_date: str,
    registry: Any,
) -> Optional[List[Any]]:
    """Return a split-adjusted copy of ``series`` if ``ticker`` has a confirmed
    split on/before ``as_of_date`` in the corporate-actions registry, else None.

    ``series`` is a sorted list of ``(date, close)``. Each close is scaled by
    ``cumulative_split_factor(ticker, date, as_of_date)`` so pre-split prices are
    comparable to the post-split regime (the same basis the audit uses). Returning
    None signals "no confirmed corporate action" so the caller can preserve the
    raw fallback (a genuine crash the split heuristic misfired on).
    """
    if registry is None:
        return None
    from common.corporate_actions import cumulative_split_factor, get_splits_only

    if not get_splits_only(ticker, registry, as_of=as_of_date):
        return None
    adjusted = []
    for d, close in series:
        # cumulative_split_factor compares ISO date strings; series dates may be
        # datetime.date objects, so coerce.
        d_str = d.isoformat() if hasattr(d, "isoformat") else str(d)
        factor = cumulative_split_factor(ticker, d_str, as_of_date, registry)
        adjusted.append((d, close * factor))
    return adjusted


def _hydrate_drawdown(
    rec_by_ticker: Dict[str, Dict[str, Any]],
    price_history_path: Optional[Path],
    as_of_date: str,
    corporate_actions: Any = None,
) -> int:
    """Ensure ``defensive_features["drawdown"]`` is fresh for every record.

    **Step A — Recompute from price_history.csv**: For ALL tickers with
    sufficient price data, compute 252-day drawdown ``(current / peak) - 1.0``
    and write into ``defensive_features["drawdown"]``, *overwriting* any
    existing pipeline value.  This guarantees a single, consistent source of
    truth — the Morningstar pipeline may carry stale drawdowns when its
    underlying price feed lags the yfinance-sourced price_history.csv.

    **Step B — Fallback**: For tickers where price_history.csv has no data,
    keep any pipeline-provided ``drawdown`` (or alias ``drawdown_current``).

    Also writes ``defensive_features["drawdown_missing_reason"]``:
      - ``""`` — drawdown is present (computed or normalized)
      - ``"no_price_series"`` — ticker not found in price_history.csv
      - ``"series_too_short"`` — has price data but bars < MIN_BARS_FOR_ESTIMATE

    Returns the number of records hydrated (overwritten or newly computed).
    """
    import csv as _csv
    from datetime import datetime as _dt

    hydrated = 0

    # --- Step A: Compute from price_history.csv for ALL tickers ---
    if price_history_path is None or not price_history_path.exists():
        # No price CSV: fall back to pipeline values (normalize keys only)
        for rec in rec_by_ticker.values():
            df = rec.get("defensive_features")
            if df is None:
                df = {}
                rec["defensive_features"] = df
            raw_dd = df.get("drawdown")
            if raw_dd is not None:
                try:
                    df["drawdown"] = float(raw_dd)
                except (ValueError, TypeError):
                    df["drawdown"] = None
            if df.get("drawdown") is not None:
                df["drawdown_missing_reason"] = ""
                continue
            for alt_key in ("drawdown_current", "drawdown_60d"):
                val = df.get(alt_key)
                if val is not None:
                    try:
                        df["drawdown"] = float(val)
                    except (ValueError, TypeError):
                        continue
                    df["drawdown_missing_reason"] = ""
                    hydrated += 1
                    break
            else:
                df.setdefault("drawdown_missing_reason", "no_price_series")
        return hydrated

    try:
        ref_date = _dt.strptime(as_of_date, "%Y-%m-%d").date()
    except ValueError:
        logger.warning(f"_hydrate_drawdown: invalid as_of_date {as_of_date}")
        return hydrated

    all_tickers = set(rec_by_ticker.keys())

    # --- Symbol alias variants ---
    def _alias_variants(sym: str) -> List[str]:
        s = sym.strip().upper()
        variants = [s]
        if "." in s:
            variants.append(s.replace(".", "-"))
        if "-" in s:
            variants.append(s.replace("-", "."))
        return variants

    alias_targets: set = set()
    for t in all_tickers:
        for v in _alias_variants(t):
            alias_targets.add(v)

    # Read prices for ALL universe tickers + XBI
    xbi_want = {"XBI"}
    prices_by_ticker: Dict[str, List[float]] = {}
    with __import__("contextlib").nullcontext():
        reader = _read_price_history_rows(price_history_path)
        for row in reader:
            ticker = (row.get("ticker") or "").upper()
            if ticker not in all_tickers and ticker not in alias_targets and ticker not in xbi_want:
                continue
            date_str = row.get("date", "")
            close_str = row.get("close", "")
            if not date_str or not close_str:
                continue
            try:
                row_date = _dt.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            if row_date > ref_date:
                continue
            try:
                close = float(close_str)
            except (ValueError, TypeError):
                continue
            if close <= 0:
                continue
            prices_by_ticker.setdefault(ticker, []).append((row_date, close))

    # --- Build alias index ---
    alias_to_csv_sym: Dict[str, str] = {}
    csv_syms_with_data = set(prices_by_ticker.keys())
    for csv_sym in sorted(csv_syms_with_data):
        for v in _alias_variants(csv_sym):
            if v in all_tickers and v not in prices_by_ticker:
                if v not in alias_to_csv_sym:
                    alias_to_csv_sym[v] = csv_sym

    WINDOW = 252
    # Resolve the corporate-actions registry once (used only by the recent-split
    # fallback below). Default-load if not injected; fail-open on any error.
    _ca_registry = corporate_actions
    if _ca_registry is None:
        try:
            from common.corporate_actions import load_actions

            _ca_registry = load_actions()
        except Exception:  # pragma: no cover - fail-open
            _ca_registry = None
    recomputed_tickers: set = set()
    for ticker in all_tickers:
        rec = rec_by_ticker[ticker]
        df = rec.get("defensive_features")
        if df is None:
            df = {}
            rec["defensive_features"] = df

        resolved_sym = None
        series = prices_by_ticker.get(ticker)
        if not series:
            csv_sym = alias_to_csv_sym.get(ticker)
            if csv_sym:
                series = prices_by_ticker.get(csv_sym)
                resolved_sym = csv_sym

        if not series or len(series) < MIN_BARS_FOR_ESTIMATE:
            # No sufficient price data — fall back to pipeline value
            raw_dd = df.get("drawdown")
            if raw_dd is not None:
                try:
                    df["drawdown"] = float(raw_dd)
                except (ValueError, TypeError):
                    df["drawdown"] = None
            if df.get("drawdown") is None:
                for alt_key in ("drawdown_current", "drawdown_60d"):
                    val = df.get(alt_key)
                    if val is not None:
                        try:
                            df["drawdown"] = float(val)
                        except (ValueError, TypeError):
                            continue
                        hydrated += 1
                        break
            if df.get("drawdown") is not None:
                df["drawdown_missing_reason"] = ""
            elif not series:
                df["drawdown_missing_reason"] = "no_price_series"
            else:
                df["drawdown_missing_reason"] = "series_too_short"
            continue

        series.sort(key=lambda x: x[0])
        unfiltered_series = list(series)  # preserve for fallback

        # Filter split artifacts before computing drawdown
        series, split_warns = _filter_price_outliers(series)
        if split_warns:
            logger.warning(
                "_hydrate_drawdown: %s has %d price outlier(s) (split artifacts), "
                "filtered before drawdown computation: %s",
                ticker,
                len(split_warns),
                "; ".join(f"{w['date']} {w['pct_change']:+.0%}" for w in split_warns),
            )
        if len(series) < MIN_BARS_FOR_ESTIMATE:
            # The split-filter over-truncated: a recent split left < MIN_BARS
            # post-split bars. Restoring the RAW unfiltered series here would
            # compute the 252-bar peak across the split (pre-split high) and
            # report a phantom deep drawdown — this is the IMMP/GOSS bug.
            if len(unfiltered_series) >= MIN_BARS_FOR_ESTIMATE and split_warns:
                # If corporate_actions.json confirms a real split, scale the
                # restored series so the peak is on the post-split basis. If no
                # action is registered (a genuine crash the −75% heuristic
                # misfired on), keep the raw series — the deep drawdown is real.
                adjusted = _corp_action_split_adjust(ticker, unfiltered_series, as_of_date, _ca_registry)
                if adjusted is not None:
                    logger.info(
                        "_hydrate_drawdown: %s recent-split fallback: filtered=%d < %d, "
                        "using corporate-action split-adjusted series (%d bars)",
                        ticker,
                        len(series),
                        MIN_BARS_FOR_ESTIMATE,
                        len(adjusted),
                    )
                    series = adjusted
                else:
                    logger.info(
                        "_hydrate_drawdown: %s split-filter fallback: "
                        "filtered=%d bars < %d, restoring unfiltered %d bars "
                        "(no corporate action — treated as real crash)",
                        ticker,
                        len(series),
                        MIN_BARS_FOR_ESTIMATE,
                        len(unfiltered_series),
                    )
                    series = unfiltered_series
            else:
                df["drawdown_missing_reason"] = "series_too_short"
                continue

        closes = [p for _, p in series]
        look = closes[-WINDOW:] if len(closes) >= WINDOW else closes
        peak = max(look)
        if peak == 0:
            df["drawdown_missing_reason"] = "series_too_short"
            continue

        old_dd = df.get("drawdown")
        dd = round((closes[-1] / peak) - 1.0, 6)
        df["drawdown"] = dd
        df["drawdown_missing_reason"] = ""
        if resolved_sym:
            df["drawdown_price_symbol"] = resolved_sym
        recomputed_tickers.add(ticker)

        # Track overwrite for logging
        if old_dd is not None:
            try:
                old_dd_f = float(old_dd)
                if abs(dd - old_dd_f) > 0.05:
                    logger.debug(
                        f"_hydrate_drawdown: {ticker} overwrite "
                        f"{old_dd_f:.4f} → {dd:.4f} (delta {dd - old_dd_f:+.4f})"
                    )
            except (ValueError, TypeError):
                pass
        hydrated += 1

    # --- XBI drawdown + relative drawdown for ALL tickers ---
    xbi_series = prices_by_ticker.get("XBI")
    xbi_dd = None
    if xbi_series and len(xbi_series) >= MIN_BARS_FOR_ESTIMATE:
        xbi_series.sort(key=lambda x: x[0])
        xbi_closes = [p for _, p in xbi_series]
        xbi_look = xbi_closes[-WINDOW:] if len(xbi_closes) >= WINDOW else xbi_closes
        xbi_peak = max(xbi_look)
        if xbi_peak > 0:
            xbi_dd = round((xbi_closes[-1] / xbi_peak) - 1.0, 6)

    for rec in rec_by_ticker.values():
        df = rec.get("defensive_features")
        if df is None:
            df = {}
            rec["defensive_features"] = df
        df["drawdown_xbi"] = xbi_dd
        dd = df.get("drawdown")
        if dd is not None and xbi_dd is not None:
            df["drawdown_rel_xbi"] = round(float(dd) - xbi_dd, 6)
        else:
            df["drawdown_rel_xbi"] = None

    # --- Defensive check: all reason values must be in the enum ---
    for t, rec in rec_by_ticker.items():
        reason = (rec.get("defensive_features") or {}).get("drawdown_missing_reason", "")
        if reason not in VALID_DRAWDOWN_MISSING_REASONS:
            raise ValueError(f"_hydrate_drawdown: ticker {t} has invalid " f"drawdown_missing_reason={reason!r}")

    return hydrated


BETA_WINDOW = 60
"""Number of aligned trading days for beta/alpha computation."""

MIN_OVERLAP_BARS = 30
"""Minimum date-aligned bars required for a meaningful beta estimate."""

XBI_STALE_THRESHOLD = 3
"""Max acceptable trading-day gap between XBI last date and other tickers.
If exceeded, beta/alpha are marked ``xbi_stale`` rather than computed
silently from stale benchmark data."""


def _hydrate_beta_rsi(
    rec_by_ticker: Dict[str, Dict[str, Any]],
    price_history_path: Optional[Path],
    as_of_date: str,
) -> int:
    """Hydrate ``beta_xbi_60d``, ``alpha_60d``, and ``rsi_14d`` from price CSV.

    All three fields use **overwrite_always** policy — ``price_history.csv``
    is the single source of truth.  Stale pipeline / cache values are always
    replaced when recomputation is possible.

    **Beta** is computed from date-aligned daily returns (inner-join by date)
    over a 60-day window.

    **Alpha** (excess return) is ``ticker_cum_return − beta × xbi_cum_return``
    over the same aligned window.

    **RSI** is 14-day Wilder-smoothed, unchanged from prior implementation.

    **XBI staleness guard**: when XBI price data is >3 trading days behind
    *as_of_date*, beta and alpha are *not* computed.  Instead the missing-
    reason field is set to ``xbi_stale``.

    Missing-reason fields stored on ``defensive_features``:

    * ``beta_xbi_60d_missing_reason``: ``""`` | ``no_xbi_series`` |
      ``xbi_stale`` | ``insufficient_overlap`` | ``series_too_short`` |
      ``zero_var_benchmark``
    * ``alpha_60d_missing_reason``: ``""`` | ``beta_missing:<reason>``

    Source-tagging fields:

    * ``beta_xbi_60d_source``: ``"price_history"`` | ``""``
    * ``alpha_60d_source``: ``"price_history"`` | ``""``

    Returns the total number of fields hydrated (beta + alpha + RSI).
    """
    import csv as _csv
    from datetime import datetime as _dt

    hydrated = 0

    if price_history_path is None or not price_history_path.exists():
        return hydrated

    all_tickers = list(rec_by_ticker.keys())

    try:
        ref_date = _dt.strptime(as_of_date, "%Y-%m-%d").date()
    except ValueError:
        return hydrated

    # ------------------------------------------------------------------
    # Parse price CSV for ALL tickers + XBI (single pass)
    # ------------------------------------------------------------------
    ticker_set = set(all_tickers) | {"XBI"}
    prices_by_ticker: Dict[str, List] = {}
    with __import__("contextlib").nullcontext():
        reader = _read_price_history_rows(price_history_path)
        for row in reader:
            ticker = (row.get("ticker") or "").upper()
            if ticker not in ticker_set:
                continue
            date_str = row.get("date", "")
            close_str = row.get("close", "")
            if not date_str or not close_str:
                continue
            try:
                row_date = _dt.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            if row_date > ref_date:
                continue
            try:
                close = float(close_str)
            except (ValueError, TypeError):
                continue
            if close <= 0:
                continue
            prices_by_ticker.setdefault(ticker, []).append((row_date, close))

    for tk in prices_by_ticker:
        prices_by_ticker[tk].sort(key=lambda x: x[0])

    # ------------------------------------------------------------------
    # Filter split artifacts from all price series
    # ------------------------------------------------------------------
    split_warnings = _validate_price_splits(prices_by_ticker)
    for tk, warns in split_warnings.items():
        logger.warning(
            "_hydrate_beta_rsi: %s has %d price outlier(s) (split artifacts), " "filtering: %s",
            tk,
            len(warns),
            "; ".join(f"{w['date']} {w['pct_change']:+.0%}" for w in warns),
        )
        prices_by_ticker[tk], _ = _filter_price_outliers(prices_by_ticker[tk])

    # ------------------------------------------------------------------
    # XBI freshness check
    # ------------------------------------------------------------------
    xbi_series = prices_by_ticker.get("XBI")
    xbi_available = xbi_series is not None and len(xbi_series) >= 2
    xbi_stale = False
    xbi_gap_days = 0

    if xbi_available:
        xbi_last_date = xbi_series[-1][0]
        # Count unique trading dates across non-XBI tickers after XBI's last
        post_xbi_dates: set = set()
        for tk, series in prices_by_ticker.items():
            if tk == "XBI":
                continue
            for d, _ in series:
                if d > xbi_last_date:
                    post_xbi_dates.add(d)
        xbi_gap_days = len(post_xbi_dates)
        if xbi_gap_days > XBI_STALE_THRESHOLD:
            xbi_stale = True
            logger.warning(
                "XBI benchmark stale: last_date=%s, as_of=%s, gap=%d trading "
                "days (threshold=%d). Beta/alpha marked xbi_stale.",
                xbi_last_date,
                as_of_date,
                xbi_gap_days,
                XBI_STALE_THRESHOLD,
            )

    # ------------------------------------------------------------------
    # Step A: beta_xbi_60d from price_history.csv (overwrite_always)
    #         + alpha_60d from same aligned window
    # ------------------------------------------------------------------
    xbi_date_close: Dict = {}
    if xbi_available:
        xbi_date_close = {d: c for d, c in xbi_series}

    for ticker in all_tickers:
        dfe = rec_by_ticker[ticker].get("defensive_features")
        if dfe is None:
            dfe = {}
            rec_by_ticker[ticker]["defensive_features"] = dfe

        # --- guard: XBI not in price CSV ---
        if not xbi_available:
            dfe["beta_xbi_60d_missing_reason"] = "no_xbi_series"
            dfe["beta_xbi_60d_source"] = ""
            dfe["alpha_60d_missing_reason"] = "beta_missing:no_xbi_series"
            dfe["alpha_60d_source"] = ""
            continue

        # --- guard: XBI stale ---
        if xbi_stale:
            dfe["beta_xbi_60d_missing_reason"] = "xbi_stale"
            dfe["beta_xbi_60d_source"] = ""
            dfe["alpha_60d_missing_reason"] = "beta_missing:xbi_stale"
            dfe["alpha_60d_source"] = ""
            continue

        t_series = prices_by_ticker.get(ticker)
        if not t_series or len(t_series) < 2:
            dfe["beta_xbi_60d_missing_reason"] = "series_too_short"
            dfe["beta_xbi_60d_source"] = ""
            dfe["alpha_60d_missing_reason"] = "beta_missing:series_too_short"
            dfe["alpha_60d_source"] = ""
            continue

        # Date-aligned inner join
        aligned = []
        for d, c in t_series:
            xc = xbi_date_close.get(d)
            if xc is not None:
                aligned.append((d, c, xc))

        if len(aligned) < MIN_OVERLAP_BARS + 1:
            dfe["beta_xbi_60d_missing_reason"] = "insufficient_overlap"
            dfe["beta_xbi_60d_source"] = ""
            dfe["alpha_60d_missing_reason"] = "beta_missing:insufficient_overlap"
            dfe["alpha_60d_source"] = ""
            continue

        # Take most recent BETA_WINDOW+1 aligned bars (or all if fewer)
        window_bars = min(BETA_WINDOW + 1, len(aligned))
        aligned = aligned[-window_bars:]

        # Compute daily returns
        n_rets = len(aligned) - 1
        t_rets = [(aligned[i][1] / aligned[i - 1][1]) - 1.0 for i in range(1, len(aligned))]
        x_rets = [(aligned[i][2] / aligned[i - 1][2]) - 1.0 for i in range(1, len(aligned))]

        # Beta = cov(t, x) / var(x)
        t_mean = sum(t_rets) / n_rets
        x_mean = sum(x_rets) / n_rets
        cov_tx = sum((t_rets[i] - t_mean) * (x_rets[i] - x_mean) for i in range(n_rets)) / n_rets
        var_x = sum((x_rets[i] - x_mean) ** 2 for i in range(n_rets)) / n_rets

        if var_x == 0:
            dfe["beta_xbi_60d_missing_reason"] = "zero_var_benchmark"
            dfe["beta_xbi_60d_source"] = ""
            dfe["alpha_60d_missing_reason"] = "beta_missing:zero_var_benchmark"
            dfe["alpha_60d_source"] = ""
            continue

        new_beta = round(cov_tx / var_x, 4)
        old_beta = dfe.get("beta_xbi_60d")
        if old_beta is not None:
            try:
                if abs(float(old_beta) - new_beta) > 0.5:
                    logger.debug(
                        "Beta overwrite %s: %.4f -> %.4f (delta=%.4f)",
                        ticker,
                        float(old_beta),
                        new_beta,
                        abs(float(old_beta) - new_beta),
                    )
            except (ValueError, TypeError):
                pass

        dfe["beta_xbi_60d"] = new_beta
        dfe["beta_xbi_60d_source"] = "price_history"
        dfe["beta_xbi_60d_missing_reason"] = ""
        hydrated += 1

        # --- Alpha from same aligned window ---
        t_cum = (aligned[-1][1] / aligned[0][1]) - 1.0
        x_cum = (aligned[-1][2] / aligned[0][2]) - 1.0
        new_alpha = round(t_cum - new_beta * x_cum, 6)

        # Log large overwrite deltas vs pipeline alpha
        mom_enh = (rec_by_ticker[ticker].get("score_breakdown") or {}).get("enhancements", {}).get("momentum") or {}
        pipeline_alpha = mom_enh.get("alpha_60d")
        if pipeline_alpha is None:
            pipeline_alpha = (rec_by_ticker[ticker].get("momentum_signal") or {}).get("alpha_60d")
        if pipeline_alpha is not None:
            try:
                if abs(float(pipeline_alpha) - new_alpha) > 0.05:
                    logger.debug(
                        "Alpha overwrite %s: %.6f -> %.6f (delta=%.6f)",
                        ticker,
                        float(pipeline_alpha),
                        new_alpha,
                        abs(float(pipeline_alpha) - new_alpha),
                    )
            except (ValueError, TypeError):
                pass

        dfe["alpha_60d"] = new_alpha
        dfe["alpha_60d_source"] = "price_history"
        dfe["alpha_60d_missing_reason"] = ""
        hydrated += 1

    # ------------------------------------------------------------------
    # Step B: rsi_14d from price_history.csv (overwrite_always — unchanged)
    # ------------------------------------------------------------------
    RSI_PERIOD = 14
    MIN_BARS_RSI = RSI_PERIOD + 1  # need at least 15 prices

    for ticker in all_tickers:
        series = prices_by_ticker.get(ticker)
        if not series or len(series) < MIN_BARS_RSI:
            continue
        closes = [p for _, p in series]

        # Wilder-smoothed RSI
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains = [max(d, 0) for d in deltas]
        losses = [max(-d, 0) for d in deltas]

        # Seed with simple average of first RSI_PERIOD changes
        avg_gain = sum(gains[:RSI_PERIOD]) / RSI_PERIOD
        avg_loss = sum(losses[:RSI_PERIOD]) / RSI_PERIOD

        # Wilder smoothing for remaining
        for i in range(RSI_PERIOD, len(deltas)):
            avg_gain = (avg_gain * (RSI_PERIOD - 1) + gains[i]) / RSI_PERIOD
            avg_loss = (avg_loss * (RSI_PERIOD - 1) + losses[i]) / RSI_PERIOD

        if avg_loss == 0:
            rsi_val = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_val = 100.0 - (100.0 / (1.0 + rs))

        dfe = rec_by_ticker[ticker].get("defensive_features")
        if dfe is None:
            dfe = {}
            rec_by_ticker[ticker]["defensive_features"] = dfe
        old_rsi = dfe.get("rsi_14d")
        new_rsi = round(rsi_val, 4)
        if old_rsi is not None:
            try:
                if abs(float(old_rsi) - new_rsi) > 5.0:
                    logger.debug(
                        "RSI overwrite %s: %.2f -> %.2f (delta=%.2f)",
                        ticker,
                        float(old_rsi),
                        new_rsi,
                        abs(float(old_rsi) - new_rsi),
                    )
            except (ValueError, TypeError):
                pass
        dfe["rsi_14d"] = new_rsi
        hydrated += 1

    # ------------------------------------------------------------------
    # Step C: vol_60d from price_history.csv (FILL-IF-MISSING)
    # Annualized daily-return volatility over the last 60 trading days.
    # Unlike beta/rsi (overwrite_always), vol_60d is only filled when absent so
    # we don't perturb the 272 tickers that already carry a value under the
    # model freeze — this closes the coverage gap for tickers whose universe
    # record lacked a defensive_features block (e.g. TEVA/CAPR/DNTH).
    # ------------------------------------------------------------------
    VOL_WINDOW = 60
    MIN_RETURNS_VOL = 30
    for ticker in all_tickers:
        dfe = rec_by_ticker[ticker].get("defensive_features")
        if dfe is None:
            dfe = {}
            rec_by_ticker[ticker]["defensive_features"] = dfe
        # Only fill when missing/invalid — respect existing values.
        existing = dfe.get("vol_60d")
        if existing not in (None, ""):
            try:
                float(existing)
                continue  # already has a usable value
            except (ValueError, TypeError):
                pass  # invalid → recompute
        series = prices_by_ticker.get(ticker)
        if not series or len(series) < MIN_RETURNS_VOL + 1:
            continue
        closes = [p for _, p in series][-(VOL_WINDOW + 1) :]
        rets = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]
        if len(rets) < MIN_RETURNS_VOL:
            continue
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)  # sample var (ddof=1)
        vol_60d = (var**0.5) * (252**0.5)
        dfe["vol_60d"] = f"{vol_60d:.4f}"
        hydrated += 1

    return hydrated


def _check_catalyst_corroboration(
    m3_summaries: Optional[Dict[str, Any]],
    ticker: str,
    catalyst_source: str,
    catalyst_family: str,
) -> Dict[str, Any]:
    """Check if the selected clinical catalyst date is corroborated."""
    if not m3_summaries or catalyst_family != "CLINICAL":
        return evaluate_corroboration(catalyst_source, "", [], catalyst_family)

    summary = m3_summaries.get(ticker.upper()) or m3_summaries.get(ticker)
    if not summary or not isinstance(summary, dict):
        return evaluate_corroboration(catalyst_source, "", [], catalyst_family)

    next_date = (summary.get("integration") or {}).get("next_catalyst_date", "")
    events = summary.get("events") or []
    return evaluate_corroboration(catalyst_source, next_date, events, catalyst_family)


def _find_nearest_catalyst_event(
    m3_summaries: Optional[Dict[str, Any]],
    ticker: str,
    max_fuzzy_days: int = 14,
    as_of_date: Optional[str] = None,
    pdufa_manual: Optional[List[Dict[str, str]]] = None,
) -> Optional[Dict[str, Any]]:
    """Find the best M3 event for a ticker, preferring high-value events.

    Five-tier lookup:
      P. Priority override: if a T1/T2 event (regulatory/pivotal) exists
         within 180 days, prefer it over any nearer T3/T4/T5 event. PDUFA
         decisions are categorically more market-moving than trial milestones.
         Checks both M3 events AND PDUFA manual calendar.
      0. Earliest future event fallback: when integration.next_catalyst_date
         is null but the events list contains future events, pick the earliest.
      1. Exact date match: event_date == integration.next_catalyst_date
      2. Fuzzy date match: closest event within ±max_fuzzy_days
      3. Source UID inference: if nearest_catalyst_source_uid starts with
         "NCT", synthesise a minimal CLINICAL event

    Returns the matched event dict, or None.
    """
    if not m3_summaries:
        return None
    summary = m3_summaries.get(ticker.upper()) or m3_summaries.get(ticker)
    if not summary or not isinstance(summary, dict):
        return None
    integration = summary.get("integration") or {}
    next_date = integration.get("next_catalyst_date")

    events = [e for e in (summary.get("events") or []) if isinstance(e, dict)]

    # ── Tier P: Priority override — prefer T1/T2 over nearer T3+ ────
    # A PDUFA decision within 180 days is always more important than a
    # CT.gov PCD that happens to be closer on the calendar.
    if as_of_date:
        from datetime import date as _date

        from decision_engine import classify_catalyst_type_tier as _tier_p

        _TIER_RANK = {"T1": 0, "T2": 1}  # lower = higher priority
        ref = _date.fromisoformat(as_of_date[:10])
        best_priority = None
        best_key = None  # (tier_rank, days) — lower is better

        # Check M3 events for T1/T2
        for event in events:
            ed = event.get("event_date", "")
            if not ed:
                continue
            try:
                days = (_date.fromisoformat(ed[:10]) - ref).days
            except (ValueError, TypeError):
                continue
            if days < 1 or days > 180:
                continue
            tier = _tier_p(event.get("event_type", ""))
            rank = _TIER_RANK.get(tier)
            if rank is None:
                continue
            key = (rank, days)
            if best_key is None or key < best_key:
                best_priority = event
                best_key = key

        # Check PDUFA manual calendar — always T1 (rank 0), wins ties with T2
        if pdufa_manual:
            ticker_upper = ticker.upper()
            for rec in pdufa_manual:
                if (rec.get("ticker", "")).upper() != ticker_upper:
                    continue
                pd = rec.get("pdufa_date", "")
                if not pd:
                    continue
                try:
                    days = (_date.fromisoformat(pd[:10]) - ref).days
                except (ValueError, TypeError):
                    continue
                if days < 1 or days > 180:
                    continue
                key = (0, days)  # T1 = rank 0
                if best_key is None or key < best_key:
                    best_priority = {
                        "event_type": "FDA_PDUFA_DATE",
                        "source": "PDUFA_MANUAL",
                        "event_date": pd,
                        "_from_pdufa_manual": True,
                    }
                    best_key = key

        if best_priority is not None:
            return best_priority

    # ── Tier 0: earliest future event fallback ───────────────────────
    if not next_date and events:
        from datetime import date as _date

        from common.hard_catalyst import is_hard_catalyst as _is_hard_t0

        ref = as_of_date or ""
        future = []
        for event in events:
            ed = event.get("event_date", "")
            if not ed:
                continue
            if ref and ed <= ref:
                continue
            is_hard = _is_hard_t0(event.get("event_type", ""), event.get("source", ""))
            future.append((ed, not is_hard, event))
        if future:
            future.sort(key=lambda x: (x[0], x[1]))
            return future[0][2]

    if not next_date:
        return None

    # ── Tier 1: exact date match ─────────────────────────────────────
    for event in events:
        if event.get("event_date") == next_date:
            return event

    # ── Tier 2: fuzzy date match ─────────────────────────────────────
    if events:
        best_event = None
        best_delta = None
        best_is_soft = True
        try:
            from datetime import date as _date

            from common.hard_catalyst import is_hard_catalyst as _is_hard_t2

            target = _date.fromisoformat(next_date[:10])
            for event in events:
                ed = event.get("event_date", "")
                if not ed:
                    continue
                try:
                    delta = abs((_date.fromisoformat(ed[:10]) - target).days)
                except (ValueError, TypeError):
                    continue
                if delta > max_fuzzy_days:
                    continue
                is_soft = not _is_hard_t2(event.get("event_type", ""), event.get("source", ""))
                if best_delta is None or (delta, is_soft) < (best_delta, best_is_soft):
                    best_delta = delta
                    best_is_soft = is_soft
                    best_event = event
        except (ValueError, TypeError):
            pass
        if best_event is not None:
            return best_event

    # ── Tier 3: infer from source UID ────────────────────────────────
    source_uid = integration.get("nearest_catalyst_source_uid", "")
    if source_uid.startswith("NCT"):
        return {
            "event_type": "CT_PRIMARY_COMPLETION",
            "source": "CTGOV_CALENDAR",
            "event_date": next_date,
            "_inferred": True,
        }

    return None


def _nearest_catalyst_source(
    m3_summaries: Optional[Dict[str, Any]],
    ticker: str,
    as_of_date: Optional[str] = None,
    pdufa_manual: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Extract the source of the nearest catalyst event from Module 3 summaries."""
    event = _find_nearest_catalyst_event(m3_summaries, ticker, as_of_date=as_of_date, pdufa_manual=pdufa_manual)
    return event.get("source", "") if event else ""


def _nearest_catalyst_event_type(
    m3_summaries: Optional[Dict[str, Any]],
    ticker: str,
    as_of_date: Optional[str] = None,
    pdufa_manual: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Extract the event_type of the nearest catalyst event from Module 3."""
    event = _find_nearest_catalyst_event(m3_summaries, ticker, as_of_date=as_of_date, pdufa_manual=pdufa_manual)
    return event.get("event_type", "") if event else ""


# ---------------------------------------------------------------------------
# Spec 078 — Catalyst quality classification (Lane A + Lane B)
# ---------------------------------------------------------------------------
# Sources treated as hard regulatory / SEC-disclosed binaries (Lane B exempt).
_CATALYST_QUALITY_EXEMPT_SOURCES: frozenset = frozenset(
    {"PDUFA_MANUAL", "FDA_ADCOM_CALENDAR", "SEC_8K_FILING", "SEC_6K_FILING"}
)
# Calendar-confidence gate for non-exempt sources (Lane B).
_CATALYST_QUALITY_CONF_THRESHOLD: float = 0.40
# Non-binary corporate event types that must not drive tier credit (Lane A).
_NON_BINARY_CATALYST_EVENT_TYPES: frozenset = frozenset(
    {
        "EARNINGS_RELEASE",
        "INVESTOR_DAY",
        "PARTNERSHIP",
        "MA_ACTIVITY",
        "LICENSING_DEAL",
        "CONFERENCE_PRESENTATION",
        "CONFERENCE_LATE_BREAKER",
        "CONFERENCE_ACCEPTED_ABSTRACT",
        "IR_EVENT",
        "PRESS_RELEASE_EVENT",
    }
)


def classify_catalyst_quality(
    catalyst_source: str,
    catalyst_event_type: str,
    calendar_confidence: str,
) -> str:
    """Return the Spec 078 catalyst-quality bucket for a single catalyst row.

    Buckets (in precedence order):
      corporate_update — non-binary corporate event type (Lane A)
      low_confidence   — calendar_confidence < threshold for non-exempt source (Lane B)
      binary_alpha     — hard regulatory / SEC-disclosed binary event
      registry_only    — CT.gov calendar-derived completion date
      ""               — no catalyst signal
    """

    # Guard NaN from pandas CSV reads (float NaN masquerades as a non-string).
    def _s(v) -> str:
        return "" if v is None or isinstance(v, float) else str(v).strip()

    src = _s(catalyst_source)
    evt = _s(catalyst_event_type)
    cal = _s(calendar_confidence)

    if not src and not evt:
        return ""

    # Lane A: non-binary corporate event types do not qualify as release-valve catalysts.
    if evt in _NON_BINARY_CATALYST_EVENT_TYPES:
        return "corporate_update"

    # Lane B: low calendar confidence for non-exempt sources.
    if src not in _CATALYST_QUALITY_EXEMPT_SOURCES:
        try:
            conf_float = float(cal) if cal else 0.0
        except (ValueError, TypeError):
            conf_float = 0.0
        if 0.0 < conf_float < _CATALYST_QUALITY_CONF_THRESHOLD:
            return "low_confidence"

    # Hard regulatory / SEC-disclosed binaries.
    if src in _CATALYST_QUALITY_EXEMPT_SOURCES:
        return "binary_alpha"

    # CT.gov calendar-derived events.
    if src.startswith("CTGOV") or src == "FDA_ADCOM_CALENDAR":
        return "registry_only"

    # Any remaining catalyst with a signal — classify conservatively.
    return "registry_only"


def _load_pdufa_manual(
    data_dir: Optional[Path] = None,
    as_of_date: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Load PDUFA manual dates via the regulatory calendar loader.

    Returns list of {ticker, pdufa_date, ...} dicts.  Normalizes old schema
    fields and applies PIT filtering when as_of_date is provided.
    """
    from common.regulatory_calendar import load_and_validate

    records, errors = load_and_validate(data_dir=data_dir, as_of_date=as_of_date)
    if errors:
        logger.warning("regulatory_calendar: %d validation issues: %s", len(errors), errors[:5])
    return records


def _load_snapshot_pdufa_manual(data_dir: Optional[Path], as_of_date: str) -> List[Dict[str, str]]:
    """Load snapshot PDUFA manual records from the active run data directory."""
    return _load_pdufa_manual(data_dir=data_dir, as_of_date=as_of_date)


def _nearest_regulatory_catalyst(
    m3_summaries: Optional[Dict[str, Any]],
    ticker: str,
    as_of_date_str: str,
    max_days: int = 180,
    pdufa_manual: Optional[List[Dict[str, str]]] = None,
    event_ledger: Optional[list] = None,
) -> Tuple[str, str]:
    """Find the nearest REGULATORY event for a ticker.

    Scans three sources in order:
      1. M3 scored events (CTGov/SEC pipeline)
      2. Event ledger entries (FDA ADCOM, FDA regulatory, EMA, PDUFA, SEC 8-K)
      3. PDUFA manual file (static fallback)

    Returns (days_str, event_type) or ("", "").
    """
    from datetime import date as _date

    best_days: Optional[int] = None
    best_type = ""

    def _try_update(event_date_str: str, event_type: str) -> None:
        nonlocal best_days, best_type
        if not event_date_str or event_date_str <= as_of_date_str:
            return
        try:
            days = (_date.fromisoformat(event_date_str[:10]) - _date.fromisoformat(as_of_date_str[:10])).days
        except (ValueError, TypeError):
            return
        if days < 1 or days > max_days:
            return
        if best_days is None or days < best_days:
            best_days = days
            best_type = event_type

    # 1. Scan M3 events
    if m3_summaries:
        summary = m3_summaries.get(ticker.upper()) or m3_summaries.get(ticker)
        if summary and isinstance(summary, dict):
            for event in summary.get("events") or []:
                if not isinstance(event, dict):
                    continue
                et = event.get("event_type", "")
                if et not in REGULATORY_EVENT_TYPES:
                    continue
                _try_update(event.get("event_date", ""), et)

    # 2. Scan event ledger (FDA ADCOM, FDA regulatory, EMA, PDUFA, etc.)
    #    This picks up regulatory events that M3's scored pipeline doesn't
    #    surface because score_catalyst_events only processes CTGov/SEC events.
    if event_ledger:
        ticker_upper = ticker.upper()
        for entry in event_ledger:
            if entry.ticker.upper() != ticker_upper:
                continue
            if entry.event_type not in REGULATORY_EVENT_TYPES:
                continue
            if entry.confidence == "LOW":
                continue
            _try_update(entry.event_date or "", entry.event_type)

    # 3. Fallback: PDUFA manual file
    if pdufa_manual:
        ticker_upper = ticker.upper()
        for rec in pdufa_manual:
            if (rec.get("ticker", "")).upper() != ticker_upper:
                continue
            _try_update(rec.get("pdufa_date", ""), "PDUFA")

    if best_days is not None:
        return (str(best_days), best_type)
    return ("", "")


# ---------------------------------------------------------------------------
# Clinical alpha z-score helpers
# ---------------------------------------------------------------------------

_CLINICAL_PHASES = frozenset(
    {
        "PHASE1",
        "PHASE2",
        "PHASE3",
        "PHASE 1",
        "PHASE 2",
        "PHASE 3",
        "PHASE1/PHASE2",
        "PHASE 1/PHASE 2",
        "PHASE2/PHASE3",
        "PHASE 2/PHASE 3",
        "EARLY_PHASE1",
    }
)

_PHASE_NUM = {
    "approved": 1.0,
    "phase 3": 0.83,
    "phase 2/3": 0.67,
    "phase 2": 0.50,
    "phase 1/2": 0.33,
    "phase 1": 0.17,
    "preclinical": 0.08,
}

_CE_WEIGHTS = {"phase": 0.35, "proximity": 0.35, "quality": 0.30}


def _parse_trial_date(s) -> Optional[date]:
    """Parse YYYY-MM-DD string to date, or None."""
    if not s:
        return None
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _clinical_proximity(days: Optional[int]) -> float:
    """Exponential decay proximity score: days to readout → [0, 1]."""
    if days is None:
        return 0.0
    if days <= 0:
        return 1.0
    return math.exp(-days / 90)


def _compute_clinical_readout_days(
    trial_records: List[Dict],
    as_of_date: str,
) -> Dict[str, Optional[int]]:
    """For each ticker, find min days to nearest credible clinical readout.

    Filters: INTERVENTIONAL, Phase 1-3 (excl Phase 4), PIT-safe.
    Priority: primary_completion_date > completion_date.
    Past PCD with no results_first_posted → 0 (imminent readout).

    Returns dict mapping ticker → days (int) or None if no qualifying trials.
    """
    as_of = datetime.strptime(as_of_date[:10], "%Y-%m-%d").date()
    # Group trials by ticker
    by_ticker: Dict[str, list] = {}
    for trial in trial_records:
        tk = trial.get("ticker", "")
        if tk:
            by_ticker.setdefault(tk.upper(), []).append(trial)

    result: Dict[str, Optional[int]] = {}
    for ticker, trials in by_ticker.items():
        best_days: Optional[int] = None
        for t in trials:
            # Study type filter
            if (t.get("study_type") or "").upper() != "INTERVENTIONAL":
                continue
            # Phase filter (Phase 1-3 only)
            phase = str(t.get("phase", "")).upper()
            if phase not in _CLINICAL_PHASES:
                continue
            # PIT gate: trial must be publicly known at as_of
            fp = _parse_trial_date(t.get("first_posted"))
            lup = _parse_trial_date(t.get("last_update_posted"))
            pit_date = fp or lup
            if pit_date is None or pit_date > as_of:
                continue

            pcd = _parse_trial_date(t.get("primary_completion_date"))
            cd = _parse_trial_date(t.get("completion_date"))
            rfp = _parse_trial_date(t.get("results_first_posted"))

            # Priority 1: PCD
            if pcd is not None:
                if pcd > as_of:
                    days = (pcd - as_of).days
                elif rfp is None:
                    # Past PCD, no results posted → imminent readout
                    days = 0
                else:
                    # Past PCD + results posted → already known, skip
                    continue
            elif cd is not None and cd > as_of:
                # Fallback: future completion_date
                days = (cd - as_of).days
            else:
                continue

            if best_days is None or days < best_days:
                best_days = days

        if best_days is not None:
            result[ticker] = best_days

    return result


def _compute_clinical_alpha_z(
    csv_rows: List[dict],
    m4_lookup: Dict[str, dict],
    readout_days: Dict[str, Optional[int]],
) -> None:
    """Compute clinical_alpha_z, clinical_readout_days, clinical_coverage_flag.

    Writes into csv_rows for all tickers with module_4 scores.
    Z-score is cross-sectional across the full ranked universe.
    """
    eligible_indices: List[int] = []
    raw_scores: List[float] = []

    for i, row in enumerate(csv_rows):
        ticker = row.get("ticker", "")
        m4 = m4_lookup.get(ticker)
        if m4 is None:
            continue

        lead_phase = str(m4.get("lead_phase", "")).lower().strip()
        phase_num = _PHASE_NUM.get(lead_phase, 0.0)
        rd = readout_days.get(ticker)
        proximity = _clinical_proximity(rd)
        clin_score = m4.get("clinical_score")
        quality = float(clin_score) / 100.0 if clin_score is not None else 0.0

        raw = _CE_WEIGHTS["phase"] * phase_num + _CE_WEIGHTS["proximity"] * proximity + _CE_WEIGHTS["quality"] * quality

        eligible_indices.append(i)
        raw_scores.append(raw)

        # Store readout metadata now (will add z later)
        csv_rows[i]["clinical_readout_days"] = rd if rd is not None else ""
        csv_rows[i]["clinical_coverage_flag"] = 1 if rd is not None else 0

    if not raw_scores:
        return

    # Winsorize at p5/p95
    n = len(raw_scores)
    if n >= 20:
        sorted_vals = sorted(raw_scores)
        p5 = sorted_vals[max(0, int(n * 0.05))]
        p95 = sorted_vals[min(n - 1, int(n * 0.95))]
    else:
        p5 = min(raw_scores)
        p95 = max(raw_scores)

    winsorized = [max(p5, min(p95, v)) for v in raw_scores]

    # Z-score
    mean_val = sum(winsorized) / len(winsorized)
    var_val = sum((v - mean_val) ** 2 for v in winsorized) / len(winsorized)
    std_val = var_val**0.5

    for j, idx in enumerate(eligible_indices):
        if std_val > 0:
            z = (winsorized[j] - mean_val) / std_val
        else:
            z = 0.0
        csv_rows[idx]["clinical_alpha_z"] = round(z, 4)


# ---------------------------------------------------------------------------
# Far-horizon catalyst hydration
# ---------------------------------------------------------------------------


def _hydrate_far_horizon_catalysts(
    csv_rows: List[dict],
    trial_records: List[dict],
    as_of_date: str,
    ruleset: "DecisionRuleset",
    m3_summaries: Optional[Dict[str, Any]] = None,
) -> int:
    """Override no_upcoming/missing tickers with far_window when a far PCD exists.

    For each row where catalyst_mode is "no_upcoming" or "missing", scans
    trial_records for active INTERVENTIONAL trials with a future PCD within
    ``ruleset.far_window_days``.  If found, sets catalyst_mode="far_window"
    and populates catalyst fields with a dampened signal.

    Also scans M3 scored events as a fallback for names not covered by
    trial_records (e.g. events from SEC 8-K, press releases, or non-CTgov
    registries that produced scored DATA_READOUT / CT_PRIMARY_COMPLETION
    events but where integration.next_catalyst_date was null).

    Returns the count of overridden tickers (for telemetry).
    """
    as_of = datetime.strptime(as_of_date[:10], "%Y-%m-%d").date()
    far_max = ruleset.far_window_days

    # Build per-ticker: min future PCD days from active interventional trials
    # Only consider PCDs beyond the current catalyst window (> 270d) but
    # within far_window_days — that's the operational gap we're filling.
    # Also include PCDs within 270d that the pipeline missed (missing mode).
    min_future_pcd: Dict[str, Tuple[int, str, str]] = {}  # ticker → (days, source, event_type)
    for t in trial_records:
        tk = t.get("ticker", "")
        if not tk:
            continue
        stype = (t.get("study_type") or "").upper()
        if "INTERVENTIONAL" not in stype:
            continue
        status = (t.get("status") or "?").upper()
        if status in _TERMINAL_NEGATIVE_STATUSES:
            continue
        pcd = t.get("primary_completion_date")
        if not pcd:
            continue
        try:
            pcd_date = datetime.strptime(pcd[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        days = (pcd_date - as_of).days
        if days <= 0:
            continue  # past PCD — not a future catalyst
        if days > far_max:
            continue  # beyond far window
        if tk not in min_future_pcd or days < min_future_pcd[tk][0]:
            min_future_pcd[tk] = (days, "CTGOV_PCD_FAR", "CT_PRIMARY_COMPLETION")

    # M3 fallback: scan scored events for names not yet covered
    if m3_summaries:
        _clinical_types = frozenset(
            {"CT_PRIMARY_COMPLETION", "CT_STUDY_COMPLETION", "DATA_READOUT", "DATA_PRESENTATION"}
        )
        as_of_str = as_of_date[:10]
        for tk_upper, summary in m3_summaries.items():
            if tk_upper in min_future_pcd:
                continue  # already covered by trial_records
            if not isinstance(summary, dict):
                continue
            for event in summary.get("events") or []:
                if not isinstance(event, dict):
                    continue
                et = event.get("event_type", "")
                if et not in _clinical_types:
                    continue
                ed = event.get("event_date", "")
                if not ed or ed <= as_of_str:
                    continue
                try:
                    days = (datetime.strptime(ed[:10], "%Y-%m-%d").date() - as_of).days
                except (ValueError, TypeError):
                    continue
                if days > far_max:
                    continue
                src = event.get("source", "M3_EVENT")
                if tk_upper not in min_future_pcd or days < min_future_pcd[tk_upper][0]:
                    min_future_pcd[tk_upper] = (days, src, et)

    # Override eligible rows
    n_overrides = 0
    for row in csv_rows:
        mode = row.get("catalyst_mode", row.get("de_catalyst_mode", ""))
        if mode not in ("no_upcoming", "missing"):
            continue
        tk = row.get("ticker", "")
        if tk not in min_future_pcd:
            continue
        days, source, event_type = min_future_pcd[tk]
        row["catalyst_mode"] = "far_window"
        row["catalyst_days"] = days
        row["catalyst_source"] = source
        row["catalyst_event_type"] = event_type
        row["catalyst_strength"] = "far"
        row["catalyst_decay_w"] = ruleset.far_window_decay_mult
        # Recompute next_catalyst_date / bounds to stay consistent with catalyst_days.
        # The canonical date block (run_screen.py:4788) ran before this hydration
        # step, so next_catalyst_date may reflect a stale M3 event date. Override
        # it here so next_catalyst_date == as_of + catalyst_days.
        _far_next = (as_of + timedelta(days=days)).strftime("%Y-%m-%d")
        row["next_catalyst_date"] = _far_next
        # Date bounds: far-window dates lack day precision; use ±15d (MONTH-grade)
        _far_base = datetime.strptime(_far_next, "%Y-%m-%d")
        row["catalyst_date_lower"] = (_far_base - timedelta(days=15)).strftime("%Y-%m-%d")
        row["catalyst_date_upper"] = (_far_base + timedelta(days=15)).strftime("%Y-%m-%d")
        n_overrides += 1

    return n_overrides


# ---------------------------------------------------------------------------
# Catalyst shadow metrics telemetry
# ---------------------------------------------------------------------------

_GOOD_CATALYST_MODES = frozenset({"specific_days", "blended_window", "far_window"})
_BAD_CATALYST_MODES = frozenset({"no_upcoming", "missing"})

_DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _find_prior_rankings(snapshot_dir: Path, current_date: str) -> Optional[dict]:
    """Find the most recent prior snapshot with a tier_dev-bearing rankings.csv.

    Returns ``{"date": str, "rows": [dict, ...]}`` or *None*.
    """
    if not snapshot_dir.exists():
        return None

    all_dates = sorted(
        [d.name for d in snapshot_dir.iterdir() if d.is_dir() and _DATE_DIR_RE.match(d.name)],
        reverse=True,
    )
    if current_date not in all_dates:
        return None

    idx = all_dates.index(current_date)
    # Walk backward (max 10 candidates) looking for a valid prior
    for candidate in all_dates[idx + 1 : idx + 11]:
        rankings_csv = snapshot_dir / candidate / "rankings.csv"
        if not rankings_csv.exists():
            continue
        with open(rankings_csv, "r", encoding="utf-8") as f:
            header = f.readline().strip()
        if "tier_dev" not in header.split(","):
            continue
        # Valid prior — read full CSV
        with open(rankings_csv, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return {"date": candidate, "rows": rows}

    return None


_TERMINAL_NEGATIVE_STATUSES = frozenset({"WITHDRAWN", "TERMINATED", "SUSPENDED"})


def _classify_coverage_buckets(
    csv_rows: List[dict],
    trial_records: List[dict],
    as_of_date: str,
    max_window: int = 270,
) -> dict:
    """Classify each ticker into a catalyst coverage bucket.

    Buckets:
      OK      — Has catalyst signal (specific_days, blended_window, or far_window)
      A       — No trials in snapshot at all
      B       — Trials exist but none are INTERVENTIONAL
      C       — All interventional trials are terminal (W/T/S)
      D       — Active interventional trials but ALL PCDs in the past
      E       — Active trials, all forward PCDs > max_window
      F       — Active trials, mix of past + far (>max_window) PCDs
      G       — Active trials with near PCD (should have catalyst — investigate)
      MISSING — catalyst_mode == "missing"

    Returns dict with per-bucket counts + aggregate keys.
    """
    as_of = datetime.strptime(as_of_date[:10], "%Y-%m-%d").date()
    all_tickers = {r.get("ticker", "") for r in csv_rows}

    # Build per-ticker trial info
    info: Dict[str, dict] = {}
    for tk in all_tickers:
        info[tk] = {
            "n_trials": 0,
            "n_interventional": 0,
            "n_active_interventional": 0,
            "near_pcds": [],
            "past_pcds": [],
            "far_pcds": [],
        }

    for t in trial_records:
        tk = t.get("ticker", "")
        if tk not in info:
            continue
        rec = info[tk]
        rec["n_trials"] += 1

        stype = (t.get("study_type") or "").upper()
        status = (t.get("status") or "?").upper()
        is_terminal = status in _TERMINAL_NEGATIVE_STATUSES

        if "INTERVENTIONAL" not in stype:
            continue
        rec["n_interventional"] += 1
        if not is_terminal:
            rec["n_active_interventional"] += 1

        pcd = t.get("primary_completion_date")
        if not pcd:
            continue
        try:
            pcd_date = datetime.strptime(pcd[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        if is_terminal:
            continue

        days = (pcd_date - as_of).days
        if days <= 0:
            rec["past_pcds"].append(days)
        elif days <= max_window:
            rec["near_pcds"].append(days)
        else:
            rec["far_pcds"].append(days)

    # Classify each ticker
    counts: Dict[str, int] = {k: 0 for k in ("OK", "A", "B", "C", "D", "E", "F", "G", "MISSING")}
    for row in csv_rows:
        tk = row.get("ticker", "")
        mode = row.get("catalyst_mode", row.get("de_catalyst_mode", "missing"))
        if mode in ("specific_days", "blended_window", "far_window"):
            bucket = "OK"
        elif mode == "missing":
            bucket = "MISSING"
        elif not tk or tk not in info:
            bucket = "MISSING"
        else:
            ti = info[tk]
            if ti["n_trials"] == 0:
                bucket = "A"
            elif ti["n_interventional"] == 0:
                bucket = "B"
            elif ti["n_active_interventional"] == 0:
                bucket = "C"
            elif ti["near_pcds"]:
                bucket = "G"
            elif ti["past_pcds"] and ti["far_pcds"]:
                bucket = "F"
            elif ti["far_pcds"]:
                bucket = "E"
            elif ti["past_pcds"]:
                bucket = "D"
            else:
                bucket = "A"
        counts[bucket] += 1

    total = len(csv_rows)
    n_no_upcoming_total = sum(counts[k] for k in "ABCDEFG")
    n_addressable_gap = counts["D"] + counts["E"] + counts["F"]

    result = {}
    for k in ("OK", "A", "B", "C", "D", "E", "F", "G", "MISSING"):
        result[f"coverage_bucket_{k}"] = counts[k]
    result["n_no_upcoming_total"] = n_no_upcoming_total
    result["n_addressable_gap"] = n_addressable_gap
    result["n_coverage_total"] = total
    return result


def _compute_shadow_metrics(
    csv_rows: List[dict],
    as_of_date: str,
    snapshot_dir: Path,
    source_mix: Optional[dict],
    trial_records: Optional[List[dict]] = None,
    max_window: int = 270,
) -> dict:
    """Compute catalyst shadow metrics for telemetry sidecar.

    All metrics are derived from the current ``csv_rows`` (already written to
    rankings.csv) and, optionally, a prior snapshot for transition/overlap
    metrics.
    """
    prior = _find_prior_rankings(snapshot_dir, as_of_date)
    prior_date = prior["date"] if prior else None

    # ----- Transition metrics (dev-only, common tickers) -----
    bad_to_good_count: Optional[int] = None
    good_to_bad_count: Optional[int] = None
    bad_to_good_tickers: Optional[List[str]] = None
    good_to_bad_tickers: Optional[List[str]] = None
    bad_to_good_in_top60: Optional[int] = None
    bad_to_good_in_top100: Optional[int] = None

    if prior is not None:
        prior_mode_by_ticker = {
            r["ticker"]: r.get("catalyst_mode", r.get("de_catalyst_mode", ""))
            for r in prior["rows"]
            if r.get("archetype") == "drug_developer"
        }
        bad_to_good_list: List[str] = []
        good_to_bad_list: List[str] = []
        for row in csv_rows:
            if row.get("archetype") != "drug_developer":
                continue
            ticker = row.get("ticker", "")
            if ticker not in prior_mode_by_ticker:
                continue  # new ticker, skip
            cur_mode = row.get("catalyst_mode", row.get("de_catalyst_mode", ""))
            prev_mode = prior_mode_by_ticker[ticker]
            if prev_mode in _BAD_CATALYST_MODES and cur_mode in _GOOD_CATALYST_MODES:
                bad_to_good_list.append(ticker)
            elif prev_mode in _GOOD_CATALYST_MODES and cur_mode in _BAD_CATALYST_MODES:
                good_to_bad_list.append(ticker)

        bad_to_good_count = len(bad_to_good_list)
        good_to_bad_count = len(good_to_bad_list)
        bad_to_good_tickers = sorted(bad_to_good_list)
        good_to_bad_tickers = sorted(good_to_bad_list)

    # ----- Distribution metrics (current snapshot, dev-only GOOD-mode) -----
    good_days: List[float] = []
    all_days: List[float] = []
    for row in csv_rows:
        if row.get("archetype") != "drug_developer":
            continue
        raw = row.get("catalyst_days", row.get("de_catalyst_days", ""))
        if raw in (None, ""):
            continue
        try:
            val = float(raw)
        except (ValueError, TypeError):
            continue
        all_days.append(val)
        cur_mode = row.get("catalyst_mode", row.get("de_catalyst_mode", ""))
        if cur_mode in _GOOD_CATALYST_MODES:
            good_days.append(val)

    median_catalyst_days_good: Optional[float] = None
    if good_days:
        median_catalyst_days_good = round(median(good_days), 1)

    p10: Optional[float] = None
    p50: Optional[float] = None
    p90: Optional[float] = None
    if len(all_days) >= 2:
        deciles = quantiles(all_days, n=10)  # returns 9 cut-points
        p10 = round(deciles[0], 1)
        p50 = round(deciles[4], 1)
        p90 = round(deciles[8], 1)
    elif len(all_days) == 1:
        p10 = p50 = p90 = round(all_days[0], 1)

    # ----- Overlap metrics (full universe, by composite_rank) -----
    top60_overlap: Optional[float] = None
    top100_overlap: Optional[float] = None

    if prior is not None:

        def _top_n_tickers(rows, n):
            valid = []
            for r in rows:
                cr = r.get("composite_rank", "")
                if cr in (None, ""):
                    continue
                try:
                    valid.append((int(cr), r.get("ticker", "")))
                except (ValueError, TypeError):
                    continue
            valid.sort()
            return {t for _, t in valid[:n]}

        cur_set_60 = _top_n_tickers(csv_rows, 60)
        prior_set_60 = _top_n_tickers(prior["rows"], 60)
        cur_set_100 = _top_n_tickers(csv_rows, 100)
        prior_set_100 = _top_n_tickers(prior["rows"], 100)

        def _jaccard(a, b):
            if not a and not b:
                return 1.0
            if not a or not b:
                return 0.0
            return len(a & b) / len(a | b)

        top60_overlap = round(_jaccard(cur_set_60, prior_set_60), 4)
        top100_overlap = round(_jaccard(cur_set_100, prior_set_100), 4)

        # B→G flips that landed in current top-N (decision-relevant)
        if bad_to_good_tickers is not None:
            bg_set = set(bad_to_good_tickers)
            bad_to_good_in_top60 = len(bg_set & cur_set_60)
            bad_to_good_in_top100 = len(bg_set & cur_set_100)

    # ----- Source attribution (from catalyst_source_mix) -----
    sec_8k_events: Optional[int] = None
    ctgov_events: Optional[int] = None
    fda_events: Optional[int] = None

    if source_mix and isinstance(source_mix.get("by_source"), dict):
        by_src = source_mix["by_source"]
        sec_8k_events = by_src.get("SEC_8K_FILING", 0)
        ctgov_events = by_src.get("CTGOV_CALENDAR", 0) + by_src.get("CTGOV", 0)
        fda_events = sum(
            by_src.get(k, 0) for k in ("FDA_CALENDAR", "FDA_ADCOM_CALENDAR", "FEDERAL_REGISTER", "FDA_PDUFA")
        )

    # ----- SEC 8-K future coverage (from catalyst_source_mix) -----
    sec8k_future_count: Optional[int] = None
    sec8k_future_tickers: Optional[int] = None
    sec8k_future_days_p50: Optional[int] = None
    sec8k_future_days_p90: Optional[int] = None
    sec8k_future_alert: Optional[str] = None

    if source_mix and isinstance(source_mix.get("sec8k_future"), dict):
        _fut = source_mix["sec8k_future"]
        sec8k_future_count = _fut.get("count", 0)
        sec8k_future_tickers = _fut.get("tickers", 0)
        sec8k_future_days_p50 = _fut.get("days_ahead_p50")
        sec8k_future_days_p90 = _fut.get("days_ahead_p90")

        # Red flag: zero future events when total 8K count is >0
        if sec8k_future_count == 0 and (sec_8k_events or 0) > 0:
            sec8k_future_alert = "ZERO_FUTURE_EVENTS"

        # Collapse detection: compare to prior shadow metrics
        if sec8k_future_count is not None and sec8k_future_alert is None and prior is not None:
            _prior_shadow_path = snapshot_dir / prior["date"] / "catalyst_shadow_metrics.json"
            if _prior_shadow_path.exists():
                try:
                    _ps = json.loads(_prior_shadow_path.read_text(encoding="utf-8"))
                    _prior_fc = _ps.get("sec8k_future_count")
                    if _prior_fc and _prior_fc > 0 and sec8k_future_count / _prior_fc < 0.50:
                        sec8k_future_alert = f"COLLAPSED_{sec8k_future_count / _prior_fc:.0%}_vs_prior"
                except (json.JSONDecodeError, OSError, TypeError, ZeroDivisionError):
                    pass

    # ----- A-tier count (dev-only) -----
    a_tier_count = sum(1 for r in csv_rows if r.get("archetype") == "drug_developer" and r.get("tier_dev") == "A")

    metrics = {
        "as_of_date": as_of_date,
        "prior_date": prior_date,
        "bad_to_good_count": bad_to_good_count,
        "good_to_bad_count": good_to_bad_count,
        "bad_to_good_tickers": bad_to_good_tickers,
        "good_to_bad_tickers": good_to_bad_tickers,
        "bad_to_good_in_top60": bad_to_good_in_top60,
        "bad_to_good_in_top100": bad_to_good_in_top100,
        "median_catalyst_days_good": median_catalyst_days_good,
        "p10_catalyst_days": p10,
        "p50_catalyst_days": p50,
        "p90_catalyst_days": p90,
        "A_tier_count": a_tier_count,
        "top60_overlap": top60_overlap,
        "top100_overlap": top100_overlap,
        "sec_8k_events": sec_8k_events,
        "ctgov_events": ctgov_events,
        "fda_events": fda_events,
        "sec8k_future_count": sec8k_future_count,
        "sec8k_future_tickers": sec8k_future_tickers,
        "sec8k_future_days_p50": sec8k_future_days_p50,
        "sec8k_future_days_p90": sec8k_future_days_p90,
        "sec8k_future_alert": sec8k_future_alert,
    }

    # ----- Coverage bucket telemetry (requires trial_records) -----
    if trial_records is not None:
        bucket_metrics = _classify_coverage_buckets(
            csv_rows,
            trial_records,
            as_of_date,
            max_window,
        )
        metrics.update(bucket_metrics)

    return metrics


# ---------------------------------------------------------------------------
# Output hygiene: deterministic defaults for z-score / continuous columns
# ---------------------------------------------------------------------------
_ALWAYS_NUMERIC_DEFAULTS = {
    "clinical_score_z_tier": 0.0,
    "coinvest_score_z": 0.0,
    "inst_delta_z": 0.0,
    "catalyst_decay_w": 0.0,
}

# Signal presence flags — written as string booleans for CSV compatibility
_SIGNAL_FLAG_DEFAULTS = {
    "has_coinvest_signal": "False",
    "has_inst_delta": "False",
    "has_catalyst_signal": "False",
}


def _ensure_defaults(csv_rows, defaults=_ALWAYS_NUMERIC_DEFAULTS):
    """Fill missing/empty/NaN values with deterministic defaults.

    This is a safety net — the z-score blocks should populate these for most
    tickers, but platform_* archetypes (devices, diagnostics, services) can
    fall through the clinical_score_z_tier cohort logic.
    """
    for row in csv_rows:
        for field, default in defaults.items():
            val = row.get(field)
            if val is None or str(val).strip() == "" or str(val).strip().lower() == "nan":
                row[field] = default
        # Signal presence flags (string booleans, never overwrite if already set)
        for flag, flag_default in _SIGNAL_FLAG_DEFAULTS.items():
            row.setdefault(flag, flag_default)


# Alpha table resolution — delegated to common.alpha_table
from common.alpha_table import check_artifact_marker as _check_artifact_marker
from common.alpha_table import resolve_alpha_table_path as _resolve_alpha_table_path_shared


def _resolve_alpha_table_path(
    ruleset: "DecisionRuleset",
    as_of_date: str,
    logger: logging.Logger,
) -> tuple:
    """Thin wrapper preserving the original 3-arg call signature.

    Derives project_root from ``__file__`` at call time (not import time)
    so tests can monkeypatch ``run_screen.__file__``.
    """
    import run_screen as _self_mod

    project_root = Path(_self_mod.__file__).resolve().parent
    return _resolve_alpha_table_path_shared(
        ruleset,
        as_of_date,
        project_root,
        logger,
    )


def load_cache_refresh_sidecar(path: Path) -> Dict[str, Any]:
    """Load cache_refresh sidecar and extract rejection metadata.

    Returns a dict with keys: sidecar, had_rejections, rejected_sources,
    summary, banner.  Safe to call on missing/corrupt files.
    """
    empty: Dict[str, Any] = {
        "sidecar": None,
        "had_rejections": False,
        "rejected_sources": [],
        "summary": {},
        "banner": "",
    }
    try:
        if not path.exists():
            return empty
        refresh = json.loads(path.read_text())
        results_list = refresh.get("results", [])
        rejected = [r for r in results_list if not r.get("accepted", True)]
        summary = {
            r["source"]: {k: r.get(k) for k in ("accepted", "reason", "count", "prior_count", "committed")}
            for r in results_list
        }
        banner = "; ".join(f"{r['source']}:{r.get('reason', '?')}" for r in rejected)
        return {
            "sidecar": path.name,
            "had_rejections": len(rejected) > 0,
            "rejected_sources": [r["source"] for r in rejected],
            "summary": summary,
            "banner": banner,
        }
    except Exception as exc:
        logger.warning("Source reliability sidecar unreadable: %s", exc)
        return empty


def _compute_config_fingerprint(
    decision_mode,
    ranking_mode,
    tier_filter,
    top_k,
    alpha_schema_mode,
    ruleset_id,
    reliability_table,
    ranker_mode="pairwise_minimal",
) -> str:
    """SHA256[:8] of deterministic config subset."""
    import hashlib

    rel_hash = hashlib.sha256(json.dumps(reliability_table or [], sort_keys=True).encode()).hexdigest()[:16]
    obj = {
        "decision_mode": decision_mode,
        "ranking_mode": ranking_mode,
        "ranker_mode": ranker_mode,
        "tier_filter": sorted(tier_filter) if tier_filter else [],
        "top_k": top_k,
        "alpha_schema_mode": alpha_schema_mode,
        "ruleset_id": ruleset_id,
        "source_reliability_table_hash": rel_hash,
    }
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:8]


# =============================================================================
# COVERAGE QUALITY ARTIFACT (v1)
# =============================================================================


def _write_coverage_quality(
    snap_path: Path,
    csv_rows: List[dict],
    metadata: dict,
    as_of_date: str,
    log: logging.Logger,
    options_freshness: Optional[Dict[str, Any]] = None,
    generated_at: Optional[str] = None,
) -> None:
    """Generate coverage_quality.json and coverage_quality.md in the snapshot dir.

    Consolidates the key coverage and rescue metrics into a single artifact
    so that every screen run produces an observable record of data health.
    """
    n_eligible = sum(1 for r in csv_rows if _is_eligible(r))
    n_total = len(csv_rows)
    dev_rows = [r for r in csv_rows if r.get("archetype") == "drug_developer"]
    n_dev = len(dev_rows)

    # -- Phase-2 health (read sidecar if present, else summarise from metadata) --
    health_path = snap_path / "phase2_health.json"
    if health_path.exists():
        health = json.loads(health_path.read_text())
    else:
        health = {"status": "UNAVAILABLE", "reasons": [], "metrics": {}}

    # -- Catalyst coverage by mode --
    mode_dist: Dict[str, int] = {}
    for r in dev_rows:
        m = r.get("catalyst_mode", "") or "missing"
        mode_dist[m] = mode_dist.get(m, 0) + 1

    n_specific = mode_dist.get("specific_days", 0)
    n_far_window = mode_dist.get("far_window", 0)
    n_no_upcoming = mode_dist.get("no_upcoming", 0)
    n_missing_cat = mode_dist.get("missing", 0)
    catalyst_coverage_pct = round(n_specific / max(n_dev, 1) * 100, 1)

    # -- Catalyst-family coverage --
    family_dist: Dict[str, int] = {}
    family_missing: List[str] = []
    for r in csv_rows:
        fam = r.get("catalyst_family", "")
        if fam:
            family_dist[fam] = family_dist.get(fam, 0) + 1
        elif _is_eligible(r):
            family_missing.append(r.get("ticker", "?"))
    n_with_family = sum(family_dist.values())
    catalyst_family_coverage_pct = round(n_with_family / max(n_total, 1) * 100, 1)

    # -- Far-window / M3-fallback rescue counters (all archetypes) --
    n_far_ctgov = 0
    n_far_m3 = 0
    n_far_total = 0
    for r in csv_rows:
        if r.get("catalyst_mode") == "far_window":
            n_far_total += 1
            src = r.get("catalyst_source", "")
            if "M3" in src.upper():
                n_far_m3 += 1
            else:
                n_far_ctgov += 1

    # -- Tier-0 fallback rescue (tickers with catalyst_strength == "far") --
    # This captures names rescued by the four-tier lookup (earliest future event)
    n_tier0_rescue = sum(1 for r in csv_rows if r.get("catalyst_strength") == "far")

    # -- Drawdown rescue --
    n_dd_rescued = sum(1 for r in csv_rows if r.get("dd_rel_margin_rescued") == "1")

    # -- Secondary regulatory coverage --
    n_reg_flagged = sum(1 for r in csv_rows if r.get("has_regulatory_upcoming_180d") == "1")
    reg_secondary_pct = round(n_reg_flagged / max(n_eligible, 1) * 100, 1)

    # Regulatory by proximity bucket
    reg_0_30 = 0
    reg_31_90 = 0
    reg_91_180 = 0
    for r in csv_rows:
        if r.get("has_regulatory_upcoming_180d") != "1":
            continue
        try:
            d = int(r.get("regulatory_days", 9999))
        except (ValueError, TypeError):
            continue
        if d <= 30:
            reg_0_30 += 1
        elif d <= 90:
            reg_31_90 += 1
        else:
            reg_91_180 += 1

    # -- Sponsor / drawdown / catalyst-component coverage --
    n_sponsor = sum(
        1 for r in csv_rows if isinstance(r.get("sponsor_tier1_count"), (int, float)) and r["sponsor_tier1_count"] > 0
    )
    sponsor_pct = round(n_sponsor / max(n_total, 1) * 100, 1)

    n_drawdown_present = sum(1 for r in csv_rows if r.get("de_drawdown") not in (None, "", "None"))
    drawdown_pct = round(n_drawdown_present / max(n_total, 1) * 100, 1)

    catalyst_comp_pct = round(metadata.get("options_diagnostics", {}).get("coverage_pct", 0.0), 1)

    # -- Missingness penalty distribution --
    miss_counts: Dict[int, int] = {}
    for r in csv_rows:
        try:
            p = int(r.get("missingness_penalty", 0))
        except (ValueError, TypeError):
            p = 0
        miss_counts[p] = miss_counts.get(p, 0) + 1

    # -- Top missing names (eligible, no catalyst family) --
    top_missing = sorted(family_missing)[:15]

    # -- Market-model disagreement summary --
    disagree_dist: Dict[str, int] = {}
    disagree_high_names: List[str] = []
    for r in csv_rows:
        d = r.get("market_model_disagreement", "")
        if d:
            disagree_dist[d] = disagree_dist.get(d, 0) + 1
        if d == "high" and _is_eligible(r):
            disagree_high_names.append(r.get("ticker", "?"))

    # -- Build JSON --
    # -- Risk-feature coverage (universe-wide) --
    # de_vol_60d had a silent gap: it was read from universe.json's
    # defensive_features block (partial) while beta/drawdown/rsi are hydrated
    # from price_history. Surface per-feature coverage over ALL screened rows so
    # a coverage regression can't hide (the exposure_missingness gate only
    # inspects held / top-K names).
    _risk_feats = ["de_vol_60d", "de_beta_xbi_60d", "de_drawdown", "de_rsi_14d", "de_alpha_60d"]
    risk_feature_coverage: Dict[str, Any] = {"n_total": n_total}
    for _f in _risk_feats:
        _present = sum(1 for r in csv_rows if str(r.get(_f, "")).strip() not in ("", "None", "nan", "NaN"))
        risk_feature_coverage[_f] = {
            "present": _present,
            "coverage_pct": round(_present / max(n_total, 1) * 100, 1),
        }

    cq: Dict[str, Any] = {
        "schema_version": "coverage_quality.v1",
        "as_of_date": as_of_date,
        "generated_at": generated_at or _deterministic_timestamp(as_of_date),
        "phase2_health": {
            "status": health.get("status"),
            "reasons": health.get("reasons", []),
            "a_count": health.get("metrics", {}).get("a_count"),
            "portfolio_size": health.get("metrics", {}).get("portfolio_size"),
        },
        "catalyst_coverage": {
            "n_dev": n_dev,
            "n_specific_days": n_specific,
            "n_far_window": n_far_window,
            "n_no_upcoming": n_no_upcoming,
            "n_missing": n_missing_cat,
            "specific_days_pct": catalyst_coverage_pct,
            "mode_distribution": dict(sorted(mode_dist.items(), key=lambda x: -x[1])),
        },
        "catalyst_family_coverage": {
            "n_with_family": n_with_family,
            "n_total": n_total,
            "coverage_pct": catalyst_family_coverage_pct,
            "family_distribution": dict(sorted(family_dist.items(), key=lambda x: -x[1])),
        },
        "risk_feature_coverage": risk_feature_coverage,
        "rescue_counters": {
            "far_window_ctgov": n_far_ctgov,
            "far_window_m3_fallback": n_far_m3,
            "far_window_total": n_far_total,
            "tier0_earliest_future_event": n_tier0_rescue,
            "drawdown_margin_rescued": n_dd_rescued,
        },
        "regulatory_secondary": {
            "n_flagged": n_reg_flagged,
            "n_eligible": n_eligible,
            "coverage_pct": reg_secondary_pct,
            "by_proximity": {
                "0_30d": reg_0_30,
                "31_90d": reg_31_90,
                "91_180d": reg_91_180,
            },
        },
        "component_coverage": {
            "sponsor_pct": sponsor_pct,
            "drawdown_pct": drawdown_pct,
            "options_pct": catalyst_comp_pct,
        },
        "missingness_penalty_distribution": dict(sorted(miss_counts.items())),
        "top_missing_names": top_missing,
        "market_model_disagreement": {
            "distribution": dict(sorted(disagree_dist.items())),
            "n_high": len(disagree_high_names),
            "high_names": sorted(disagree_high_names)[:15],
        },
        "options_data_freshness": options_freshness or {},
        "term_structure_flags": {
            "n_mismatch": sum(1 for r in csv_rows if r.get("ts_flag_type") == "MARKET_SEES_SOONER"),
            "n_blind_spot": sum(1 for r in csv_rows if r.get("ts_flag_type") == "BLIND_SPOT"),
            "n_not_pricing": sum(1 for r in csv_rows if r.get("ts_flag_type") == "MARKET_NOT_PRICING_EVENT"),
            "flagged_names": sorted(r.get("ticker", "?") for r in csv_rows if r.get("ts_flag") == "1")[:20],
        },
    }

    # -- Write JSON --
    cq_json_path = snap_path / "coverage_quality.json"
    with open(cq_json_path, "w", encoding="utf-8") as f:  # coverage_quality.json — read by collection health
        json.dump(cq, f, indent=2, default=str)
        f.write("\n")

    # -- Write Markdown --
    h = cq["phase2_health"]
    cc = cq["catalyst_coverage"]
    cf = cq["catalyst_family_coverage"]
    rc = cq["rescue_counters"]
    rs = cq["regulatory_secondary"]
    comp = cq["component_coverage"]

    lines = [
        f"# Coverage Quality Report — {as_of_date}",
        "",
    ]

    # Staleness warning
    _of = options_freshness or {}
    if _of and not _of.get("all_fresh", True):
        _tt = _of.get("tastytrade_status", "unknown")
        _mv = _of.get("massive_status", "unknown")
        lines.append(f"**OPTIONS DATA STALE** — Tastytrade: {_tt.upper()}, Massive: {_mv.upper()}")
        lines.append(
            "Options signals (quality composite, disagreement overlay, term structure flags) "
            "may be unreliable for this run."
        )
        lines.append("")

    lines.append(f"## Phase-2 Health: **{h['status']}**")
    lines.append("")
    if h["reasons"]:
        for reason in h["reasons"]:
            lines.append(f"- {reason}")
    else:
        lines.append("No issues detected.")
    lines.append("")
    lines.append(f"- A-tier count: {h['a_count']}")
    lines.append(f"- Portfolio size: {h['portfolio_size']}")
    lines.append("")

    lines.append("## Catalyst Coverage (drug_developer)")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| specific_days | {cc['n_specific_days']}/{cc['n_dev']} ({cc['specific_days_pct']}%) |")
    lines.append(f"| far_window | {cc['n_far_window']} |")
    lines.append(f"| no_upcoming | {cc['n_no_upcoming']} |")
    lines.append(f"| missing | {cc['n_missing']} |")
    lines.append("")

    lines.append("## Catalyst-Family Coverage")
    lines.append("")
    lines.append(f"**{cf['n_with_family']}/{cf['n_total']}** ({cf['coverage_pct']}%)")
    lines.append("")
    if cf["family_distribution"]:
        lines.append("| Family | Count |")
        lines.append("|--------|-------|")
        for fam, cnt in cf["family_distribution"].items():
            lines.append(f"| {fam} | {cnt} |")
        lines.append("")

    lines.append("## Rescue Counters")
    lines.append("")
    lines.append(f"| Rescue Type | Count |")
    lines.append(f"|-------------|-------|")
    lines.append(f"| Far-window (CTGov PCD) | {rc['far_window_ctgov']} |")
    lines.append(f"| Far-window (M3 fallback) | {rc['far_window_m3_fallback']} |")
    lines.append(f"| Tier-0 earliest future event | {rc['tier0_earliest_future_event']} |")
    lines.append(f"| Drawdown margin rescue | {rc['drawdown_margin_rescued']} |")
    lines.append("")

    lines.append("## Secondary Regulatory Coverage")
    lines.append("")
    lines.append(f"**{rs['n_flagged']}/{rs['n_eligible']}** ({rs['coverage_pct']}%)")
    lines.append("")
    lines.append(f"| Window | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| 0-30d | {rs['by_proximity']['0_30d']} |")
    lines.append(f"| 31-90d | {rs['by_proximity']['31_90d']} |")
    lines.append(f"| 91-180d | {rs['by_proximity']['91_180d']} |")
    lines.append("")

    lines.append("## Component Coverage")
    lines.append("")
    lines.append(f"| Component | Coverage |")
    lines.append(f"|-----------|----------|")
    lines.append(f"| Sponsor (13F) | {comp['sponsor_pct']}% |")
    lines.append(f"| Drawdown | {comp['drawdown_pct']}% |")
    lines.append(f"| Options | {comp['options_pct']}% |")
    lines.append("")

    if top_missing:
        lines.append("## Top Missing Names (eligible, no catalyst family)")
        lines.append("")
        lines.append(", ".join(top_missing))
        lines.append("")

    mmd = cq.get("market_model_disagreement", {})
    if mmd.get("n_high", 0) > 0:
        lines.append("## Market-Model Disagreement (shadow diagnostic)")
        lines.append("")
        dd = mmd.get("distribution", {})
        lines.append(f"| Level | Count |")
        lines.append(f"|-------|-------|")
        for level in ["low", "medium", "high"]:
            lines.append(f"| {level} | {dd.get(level, 0)} |")
        lines.append("")
        lines.append(f"**High-disagreement names** ({mmd['n_high']}): " f"{', '.join(mmd.get('high_names', []))}")
        lines.append("")

    tsf = cq.get("term_structure_flags", {})
    n_ts_total = tsf.get("n_mismatch", 0) + tsf.get("n_blind_spot", 0) + tsf.get("n_not_pricing", 0)
    if n_ts_total > 0:
        lines.append("## Term Structure Flags (Agent 0 staleness / blind spot)")
        lines.append("")
        lines.append("| Flag Type | Count |")
        lines.append("|-----------|-------|")
        lines.append(f"| MARKET_SEES_SOONER | {tsf.get('n_mismatch', 0)} |")
        lines.append(f"| MARKET_NOT_PRICING_EVENT | {tsf.get('n_not_pricing', 0)} |")
        lines.append(f"| BLIND_SPOT | {tsf.get('n_blind_spot', 0)} |")
        lines.append("")
        flagged = tsf.get("flagged_names", [])
        if flagged:
            lines.append(f"Flagged: {', '.join(flagged)}")
            lines.append("")

    cq_md_path = snap_path / "coverage_quality.md"
    with open(cq_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log.info(
        "[COVERAGE] %s — catalyst=%s/%s (%s%%), family=%s/%s (%s%%), reg_secondary=%s%%, "
        "rescue: far=%s (ctgov=%s, m3=%s), tier0=%s, dd=%s",
        h["status"],
        cc["n_specific_days"],
        cc["n_dev"],
        cc["specific_days_pct"],
        cf["n_with_family"],
        cf["n_total"],
        cf["coverage_pct"],
        rs["coverage_pct"],
        rc["far_window_total"],
        rc["far_window_ctgov"],
        rc["far_window_m3_fallback"],
        rc["tier0_earliest_future_event"],
        rc["drawdown_margin_rescued"],
    )


# =========================================================================
# save_validation_snapshot phase helpers
# Extracted to create testable phase boundaries within the God function.
# =========================================================================


def _enrich_market_data_fields(
    csv_rows: List[Dict[str, Any]],
    market_data_by_ticker: Optional[Dict[str, Dict]] = None,
) -> None:
    """PHASE 2a: Inject market data fields for Event EV expectation model."""
    _ensure_defaults(csv_rows)
    _mkt_lookup = market_data_by_ticker or {}
    for _r in csv_rows:
        _tk = (_r.get("ticker") or "").upper()
        _md = _mkt_lookup.get(_tk, {})
        if _md:
            if "short_interest_pct" not in _r or not _r.get("short_interest_pct"):
                _r["short_interest_pct"] = _md.get("short_percent", "")
            if "close_price" not in _r or not _r.get("close_price"):
                _r["close_price"] = _md.get("price", "")
            if "market_cap_mm" not in _r or not _r.get("market_cap_mm"):
                _mcap = _md.get("market_cap")
                _r["market_cap_mm"] = round(_mcap / 1e6, 1) if _mcap else ""


def _enrich_applicability_flags(csv_rows: List[Dict[str, Any]]) -> None:
    """PHASE 2b: Set applicability flags for scorecard N/A-aware missingness."""
    for _r in csv_rows:
        _arch = _r.get("archetype", "")
        _r["has_commercial_quality"] = 1 if _arch != "drug_developer" else 0
        _r["has_clinical_optionality_dev"] = 1 if _arch == "drug_developer" else 0


def _finalize_priced_move(csv_rows: List[Dict[str, Any]]) -> None:
    """PHASE 2z: Final field injection — priced_move_pct from straddle_price.

    Must run AFTER all options scoring (straddle, PCR, etc) and BEFORE CSV write.
    This was the source of a P0 execution-order bug when it ran too early.

    Unit conversion: straddle_price / implied_event_move are in decimal
    (0.39 = 39% move), but EES base-rate tables and conditional model
    use percentage points (39.0 = 39% move). Multiply by 100.
    """
    _n_set = 0
    _n_suspicious = 0
    for _r in csv_rows:
        if not _r.get("priced_move_pct") and _r.get("straddle_price"):
            try:
                _sp = float(_r["straddle_price"])
                _pm = round(_sp * 100.0, 4)
                _r["priced_move_pct"] = _pm
                _n_set += 1
                # Sanity: priced_move_pct should be 1-200 range (percentage points)
                if _pm < 1.0 or _pm > 500.0:
                    _n_suspicious += 1
            except (ValueError, TypeError):
                pass
    if _n_suspicious > 0:
        logger.warning(
            "[Phase2z] %d/%d priced_move_pct values outside [1, 500] — possible unit error",
            _n_suspicious,
            _n_set,
        )


def save_validation_snapshot(
    snapshot_dir: Path,
    as_of_date: str,
    results: Dict[str, Any],
    version: str,
    decision_mode: str = "observe",
    ruleset: Optional[DecisionRuleset] = None,
    price_history_path: Optional[Path] = None,
    market_data_by_ticker: Optional[Dict[str, Dict]] = None,
    trial_records: Optional[List[Dict]] = None,
    alpha_schema_mode: str = "fail",
    inputs_manifest_mode: str = "off",
    ranking_mode: str = "decision",
    prior_snapshot_dir: Optional[Path] = None,
    ctgov_cache_date: Optional[str] = None,
    phase_scores_version: str = "v3",
    force_overwrite: bool = False,
    regime_result: Optional[Dict[str, Any]] = None,
    ranker_mode: str = "pairwise_minimal",
    data_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Save a lightweight validation snapshot for future forward-looking backtests.

    Writes two files into snapshot_dir/YYYY-MM-DD/:
      - rankings.csv: one row per ranked ticker with scores and signals
      - metadata.json: run parameters, version, archetype distribution

    These snapshots capture the model's *point-in-time* rankings so that
    future price data can measure predictive accuracy (IC, decile lift, etc.)
    without survivorship bias from using only current-day rankings.

    Args:
        snapshot_dir: Base directory for snapshots (e.g. data/snapshots)
        as_of_date: Screen date string (YYYY-MM-DD)
        results: Full pipeline results dict
        version: Pipeline version string
        decision_mode: "observe" (sort by composite_rank), "actionable"
                       (sort by tier-based deterministic ordering), or "phase2"
                       (observe + emit decision_portfolio.csv/json with A+B filter)
        ruleset: DecisionRuleset config (defaults to DEFAULT_RULESET)
        price_history_path: Path to price_history.csv for drawdown hydration
        ranking_mode: "decision" (sort rankings.csv by DE actionable sort key)
                      or "composite" (legacy: re-sort by composite_rank in
                      observe/phase2 modes)

    Returns:
        Path to the snapshot directory, or None if save failed
    """
    snap_path = snapshot_dir / as_of_date

    # CCFT Frozen: refuse to overwrite an existing snapshot unless
    # the caller explicitly opts in via --force-overwrite.
    _existing_csv = snap_path / "rankings.csv"
    if _existing_csv.exists() and not force_overwrite:
        logger.warning(
            "Snapshot already exists at %s — refusing to overwrite "
            "(CCFT Frozen principle). Use --force-overwrite to replace.",
            snap_path,
        )
        return None

    try:
        snap_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning(f"Could not create snapshot directory {snap_path}: {e}")
        return None

    _prior_dir = prior_snapshot_dir or snapshot_dir

    ranked = results.get("module_5_composite", {}).get("ranked_securities", [])
    if not ranked:
        logger.warning("No ranked securities to snapshot")
        return None

    # Validate critical fields before writing snapshot — catch corrupt module 5
    # output early rather than producing CSV rows with empty fields.
    _REQUIRED_SNAPSHOT_FIELDS = {"ticker", "composite_rank"}
    _sample = ranked[0] if ranked else {}
    _missing = _REQUIRED_SNAPSHOT_FIELDS - set(_sample.keys())
    if _missing:
        logger.error(
            "Module 5 output missing required fields %s in first record — "
            "aborting snapshot to prevent corrupt output.",
            _missing,
        )
        return None

    ranked = sorted(ranked, key=lambda x: x.get("composite_rank", 999))
    archetypes = results.get("company_archetypes", {})
    industry_groups = results.get("industry_groups", {})
    m3_summaries = (results.get("module_3_catalyst") or {}).get("summaries")

    # Load PDUFA manual dates once for secondary regulatory catalyst scan.
    # This ensures tickers with PDUFA dates in the manual file get flagged
    # even when M3 events don't include the PDUFA entry.
    _pdufa_manual = _load_snapshot_pdufa_manual(data_dir=data_dir, as_of_date=as_of_date)

    # Load source reliability table (empirical slip-based trust actions).
    # Used to penalize noisy calendar sources in priority ranking.
    _reliability_table: list = []
    try:
        from common.source_reliability import load_reliability_table

        _rel_root = Path(__file__).resolve().parent / "artifacts" / "calendar_source_reliability"
        if _rel_root.is_dir():
            _rel_dates = sorted(d.name for d in _rel_root.iterdir() if d.is_dir() and d.name <= as_of_date)
            if _rel_dates:
                _rel_path = _rel_root / _rel_dates[-1] / "source_reliability.json"
                _reliability_table = load_reliability_table(_rel_path)
    except Exception as exc:
        logger.warning("Source reliability load failed: %s", exc)
        _reliability_table = []

    # Build the event ledger for direct regulatory event scanning.
    # M3's scored events (summaries[ticker].events) only include CTGov/SEC
    # pipeline events — not FDA ADCOM, FDA regulatory, EMA, or PDUFA ledger
    # entries.  Building the ledger here and scanning it directly picks up
    # additional regulatory sources (FDA AdCom, EMA committee, etc.).
    _event_ledger: list = []
    try:
        from event_ledger import LedgerConfig, build_event_ledger

        _event_ledger = build_event_ledger(
            date.fromisoformat(as_of_date),
            LedgerConfig(),
        )
        logger.info("Snapshot: event ledger built with %d entries for regulatory scan", len(_event_ledger))
    except Exception as exc:
        logger.warning("Snapshot: event ledger build failed (%s) — regulatory scan uses M3+PDUFA only", exc)

    # --- FDA AdCom vote features (passive pilot) ---
    _adcom_vote_lookup: Dict[str, Dict[str, Any]] = {}
    try:
        _fda_cache_dir = Path(__file__).resolve().parent / "cache" / "fda"
        _adcom_cache_path = _fda_cache_dir / f"adcom_calendar_{as_of_date}.json"
        if not _adcom_cache_path.exists():
            # Fallback: most recent adcom cache at or before as_of_date
            _adcom_candidates = sorted(
                (
                    p
                    for p in _fda_cache_dir.glob("adcom_calendar_*.json")
                    if p.name <= f"adcom_calendar_{as_of_date}.json"
                ),
                reverse=True,
            )
            if _adcom_candidates:
                _adcom_cache_path = _adcom_candidates[0]
                logger.info("Snapshot: AdCom cache fallback to %s", _adcom_cache_path.name)
        if _adcom_cache_path.exists():
            _adcom_raw = json.loads(_adcom_cache_path.read_text())
            _adcom_vote_lookup = build_adcom_vote_lookup(_adcom_raw, as_of_date)
            logger.info("Snapshot: AdCom vote lookup built for %d tickers", len(_adcom_vote_lookup))
    except Exception as exc:
        logger.warning("Snapshot: AdCom vote lookup failed (%s) — columns will be empty", exc)

    # --- Build company name lookup from Module 1 universe ---
    # Some tickers in universe.json have market_data.company_name set to the
    # sector label (e.g. "Healthcare") instead of the real company name.
    # Detect these and substitute from the override table below.
    _SECTOR_LABELS = frozenset(
        {
            "Healthcare",
            "Biotechnology",
            "Technology",
            "Financial",
            "Industrials",
            "Consumer Defensive",
            "Consumer Cyclical",
            "Basic Materials",
            "Energy",
            "Communication Services",
            "Real Estate",
            "Utilities",
        }
    )
    _NAME_OVERRIDES: Dict[str, str] = {
        "ABEO": "Abeona Therapeutics Inc.",
        "ABOS": "Acumen Pharmaceuticals, Inc.",
        "ACRV": "Acrivon Therapeutics, Inc.",
        "AKTX": "Akari Therapeutics Plc",
        "ALDX": "Aldeyra Therapeutics, Inc.",
        "ALGS": "Aligos Therapeutics, Inc.",
        "ARTV": "Artiva Biotherapeutics, Inc.",
        "ASMB": "Assembly Biosciences, Inc.",
        "BLTE": "Belite Bio, Inc",
        "BMEA": "Biomea Fusion, Inc.",
        "CELC": "Celcuity Inc.",
        "CERS": "Cerus Corporation",
        "CLYM": "Climb Bio, Inc.",
        "DNTH": "Dianthus Therapeutics, Inc.",
        "DRUG": "Bright Minds Biosciences Inc.",
        "ELDN": "Eledon Pharmaceuticals, Inc.",
        "ENGN": "enGene Holdings Inc.",
        "GBIO": "Generation Bio Co.",
        "HOWL": "Werewolf Therapeutics, Inc.",
        "IKT": "Inhibikase Therapeutics, Inc.",
        "IMTX": "Immatics N.V.",
        "JBIO": "Jade Biosciences, Inc.",
        "LBRX": "LB Pharmaceuticals Inc",
        "LYRA": "Lyra Therapeutics, Inc.",
        "MLTX": "MoonLake Immunotherapeutics",
        "NAUT": "Nautilus Biotechnology, Inc.",
        "NKTX": "Nkarta, Inc.",
        "OCS": "Oculis Holding AG",
        "PRLD": "Prelude Therapeutics Incorporated",
        "PRQR": "ProQR Therapeutics N.V.",
        "PVLA": "Palvella Therapeutics, Inc.",
        "SABS": "SAB Biotherapeutics, Inc.",
        "SERA": "Sera Prognostics, Inc.",
        "SKYE": "Skye Bioscience, Inc.",
        "TARA": "Protara Therapeutics, Inc.",
        "TCRX": "TScan Therapeutics, Inc.",
        "TENX": "Tenax Therapeutics, Inc.",
        "VKTX": "Viking Therapeutics, Inc.",
        "VRDN": "Viridian Therapeutics, Inc.",
    }
    _ticker_to_name: Dict[str, str] = {}
    for sec in (results.get("module_1_universe") or {}).get("active_securities", []):
        _t = sec.get("ticker")
        if _t:
            raw_name = sec.get("company_name") or ""
            if raw_name in _SECTOR_LABELS:
                _ticker_to_name[_t] = _NAME_OVERRIDES.get(_t, "")
            else:
                _ticker_to_name[_t] = raw_name

    # --- Helper: extract a named component's normalized score from component_scores ---
    def _component_score(rec, name):
        for c in rec.get("component_scores") or []:
            if c.get("name") == name:
                return c.get("normalized")
        return None

    # --- Load earnings calendar (best-effort, non-blocking) ---
    _earnings_by_ticker: Dict[str, str] = {}
    try:
        import glob as _glob_mod

        _earnings_dir = Path(__file__).resolve().parent / "artifacts" / "earnings_sync"
        _raw_files = sorted(_earnings_dir.glob("earnings_raw_*.json"), reverse=True)
        if _raw_files:
            with open(_raw_files[0]) as _ef:
                _earnings_raw = json.load(_ef)
            for _er in _earnings_raw.get("rows", []):
                _et = _er.get("symbol", "")
                _ed = _er.get("earnings_date", "")
                if _et and _ed:
                    # Keep earliest upcoming date per ticker
                    if _et not in _earnings_by_ticker or _ed < _earnings_by_ticker[_et]:
                        _earnings_by_ticker[_et] = _ed
            logger.info("Snapshot: loaded %d earnings dates from %s", len(_earnings_by_ticker), _raw_files[0].name)
    except Exception as exc:
        logger.debug("Snapshot: earnings calendar not available (%s)", exc)

    # --- Load AACT execution scores (best-effort, non-blocking) ---
    _aact_exec_by_ticker: Dict[str, float] = {}
    try:
        _aact_dir = Path(__file__).resolve().parent / "artifacts" / "aact_deltas"
        _aact_files = sorted(_aact_dir.glob("aact_deltas_*.json"), reverse=True)
        if _aact_files:
            with open(_aact_files[0]) as _af:
                _aact_data = json.load(_af)
            for _at in _aact_data.get("tickers", []):
                _atk = _at.get("ticker", "")
                _aes = _at.get("execution_score", 0)
                if _atk:
                    _aact_exec_by_ticker[_atk] = _aes
            logger.info(
                "Snapshot: loaded %d AACT execution scores from %s", len(_aact_exec_by_ticker), _aact_files[0].name
            )
    except Exception as exc:
        logger.debug("Snapshot: AACT deltas not available (%s)", exc)

    # --- Build ranking rows ---
    csv_rows = []
    for rec in ranked:
        ticker = rec.get("ticker", "")
        # Extract signal scores from nested dicts (momentum/valuation
        # are not in component_scores; catalyst/smart_money/clinical/
        # financial use component_scores.normalized as single source)
        mom = rec.get("momentum_signal") or {}
        val = rec.get("valuation_signal") or {}
        row = {
            "ticker": ticker,
            "company_name": _ticker_to_name.get(ticker, ""),
            "composite_rank": rec.get("composite_rank"),
            "composite_score": rec.get("composite_score"),
            "score_rank_pct": rec.get("score_rank_pct"),
            "score_z": rec.get("score_z"),
            "composite_score_attn": rec.get("composite_score_attn"),
            "score_rank_pct_attn": rec.get("score_rank_pct_attn"),
            "score_z_attn": rec.get("score_z_attn"),
            "stage_bucket": rec.get("stage_bucket"),
            "market_cap_bucket": rec.get("market_cap_bucket"),
            "severity": rec.get("severity"),
            "archetype": archetypes.get(ticker, ""),
            "industry_group": industry_groups.get(ticker, ""),
            "momentum_score": mom.get("momentum_score"),
            "catalyst_score": _component_score(rec, "catalyst"),
            "smart_money_score": _component_score(rec, "smart_money"),
            "valuation_score": val.get("valuation_score"),
            "clinical_score": _component_score(rec, "clinical"),
            "financial_score": _component_score(rec, "financial"),
            "confidence_overall": rec.get("confidence_overall"),
            "next_earnings_date": _earnings_by_ticker.get(ticker, ""),
            "aact_execution_score": _aact_exec_by_ticker.get(ticker, ""),
        }
        csv_rows.append(row)

    # --- Compute clinical_optionality_pct_dev (inverted percentile within dev cohort) ---
    # Empirical finding: in dev-stage (drug_developer), lower clinical_score →
    # higher forward returns.  clinical_optionality_pct_dev inverts the ranking so
    # higher = more optionality = empirically better forward performance.
    dev_rows = [
        (i, r)
        for i, r in enumerate(csv_rows)
        if r.get("archetype") == "drug_developer" and r.get("clinical_score") not in (None, "")
    ]
    if dev_rows:
        # Sort dev tickers by clinical_score DESC then ticker ASC (deterministic)
        dev_sorted = sorted(dev_rows, key=lambda x: (-float(x[1]["clinical_score"]), x[1].get("ticker", "")))
        n_dev = len(dev_sorted)
        for rank_i, (row_idx, _) in enumerate(dev_sorted, start=1):
            # Continuity-corrected percentile: highest clinical_score → highest pct
            p = (n_dev - rank_i + 0.5) / n_dev
            p = max(0.001, min(0.999, p))
            csv_rows[row_idx]["clinical_rank_pct_dev"] = round(p, 6)
            csv_rows[row_idx]["clinical_optionality_pct_dev"] = round(1.0 - p, 6)

    # --- Compute clinical_alpha_z (cross-sectional z within dev cohort) ---
    if trial_records:
        _readout_days = _compute_clinical_readout_days(trial_records, as_of_date)
        _m4_lookup = {s["ticker"]: s for s in (results.get("module_4_clinical", {}).get("scores") or [])}
        _compute_clinical_alpha_z(csv_rows, _m4_lookup, _readout_days)

    # --- Compute clinical_score_z (PIT-safe, cross-sectional within cohort) ---
    # Each archetype cohort gets its own z-score (never mixed).
    _CLINICAL_COHORTS = {
        "drug_developer": "dev",
        "commercial_biotech": "comm_biotech",
        "commercial_pharma": "comm_pharma",
    }
    _dev_indices = []  # still needed for tier z-score below
    for _cohort_arch in _CLINICAL_COHORTS:
        _cohort_idx = [
            i
            for i, r in enumerate(csv_rows)
            if r.get("archetype") == _cohort_arch
            and r.get("clinical_score") is not None
            and r.get("clinical_score") != ""
        ]
        if _cohort_arch == "drug_developer":
            _dev_indices = _cohort_idx
        if _cohort_idx:
            _scores = [float(csv_rows[i]["clinical_score"]) for i in _cohort_idx]
            _mean = sum(_scores) / len(_scores)
            _var = sum((s - _mean) ** 2 for s in _scores) / len(_scores)
            _std = _var**0.5
            for j, idx in enumerate(_cohort_idx):
                if _std > 0:
                    csv_rows[idx]["clinical_score_z"] = round((_scores[j] - _mean) / _std, 4)
                else:
                    csv_rows[idx]["clinical_score_z"] = 0.0

    # NOTE: clinical_score_z_tier computed AFTER DE loop (needs tier_dev + tier_commercial)
    from collections import defaultdict as _defaultdict

    # --- Clinical Calendar Alpha v2 features ---
    if trial_records:
        from common.clinical_calendar_alpha import (
            CalendarAlphaConfig,
            compose_clinical_score_v2,
            compute_competitive_intensity,
            compute_design_quality,
            compute_endpoint_strength,
            compute_execution_momentum,
            compute_program_features,
            compute_readout_density,
            z_score_dict,
        )

        _active_tickers = [r.get("ticker", "") for r in csv_rows if r.get("ticker")]
        _ccav2_program = compute_program_features(trial_records, as_of_date)
        _ccav2_readout = compute_readout_density(trial_records, as_of_date)
        _ccav2_design = compute_design_quality(trial_records, as_of_date)
        _ccav2_endpoint = compute_endpoint_strength(trial_records, as_of_date)

        # Prior trial loading for execution momentum
        _prior_trials = None
        _ctgov_root = Path(__file__).parent / "cache" / "ctgov"
        if _ctgov_root.is_dir():
            import re as _re_mod

            _date_files = sorted(
                [
                    f.stem.replace("trial_records_", "")
                    for f in _ctgov_root.glob("trial_records_*.json")
                    if _re_mod.match(r"^trial_records_\d{4}-\d{2}-\d{2}\.json$", f.name)
                ],
                reverse=True,
            )
            _prior_date = next((d for d in _date_files if d < as_of_date), None)
            if _prior_date:
                _prior_path = _ctgov_root / f"trial_records_{_prior_date}.json"
                try:
                    import json as _json_mod

                    with open(_prior_path) as _pf:
                        _prior_trials = _json_mod.load(_pf)
                    logger.info(f"  Calendar Alpha v2: prior trials loaded from {_prior_date}")
                except Exception as _pt_err:
                    logger.warning(f"  Calendar Alpha v2: failed to load prior trials: {_pt_err}")

        _ccav2_momentum = compute_execution_momentum(trial_records, _prior_trials, as_of_date)

        # Competitive intensity (build landscape once, score all tickers)
        try:
            _ccav2_competition = compute_competitive_intensity(
                trial_records,
                _active_tickers,
                as_of_date,
            )
        except Exception as _comp_err:
            logger.warning(f"  Calendar Alpha v2: competitive intensity failed: {_comp_err}")
            _ccav2_competition = {}

        # Cross-sectional z-scores for composition
        _z_rc = z_score_dict({tk: v["readout_curve_score"] for tk, v in _ccav2_readout.items()})
        _z_rd = z_score_dict({tk: v.get("readout_density_90", 0) for tk, v in _ccav2_readout.items()})
        _z_mom = z_score_dict({tk: v["execution_momentum_score"] for tk, v in _ccav2_momentum.items()})
        _z_des = z_score_dict({tk: v["design_quality_score"] for tk, v in _ccav2_design.items()})
        _z_ep = z_score_dict({tk: v["endpoint_strength_score"] for tk, v in _ccav2_endpoint.items()})
        # competition z already computed inside compute_competitive_intensity

        _ccav2_config = CalendarAlphaConfig(
            enable_sizing=ruleset.enable_clinical_sizing,
        )

        # Compose v2 score and write columns into csv_rows
        _ccav2_count = 0
        for row in csv_rows:
            tk = row.get("ticker", "").upper()
            if not tk:
                continue

            prog = _ccav2_program.get(tk, {})
            rd = _ccav2_readout.get(tk, {})
            mom = _ccav2_momentum.get(tk, {})
            des = _ccav2_design.get(tk, {})
            ep = _ccav2_endpoint.get(tk, {})
            comp = _ccav2_competition.get(tk, {})

            # Merge features for compose call
            merged = {}
            merged.update(prog)
            merged.update(rd)
            merged.update(mom)
            merged.update(des)
            merged.update(ep)
            merged.update(comp)

            # Base clinical_score
            cs_raw = row.get("clinical_score")
            cs = float(cs_raw) if cs_raw is not None and cs_raw != "" else None

            v2_score, sizing_mult, _tags = compose_clinical_score_v2(
                cs,
                merged,
                _ccav2_config,
                z_readout_curve=_z_rc.get(tk, 0.0),
                z_readout_density=_z_rd.get(tk, 0.0),
                z_momentum=_z_mom.get(tk, 0.0),
                z_design=_z_des.get(tk, 0.0),
                z_endpoint=_z_ep.get(tk, 0.0),
                z_competition=comp.get("competitive_intensity_z", 0.0),
            )

            row["clinical_score_v2"] = v2_score if v2_score is not None else ""
            row["lead_program_phase"] = prog.get("lead_program_phase", "")
            row["lead_program_readout_days"] = (
                prog.get("lead_program_readout_days") if prog.get("lead_program_readout_days") is not None else ""
            )
            row["program_count"] = prog.get("program_count", "")
            row["program_diversification"] = prog.get("program_diversification_score", "")
            row["readout_curve_score"] = rd.get("readout_curve_score", "")
            row["readout_density_90"] = rd.get("readout_density_90", "")
            row["late_stage_readouts_180"] = rd.get("late_stage_readouts_180", "")
            row["execution_momentum"] = mom.get("execution_momentum_score", "")
            row["design_quality_score"] = des.get("design_quality_score", "")
            row["endpoint_strength_score"] = ep.get("endpoint_strength_score", "")
            row["therapeutic_area"] = ep.get("therapeutic_area", "")
            row["competitive_intensity_z"] = comp.get("competitive_intensity_z", "")
            row["crowding_level"] = comp.get("crowding_level", "")
            row["sizing_multiplier_clinical"] = sizing_mult
            if v2_score is not None:
                _ccav2_count += 1

        logger.info(f"  Calendar Alpha v2: {_ccav2_count} tickers scored")

    # --- Clinical Quality Score (Spec 057, monitor-only) ---
    if trial_records:
        from common.clinical_quality_score import compute_clinical_quality_scores

        _cq_results = compute_clinical_quality_scores(trial_records, as_of_date)
        _cq_count = 0
        for row in csv_rows:
            tk = row.get("ticker", "").upper()
            cq = _cq_results.get(tk)
            if cq:
                row["clinical_quality_score"] = cq.clinical_quality_score
                row["clinical_quality_confidence"] = cq.clinical_quality_confidence
                row["endpoint_strength_tier"] = cq.endpoint_strength_tier
                row["design_rigor_tier"] = cq.design_rigor_tier
                row["prior_evidence_tier"] = cq.prior_evidence_tier
                row["mechanism_maturity_tier"] = cq.mechanism_maturity_tier
                row["clinical_quality_notes"] = cq.notes
                _cq_count += 1
            else:
                row["clinical_quality_score"] = ""
                row["clinical_quality_confidence"] = ""
                row["endpoint_strength_tier"] = ""
                row["design_rigor_tier"] = ""
                row["prior_evidence_tier"] = ""
                row["mechanism_maturity_tier"] = ""
                row["clinical_quality_notes"] = ""
        logger.info(f"  Clinical Quality Score: {_cq_count} tickers scored (monitor-only)")

    # --- Compute commercial_quality_pct (percentile within commercial cohort) ---
    _CQ_WEIGHTS = {"financial": 0.45, "valuation": 0.35, "momentum": 0.20}

    _non_dev_arch = lambda a: a and a != "drug_developer"
    comm_rows = [
        (i, r)
        for i, r in enumerate(csv_rows)
        if _non_dev_arch(r.get("archetype", ""))
        and r.get("financial_score") is not None
        and r.get("valuation_score") is not None
    ]
    if comm_rows:
        # Compute raw quality score
        for idx, r in comm_rows:
            cq = (
                _CQ_WEIGHTS["financial"] * (_safe_float(r["financial_score"], default=0.0))
                + _CQ_WEIGHTS["valuation"] * (_safe_float(r["valuation_score"], default=0.0))
                + _CQ_WEIGHTS["momentum"] * (_safe_float(r.get("momentum_score"), default=0.0))
            )
            csv_rows[idx]["commercial_quality"] = round(cq, 6)

        # Percentile rank (higher raw → higher pct)
        comm_sorted = sorted(
            comm_rows,
            key=lambda x: (csv_rows[x[0]].get("commercial_quality", 0), x[1].get("ticker", "")),
        )
        n_comm = len(comm_sorted)
        for rank_i, (row_idx, _) in enumerate(comm_sorted, start=1):
            p = (rank_i - 0.5) / n_comm
            p = max(0.001, min(0.999, p))
            csv_rows[row_idx]["commercial_quality_pct"] = round(p, 6)

    # --- Decision Engine v1: compute eligibility, overlays, sizing, tiering ---
    rec_by_ticker = {rec["ticker"]: rec for rec in ranked}

    # Hydrate missing drawdown data (key normalization + price-history fallback)
    if price_history_path is not None:
        n_hydrated = _hydrate_drawdown(rec_by_ticker, price_history_path, as_of_date)
        if n_hydrated:
            logger.info(f"Hydrated drawdown for {n_hydrated} tickers")

    # Hydrate beta_xbi_60d (defensive cache) + rsi_14d (price-history)
    n_beta_rsi = _hydrate_beta_rsi(rec_by_ticker, price_history_path, as_of_date)
    if n_beta_rsi:
        logger.info(f"Hydrated beta/RSI for {n_beta_rsi} fields")

    # Cost-aware sizing: initialize cost schedule when enabled
    cost_schedule = None
    rep_weight = 0.0
    if ruleset and ruleset.enable_cost_haircut and market_data_by_ticker:
        cost_schedule = CostSchedule(
            impact_cap_bps=ruleset.cost_impact_cap_bps,
        )
        rep_weight = 100.0 / PHASE2_DEFAULT_TOP_K

    # --- Alpha signal contract: validate DE inputs ---
    _alpha_schema_mode = alpha_schema_mode
    try:
        from alpha_signal_contract import validate_alpha_inputs

        _alpha_input_diag = validate_alpha_inputs(
            rec_by_ticker,
            csv_rows,
            schema_mode=_alpha_schema_mode,
        )
    except Exception as _asc_err:
        logger.warning(f"Alpha contract input validation error: {_asc_err}")
        _alpha_input_diag = None

    for row in csv_rows:
        ticker = row.get("ticker", "")
        rec = rec_by_ticker.get(ticker)
        if rec is None:
            continue
        arch = archetypes.get(ticker, "")
        opt = row.get("clinical_optionality_pct_dev")
        opt_float = float(opt) if opt is not None else None

        est_cost_bps = None
        if cost_schedule is not None:
            md = market_data_by_ticker.get(ticker, {})
            adv = (md.get("avg_volume") or 0) * (md.get("price") or 0)
            if adv > 0:
                cost_est = estimate_trade_cost(rep_weight, adv, cost_schedule)
                est_cost_bps = cost_est.round_trip_bps

        cq_pct = row.get("commercial_quality_pct")
        cq_pct_float = float(cq_pct) if cq_pct is not None else None

        _md = market_data_by_ticker.get(ticker, {}) if market_data_by_ticker else {}
        _mkt_cap = _md.get("market_cap")

        decision = compute_decision_fields(
            rec,
            arch,
            opt_float,
            ruleset=ruleset,
            est_cost_bps=est_cost_bps,
            commercial_quality_pct=cq_pct_float,
            market_cap=float(_mkt_cap) if _mkt_cap is not None else None,
        )
        row.update(decision)

        # Red flag audit trail: persist detection results from rec
        row["fundamental_red_flag"] = "1" if rec.get("fundamental_red_flag") else "0"
        row["fundamental_red_flag_reasons"] = ",".join(sorted(rec.get("fundamental_red_flag_reasons") or []))
        _surv_m = (rec.get("survivability_signal") or {}).get("metrics") or {}
        row["fundamental_red_flag_inputs"] = json.dumps(
            {
                "stage_bucket": rec.get("stage_bucket"),
                "lead_phase": rec.get("lead_phase"),
                "has_revenue": rec.get("has_revenue"),
                "tier1_count": (rec.get("coinvest") or {}).get("tier1_count"),
                "runway_months": _surv_m.get("effective_runway_months"),
                "burn_ttm": _surv_m.get("burn_ttm"),
                "cash_total": _surv_m.get("cash_total"),
                "surv_score": (rec.get("survivability_signal") or {}).get("score"),
                "dilution_risk": (rec.get("dilution_risk_signal") or {}).get("risk_level"),
            },
            sort_keys=True,
        )

        row["est_cost_bps"] = est_cost_bps if est_cost_bps is not None else ""
        # Signal presence flag for catalyst
        row["has_catalyst_signal"] = str(decision.get("catalyst_mode", "missing") != "missing")

        # Persist decision engine INPUT fields for archive self-containment.
        # These de_-prefixed columns capture the raw inputs the engine read,
        # so future archives don't need enrichment scripts.
        cd = rec.get("catalyst_decay") or {}
        df = rec.get("defensive_features") or {}
        coinv = rec.get("coinvest") or {}
        mom_enh = (rec.get("score_breakdown") or {}).get("enhancements", {}).get("momentum") or {}
        row["de_catalyst_days"] = cd.get("days_to_catalyst", "")
        row["de_catalyst_in_window"] = cd.get("in_optimal_window", "")
        row["de_catalyst_mode"] = decision.get("catalyst_mode", "")
        # Alpha: prefer price_history recompute, then pipeline fallback
        row["de_alpha_60d"] = df.get("alpha_60d", "")
        if row["de_alpha_60d"] == "":
            row["de_alpha_60d"] = mom_enh.get("alpha_60d", "")
        if row["de_alpha_60d"] == "":
            mom_top = rec.get("momentum_signal") or {}
            row["de_alpha_60d"] = mom_top.get("alpha_60d", "")
        row["de_alpha_60d_source"] = df.get("alpha_60d_source", "")
        row["de_alpha_60d_missing_reason"] = df.get("alpha_60d_missing_reason", "")
        t1 = coinv.get("tier1_count")
        row["de_tier1_count"] = t1 if t1 is not None else ""
        row["de_beta_xbi_60d"] = df.get("beta_xbi_60d", "")
        row["de_beta_xbi_60d_source"] = df.get("beta_xbi_60d_source", "")
        row["de_beta_xbi_60d_missing_reason"] = df.get("beta_xbi_60d_missing_reason", "")
        row["de_drawdown"] = df.get("drawdown", "")
        row["de_drawdown_missing_reason"] = df.get("drawdown_missing_reason", "")
        row["de_rsi_14d"] = df.get("rsi_14d", "")
        row["de_vol_60d"] = df.get("vol_60d", "")
        row["de_drawdown_xbi"] = df.get("drawdown_xbi", "")
        row["de_drawdown_rel_xbi"] = df.get("drawdown_rel_xbi", "")

        # Returns provenance: live screen always uses Morningstar pipeline
        row["returns_source"] = "morningstar"

        # Catalyst source provenance: extract from Module 3 nearest event.
        # Pass PDUFA manual so T1 regulatory events override nearer T3 CT.gov milestones.
        row["catalyst_source"] = _nearest_catalyst_source(
            m3_summaries, ticker, as_of_date=as_of_date, pdufa_manual=_pdufa_manual
        )
        row["catalyst_event_type"] = _nearest_catalyst_event_type(
            m3_summaries, ticker, as_of_date=as_of_date, pdufa_manual=_pdufa_manual
        )
        row["is_hard_catalyst"] = "1" if _is_hard_catalyst(row["catalyst_event_type"], row["catalyst_source"]) else "0"
        row["catalyst_family"] = classify_catalyst_family(row["catalyst_event_type"])
        # Fix catalyst_type_tier: DE computes it from overlays which lack event_type.
        # Override with the correct tier now that event_type is known.
        from decision_engine import classify_catalyst_type_tier

        row["catalyst_type_tier"] = classify_catalyst_type_tier(row["catalyst_event_type"])
        # Corroboration check: noisy clinical sources need independent confirmation
        _corr = _check_catalyst_corroboration(m3_summaries, ticker, row["catalyst_source"], row["catalyst_family"])
        row["catalyst_corroborated"] = "1" if _corr["corroborated"] else "0"
        row["catalyst_trust_status"] = _corr["trust_status"]
        row["catalyst_trust_reason"] = _corr["trust_reason"]
        row["binary_quality_score"] = compute_binary_quality_score(row)
        row.update(compute_event_quality_features(row))
        row.update(compute_clinical_91_180_quality(row))
        # FDA AdCom vote features (passive pilot)
        row.update(_adcom_vote_lookup.get(ticker.upper(), {col: "" for col in ADCOM_VOTE_COLUMNS}))

        # Secondary regulatory catalyst (independent of primary nearest)
        _reg_days, _reg_et = _nearest_regulatory_catalyst(
            m3_summaries,
            ticker,
            as_of_date,
            pdufa_manual=_pdufa_manual,
            event_ledger=_event_ledger,
        )
        row["regulatory_days"] = _reg_days
        row["regulatory_event_type"] = _reg_et
        row["has_regulatory_upcoming_180d"] = "1" if _reg_days else "0"
        # Stamp calendar confidence for portfolio construction
        _rc = "HIGH"  # default; M3/ledger events are inherently high-confidence
        if _reg_days and _pdufa_manual:
            for _prec in _pdufa_manual:
                if (_prec.get("ticker", "")).upper() == ticker.upper():
                    _rc = (_prec.get("confidence") or "HIGH").upper()
                    break
        row["regulatory_confidence"] = _rc if _reg_days else ""

        # Catalyst priority (resolve from event_type + source via ruleset policy)
        rs = ruleset or DEFAULT_RULESET
        row["cat_priority"] = resolve_catalyst_priority(
            row["catalyst_event_type"],
            row["catalyst_source"],
            rs,
        )

        # --- Explainability: top_3_drivers ---
        _score_cols = [
            "clinical_score",
            "catalyst_score",
            "momentum_score",
            "financial_score",
            "smart_money_score",
        ]
        _vals = {}
        for sc in _score_cols:
            v = row.get(sc)
            if v is not None and v != "":
                try:
                    _vals[sc] = float(v)
                except (ValueError, TypeError):
                    pass
        if _vals:
            _mean = sum(_vals.values()) / len(_vals)
            _devs = [(k, v - _mean) for k, v in _vals.items()]
            _devs.sort(key=lambda x: (-abs(x[1]), x[0]))
            row["top_3_drivers"] = ";".join(f"{k}:{d:+.1f}" for k, d in _devs[:3])
        else:
            row["top_3_drivers"] = ""

        # --- Explainability: catalyst_reason_detail ---
        _cat_days = row.get("catalyst_days", "")
        _cat_days_str = str(_cat_days) if _cat_days != "" else "NA"
        _decay_w = row.get("catalyst_decay_w", "")
        _decay_w_str = str(_decay_w) if _decay_w != "" else "NA"
        row["catalyst_reason_detail"] = (
            f"tier={row.get('tier_dev', '')}"
            f";reason={row.get('tier_reason', '')}"
            f";cat_days={_cat_days_str}"
            f";strength={row.get('catalyst_strength', '')}"
            f";decay_w={_decay_w_str}"
        )

    # --- Catalyst calendar v2: persist canonical date fields ---
    # Derive next_catalyst_date from catalyst_days + as_of_date (canonical, computed once)
    from datetime import timedelta as _td

    for row in csv_rows:
        _cd_raw = row.get("catalyst_days", "")
        try:
            _cd_int = int(float(_cd_raw)) if _cd_raw else 0
        except (ValueError, TypeError):
            _cd_int = 0

        # Canonical date
        if _cd_int > 0:
            _next_date = (datetime.strptime(as_of_date, "%Y-%m-%d") + _td(days=_cd_int)).strftime("%Y-%m-%d")
            row["next_catalyst_date"] = _next_date
        else:
            row["next_catalyst_date"] = ""

        # Date precision (prefer clinical_days_precision, fall back to heuristic)
        _prec = row.get("clinical_days_precision", "").strip()
        row["catalyst_date_precision"] = _prec if _prec else ("DAY" if row.get("is_hard_catalyst") == "1" else "")

        # Date bounds
        if _cd_int > 0 and row["next_catalyst_date"]:
            _base = datetime.strptime(row["next_catalyst_date"], "%Y-%m-%d")
            if _prec == "DAY":
                row["catalyst_date_lower"] = row["next_catalyst_date"]
                row["catalyst_date_upper"] = row["next_catalyst_date"]
            elif _prec == "MONTH":
                row["catalyst_date_lower"] = (_base - _td(days=15)).strftime("%Y-%m-%d")
                row["catalyst_date_upper"] = (_base + _td(days=15)).strftime("%Y-%m-%d")
            elif _prec in ("QUARTER", "HALF_YEAR"):
                row["catalyst_date_lower"] = (_base - _td(days=45)).strftime("%Y-%m-%d")
                row["catalyst_date_upper"] = (_base + _td(days=45)).strftime("%Y-%m-%d")
            else:
                row["catalyst_date_lower"] = ""
                row["catalyst_date_upper"] = ""
        else:
            row["catalyst_date_lower"] = ""
            row["catalyst_date_upper"] = ""

        # Calendar confidence score [0, 1]
        _conf = 0.0
        _has_signal = bool(_cd_int > 0 or row.get("catalyst_event_type", "").strip())

        if _has_signal:
            # Precision component (0.0 - 0.40)
            _prec_map = {"DAY": 0.40, "WEEK": 0.30, "MONTH": 0.20, "QUARTER": 0.10}
            _conf += _prec_map.get(row.get("catalyst_date_precision", ""), 0.10)

            # Source component (0.0 - 0.30)
            _src = row.get("catalyst_source", "").strip()
            _src_map = {"SEC_8K_FILING": 0.30, "CTGOV_CALENDAR": 0.25, "CTGOV_PCD_FAR": 0.15}
            _conf += _src_map.get(_src, 0.05)

            # Hard catalyst bonus (0.0 - 0.20)
            if row.get("is_hard_catalyst") == "1":
                _conf += 0.20

            # Consistency bonus (catalyst_days ≈ de_catalyst_days)
            _de_cd = 0
            try:
                _de_cd = int(float(row.get("de_catalyst_days", "") or 0))
            except (ValueError, TypeError):
                pass
            if _cd_int > 0 and _de_cd > 0 and abs(_cd_int - _de_cd) <= 7:
                _conf += 0.10

        row["calendar_confidence"] = str(round(min(1.0, _conf), 4)) if _has_signal else ""
        row["has_catalyst_signal"] = "1" if _has_signal else "0"
        row["has_tradeable_calendar"] = "1" if _conf >= 0.50 and 0 < _cd_int <= 120 else "0"
        # Spec 078: per-row catalyst quality classification (Lane A + Lane B).
        row["catalyst_quality"] = classify_catalyst_quality(
            row.get("catalyst_source", ""),
            row.get("catalyst_event_type", ""),
            row.get("calendar_confidence", ""),
        )

    _n_tradeable_cal = sum(1 for r in csv_rows if r.get("has_tradeable_calendar") == "1")
    logger.info(
        "Catalyst calendar v2: %d/%d with signal, %d tradeable calendars",
        sum(1 for r in csv_rows if r.get("has_catalyst_signal") == "1"),
        len(csv_rows),
        _n_tradeable_cal,
    )

    # --- Compute clinical_score_z_tier (needs tier_dev + tier_commercial from DE) ---
    # Drug developers: z within tier_dev groups
    _tier_groups = _defaultdict(list)  # tier_dev -> [(index, score)]
    for i in _dev_indices:
        tier = csv_rows[i].get("tier_dev", "")
        if tier:
            _tier_groups[tier].append((i, float(csv_rows[i]["clinical_score"])))
    for _tier, _pairs in _tier_groups.items():
        if len(_pairs) < 2:
            for idx, _ in _pairs:
                csv_rows[idx]["clinical_score_z_tier"] = 0.0
            continue
        _t_scores = [s for _, s in _pairs]
        _t_mean = sum(_t_scores) / len(_t_scores)
        _t_var = sum((s - _t_mean) ** 2 for s in _t_scores) / len(_t_scores)
        _t_std = _t_var**0.5
        for idx, s in _pairs:
            if _t_std > 0:
                csv_rows[idx]["clinical_score_z_tier"] = round((s - _t_mean) / _t_std, 4)
            else:
                csv_rows[idx]["clinical_score_z_tier"] = 0.0

    # Commercial cohorts: z within (archetype, tier_commercial) groups
    _COMM_ARCHETYPES = ("commercial_biotech", "commercial_pharma")
    _comm_tier_groups = _defaultdict(list)  # (archetype, tier_commercial) -> [(index, score)]
    for i, r in enumerate(csv_rows):
        arch = r.get("archetype", "")
        if arch not in _COMM_ARCHETYPES:
            continue
        cs = r.get("clinical_score")
        if cs is None or cs == "":
            continue
        tier = r.get("tier_commercial", "")
        if not tier:
            r["clinical_score_z_tier"] = 0.0
            continue
        _comm_tier_groups[(arch, tier)].append((i, float(cs)))
    for (_arch, _tier), _pairs in _comm_tier_groups.items():
        if len(_pairs) < 2:
            for idx, _ in _pairs:
                csv_rows[idx]["clinical_score_z_tier"] = 0.0
            continue
        _t_scores = [s for _, s in _pairs]
        _t_mean = sum(_t_scores) / len(_t_scores)
        _t_var = sum((s - _t_mean) ** 2 for s in _t_scores) / len(_t_scores)
        _t_std = _t_var**0.5
        for idx, s in _pairs:
            if _t_std > 0:
                csv_rows[idx]["clinical_score_z_tier"] = round((s - _t_mean) / _t_std, 4)
            else:
                csv_rows[idx]["clinical_score_z_tier"] = 0.0

    # --- Clinical Calendar Alpha v2 z-score (cross-sectional, ddof=0) ---
    # Computed over all tickers with a non-empty clinical_score_v2.
    _cv2_pairs = []  # [(index, value)]
    for _ci, _cr in enumerate(csv_rows):
        _cv2_raw = _cr.get("clinical_score_v2")
        if _cv2_raw is not None and _cv2_raw != "":
            try:
                _cv2_pairs.append((_ci, float(_cv2_raw)))
            except (ValueError, TypeError):
                pass
    if _cv2_pairs:
        _cv2_vals = [v for _, v in _cv2_pairs]
        _cv2_mean = sum(_cv2_vals) / len(_cv2_vals)
        _cv2_var = sum((v - _cv2_mean) ** 2 for v in _cv2_vals) / len(_cv2_vals)
        _cv2_std = _cv2_var**0.5
        for _ci, _cv2_v in _cv2_pairs:
            if _cv2_std > 0:
                csv_rows[_ci]["clinical_score_v2_z"] = round((_cv2_v - _cv2_mean) / _cv2_std, 4)
            else:
                csv_rows[_ci]["clinical_score_v2_z"] = 0.0
    # Default for tickers without clinical_score_v2
    for _cr in csv_rows:
        if "clinical_score_v2_z" not in _cr:
            _cr["clinical_score_v2_z"] = 0.0

    # --- Coinvest z-score: size-residualized cross-sectional z ---
    # Regresses sponsor_tier1_count on log(market_cap) to remove the mechanical
    # size confound (~21% attribution to size, per Spec 049 audit), then z-scores
    # the residual. The residualized version retains ~79% of the raw signal's
    # discriminative power while removing market-cap-driven overlap counting.
    _coinvest_mode = ruleset.coinvest_score_mode if ruleset else "tier1_count"
    _sparse_mode = ruleset.sparse_signal_mode if ruleset else "legacy"
    if _coinvest_mode == "tier1_count":
        _cz_pairs = []  # [(index, raw_value, has_signal, log_mcap_or_None)]
        for _ci, _cr in enumerate(csv_rows):
            _t1 = _cr.get("sponsor_tier1_count")
            _has = isinstance(_t1, (int, float))
            _raw = float(_t1) if _has else 0.0
            # Get market cap for size residualization
            _tk = _cr.get("ticker", "")
            _md = market_data_by_ticker.get(_tk, {}) if market_data_by_ticker else {}
            _mcap = _md.get("market_cap")
            _log_mcap = math.log(float(_mcap)) if _mcap and float(_mcap) > 0 else None
            _cz_pairs.append((_ci, _raw, _has, _log_mcap))
            csv_rows[_ci]["has_coinvest_signal"] = str(_has)

        if _cz_pairs:
            # --- Size residualization via OLS: coinvest = a + b*log(mcap) + residual ---
            # Use only tickers with both coinvest data and market cap
            _ols_pairs = [
                (v, lm) for _, v, has, lm in _cz_pairs if (has or _sparse_mode != "exclude_missing") and lm is not None
            ]
            _resid_by_idx = {}
            if len(_ols_pairs) >= 10:
                # OLS: b = cov(x,y)/var(x), a = mean(y) - b*mean(x)
                _n = len(_ols_pairs)
                _y_vals = [p[0] for p in _ols_pairs]
                _x_vals = [p[1] for p in _ols_pairs]
                _y_mean = sum(_y_vals) / _n
                _x_mean = sum(_x_vals) / _n
                _cov_xy = sum((_y_vals[i] - _y_mean) * (_x_vals[i] - _x_mean) for i in range(_n)) / _n
                _var_x = sum((_x_vals[i] - _x_mean) ** 2 for i in range(_n)) / _n
                _var_y = sum((_y_vals[i] - _y_mean) ** 2 for i in range(_n)) / _n
                if _var_x > 1e-12:
                    _beta = _cov_xy / _var_x
                    _alpha = _y_mean - _beta * _x_mean
                else:
                    _beta, _alpha = 0.0, _y_mean
                # Compute residuals for ALL tickers (including those without mcap → use raw)
                for _ci, _cv, _has, _lm in _cz_pairs:
                    if _lm is not None:
                        _predicted = _alpha + _beta * _lm
                        _resid_by_idx[_ci] = _cv - _predicted
                    else:
                        # No market cap → use demeaned raw (no size adjustment possible)
                        _resid_by_idx[_ci] = _cv - _y_mean
                logger.info(
                    "  Coinvest size residualization: beta=%.3f (log_mcap), n=%d, " "R²=%.3f",
                    _beta,
                    _n,
                    ((_cov_xy**2 / (_var_x * _var_y)) if (_var_x > 1e-12 and _var_y > 1e-12) else 0.0),
                )
            else:
                # Too few observations for OLS — fall back to raw demeaning
                _cz_real = [v for _, v, has, _ in _cz_pairs if has or _sparse_mode != "exclude_missing"]
                _cz_mean = sum(_cz_real) / len(_cz_real) if _cz_real else 0.0
                for _ci, _cv, _has, _ in _cz_pairs:
                    _resid_by_idx[_ci] = _cv - _cz_mean
                logger.info("  Coinvest size residualization: skipped (<%d obs), using demeaned raw", 10)

            # Z-score the residuals
            _resid_vals = [
                _resid_by_idx[_ci]
                for _ci, _, _has, _ in _cz_pairs
                if (_has or _sparse_mode != "exclude_missing") and _ci in _resid_by_idx
            ]
            if _resid_vals:
                _rz_mean = sum(_resid_vals) / len(_resid_vals)
                _rz_var = sum((v - _rz_mean) ** 2 for v in _resid_vals) / len(_resid_vals)
                _rz_std = _rz_var**0.5
            else:
                _rz_mean, _rz_std = 0.0, 0.0

            for _ci, _cv, _has, _ in _cz_pairs:
                if _sparse_mode == "exclude_missing" and not _has:
                    csv_rows[_ci]["coinvest_score_z"] = 0.0
                elif _rz_std > 0 and _ci in _resid_by_idx:
                    csv_rows[_ci]["coinvest_score_z"] = round((_resid_by_idx[_ci] - _rz_mean) / _rz_std, 4)
                else:
                    csv_rows[_ci]["coinvest_score_z"] = 0.0
                # Human-readable tag (still based on raw count for operator visibility)
                _t1_int = int(_cv) if _cv == _cv else 0
                csv_rows[_ci]["coinvest_tag"] = f"elite_{_t1_int}" if _t1_int > 0 else ""

            # --- Filing-age decay: exponential downweight of stale 13F data ---
            # 13F filings arrive ~45 days after quarter-end. By month 4+, the
            # information is increasingly stale. Apply exp(-age/half_life) so that
            # a 2-month-old filing has full weight but a 6-month-old filing is
            # attenuated. Half-life of 90 days means: 45d→0.71x, 90d→0.50x,
            # 180d→0.25x. This is a continuous replacement for the binary
            # fresh/stale recency gate.
            _DECAY_HALF_LIFE_DAYS = 90
            _decay_applied = 0
            for _ci, _, _, _ in _cz_pairs:
                _age_raw = csv_rows[_ci].get("coinvest_filing_age_days")
                _z_raw = csv_rows[_ci].get("coinvest_score_z", 0.0)
                if _age_raw and isinstance(_age_raw, (int, float)) and _age_raw > 0 and _z_raw != 0:
                    _decay = math.exp(-_age_raw / _DECAY_HALF_LIFE_DAYS * math.log(2))
                    csv_rows[_ci]["coinvest_score_z"] = round(_z_raw * _decay, 4)
                    _decay_applied += 1
            if _decay_applied:
                logger.info(
                    "  Coinvest filing-age decay: half_life=%dd, applied to %d/%d tickers",
                    _DECAY_HALF_LIFE_DAYS,
                    _decay_applied,
                    len(_cz_pairs),
                )

    # --- Compute institutional summary + delta (for z-score, written as sidecar later) ---
    inst_summary = None
    inst_delta = None
    if decision_mode == "phase2":
        try:
            from institutional_summary import (
                _find_prior_institutional_summary,
                _get_filing_period,
                build_institutional_summary,
                compute_institutional_delta,
            )

            _uni_tickers = {r.get("ticker", "") for r in csv_rows if r.get("ticker")}
            # nearest_prior_days=95 lets non-quarter-end monthly regen dates
            # use the most recent prior 13F cache (previous quarter-end or
            # recent daily cache). Exact-match dates are unaffected. The
            # resulting summary's cache_as_of_date records the actual source,
            # so _find_prior_institutional_summary's PIT guard still rejects
            # stale-source summaries as priors for cross-date delta.
            inst_summary = build_institutional_summary(as_of_date, _uni_tickers, nearest_prior_days=95)
            if inst_summary:
                # Use cross-quarter comparison to get real filing-period deltas
                _cache_base = Path(__file__).resolve().parent / "data" / "caches" / "sec_13f" / "PIT"
                _cur_filing = _get_filing_period(_cache_base, as_of_date)
                prior_inst = _find_prior_institutional_summary(
                    _prior_dir,
                    as_of_date,
                    cross_quarter=True,
                    current_filing_period=_cur_filing,
                    max_candidates=200,
                )
                if prior_inst:
                    inst_delta = compute_institutional_delta(inst_summary, prior_inst)
        except Exception as e:
            logger.warning("Could not compute institutional summary/delta: %s", e)

    # --- Registry transition detection ---
    # When the manager registry changes (CIK corrections, additions, removals),
    # inst_delta values are contaminated for one cycle because the "prior" and
    # "current" institutional summaries used different manager sets.
    _inst_registry_transition = False
    if inst_summary and prior_inst:
        _cur_mgr_count = inst_summary.get("elite_managers_with_filing", 0)
        _pri_mgr_count = prior_inst.get("elite_managers_with_filing", 0)
        if _cur_mgr_count != _pri_mgr_count:
            _inst_registry_transition = True
            logger.warning(
                "Registry transition detected: current=%d managers, prior=%d managers. "
                "inst_delta values may be contaminated this cycle.",
                _cur_mgr_count,
                _pri_mgr_count,
            )

    # --- Institutional delta z-score: cross-sectional z of net_elite_holders_delta ---
    if inst_delta:
        _id_pairs = []  # [(index, value, has_signal)]
        _inst_tickers = inst_delta.get("tickers", {})
        for _di, _dr in enumerate(csv_rows):
            _tk = _dr.get("ticker", "")
            _tk_delta = _inst_tickers.get(_tk, {})
            _has_inst = _tk in _inst_tickers
            _net = _tk_delta.get("net_elite_holders_delta", 0)
            csv_rows[_di]["inst_delta_net"] = _net
            csv_rows[_di]["inst_delta_new"] = _tk_delta.get("elite_new_count", 0)
            csv_rows[_di]["inst_delta_exit"] = _tk_delta.get("elite_exit_count", 0)
            csv_rows[_di]["has_inst_delta"] = str(_has_inst)
            _id_pairs.append((_di, float(_net), _has_inst))
        # z-score (ddof=0)
        if _sparse_mode == "exclude_missing":
            _id_real = [v for _, v, has in _id_pairs if has]
        else:
            _id_real = [v for _, v, _ in _id_pairs]
        if _id_real:
            _id_mean = sum(_id_real) / len(_id_real)
            _id_var = sum((v - _id_mean) ** 2 for v in _id_real) / len(_id_real)
            _id_std = _id_var**0.5
        else:
            _id_mean, _id_std = 0.0, 0.0
        for _di, _dv, _has_i in _id_pairs:
            if _sparse_mode == "exclude_missing" and not _has_i:
                csv_rows[_di]["inst_delta_z"] = 0.0
            elif _id_std > 0:
                csv_rows[_di]["inst_delta_z"] = round((_dv - _id_mean) / _id_std, 4)
            else:
                csv_rows[_di]["inst_delta_z"] = 0.0
    else:
        for _dr in csv_rows:
            _dr.setdefault("inst_delta_z", 0.0)
            _dr.setdefault("inst_delta_net", 0)
            _dr.setdefault("inst_delta_new", 0)
            _dr.setdefault("inst_delta_exit", 0)

    # --- Institutional delta coverage guard ---
    # When too few tickers have nonzero net delta, the z-scores are driven by
    # a handful of names — fragile signal.  Zero out z if coverage < threshold.
    _inst_nonzero_count = sum(1 for r in csv_rows if r.get("inst_delta_net", 0) != 0)
    _inst_nonzero_pct = round(100.0 * _inst_nonzero_count / len(csv_rows), 2) if csv_rows else 0.0
    for _dr in csv_rows:
        _dr["inst_delta_nonzero_pct"] = _inst_nonzero_pct
    _inst_min_pct = ruleset.institutional_sort_min_nonzero_pct if ruleset else 10.0
    if _inst_nonzero_pct < _inst_min_pct:
        if _inst_nonzero_count > 0:
            logger.warning(
                "inst_delta_z coverage guard: only %.1f%% of tickers have "
                "nonzero net delta (%d/%d) — below %.1f%% threshold; "
                "zeroing all inst_delta_z values",
                _inst_nonzero_pct,
                _inst_nonzero_count,
                len(csv_rows),
                _inst_min_pct,
            )
        for _dr in csv_rows:
            _dr["inst_delta_z"] = 0.0

    # --- Institutional flow diagnostics: absolute vs relative signal ---
    # Detects names where absolute institutional flow is positive but
    # cross-sectional z-score is negative (relative underperformance in
    # a strong cohort). Also flags reverse case.
    for _dr in csv_rows:
        _net = _dr.get("inst_delta_net", 0)
        _new = _dr.get("inst_delta_new", 0)
        _exit = _dr.get("inst_delta_exit", 0)
        _iz = float(_dr.get("inst_delta_z", 0))

        try:
            _net = int(_net)
            _new = int(_new)
            _exit = int(_exit)
        except (ValueError, TypeError):
            _net = _new = _exit = 0

        # Absolute flow: net positive means more new entrants than exits
        _abs_positive = _net > 0 or _new > _exit
        _abs_has_new = _new > 0
        _abs_negative = _net < 0 and _exit > _new
        _z_negative = _iz < -0.3
        _z_positive = _iz > 0.3

        # Registry transition guard
        _dr["inst_delta_regime"] = "transition" if _inst_registry_transition else "clean"

        if _inst_registry_transition:
            # Transition regime: deltas contaminated by registry change
            _dr["inst_flow_abs_positive"] = ""
            _dr["inst_flow_abs_negative"] = ""
            _dr["inst_relative_underperformance"] = ""
            _dr["inst_relative_outperformance"] = ""
            _dr["inst_flow_diagnostic"] = "registry_transition: delta contaminated, ignore relative signal this cycle"
        else:
            # Clean regime: apply flags normally
            _dr["inst_flow_abs_positive"] = "1" if _abs_positive else "0"
            _dr["inst_flow_abs_negative"] = "1" if _abs_negative else "0"
            _dr["inst_relative_underperformance"] = "1" if _abs_positive and _z_negative else "0"
            _dr["inst_relative_outperformance"] = "1" if _abs_negative and _z_positive else "0"

            _reasons = []
            if _abs_positive and _z_negative:
                _reasons.append(
                    f"abs_flow_positive(net={_net},new={_new}) but z={_iz:+.2f} (relative underperformance)"
                )
            elif _abs_negative and _abs_has_new and _z_negative:
                _reasons.append(
                    f"mixed_flow(net={_net},new={_new},exit={_exit}) z={_iz:+.2f} (net negative with new entrants)"
                )
            if _abs_negative and _z_positive:
                _reasons.append(
                    f"abs_flow_negative(net={_net},exit={_exit}) but z={_iz:+.2f} (relative outperformance)"
                )
            if abs(_iz) > 1.5:
                _reasons.append(f"extreme_inst_delta_z={_iz:+.2f}")
            _dr["inst_flow_diagnostic"] = "; ".join(_reasons) if _reasons else ""

    _n_transition = sum(1 for r in csv_rows if r.get("inst_delta_regime") == "transition")
    _n_rel_under = sum(1 for r in csv_rows if r.get("inst_relative_underperformance") == "1")
    _n_rel_over = sum(1 for r in csv_rows if r.get("inst_relative_outperformance") == "1")
    if _n_transition > 0:
        logger.info(
            "Institutional flow diagnostics: TRANSITION regime (%d names), flags suppressed",
            _n_transition,
        )
    elif _n_rel_under > 0 or _n_rel_over > 0:
        logger.info(
            "Institutional flow diagnostics: %d relative underperformers, %d relative outperformers",
            _n_rel_under,
            _n_rel_over,
        )

    # --- Clinical adjustment telemetry (mirrors DE formula, for monitoring) ---
    _clin_adj_dev = 0
    _clin_adj_comm = 0
    if ruleset and ruleset.enable_clinical_sort_signal:
        _stage_mults_d = dict(ruleset.clinical_stage_mults)
        for r in csv_rows:
            _cz = r.get("clinical_score_z_tier")
            if _cz is None or _cz == "":
                continue
            _cz = float(_cz)
            _stage = str(r.get("stage_bucket", ""))
            _sm = _stage_mults_d.get(_stage, 0.0)
            if ruleset.clinical_positive_only:
                _cz_eff = min(2.0, max(0.0, _cz))
            else:
                _cz_eff = max(-2.0, min(2.0, _cz))
            _adj = ruleset.clinical_sort_weight * _cz_eff * _sm
            if _adj != 0.0:
                _arch = r.get("archetype", "")
                if _arch == "drug_developer":
                    _clin_adj_dev += 1
                elif _arch in ("commercial_biotech", "commercial_pharma"):
                    _clin_adj_comm += 1
    logger.info(f"  Clinical sort adjustment: dev={_clin_adj_dev}, comm={_clin_adj_comm}")

    # --- Far-horizon catalyst hydration (opt-in, after DE sets catalyst_mode) ---
    _n_far_window = 0
    if ruleset and ruleset.far_window_days > 0:
        _n_far_window = _hydrate_far_horizon_catalysts(
            csv_rows,
            trial_records or [],
            as_of_date,
            ruleset,
            m3_summaries=m3_summaries,
        )
        if _n_far_window:
            logger.info(f"  Far-horizon catalyst: {_n_far_window} tickers overridden to far_window")

    # --- Alpha cohort scoring (opt-in via sort_anchor or composite_engine) ---
    _alpha_table_path_resolved = None
    _alpha_table_source = None  # "artifact"|"preexisting_local"|"rebuilt_in_run"|"static"|"static_fallback"
    _alpha_table = None
    _alpha_cells_populated = 0

    _run_alpha_cohort = ruleset and (
        ruleset.sort_anchor == "alpha_cohort" or ruleset.composite_engine == "alpha_cohort"
    )
    if _run_alpha_cohort:
        from module_5_alpha_cohort import attach_alpha_scores, load_alpha_cohort_table

        # Policy-based table resolution (adaptive alpha)
        _alpha_table_path_resolved, _alpha_table_source = _resolve_alpha_table_path(
            ruleset,
            as_of_date,
            logger,
        )
        _alpha_table = load_alpha_cohort_table(_alpha_table_path_resolved)
        _alpha_cells_populated = sum(1 for c in _alpha_table.get("cells", {}).values() if c.get("n", 0) > 0)
        attach_alpha_scores(
            csv_rows,
            _alpha_table,
            shrink_k=ruleset.alpha_cohort_shrink_k,
            clip_min=ruleset.alpha_cohort_clip_min,
            clip_max=ruleset.alpha_cohort_clip_max,
        )
        logger.info(f"  Alpha cohort scoring: {len(csv_rows)} tickers scored")

    # --- Alpha cohort composite override (opt-in via composite_engine) ---
    if ruleset and ruleset.composite_engine == "alpha_cohort":
        from common.score_to_er import attach_rank_and_z as _rerank

        # Overwrite composite_score with alpha_cohort_raw
        for row in csv_rows:
            row["composite_score"] = row.get("alpha_cohort_raw", 0.0)
        # Recompute score_rank_pct and score_z from new composite_score
        _rerank(csv_rows, score_key="composite_score")
        # Recompute composite_rank (1 = highest composite_score)
        _idx_sorted = sorted(
            range(len(csv_rows)),
            key=lambda i: (-float(csv_rows[i].get("composite_score", 0)), csv_rows[i].get("ticker", "")),
        )
        for rank, idx in enumerate(_idx_sorted, start=1):
            csv_rows[idx]["composite_rank"] = rank
        logger.info("  Composite engine: alpha_cohort (overrode composite_score/rank/pct)")

    # --- Alpha signal contract: validate DE outputs ---
    try:
        from alpha_signal_contract import validate_alpha_outputs

        _alpha_output_diag = validate_alpha_outputs(
            csv_rows,
            schema_mode=_alpha_schema_mode,
            alpha_cohort_enabled=bool(_run_alpha_cohort),
        )
    except Exception as _aso_err:
        logger.warning(f"Alpha contract output validation error: {_aso_err}")
        _alpha_output_diag = None

    # =========================================================================
    # PHASE 2: ENRICH — inject market data, applicability flags, options
    # This phase adds fields to csv_rows but does not change scores or ranks.
    # All enrichment must complete before PHASE 3 (WRITE).
    # =========================================================================

    _enrich_market_data_fields(csv_rows, market_data_by_ticker)
    _enrich_applicability_flags(csv_rows)

    # --- Spec 050/051: Selector + Ranker scoring (mode-dispatched) ---
    _eligible_for_selector = [r for r in csv_rows if _is_eligible(r)]
    if _eligible_for_selector:
        _sel_results = compute_selector_scores(_eligible_for_selector, config=A4_SELECTOR_CONFIG)
        for _row, _sr in zip(_eligible_for_selector, _sel_results):
            _row["selector_score"] = _sr.selector_score
            _row["selector_rank_bucket"] = _sr.selector_rank_bucket
            _row["selector_clinical_block"] = _sr.clinical_block
            _row["selector_catalyst_block"] = _sr.catalyst_block
            _row["selector_survivability_block"] = _sr.survivability_block
            _row["selector_institutional_block"] = _sr.institutional_block
            _row["selector_market_block"] = _sr.market_structure_block

        _sel_scores = [_sr.selector_score for _sr in _sel_results]
        _sel_buckets = [_sr.selector_rank_bucket for _sr in _sel_results]

        # Pre-assign temporary actionable_rank by selector_score descending.
        # Needed by filter_cohort in score_snapshot (real rank assigned after sort).
        _sel_ranked = sorted(
            enumerate(_eligible_for_selector),
            key=lambda x: -_safe_float(x[1].get("selector_score"), default=0.0),
        )
        for _tmp_rank, (_idx, _row) in enumerate(_sel_ranked, start=1):
            _row["actionable_rank"] = _tmp_rank

        # Regime modulation (used by clinical_50 path; persisted for all modes)
        _regime_label = (regime_result or {}).get("regime", "UNKNOWN")
        _regime_mod = get_regime_modulation(_regime_label)
        for _row in csv_rows:
            _row["regime_label"] = _regime_label

        # ------------------------------------------------------------------
        # Clinical_50 ranker (production or shadow depending on ranker_mode)
        # ------------------------------------------------------------------
        from ranker_engine import DEFAULT_RANKER_CONFIG, RankerConfig

        _ranker_cfg = DEFAULT_RANKER_CONFIG
        if _regime_mod.ranker_max_adj_mult != 1.0:
            _ranker_cfg = RankerConfig(
                max_adjustment_pct=DEFAULT_RANKER_CONFIG.max_adjustment_pct * _regime_mod.ranker_max_adj_mult
            )
        _rnk_results = compute_ranker_adjustments(_eligible_for_selector, _sel_scores, _sel_buckets, config=_ranker_cfg)
        _n_ranker_active = sum(1 for _rr in _rnk_results if _rr.ranker_active)

        # ------------------------------------------------------------------
        # Pairwise v2 ranker (production or shadow depending on ranker_mode).
        # Production artifact = capped Family C live-pilot vector
        # (model_variant="deployed_live_pilot"; coinvest_score_z weight capped
        # vs trained minimal_v2; financial_score weight unchanged). The artifact's
        # `provenance` block is authoritative for deployed weights.
        # ------------------------------------------------------------------
        _rv2_model_path = Path("production_data/ranker_v2_model.json")
        _rv2_scored = 0
        _rv2_by_ticker: Dict[str, Dict[str, Any]] = {}
        _rv2_config_id = "?"
        _rv2_model_variant = "?"
        _rv2_deployment_delta = ""
        _rv2_ok = False
        if _rv2_model_path.exists():
            try:
                _rv2_artifact = json.loads(_rv2_model_path.read_text(encoding="utf-8"))
                _rv2_model = model_from_dict(_rv2_artifact["model"])
                _rv2_config_id = _rv2_artifact.get("config_id", "?")
                _rv2_provenance = _rv2_artifact.get("provenance", {}) or {}
                _rv2_model_variant = _rv2_provenance.get("model_variant", "unspecified")
                _rv2_deployment_delta = _rv2_provenance.get("deployment_delta", "")
                _rv2_results = score_snapshot(csv_rows, _rv2_model, PRODUCTION_RANKER_V2_CONFIG)
                _rv2_by_ticker = {r["ticker"]: r for r in _rv2_results}
                _rv2_scored = sum(1 for r in _rv2_results if r["ranker_v2_score"] is not None)
                _rv2_ok = True
            except Exception as _rv2_err:
                logger.warning(f"  Ranker v2 scoring failed: {_rv2_err}")
        else:
            if ranker_mode == "pairwise_minimal":
                raise FileNotFoundError(
                    f"ranker_mode=pairwise_minimal requires {_rv2_model_path} but file not found. "
                    "Restore the model artifact, or pass --ranker-mode=clinical_50 for the legacy fallback."
                )

        # ------------------------------------------------------------------
        # Mode dispatch: wire production ranker into final_score
        # ------------------------------------------------------------------
        if ranker_mode == "pairwise_minimal" and _rv2_ok:
            # --- PRODUCTION: pairwise v2 ---
            # final_score = ranker_v2_score for cohort members;
            # non-cohort eligible get selector_score scaled down so they sort below cohort.
            for _row in _eligible_for_selector:
                _tk = _row.get("ticker", "")
                _rv2 = _rv2_by_ticker.get(_tk)
                if _rv2 and _rv2["ranker_v2_score"] is not None:
                    _row["ranker_active"] = "1"
                    _row["ranker_adjustment"] = 0.0
                    _row["final_score"] = _rv2["ranker_v2_score"]
                    _row["ranker_v2_score"] = _rv2["ranker_v2_score"]
                    _row["ranker_v2_rank"] = _rv2["ranker_v2_rank"]
                    # Block scores not applicable for pairwise mode
                    _row["ranker_options_block"] = ""
                    _row["ranker_inst_block"] = ""
                    _row["ranker_aact_block"] = ""
                else:
                    # Non-cohort eligible: use selector_score * small factor
                    # so they sort below all cohort members but maintain
                    # relative ordering among themselves.
                    _row["ranker_active"] = "0"
                    _row["ranker_adjustment"] = 0.0
                    _sel = _safe_float(_row.get("selector_score"), default=0.0)
                    _row["final_score"] = _sel * 0.0001
                    _row["ranker_v2_score"] = ""
                    _row["ranker_v2_rank"] = ""
                    _row["ranker_options_block"] = ""
                    _row["ranker_inst_block"] = ""
                    _row["ranker_aact_block"] = ""

            # Shadow: clinical_50 results (for forward comparison logging)
            _shadow_clinical50 = {}
            for _row, _rr in zip(_eligible_for_selector, _rnk_results):
                _tk = _row.get("ticker", "")
                _shadow_clinical50[_tk] = {
                    "ranker_active": _rr.ranker_active,
                    "final_score": _rr.final_score,
                }

            logger.info(
                f"  Selector: {len(_sel_results)} scored | "
                f"Ranker [PROD]: pairwise_minimal, {_rv2_scored} scored "
                f"(model {_rv2_config_id}, variant={_rv2_model_variant}) | "
                f"Ranker [SHADOW]: clinical_50, {_n_ranker_active} active (regime={_regime_label})"
            )
            if _rv2_deployment_delta:
                logger.info(f"  Ranker [PROD] deployment_delta: {_rv2_deployment_delta}")

        else:
            # --- PRODUCTION: clinical_50 (legacy / fallback) ---
            for _row, _rr in zip(_eligible_for_selector, _rnk_results):
                _row["ranker_active"] = "1" if _rr.ranker_active else "0"
                _row["ranker_adjustment"] = _rr.ranker_adjustment
                _row["final_score"] = _rr.final_score
                _row["ranker_options_block"] = _rr.options_block
                _row["ranker_inst_block"] = _rr.inst_block
                _row["ranker_aact_block"] = _rr.aact_block

            # Populate ranker_v2 shadow columns
            for _row in csv_rows:
                _tk = _row.get("ticker", "")
                _rv2 = _rv2_by_ticker.get(_tk)
                if _rv2 and _rv2["ranker_v2_score"] is not None:
                    _row["ranker_v2_score"] = _rv2["ranker_v2_score"]
                    _row["ranker_v2_rank"] = _rv2["ranker_v2_rank"]
                else:
                    _row["ranker_v2_score"] = ""
                    _row["ranker_v2_rank"] = ""

            _shadow_clinical50 = {}  # no shadow needed when clinical_50 IS production

            if ranker_mode == "pairwise_minimal" and not _rv2_ok:
                logger.warning("  Ranker mode pairwise_minimal requested but model failed; fell back to clinical_50")
            logger.info(
                f"  Selector: {len(_sel_results)} scored | "
                f"Ranker [PROD]: clinical_50, {_n_ranker_active} active (regime={_regime_label})"
                + (f" | Ranker [SHADOW]: pairwise_minimal, {_rv2_scored} scored" if _rv2_ok else "")
            )

        # ------------------------------------------------------------------
        # Shadow comparison: compute overlap and log to snapshot dir
        # ------------------------------------------------------------------
        if _rv2_ok and _eligible_for_selector:
            # Build top-30 lists for production and shadow
            _prod_top30_tickers = sorted(
                _eligible_for_selector,
                key=lambda r: -_safe_float(r.get("final_score"), default=0.0),
            )[:30]
            _prod_top30_set = {r.get("ticker") for r in _prod_top30_tickers}

            if ranker_mode == "pairwise_minimal":
                # Shadow is clinical_50: build its top-30 from shadow results
                _shadow_scored = []
                for _row in _eligible_for_selector:
                    _tk = _row.get("ticker", "")
                    _sh = _shadow_clinical50.get(_tk)
                    _shadow_scored.append((_tk, _sh["final_score"] if _sh else 0.0))
                _shadow_scored.sort(key=lambda x: -x[1])
                _shadow_top30_set = {t for t, _ in _shadow_scored[:30]}
                _shadow_ranker_name = "clinical_50"
            else:
                # Shadow is pairwise: build its top-30 from v2 scores
                _shadow_scored = []
                for _row in _eligible_for_selector:
                    _tk = _row.get("ticker", "")
                    _rv2 = _rv2_by_ticker.get(_tk)
                    _sc = _rv2["ranker_v2_score"] if (_rv2 and _rv2["ranker_v2_score"] is not None) else 0.0
                    _shadow_scored.append((_tk, _sc))
                _shadow_scored.sort(key=lambda x: -x[1])
                _shadow_top30_set = {t for t, _ in _shadow_scored[:30]}
                _shadow_ranker_name = "pairwise_minimal"

            _overlap = _prod_top30_set & _shadow_top30_set
            _prod_only = sorted(_prod_top30_set - _shadow_top30_set)
            _shadow_only = sorted(_shadow_top30_set - _prod_top30_set)

            _shadow_comparison = {
                "snapshot_date": as_of_date,
                "production_ranker": ranker_mode,
                "shadow_ranker": _shadow_ranker_name,
                "production_top30": sorted(_prod_top30_set),
                "shadow_top30": sorted(_shadow_top30_set),
                "overlap_count": len(_overlap),
                "overlap_pct": round(len(_overlap) / 30 * 100, 1),
                "production_only": _prod_only,
                "shadow_only": _shadow_only,
                "n_swaps": len(_prod_only),
            }

            # --- 2-feature ranker shadow (scoring audit candidate) ---
            # Skip if the 2feat model is identical to production (avoids
            # wasting CPU on a comparison that always yields 30/30 overlap).
            _shadow_comparison["ranker_shadow_2feat_status"] = "NOT_RUN"
            try:
                _2feat_model_path = Path("production_data/ranker_v2_model_2feat.json")
                if _2feat_model_path.exists():
                    _2feat_raw = _2feat_model_path.read_text()
                    _prod_raw = _rv2_model_path.read_text()
                    if _2feat_raw == _prod_raw:
                        _shadow_comparison["ranker_shadow_2feat_status"] = "IDENTICAL_TO_PROD"
                        logger.debug("  2feat shadow skipped: model identical to production")
                    else:
                        _2feat_model = json.loads(_2feat_raw)
                        _2f_feats = _2feat_model["model"]["feature_names"]
                        _2f_weights = _2feat_model["model"]["weights"]
                        _2f_bias = _2feat_model["model"]["bias"]

                        # Z-score within top-60 (same as production ranker)
                        _2f_top60 = sorted(
                            _eligible_for_selector,
                            key=lambda r: -_safe_float(r.get("selector_score"), default=0.0),
                        )[:60]
                        _2f_means = {}
                        _2f_stds = {}
                        for _f in _2f_feats:
                            _vals = [_safe_float(r.get(_f), default=0.0) for r in _2f_top60]
                            _2f_means[_f] = sum(_vals) / max(len(_vals), 1)
                            _var = sum((_v - _2f_means[_f]) ** 2 for _v in _vals) / max(len(_vals), 1)
                            _2f_stds[_f] = _var**0.5 if _var > 0 else 1.0

                        _2f_scored = []
                        for _row in _2f_top60:
                            _s = _2f_bias
                            for _k, _f in enumerate(_2f_feats):
                                _raw = _safe_float(_row.get(_f), default=0.0)
                                _z = (_raw - _2f_means[_f]) / _2f_stds[_f]
                                _s += _2f_weights[_k] * _z
                            _2f_scored.append((_row.get("ticker", ""), _s))
                        _2f_scored.sort(key=lambda x: -x[1])
                        _2f_top30 = {t for t, _ in _2f_scored[:30]}

                        _2f_overlap = _prod_top30_set & _2f_top30
                        _shadow_comparison["shadow_2feat_top30"] = sorted(_2f_top30)
                        _shadow_comparison["shadow_2feat_overlap"] = len(_2f_overlap)
                        _shadow_comparison["shadow_2feat_overlap_pct"] = round(len(_2f_overlap) / 30 * 100, 1)
                        _shadow_comparison["shadow_2feat_swaps"] = sorted(_prod_top30_set - _2f_top30)
                        _shadow_comparison["ranker_shadow_2feat_status"] = "OK"
            except Exception as _2f_err:
                _shadow_comparison["ranker_shadow_2feat_status"] = "FAILED"
                _shadow_comparison["ranker_shadow_2feat_error"] = f"{type(_2f_err).__name__}: {_2f_err}"
                logger.warning("2-feature shadow failed: %s", _2f_err)

            # Write comparison to snapshot dir (non-blocking)
            try:
                _comp_path = snap_path / "ranker_shadow_comparison.json"
                with open(_comp_path, "w", encoding="utf-8") as f:
                    json.dump(_shadow_comparison, f, indent=2, default=str)
                    f.write("\n")
                _2f_info = ""
                if "shadow_2feat_overlap" in _shadow_comparison:
                    _2f_info = f", 2feat: {_shadow_comparison['shadow_2feat_overlap']}/30"
                logger.info(
                    f"  Ranker shadow: {len(_overlap)}/30 overlap ({_shadow_comparison['overlap_pct']}%), "
                    f"{len(_prod_only)} swaps{_2f_info} -> ranker_shadow_comparison.json"
                )
            except OSError as _cmp_err:
                logger.warning(f"  Could not write ranker_shadow_comparison.json: {_cmp_err}")

    # Ensure ranker_v2 columns exist on all rows (including ineligible)
    for _row in csv_rows:
        if "ranker_v2_score" not in _row:
            _row["ranker_v2_score"] = ""
        if "ranker_v2_rank" not in _row:
            _row["ranker_v2_rank"] = ""

    # --- Actionable ordering: sort + assign rank + compute weights ---
    _rs = ruleset or DEFAULT_RULESET
    logger.info(
        f"  Sort anchor: {_rs.sort_anchor}, catalyst_priority_mode: {_rs.catalyst_priority_mode}, ruleset_id: {_rs.ruleset_id}"
    )
    csv_rows.sort(
        key=lambda r: compute_actionable_sort_key(
            decision_fields=r,
            archetype=r.get("archetype", ""),
            optionality=_safe_float(r.get("clinical_optionality_pct_dev")),
            composite_rank=r.get("composite_rank"),
            ticker=r.get("ticker", ""),
            catalyst_event_type=r.get("catalyst_event_type", ""),
            catalyst_source=r.get("catalyst_source", ""),
            ruleset=ruleset,
            tiebreaker_pct=(
                _safe_float(r.get("alpha_cohort_pct"))
                if ruleset and ruleset.sort_anchor == "alpha_cohort"
                else (
                    _safe_float(r.get("commercial_quality_pct"))
                    if r.get("archetype", "").startswith("commercial_")
                    else _safe_float(r.get("clinical_optionality_pct_dev"))
                )
            ),
            alpha_raw=_safe_float(r.get("alpha_cohort_raw")),
        )
    )

    # Populate sort contribution diagnostics (same inputs as sort key above)
    for r in csv_rows:
        total_adj, cmap = compute_sort_contribs(
            decision_fields=r,
            archetype=r.get("archetype", ""),
            ruleset=ruleset,
            tiebreaker_pct=(
                _safe_float(r.get("alpha_cohort_pct"))
                if ruleset and ruleset.sort_anchor == "alpha_cohort"
                else (
                    _safe_float(r.get("commercial_quality_pct"))
                    if r.get("archetype", "").startswith("commercial_")
                    else _safe_float(r.get("clinical_optionality_pct_dev"))
                )
            ),
            alpha_raw=_safe_float(r.get("alpha_cohort_raw")),
            catalyst_event_type=r.get("catalyst_event_type", ""),
            catalyst_source=r.get("catalyst_source", ""),
        )
        r["de_sort_total_adj"] = round(total_adj, 6)
        for k in SORT_CONTRIB_KEYS:
            r[f"de_sort_contrib_{k}"] = round(cmap[k], 6)

    # Assign actionable_rank: eligible rows get 1..N, ineligible get blank
    eligible_rows = [r for r in csv_rows if _is_eligible(r)]
    ineligible_rows = [r for r in csv_rows if not _is_eligible(r)]
    _n_eligible = len(eligible_rows)
    _alpha_present = sum(1 for r in eligible_rows if (_safe_float(r.get("alpha_cohort_raw")) or 0.0) != 0.0)
    for i, row in enumerate(eligible_rows, start=1):
        row["actionable_rank"] = i
    for row in ineligible_rows:
        row["actionable_rank"] = ""
        row["target_weight_pct"] = ""

    # Compute target weights for eligible rows
    compute_target_weights(eligible_rows, ruleset=ruleset)

    # --- Output contract: weights only for portfolio names ---
    # Determine top-30 portfolio set by final_score. Only these names
    # get target_weight_pct in rankings.csv. All other eligible names
    # get blank weight (they appear in the research list but not the buy list).
    # portfolio_positions.csv recomputes weights for just this subset later.
    _top30_by_score = sorted(eligible_rows, key=lambda r: -_safe_float(r.get("final_score"), default=0.0))
    _portfolio_tickers = {r.get("ticker") for r in _top30_by_score[:30]}
    for row in eligible_rows:
        if row.get("ticker") not in _portfolio_tickers:
            row["target_weight_pct"] = ""

    # Alpha modifier A/B diagnostics removed (alpha_modifier deprecated & inert).

    # --- Final CSV row ordering ---
    # In decision ranking mode (default): eligible rows first (in DE order),
    # then ineligible rows sorted deterministically by composite_rank + ticker.
    # In composite ranking mode (legacy): re-sort all rows by composite_rank.
    if ranking_mode == "decision":
        # Re-sort eligible rows for the rankings report:
        #   1. target_weight_pct descending (highest weight first)
        #   2. tier_any ascending by quality (A=0, B=1, C=2, D/blank=3)
        #   3. actionable_rank ascending (lower rank first as tiebreaker)
        _TIER_REPORT_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3}

        def _report_sort_key(r):
            # Weight: negate for descending sort (higher weight → more negative → first)
            w = _safe_float(r.get("target_weight_pct"), default=0.0)
            tier = _TIER_REPORT_ORDER.get(r.get("tier_any", ""), 3)
            rank = _safe_float(r.get("actionable_rank"), default=9999)
            return (-w, tier, rank)

        eligible_rows.sort(key=_report_sort_key)

        # Sort ineligible rows deterministically: composite_rank asc, ticker asc.
        def _inelig_sort_key(r):
            cr = r.get("composite_rank")
            try:
                cr_val = float(cr) if cr not in (None, "") else 1e18
            except (ValueError, TypeError):
                cr_val = 1e18
            return (cr_val, r.get("ticker", ""))

        ineligible_rows.sort(key=_inelig_sort_key)
        csv_rows = eligible_rows + ineligible_rows
    elif decision_mode in ("observe", "phase2"):
        # Legacy composite ordering
        has_composite = any(r.get("composite_rank") not in (None, "") for r in csv_rows)
        if has_composite:
            csv_rows.sort(
                key=lambda r: (
                    int(r["composite_rank"]) if r.get("composite_rank") not in (None, "") else 9999,
                    r.get("ticker", ""),
                )
            )
        else:
            csv_rows.sort(
                key=lambda r: (
                    int(r["actionable_rank"]) if r.get("actionable_rank") not in (None, "") else 9999,
                    r.get("ticker", ""),
                )
            )

    # --- Hard catalyst source forward-carry (Spec 011) ---
    # Once a hard catalyst source (SEC_8K_FILING, etc.) appears for a ticker,
    # carry it forward into subsequent snapshots until the event date passes.
    # Prevents intermittent 8-K fetch gaps from resetting is_hard_catalyst.
    try:
        from common.hard_catalyst_carry import forward_carry_hard_catalysts

        _carry_state_dir = snap_path.parent.parent / "state"
        _carry_state_dir.mkdir(parents=True, exist_ok=True)
        _carry_file = _carry_state_dir / "hard_catalyst_carry.json"
        _as_of = date.fromisoformat(as_of_date) if isinstance(as_of_date, str) else as_of_date
        forward_carry_hard_catalysts(csv_rows, _as_of, _carry_file)
        # Re-classify catalyst_type_tier after carry may have changed event_type
        from decision_engine import classify_catalyst_type_tier as _classify_ctt

        for _row in csv_rows:
            _evt = _row.get("catalyst_event_type", "")
            if _evt:
                _row["catalyst_type_tier"] = _classify_ctt(_evt)
                _row["catalyst_family"] = classify_catalyst_family(_evt)
    except Exception as _carry_exc:
        logger.debug("Hard catalyst forward-carry skipped: %s", _carry_exc)

    # --- Options data freshness preflight ---
    _options_freshness: Dict[str, Any] = {
        "tastytrade_status": "unknown",
        "tastytrade_as_of": "",
        "massive_status": "unknown",
        "massive_as_of": "",
        "all_fresh": False,
    }
    try:
        from common.options_diagnostics import _has_credentials as _tt_has_creds

        if _tt_has_creds():
            _options_freshness["tastytrade_status"] = "ok"
            _options_freshness["tastytrade_as_of"] = as_of_date
        else:
            _options_freshness["tastytrade_status"] = "no_credentials"
            logger.warning("[OPTIONS_WARM] Tastytrade: no credentials (TT_SECRET / TT_REFRESH not set)")
    except Exception as _tt_exc:
        _options_freshness["tastytrade_status"] = "failed"
        logger.warning("[OPTIONS_WARM] Tastytrade preflight failed: %s", _tt_exc)

    try:
        from common.options_history_massive import _get_config as _massive_config

        _mc = _massive_config()
        if _mc.get("api_key"):
            _options_freshness["massive_status"] = "ok"
            _options_freshness["massive_as_of"] = as_of_date
        else:
            _options_freshness["massive_status"] = "no_credentials"
    except Exception:
        _options_freshness["massive_status"] = "not_available"

    _options_freshness["all_fresh"] = _options_freshness["tastytrade_status"] == "ok"
    _tt_status = _options_freshness["tastytrade_status"]
    _mv_status = _options_freshness["massive_status"]
    logger.info("[OPTIONS_WARM] Tastytrade: %s, Massive: %s", _tt_status, _mv_status)

    # --- Options diagnostics (passive, tastytrade) ---
    # Restricted to catalyst-relevant names only to limit API latency.
    # Non-blocking: degrades gracefully if credentials missing or API unavailable.
    try:
        from common.options_diagnostics import fetch_options_with_fallback, select_catalyst_tickers

        _opt_tickers = select_catalyst_tickers(csv_rows)
        if _opt_tickers:
            _opt_lookup = fetch_options_with_fallback(_opt_tickers, as_of_date)
            _opt_with_data = sum(1 for v in _opt_lookup.values() if str(v.get("opt_has_data", "0")) == "1")
            # Update freshness based on actual fetch result
            if _opt_with_data > 0:
                _options_freshness["tastytrade_status"] = "ok"
                _options_freshness["tastytrade_as_of"] = as_of_date
            elif _options_freshness["tastytrade_status"] == "ok":
                _options_freshness["tastytrade_status"] = "stale"
            _options_freshness["all_fresh"] = _options_freshness["tastytrade_status"] == "ok"
            for row in csv_rows:
                tk = (row.get("ticker") or "").upper()
                row.update(_opt_lookup.get(tk, empty_diagnostics()))
            logger.info(
                "Options diagnostics: %d/%d tickers enriched, %d with data",
                len(_opt_lookup),
                len(_opt_tickers),
                _opt_with_data,
            )
            if _opt_with_data == 0:
                _sample_basis = next((v.get("opt_diagnostic_basis", "") for v in _opt_lookup.values()), "")
                if _sample_basis == "no_credentials":
                    logger.warning("Options diagnostics: TT_SECRET / TT_REFRESH not set — all opt_* columns empty")
                else:
                    logger.warning("Options diagnostics: 0 tickers with data (basis: %s)", _sample_basis)
        else:
            for row in csv_rows:
                row.update(empty_diagnostics())
    except Exception as exc:
        logger.warning("Options diagnostics failed (%s) — columns will be empty", exc)
        for row in csv_rows:
            row.update(empty_diagnostics(f"error: {exc}"))

    # --- Massive chain analytics (deep per-strike data) ---
    # Populates opt_rr_25d, opt_put_call_skew from chain, plus straddle pricing
    # and OI/volume metrics for the crowding panel. Non-blocking.
    _chain_analytics: Dict[str, Dict[str, Any]] = {}
    try:
        from common.massive_chain_analytics import warm_chain_analytics

        # Use catalyst-relevant tickers (same set as Tastytrade)
        _chain_tickers = [
            r.get("ticker", "")
            for r in csv_rows
            if r.get("opt_has_data") == "1" or r.get("catalyst_mode") == "specific_days"
        ][
            :100
        ]  # cap to avoid API overload

        if _chain_tickers and _options_freshness.get("massive_status") == "ok":
            # Build price lookup from market_data.json
            _chain_prices: Dict[str, float] = {}
            _mkt_json = snap_path.parent.parent / "production_data" / "market_data.json"
            if not _mkt_json.exists():
                _mkt_json = Path("production_data") / "market_data.json"
            if _mkt_json.exists():
                try:
                    for _rec in json.loads(_mkt_json.read_text()):
                        if isinstance(_rec, dict) and "ticker" in _rec:
                            _p = _rec.get("price") or _rec.get("regularMarketPrice") or _rec.get("previousClose") or 0
                            if _p:
                                _chain_prices[_rec["ticker"].upper()] = float(_p)
                except Exception as exc:
                    logger.warning("market_data.json price parse failed for chain analytics: %s", exc)

            _cat_days_map = {}
            for _r in csv_rows:
                _tk = _r.get("ticker", "").upper()
                try:
                    _cd = int(float(_r.get("catalyst_days", 0)))
                except (ValueError, TypeError):
                    _cd = 0
                if _tk and _cd > 0:
                    _cat_days_map[_tk] = _cd
            _chain_analytics = warm_chain_analytics(
                _chain_tickers,
                snap_path,
                as_of_date,
                _chain_prices,
                catalyst_days_by_ticker=_cat_days_map,
            )

            # Enrich csv_rows with chain analytics fields
            for row in csv_rows:
                tk = row.get("ticker", "")
                ca = _chain_analytics.get(tk, {})
                if ca.get("status") == "ok":
                    # Populate empty opt_rr_25d and opt_put_call_skew from chain
                    if not row.get("opt_rr_25d") and ca.get("rr_25d") is not None:
                        row["opt_rr_25d"] = str(round(ca["rr_25d"], 4))
                    if not row.get("opt_put_call_skew") and ca.get("put_call_skew") is not None:
                        row["opt_put_call_skew"] = str(round(ca["put_call_skew"], 4))
                    # Store chain-derived fields for downstream use
                    # Prefer catalyst-aligned straddle when available
                    row["_chain_straddle_price"] = ca.get("catalyst_straddle_price") or ca.get("straddle_price")
                    row["_chain_actual_implied_move"] = ca.get("catalyst_straddle_implied_move") or ca.get(
                        "actual_implied_move"
                    )
                    row["_chain_oi_concentration"] = ca.get("oi_concentration")
                    row["_chain_near_term_vol_share"] = ca.get("near_term_volume_share")
                    # IV crush metrics
                    if ca.get("iv_crush_breakeven_pct") is not None:
                        row["iv_crush_breakeven_pct"] = str(round(ca["iv_crush_breakeven_pct"], 4))
                    if ca.get("crush_adjusted_implied_move") is not None:
                        row["crush_adjusted_implied_move"] = str(round(ca["crush_adjusted_implied_move"], 4))

            logger.info(
                "[MASSIVE_CHAIN] Enriched %d/%d tickers with chain analytics",
                sum(1 for ca in _chain_analytics.values() if ca.get("status") == "ok"),
                len(_chain_tickers),
            )
    except Exception as _mc_exc:
        logger.debug("Massive chain analytics skipped: %s", _mc_exc)

    # --- Straddle mispricing / cheap_vol_score ---
    try:
        from common.straddle_mispricing import compute_cheap_vol_score

        _emt_path = Path(__file__).resolve().parent / "data" / "research" / "event_move_table.json"
        if _emt_path.exists():
            import json as _json_strad

            with open(_emt_path, encoding="utf-8") as _f:
                _emt_raw = _json_strad.load(_f)
            _event_move_table = _emt_raw.get("table", _emt_raw)
        else:
            _event_move_table = {}

        _n_straddle_scored = 0
        for row in csv_rows:
            atm_iv = row.get("opt_atm_iv") or row.get("_chain_actual_implied_move")
            cat_days_s = row.get("catalyst_days", "")
            cat_fam = row.get("catalyst_family", "")
            phase = row.get("lead_program_phase", row.get("archetype", ""))
            ta = row.get("therapeutic_area", "")
            straddle = row.get("_chain_straddle_price")
            price_val = row.get("_last_close") or row.get("close_price") or row.get("close")

            try:
                iv_f = float(atm_iv) if atm_iv else 0
                cd_i = int(float(cat_days_s)) if cat_days_s else 0
            except (ValueError, TypeError):
                continue
            if iv_f <= 0 or cd_i <= 0:
                continue

            cvs = compute_cheap_vol_score(
                iv_f,
                cd_i,
                cat_fam,
                phase,
                ta,
                _event_move_table,
                actual_straddle_price=float(straddle) if straddle else None,
                underlying_price=float(price_val) if price_val else None,
            )
            if cvs.get("cheap_vol_score") is not None:
                row["cheap_vol_score"] = str(round(cvs["cheap_vol_score"], 4))
                row["vol_classification"] = cvs.get("vol_classification", "")
                row["straddle_price"] = str(round(cvs["implied_move"], 4)) if cvs.get("implied_move") else ""
                _n_straddle_scored += 1
        logger.info("[STRADDLE] Scored %d/%d tickers", _n_straddle_scored, len(csv_rows))
    except Exception as _strad_exc:
        logger.warning("Straddle mispricing skipped: %s", _strad_exc)

    # --- Pre-event put/call ratio (from Massive day aggs, shadow) ---
    # Computes volume-based put/call ratio over a 10-day lookback from
    # cached day aggs.  Injected as pre_event_put_call_ratio for the
    # pcr_penalty_mode sort contribution (build_window CLINICAL only).
    _PCR_LOOKBACK = 10
    try:
        from common.options_history_massive import ingest_day_aggs

        _pcr_cache_root = Path(__file__).resolve().parent / "data" / "caches" / "massive_options" / "day_aggs"
        _pcr_aod = date.fromisoformat(as_of_date) if isinstance(as_of_date, str) else as_of_date
        # Collect day aggs for lookback window
        _pcr_put: Dict[str, int] = {}
        _pcr_call: Dict[str, int] = {}
        _pcr_loaded = 0
        for _pcr_offset in range(_PCR_LOOKBACK + 4):  # +4 for weekends
            _pcr_dt = _pcr_aod - timedelta(days=_pcr_offset)
            if _pcr_dt.weekday() >= 5:
                continue
            _pcr_file = _pcr_cache_root / str(_pcr_dt.year) / f"{_pcr_dt.month:02d}" / f"{_pcr_dt.isoformat()}.csv.gz"
            if not _pcr_file.exists():
                continue
            import gzip as _pcr_gz

            try:
                with _pcr_gz.open(_pcr_file, "rt") as _pf:
                    _pcr_reader = csv.DictReader(_pf)
                    for _pcr_rec in _pcr_reader:
                        _pcr_tk = _pcr_rec.get("ticker", "")
                        if not _pcr_tk:
                            continue
                        # Parse underlying from OCC ticker: O:UNDERLYING...
                        _pcr_stripped = _pcr_tk[2:] if _pcr_tk.startswith("O:") else _pcr_tk
                        # Find the date portion (6 digits after underlying)
                        _pcr_ul = ""
                        for _ci, _ch in enumerate(_pcr_stripped):
                            if _ch.isdigit():
                                _pcr_ul = _pcr_stripped[:_ci]
                                break
                        if not _pcr_ul:
                            continue
                        _pcr_vol = int(float(_pcr_rec.get("volume", 0) or 0))
                        if _pcr_vol <= 0:
                            continue
                        # Determine put/call from last letter before strike digits
                        _pcr_is_put = False
                        for _pch in reversed(_pcr_stripped):
                            if _pch == "P":
                                _pcr_is_put = True
                                break
                            elif _pch == "C":
                                break
                            elif _pch.isdigit():
                                continue
                            else:
                                break
                        if _pcr_is_put:
                            _pcr_put[_pcr_ul] = _pcr_put.get(_pcr_ul, 0) + _pcr_vol
                        else:
                            _pcr_call[_pcr_ul] = _pcr_call.get(_pcr_ul, 0) + _pcr_vol
                _pcr_loaded += 1
            except (OSError, ValueError, KeyError) as _pcr_file_exc:
                logger.debug("[PCR] Skipped day agg %s: %s", _pcr_file, _pcr_file_exc)
            if _pcr_loaded >= _PCR_LOOKBACK:
                break

        _n_pcr = 0
        for row in csv_rows:
            tk = row.get("ticker", "")
            p = _pcr_put.get(tk, 0)
            c = _pcr_call.get(tk, 0)
            if p + c > 0:
                row["pre_event_put_call_ratio"] = str(round(p / (p + c), 4))
                _n_pcr += 1
            else:
                row["pre_event_put_call_ratio"] = ""
        logger.info(
            "[PCR] Computed put/call ratio for %d/%d tickers (%d trading days)", _n_pcr, len(csv_rows), _pcr_loaded
        )
    except Exception as _pcr_exc:
        logger.debug("Pre-event put/call ratio skipped: %s", _pcr_exc)

    # --- Options quality composite (derived from diagnostics) ---
    for row in csv_rows:
        row.update(compute_options_quality_composite(row))

    # --- Morningstar research diagnostics (from signal engine results) ---
    _ms_scores = results.get("enhancements", {}).get("morningstar_scores", {})
    _ms_ticker_scores = _ms_scores.get("scores_by_ticker", {}) if isinstance(_ms_scores, dict) else {}
    _ms_enriched = 0
    for row in csv_rows:
        _tk = row.get("ticker", "")
        _ms_result = _ms_ticker_scores.get(_tk, {})
        if isinstance(_ms_result, dict) and _ms_result.get("status") == "SUCCESS":
            for _field in (
                "ms_volatility_3yr",
                "ms_volatility_5yr",
                "ms_star_rating",
                "ms_return_ytd",
                "ms_return_annualized_3yr",
                "ms_return_annualized_5yr",
            ):
                row[_field] = _ms_result.get(_field, "")
                if row[_field] is None:
                    row[_field] = ""
            _ms_enriched += 1
    if _ms_enriched > 0:
        logger.info("[MS] Morningstar research diagnostics enriched %d/%d tickers", _ms_enriched, len(csv_rows))

    # --- Options verdict research features (from fused 4-lens artifact) ---
    try:
        from common.options_verdict_features import enrich_csv_rows_with_verdict

        _verdict_path = Path("artifacts") / "options_verdict" / f"{as_of_date}_verdict.json"
        _verdict_data = None
        if _verdict_path.exists():
            import json as _json_mod

            _verdict_data = _json_mod.loads(_verdict_path.read_text(encoding="utf-8"))
        _n_ovf = enrich_csv_rows_with_verdict(csv_rows, _verdict_data)
        if _n_ovf > 0:
            logger.info("[OVF] Options verdict features enriched %d/%d tickers", _n_ovf, len(csv_rows))
    except Exception as _ovf_exc:
        logger.debug("Options verdict features skipped: %s", _ovf_exc)

    # --- Options Monitor v1.1 research features (Spec 040) ---
    try:
        from common.options_monitor_v11_features import compute_v11_features
        from common.options_monitor_v11_model import compute_full_verdict

        _om11_n = 0
        for row in csv_rows:
            # Extract available inputs from row
            _cat_days = row.get("catalyst_days", "")
            _cat_days_f = float(_cat_days) if _cat_days and _cat_days != "" else None
            _features = compute_v11_features(
                event_window_flag=_cat_days_f is not None and _cat_days_f <= 14,
                hard_catalyst_flag=row.get("is_hard_catalyst", "") == "1",
                catalyst_class=(row.get("catalyst_family", "other") or "other").lower(),
            )
            _full = compute_full_verdict(_features)
            # Map to ovf11_ prefix for CSV
            row["ovf11_ep"] = _full.get("om11_factor_event_premium", "")
            row["ovf11_sr"] = _full.get("om11_factor_surface_repricing", "")
            row["ovf11_sk"] = _full.get("om11_factor_skew_tail", "")
            row["ovf11_dv"] = _full.get("om11_factor_divergence", "")
            row["ovf11_quality"] = _full.get("om11_chain_quality", "")
            row["ovf11_confidence"] = _full.get("om11_confidence", "")
            row["ovf11_score"] = _full.get("om11_score_final", "")
            row["ovf11_primary_factor"] = _full.get("om11_primary_factor", "")
            row["ovf11_monitor_verdict"] = _full.get("om11_monitor_verdict", "")
            row["ovf11_trade_bias"] = _full.get("om11_trade_bias", "")
            row["ovf11_event_window_flag"] = _full.get("om11_event_window_flag", "")
            row["ovf11_catalyst_class"] = _full.get("om11_catalyst_class", "")
            _om11_n += 1
        logger.info("[OM11] Options Monitor v1.1 features computed for %d tickers", _om11_n)
    except Exception as _om11_exc:
        logger.debug("Options Monitor v1.1 features skipped: %s", _om11_exc)

    # --- Market-model disagreement overlay (shadow diagnostic) ---
    # Surfaces where model quality diverges from market-implied event magnitude.
    # Not used in ranking — diagnostics and review-queue only.
    # Pre-filter: only compute for catalyst_days <= 180 to avoid stale-window
    # artifacts where distant PCDs inflate implied_event_move mechanically.
    _POS_DIV_MAX_DAYS = 180
    try:
        from common.pos_divergence import _safe_float as _pd_float
        from common.pos_divergence import _safe_int as _pd_int
        from common.pos_divergence import compute_implied_event_move, z_score_array

        _pos_atm = [_pd_float(r.get("opt_atm_iv")) for r in csv_rows]
        _pos_days = [_pd_int(r.get("catalyst_days")) for r in csv_rows]
        _pos_model = [_pd_float(r.get("composite_score")) for r in csv_rows]

        # Mask out names beyond the decision engine's catalyst window
        _implied = [
            compute_implied_event_move(iv, d) if 0 < d <= _POS_DIV_MAX_DAYS else float("nan")
            for iv, d in zip(_pos_atm, _pos_days)
        ]
        _implied_z = z_score_array(_implied)
        _model_z = z_score_array(_pos_model)

        _n_disagreement = 0
        for i, row in enumerate(csv_rows):
            im = _implied[i]
            row["implied_event_move"] = round(im, 4) if not math.isnan(im) else ""
            mz = _model_z[i]
            iz = _implied_z[i]
            if not math.isnan(mz) and not math.isnan(iz):
                div = round(mz - iz, 4)
                row["pos_divergence"] = div
                abs_div = abs(div)
                if abs_div >= 1.5:
                    row["market_model_disagreement"] = "high"
                    _n_disagreement += 1
                elif abs_div >= 0.75:
                    row["market_model_disagreement"] = "medium"
                else:
                    row["market_model_disagreement"] = "low"
            else:
                row["pos_divergence"] = ""
                row["market_model_disagreement"] = ""

        # Phase 3 guard: demote high-disagreement when options data is stale
        if not _options_freshness["all_fresh"]:
            _demoted = 0
            for row in csv_rows:
                if row.get("market_model_disagreement") == "high":
                    row["market_model_disagreement"] = "medium"
                    _demoted += 1
            if _demoted:
                logger.warning(
                    "[POS_DIV] Options data stale — demoted %d high-disagreement to medium",
                    _demoted,
                )
            _n_disagreement = 0  # reset count after demotion

        if _n_disagreement:
            logger.info(
                "[POS_DIV] Market-model disagreement: %d high-disagreement names",
                _n_disagreement,
            )
    except Exception as _pd_exc:
        logger.debug("Market-model disagreement overlay skipped: %s", _pd_exc)
        for row in csv_rows:
            row.setdefault("implied_event_move", "")
            row.setdefault("pos_divergence", "")
            row.setdefault("market_model_disagreement", "")

    # --- Term structure validation (Agent 0 staleness / blind spot flags) ---
    # Purely diagnostic: flags catalyst date mismatches and unsurfaced events.
    try:
        from common.term_structure_validator import validate_term_structure

        # Compute cross-sectional median IV as baseline
        _all_ivs = [
            float(r.get("opt_atm_iv", 0))
            for r in csv_rows
            if r.get("opt_has_data") == "1" and r.get("opt_atm_iv", "") not in ("", "0")
        ]
        _baseline_iv = sorted(_all_ivs)[len(_all_ivs) // 2] if _all_ivs else None

        _ts_mismatch = 0
        _ts_blind = 0
        for row in csv_rows:
            _cd_raw = row.get("catalyst_days", "")
            _cd = int(float(_cd_raw)) if _cd_raw not in ("", None) else None
            _slope_raw = row.get("opt_term_slope", "")
            _slope = float(_slope_raw) if _slope_raw not in ("", None) else None
            _iv_raw = row.get("opt_atm_iv", "")
            _iv = float(_iv_raw) if _iv_raw not in ("", None) else None

            ts = validate_term_structure(
                catalyst_days=_cd,
                catalyst_mode=row.get("catalyst_mode"),
                opt_term_slope=_slope,
                opt_atm_iv=_iv,
                baseline_iv=_baseline_iv,
            )
            row["ts_flag"] = "1" if ts["flag"] else ""
            row["ts_flag_type"] = ts["flag_type"]
            row["ts_flag_reason"] = ts["reason"]
            if ts["flag_type"] == "MARKET_SEES_SOONER":
                _ts_mismatch += 1
            elif ts["flag_type"] == "BLIND_SPOT":
                _ts_blind += 1

        # Phase 3 guard: demote BLIND_SPOT when options data is stale
        if not _options_freshness["all_fresh"]:
            _bs_demoted = 0
            for row in csv_rows:
                if row.get("ts_flag_type") == "BLIND_SPOT":
                    row["ts_flag_type"] = "BLIND_SPOT_UNCONFIRMED"
                    _bs_demoted += 1
            if _bs_demoted:
                logger.warning(
                    "[TS_VALID] Options data stale — demoted %d BLIND_SPOT to BLIND_SPOT_UNCONFIRMED",
                    _bs_demoted,
                )

        if _ts_mismatch or _ts_blind:
            logger.info(
                "[TS_VALID] Term structure flags: %d mismatch, %d blind_spot",
                _ts_mismatch,
                _ts_blind,
            )
    except Exception as _ts_exc:
        logger.debug("Term structure validation skipped: %s", _ts_exc)
        for row in csv_rows:
            row.setdefault("ts_flag", "")
            row.setdefault("ts_flag_type", "")
            row.setdefault("ts_flag_reason", "")

    # --- Review queue artifact ---
    _review_queue: List[dict] = []
    try:
        from common.review_queue import (
            build_review_queue,
            compute_blind_spot_streaks,
            compute_queue_summary,
            format_review_queue_md,
            write_review_queue_csv,
        )

        # Find previous snapshot for blind-spot streak continuity
        _prev_rankings = None
        _snap_parent = snap_path.parent
        _prior_dates = sorted(
            d.name
            for d in _snap_parent.iterdir()
            if d.is_dir() and d.name[:4].isdigit() and d.name < as_of_date and "__" not in d.name
        )
        if _prior_dates:
            _prev_rankings = _snap_parent / _prior_dates[-1] / "rankings.csv"

        _state_dir = snap_path.parent.parent / "state"
        _blind_streaks = compute_blind_spot_streaks(csv_rows, _prev_rankings, _state_dir)

        # Inject blind spot days into csv_rows for downstream use
        for row in csv_rows:
            row["ts_blind_spot_days"] = _blind_streaks.get(row.get("ticker", ""), 0)

        _review_queue = build_review_queue(
            csv_rows,
            options_data_fresh=_options_freshness.get("all_fresh", False),
            blind_spot_streaks=_blind_streaks,
        )

        if _review_queue:
            write_review_queue_csv(_review_queue, snap_path / "review_queue.csv", as_of_date)
            _rq_md = format_review_queue_md(_review_queue, as_of_date, _options_freshness.get("all_fresh", False))
            with open(snap_path / "review_queue.md", "w", encoding="utf-8") as f:
                f.write(_rq_md)

        _rq_summary = compute_queue_summary(_review_queue)
        _n_noadd = _rq_summary.get("no_add_until_review", 0)
        _n_haircut = _rq_summary.get("size_haircut", 0)
        _n_review = _rq_summary.get("manual_review_required", 0)
        _n_monitor = _rq_summary.get("monitor_only", 0)
        logger.info(
            "[REVIEW_QUEUE] %d no-add, %d haircut, %d manual-review, %d monitor",
            _n_noadd,
            _n_haircut,
            _n_review,
            _n_monitor,
        )

    except Exception as _rq_exc:
        logger.debug("Review queue skipped: %s", _rq_exc)

    # --- Surface signal enrichment (Spec 020) ---
    try:
        from common.options_surface_signals import enrich_row_with_surface_signals, load_historical_iv_feature_history

        _iv_hist_path = Path(__file__).resolve().parent / "data" / "research" / "historical_iv_features.csv"
        if _iv_hist_path.exists():
            _iv_hist = load_historical_iv_feature_history(_iv_hist_path)
            _n_enriched = 0
            for row in csv_rows:
                ticker = (row.get("ticker") or "").upper()
                hist = _iv_hist.get(ticker, [])
                if hist:
                    enrich_row_with_surface_signals(row, hist, as_of_date)
                    if row.get("surface_signal_quality") in ("ok", "partial"):
                        _n_enriched += 1
            logger.info("[SURFACE] Enriched %d/%d rows with surface signals", _n_enriched, len(csv_rows))
        else:
            logger.debug("[SURFACE] historical_iv_features.csv not found — skipping")
    except Exception as _surf_exc:
        logger.debug("Surface signal enrichment skipped: %s", _surf_exc)

    # --- Options mispricing review queue ---
    try:
        from common.options_review_queue import build_options_review_queue, write_options_review_queue

        _opt_queue = build_options_review_queue(csv_rows)
        write_options_review_queue(_opt_queue, snap_path, as_of_date)
        _oqs = _opt_queue["summary"]
        logger.info(
            "[OPTIONS_QUEUE] queued=%d hard=%d soft_skipped=%d cheap=%d rich=%d disagree=%d ts=%d skew=%d surf_hi=%d surf_med=%d iv_up=%d",
            _oqs["n_total"],
            _oqs["n_hard_catalyst"],
            _oqs["n_soft_skipped"],
            _oqs["n_cheap_straddle"],
            _oqs["n_rich_straddle"],
            _oqs["n_high_disagreement"],
            _oqs["n_term_structure_flag"],
            _oqs["n_extreme_skew"],
            _oqs.get("n_surface_move_high", 0),
            _oqs.get("n_surface_move_med", 0),
            _oqs.get("n_iv_ramp_rising", 0),
        )
    except Exception as _oq_exc:
        logger.debug("Options review queue skipped: %s", _oq_exc)

    # =========================================================================
    # PHASE 2z: Final field injection (must be last enrichment before write)
    # =========================================================================
    _finalize_priced_move(csv_rows)

    # --- EES v2: score + gate enrichment (must run before CSV write) ---
    _ees_scores_for_sidecar = None
    try:
        from event_ev.expectation_error_model import enrich_csv_rows as _enrich_ees_pre

        _ees_scores_for_sidecar = _enrich_ees_pre(csv_rows, as_of_date)
    except Exception as _ees_pre_exc:
        logger.warning("EES pre-write enrichment failed: %s", _ees_pre_exc, exc_info=True)

    # --- Conditional Model: biomarker/subgroup mispricing (diagnostic only) ---
    _cond_overlays_for_sidecar = None
    try:
        from event_ev.conditional_model import enrich_csv_rows as _enrich_conditional

        _cond_overlays_for_sidecar = _enrich_conditional(csv_rows, as_of_date)
    except Exception as _cond_exc:
        logger.warning("Conditional model enrichment failed: %s", _cond_exc, exc_info=True)

    # --- Execution Capacity: slippage/sizing realism (construction only) ---
    _exec_overlays_for_sidecar = None
    try:
        from event_ev.execution_capacity import enrich_csv_rows as _enrich_execution

        _exec_overlays_for_sidecar = _enrich_execution(csv_rows, as_of_date)
    except Exception as _exec_exc:
        logger.warning("Execution capacity enrichment failed: %s", _exec_exc, exc_info=True)

    # --- EES v3: conditional misprice + expected move (diagnostic overlay) ---
    # MUST run after both EES v2 and conditional model (reads their output columns)
    _v3_overlays_for_sidecar = None
    try:
        from event_ev.ees_v3 import enrich_csv_rows as _enrich_v3

        _v3_overlays_for_sidecar = _enrich_v3(csv_rows, as_of_date)
    except Exception as _v3_exc:
        logger.warning("EES v3 enrichment failed: %s", _v3_exc, exc_info=True)

    # --- Runway Severity: financing-truth cross-layer feature (diagnostic overlay) ---
    _runway_overlays_for_sidecar = None
    try:
        from event_ev.runway_severity import enrich_csv_rows as _enrich_runway

        logger.info("Starting runway severity enrichment for %d rows", len(csv_rows))
        _runway_overlays_for_sidecar = _enrich_runway(csv_rows, as_of_date)
        # Verify enrichment was successful
        _ev_sev_populated = sum(1 for r in csv_rows if "ev_severity_score" in r and r.get("ev_severity_score"))
        _ev_sev_col_exists = sum(1 for r in csv_rows if "ev_severity_score" in r)
        logger.info(
            "Runway severity enrichment: %d/%d rows with column; %d with value",
            _ev_sev_col_exists,
            len(csv_rows),
            _ev_sev_populated,
        )

        if _ev_sev_col_exists == 0:
            logger.error(
                "CRITICAL: Runway enrichment added 0 columns. csv_rows type=%s, sample row keys=%s",
                type(csv_rows),
                list(csv_rows[0].keys())[:5] if csv_rows else "EMPTY",
            )
    except Exception as _runway_exc:
        logger.warning("Runway severity enrichment failed: %s", _runway_exc, exc_info=True)
        logger.info("Proceeding without runway severity enrichment")

    # --- Runway Transition Model: shadow-only state transition probabilities (Phase 1) ---
    # Markov Chain state transitions for runway financing risk.
    # Emits shadow columns only: transition_runway_state, transition_p_runway_worse_60d, etc.
    # No effect on ranking/selector/truth gate in v0.
    _transition_overlays_for_sidecar = None
    try:
        from event_ev.transition_model import enrich_csv_rows as _enrich_transition

        logger.info("Starting runway transition enrichment for %d rows", len(csv_rows))
        _snap_dir = Path(__file__).resolve().parent / "data" / "snapshots"
        _transition_overlays_for_sidecar = _enrich_transition(csv_rows, as_of_date, snap_dir=_snap_dir)
        _trans_col_exists = sum(1 for r in csv_rows if "transition_runway_state" in r)
        _transition_populated = sum(
            1 for r in csv_rows if "transition_runway_state" in r and r.get("transition_runway_state")
        )
        logger.info(
            "Runway transition enrichment: %d/%d rows with column; %d with value",
            _trans_col_exists,
            len(csv_rows),
            _transition_populated,
        )

        if _trans_col_exists == 0:
            logger.error("CRITICAL: Transition enrichment added 0 columns. csv_rows type=%s", type(csv_rows))
    except Exception as _transition_exc:
        logger.warning("Runway transition enrichment failed: %s", _transition_exc, exc_info=True)
        logger.info("Proceeding without runway transition enrichment")

    # --- Insider Form 4: diagnostic pass-through (scoring lane closed 2026-04-05) ---
    # Computes insider_net_buy_value_90d on demand from data/form4/raw/{ticker}.json
    # at this exact as_of_date (PIT-safe filing_date windowing). Missing raw file
    # leaves the column blank (NA); present-but-inactive yields 0.0.
    try:
        from common.insider_enrichment import enrich_rows_with_insider_net_buy_value

        _form4_raw_dir = Path(__file__).resolve().parent / "data" / "form4" / "raw"
        enrich_rows_with_insider_net_buy_value(csv_rows, as_of_date, _form4_raw_dir)
    except Exception as _insider_exc:
        logger.warning("Insider enrichment failed: %s", _insider_exc, exc_info=True)

    # --- Options features Stage 1: shadow quality/gap scores (no rank impact) ---
    try:
        from common.options_features import enrich_csv_rows as _enrich_options_features

        _opts_summary = _enrich_options_features(csv_rows, as_of_date)
        logger.info(
            "[options_features] usable=%d thin=%d missing=%d confirmed=%d gap+=%d gap-=%d",
            _opts_summary.get("n_usable", 0),
            _opts_summary.get("n_thin", 0),
            _opts_summary.get("n_missing", 0),
            _opts_summary.get("n_confirmed", 0),
            _opts_summary.get("n_gap_positive", 0),
            _opts_summary.get("n_gap_negative", 0),
        )
    except Exception as _opts_feat_exc:
        logger.warning("Options features enrichment failed: %s", _opts_feat_exc, exc_info=True)

    # =========================================================================
    # PHASE 3: WRITE — rankings.csv, checksum, manifest
    # =========================================================================

    # --- Display-only development_stage column ---
    # Final pass over csv_rows just before write. Does NOT touch any scoring,
    # selector, ranker, eligibility, or decision-engine field. See
    # _derive_development_stage() near PHASE_ORDER for the precedence rules.
    _m4_lead_phase_lookup = {
        s.get("ticker", ""): s.get("lead_phase", "") for s in (results.get("module_4_clinical", {}).get("scores") or [])
    }
    for _row in csv_rows:
        _stage, _source, _raw_phase = _derive_development_stage(
            _row, _m4_lead_phase_lookup.get(_row.get("ticker", ""), "")
        )
        _row["development_stage"] = _stage
        _row["development_stage_source"] = _source
        _row["lead_program_phase_raw"] = _raw_phase

    # --- Display-only ranker_v2 cohort_membership_streak (per audit 2026-04-26) ---
    # Walks back through dated dirs under _prior_dir to count consecutive prior
    # days with the same cohort membership state. Read-only against historical
    # rankings.csv files; never mutates scoring/selector/ranker logic.
    # _prior_dir (not snap_path.parent) — snap_path is a staging dir in production.
    _annotate_cohort_membership_streaks(csv_rows, snap_path, prior_snapshot_dir=_prior_dir)

    # --- Cohort churn diagnostic alert (per audit 2026-04-26) ---
    # Writes snap_path/cohort_churn_alert.json comparing today's v2 cohort to
    # the prior snapshot. severity=warn when churn >= 10% (boundary noise
    # threshold from the audit).
    try:
        _churn_alert = _write_cohort_churn_alert(csv_rows, snap_path, prior_snapshot_dir=_prior_dir)
        if _churn_alert and _churn_alert.get("severity") == "warn":
            logger.warning(
                f"Cohort churn WARN: {_churn_alert['churn_n']} names "
                f"({_churn_alert['churn_pct']}%) flipped vs {_churn_alert['prior_snapshot']}"
            )
    except Exception as _churn_exc:  # pragma: no cover — defensive
        logger.warning(f"Cohort churn alert generation failed: {_churn_exc}")

    # --- Safety check: ensure missing enrichment columns have default values ---
    # If enrichment functions fail silently, ensure columns exist before write.
    # This should not happen if enrichments are working, but provides a safety net.
    _missing_col_count = {}
    for row in csv_rows:
        # Ensure severity columns exist (should be set by enrich_csv_rows)
        if "ev_severity_score" not in row:
            row["ev_severity_score"] = ""  # Empty indicates enrichment failed
            _missing_col_count["ev_severity_score"] = _missing_col_count.get("ev_severity_score", 0) + 1
        if "runway_buffer_months" not in row:
            row["runway_buffer_months"] = ""
            _missing_col_count["runway_buffer_months"] = _missing_col_count.get("runway_buffer_months", 0) + 1
        if "financing_truth_gate" not in row:
            row["financing_truth_gate"] = ""
            _missing_col_count["financing_truth_gate"] = _missing_col_count.get("financing_truth_gate", 0) + 1
        if "dilution_haircut" not in row:
            row["dilution_haircut"] = ""
            _missing_col_count["dilution_haircut"] = _missing_col_count.get("dilution_haircut", 0) + 1
        if "size_multiplier" not in row:
            row["size_multiplier"] = ""
            _missing_col_count["size_multiplier"] = _missing_col_count.get("size_multiplier", 0) + 1
        if "severity_bucket" not in row:
            row["severity_bucket"] = ""
            _missing_col_count["severity_bucket"] = _missing_col_count.get("severity_bucket", 0) + 1
        if "severity_notes" not in row:
            row["severity_notes"] = ""
            _missing_col_count["severity_notes"] = _missing_col_count.get("severity_notes", 0) + 1
        # Ensure transition columns exist (should be set by enrich_csv_rows)
        if "transition_runway_state" not in row:
            row["transition_runway_state"] = ""
            _missing_col_count["transition_runway_state"] = _missing_col_count.get("transition_runway_state", 0) + 1
        if "transition_p_runway_worse_60d" not in row:
            row["transition_p_runway_worse_60d"] = ""
            _missing_col_count["transition_p_runway_worse_60d"] = (
                _missing_col_count.get("transition_p_runway_worse_60d", 0) + 1
            )
        if "transition_p_financing_90d" not in row:
            row["transition_p_financing_90d"] = ""
            _missing_col_count["transition_p_financing_90d"] = (
                _missing_col_count.get("transition_p_financing_90d", 0) + 1
            )
        if "transition_p_distress_90d" not in row:
            row["transition_p_distress_90d"] = ""
            _missing_col_count["transition_p_distress_90d"] = _missing_col_count.get("transition_p_distress_90d", 0) + 1
        # Stage 1 options shadow features (set by common.options_features.enrich_csv_rows)
        for _opt_col in (
            "options_quality_score",
            "options_quality_status",
            "event_premium_iv_pp",
            "event_premium_ratio",
            "expectation_gap_score",
            "options_shadow_verdict",
        ):
            if _opt_col not in row:
                row[_opt_col] = ""
                _missing_col_count[_opt_col] = _missing_col_count.get(_opt_col, 0) + 1

    if _missing_col_count:
        logger.warning("Missing enrichment columns added with empty defaults: %s", _missing_col_count)

    # --- Write rankings CSV ---
    csv_path = snap_path / "rankings.csv"
    try:
        import tempfile as _tf

        _fd, _tmp = _tf.mkstemp(dir=snap_path, prefix=".tmp_rankings_", suffix=".csv")
        try:
            with os.fdopen(_fd, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=SNAPSHOT_COLUMNS,
                    extrasaction="ignore",
                    lineterminator="\n",
                )
                writer.writeheader()
                for row in csv_rows:
                    writer.writerow(row)
            Path(_tmp).replace(csv_path)
        except Exception:
            try:
                os.unlink(_tmp)
            except OSError:
                pass
            raise
    except OSError as e:
        logger.warning(f"Could not write snapshot CSV: {e}")
        return None

    # --- Write SHA-256 checksum sidecar for rankings.csv ---
    try:
        from common.snapshot_integrity import write_checksum

        write_checksum(csv_path)
    except Exception as exc:
        logger.debug("Checksum sidecar write failed: %s", exc)

    # --- Write snapshot manifest sidecar ---
    try:
        from common.snapshot_manifest import write_snapshot_manifest

        write_snapshot_manifest(snap_path)
    except Exception as exc:
        logger.debug("Snapshot manifest write failed: %s", exc)

    # --- Write rank-depth Top-60 shadow sidecar (annotation only) ---
    # VALIDATION_INFRASTRUCTURE / RANK_DEPTH_SHADOW_TRACKING / NO_MODEL_CHANGE.
    # Non-blocking: a failure here must never affect the production snapshot.
    try:
        _rd_path = export_rank_depth_top60_sidecar(csv_rows, snap_path, as_of_date)
        logger.info(f"  Rank-depth Top-60 sidecar written: {_rd_path}")
    except Exception as exc:
        logger.warning(f"  rank_depth_top60.csv sidecar write failed (non-blocking): {exc}")

    # --- Write options diagnostics sidecar ---
    try:
        from common.options_snapshot import write_options_snapshot

        write_options_snapshot(snap_path, csv_rows, as_of_date)
    except Exception as exc:
        logger.warning("Options diagnostics snapshot failed (%s) — skipping sidecar", exc)

    # --- Spec 059: Options event overlay artifacts ---
    try:
        from event_ev.catalyst_risk_overlay import build_catalyst_risk_matrix, collect_escalated_alerts
        from event_ev.implied_realized_calibration import make_forward_log_entry
        from event_ev.surface_diagnostics import detect_surface_anomalies

        # Forward logging — record implied moves for pending catalysts (grows calibration table)
        _fwd_log = []
        for _row in csv_rows:
            _liq = (_row.get("opt_liquidity_state") or "").strip()
            _imp = _row.get("implied_event_move", "")
            _cdays = _row.get("catalyst_days", "")
            if _liq == "liquid" and _imp and _cdays:
                try:
                    _imp_f = float(_imp)
                    _cdays_i = int(float(_cdays))
                    if _imp_f > 0 and 0 < _cdays_i <= 90:
                        _fwd_log.append(
                            make_forward_log_entry(
                                ticker=_row.get("ticker", ""),
                                event_type=_row.get("catalyst_event_type", ""),
                                event_family=_row.get("catalyst_family", ""),
                                phase=_row.get("lead_program_phase", ""),
                                implied_event_move=_imp_f,
                                catalyst_days=_cdays_i,
                                opt_liquidity_state=_liq,
                                snapshot_date=as_of_date,
                            )
                        )
                except (ValueError, TypeError):
                    pass
        if _fwd_log:
            _fwd_path = snap_path / "options_forward_log.json"
            with open(_fwd_path, "w", encoding="utf-8") as f:
                json.dump({"as_of_date": as_of_date, "n_entries": len(_fwd_log), "entries": _fwd_log}, f, indent=2)
            logger.info("[SPEC059] Forward log: %d entries written", len(_fwd_log))

        # Surface anomaly detection
        _surface_rows = []
        for _row in csv_rows:
            _surface_rows.append(
                {
                    "ticker": _row.get("ticker", ""),
                    "opt_front_iv": _row.get("opt_front_iv", ""),
                    "opt_back_iv": _row.get("opt_back_iv", ""),
                    "opt_atm_iv": _row.get("opt_atm_iv", ""),
                    "opt_liquidity_state": _row.get("opt_liquidity_state", "absent"),
                    "catalyst_days": _row.get("catalyst_days", ""),
                    "event_family": _row.get("catalyst_family", ""),
                }
            )
        _anomalies = detect_surface_anomalies(_surface_rows)
        if _anomalies:
            with open(snap_path / "surface_anomalies.json", "w", encoding="utf-8") as f:
                json.dump(
                    {"as_of_date": as_of_date, "n_anomalies": len(_anomalies), "anomalies": _anomalies}, f, indent=2
                )
            logger.info("[SPEC059] Surface anomalies: %d flagged", len(_anomalies))

        # Catalyst risk matrix (book names near catalysts)
        _risk_rows = []
        for _row in csv_rows:
            _ar = _row.get("actionable_rank", "")
            if _ar and str(_ar).strip().isdigit() and int(_ar) <= 30:
                _risk_rows.append(
                    {
                        "ticker": _row.get("ticker", ""),
                        "catalyst_days": _row.get("catalyst_days", ""),
                        "implied_event_move": _row.get("implied_event_move", ""),
                        "opt_atm_iv": _row.get("opt_atm_iv", ""),
                        "opt_iv_regime": _row.get("opt_iv_regime", ""),
                        "opt_liquidity_state": _row.get("opt_liquidity_state", "absent"),
                        "underlying_price": _row.get("close_price", _row.get("underlying_price", "")),
                        "weight_pct": _row.get("weight_pct", "3.33"),
                        "event_family": _row.get("catalyst_family", ""),
                    }
                )
        _matrix = build_catalyst_risk_matrix(_risk_rows, max_days=30)
        _alerts = collect_escalated_alerts(_risk_rows)
        if _matrix or _alerts:
            with open(snap_path / "catalyst_risk_overlay.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "as_of_date": as_of_date,
                        "risk_matrix": _matrix,
                        "n_risk_names": len(_matrix),
                        "escalated_alerts": _alerts,
                        "n_alerts": len(_alerts),
                    },
                    f,
                    indent=2,
                )
            logger.info(
                "[SPEC059] Risk matrix: %d names, %d alerts (%d critical)",
                len(_matrix),
                len(_alerts),
                sum(1 for a in _alerts if a.get("severity") == "critical"),
            )
    except Exception as _s059_exc:
        logger.debug("Spec 059 overlay artifacts skipped: %s", _s059_exc)

    # --- Write regulatory coverage telemetry sidecar ---
    try:
        from common.regulatory_coverage_telemetry import write_telemetry as _write_reg_tel

        _write_reg_tel(snap_path, csv_rows, as_of_date, snapshots_dir=snap_path.parent)
    except Exception as exc:
        logger.warning("Regulatory coverage telemetry failed (%s) — skipping sidecar", exc)

    # --- Expectation Error Model v2 (sidecar artifacts — scores already injected pre-write) ---
    try:
        from event_ev.expectation_error_model import build_gate_diagnostics as _build_gate_diag
        from event_ev.expectation_error_model import build_gate_performance as _build_gate_perf

        _ees_scores = _ees_scores_for_sidecar
        if _ees_scores:
            _ees_dicts = [s.to_dict() for s in _ees_scores]
            _ees_dicts.sort(key=lambda d: d.get("ees_v2_score", 0), reverse=True)
            with open(snap_path / "expectation_error_overlay.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "as_of_date": as_of_date,
                        "model_version": "ees_v2.0",
                        "n_scored": len(_ees_scores),
                        "top_10": _ees_dicts[:10],
                        "bottom_10": _ees_dicts[-10:],
                    },
                    f,
                    indent=2,
                )

            # Gate diagnostics sidecar
            _diag = _build_gate_diag(_ees_scores, csv_rows, as_of_date)
            with open(snap_path / "ees_gate_diagnostics.json", "w", encoding="utf-8") as f:
                json.dump(_diag, f, indent=2)
            logger.info(
                "[EES] v2: %d scored, %d eligible (%d%%), Q-fail=%d, T-fail=%d, corr=%.3f",
                _diag["universe"]["total"],
                _diag["universe"]["eligible"],
                _diag["universe"]["pct_eligible"],
                _diag["universe"]["quality_fail"],
                _diag["universe"]["trap_fail"],
                _diag.get("quality_trap_correlation") or 0,
            )

            # Gate performance tracker (compares to prior snapshot)
            _prior_csv = None
            if _prior_dir and (_prior_dir / "rankings.csv").exists():
                import csv as _csv_mod

                with open(_prior_dir / "rankings.csv", newline="", encoding="utf-8") as _pf:
                    _prior_csv = list(_csv_mod.DictReader(_pf))
            _perf = _build_gate_perf(csv_rows, _prior_csv, as_of_date)
            if _perf:
                with open(snap_path / "ees_gate_performance.json", "w", encoding="utf-8") as f:
                    json.dump(_perf, f, indent=2)
                _e_ret = _perf["eligible"].get("mean_ret")
                _t_ret = _perf["trap_fail"].get("mean_ret")
                logger.info(
                    "[EES] Gate perf: eligible=%s%%, trap_fail=%s%% (confirm underperformance=%s)",
                    _e_ret,
                    _t_ret,
                    "YES" if _e_ret is not None and _t_ret is not None and _e_ret > _t_ret else "NO/INSUFF",
                )
            # Execution stress report (daily monitoring)
            try:
                from event_ev.portfolio_sizing import build_execution_stress_report, compute_weights

                _b6_map = {}
                _trap_map = {}
                for _row in csv_rows:
                    _t = _row.get("ticker", "")
                    _sel = _row.get("selector_score")
                    if _t and _sel is not None:
                        try:
                            _b6_map[_t] = float(_sel)
                        except (ValueError, TypeError):
                            pass
                    _trap_val = _row.get("trap_overlay_score")
                    if _t and _trap_val is not None:
                        try:
                            _trap_map[_t] = float(_trap_val)
                        except (ValueError, TypeError):
                            pass

                # Dollar volumes: avg_volume × price from market_data
                _dv_map = {}
                if market_data_by_ticker:
                    for _t, _md in market_data_by_ticker.items():
                        _avg_vol = _md.get("avg_volume") or _md.get("avg_volume_90d")
                        _px = _md.get("price")
                        if _avg_vol and _px:
                            try:
                                _dv_map[_t] = float(_avg_vol) * float(_px)
                            except (ValueError, TypeError):
                                pass

                # Build top-30 portfolio weights for stress check
                _elig_tickers = [
                    _row.get("ticker", "")
                    for _row in csv_rows
                    if _row.get("ees_eligible") is True or str(_row.get("ees_eligible", "")).lower().strip() == "true"
                ]
                _ranked = sorted(
                    [_t for _t in _elig_tickers if _t in _b6_map],
                    key=lambda _t: _b6_map[_t],
                    reverse=True,
                )[:30]

                if len(_ranked) >= 10 and _b6_map and _trap_map:
                    # Spec 057: Build CQ score map for conviction tilt (shadow, disabled by default)
                    _cq_map = {}
                    for _row in csv_rows:
                        _t = _row.get("ticker", "")
                        _cq = _row.get("clinical_quality_score")
                        if _t and _cq is not None and _cq != "":
                            try:
                                _cq_map[_t] = float(_cq)
                            except (ValueError, TypeError):
                                pass

                    _weights = compute_weights(
                        _ranked,
                        _b6_map,
                        _trap_map,
                        alpha=1.5,
                        cq_scores=_cq_map if _cq_map else None,
                        cq_tilt_strength=0.0,  # disabled — set to 0.15 to activate
                    )
                    _positions = [
                        {"ticker": _t, "weight": _weights.get(_t, 0)} for _t in _ranked if _weights.get(_t, 0) > 0
                    ]

                    # Run stress report at $5M base + 1.5x stress
                    for _sf, _sf_label in [(1.0, "base"), (1.5, "stress")]:
                        _stress = build_execution_stress_report(_positions, _dv_map, 5_000_000, stress_factor=_sf)
                        _stress_path = snap_path / f"execution_stress_{_sf_label}.json"
                        with open(_stress_path, "w", encoding="utf-8") as f:
                            json.dump(_stress, f, indent=2)

                    _base_stress = build_execution_stress_report(_positions, _dv_map, 5_000_000)
                    _tc = _base_stress.get("tail_concentration", {})
                    logger.info(
                        "[EES] Execution stress: %d positions, top3_participation=%.1f%%, " ">5%%ADV=%d, >20%%ADV=%d",
                        _base_stress.get("n_positions", 0),
                        _tc.get("top3_participation_weight_pct", 0),
                        _tc.get("n_above_5pct_adv", 0),
                        _tc.get("n_above_20pct_adv", 0),
                    )
            except Exception as _stress_exc:
                logger.warning("Execution stress report failed: %s", _stress_exc, exc_info=True)

    except Exception as _ees_exc:
        logger.debug("Expectation Error Model overlay skipped: %s", _ees_exc)

    # --- Conditional Model sidecar (diagnostic overlay) ---
    try:
        if _cond_overlays_for_sidecar:
            _cond_dicts = [o.to_dict() for o in _cond_overlays_for_sidecar]
            _cond_dicts.sort(key=lambda d: d.get("conditional_gap_score", 0), reverse=True)
            _n_selected = sum(1 for d in _cond_dicts if "selected" in d.get("conditional_bucket", ""))
            _n_enriched = sum(1 for d in _cond_dicts if "enriched" in d.get("conditional_bucket", ""))
            with open(snap_path / "conditional_model_overlay.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "as_of_date": as_of_date,
                        "model_version": "conditional_v1.0",
                        "n_scored": len(_cond_dicts),
                        "n_biomarker_selected": _n_selected,
                        "n_enriched": _n_enriched,
                        "top_10_gap": _cond_dicts[:10],
                        "bottom_10_gap": _cond_dicts[-10:],
                    },
                    f,
                    indent=2,
                )
            logger.info(
                "[ConditionalModel] Sidecar: %d scored, %d selected, %d enriched",
                len(_cond_dicts),
                _n_selected,
                _n_enriched,
            )
    except Exception as _cond_sidecar_exc:
        logger.debug("Conditional model sidecar skipped: %s", _cond_sidecar_exc)

    # --- Execution Capacity sidecar (construction overlay) ---
    try:
        if _exec_overlays_for_sidecar:
            _exec_dicts = [o.to_dict() for o in _exec_overlays_for_sidecar]
            _exec_dicts.sort(key=lambda d: d.get("execution_capacity_score", 0))
            _n_unrest = sum(1 for d in _exec_dicts if d.get("execution_bucket") == "unrestricted")
            _n_reduced = sum(1 for d in _exec_dicts if d.get("execution_bucket") == "reduced")
            _n_micro = sum(1 for d in _exec_dicts if d.get("execution_bucket") == "micro_size_only")
            _n_untrade = sum(1 for d in _exec_dicts if d.get("execution_bucket") == "untradeable")
            with open(snap_path / "execution_capacity_overlay.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "as_of_date": as_of_date,
                        "model_version": "execution_v1.0",
                        "n_scored": len(_exec_dicts),
                        "buckets": {
                            "unrestricted": _n_unrest,
                            "reduced": _n_reduced,
                            "micro_size_only": _n_micro,
                            "untradeable": _n_untrade,
                        },
                        "most_constrained_10": _exec_dicts[:10],
                        "least_constrained_10": _exec_dicts[-10:],
                    },
                    f,
                    indent=2,
                )
            logger.info(
                "[ExecutionCapacity] Sidecar: %d scored — %d unrestricted, %d reduced, %d micro, %d untradeable",
                len(_exec_dicts),
                _n_unrest,
                _n_reduced,
                _n_micro,
                _n_untrade,
            )
    except Exception as _exec_sidecar_exc:
        logger.debug("Execution capacity sidecar skipped: %s", _exec_sidecar_exc)

    # --- EES v3 sidecar (diagnostic overlay) ---
    try:
        if _v3_overlays_for_sidecar:
            _v3_dicts = [o.to_dict() for o in _v3_overlays_for_sidecar]
            _v3_dicts.sort(key=lambda d: d.get("ees_v3_score", 0), reverse=True)
            _n_v3_pass = sum(1 for d in _v3_dicts if d.get("ees_v3_gate"))
            _n_v3_misprice = sum(1 for d in _v3_dicts if d.get("misprice_available"))
            with open(snap_path / "ees_v3_overlay.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "as_of_date": as_of_date,
                        "model_version": "ees_v3.0",
                        "n_scored": len(_v3_dicts),
                        "n_with_misprice": _n_v3_misprice,
                        "n_pass_gate": _n_v3_pass,
                        "top_10": _v3_dicts[:10],
                        "bottom_10": _v3_dicts[-10:],
                    },
                    f,
                    indent=2,
                )
            logger.info(
                "[EES v3] Sidecar: %d scored, %d with misprice data, %d pass gate",
                len(_v3_dicts),
                _n_v3_misprice,
                _n_v3_pass,
            )
    except Exception as _v3_sidecar_exc:
        logger.debug("EES v3 sidecar skipped: %s", _v3_sidecar_exc)

    # --- Runway Severity sidecar (financing-truth overlay) ---
    try:
        if _runway_overlays_for_sidecar:
            _rwy_dicts = [o.to_dict() for o in _runway_overlays_for_sidecar]
            _rwy_dicts.sort(key=lambda d: d.get("runway_severity_score", 0), reverse=True)
            _n_gate_fail = sum(1 for d in _rwy_dicts if not d.get("financing_truth_gate"))
            _bucket_dist = {}
            for d in _rwy_dicts:
                b = d.get("severity_bucket", "?")
                _bucket_dist[b] = _bucket_dist.get(b, 0) + 1
            with open(snap_path / "runway_severity_overlay.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "as_of_date": as_of_date,
                        "model_version": "runway_severity_v1.0",
                        "n_scored": len(_rwy_dicts),
                        "n_gate_fail": _n_gate_fail,
                        "bucket_distribution": _bucket_dist,
                        "most_severe_10": _rwy_dicts[:10],
                        "least_severe_10": _rwy_dicts[-10:],
                    },
                    f,
                    indent=2,
                )
            logger.info(
                "[RunwaySeverity] Sidecar: %d scored, %d gate failures, buckets: %s",
                len(_rwy_dicts),
                _n_gate_fail,
                " ".join(f"{k}={v}" for k, v in sorted(_bucket_dist.items())),
            )
    except Exception as _rwy_sidecar_exc:
        logger.debug("Runway severity sidecar skipped: %s", _rwy_sidecar_exc)

    # --- Runway Transition Model sidecar (Phase 1 shadow columns) ---
    try:
        if _transition_overlays_for_sidecar:
            _trans_dicts = [o.to_dict() for o in _transition_overlays_for_sidecar]
            _trans_dicts.sort(key=lambda d: d.get("p_distress_90d", 0), reverse=True)
            _state_dist = {}
            for d in _trans_dicts:
                s = d.get("current_state", "?")
                _state_dist[s] = _state_dist.get(s, 0) + 1
            _n_pooled = sum(1 for d in _trans_dicts if d.get("is_pooled_estimate"))
            with open(snap_path / "transition_model_overlay.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "as_of_date": as_of_date,
                        "model_version": "transition_model_v0.1",
                        "phase": "Phase 1 (runway chain shadow-only)",
                        "n_scored": len(_trans_dicts),
                        "n_using_pooled_estimate": _n_pooled,
                        "state_distribution": _state_dist,
                        "highest_distress_risk_10": _trans_dicts[:10],
                        "all_results": _trans_dicts,
                    },
                    f,
                    indent=2,
                )
            logger.info(
                "[RunwayTransition] Sidecar: %d scored, %d pooled, state dist: %s",
                len(_trans_dicts),
                _n_pooled,
                " ".join(f"{k}={v}" for k, v in sorted(_state_dist.items())),
            )
    except Exception as _trans_sidecar_exc:
        logger.debug("Transition model sidecar skipped: %s", _trans_sidecar_exc)

    # --- Write eligibility summary sidecar ---
    try:
        from decision_engine_codes import canonicalize_reasons

        _elig_counts: Dict[str, int] = {}
        _elig_examples: Dict[str, List[str]] = {}
        _n_eligible = 0
        _n_ineligible = 0
        for _row in csv_rows:
            _raw = _row.get("ineligible_reasons", "")
            _codes = canonicalize_reasons(_raw)
            if _codes:
                _n_ineligible += 1
                for _c in _codes:
                    _elig_counts[_c] = _elig_counts.get(_c, 0) + 1
                    _elig_examples.setdefault(_c, []).append(_row.get("ticker", ""))
            else:
                _n_eligible += 1
        # Cap examples at 10 tickers, sorted alphabetically for determinism
        for _c in _elig_examples:
            _elig_examples[_c] = sorted(_elig_examples[_c])[:10]

        _elig_summary = {
            "schema": "eligibility_summary.v1",
            "as_of_date": as_of_date,
            "n_total": len(csv_rows),
            "n_eligible": _n_eligible,
            "n_ineligible": _n_ineligible,
            "counts_by_reason": dict(sorted(_elig_counts.items(), key=lambda x: (-x[1], x[0]))),
            "examples_by_reason": dict(
                sorted(_elig_examples.items(), key=lambda x: (-_elig_counts.get(x[0], 0), x[0]))
            ),
        }
        with open(snap_path / "eligibility_summary.json", "w", encoding="utf-8") as f:
            json.dump(_elig_summary, f, indent=2, sort_keys=False)
            f.write("\n")

        # Markdown sidecar
        _md_lines = [
            f"# Eligibility Summary — {as_of_date}",
            "",
            f"| Metric | Count |",
            f"|--------|-------|",
            f"| Total  | {len(csv_rows)} |",
            f"| Eligible | {_n_eligible} |",
            f"| Ineligible | {_n_ineligible} |",
            "",
            "## Counts by Reason",
            "",
            "| Reason | Count |",
            "|--------|-------|",
        ]
        for _c, _cnt in sorted(_elig_counts.items(), key=lambda x: (-x[1], x[0])):
            _md_lines.append(f"| {_c} | {_cnt} |")
        _md_lines.append("")
        _md_lines.append("## Examples (top 5 reasons)")
        _md_lines.append("")
        _top5 = sorted(_elig_counts.items(), key=lambda x: (-x[1], x[0]))[:5]
        for _c, _cnt in _top5:
            _tickers = ", ".join(_elig_examples.get(_c, []))
            _md_lines.append(f"**{_c}** ({_cnt}): {_tickers}")
            _md_lines.append("")
        with open(snap_path / "eligibility_summary.md", "w", encoding="utf-8") as f:
            f.write("\n".join(_md_lines))
            f.write("\n")
    except Exception as e:
        logger.warning("Could not write eligibility_summary: %s", e)

    # --- Write catalyst source mix sidecar JSON ---
    source_mix = (results.get("module_3_catalyst") or {}).get("catalyst_source_mix")
    if source_mix:
        try:
            with open(snap_path / "catalyst_source_mix.json", "w", encoding="utf-8") as f:
                json.dump(source_mix, f, indent=2, default=str)
                f.write("\n")
        except OSError as e:
            logger.warning(f"Could not write catalyst_source_mix.json: {e}")

    # --- Write catalyst shadow metrics sidecar ---
    try:
        shadow = _compute_shadow_metrics(
            csv_rows=csv_rows,
            as_of_date=as_of_date,
            snapshot_dir=_prior_dir,
            source_mix=source_mix,
            trial_records=trial_records,
        )
        with open(snap_path / "catalyst_shadow_metrics.json", "w", encoding="utf-8") as f:
            json.dump(shadow, f, indent=2, default=str)
            f.write("\n")
        # Log extraction diagnostics when future coverage alert fires
        if shadow.get("sec8k_future_alert"):
            try:
                from wake_robin_data_pipeline.collectors.sec_8k_catalyst_collector import get_extraction_diagnostics

                diag = get_extraction_diagnostics()
                logger.warning(
                    "8-K future coverage alert: %s | Stages: %s",
                    shadow["sec8k_future_alert"],
                    diag["counters"],
                )
                for ex in diag["dropped_examples"][:5]:
                    logger.warning(
                        "  Dropped: %s %s → %s | '%s'",
                        ex["ticker"],
                        ex["event_type"],
                        ex["drop_reason"],
                        ex["snippet"][:80],
                    )
            except ImportError:
                pass
    except Exception as e:
        logger.warning("Could not write catalyst_shadow_metrics.json: %s", e)

    # --- Write cache health sentinel sidecar ---
    try:
        from cache_health import _worst, compute_cache_health, load_prior_cache_health

        _by_src = (source_mix or {}).get("by_source", {})
        # Use pre-dedup counts for cache health: fuzzy dedup absorbs 8-K events
        # into calendar events, zeroing out their source tag in post-dedup counts.
        _pre_dedup_src = (source_mix or {}).get("pre_dedup_by_source", _by_src)
        _sec8k_count = _pre_dedup_src.get("SEC_8K_FILING", 0)
        _ctgov_count = _by_src.get("CTGOV_CALENDAR", 0) + _by_src.get("CTGOV", 0)
        _ema_agenda_count = sum(v for k, v in _by_src.items() if k.startswith("EMA_") and k.endswith("_AGENDA"))
        _ema_outcomes_count = _by_src.get("EMA_MEETING_HIGHLIGHTS", 0)
        _prior_health = load_prior_cache_health(_prior_dir, as_of_date)
        _prior_sec8k = (_prior_health or {}).get("sec8k", {}).get("count")
        _prior_ctgov = (_prior_health or {}).get("ctgov", {}).get("count")
        _prior_ema_agenda = (_prior_health or {}).get("ema_agenda", {}).get("count")
        _prior_ema_outcomes = (_prior_health or {}).get("ema_outcomes", {}).get("count")
        _cache_health = compute_cache_health(
            sec8k_count=_sec8k_count,
            ctgov_count=_ctgov_count,
            prior_sec8k_count=_prior_sec8k,
            prior_ctgov_count=_prior_ctgov,
            as_of_date=as_of_date,
            ema_agenda_count=_ema_agenda_count if _ema_agenda_count else None,
            ema_outcomes_count=_ema_outcomes_count if _ema_outcomes_count else None,
            prior_ema_agenda_count=_prior_ema_agenda,
            prior_ema_outcomes_count=_prior_ema_outcomes,
        )

        # Load cache refresh sidecar and merge into cache_health
        _refresh_path = Path(__file__).parent / "cache" / f"cache_refresh_{as_of_date}.json"
        _cr = load_cache_refresh_sidecar(_refresh_path)
        _cache_health["cache_refresh_sidecar"] = _cr["sidecar"]
        _cache_health["cache_refresh_had_rejections"] = _cr["had_rejections"]
        _cache_health["cache_refresh_rejected_sources"] = _cr["rejected_sources"]
        _cache_health["cache_refresh_summary"] = _cr["summary"] or None
        if _cr["had_rejections"]:
            _cache_health["overall_status"] = _worst(_cache_health["overall_status"], "bad")
            _cache_health["degraded_run"] = True

        with open(snap_path / "cache_health.json", "w", encoding="utf-8") as f:
            json.dump(_cache_health, f, indent=2)
            f.write("\n")
        if _cache_health["degraded_run"]:
            _parts = [
                f"sec8k={_cache_health['sec8k']['status']}",
                f"ctgov={_cache_health['ctgov']['status']}",
            ]
            if "ema_agenda" in _cache_health:
                _parts.append(f"ema_agenda={_cache_health['ema_agenda']['status']}")
            if "ema_outcomes" in _cache_health:
                _parts.append(f"ema_outcomes={_cache_health['ema_outcomes']['status']}")
            logger.warning(
                "[CACHE] Cache health: %s (%s)",
                _cache_health["overall_status"],
                " ".join(_parts),
            )
        if _cr["had_rejections"]:
            logger.warning(
                "[CACHE REFRESH REJECTED] %s (see %s)",
                _cr["banner"],
                _refresh_path.name,
            )
    except Exception as e:
        logger.warning("Could not write cache_health.json: %s", e)
        _cache_health = {}
        _cr = {"sidecar": None, "had_rejections": False, "rejected_sources": [], "summary": {}}

    _cache_degraded = _cache_health.get("degraded_run", False)
    _cache_health_status = _cache_health.get("overall_status", "ok")

    # --- Write institutional summary sidecar (phase2 only) ---
    # inst_summary and inst_delta already computed above (before sort)
    if decision_mode == "phase2" and inst_summary:
        try:
            with open(snap_path / "institutional_summary.json", "w", encoding="utf-8") as f:
                json.dump(inst_summary, f, indent=2, sort_keys=False)
                f.write("\n")
            logger.info(
                "[INST] Institutional summary: %d/%d tickers",
                inst_summary["tickers_with_signal"],
                inst_summary["tickers_in_universe"],
            )
        except Exception as e:
            logger.warning("Could not write institutional_summary.json: %s", e)
        if inst_delta:
            try:
                with open(snap_path / "institutional_summary_delta.json", "w", encoding="utf-8") as f:
                    json.dump(inst_delta, f, indent=2, sort_keys=False)
                    f.write("\n")
                logger.info(
                    "[INST] Delta: %s → %s, %d common tickers",
                    inst_delta.get("prior_date", "?"),
                    inst_delta.get("as_of_date", "?"),
                    inst_delta.get("tickers_common", 0),
                )
            except Exception as e:
                logger.warning("Could not write institutional_summary_delta.json: %s", e)

    # --- Write decision ruleset sidecar JSON for reproducibility ---
    rs = ruleset or DEFAULT_RULESET
    try:
        rs.to_json(str(snap_path / "decision_ruleset.json"))
    except OSError as e:
        logger.warning(f"Could not write decision_ruleset.json: {e}")

    # --- Eligibility debug diagnostics (phase2 only) ---
    if decision_mode == "phase2":
        try:
            from decision_engine import compute_gate_margins

            _elig_rs = ruleset or DEFAULT_RULESET
            elig_debug = []
            for row in csv_rows:
                _t = row.get("ticker", "")
                _rec = rec_by_ticker.get(_t)
                if _rec is None:
                    continue
                _df = _rec.get("defensive_features") or {}
                _gm = compute_gate_margins(_rec, ruleset=_elig_rs)
                elig_debug.append(
                    {
                        "ticker": _t,
                        "dd_abs": _df.get("drawdown"),
                        "dd_rel": _df.get("drawdown_rel_xbi"),
                        "dd_xbi": _df.get("drawdown_xbi"),
                        "abs_breach": (
                            _df.get("drawdown") is not None and _df.get("drawdown") < _elig_rs.drawdown_gate
                        ),
                        "rel_breach": (
                            _df.get("drawdown_rel_xbi") is not None
                            and _df.get("drawdown_rel_xbi") < _elig_rs.drawdown_rel_xbi_gate
                        ),
                        "gate_mode": _elig_rs.drawdown_gate_mode,
                        "require_both": _elig_rs.drawdown_gate_require_both,
                        "rescue_enabled": _elig_rs.enable_dd_rel_margin_rescue,
                        "rescue_threshold": _elig_rs.dd_rel_margin_rescue_threshold,
                        "dd_combined_passed": next(
                            (g["passed"] for g in _gm["gates"] if g["gate"] == "deep_drawdown"),
                            None,
                        ),
                        "dd_abs_margin": _gm.get("dd_abs_margin"),
                        "dd_rel_margin": _gm.get("dd_rel_margin"),
                        "rescued_by_rel": _gm.get("rescued_by_rel"),
                        "final_eligible": row.get("eligible"),
                        "ineligible_reasons": row.get("ineligible_reasons", ""),
                    }
                )
            with open(snap_path / "eligibility_debug.json", "w", encoding="utf-8") as f:
                json.dump(elig_debug, f, indent=2, default=str)
                f.write("\n")
            logger.info(f"Eligibility debug: {len(elig_debug)} tickers -> eligibility_debug.json")
        except Exception as e:
            logger.warning(f"Could not write eligibility_debug.json: {e}")

    # --- Phase-2 decision portfolio (full DE-ranked research list) ---
    if decision_mode == "phase2":
        rs = ruleset or DEFAULT_RULESET

        # Use the same ordered list as rankings.csv (eligible first, then
        # ineligible) — no Top-N truncation, no weight computation.
        portfolio_rows = list(csv_rows)

        # Write decision_portfolio.csv
        portfolio_csv_path = snap_path / "decision_portfolio.csv"
        try:
            with open(portfolio_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=PHASE2_PORTFOLIO_COLUMNS,
                    extrasaction="ignore",
                    lineterminator="\n",
                )
                writer.writeheader()
                for row in portfolio_rows:
                    writer.writerow(row)
            logger.info(f"Phase-2 research list: {len(portfolio_rows)} securities -> " f"{portfolio_csv_path.name}")
        except OSError as e:
            logger.warning(f"Could not write decision_portfolio.csv: {e}")

        # Write decision_portfolio.json (structured payload)
        portfolio_json_path = snap_path / "decision_portfolio.json"
        try:
            n_eligible = sum(1 for r in portfolio_rows if _is_eligible(r))
            portfolio_payload = {
                "schema_version": SCHEMA_DECISION_PORTFOLIO,
                "generator_version": VERSION,
                "config_fingerprint": None,  # populated in metadata.json
                "snapshot_date": as_of_date,
                "decision_engine_version": DE_VERSION,
                "ruleset_id": rs.ruleset_id,
                "ruleset_path": str(
                    PHASE2_DEFAULT_RULESET_PATH if PHASE2_DEFAULT_RULESET_PATH.exists() else "built-in"
                ),
                "n_securities": len(portfolio_rows),
                "n_eligible": n_eligible,
                "positions": [
                    {
                        "ticker": r.get("ticker", ""),
                        "company_name": r.get("company_name", ""),
                        "actionable_rank": r.get("actionable_rank", ""),
                        "tier_any": r.get("tier_any", ""),
                        "tier_dev": r.get("tier_dev", ""),
                        "tier_commercial": r.get("tier_commercial", ""),
                        "size_band": r.get("size_band", ""),
                        "tier_reason": r.get("tier_reason", ""),
                        "catalyst_mode": r.get("catalyst_mode", ""),
                        "catalyst_days": r.get("catalyst_days", ""),
                        "mom_state": r.get("mom_state", ""),
                        "risk_flags": r.get("risk_flags", ""),
                        "archetype": r.get("archetype", ""),
                        "development_stage": r.get("development_stage", ""),
                        "development_stage_source": r.get("development_stage_source", ""),
                        "eligible": r.get("eligible", ""),
                        "composite_rank": r.get("composite_rank", ""),
                        "composite_score": r.get("composite_score", ""),
                        "next_earnings_date": r.get("next_earnings_date", ""),
                    }
                    for r in portfolio_rows
                ],
            }
            with open(portfolio_json_path, "w", encoding="utf-8") as f:
                json.dump(portfolio_payload, f, indent=2, default=str)
                f.write("\n")
            logger.info(f"Phase-2 research list JSON -> {portfolio_json_path.name}")
        except OSError as e:
            logger.warning(f"Could not write decision_portfolio.json: {e}")

        # --- Portfolio positions (top-K weighted subset for operations/health/delta) ---
        # Spec 050: top-30 by final_score (selector + ranker), no tier filter.
        # eligible_rows is already sorted by the report sort key (weight→tier→rank),
        # so re-sort by final_score descending to get the correct top-K.
        top_k = POSITIONS_TOP_K
        _pos_by_score = sorted(eligible_rows, key=lambda r: -_safe_float(r.get("final_score"), default=0.0))
        position_rows = [dict(r) for r in _pos_by_score[:top_k]]

        # Recompute weights for this subset (normalizes to 100%)
        if decision_mode == "phase2":
            # Phase 2: equal weights (paper-only tracking)
            equal_weight = round(100.0 / len(position_rows), 2) if position_rows else 0.0
            for row in position_rows:
                row["target_weight_pct"] = equal_weight
        else:
            compute_target_weights(position_rows, ruleset=ruleset)

        # Write portfolio_positions.csv
        positions_csv_path = snap_path / "portfolio_positions.csv"
        try:
            with open(positions_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=PORTFOLIO_POSITIONS_COLUMNS,
                    extrasaction="ignore",
                    lineterminator="\n",
                )
                writer.writeheader()
                for row in position_rows:
                    writer.writerow(row)
            logger.info(f"Phase-2 positions: {len(position_rows)} securities -> " f"{positions_csv_path.name}")
        except OSError as e:
            logger.warning(f"Could not write portfolio_positions.csv: {e}")

        # Write portfolio_positions.json
        positions_json_path = snap_path / "portfolio_positions.json"
        try:
            total_wt = round(
                sum(
                    float(r.get("target_weight_pct", 0))
                    for r in position_rows
                    if r.get("target_weight_pct") not in (None, "")
                ),
                2,
            )
            # Compute Path C drawdown vs XBI for governance monitoring (locked Day 1 portfolio only)
            portfolio_holdings = {
                r.get("ticker", ""): float(r.get("target_weight_pct", 0))
                for r in position_rows
                if r.get("ticker", "") and r.get("target_weight_pct", "") not in (None, "")
            }
            drawdown_result = compute_drawdown_vs_xbi(
                portfolio_holdings,
                snapshot_date=as_of_date,
                baseline_date="2026-05-29",
                price_history_path=Path("production_data/price_history_split_adj.csv"),
            )

            positions_payload = {
                "schema_version": SCHEMA_PORTFOLIO_POSITIONS,
                "generator_version": VERSION,
                "config_fingerprint": None,  # populated in metadata.json
                "snapshot_date": as_of_date,
                "decision_engine_version": DE_VERSION,
                "ruleset_id": rs.ruleset_id,
                "top_k": top_k,
                "tier_filter": "all",  # Spec 050: no tier filter, top-K by final_score
                "n_positions": len(position_rows),
                "total_weight_pct": total_wt,
                "drawdown_vs_xbi_pp": drawdown_result["pp"],
                "drawdown_vs_xbi_status": drawdown_result["status"],
                "drawdown_vs_xbi_baseline_date": drawdown_result["baseline_date"],
                "drawdown_vs_xbi_latest_date": drawdown_result["latest_date"],
                "positions": [
                    {
                        "ticker": r.get("ticker", ""),
                        "company_name": r.get("company_name", ""),
                        "actionable_rank": r.get("actionable_rank", ""),
                        "target_weight_pct": r.get("target_weight_pct", ""),
                        "tier_any": r.get("tier_any", ""),
                        "tier_dev": r.get("tier_dev", ""),
                        "tier_commercial": r.get("tier_commercial", ""),
                        "size_band": r.get("size_band", ""),
                        "catalyst_days": r.get("catalyst_days", ""),
                        "catalyst_mode": r.get("catalyst_mode", ""),
                        "mom_state": r.get("mom_state", ""),
                        "archetype": r.get("archetype", ""),
                    }
                    for r in position_rows
                ],
            }
            with open(positions_json_path, "w", encoding="utf-8") as f:
                json.dump(positions_payload, f, indent=2, default=str)
                f.write("\n")
            logger.info(f"Phase-2 positions JSON -> {positions_json_path.name}")
        except OSError as e:
            logger.warning(f"Could not write portfolio_positions.json: {e}")

        # --- P0 sidecar diff for EES v3 promotion battery (Spec 064) ---
        # Observational only: captures what EES v3 would filter from the
        # 2-feat ranker baseline, per-snapshot, plus coverage health for
        # priced_move_pct / short_interest_pct / gap-computable rows.
        # Does NOT alter selection, ranking, or execution. See
        # specs/changes/spec_064_ees_v3_promotion_battery.md for thresholds.
        try:
            if _v3_overlays_for_sidecar:
                from event_ev.ees_v3 import DEFAULT_W_EXPECTED as _V3_W_EXPECTED
                from event_ev.ees_v3 import DEFAULT_W_MISPRICE as _V3_W_MISPRICE

                _EPS_COV = 1e-9
                _v3_by_ticker = {o.ticker: o for o in _v3_overlays_for_sidecar}
                _row_by_ticker = {r.get("ticker", ""): r for r in csv_rows}

                # Coverage health — evaluated over all scored rows so operators
                # see data-plane stability even when eligibility shrinks the
                # universe. These thresholds gate downstream WS4 / P2 work.
                _n_total = len(csv_rows)
                _n_pm = _n_si = _n_gap = _n_full = 0
                for _r in csv_rows:
                    _pm = _safe_float(_r.get("priced_move_pct"), default=None)
                    _si = _safe_float(_r.get("short_interest_pct"), default=None)
                    if _pm is not None:
                        _n_pm += 1
                        if abs(_pm) > _EPS_COV:
                            _n_gap += 1
                    if _si is not None:
                        _n_si += 1
                    _ov = _v3_by_ticker.get(_r.get("ticker", ""))
                    if _ov is not None and _ov.misprice_available:
                        _n_full += 1

                def _pct(num: int, den: int) -> float:
                    return round(num / den, 4) if den > 0 else 0.0

                _cov = {
                    "total_scored": _n_total,
                    "pct_priced_move": _pct(_n_pm, _n_total),
                    "pct_short_interest": _pct(_n_si, _n_total),
                    "pct_gap_computable": _pct(_n_gap, _n_total),
                    "pct_full_ees_computable": _pct(_n_full, _n_total),
                }
                _hard_fail: List[str] = []
                if _cov["pct_priced_move"] < 0.60:
                    _hard_fail.append("priced_move_below_0.60")
                if _cov["pct_short_interest"] < 0.70:
                    _hard_fail.append("short_interest_below_0.70")
                if _cov["pct_gap_computable"] < 0.50:
                    _hard_fail.append("gap_computable_below_0.50")
                if _cov["pct_full_ees_computable"] < 0.40:
                    _hard_fail.append("full_ees_below_0.40")
                _cov["hard_fail_flags"] = _hard_fail

                # Baseline = top-K position rows (already written above). Keep
                # order so downstream diffs can do rank-wise comparisons.
                _baseline_tickers = [r.get("ticker", "") for r in position_rows if r.get("ticker")]
                _baseline_set = set(_baseline_tickers)

                def _driver_component(ov) -> str:
                    _cm = _V3_W_MISPRICE * ov.conditional_misprice_z
                    _em = _V3_W_EXPECTED * ov.conditional_expected_move_z
                    return "conditional_misprice" if abs(_cm) >= abs(_em) else "expected_move"

                def _v2_context(row: Dict[str, Any]) -> Dict[str, Optional[float]]:
                    # Operator context — NOT part of the v3 composite, but
                    # useful when reviewing filtered names.
                    return {
                        "trap_overlay_score": _safe_float(row.get("trap_overlay_score"), default=None),
                        "timing_decay_risk_score": _safe_float(row.get("timing_decay_risk_score"), default=None),
                        "quality_overlay_score": _safe_float(row.get("quality_overlay_score"), default=None),
                    }

                _would_filter: List[Dict[str, Any]] = []
                _would_retain_count = 0
                for _t in _baseline_tickers:
                    _ov = _v3_by_ticker.get(_t)
                    _row = _row_by_ticker.get(_t, {})
                    if _ov is None:
                        _would_retain_count += 1
                        continue
                    if not _ov.ees_v3_gate:
                        _would_filter.append(
                            {
                                "ticker": _t,
                                "ees_v3_score": round(_ov.ees_v3_score, 4),
                                "ees_v3_pctile": round(_ov.ees_v3_pctile, 1),
                                "misprice_available": _ov.misprice_available,
                                "driver_component": _driver_component(_ov),
                                "v2_context": _v2_context(_row),
                            }
                        )
                    else:
                        _would_retain_count += 1

                # would_add — not in baseline but ranks in EES v3 top 5%
                _would_add: List[Dict[str, Any]] = []
                for _r in csv_rows:
                    _t = _r.get("ticker", "")
                    if not _t or _t in _baseline_set or not _is_eligible(_r):
                        continue
                    _ov = _v3_by_ticker.get(_t)
                    if _ov is None or _ov.ees_v3_pctile < 95.0:
                        continue
                    _would_add.append(
                        {
                            "ticker": _t,
                            "ees_v3_score": round(_ov.ees_v3_score, 4),
                            "ees_v3_pctile": round(_ov.ees_v3_pctile, 1),
                            "misprice_available": _ov.misprice_available,
                            "driver_component": _driver_component(_ov),
                        }
                    )
                _would_add.sort(key=lambda d: -d["ees_v3_score"])

                _sidecar_payload = {
                    "snapshot_date": as_of_date,
                    "spec": "spec_064_ees_v3_promotion_battery",
                    "ees_version": _v3_overlays_for_sidecar[0].model_version,
                    "baseline": {
                        "selector": "A4_2feat_ranker",
                        "ruleset_id": rs.ruleset_id,
                        "n_selected": len(_baseline_tickers),
                        "selected_tickers": _baseline_tickers,
                    },
                    "ees_overlay": {
                        "n_scored": len(_v3_overlays_for_sidecar),
                        "would_filter": _would_filter,
                        "would_add": _would_add,
                        "would_retain_count": _would_retain_count,
                    },
                    "coverage": _cov,
                    "forward_outcomes": {
                        "horizon_days": 21,
                        "status": "pending",
                    },
                    "note": (
                        "Observational overlay. Does not affect ranking, "
                        "selection, or execution. See "
                        "specs/changes/spec_064_ees_v3_promotion_battery.md"
                    ),
                }
                with open(snap_path / "ees_sidecar_diff.json", "w", encoding="utf-8") as f:
                    json.dump(_sidecar_payload, f, indent=2, default=str)
                    f.write("\n")
                logger.info(
                    "[EES sidecar] filter=%d add=%d retain=%d | coverage pm=%.0f%% si=%.0f%% gap=%.0f%% full=%.0f%% | hard_fail=%s",
                    len(_would_filter),
                    len(_would_add),
                    _would_retain_count,
                    _cov["pct_priced_move"] * 100,
                    _cov["pct_short_interest"] * 100,
                    _cov["pct_gap_computable"] * 100,
                    _cov["pct_full_ees_computable"] * 100,
                    ",".join(_hard_fail) if _hard_fail else "none",
                )
        except Exception as _sidecar_exc:
            logger.debug("EES sidecar diff skipped: %s", _sidecar_exc, exc_info=True)

    # --- Catalyst coverage diagnostics ---
    dev_csv_rows = [r for r in csv_rows if r.get("archetype") == "drug_developer"]
    cat_modes = {}
    for r in dev_csv_rows:
        m = r.get("catalyst_mode", "") or "_blank"
        cat_modes[m] = cat_modes.get(m, 0) + 1
    n_dev = len(dev_csv_rows)
    n_specific = cat_modes.get("specific_days", 0)
    if n_dev:
        logger.info(f"[CATALYST] Dev coverage: {n_specific}/{n_dev} specific_days " f"({100*n_specific/n_dev:.1f}%)")
    else:
        logger.info("[CATALYST] No dev rows")
    for mode, cnt in sorted(cat_modes.items(), key=lambda x: -x[1]):
        logger.info(f"[CATALYST]   {mode}: {cnt}")
    if n_specific > 0:
        specific_rows = sorted(
            [r for r in dev_csv_rows if r.get("catalyst_mode") == "specific_days"],
            key=lambda r: float(r.get("catalyst_days") or 9999),
        )
        for r in specific_rows[:10]:
            logger.info(
                f"[CATALYST]   {r['ticker']:6s} days={str(r.get('catalyst_days','?')):>4s} "
                f"tier={r.get('tier_dev','?')}"
            )

    # --- Compute per-component score diagnostics ---
    score_columns = [
        "composite_score",
        "momentum_score",
        "catalyst_score",
        "smart_money_score",
        "valuation_score",
        "clinical_score",
        "financial_score",
        "confidence_overall",
    ]
    score_diagnostics = {}
    for col in score_columns:
        vals = []
        for rec in ranked:
            if col == "momentum_score":
                v = (rec.get("momentum_signal") or {}).get("momentum_score")
            elif col == "catalyst_score":
                v = _component_score(rec, "catalyst")
            elif col == "smart_money_score":
                v = _component_score(rec, "smart_money")
            elif col == "valuation_score":
                v = (rec.get("valuation_signal") or {}).get("valuation_score")
            elif col == "clinical_score":
                v = _component_score(rec, "clinical")
            elif col == "financial_score":
                v = _component_score(rec, "financial")
            else:
                v = rec.get(col)
            if v is not None:
                vals.append(v)
        n_total = len(vals)
        n_unique = len(set(vals))
        score_diagnostics[col] = {
            "n_total": n_total,
            "n_unique": n_unique,
            "tie_pct": round(1.0 - n_unique / n_total, 4) if n_total > 0 else None,
        }

    # Catalyst zero-signal breakdown: separate neutral-base tickers from real ties
    cat_diag = score_diagnostics.get("catalyst_score", {})
    cat_raw_vals = []
    for rec in ranked:
        for cs in rec.get("component_scores", []):
            if cs.get("name") == "catalyst" and cs.get("raw") is not None:
                cat_raw_vals.append(float(str(cs["raw"])))
    if cat_raw_vals:
        neutral_count = sum(1 for v in cat_raw_vals if abs(v - 50.0) < 0.01)
        cat_diag["zero_signal_count"] = neutral_count
        non_neutral = [v for v in cat_raw_vals if abs(v - 50.0) >= 0.01]
        nn_unique = len(set(str(round(v, 6)) for v in non_neutral))
        cat_diag["signal_n"] = len(non_neutral)
        cat_diag["signal_unique"] = nn_unique
        cat_diag["signal_tie_pct"] = round(1.0 - nn_unique / len(non_neutral), 4) if non_neutral else None

    # Clinical zero-signal breakdown: tickers with no trials
    clin_diag = score_diagnostics.get("clinical_score", {})
    clin_raw_vals = []
    for rec in ranked:
        for cs in rec.get("component_scores", []):
            if cs.get("name") == "clinical" and cs.get("raw") is not None:
                clin_raw_vals.append(float(str(cs["raw"])))
    if clin_raw_vals:
        from collections import Counter

        raw_counter = Counter(clin_raw_vals)
        floor_val, floor_count = raw_counter.most_common(1)[0]
        if floor_count >= 5:
            clin_diag["zero_signal_count"] = floor_count
            clin_diag["zero_signal_raw"] = round(floor_val, 4)
            non_floor = [v for v in clin_raw_vals if abs(v - floor_val) > 0.01]
            nf_unique = len(set(str(round(v, 6)) for v in non_floor))
            clin_diag["signal_n"] = len(non_floor)
            clin_diag["signal_unique"] = nf_unique
            clin_diag["signal_tie_pct"] = round(1.0 - nf_unique / len(non_floor), 4) if non_floor else None

    # Top offenders: components sorted by tie_pct descending (for quick triage)
    score_diagnostics["_top_tied"] = sorted(
        [k for k in score_diagnostics],
        key=lambda k: score_diagnostics[k].get("tie_pct") or 0,
        reverse=True,
    )

    # --- Ranking diagnostics: uniqueness of the actual rank-driving fields ---
    ranking_diagnostics = {}
    for field in ["composite_rank", "score_rank_pct", "score_z", "composite_score"]:
        vals = [rec.get(field) for rec in ranked if rec.get(field) is not None]
        n_total = len(vals)
        n_unique = len(set(str(v) for v in vals))
        ranking_diagnostics[field] = {
            "n_total": n_total,
            "n_unique": n_unique,
            "tie_pct": round(1.0 - n_unique / n_total, 4) if n_total > 0 else None,
        }

    # --- Write metadata JSON ---
    summary = results.get("summary", {})
    clinical_filter = summary.get("clinical_activity_filter", {})
    archetype_counts = {}
    for a in archetypes.values():
        archetype_counts[a] = archetype_counts.get(a, 0) + 1

    # --- Provenance fields ---
    _rs = ruleset or DEFAULT_RULESET
    _git = get_git_info(Path(__file__).resolve().parent)
    _ruleset_path = snap_path / "decision_ruleset.json"
    _ruleset_hash = sha256_file(str(_ruleset_path)) if _ruleset_path.exists() else None
    _config_fingerprint = _compute_config_fingerprint(
        decision_mode=decision_mode,
        ranking_mode=ranking_mode,
        tier_filter=PHASE2_DEFAULT_TIER_FILTER,
        top_k=PHASE2_DEFAULT_TOP_K,
        alpha_schema_mode=alpha_schema_mode,
        ruleset_id=_rs.ruleset_id,
        reliability_table=_reliability_table,
        ranker_mode=ranker_mode,
    )

    _phase_scores_version = phase_scores_version

    metadata = {
        "schema_version": SCHEMA_METADATA,
        "generator_version": VERSION,
        "config_fingerprint": _config_fingerprint,
        "as_of_date": as_of_date,
        "saved_at": _deterministic_timestamp(as_of_date),
        "decision_mode": decision_mode,
        "ranking_mode": ranking_mode,
        "ranker_mode": ranker_mode,
        "version": version,
        "ruleset_id": _rs.ruleset_id,
        "ruleset_file": getattr(_rs, "_source_path", None),
        "ruleset_hash": _ruleset_hash,
        "engine_version": DE_VERSION,
        "phase_scores_version": _phase_scores_version,
        "git_sha": _git.get("commit_sha"),
        "ticker_count": len(ranked),
        "source_type": "live_pipeline_v3",
        "scoring_model": "module_5_v3_ic_enhanced",
        "total_evaluated": summary.get("total_evaluated"),
        "active_universe": summary.get("active_universe"),
        "clinical_excluded": clinical_filter.get("excluded_count", 0),
        "clinical_exempted": clinical_filter.get("exempted_count", 0),
        "archetype_distribution": dict(sorted(archetype_counts.items())),
        "ranking_diagnostics": ranking_diagnostics,
        "score_diagnostics": score_diagnostics,
        "dev_optionality_stats": (
            {
                "n_dev": len(dev_rows),
                "n_dev_with_clinical": len(dev_rows),
                "min_clinical": round(min(float(r["clinical_score"]) for _, r in dev_rows), 4) if dev_rows else None,
                "max_clinical": round(max(float(r["clinical_score"]) for _, r in dev_rows), 4) if dev_rows else None,
            }
            if dev_rows
            else {"n_dev": 0}
        ),
        "input_hashes": results.get("run_metadata", {}).get("input_hashes", {}),
        "clinical_sort_telemetry": {
            "n_nonzero_clin_adj_dev": _clin_adj_dev,
            "n_nonzero_clin_adj_comm": _clin_adj_comm,
            "ruleset_id": ruleset.ruleset_id if ruleset else None,
            "enable_clinical_sort_signal": ruleset.enable_clinical_sort_signal if ruleset else False,
        },
        "far_window_telemetry": {
            "n_far_window_overrides": _n_far_window,
            "far_window_days": ruleset.far_window_days if ruleset else None,
            "far_window_decay_mult": ruleset.far_window_decay_mult if ruleset else None,
        },
        "alpha_cohort_telemetry": (
            {
                "enabled": _run_alpha_cohort if _run_alpha_cohort else False,
                "composite_engine": ruleset.composite_engine if ruleset else "legacy",
                "table_path": ruleset.alpha_cohort_table_path if ruleset else None,
                "shrink_k": ruleset.alpha_cohort_shrink_k if ruleset else None,
            }
            if _run_alpha_cohort
            else {}
        ),
        "alpha_table_telemetry": {
            "table_policy": ruleset.alpha_table_rebuild_policy if ruleset else "never",
            "table_path": str(_alpha_table_path_resolved) if _alpha_table_path_resolved else None,
            "alpha_table_source": _alpha_table_source,
            "alpha_table_marker_present": (
                _check_artifact_marker(Path(_alpha_table_path_resolved))[0] if _alpha_table_path_resolved else False
            ),
            "table_build_info": (_alpha_table or {}).get("_build_info"),
            "cells_populated": _alpha_cells_populated,
            "alpha_score_present": _alpha_present,
            "alpha_score_missing": _n_eligible - _alpha_present,
        },
        "coinvest_coverage": {
            "tickers_with_signal": sum(
                1
                for r in csv_rows
                if isinstance(r.get("sponsor_tier1_count"), (int, float)) and r["sponsor_tier1_count"] > 0
            ),
            "ticker_count": len(csv_rows),
            "coverage_pct": round(
                100
                * sum(
                    1
                    for r in csv_rows
                    if isinstance(r.get("sponsor_tier1_count"), (int, float)) and r["sponsor_tier1_count"] > 0
                )
                / max(len(csv_rows), 1),
                1,
            ),
        },
        "alpha_contract_telemetry": {
            "input": _alpha_input_diag.to_dict() if _alpha_input_diag else {"skipped": True},
            "output": _alpha_output_diag.to_dict() if _alpha_output_diag else {"skipped": True},
        },
        "inputs_manifest": {
            "mode": inputs_manifest_mode,
            "path": "inputs_manifest.json" if inputs_manifest_mode != "off" else None,
            "all_required_present": (
                results.get("run_metadata", {})
                .get("inputs_manifest", {})
                .get("validation", {})
                .get("all_required_present")
                if inputs_manifest_mode != "off"
                else None
            ),
        },
        "cache_health_overall_status": _cache_health_status,
        "cache_degraded_run": _cache_degraded,
        "suppress_churn_comparisons": _cache_degraded,
        "cache_refresh_sidecar": _cr["sidecar"],
        "cache_refresh_had_rejections": _cr["had_rejections"],
        "cache_refresh_rejected_sources": _cr["rejected_sources"],
        "cache_refresh_summary": _cr["summary"] or None,
        "pseudo_pit_version": 2 if results.get("run_metadata", {}).get("pit_mode", "off") != "off" else 1,
    }

    # --- data_sources provenance (for anti-lookahead validation) ---
    _data_sources: Dict[str, Any] = {}
    if inst_summary and inst_summary.get("cache_as_of_date"):
        _data_sources["sec_13f"] = {
            "effective_date": inst_summary["cache_as_of_date"],
            "lag_days": 45,
        }
    if ctgov_cache_date:
        _data_sources["ctgov"] = {"effective_date": ctgov_cache_date}
    # 8-K cache is always keyed by as_of_date in the production pipeline
    _data_sources["sec_8k"] = {"effective_date": as_of_date}
    metadata["data_sources"] = _data_sources

    # Regulatory coverage telemetry (all sources: M3, event ledger, manual calendar)
    _n_pdufa_events = len(_pdufa_manual)
    _n_reg_flagged = sum(1 for r in csv_rows if r.get("has_regulatory_upcoming_180d") == "1")
    _eligible_count = sum(1 for r in csv_rows if _is_eligible(r))
    _reg_type_counts: Dict[str, int] = {}
    for r in csv_rows:
        et = r.get("regulatory_event_type", "")
        if et:
            _reg_type_counts[et] = _reg_type_counts.get(et, 0) + 1
    _n_ledger_reg = 0
    _ledger_source_counts: Dict[str, int] = {}
    if _event_ledger:
        for entry in _event_ledger:
            if entry.event_type in REGULATORY_EVENT_TYPES:
                _n_ledger_reg += 1
                src = entry.source or "unknown"
                _ledger_source_counts[src] = _ledger_source_counts.get(src, 0) + 1
    # Manual calendar telemetry (with quality selection diagnostics)
    try:
        from common.regulatory_calendar import CalendarPolicy, get_calendar_telemetry, select_quality_entries

        _cal_used, _cal_diag = select_quality_entries(
            _pdufa_manual,
            as_of_date,
            n_eligible=_eligible_count,
            reliability_table=_reliability_table or None,
        )
        _cal_telemetry = get_calendar_telemetry(_cal_used, selection_diag=_cal_diag)
    except Exception:
        _cal_telemetry = {}
        _cal_diag = {}
    metadata["regulatory_coverage"] = {
        "manual_calendar_n_records": _n_pdufa_events,
        "manual_calendar_detail": _cal_telemetry,
        "event_ledger_regulatory_entries": _n_ledger_reg,
        "event_ledger_sources": _ledger_source_counts,
        "n_eligible_flagged": _n_reg_flagged,
        "regulatory_secondary_coverage_pct": round(_n_reg_flagged / max(_eligible_count, 1) * 100, 1),
        "flagged_event_types": _reg_type_counts,
        "reg_calendar_entries_raw": _n_pdufa_events,
        "reg_calendar_entries_pit": _cal_diag.get("input_count", _n_pdufa_events),
        "reg_calendar_entries_used": _cal_diag.get("output_count", _n_pdufa_events),
        "reg_calendar_coverage_pct_eligible": _cal_diag.get(
            "coverage_pct",
            round(len({r.get("ticker") for r in _pdufa_manual}) / max(_eligible_count, 1) * 100, 1),
        ),
    }

    _reg_cov_pct = metadata["regulatory_coverage"].get("regulatory_secondary_coverage_pct", 0.0)
    if _reg_cov_pct < 5.0:
        logger.warning("Regulatory secondary coverage %.1f%% < 5.0%% — consider expanding event sources", _reg_cov_pct)

    # --- Per-ticker source reliability annotation ---
    _rel_action_counts: Dict[str, int] = {}
    if _reliability_table:
        try:
            from common.source_reliability import compute_priority_penalty, get_source_action

            for r in csv_rows:
                src = r.get("catalyst_source", "")
                conf = r.get("confidence_overall", "")
                fam = r.get("catalyst_family", "")
                if src:
                    action, _reason = get_source_action(
                        _reliability_table,
                        src,
                        conf,
                        fam,
                    )
                    penalty = compute_priority_penalty(action)
                    r["source_reliability_action"] = action
                    r["source_reliability_penalty"] = str(round(penalty, 1)) if penalty else ""
                    _rel_action_counts[action] = _rel_action_counts.get(action, 0) + 1
                else:
                    r["source_reliability_action"] = ""
                    r["source_reliability_penalty"] = ""
        except Exception:
            pass

    metadata["source_reliability"] = {
        "table_loaded": bool(_reliability_table),
        "table_buckets": len(_reliability_table),
        "action_counts": _rel_action_counts,
    }

    # Options diagnostics coverage telemetry
    _n_opt_candidates = sum(1 for r in csv_rows if r.get("opt_has_data") is not None)
    _n_opt_has_data = sum(1 for r in csv_rows if str(r.get("opt_has_data", "0")) == "1")
    _n_opt_usable = sum(1 for r in csv_rows if r.get("opt_use_for_judgment") == "YES")
    _n_opt_composite = sum(1 for r in csv_rows if r.get("options_quality_composite", "") != "")
    _n_opt_liquid = sum(1 for r in csv_rows if r.get("opt_liquidity_state") == "liquid")
    _n_opt_thin = sum(1 for r in csv_rows if r.get("opt_liquidity_state") == "thin")
    _opt_basis_counts: Dict[str, int] = {}
    for r in csv_rows:
        basis = r.get("opt_diagnostic_basis", "")
        if basis:
            _opt_basis_counts[basis] = _opt_basis_counts.get(basis, 0) + 1
    _opt_has_credentials = (
        not any(r.get("opt_diagnostic_basis") == "no_credentials" for r in csv_rows[:1]) if csv_rows else False
    )
    metadata["options_diagnostics"] = {
        "has_credentials": _opt_has_credentials,
        "n_candidates": _n_opt_candidates,
        "n_with_data": _n_opt_has_data,
        "n_liquid": _n_opt_liquid,
        "n_thin": _n_opt_thin,
        "n_use_for_judgment": _n_opt_usable,
        "n_options_quality_composite": _n_opt_composite,
        "coverage_pct": round(_n_opt_has_data / max(_n_opt_candidates, 1) * 100, 1),
        "diagnostic_basis_counts": dict(sorted(_opt_basis_counts.items())),
        "ab_ready": _n_opt_composite > 0,
    }

    meta_path = snap_path / "metadata.json"
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
            f.write("\n")
    except OSError as e:
        logger.warning(f"Could not write snapshot metadata: {e}")
        return None

    # Write inputs_manifest.json sidecar
    if inputs_manifest_mode in ("write", "verify"):
        manifest_data = results.get("run_metadata", {}).get("inputs_manifest")
        if manifest_data:
            try:
                with open(snap_path / "inputs_manifest.json", "w", encoding="utf-8") as f:
                    json.dump(manifest_data, f, indent=2, sort_keys=True)
                    f.write("\n")
            except OSError as e:
                logger.warning(f"Could not write inputs_manifest.json: {e}")

    # --- Coverage quality artifact (v1) ---
    try:
        _write_coverage_quality(
            snap_path,
            csv_rows,
            metadata,
            as_of_date,
            logger,
            _options_freshness,
            generated_at=_deterministic_timestamp(as_of_date),
        )
    except Exception as _cq_exc:
        logger.debug("Coverage quality artifact skipped: %s", _cq_exc)

    # Patch review_queue_summary into coverage_quality.json (must run after both are written)
    if _review_queue:
        try:
            from common.review_queue import compute_queue_summary as _cqs

            _cq_path = snap_path / "coverage_quality.json"
            if _cq_path.exists():
                _cq = json.loads(_cq_path.read_text())
                _cq["review_queue_summary"] = _cqs(_review_queue)
                with open(_cq_path, "w", encoding="utf-8") as f:
                    json.dump(_cq, f, indent=2, default=str)
                    f.write("\n")
        except Exception:
            pass

    # --- Spec 062: Options Expression Overlay (shadow-only) ---
    # Evaluates recommendations for all names with catalyst + EES data.
    # Logs decisions (all names) and attribution (tradeable only).
    # Outputs a sidecar summary. Zero production impact.
    try:
        from event_ev.expression_attribution import append_records_idempotent as _append_expr_records
        from event_ev.expression_attribution import build_attribution_record as _build_expr_attr_record
        from event_ev.expression_attribution import build_decision_record as _build_expr_dec_record
        from event_ev.expression_attribution import evaluate_kill_switches, load_attribution_log
        from event_ev.expression_layer import build_recommendation as _build_expr_rec

        _expr_decisions = []
        _expr_dec_records: List[Dict[str, Any]] = []
        _expr_attr_records: List[Dict[str, Any]] = []
        _expr_n_tradeable = 0
        _expr_n_rejected = 0
        _expr_by_type: Dict[str, int] = {}
        _expr_by_class: Dict[str, int] = {}
        _expr_by_gate: Dict[str, int] = {}

        # Load prior attribution log for kill-switch state
        _attr_log_path = Path("data/expression_attribution_log.jsonl")
        _dec_log_path = Path("data/expression_decision_log.jsonl")
        _prior_attr = load_attribution_log(_attr_log_path)
        _ks_state = evaluate_kill_switches(_prior_attr)

        _ees_by_ticker: Dict[str, Any] = {}
        if _ees_scores_for_sidecar:
            for _ees_obj in _ees_scores_for_sidecar:
                _ees_by_ticker[_ees_obj.ticker] = _ees_obj

        for _row in csv_rows:
            _tk = _row.get("ticker", "")
            if not _tk:
                continue

            # Need both catalyst_days and EES to evaluate
            _cd_raw = _row.get("catalyst_days", "")
            try:
                _cd = int(float(_cd_raw)) if _cd_raw else 0
            except (ValueError, TypeError):
                _cd = 0
            if _cd <= 0:
                continue

            _ees_obj = _ees_by_ticker.get(_tk)
            if _ees_obj is None:
                continue

            # Build lightweight typed proxies from CSV row fields
            from event_ev.data_contracts import CatalystNode as _CN
            from event_ev.data_contracts import CrowdBelief as _CB
            from event_ev.data_contracts import ExpectationErrorScore as _EES
            from event_ev.data_contracts import OutcomeProbabilities as _OP
            from event_ev.data_contracts import ScenarioPayoffs as _SP
            from event_ev.data_contracts import TimingEstimate as _TE

            _node_id = f"{_tk}_{as_of_date}"

            # Compute expected_date from catalyst_days + as_of_date
            from datetime import date as _date_cls
            from datetime import timedelta as _td_cls

            _expected_date = str(_date_cls.fromisoformat(as_of_date) + _td_cls(days=_cd))

            _node = _CN(
                ticker=_tk,
                event_family=str(_row.get("catalyst_family", "CLINICAL")).upper(),
                event_type=str(_row.get("catalyst_event_type", "")),
                event_subtype="",
                expected_date=_expected_date,
                date_range_start=None,
                date_range_end=None,
                date_precision=_row.get("catalyst_date_precision", "MONTH"),
                date_confidence=float(_row.get("catalyst_date_confidence", 0.5) or 0.5),
                source=str(_row.get("catalyst_source", "")),
                source_uid=str(_row.get("nearest_catalyst_source_uid", _tk)),
                disclosed_at=as_of_date,
                phase=str(_row.get("phase", "unknown")),
                indication=str(_row.get("primary_indication", "")),
                node_id=_node_id,
            )

            # OutcomeProbabilities: use priors if available, else 50/40/10
            _p_hit = _safe_float(_row.get("p_hit"), default=0.50)
            _p_miss = _safe_float(_row.get("p_miss"), default=0.40)
            _p_mixed = max(0.0, round(1.0 - _p_hit - _p_miss, 4))
            _out_conf = _safe_float(_row.get("outcome_confidence"), default=0.50)
            try:
                _outcome = _OP(
                    node_id=_node_id,
                    as_of_date=as_of_date,
                    p_hit=_p_hit,
                    p_miss=_p_miss,
                    p_mixed=_p_mixed,
                    confidence=_out_conf,
                    prior_source="csv_proxy",
                )
            except ValueError:
                continue  # p_hit + p_miss + p_mixed != 1.0

            _pm_pct = _safe_float(_row.get("priced_move_pct"))
            # mispricing_score: prefer dedicated field, fall back to
            # conditional_misprice_score from EES (both measure model-vs-market gap)
            _misprice = _safe_float(_row.get("mispricing_score"))
            _misprice_source = "true"
            if _misprice is None or _misprice == 0.0:
                _misprice = _safe_float(_row.get("conditional_misprice_score"), default=0.0)
                _misprice_source = "proxy_cond_misprice"

            _crowd = _CB(
                node_id=_node_id,
                as_of_date=as_of_date,
                implied_p_hit=max(0.0, min(1.0, _p_hit - _misprice)),
                belief_direction="NEUTRAL",
                belief_intensity=0.5,
                priced_move_pct=_pm_pct,
                mispricing_score=_misprice,
            )

            # scenario_ev: prefer dedicated field, fall back to EES-derived estimate.
            # When payoff engine is not in the production pipeline, derive from
            # priced_move_pct * mispricing direction (gives a rough EV proxy).
            _sev = _safe_float(_row.get("scenario_ev"))
            _ev_source = "true"
            if _sev is None or _sev == 0.0:
                _up_est = _safe_float(_row.get("upside_hit"))
                _dn_est = _safe_float(_row.get("downside_miss"))
                if _up_est is not None and _dn_est is not None:
                    _sev = round(_p_hit * _up_est + _p_miss * _dn_est + _p_mixed * 0.0, 4)
                    _ev_source = "proxy_payoff"
                elif _pm_pct is not None and _pm_pct > 0 and _misprice != 0:
                    # Crude proxy: priced_move_pct * mispricing_direction * scaling
                    # If model says P(HIT) > market (misprice > 0), EV is positive
                    _sev = round(_pm_pct * 100.0 * _misprice * 0.5, 4)
                    _ev_source = "proxy_pm"
                else:
                    _sev = round(_p_hit * 20.0 + _p_miss * (-20.0), 4)
                    _ev_source = "fallback"
            _up = _safe_float(_row.get("upside_hit"), default=20.0)
            _down = _safe_float(_row.get("downside_miss"), default=-20.0)
            _asym = abs(_up / _down) if _down != 0 else 1.0

            _payoff = _SP(
                node_id=_node_id,
                as_of_date=as_of_date,
                upside_hit=_up,
                downside_miss=_down,
                move_mixed=_safe_float(_row.get("move_mixed"), default=0.0),
                scenario_ev=_sev,
                asymmetry_ratio=round(_asym, 4),
                downside_adjusted_ev=_safe_float(_row.get("downside_adjusted_ev"), default=_sev),
                kelly_fraction=_safe_float(_row.get("kelly_fraction"), default=0.0),
                analog_count=int(_safe_float(_row.get("analog_count"), default=10)),
                analog_confidence=str(_row.get("analog_confidence", "low") or "low"),
            )

            _timing = _TE(
                node_id=_node_id,
                as_of_date=as_of_date,
                prob_on_time=_safe_float(_row.get("prob_on_time"), default=0.60),
                prob_slip=_safe_float(_row.get("prob_slip"), default=0.20),
                prob_early=_safe_float(_row.get("prob_early"), default=0.10),
                expected_delay_days=_safe_float(_row.get("expected_delay_days"), default=0.0),
                median_arrival_days=float(_cd),
                hazard_rate=0.03,
            )

            _rec = _build_expr_rec(
                node=_node,
                outcome=_outcome,
                crowd=_crowd,
                payoff=_payoff,
                ees=_ees_obj,
                timing=_timing,
                as_of_date=as_of_date,
                opt_liquidity_state=str(_row.get("opt_liquidity_state", "")),
                opt_atm_iv=_safe_float(_row.get("opt_atm_iv")),
                opt_front_iv=_safe_float(_row.get("opt_front_iv")),
                opt_back_iv=_safe_float(_row.get("opt_back_iv")),
                opt_rr_25d=_safe_float(_row.get("opt_rr_25d")),
                bid_ask_spread_pct=_safe_float(_row.get("bid_ask_spread_pct")),
                priced_move_pct=_pm_pct,
                quote_fresh=True,
            )

            # Inject EV source provenance (dict inside frozen dataclass is mutable)
            _rec.inputs_used["ev_source"] = _ev_source
            _rec.inputs_used["misprice_source"] = _misprice_source

            # Decision record (ALL names) — batched, written once after the loop
            _ks_active = not _ks_state.get("overlay_enabled", True)
            _ks_reason = "; ".join(_ks_state.get("triggered_rules", [])) if _ks_active else None
            _expr_dec_records.append(
                _build_expr_dec_record(
                    _rec,
                    timestamp=as_of_date + "T00:00:00",
                    kill_switch_active=_ks_active,
                    kill_switch_reason=_ks_reason,
                )
            )
            _expr_decisions.append(_rec)

            # Attribution record (tradeable only) — batched
            if _rec.is_tradeable and not _ks_active:
                _expr_attr_records.append(_build_expr_attr_record(_rec, timestamp=as_of_date + "T00:00:00"))
                _expr_n_tradeable += 1
            else:
                _expr_n_rejected += 1

            # Tally counts
            _expr_by_type[_rec.mispricing_type] = _expr_by_type.get(_rec.mispricing_type, 0) + 1
            _expr_by_class[_rec.overlay_class] = _expr_by_class.get(_rec.overlay_class, 0) + 1
            for _gf in _rec.gate_failures:
                _expr_by_gate[_gf] = _expr_by_gate.get(_gf, 0) + 1

        # Persist logs once per run. Idempotent per (ticker, node_id): a re-run
        # for the same as_of date replaces its own records instead of appending
        # duplicates that would skew kill-switch metrics (#495).
        _dec_write = _append_expr_records(_dec_log_path, _expr_dec_records)
        _attr_write = _append_expr_records(_attr_log_path, _expr_attr_records)
        if _dec_write["replaced"] or _attr_write["replaced"]:
            logger.info(
                "Expression logs: replaced %d decision / %d attribution records from a prior same-day run",
                _dec_write["replaced"],
                _attr_write["replaced"],
            )

        # Calibration diagnostics — EV source distribution + score distributions
        _ev_sources: Dict[str, int] = {}
        _misprice_sources: Dict[str, int] = {}
        _ev_vals: List[float] = []
        _misprice_vals: List[float] = []
        _belief_vals: List[float] = []
        _permission_vals: List[float] = []
        for _r in _expr_decisions:
            _evs = _r.inputs_used.get("ev_source", "unknown")
            _ev_sources[_evs] = _ev_sources.get(_evs, 0) + 1
            _ms = _r.inputs_used.get("misprice_source", "unknown")
            _misprice_sources[_ms] = _misprice_sources.get(_ms, 0) + 1
            _ev_vals.append(_r.scenario_ev)
            _misprice_vals.append(abs(_r.inputs_used.get("mispricing_score", 0) or 0))
            _belief_vals.append(_r.belief_strength)
            _permission_vals.append(_r.permission_to_express)

        def _dist_summary(vals):
            if not vals:
                return {}
            s = sorted(vals)
            n = len(s)
            return {
                "n": n,
                "min": round(s[0], 4),
                "p25": round(s[n // 4], 4),
                "median": round(s[n // 2], 4),
                "p75": round(s[3 * n // 4], 4),
                "max": round(s[-1], 4),
                "mean": round(sum(s) / n, 4),
                "pct_above_threshold": round(sum(1 for v in s if v >= 3.0) / n * 100, 1),
            }

        _calibration = {
            "ev_source_distribution": dict(sorted(_ev_sources.items())),
            "misprice_source_distribution": dict(sorted(_misprice_sources.items())),
            "scenario_ev_distribution": _dist_summary(_ev_vals),
            "abs_mispricing_score_distribution": _dist_summary(_misprice_vals),
            "belief_strength_distribution": _dist_summary(_belief_vals),
            "permission_to_express_distribution": _dist_summary(_permission_vals),
        }

        # Write sidecar summary
        _expr_summary = {
            "schema": "expression_overlay_summary.v1",
            "as_of_date": as_of_date,
            "n_evaluated": len(_expr_decisions),
            "n_tradeable": _expr_n_tradeable,
            "n_rejected": _expr_n_rejected,
            "counts_by_mispricing_type": dict(sorted(_expr_by_type.items())),
            "counts_by_overlay_class": dict(sorted(_expr_by_class.items())),
            "counts_rejected_by_gate": dict(sorted(_expr_by_gate.items(), key=lambda x: -x[1])),
            "calibration": _calibration,
            "kill_switch_state": {
                "overlay_enabled": _ks_state.get("overlay_enabled", True),
                "sizing_enabled": _ks_state.get("sizing_enabled", True),
                "disabled_types": _ks_state.get("disabled_types", []),
                "triggered_rules": _ks_state.get("triggered_rules", []),
                "evaluation_status": _ks_state.get("evaluation_status", "not_evaluated"),
            },
            "attribution_log_path": str(_attr_log_path),
            "decision_log_path": str(_dec_log_path),
        }
        with open(snap_path / "expression_overlay_summary.json", "w", encoding="utf-8") as f:
            json.dump(_expr_summary, f, indent=2, default=str)
            f.write("\n")

        # Tradeable recommendations sidecar (for operator review)
        _tradeable_recs = [r.to_dict() for r in _expr_decisions if r.is_tradeable]
        if _tradeable_recs:
            with open(snap_path / "expression_recommendations.json", "w", encoding="utf-8") as f:
                json.dump(_tradeable_recs, f, indent=2, default=str)
                f.write("\n")

        logger.info(
            "[EXPR] Expression overlay: %d evaluated, %d tradeable, %d rejected, " "kill_switch=%s",
            len(_expr_decisions),
            _expr_n_tradeable,
            _expr_n_rejected,
            "OFF" if _ks_state.get("overlay_enabled", True) else "ON",
        )
    except Exception as _expr_exc:
        logger.debug("Expression overlay skipped: %s", _expr_exc)

    logger.info(f"[SNAPSHOT] Saved validation snapshot: {snap_path} " f"({len(ranked)} tickers, v{version})")
    return snap_path


# =============================================================================
# SNAPSHOT COMPARISON
# =============================================================================


def compare_snapshots_detailed(
    state_dir: Path,
    current_date: str,
    prior_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare two CT.gov state snapshots with detailed hash and field-diff summary.

    Args:
        state_dir: Directory containing state_YYYY-MM-DD.jsonl files
        current_date: Current snapshot date (YYYY-MM-DD)
        prior_date: Prior snapshot date (default: day before current_date)

    Returns:
        Dict with comparison results including:
        - file paths, line counts, SHA256 hashes
        - field changes detected
        - records added/removed
    """
    from datetime import timedelta

    current_date_obj = to_date_object(current_date)
    if prior_date:
        prior_date_obj = to_date_object(prior_date)
    else:
        prior_date_obj = current_date_obj - timedelta(days=1)
        prior_date = prior_date_obj.isoformat()

    current_file = state_dir / f"state_{current_date}.jsonl"
    prior_file = state_dir / f"state_{prior_date}.jsonl"

    result = {
        "current_date": current_date,
        "prior_date": prior_date,
        "current_file": str(current_file),
        "prior_file": str(prior_file),
        "current_exists": current_file.exists(),
        "prior_exists": prior_file.exists(),
        "current_sha256": None,
        "prior_sha256": None,
        "current_line_count": 0,
        "prior_line_count": 0,
        "files_identical": False,
        "field_changes_detected": 0,
        "records_added": 0,
        "records_removed": 0,
        "top_changed_fields": {},
        "comparison_status": "OK",
    }

    # Check if files exist
    if not current_file.exists():
        result["comparison_status"] = "CURRENT_MISSING"
        return result
    if not prior_file.exists():
        result["comparison_status"] = "PRIOR_MISSING"
        return result

    # Compute hashes and line counts
    current_bytes = current_file.read_bytes()
    prior_bytes = prior_file.read_bytes()

    result["current_sha256"] = hashlib.sha256(current_bytes).hexdigest()
    result["prior_sha256"] = hashlib.sha256(prior_bytes).hexdigest()
    result["current_line_count"] = current_bytes.count(b"\n")
    result["prior_line_count"] = prior_bytes.count(b"\n")

    # Check if files are identical
    if result["current_sha256"] == result["prior_sha256"]:
        result["files_identical"] = True
        result["comparison_status"] = "IDENTICAL"
        return result

    # Parse and compare records
    def parse_jsonl(file_path: Path) -> Dict[str, Dict]:
        records = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        # Use (ticker, nct_id) as key
                        key = (record.get("ticker", ""), record.get("nct_id", ""))
                        records[key] = record
                    except json.JSONDecodeError:
                        continue
        return records

    current_records = parse_jsonl(current_file)
    prior_records = parse_jsonl(prior_file)

    current_keys = set(current_records.keys())
    prior_keys = set(prior_records.keys())

    result["records_added"] = len(current_keys - prior_keys)
    result["records_removed"] = len(prior_keys - current_keys)

    # Check field changes in common records
    common_keys = current_keys & prior_keys
    field_changes = {}
    for key in common_keys:
        curr = current_records[key]
        prev = prior_records[key]
        for field in set(curr.keys()) | set(prev.keys()):
            if curr.get(field) != prev.get(field):
                field_changes[field] = field_changes.get(field, 0) + 1
                result["field_changes_detected"] += 1

    # Top changed fields
    sorted_fields = sorted(field_changes.items(), key=lambda x: -x[1])[:10]
    result["top_changed_fields"] = dict(sorted_fields)

    return result


def write_diagnostics_output(
    diagnostics_path: Path,
    results: Dict[str, Any],
    args: argparse.Namespace,
    catalyst_window: tuple[int, int],
) -> None:
    """
    Write structured diagnostics to JSON sidecar file.

    Args:
        diagnostics_path: Output path for diagnostics JSON
        results: Full pipeline results
        args: Parsed CLI arguments
        catalyst_window: Resolved catalyst window tuple
    """
    diagnostics = {
        "schema_version": "1.0.0",
        "generated_by": "run_screen.py",
        "run_metadata": {
            "as_of_date": args.as_of_date,
            "orchestrator_version": VERSION,
            "cli_arguments": {
                "log_level": args.log_level,
                "diagnostics": args.diagnostics,
                "catalyst_window": f"{catalyst_window[0]}-{catalyst_window[1]}",
                "catalyst_window_preset": args.catalyst_window_preset,
                "catalyst_decay": args.catalyst_decay,
                "catalyst_half_life_days": args.catalyst_half_life_days,
                "enable_smart_money": args.enable_smart_money,
                "smart_money_source": args.smart_money_source,
                "enable_enhancements": args.enable_enhancements,
                "enable_short_interest": args.enable_short_interest,
                "enable_coinvest": args.enable_coinvest,
                "min_component_coverage": args.min_component_coverage,
                "min_component_coverage_mode": args.min_component_coverage_mode,
            },
        },
        "pipeline_summary": results.get("summary", {}),
        "module_diagnostics": {},
        "component_coverage": {},
        "gating_analysis": {},
    }

    # Extract module diagnostics
    if "module_1_universe" in results:
        m1 = results["module_1_universe"]
        diagnostics["module_diagnostics"]["module_1"] = {
            "active_count": len(m1.get("active_securities", [])),
            "excluded_count": len(m1.get("excluded_securities", [])),
            "diagnostic_counts": m1.get("diagnostic_counts", {}),
        }

    if "module_2_financial" in results:
        m2 = results["module_2_financial"]
        diagnostics["module_diagnostics"]["module_2"] = {
            "scored_count": len(m2.get("scores", [])),
            "diagnostic_counts": m2.get("diagnostic_counts", {}),
        }

    if "module_3_catalyst" in results:
        m3 = results["module_3_catalyst"]
        diagnostics["module_diagnostics"]["module_3"] = {
            "events_total": m3.get("diagnostic_counts", {}).get("events_detected_total", 0),
            "tickers_with_events": m3.get("diagnostic_counts", {}).get("tickers_with_events", 0),
            "events_by_type": m3.get("diagnostic_counts", {}).get("events_by_type", {}),
            "events_by_severity": m3.get("diagnostic_counts", {}).get("events_by_severity", {}),
        }

    if "module_4_clinical" in results:
        m4 = results["module_4_clinical"]
        diagnostics["module_diagnostics"]["module_4"] = {
            "scored_count": len(m4.get("scores", [])),
            "diagnostic_counts": m4.get("diagnostic_counts", {}),
        }

    if "module_5_composite" in results:
        m5 = results["module_5_composite"]
        diagnostics["module_diagnostics"]["module_5"] = {
            "ranked_count": len(m5.get("ranked_securities", [])),
            "excluded_count": len(m5.get("excluded_securities", [])),
            "run_status": m5.get("run_status", "UNKNOWN"),
            "degraded_components": m5.get("degraded_components", []),
            "health_warnings": m5.get("health_warnings", []),
            "health_errors": m5.get("health_errors", []),
        }
        diagnostics["component_coverage"] = m5.get("component_coverage", {})
        diagnostics["gating_analysis"] = {
            "gated_component_counts": m5.get("gated_component_counts", {}),
            "confidence_gated_tickers": m5.get("confidence_gated_tickers", {}),
        }

    # Enhancement diagnostics
    if "enhancements" in results:
        enh = results["enhancements"]
        diagnostics["enhancement_diagnostics"] = {}

        if enh.get("regime"):
            diagnostics["enhancement_diagnostics"]["regime"] = {
                "regime": enh["regime"].get("regime"),
                "confidence": str(enh["regime"].get("confidence", "0")),
                "staleness": enh["regime"].get("staleness"),
            }

        if enh.get("pos_scores"):
            pos = enh["pos_scores"]
            diagnostics["enhancement_diagnostics"]["pos"] = pos.get("diagnostic_counts", {})

        if enh.get("short_interest_scores"):
            si = enh["short_interest_scores"]
            diagnostics["enhancement_diagnostics"]["short_interest"] = si.get("diagnostic_counts", {})

        if enh.get("dilution_risk_scores"):
            dr = enh["dilution_risk_scores"]
            diagnostics["enhancement_diagnostics"]["dilution_risk"] = dr.get("diagnostic_counts", {})

    write_json_output(diagnostics_path, diagnostics, secure=True)
    logger.info(f"  Diagnostics written to: {diagnostics_path}")


def log_smart_money_debug(
    institutional_momentum: Optional[Dict],
    market_data_by_ticker: Dict[str, Dict],
    active_tickers: List[str],
    data_dir: Path,
    smart_money_source: str,
) -> None:
    """
    Print detailed smart money diagnostics when --debug-smart-money is enabled.

    Args:
        institutional_momentum: Loaded institutional momentum data
        market_data_by_ticker: Market data dict indexed by ticker
        active_tickers: List of active tickers from Module 1
        data_dir: Data directory path
        smart_money_source: Smart money source setting ('13f', 'internal', 'auto')
    """
    logger.info("=" * 60)
    logger.info("SMART MONEY DEBUG DIAGNOSTICS")
    logger.info("=" * 60)

    # File resolution
    momentum_file = data_dir / "momentum_results.json"
    root_momentum_file = Path("momentum_results.json")
    holdings_file = data_dir / "holdings_snapshots.json"
    coinvest_file = data_dir / "coinvest_signals.json"

    logger.info(f"Source setting: --smart-money-source={smart_money_source}")
    logger.info("")
    logger.info("File paths searched:")
    logger.info(f"  {momentum_file}: {'EXISTS' if momentum_file.exists() else 'NOT FOUND'}")
    logger.info(f"  {root_momentum_file}: {'EXISTS' if root_momentum_file.exists() else 'NOT FOUND'}")
    logger.info(f"  {holdings_file}: {'EXISTS' if holdings_file.exists() else 'NOT FOUND'}")
    logger.info(f"  {coinvest_file}: {'EXISTS' if coinvest_file.exists() else 'NOT FOUND'}")
    logger.info("")

    if institutional_momentum is None:
        logger.info("Institutional momentum data: NOT LOADED")
        logger.info("  Reason: No momentum_results.json found in data_dir or root")
        logger.info("=" * 60)
        return

    # Data statistics
    rankings = institutional_momentum.get("rankings", [])
    summary = institutional_momentum.get("summary", {})

    logger.info(f"Institutional momentum data: LOADED")
    logger.info(f"  Total tickers in file: {len(rankings)}")
    logger.info(f"  Coordinated buys: {len(summary.get('coordinated_buys', []))}")
    logger.info(f"  Coordinated sells: {len(summary.get('coordinated_sells', []))}")
    logger.info("")

    # Universe overlap analysis
    momentum_tickers = {r.get("ticker", "").upper() for r in rankings if r.get("ticker")}
    active_set = {t.upper() for t in active_tickers}

    overlap = momentum_tickers & active_set
    in_momentum_not_active = momentum_tickers - active_set
    in_active_not_momentum = active_set - momentum_tickers

    logger.info("Universe overlap:")
    logger.info(f"  Tickers in momentum file: {len(momentum_tickers)}")
    logger.info(f"  Active tickers (Module 1): {len(active_set)}")
    logger.info(f"  Overlap (can be scored): {len(overlap)}")
    logger.info(f"  In momentum but not active: {len(in_momentum_not_active)}")
    logger.info(f"  In active but not momentum: {len(in_active_not_momentum)}")

    if in_momentum_not_active and len(in_momentum_not_active) <= 10:
        logger.info(f"    Examples not in active: {sorted(in_momentum_not_active)[:10]}")
    if in_active_not_momentum and len(in_active_not_momentum) <= 10:
        logger.info(f"    Examples missing momentum: {sorted(in_active_not_momentum)[:10]}")
    logger.info("")

    # Injection analysis
    injected_count = 0
    not_injected_reasons = {"no_market_data": 0, "already_has_return": 0, "invalid_score": 0}

    for ranking in rankings:
        ticker = ranking.get("ticker", "").upper()
        momentum_score = ranking.get("momentum_score")

        if ticker not in market_data_by_ticker:
            not_injected_reasons["no_market_data"] += 1
        elif market_data_by_ticker[ticker].get("return_60d") is not None:
            # Check if it was us who set it
            if market_data_by_ticker[ticker].get("_momentum_source") == "13f":
                injected_count += 1
            else:
                not_injected_reasons["already_has_return"] += 1
        elif momentum_score is None:
            not_injected_reasons["invalid_score"] += 1

    logger.info("Injection analysis (into market_data_by_ticker):")
    logger.info(f"  Successfully injected: {injected_count}")
    logger.info(f"  Not injected - no market data: {not_injected_reasons['no_market_data']}")
    logger.info(f"  Not injected - already has return_60d: {not_injected_reasons['already_has_return']}")
    logger.info(f"  Not injected - invalid score: {not_injected_reasons['invalid_score']}")
    logger.info("=" * 60)


# =============================================================================
# CHECKPOINTING / MANIFEST / AUDIT — delegated to run_screen_checkpoint.py
# Re-exports below maintain backward compatibility for existing importers.
# =============================================================================

from run_screen_checkpoint import (  # noqa: E402,F811
    CHECKPOINT_MODULES,
    DEPENDENCY_REGISTRY,
    MANIFEST_VERSION,
    InputDependency,
    _bundle_relpath,
    _count_records,
    _resolve_cache_paths,
    append_audit_log,
    build_inputs_manifest,
    create_audit_record,
    create_replay_bundle,
    extract_replay_bundle,
    get_resume_module_index,
    load_checkpoint,
    save_checkpoint,
    validate_inputs_dry_run,
    verify_against_prior_manifest,
    verify_input_freshness,
    verify_inputs_manifest,
)

# --- Old checkpoint/manifest/audit implementations removed ---
# See run_screen_checkpoint.py for the canonical implementations.
# The re-exports above ensure backward compatibility.

# Marker to find the deletion boundary — search for _load_module5_weights below
_CHECKPOINT_SECTION_REMOVED = True


def _load_module5_weights(path: Path) -> Optional[Dict[str, "Decimal"]]:
    """
    Load calibrated Module 5 weights from JSON.

    Expected format:
        { "feature_weights": {"valuation_normalized": -0.35, ...}, ... }

    Returns a dict mapping component name -> Decimal weight suitable for
    passing to compute_module_5_composite_v3(weights=...).
    Returns None if file missing/invalid (with warning).
    """
    from decimal import Decimal

    if not path.exists():
        logger.warning(f"Module 5 weights file not found: {path} — using defaults")
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load Module 5 weights from {path}: {e} — using defaults")
        return None

    feature_weights = data.get("feature_weights")
    if not isinstance(feature_weights, dict):
        logger.warning(f"Invalid Module 5 weights format in {path} — using defaults")
        return None

    # Map calibrated feature names to Module 5 component names.
    # Calibrator outputs "valuation_normalized" → Module 5 expects "valuation".
    def _priority(name: str) -> int:
        if name.endswith("_normalized"):
            return 0
        if name.endswith("_raw"):
            return 1
        if name.endswith("_contribution"):
            return 2
        if name.endswith("_confidence"):
            return 3
        return 4

    weights: Dict[str, Decimal] = {}
    for feat_name, w in sorted(feature_weights.items(), key=lambda kv: (_priority(kv[0]), kv[0])):
        # Validate numeric
        try:
            w_f = float(w)
        except (TypeError, ValueError):
            logger.warning(f"Invalid weight for {feat_name} in {path}: {w!r} — skipping")
            continue

        # Strip _normalized/_raw/_contribution suffix to get component name
        component = feat_name
        for suffix in ("_normalized", "_raw", "_contribution", "_confidence"):
            if feat_name.endswith(suffix):
                component = feat_name[: -len(suffix)]
                break
        if abs(w_f) < 1e-10:
            continue  # Skip zero-weight features (dropped/unused)
        if component in weights:
            logger.warning(f"Duplicate component weight for {component} from {feat_name} — keeping first")
            continue
        weights[component] = Decimal(str(w_f))

    if not weights:
        logger.warning(f"All weights are zero in {path} — using defaults")
        return None

    logger.info(f"Loaded calibrated Module 5 weights from {path}:")
    for comp in sorted(weights.keys()):
        logger.info(f"  {comp}: {weights[comp]}")

    return weights


# ── Schema v2 bundle loading + regime-conditioned weight selection ────


def _parse_feature_weights_to_components(
    feature_weights: Dict[str, Any],
) -> Optional[Dict[str, "Decimal"]]:
    """
    Convert a feature_weights dict to component name → Decimal mapping.

    Shared logic extracted from _load_module5_weights to reuse for bundle parsing.
    """
    from decimal import Decimal

    def _priority(name: str) -> int:
        if name.endswith("_normalized"):
            return 0
        if name.endswith("_raw"):
            return 1
        if name.endswith("_contribution"):
            return 2
        if name.endswith("_confidence"):
            return 3
        return 4

    weights: Dict[str, Decimal] = {}
    for feat_name, w in sorted(feature_weights.items(), key=lambda kv: (_priority(kv[0]), kv[0])):
        try:
            w_f = float(w)
        except (TypeError, ValueError):
            continue
        component = feat_name
        for suffix in ("_normalized", "_raw", "_contribution", "_confidence"):
            if feat_name.endswith(suffix):
                component = feat_name[: -len(suffix)]
                break
        if abs(w_f) < 1e-10:
            continue
        if component in weights:
            continue
        weights[component] = Decimal(str(w_f))

    return weights if weights else None


def _load_module5_weights_bundle(path: Path) -> Optional[Dict[str, Any]]:
    """
    Load a Module 5 weight bundle from JSON (schema_version 1 or 2).

    Schema v1 (existing format):
        { "feature_weights": { ... } }
        → Treated as global-only, no regime weights.

    Schema v2 (bundle format):
        {
          "schema_version": 2,
          "blend": { "k": 25, "min_fit_weeks": 5 },
          "global": { "feature_weights": { ... } },
          "by_regime": {
            "BULL": { "n_weeks": 71, "feature_weights": { ... } },
            ...
          }
        }

    Returns a normalized bundle dict:
        {
          "global": Dict[str, Decimal],
          "by_regime": Dict[str, Dict[str, Decimal]],
          "n_weeks": Dict[str, int],
          "blend_k": int,
          "min_fit_weeks": int,
        }
    Returns None if file missing/invalid.
    """
    from decimal import Decimal

    if not path.exists():
        logger.warning(f"Module 5 weights file not found: {path} — using defaults")
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load Module 5 weights from {path}: {e} — using defaults")
        return None

    schema_version = data.get("schema_version", 1)

    if schema_version == 1:
        # Legacy format: wrap as global-only bundle
        feature_weights = data.get("feature_weights")
        if not isinstance(feature_weights, dict):
            logger.warning(f"Invalid Module 5 weights format in {path} — using defaults")
            return None
        global_weights = _parse_feature_weights_to_components(feature_weights)
        if global_weights is None:
            logger.warning(f"All weights are zero in {path} — using defaults")
            return None

        logger.info(f"Loaded Module 5 weights (schema v1, global only) from {path}:")
        for comp in sorted(global_weights.keys()):
            logger.info(f"  {comp}: {global_weights[comp]}")

        return {
            "global": global_weights,
            "by_regime": {},
            "n_weeks": {},
            "blend_k": 25,
            "min_fit_weeks": 5,
        }

    elif schema_version == 2:
        # Bundle format
        global_block = data.get("global", {})
        global_fw = global_block.get("feature_weights")
        if not isinstance(global_fw, dict):
            logger.warning(f"Schema v2 bundle missing global.feature_weights in {path} — using defaults")
            return None
        global_weights = _parse_feature_weights_to_components(global_fw)
        if global_weights is None:
            logger.warning(f"All global weights are zero in {path} — using defaults")
            return None

        # Parse blend params
        blend_cfg = data.get("blend", {})
        blend_k = int(blend_cfg.get("k", 25))
        min_fit_weeks = int(blend_cfg.get("min_fit_weeks", 5))

        # Parse per-regime weights
        by_regime: Dict[str, Dict[str, Decimal]] = {}
        n_weeks: Dict[str, int] = {}
        for regime_name, regime_block in data.get("by_regime", {}).items():
            regime_fw = regime_block.get("feature_weights")
            if not isinstance(regime_fw, dict):
                continue
            parsed = _parse_feature_weights_to_components(regime_fw)
            if parsed is not None:
                by_regime[regime_name] = parsed
                n_weeks[regime_name] = int(regime_block.get("n_weeks", 0))

        logger.info(f"Loaded Module 5 weights bundle (schema v2) from {path}:")
        logger.info(f"  Global weights: {', '.join(f'{k}={v}' for k, v in sorted(global_weights.items()))}")
        logger.info(f"  Regimes: {sorted(by_regime.keys()) or '(none)'}")
        logger.info(f"  Blend k={blend_k}, min_fit_weeks={min_fit_weeks}")

        return {
            "global": global_weights,
            "by_regime": by_regime,
            "n_weeks": n_weeks,
            "blend_k": blend_k,
            "min_fit_weeks": min_fit_weeks,
        }

    else:
        logger.warning(f"Unknown schema_version {schema_version} in {path} — using defaults")
        return None


def _select_module5_weights(
    bundle: Dict[str, Any],
    regime: Optional[str],
    mode: str = "global",
) -> Dict[str, "Decimal"]:
    """
    Select the effective Module 5 weights from a bundle given current regime.

    Parameters
    ----------
    bundle : normalized bundle from _load_module5_weights_bundle
    regime : current market regime label (e.g., "BULL", "BEAR", "CHOP")
    mode : one of "global", "regime", "blended"

    Returns
    -------
    Dict[str, Decimal] mapping component name → weight
    """
    from decimal import Decimal

    global_w = bundle["global"]

    if mode == "global":
        logger.info("  Module 5 weight mode: global (ignoring regime)")
        return global_w

    by_regime = bundle.get("by_regime", {})
    n_weeks = bundle.get("n_weeks", {})
    blend_k = bundle.get("blend_k", 25)
    min_fit_weeks = bundle.get("min_fit_weeks", 5)

    regime_w = by_regime.get(regime) if regime else None

    if regime_w is None:
        logger.info(f"  Module 5 weight mode: {mode}, regime={regime} " f"→ no regime weights available, using global")
        return global_w

    regime_n = n_weeks.get(regime, 0)

    if mode == "regime":
        if regime_n < min_fit_weeks:
            logger.info(
                f"  Module 5 weight mode: regime, regime={regime}, "
                f"n_weeks={regime_n} < min_fit_weeks={min_fit_weeks} → global fallback"
            )
            return global_w
        logger.info(
            f"  Module 5 weight mode: regime, regime={regime}, " f"n_weeks={regime_n} → using pure regime weights"
        )
        return regime_w

    # mode == "blended"
    if regime_n < min_fit_weeks:
        logger.info(
            f"  Module 5 weight mode: blended, regime={regime}, "
            f"n_weeks={regime_n} < min_fit_weeks={min_fit_weeks} → global fallback"
        )
        return global_w

    alpha = Decimal(str(regime_n)) / (Decimal(str(regime_n)) + Decimal(str(blend_k)))
    blended: Dict[str, Decimal] = {}
    all_components = set(global_w.keys()) | set(regime_w.keys())
    for comp in sorted(all_components):
        g = global_w.get(comp, Decimal("0"))
        r = regime_w.get(comp, Decimal("0"))
        blended[comp] = alpha * r + (Decimal("1") - alpha) * g

    logger.info(f"  Module 5 weight mode: blended, regime={regime}, " f"n_weeks={regime_n}, alpha={float(alpha):.3f}")
    for comp in sorted(blended.keys()):
        logger.info(
            f"    {comp}: {blended[comp]:.6f} "
            f"(global={global_w.get(comp, Decimal('0')):.6f}, "
            f"regime={regime_w.get(comp, Decimal('0')):.6f})"
        )

    return blended


def _normalize_m3_regime(regime: Optional[str]) -> Optional[str]:
    """
    Map production regime labels to the coarse {BULL, BEAR, CHOP} buckets
    used by M3 walk-forward weight bundles.

    Production's RegimeDetectionEngine emits richer labels (VOLATILITY_SPIKE,
    SECTOR_ROTATION, RECESSION_RISK, etc.) that don't exist in the bundle.
    Without mapping, blended mode would always take global fallback.
    """
    if not regime or regime == "UNKNOWN":
        return None
    r = str(regime).upper()
    if r in ("BULL", "BEAR", "CHOP", "CHOP_LOWVOL", "CHOP_HIGHVOL"):
        return r
    if "BULL" in r:
        return "BULL"
    if any(k in r for k in ("BEAR", "RECESSION", "CRISIS", "VOLATILITY_SPIKE")):
        return "BEAR"
    # Everything else (SECTOR_ROTATION, SECTOR_DISLOCATION, unclear) → CHOP
    return "CHOP"


def _build_m3_config(
    sec_8k_mode: str = "cache_only",
    sec_multi_form_mode: str = "cache_only",
    fda_regulatory_mode: str = "cache_only",
    ctgov_cache_dir=None,
) -> "Module3Config":
    """Build Module3Config, optionally overriding SEC 8-K mode."""
    cfg = Module3Config()
    cfg.enable_sec_8k_catalysts = sec_8k_mode
    cfg.enable_sec_multi_form = sec_multi_form_mode
    cfg.enable_fda_regulatory = fda_regulatory_mode
    cfg.ctgov_cache_dir = ctgov_cache_dir
    return cfg


def run_screening_pipeline(
    as_of_date: str,
    data_dir: Path,
    universe_tickers: Optional[List[str]] = None,
    enable_coinvest: bool = False,
    enable_enhancements: bool = True,
    enable_short_interest: bool = True,
    checkpoint_dir: Optional[Path] = None,
    resume_from: Optional[str] = None,
    audit_log_path: Optional[Path] = None,
    pipeline_timeout: int = DEFAULT_PIPELINE_TIMEOUT,
    # New governance-friendly parameters
    catalyst_window: Optional[tuple[int, int]] = None,
    catalyst_decay: str = "step",
    catalyst_half_life_days: int = 30,
    enable_smart_money: bool = False,
    smart_money_source: str = "auto",
    debug_smart_money: bool = False,
    compare_snapshots: bool = False,
    snapshot_date_prior: Optional[str] = None,
    min_component_coverage: Optional[float] = None,
    min_component_coverage_mode: str = "applied",
    # Clinical activity filter parameters
    min_trials: int = 5,
    min_phase: str = "phase1",
    no_clinical_filter: bool = False,
    # Clustering parameters
    enable_clustering: bool = False,
    cluster_method: str = "indication",
    cluster_threshold: float = 0.70,
    # Defensive overlay parameters
    apply_defensive_multiplier: bool = False,
    defensive_config: str = "default",
    defensive_cache: Optional[str] = None,
    enable_position_sizing: bool = False,
    pit_mode: str = "strict",
    output_dir: Optional[Path] = None,
    # SEC 8-K live mode
    sec_8k_mode: str = "cache_only",
    # SEC multi-form (10-Q, 10-K, 6-K) mode
    sec_multi_form_mode: str = "cache_only",
    # FDA regulatory notices (Federal Register) mode
    fda_regulatory_mode: str = "cache_only",
    # Morningstar data mode: off skips vol enrichment + signal engine
    morningstar_mode: str = "on",
    # Module 5 calibrated weights
    module5_weights_path: Optional[Path] = None,
    module5_weights_mode: str = "global",
    # Scoring mode (Baker-style or default)
    scoring_mode: Optional[str] = None,
    # Override ctgov cache directory (None = auto-detect from repo; False = disable)
    ctgov_cache_dir: Optional[Any] = None,
    # Input manifest mode: "off", "write", "verify"
    inputs_manifest_mode: str = "off",
    inputs_manifest_verify_path: Optional[Path] = None,
    # True when replaying a frozen bundle: tolerate now-required deps that predate
    # the bundle's manifest (schema evolution) rather than hard-failing drift.
    from_replay_bundle: bool = False,
    phase_scores_v2: bool = False,
    phase_scores_v3: bool = True,
    phase_scores_v1: bool = False,
) -> Dict[str, Any]:
    """
    Execute full screening pipeline with deterministic guarantees.

    Args:
        as_of_date: Analysis date (YYYY-MM-DD) - REQUIRED, no defaults
        data_dir: Directory containing input data files
        universe_tickers: Optional whitelist of tickers
        enable_coinvest: Include co-invest overlay (requires coinvest_signals.json)
        enable_enhancements: Enable PoS + Regime modules (requires market_snapshot.json)
        enable_short_interest: Enable short interest signals (requires short_interest.json)
        checkpoint_dir: Directory for saving/loading checkpoints
        resume_from: Module name to resume from (e.g., "module_3")
        audit_log_path: Path to audit log file (JSONL format)
        pipeline_timeout: Maximum execution time in seconds (default: 1 hour)
        catalyst_window: Tuple (start_days, end_days) for optimal catalyst window
        catalyst_decay: Decay function ('step', 'linear', 'exp')
        catalyst_half_life_days: Half-life for exponential decay
        enable_smart_money: Enable smart money signal in composite
        smart_money_source: Smart money data source ('13f', 'internal', 'auto')
        debug_smart_money: Print detailed smart money diagnostics
        compare_snapshots: Run detailed snapshot comparison
        snapshot_date_prior: Override prior snapshot date for comparison
        min_component_coverage: Minimum component coverage for rankability
        min_component_coverage_mode: Coverage mode ('applied' or 'present')

    Returns:
        Complete screening results with provenance

    Raises:
        ValueError: If as_of_date is invalid
        FileNotFoundError: If required data files missing
        OperationTimeoutError: If pipeline exceeds timeout
        PathTraversalError: If data_dir path is suspicious
    """
    # CRITICAL: Validate as_of_date FIRST (no implicit defaults)
    validate_as_of_date_param(as_of_date)

    # Default output_dir to data_dir for backwards compatibility
    if output_dir is None:
        output_dir = data_dir

    # SECURITY: Validate data_dir exists and is a directory
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not data_dir.is_dir():
        raise ValueError(f"Data path is not a directory: {data_dir}")

    # SECURITY: Check data_dir is not a symlink to prevent traversal
    if data_dir.is_symlink():
        raise SymlinkError(f"Data directory is a symbolic link (security risk): {data_dir}")

    # RESOURCE: Check available memory before loading large data files
    try:
        require_minimum_memory(min_mb=500)
    except RuntimeError as e:
        logger.warning(f"Memory check: {e}")
        # Continue anyway but log the warning

    # PREFLIGHT: Validate env vars and data files before processing
    try:
        from common.startup_preflight import run_preflight

        preflight = run_preflight(data_dir=data_dir, check_env=True, check_files=True)
        if not preflight.ok:
            logger.error("Preflight check FAILED:\n%s", preflight.summary())
            raise RuntimeError(
                f"Preflight failed with {len(preflight.hard)} hard error(s). "
                f"Fix these before running the pipeline. Run details:\n{preflight.summary()}"
            )
    except ImportError:
        logger.debug("startup_preflight not available — skipping")

    logger.info(f"[{as_of_date}] Starting screening pipeline...")
    logger.info(f"  Data directory: {data_dir}")
    if checkpoint_dir:
        logger.info(f"  Checkpoint directory: {checkpoint_dir}")
    if resume_from:
        logger.info(f"  Resuming from: {resume_from}")

    # Determine resume index
    resume_index = get_resume_module_index(resume_from)

    # Compute content hashes for audit trail.
    # DETERMINISM: Exclude run_log files (they contain timestamps that change each run).
    content_hashes = _compute_run_input_hashes(data_dir)

    # Create audit record if audit log path provided
    if audit_log_path:
        audit_record = create_audit_record(as_of_date, data_dir, content_hashes)
        append_audit_log(audit_log_path, audit_record)
        logger.info(f"  Audit record written to: {audit_log_path}")

    # Build input manifest (before data loading for early-fail verify)
    _inputs_manifest = None
    if inputs_manifest_mode in ("write", "verify"):
        _manifest_conditions = {
            "enable_coinvest": enable_coinvest,
            "enable_enhancements": enable_enhancements,
            "decision_phase2": False,  # resolved in main(), post-patched
            "pit_mode": pit_mode != "off",
            "sec_8k": sec_8k_mode != "off",
            "fda_adcom": True,
        }
        _manifest_resolved = _resolve_cache_paths(as_of_date, data_dir, ctgov_cache_dir)
        _inputs_manifest = build_inputs_manifest(
            as_of_date,
            data_dir,
            _manifest_conditions,
            _manifest_resolved,
        )
        if inputs_manifest_mode == "verify":
            # First: basic required-file check
            if not verify_inputs_manifest(_inputs_manifest):
                raise FileNotFoundError(
                    "Input manifest verify failed: " + "; ".join(_inputs_manifest["validation"]["errors"])
                )
            # Second: drift check against prior manifest
            if not inputs_manifest_verify_path:
                raise FileNotFoundError("--inputs-manifest verify requires --inputs-manifest-path")
            if not inputs_manifest_verify_path.exists():
                raise FileNotFoundError(f"Prior inputs_manifest not found: {inputs_manifest_verify_path}")
            with open(inputs_manifest_verify_path, "r", encoding="utf-8") as _mf:
                _prior_manifest = json.load(_mf)
            drift_errors = verify_against_prior_manifest(
                _inputs_manifest,
                _prior_manifest,
                allow_new_required_deps=from_replay_bundle,
            )
            if drift_errors:
                for de in drift_errors:
                    logger.error(f"[INPUT MANIFEST DRIFT] {de}")
                raise FileNotFoundError("Input manifest drift vs prior: " + "; ".join(drift_errors))
            logger.info("[INPUT MANIFEST] Verify passed — no drift vs prior manifest")
        logger.info(
            f"[INPUT MANIFEST] Built manifest: "
            f"{len(_inputs_manifest['dependencies'])} deps, "
            f"{len(_inputs_manifest['validation']['errors'])} errors, "
            f"{len(_inputs_manifest['validation']['warnings'])} warnings"
        )

    # ==========================================================================
    # PARALLEL I/O PRELOAD: fire all 4 independent JSON reads concurrently.
    # File I/O releases the GIL, so ThreadPoolExecutor gives real parallelism.
    # _load_json_threadsafe omits SIGALRM (signal.alarm is main-thread-only).
    # Path resolution for trial_records must happen here so the trial future
    # can start alongside universe/financial/market.
    # ==========================================================================
    _preload_t0 = __import__("time").time()
    if ctgov_cache_dir is False:
        _trial_preload_cache: Optional[Path] = None
    else:
        _trial_preload_root = Path(ctgov_cache_dir) if ctgov_cache_dir else Path(__file__).parent / "cache" / "ctgov"
        _trial_preload_cache = _trial_preload_root / f"trial_records_{as_of_date}.json"
        if not _trial_preload_cache.exists():
            _trial_preload_cache = None
    _trial_preload_path: Path = (
        _trial_preload_cache if _trial_preload_cache is not None else data_dir / "trial_records.json"
    )

    _ms_preload_path = data_dir / "morningstar_price_history.json"
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as _preload_pool:
        _fut_universe = _preload_pool.submit(_load_json_threadsafe, data_dir / "universe.json", "Universe")
        _fut_financial = _preload_pool.submit(_load_json_threadsafe, data_dir / "financial_records.json", "Financial")
        _fut_trials = _preload_pool.submit(_load_json_threadsafe, _trial_preload_path, "Trials")
        _fut_market = _preload_pool.submit(_load_json_threadsafe, data_dir / "market_data.json", "Market data")
        _fut_morningstar = (
            _preload_pool.submit(_load_json_any_threadsafe, _ms_preload_path, "Morningstar")
            if _ms_preload_path.exists()
            else None
        )
        _preloaded: Dict[str, Any] = {
            "universe": _fut_universe.result(),
            "financial": _fut_financial.result(),
            "trials": _fut_trials.result(),
            "market": _fut_market.result(),
            "morningstar": _fut_morningstar.result() if _fut_morningstar is not None else None,
        }
    print(f"[timing] preload (parallel I/O): {__import__('time').time() - _preload_t0:.2f}s", flush=True)

    # Load input data
    logger.info("[1/7] Loading input data...")
    _t_load_data = __import__("time").time()
    universe_path = data_dir / "universe.json"
    logger.info(f"  Loading universe from: {universe_path}")
    logger.info(f"  Universe file exists: {universe_path.exists()}")
    if universe_path.exists():
        logger.info(f"  Universe file size: {universe_path.stat().st_size} bytes")
    raw_universe = _preloaded["universe"]
    logger.info(f"  Loaded {len(raw_universe)} records from universe.json")
    print(f"DEBUG_TRACE: Loaded {len(raw_universe)} records from {universe_path}", flush=True)
    # Pruning observability: record the loaded count so the cheap-exclusion
    # summary (delisted + PIT survivorship) can be logged in one place before
    # any expensive enrichment / live fetch / module work begins.
    _universe_loaded_n = len(raw_universe)

    # Normalize legacy field aliases (e.g. 'financials' → 'financial_data')
    # before any downstream code inspects these keys.
    for rec in raw_universe:
        normalize_financial_field_alias(rec)

    # --- Delisted filter: exclude tickers marked status="delisted" in universe.json ---
    # These are acquired/merged/OTC-moved companies with no active price feed.
    # Kept in universe.json for history; excluded from all active screening.
    _delisted_excl = [r.get("ticker") for r in raw_universe if isinstance(r, dict) and r.get("status") == "delisted"]
    if _delisted_excl:
        raw_universe = [r for r in raw_universe if not (isinstance(r, dict) and r.get("status") == "delisted")]
        logger.info(
            f"  Delisted filter: excluded {len(_delisted_excl)} tickers " f"(status=delisted): {sorted(_delisted_excl)}"
        )
        print(f"DEBUG_TRACE: Delisted filter: excluded {sorted(_delisted_excl)}", flush=True)

    # Extract full universe tickers BEFORE any filtering (for Module 3 stability)
    # This ensures Module 3 always processes the same population regardless of Module 1 filtering
    full_universe_tickers = frozenset(r.get("ticker") for r in raw_universe if r.get("ticker"))
    # Compute universe hash for state file namespacing (prevents churn on population changes)
    universe_hash = hashlib.sha256(json.dumps(sorted(full_universe_tickers)).encode()).hexdigest()[:8]
    logger.info(f"  Full universe: {len(full_universe_tickers)} tickers (hash: {universe_hash})")
    if len(raw_universe) == 0:
        logger.error(f"  ERROR: raw_universe is empty! File content may be corrupt.")
        logger.error(f"  universe_path: {universe_path}")
        if universe_path.exists():
            try:
                first_100_bytes = universe_path.read_text()[:100]
                logger.error(f"  First 100 bytes: {first_100_bytes}")
            except Exception as e:
                logger.error(f"  Could not read file: {e}")

    # --- PIT survivorship filter: exclude tickers that hadn't IPO'd or had delisted ---
    # Only applies when pit_mode is active (not "off")
    if pit_mode != "off":
        _ipo_dates_path = data_dir / "ipo_dates.json"
        if _ipo_dates_path.exists():
            with open(_ipo_dates_path, "r") as _f:
                _ipo_data = json.load(_f)
            _ipo_tickers = _ipo_data.get("tickers", {})
            _as_of = as_of_date  # already a string "YYYY-MM-DD"
            _pre_count = len(raw_universe)
            _excluded_ipo = []
            _excluded_delist = []
            _DELIST_BUFFER_DAYS = 45  # ~30 trading days ≈ 45 calendar days

            _delist_cutoff = (datetime.strptime(_as_of, "%Y-%m-%d") - timedelta(days=_DELIST_BUFFER_DAYS)).strftime(
                "%Y-%m-%d"
            )

            def _survivorship_ok(rec):
                """Return True if ticker passes PIT survivorship checks."""
                t = rec.get("ticker", "")
                if t not in _ipo_tickers:
                    return True  # No data — keep (conservative)
                dates = _ipo_tickers[t]
                first_date = dates.get("first_price_date", "")
                last_date = dates.get("last_price_date", "")
                # IPO check: first price date must be on or before as_of_date
                if first_date and first_date > _as_of:
                    _excluded_ipo.append(t)
                    return False
                # Delist check: last price date must not be too far before as_of_date
                if last_date and last_date < _delist_cutoff:
                    _excluded_delist.append(t)
                    return False
                return True

            raw_universe = [r for r in raw_universe if _survivorship_ok(r)]
            _post_count = len(raw_universe)
            print(f"DEBUG_TRACE: PIT filter: {_pre_count} → {_post_count}", flush=True)

            if _excluded_ipo:
                logger.info(
                    f"  PIT survivorship: excluded {len(_excluded_ipo)} pre-IPO tickers "
                    f"(first price after {_as_of}): {sorted(_excluded_ipo)}"
                )
            if _excluded_delist:
                logger.info(
                    f"  PIT survivorship: excluded {len(_excluded_delist)} delisted tickers "
                    f"(last price before {_delist_cutoff}): {sorted(_excluded_delist)}"
                )
            if _pre_count != _post_count:
                logger.info(
                    f"  PIT survivorship filter: {_pre_count} → {_post_count} tickers "
                    f"(removed {_pre_count - _post_count})"
                )
                # Recompute full_universe_tickers after filtering
                full_universe_tickers = frozenset(r.get("ticker") for r in raw_universe if r.get("ticker"))
        else:
            logger.info("  PIT survivorship: ipo_dates.json not found — skipping filter")

    # Validate tickers in universe (fail-loud on contamination)
    if HAS_TICKER_VALIDATION:
        universe_tickers_to_validate = [r.get("ticker") for r in raw_universe if r.get("ticker")]
        validation_result = validate_ticker_list(universe_tickers_to_validate)
        if validation_result["invalid"]:
            invalid_sample = list(validation_result["invalid"].items())[:5]
            raise ValueError(
                f"Universe contains {len(validation_result['invalid'])} invalid tickers. "
                f"Examples: {invalid_sample}. "
                f"Run: python src/scripts/clean_universe.py to fix."
            )
        logger.info(f"  Universe validation passed: {validation_result['stats']['valid_count']} valid tickers")
    # Consolidated cheap-exclusion summary (delisted + PIT survivorship) BEFORE
    # any expensive enrichment / live fetch / module work. Surfaces how many
    # names were pruned by cheap filters so stale/invalid names never silently
    # reach expensive stages. Behavior-preserving: only the existing filters run.
    _surviving_n = len(raw_universe)
    logger.info(
        f"  [prune] universe: loaded {_universe_loaded_n} → {_surviving_n} after cheap exclusions "
        f"(pruned {_universe_loaded_n - _surviving_n}) before expensive modules"
    )
    print(f"[timing] load universe: {__import__('time').time() - _t_load_data:.2f}s", flush=True)

    _t_load_financial = __import__("time").time()
    financial_records = _preloaded["financial"]

    # Financial records staleness check
    _fin_path = data_dir / "financial_records.json"
    if _fin_path.exists():
        _as_of = datetime.strptime(as_of_date, "%Y-%m-%d").date()
        _fin_age_days = (_as_of - datetime.fromtimestamp(_fin_path.stat().st_mtime).date()).days
        if _fin_age_days > 30:
            logger.warning(
                f"  financial_records.json is {_fin_age_days} days old — " f"cash/runway data may be outdated"
            )

    # --- PIT financial override: replace static financial records with point-in-time data ---
    # When pit_mode is active and PIT fact stores exist, use EDGAR filing dates
    # to reconstruct financial snapshots as they were known on as_of_date.
    _pit_fin_dir = data_dir / "pit_financials"
    if pit_mode != "off" and _pit_fin_dir.is_dir():
        from pit_financials import pit_financial_snapshot

        # Parallel preload all PIT financial JSON files (replaces 341 serial opens)
        _pit_preload_t0 = __import__("time").time()
        _pit_preloaded_data: Dict[str, dict] = {}
        _pit_paths = list(_pit_fin_dir.glob("*.json"))
        if _pit_paths:
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as _pit_pool:
                _pit_futs = {p.stem.upper(): _pit_pool.submit(_load_pit_json_worker, p) for p in _pit_paths}
                for _t, _f in _pit_futs.items():
                    try:
                        _pit_preloaded_data[_t] = _f.result()
                    except Exception:
                        pass  # fall back to file load for this ticker
            print(
                f"[timing] pit_financial preload ({len(_pit_preloaded_data)}/{len(_pit_paths)} files): "
                f"{__import__('time').time() - _pit_preload_t0:.2f}s",
                flush=True,
            )

        # Index existing static records by ticker for in-place replacement
        _fin_by_ticker = {}
        for _idx, _rec in enumerate(financial_records):
            _t = _rec.get("ticker")
            if _t:
                _fin_by_ticker[_t] = _idx

        _pit_overridden = 0
        _pit_fallback = 0
        _pit_total = len(_fin_by_ticker)

        for _ticker, _idx in _fin_by_ticker.items():
            _pit_snap = pit_financial_snapshot(
                _ticker,
                as_of_date,
                _pit_fin_dir,
                preloaded_data=_pit_preloaded_data.get(_ticker),
            )
            if _pit_snap is not None:
                financial_records[_idx] = _pit_snap
                _pit_overridden += 1
            else:
                _pit_fallback += 1

        logger.info(
            f"  PIT financials: {_pit_overridden}/{_pit_total} tickers overridden, "
            f"{_pit_fallback} fallback to static"
        )
    elif pit_mode != "off":
        logger.debug("  PIT financials: pit_financials/ directory not found — using static financial_records.json")
    print(f"[timing] load financial: {__import__('time').time() - _t_load_financial:.2f}s", flush=True)

    # Resolve trial_records: prefer PIT-filtered cache if available
    # ctgov_cache_dir=False disables cache lookup (used by tests)
    _t_load_trials = __import__("time").time()
    if ctgov_cache_dir is False:
        _ctgov_cache = None
    else:
        _cache_root = Path(ctgov_cache_dir) if ctgov_cache_dir else Path(__file__).parent / "cache" / "ctgov"
        _ctgov_cache = _cache_root / f"trial_records_{as_of_date}.json"
        if not _ctgov_cache.exists():
            _ctgov_cache = None

    if _ctgov_cache is not None:
        logger.info(f"  Using PIT-filtered CTGov cache: {_ctgov_cache.name}")
        trial_records_path = _ctgov_cache
    else:
        trial_records_path = data_dir / "trial_records.json"
        if ctgov_cache_dir is not False:
            if pit_mode == "strict":
                _expected_cache = (
                    Path(ctgov_cache_dir) if ctgov_cache_dir else Path(__file__).parent / "cache" / "ctgov"
                ) / f"trial_records_{as_of_date}.json"
                raise FileNotFoundError(
                    f"PIT cache miss: {_expected_cache} not found and --pit-mode=strict. "
                    f"Run: python warm_caches.py --sources ctgov --as-of-date {as_of_date}, "
                    f"or pass --pit-mode=degrade to fall back to {trial_records_path.name}."
                )
            logger.warning(
                f"  PIT cache miss: trial_records_{as_of_date}.json not found — "
                f"falling back to {trial_records_path.name} (may be stale, pit_mode={pit_mode})"
            )

    trial_records = _preloaded["trials"]  # already loaded by parallel preload

    # PIT safety net: if we fell back to unfiltered trial_records.json, apply
    # a runtime PIT filter so records must have been posted before the screen
    # date. Same-day CTGov posts are unavailable to historical scoring.
    if _ctgov_cache is None and ctgov_cache_dir is not False and trial_records:
        _pre_count = len(trial_records)
        trial_records, _pit_cutoff = _filter_ctgov_fallback_records_for_pit(trial_records, as_of_date)
        _post_count = len(trial_records)
        if _pre_count != _post_count:
            logger.info(
                f"  PIT fallback filter: {_pre_count} → {_post_count} trials "
                f"(removed {_pre_count - _post_count} with posting date > {_pit_cutoff})"
            )
    print(f"[timing] load trials: {__import__('time').time() - _t_load_trials:.2f}s", flush=True)

    _t_load_market = __import__("time").time()
    market_records = _preloaded["market"]

    # --- Ingestion schema validation (advisory, non-blocking) ---
    for _label, _validator, _data in [
        ("market_data", validate_market_data_schema, market_records),
        ("financial_records", validate_financial_records_schema, financial_records),
        ("trial_records", validate_trial_records_schema, trial_records),
    ]:
        try:
            _ok, _invalid = _validator(_data, strict=False, raise_on_error=False)
            if not _ok:
                logger.warning(
                    f"Schema validation ({_label}): {len(_invalid)} invalid record(s) "
                    f"— first: {_invalid[0] if _invalid else '?'}"
                )
            else:
                logger.debug(f"Schema validation ({_label}): OK ({len(_data)} records)")
        except Exception as _exc:
            logger.warning(f"Schema validation ({_label}): {_exc}")

    # Convert market_records list to dict keyed by ticker for Module 5
    # This enables volatility adjustment, momentum signal, and other enhancements
    # IMPORTANT: Normalize ticker keys to uppercase for consistent lookups
    # (Module 5 uses .upper() for ticker lookups)
    market_data_by_ticker = {}
    if market_records:
        for record in market_records:
            if isinstance(record, dict) and "ticker" in record:
                ticker = record["ticker"].upper()  # Normalize to uppercase for consistent lookups
                market_data_by_ticker[ticker] = record
        logger.info(f"  Market data indexed for {len(market_data_by_ticker)} tickers")

    # Validate delisted blacklist against live market data (advisory)
    if HAS_TICKER_VALIDATION and market_records:
        blacklist_warnings = validate_blacklist(market_records)
        if blacklist_warnings:
            logger.warning(f"Blacklist validation: {len(blacklist_warnings)} conflict(s)")
            for bw in blacklist_warnings:
                logger.warning(f"  {bw}")

    # Compute annualized volatility from Morningstar daily prices (HS377)
    # This enriches market_data_by_ticker with volatility_252d for Module 5's vol adjustment
    ms_price_path = data_dir / "morningstar_price_history.json"
    if morningstar_mode != "off" and ms_price_path.exists() and market_data_by_ticker:
        vol_enriched = enrich_volatility_from_morningstar_prices(
            price_history_path=ms_price_path,
            market_data_by_ticker=market_data_by_ticker,
            preloaded_data=_preloaded.get("morningstar"),
        )
        if vol_enriched > 0:
            logger.info(f"  Volatility (252d) computed for {vol_enriched} tickers")
    else:
        if not ms_price_path.exists():
            logger.info("  Morningstar price history not found, skipping volatility computation")

    # Compute momentum returns from price history
    # This enriches market_data_by_ticker with return_20d, return_60d, return_120d and XBI benchmarks
    price_history_path = data_dir / "price_history.csv"
    if price_history_path.exists() and market_data_by_ticker:
        momentum_enriched = compute_momentum_from_price_history(
            price_history_path=price_history_path,
            as_of_date=as_of_date,
            market_data_by_ticker=market_data_by_ticker,
        )
        if momentum_enriched > 0:
            logger.info(f"  Momentum data computed for {momentum_enriched} tickers")
    else:
        if not price_history_path.exists():
            logger.info("  Price history not found, skipping momentum computation")
    print(f"[timing] load market + enrich: {__import__('time').time() - _t_load_market:.2f}s", flush=True)

    coinvest_signals = None
    coinvest_audit_data = {}  # PIT audit counters for run_metadata
    raw_holdings_for_baker = None  # Preserved for Baker overlay post-processing
    holdings_schema_data = None  # Schema info for run_metadata
    if enable_coinvest:
        # Try loading coinvest_signals.json first
        coinvest_file = data_dir / "coinvest_signals.json"
        if coinvest_file.exists():
            logger.info("  Loading co-invest signals...")
            # Accept three shapes for backward compatibility:
            #   1. dict {TICKER: {...}}                        — pre-computed signals
            #   2. wrapped dict {"tickers": {TICKER: {...}}}    — builder output (coinvest_features.v1)
            #   3. list [{"ticker": "X", ...}, ...]             — legacy list format
            # Downstream consumers (module_5_composite._enrich_with_coinvest) expect a dict.
            with open(coinvest_file, "r", encoding="utf-8") as _cf:
                _raw = json.load(_cf)
            if isinstance(_raw, dict):
                coinvest_signals = _raw.get("tickers", _raw)
            elif isinstance(_raw, list):
                coinvest_signals = {
                    (entry.get("ticker") or "").upper(): entry
                    for entry in _raw
                    if isinstance(entry, dict) and entry.get("ticker")
                }
            else:
                raise ValueError(f"coinvest_signals.json must be a dict or list, got {type(_raw).__name__}")
        else:
            # Fallback: try loading holdings file and convert to coinvest format
            # Prefer holdings_detailed.json (dict format with conviction data) over holdings_snapshots.json (list format)
            holdings_file = data_dir / "holdings_detailed.json"
            if not holdings_file.exists():
                holdings_file = data_dir / "holdings_snapshots.json"
            if holdings_file.exists():
                logger.info(f"  Loading {holdings_file.name} for smart money signals...")
                with open(holdings_file, "r", encoding="utf-8") as f:
                    holdings_payload = json.load(f)

                coinvest_audit = {}
                holdings_schema = None
                holdings_snapshots = holdings_payload

                # holdings_detailed.json is expected to be a wrapper: {"_schema":..., "tickers": {...}}
                if holdings_file.name == "holdings_detailed.json":
                    if not (isinstance(holdings_payload, dict) and "tickers" in holdings_payload):
                        raise RuntimeError(
                            "holdings_detailed.json present but wrong shape: expected top-level 'tickers'."
                        )
                    holdings_schema = holdings_payload.get("_schema", {})
                    holdings_snapshots = holdings_payload["tickers"]
                    _validate_holdings_detailed_payload(holdings_schema, holdings_snapshots)
                    logger.info(f"  Detected nested 'tickers' key - extracting {len(holdings_snapshots)} tickers")
                # Legacy extractor format: allow nested tickers for non-detailed files (non-fatal)
                elif isinstance(holdings_snapshots, dict) and "tickers" in holdings_snapshots:
                    logger.info(
                        f"  Detected nested 'tickers' key - extracting {len(holdings_snapshots['tickers'])} tickers"
                    )
                    holdings_snapshots = holdings_snapshots["tickers"]

                # Preserve raw holdings for Baker overlay (before coinvest conversion)
                if holdings_snapshots and isinstance(holdings_snapshots, dict):
                    raw_holdings_for_baker = holdings_snapshots

                if holdings_snapshots and isinstance(holdings_snapshots, dict):
                    # Dict format: {"TICKER": {"holdings": {...}}} - use full conversion with conviction
                    coinvest_signals = _convert_holdings_to_coinvest(
                        holdings_snapshots, data_dir, as_of_date, audit_out=coinvest_audit
                    )
                    logger.info(
                        f"  Converted holdings (dict format) to coinvest signals for {len(coinvest_signals)} tickers"
                    )
                    # Log conviction stats
                    conviction_tickers = sum(1 for v in coinvest_signals.values() if v.get("conviction_overlap", 0) > 0)
                    logger.info(f"  Conviction overlap computed for {conviction_tickers} tickers")
                    # Store audit + schema for later persistence to run_metadata
                    coinvest_audit_data.update(coinvest_audit)
                    if holdings_schema is not None:
                        holdings_schema_data = holdings_schema
                elif holdings_snapshots and isinstance(holdings_snapshots, list):
                    # List format: [{"ticker": "X", "tier1_count": N, ...}] - direct mapping
                    coinvest_signals = {}
                    for entry in holdings_snapshots:
                        ticker = entry.get("ticker", "").upper()
                        if not ticker:
                            continue
                        tier1_count = entry.get("tier1_count", 0)
                        tier2_count = entry.get("tier2_count", 0)
                        manager_count = entry.get("manager_count", 0)

                        # Build holder_tiers dict: tier1 holders get tier=1, rest get tier=2
                        # Since we don't have individual holder names, use placeholder names
                        holder_tiers = {}
                        holders = []
                        for i in range(tier1_count):
                            name = f"Tier1_Holder_{i+1}"
                            holder_tiers[name] = 1
                            holders.append(name)
                        for i in range(tier2_count):
                            name = f"Tier2_Holder_{i+1}"
                            holder_tiers[name] = 2
                            holders.append(name)

                        coinvest_signals[ticker] = {
                            "coinvest_overlap_count": tier1_count,  # tier1 = overlap
                            "coinvest_holders": holders,
                            "holder_tiers": holder_tiers,
                            "position_changes": {},  # No change data in list format
                            "tier1_count": tier1_count,
                            "tier2_count": tier2_count,
                            "manager_count": manager_count,
                            "total_value": entry.get("total_value", 0),
                            "total_shares": entry.get("total_shares", 0),
                        }
                    logger.info(
                        f"  Converted holdings (list format) to coinvest signals for {len(coinvest_signals)} tickers"
                    )
                    tier1_tickers = sum(1 for v in coinvest_signals.values() if v.get("tier1_count", 0) > 0)
                    logger.info(
                        f"  Tier-1 coverage: {tier1_tickers}/{len(coinvest_signals)} tickers have tier-1 holders"
                    )
            else:
                logger.warning(
                    f"  --enable-coinvest specified but neither coinvest_signals.json nor holdings_snapshots.json found"
                )

    # Enhancement data (optional)
    market_snapshot = None
    short_interest_data = None
    regime_result = None  # Default; set by regime engine if enhancements enabled

    if enable_enhancements or enable_short_interest:
        if not HAS_ENHANCEMENTS:
            logger.warning("  Enhancement modules not available (import failed)")
            enable_enhancements = False
            enable_short_interest = False

    if enable_enhancements:
        # Market snapshot for regime detection (VIX, XBI vs SPY, Fed rates)
        snapshot_file = data_dir / "market_snapshot.json"
        if snapshot_file.exists():
            logger.info("  Loading market snapshot for regime detection...")
            with open(snapshot_file, "r", encoding="utf-8") as f:
                market_snapshot = json.load(f)

            # Regime-input preflight — OBSERVABILITY ONLY.
            #
            # tools/check_market_snapshot_freshness.py was fully implemented and
            # tested but wired into nothing, so the regime input ran unguarded.
            # This closes the observability half: the check now runs against the
            # real snapshot on every production run and reports staleness or a
            # wholesale feed failure.
            #
            # market_snapshot is deliberately NOT nullified. Nullifying it would
            # move regime detection and therefore the regime pruner, which is
            # frozen under the DEM candidate freeze / NO_MODEL_CHANGE window, and
            # any authorized behaviour change resets the out-of-sample clock
            # (docs/FORWARD_VALIDATION_PROTOCOL.md §1). The nullification half
            # stays pending an explicit operator freeze lift; see
            # tests/test_market_snapshot_preflight_wire.py.
            regime_input_diagnostic = _check_snapshot_freshness(snapshot_file, reference_date=as_of_date)
            _emit_snapshot_diagnostic(regime_input_diagnostic)
            if not regime_input_diagnostic.get("ok"):
                logger.warning(
                    "  REGIME_INPUT_STALE_OR_INVALID: %s",
                    "; ".join(regime_input_diagnostic.get("issues", [])) or "no detail",
                )
                logger.warning("  Observability only — market_snapshot NOT nullified (frozen behaviour).")
        else:
            logger.warning(f"  --enable-enhancements specified but {snapshot_file} not found")
            logger.warning("  Regime detection will use UNKNOWN regime with neutral weights")

    if enable_enhancements or enable_short_interest:
        # Short interest data
        si_file = data_dir / "short_interest.json"
        if si_file.exists():
            logger.info("  Loading short interest data...")
            short_interest_data = load_json_data(si_file, "Short interest")
        else:
            logger.info("  Short interest data not found, will skip SI signals")

    # 13F institutional momentum data (coordinated buys/sells among holders)
    # This enhances the smart money signal with momentum detection
    institutional_momentum = None
    momentum_file = data_dir / "momentum_results.json"
    if momentum_file.exists():
        logger.info("  Loading 13F institutional momentum data...")
        with open(momentum_file, "r", encoding="utf-8") as f:
            institutional_momentum = json.load(f)
        logger.info(f"  Institutional momentum: {len(institutional_momentum.get('rankings', []))} tickers scored")
    else:
        # Also check root directory
        root_momentum_file = Path("momentum_results.json")
        if root_momentum_file.exists():
            logger.info("  Loading 13F institutional momentum data from root...")
            with open(root_momentum_file, "r", encoding="utf-8") as f:
                institutional_momentum = json.load(f)
            logger.info(f"  Institutional momentum: {len(institutional_momentum.get('rankings', []))} tickers scored")

    # Inject 13F institutional momentum into market_data_by_ticker as pseudo "return_60d"
    # This allows Module 5 v3 to use institutional momentum as a price momentum proxy
    # Conversion: momentum_score (0-100) -> return_60d as (score - 50) / 100
    # e.g., score=75 -> +0.25, score=50 -> 0, score=25 -> -0.25
    if institutional_momentum and market_data_by_ticker:
        injected_count = 0
        coordinated_buys = set(institutional_momentum.get("summary", {}).get("coordinated_buys", []))
        coordinated_sells = set(institutional_momentum.get("summary", {}).get("coordinated_sells", []))

        for ranking in institutional_momentum.get("rankings", []):
            ticker = ranking.get("ticker")
            momentum_score = ranking.get("momentum_score")

            # CRITICAL: Use safe_numeric_check to handle momentum_score=0 correctly
            # The pattern `if momentum_score` would incorrectly skip valid zero scores
            if ticker and safe_numeric_check(momentum_score) and ticker in market_data_by_ticker:
                try:
                    score = float(momentum_score)
                    # Convert to pseudo-return: (score - 50) / 100
                    pseudo_return = (score - 50) / 100

                    # Inject into market data (only if return_60d not already set)
                    if market_data_by_ticker[ticker].get("return_60d") is None:
                        market_data_by_ticker[ticker]["return_60d"] = pseudo_return
                        # Also set benchmark to 0 so alpha = return (relative strength)
                        market_data_by_ticker[ticker]["xbi_return_60d"] = 0.0
                        # Mark source as 13F (for observability - distinguishes from prices)
                        market_data_by_ticker[ticker]["_momentum_source"] = "13f"
                        # Add flags for coordinated activity
                        if ticker in coordinated_buys:
                            market_data_by_ticker[ticker]["_13f_coordinated_buy"] = True
                        if ticker in coordinated_sells:
                            market_data_by_ticker[ticker]["_13f_coordinated_sell"] = True
                        injected_count += 1
                except (ValueError, TypeError):
                    pass

        if injected_count > 0:
            logger.info(f"  Injected 13F momentum for {injected_count} tickers into market data")

    # Debug smart money if requested (before Module 1 so we can see pre-filter state)
    if debug_smart_money:
        # We'll call this again after Module 1 with active_tickers
        log_smart_money_debug(
            institutional_momentum=institutional_momentum,
            market_data_by_ticker=market_data_by_ticker,
            active_tickers=list(full_universe_tickers),  # Use full universe initially
            data_dir=data_dir,
            smart_money_source=smart_money_source,
        )

    # Module 1: Universe filtering
    logger.info("[2/7] Module 1: Universe filtering...")
    _t_m1 = __import__("time").time()
    m1_result = None
    if resume_index > 0 and checkpoint_dir:
        m1_result = load_checkpoint(checkpoint_dir, "module_1", as_of_date)

    if m1_result is None:
        m1_result = compute_module_1_universe(
            raw_records=raw_universe,
            as_of_date=as_of_date,  # Explicit threading
            universe_tickers=universe_tickers,
        )
        if checkpoint_dir:
            save_checkpoint(checkpoint_dir, "module_1", as_of_date, m1_result)

    # Validate Module 1 output schema
    validate_pipeline_handoff("module_1", "module_2", m1_result)

    active_tickers = [s["ticker"] for s in m1_result["active_securities"]]
    logger.info(f"  Active: {len(active_tickers)}, Excluded: {len(m1_result['excluded_securities'])}")
    print(f"[timing] module_1: {__import__('time').time() - _t_m1:.2f}s", flush=True)

    # Module 2: Financial health
    logger.info("[3/7] Module 2: Financial health...")
    _t_m2 = __import__("time").time()
    m2_result = None
    if resume_index > 1 and checkpoint_dir:
        m2_result = load_checkpoint(checkpoint_dir, "module_2", as_of_date)

    if m2_result is None:
        m2_result = compute_module_2_financial(
            financial_records=financial_records,
            active_tickers=set(active_tickers),
            as_of_date=as_of_date,
            raw_universe=raw_universe,
            market_records=market_records,
        )
        if checkpoint_dir:
            save_checkpoint(checkpoint_dir, "module_2", as_of_date, m2_result)

    # Validate Module 2 output schema
    validate_pipeline_handoff("module_2", "module_5", m2_result)

    diag = m2_result.get("diagnostic_counts", {})
    logger.info(
        f"  Scored: {diag.get('scored', len(m2_result.get('scores', [])))}, " f"Missing: {diag.get('missing', 'N/A')}"
    )
    print(f"[timing] module_2: {__import__('time').time() - _t_m2:.2f}s", flush=True)

    # ========================================================================
    # Module 3: Catalyst Detection (NEW: Delta-based event detection)
    # ========================================================================

    logger.info("[4/7] Module 3: Catalyst detection...")
    _t_m3 = __import__("time").time()
    m3_result = None
    if resume_index > 2 and checkpoint_dir:
        m3_result = load_checkpoint(checkpoint_dir, "module_3", as_of_date)

    if m3_result is None:
        # Convert as_of_date string to date object for Module 3
        as_of_date_obj = to_date_object(as_of_date)

        # Create state directory namespaced by universe hash
        # This prevents churn gate from firing when Module 1 filtering changes
        # because snapshots from different universe populations are kept separate
        state_dir = output_dir / "ctgov_state" / universe_hash
        state_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"  Using state directory: {state_dir.name} (universe hash)")

        # Run snapshot comparison if requested
        if compare_snapshots:
            logger.info("  Running snapshot comparison (--compare-snapshots)...")
            snapshot_comparison = compare_snapshots_detailed(
                state_dir=state_dir,
                current_date=as_of_date,
                prior_date=snapshot_date_prior,
            )
            logger.info("  " + "=" * 56)
            logger.info("  SNAPSHOT COMPARISON RESULTS")
            logger.info("  " + "=" * 56)
            logger.info(f"  Current: {snapshot_comparison['current_file']}")
            logger.info(f"  Prior:   {snapshot_comparison['prior_file']}")
            logger.info(f"  Status:  {snapshot_comparison['comparison_status']}")

            if snapshot_comparison["comparison_status"] in ("OK", "IDENTICAL"):
                logger.info(f"  Current SHA256: {snapshot_comparison['current_sha256'][:16]}...")
                logger.info(f"  Prior SHA256:   {snapshot_comparison['prior_sha256'][:16]}...")
                logger.info(f"  Current lines:  {snapshot_comparison['current_line_count']}")
                logger.info(f"  Prior lines:    {snapshot_comparison['prior_line_count']}")

                if snapshot_comparison["files_identical"]:
                    logger.warning("  ⚠️  Files are IDENTICAL (same SHA256)")
                else:
                    logger.info(f"  Records added:   {snapshot_comparison['records_added']}")
                    logger.info(f"  Records removed: {snapshot_comparison['records_removed']}")
                    logger.info(f"  Field changes:   {snapshot_comparison['field_changes_detected']}")
                    if snapshot_comparison["top_changed_fields"]:
                        logger.info(f"  Top changed fields: {snapshot_comparison['top_changed_fields']}")
            logger.info("  " + "=" * 56)

        # Run Module 3A with FULL UNIVERSE tickers (not post-Module-1 filtered)
        # This ensures stable delta comparisons regardless of Module 1 filtering
        m3_result = compute_module_3_catalyst(
            trial_records_path=trial_records_path,  # Resolved above (PIT cache or production)
            trial_records=trial_records,  # pass pre-loaded list; avoids re-parsing 15MB file
            state_dir=state_dir,  # State directory for snapshots (namespaced by universe)
            active_tickers=full_universe_tickers,  # FULL universe, not filtered active_tickers
            as_of_date=as_of_date_obj,  # Date object
            market_calendar=SimpleMarketCalendar(),  # Market calendar for weekends
            config=_build_m3_config(
                sec_8k_mode, sec_multi_form_mode, fda_regulatory_mode, ctgov_cache_dir=ctgov_cache_dir
            ),
            output_dir=output_dir,  # Output directory for catalyst_events_*.json
            pit_mode=pit_mode,
        )
        if checkpoint_dir:
            save_checkpoint(checkpoint_dir, "module_3", as_of_date, m3_result)

    # Validate Module 3 output schema
    validate_pipeline_handoff("module_3", "module_5", m3_result)

    # Extract results (summaries is already a dict keyed by ticker)
    catalyst_summaries = m3_result["summaries"]
    diag3 = m3_result.get("diagnostic_counts", {})

    # Print diagnostics
    logger.info(
        f"  Events detected: {diag3.get('events_detected_total', 0)}, "
        f"Tickers with events: {diag3.get('tickers_with_events', 0)}/{diag3.get('tickers_analyzed', 0)}, "
        f"Severe negatives: {diag3.get('tickers_with_severe_negative', 0)}"
    )

    # Log PIT violation if detected
    pit_violation = m3_result.get("pit_violation")
    if pit_violation:
        logger.error(
            f"  PIT VIOLATION: {pit_violation['source']} dated {pit_violation['source_date']} > as_of_date {pit_violation['as_of_date']}"
        )
        logger.error(f"  Catalyst mode: {m3_result.get('catalyst_mode', 'unknown')}")

    # ========================================================================
    # End Module 3
    # ========================================================================
    print(f"[timing] module_3: {__import__('time').time() - _t_m3:.2f}s", flush=True)

    # Module 4: Clinical development
    logger.info("[5/7] Module 4: Clinical development...")
    _t_m4 = __import__("time").time()
    m4_result = None
    if resume_index > 3 and checkpoint_dir:
        m4_result = load_checkpoint(checkpoint_dir, "module_4", as_of_date)

    # Check for pre-computed clinical feature cache (same pattern as CTGov cache)
    _clinical_cache = Path(__file__).parent / "cache" / "clinical" / f"clinical_features_{as_of_date}.json"
    if m4_result is None and _clinical_cache.exists():
        logger.info(f"  Using cached clinical features: {_clinical_cache.name}")
        _cached = json.loads(_clinical_cache.read_text(encoding="utf-8"))
        m4_result = {
            "as_of_date": _cached["as_of_date"],
            "scores": _cached["scores"],
            "diagnostic_counts": {
                "scored": _cached["diagnostic_counts"].get("tickers", 0),
                "total_trials_raw": _cached["diagnostic_counts"].get("trials_raw", 0),
                "total_trials_unique": _cached["diagnostic_counts"].get("trials_after_pit", 0),
                "total_trials": _cached["diagnostic_counts"].get("trials_after_pit", 0),
                "pit_filtered": 0,
                "pit_fields_used": {},
            },
            "provenance": _cached.get("provenance", {}).get("module4_provenance", {}),
        }

    if m4_result is None:
        _m4_phase_scores = None
        if phase_scores_v1:
            _m4_phase_scores = None  # Wong defaults
            logger.info("[M4] Using PHASE_SCORES_V1 (Wong reference, explicit rollback)")
        elif phase_scores_v3:
            from module_4_clinical_dev import PHASE_SCORES_V3

            _m4_phase_scores = PHASE_SCORES_V3
            logger.info("[M4] Using PHASE_SCORES_V3 (Spec 024, default-on, survivorship-adjusted)")
        elif phase_scores_v2:
            from module_4_clinical_dev import PHASE_SCORES_V2

            _m4_phase_scores = PHASE_SCORES_V2
            logger.info("[M4] Using PHASE_SCORES_V2 (Spec 023, ±2 capped)")
        m4_result = compute_module_4_clinical_dev(
            trial_records=trial_records,
            active_tickers=active_tickers,
            as_of_date=as_of_date,  # Explicit threading
            phase_scores=_m4_phase_scores,
        )
        if checkpoint_dir:
            save_checkpoint(checkpoint_dir, "module_4", as_of_date, m4_result)

    # Validate Module 4 output schema
    validate_pipeline_handoff("module_4", "module_5", m4_result)

    diag = m4_result.get("diagnostic_counts", {})
    logger.info(
        f"  Scored: {diag.get('scored', len(m4_result.get('scores', [])))}, "
        f"Trials evaluated: {diag.get('total_trials', 'N/A')}, "
        f"PIT filtered: {diag.get('pit_filtered', 'N/A')}"
    )
    print(f"[timing] module_4: {__import__('time').time() - _t_m4:.2f}s", flush=True)

    # ========================================================================
    # Company Archetype Classification
    # ========================================================================
    # Build archetype map from industry (market_data) and revenue (Module 2) signals.
    # Non-drug-developer archetypes are exempt from the clinical trial gate.
    m2_scores_by_ticker = {s["ticker"]: s for s in m2_result.get("scores", []) if "ticker" in s}
    company_archetypes = {}
    for ticker in active_tickers:
        md = market_data_by_ticker.get(ticker, {})
        industry = md.get("industry", "Biotechnology")
        m2 = m2_scores_by_ticker.get(ticker, {})
        has_revenue = m2.get("has_revenue", False)
        revenue_scale_bucket = m2.get("revenue_scale_bucket", "pre_revenue")
        company_archetypes[ticker] = classify_company_archetype(
            ticker=ticker,
            industry=industry,
            has_revenue=has_revenue,
            revenue_scale_bucket=revenue_scale_bucket,
        )

    # Log archetype distribution
    archetype_counts = {}
    for arch in company_archetypes.values():
        archetype_counts[arch] = archetype_counts.get(arch, 0) + 1
    logger.info(f"  Company archetypes: {dict(sorted(archetype_counts.items()))}")

    # ========================================================================
    # Morningstar Industry Group Classification
    # ========================================================================
    # Load PIT-stamped industry classifications from cache.
    # Falls back to Yahoo→Morningstar mapping if cache unavailable.
    from wake_robin_data_pipeline.morningstar_industry_provider import (
        YAHOO_TO_MSTAR_INDUSTRY_GROUP,
        load_industry_classifications,
    )

    industry_groups: Dict[str, str] = load_industry_classifications(as_of_date)
    if not industry_groups:
        # Fallback: derive from Yahoo industry in market_data
        for ticker in active_tickers:
            md = market_data_by_ticker.get(ticker, {})
            yahoo_ind = md.get("industry", "")
            mstar_group = YAHOO_TO_MSTAR_INDUSTRY_GROUP.get(yahoo_ind, "")
            if mstar_group:
                industry_groups[ticker] = mstar_group
        logger.info(
            "  Industry groups: %d from Yahoo fallback (no Morningstar cache for %s)",
            len(industry_groups),
            as_of_date,
        )
    else:
        logger.info("  Industry groups: %d from Morningstar cache", len(industry_groups))

    # ========================================================================
    # Clinical Activity Filter (excludes low-activity tickers from ranking)
    # ========================================================================
    clinical_exclusions = []
    clinical_exemptions = []
    if not no_clinical_filter:
        excluded_tickers, exclusion_details, exemption_details = apply_clinical_activity_filter(
            m4_scores=m4_result.get("scores", []),
            min_trials=min_trials,
            min_phase=min_phase,
            archetypes=company_archetypes,
        )
        clinical_exclusions = exclusion_details
        clinical_exemptions = exemption_details

        if excluded_tickers or exemption_details:
            logger.info(
                f"[5.1/7] Clinical activity filter: excluding {len(excluded_tickers)} tickers, "
                f"exempting {len(exemption_details)} non-drug-developers "
                f"(min_trials={min_trials}, min_phase={min_phase})"
            )

            # Remove excluded tickers from active_tickers
            active_tickers = [t for t in active_tickers if t not in set(excluded_tickers)]

            # Log sample exclusions
            for ex in exclusion_details[:5]:
                logger.info(f"    Excluded {ex['ticker']}: {ex['details']}")
            if len(exclusion_details) > 5:
                logger.info(f"    ... and {len(exclusion_details) - 5} more")

            # Log sample exemptions
            for ex in exemption_details[:5]:
                logger.info(f"    Exempt {ex['ticker']}: {ex['archetype']} (trials={ex['trial_count']})")
            if len(exemption_details) > 5:
                logger.info(f"    ... and {len(exemption_details) - 5} more exemptions")
    else:
        logger.info("[5.1/7] Clinical activity filter: DISABLED (--no-clinical-filter)")

    # ========================================================================
    # Enhancement Layer: PoS, Short Interest, Regime Detection
    # ========================================================================
    enhancement_result = None

    # Run enhancement layer if enhancements enabled OR if short interest enabled with data
    def _safe_run_engine(engine_name: str, fn, *args, **kwargs):
        """Run an enhancement engine with error protection.

        Returns the engine result on success, or None on failure.
        Logs the error at WARNING level so the pipeline can continue.
        """
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            logger.warning("Enhancement engine '%s' failed: %s — skipping", engine_name, exc)
            return None

    should_run_enhancements = (enable_enhancements and HAS_ENHANCEMENTS) or (
        enable_short_interest and short_interest_data
    )

    if should_run_enhancements:
        if enable_enhancements and HAS_ENHANCEMENTS:
            logger.info("[5.5/7] Enhancement Layer: PoS + Regime + Short Interest...")
        else:
            logger.info("[5.5/7] Enhancement Layer: Short Interest only...")

        # Check for checkpoint
        if resume_index > 4 and checkpoint_dir:
            enhancement_result = load_checkpoint(checkpoint_dir, "enhancements", as_of_date)

        if enhancement_result is None:
            as_of_date_obj = to_date_object(as_of_date)

            # Initialize engines (PoS/Regime only if full enhancements enabled)
            pos_engine = None
            regime_engine = None
            si_engine = None
            if enable_enhancements and HAS_ENHANCEMENTS:
                try:
                    pos_engine = ProbabilityOfSuccessEngine()
                except Exception as exc:
                    logger.warning("PoS engine init failed: %s — skipping", exc)
                try:
                    regime_engine = RegimeDetectionEngine()
                except Exception as exc:
                    logger.warning("Regime engine init failed: %s — skipping", exc)
            if short_interest_data:
                try:
                    si_engine = ShortInterestSignalEngine()
                except Exception as exc:
                    logger.warning("SI engine init failed: %s — skipping", exc)

            # Step 1: Detect market regime with enhanced macro data
            regime_result = None
            macro_snapshot = None

            # Collect live macro data (yield curve, HY spread, fund flows) if available
            if regime_engine and HAS_MACRO_COLLECTOR:
                try:
                    logger.info("  Collecting macro data (FRED + yfinance)...")
                    macro_collector = MacroDataCollector()
                    macro_snapshot = macro_collector.collect_snapshot(as_of_date_obj)
                    macro_params = macro_collector.to_regime_engine_params(macro_snapshot)

                    logger.info(f"    Yield curve: {macro_snapshot.yield_curve_slope_bps} bps")
                    logger.info(f"    HY spread: {macro_snapshot.hy_credit_spread_bps} bps")
                    logger.info(f"    Biotech flows (XBI+IBB): {macro_snapshot.biotech_fund_flows_mm} MM")
                    logger.info(f"    Data quality: {macro_snapshot.data_quality.get('completeness', 'N/A')}")
                except Exception as e:
                    logger.warning(f"  Macro data collection failed: {e}")
                    macro_params = {}
            else:
                macro_params = {}

            if regime_engine and market_snapshot:
                try:
                    # Extract data date from provenance for staleness check
                    data_date_str = None
                    if "provenance" in market_snapshot:
                        data_date_str = market_snapshot["provenance"].get("as_of_date")
                    data_as_of_date = date.fromisoformat(data_date_str) if data_date_str else None

                    # Enhanced regime detection with macro data
                    regime_result = regime_engine.detect_regime(
                        vix_current=Decimal(str(market_snapshot.get("vix", "20"))),
                        xbi_vs_spy_30d=Decimal(str(market_snapshot.get("xbi_vs_spy_30d", "0"))),
                        fed_rate_change_3m=(
                            Decimal(str(market_snapshot.get("fed_rate_change_3m", "0")))
                            if market_snapshot.get("fed_rate_change_3m") is not None
                            else None
                        ),
                        as_of_date=as_of_date_obj,
                        data_as_of_date=data_as_of_date,
                        **macro_params,  # Include yield curve, HY spread, fund flows
                    )

                    # Log regime with staleness info if applicable
                    staleness = regime_result.get("staleness")
                    indicators = regime_result.get("indicators", {})
                    if staleness and staleness.get("is_stale"):
                        logger.warning(f"  Regime: UNKNOWN (data stale: {staleness['age_days']} days old)")
                    elif staleness and staleness.get("action") == "HAIRCUT_APPLIED":
                        logger.info(
                            f"  Regime: {regime_result['regime']} (confidence: {regime_result['confidence']}, {staleness['age_days']}d stale → {staleness['haircut_multiplier']}x haircut)"
                        )
                    else:
                        logger.info(f"  Regime: {regime_result['regime']} (confidence: {regime_result['confidence']})")

                    # Log enhanced indicators if macro data was used
                    if macro_params:
                        logger.info(f"    Yield curve state: {indicators.get('yield_curve_state', 'N/A')}")
                        logger.info(f"    Credit environment: {indicators.get('credit_environment', 'N/A')}")
                        logger.info(f"    Fund flow state: {indicators.get('fund_flow_state', 'N/A')}")
                except Exception as e:
                    logger.warning(f"  Regime detection failed: {e}")
                    regime_result = {"regime": "UNKNOWN", "signal_adjustments": {}}
            elif regime_engine:
                regime_result = {"regime": "UNKNOWN", "signal_adjustments": {}}
                logger.info("  Regime: UNKNOWN (no market snapshot)")
            else:
                regime_result = {"regime": "UNKNOWN", "signal_adjustments": {}}

            # Step 2: Calculate PoS scores for each ticker (only if pos_engine available)
            pos_result = None
            if pos_engine:
                # Build universe data from trial records (extract stage and indication)
                pos_universe = []
                ticker_stage_map = {}  # ticker -> {stage, indication}

                # Build stage map from clinical results (including design score for PoS)
                for clinical_score in m4_result.get("scores", []):
                    ticker = clinical_score.get("ticker")
                    if ticker:
                        # Extract design_score and normalize to 0.7-1.3 multiplier for PoS
                        # design_score is 0-25 scale; 12 = baseline (1.0x)
                        raw_design = Decimal(str(clinical_score.get("design_score", "12")))
                        # Linear mapping: 0 -> 0.7, 12 -> 1.0, 25 -> 1.3
                        design_quality = Decimal("0.7") + (raw_design / Decimal("25")) * Decimal("0.6")
                        design_quality = max(Decimal("0.7"), min(Decimal("1.3"), design_quality))

                        # Extract pipeline metrics for commercial-stage differentiation
                        pipeline_trial_count = clinical_score.get("n_trials_unique", 0)
                        # Estimate phase diversity from phase_progress (0-5 scale maps to phases)
                        # Note: phase_progress may be a string like "3.5", convert via float first
                        phase_progress_raw = clinical_score.get("phase_progress", 0) or 0
                        phase_progress = int(float(phase_progress_raw))
                        # Phase diversity: if progress >= 3, likely have trials in multiple phases
                        pipeline_phase_diversity = min(phase_progress, 5) if phase_progress else 1

                        ticker_stage_map[ticker] = {
                            "base_stage": clinical_score.get("lead_phase", "phase_2"),
                            "indication": clinical_score.get("lead_indication"),
                            "trial_design_quality": design_quality,
                            "pipeline_trial_count": pipeline_trial_count,
                            "pipeline_phase_diversity": pipeline_phase_diversity,
                        }

                # Use IndicationMapper to auto-detect indications from trial conditions
                indication_mapper = IndicationMapper()
                indication_map = indication_mapper.map_universe(
                    tickers=active_tickers, trial_records=trial_records, as_of_date=as_of_date_obj
                )
                logger.info(f"  Indication mapping: {len(indication_map)} tickers mapped")

                for ticker in active_tickers:
                    stage_info = ticker_stage_map.get(ticker, {"base_stage": "phase_2"})
                    # Get indication from mapper (auto-detected from conditions)
                    mapped_indication = indication_map.get(ticker.upper(), {}).get("indication")
                    # Fall back to clinical data if mapper returns None
                    final_indication = mapped_indication or stage_info.get("indication")
                    pos_universe.append(
                        {
                            "ticker": ticker,
                            "base_stage": stage_info.get("base_stage", "phase_2"),
                            "indication": final_indication,
                            "trial_design_quality": stage_info.get("trial_design_quality"),
                            "pipeline_trial_count": stage_info.get("pipeline_trial_count", 0),
                            "pipeline_phase_diversity": stage_info.get("pipeline_phase_diversity", 1),
                        }
                    )

                pos_result = _safe_run_engine("PoS", pos_engine.score_universe, pos_universe, as_of_date_obj)
                if pos_result is None:
                    logger.info("  PoS: skipped (engine error)")
                else:
                    pos_diag = pos_result["diagnostic_counts"]
                    conf_dist = pos_diag.get("confidence_distribution", {})
                    logger.info(
                        f"  PoS: mapped={pos_diag['indication_coverage_pct']} | "
                        f"effective(>=0.40)={pos_diag.get('effective_coverage_pct', 'N/A')} | "
                        f"conf(H/M/L)={conf_dist.get('high', 0)}/{conf_dist.get('medium', 0)}/{conf_dist.get('low', 0)}"
                    )

            # Step 3: Calculate short interest signals (if data available)
            si_result = None
            if si_engine and short_interest_data:
                # Build SI universe from short interest data
                si_map = {r.get("ticker"): r for r in short_interest_data}
                si_universe = []
                for ticker in active_tickers:
                    si_info = si_map.get(ticker, {})
                    si_universe.append(
                        {
                            "ticker": ticker,
                            "short_interest_pct": si_info.get("short_interest_pct"),
                            "days_to_cover": si_info.get("days_to_cover"),
                            "short_interest_change_pct": si_info.get("short_interest_change_pct"),
                            "institutional_long_pct": si_info.get("institutional_long_pct"),
                        }
                    )

                si_result = _safe_run_engine("ShortInterest", si_engine.score_universe, si_universe, as_of_date_obj)
                if si_result:
                    logger.info(
                        f"  SI scored: {si_result['diagnostic_counts']['total_scored']}, "
                        f"Coverage: {si_result['diagnostic_counts']['data_coverage_pct']}"
                    )

            # Step 4: Calculate accuracy enhancements (if available)
            accuracy_result = None
            if HAS_ACCURACY_ENHANCEMENTS:
                try:
                    accuracy_adapter = AccuracyEnhancementsAdapter()

                    # Build trial data map from trial records
                    trial_data_map = {}
                    for trial in trial_records:
                        ticker = trial.get("lead_sponsor_ticker") or trial.get("ticker")
                        if ticker:
                            ticker = ticker.upper()
                            if ticker not in trial_data_map:
                                trial_data_map[ticker] = trial
                            # Keep most recent trial per ticker
                            elif trial.get("last_update_posted", "") > trial_data_map[ticker].get(
                                "last_update_posted", ""
                            ):
                                trial_data_map[ticker] = trial

                    # Build financial data map
                    financial_data_map = {}
                    for score in m2_result.get("scores", []):
                        ticker = score.get("ticker")
                        if ticker:
                            financial_data_map[ticker] = score

                    # Get VIX from market snapshot
                    vix_current = None
                    if market_snapshot:
                        vix_current = Decimal(str(market_snapshot.get("vix", "20")))

                    # Compute accuracy adjustments for universe
                    universe_for_accuracy = [{"ticker": t.upper()} for t in active_tickers]
                    accuracy_adjustments = _safe_run_engine(
                        "Accuracy",
                        accuracy_adapter.compute_universe_adjustments,
                        universe=universe_for_accuracy,
                        trial_data_map=trial_data_map,
                        financial_data_map=financial_data_map,
                        as_of_date=as_of_date_obj,
                        vix_current=vix_current,
                        market_regime=regime_result.get("regime") if regime_result else None,
                    )

                    if accuracy_adjustments is None:
                        logger.info("  Accuracy enhancements: skipped (engine error)")
                    else:
                        accuracy_diag = accuracy_adapter.get_diagnostic_counts()
                        logger.info(
                            f"  Accuracy enhancements: {accuracy_diag['total']} tickers, "
                            f"staleness={accuracy_diag['staleness_coverage_pct']}%, "
                            f"regulatory={accuracy_diag['regulatory_coverage_pct']}%"
                        )

                        # Convert to serializable format
                        accuracy_result = {
                            "adjustments": {
                                ticker: {
                                    "clinical_adjustment": str(adj.clinical_adjustment),
                                    "financial_adjustment": str(adj.financial_adjustment),
                                    "catalyst_adjustment": str(adj.catalyst_adjustment),
                                    "regulatory_bonus": str(adj.regulatory_bonus),
                                    "confidence": adj.confidence,
                                    "adjustments_applied": adj.adjustments_applied,
                                }
                                for ticker, adj in accuracy_adjustments.items()
                            },
                            "diagnostic_counts": accuracy_diag,
                        }

                except Exception as _exc:
                    logger.warning("Enhancement 'Accuracy' failed: %s — skipping", _exc)

            # Step 5: Calculate dilution risk scores (if available)
            dilution_risk_result = None
            if HAS_DILUTION_RISK:
                try:
                    dilution_engine = DilutionRiskEngine()

                    # Build dilution risk universe from financial + market + catalyst data
                    dilution_universe = []
                    financial_by_ticker = {r.get("ticker"): r for r in financial_records if r.get("ticker")}
                    market_by_ticker = market_data_by_ticker  # Already indexed

                    for ticker in active_tickers:
                        fin_data = financial_by_ticker.get(ticker, {})
                        mkt_data = market_by_ticker.get(ticker.upper(), {})

                        # Get next catalyst date from Module 3 summaries
                        catalyst_summary = catalyst_summaries.get(ticker.upper()) or catalyst_summaries.get(ticker)
                        next_catalyst_date = None
                        if catalyst_summary:
                            # Handle both dict and object formats
                            if isinstance(catalyst_summary, dict):
                                next_catalyst_date = catalyst_summary.get("next_catalyst_date")
                            elif hasattr(catalyst_summary, "next_catalyst_date"):
                                next_catalyst_date = catalyst_summary.next_catalyst_date

                        # Calculate quarterly burn from NetIncome or R&D (annualized / 4)
                        quarterly_burn = None
                        net_income = fin_data.get("NetIncome")
                        if net_income is not None and net_income < 0:
                            # NetIncome is annualized, divide by 4 for quarterly
                            quarterly_burn = Decimal(str(net_income)) / Decimal("4")
                        elif fin_data.get("R&D"):
                            # Fall back to R&D as burn proxy (annualized / 4, negative)
                            quarterly_burn = -Decimal(str(fin_data.get("R&D"))) / Decimal("4")

                        dilution_universe.append(
                            {
                                "ticker": ticker,
                                "quarterly_cash": (
                                    Decimal(str(fin_data.get("CashAndSecurities") or fin_data.get("Cash")))
                                    if (fin_data.get("CashAndSecurities") or fin_data.get("Cash"))
                                    else None
                                ),
                                "quarterly_burn": quarterly_burn,
                                "next_catalyst_date": next_catalyst_date,
                                "market_cap": (
                                    Decimal(str(mkt_data.get("market_cap"))) if mkt_data.get("market_cap") else None
                                ),
                                "avg_daily_volume_90d": (
                                    int(mkt_data.get("avg_volume_90d", 0)) if mkt_data.get("avg_volume_90d") else None
                                ),
                            }
                        )

                    dilution_risk_result = _safe_run_engine(
                        "DilutionRisk", dilution_engine.score_universe, dilution_universe, as_of_date_obj
                    )
                    diag_dr = dilution_risk_result.get("diagnostic_counts", {})
                    risk_dist = diag_dr.get("risk_distribution", {})
                    logger.info(
                        f"  Dilution risk: {diag_dr.get('total_scored', 0)} scored, "
                        f"HIGH={risk_dist.get('HIGH_RISK', 0)}, "
                        f"MED={risk_dist.get('MEDIUM_RISK', 0)}, "
                        f"LOW={risk_dist.get('LOW_RISK', 0)}"
                    )

                except Exception as _exc:
                    logger.warning("Enhancement 'DilutionRisk' failed: %s — skipping", _exc)

            # Step 6: Calculate timeline slippage scores (if available)
            timeline_slippage_result = None
            if HAS_TIMELINE_SLIPPAGE:
                slippage_engine = TimelineSlippageEngine()

                # Build trial data by ticker for slippage analysis
                slippage_universe = [{"ticker": t} for t in active_tickers]
                current_trials_by_ticker = {}

                for trial in trial_records:
                    ticker = trial.get("lead_sponsor_ticker") or trial.get("ticker")
                    if ticker:
                        ticker_upper = ticker.upper()
                        if ticker_upper not in current_trials_by_ticker:
                            current_trials_by_ticker[ticker_upper] = []
                        current_trials_by_ticker[ticker_upper].append(trial)

                if slippage_universe and current_trials_by_ticker:
                    try:
                        timeline_slippage_result = slippage_engine.score_universe(
                            universe=slippage_universe,
                            current_trials_by_ticker=current_trials_by_ticker,
                            prior_trials_by_ticker=None,  # Would need historical snapshots
                            as_of_date=as_of_date_obj,
                        )
                        diag_ts = timeline_slippage_result.get("diagnostic_counts", {})
                        logger.info(
                            f"  Timeline slippage: {diag_ts.get('total_scored', 0)} scored, "
                            f"repeat_offenders={diag_ts.get('repeat_offenders', 0)}"
                        )
                    except Exception as e:
                        logger.warning(f"  Timeline slippage scoring failed: {e}")
                        timeline_slippage_result = None

            # Step 7: Calculate FDA designation scores (if available)
            fda_designation_result = None
            if HAS_FDA_DESIGNATIONS:
                fda_engine = FDADesignationEngine()

                # Load FDA designations from file (comprehensive) or fall back to sample data
                fda_designations_path = Path(data_dir) / "fda_designations.json"
                if fda_designations_path.exists():
                    try:
                        with open(fda_designations_path) as f:
                            fda_data = json.load(f)
                        designations_list = fda_data.get("designations", [])
                        loaded_count = fda_engine.load_designations(designations_list)
                        logger.info(f"  Loaded {loaded_count} FDA designations from {fda_designations_path.name}")
                    except Exception as e:
                        logger.warning(f"  Failed to load FDA designations: {e}, using sample data")
                        fda_engine.load_designations(generate_sample_designations())
                else:
                    logger.info("  No fda_designations.json found, using sample data")
                    fda_engine.load_designations(generate_sample_designations())

                fda_universe = [{"ticker": t.upper()} for t in active_tickers]
                fda_designation_result = fda_engine.score_universe(fda_universe, as_of_date_obj)

                diag_fda = fda_designation_result.get("diagnostic_counts", {})
                logger.info(
                    f"  FDA designations: {diag_fda.get('total_scored', 0)} scored, "
                    f"with_designations={diag_fda.get('with_designations', 0)}"
                )

            # Step 8: Calculate pipeline diversity scores (if available)
            pipeline_diversity_result = None
            if HAS_PIPELINE_DIVERSITY:
                try:
                    diversity_engine = PipelineDiversityEngine()

                    # Build clinical data map from Module 4 results
                    clinical_data_map = {}
                    for clinical_score in m4_result.get("scores", []):
                        ticker = clinical_score.get("ticker")
                        if ticker:
                            clinical_data_map[ticker.upper()] = {
                                "n_trials_unique": clinical_score.get("n_trials_unique", 0),
                                "lead_phase": clinical_score.get("lead_phase", "phase_2"),
                            }

                    diversity_universe = []
                    for t in active_tickers:
                        ticker_upper = t.upper()
                        diversity_universe.append(
                            {
                                "ticker": ticker_upper,
                                "clinical_data": clinical_data_map.get(ticker_upper, {}),
                            }
                        )

                    pipeline_diversity_result = _safe_run_engine(
                        "PipelineDiversity",
                        diversity_engine.score_universe,
                        diversity_universe,
                        trial_records,
                        as_of_date_obj,
                    )

                    diag_pd = pipeline_diversity_result.get("diagnostic_counts", {})
                    risk_dist = diag_pd.get("risk_distribution", {})
                    logger.info(
                        f"  Pipeline diversity: {diag_pd.get('total_scored', 0)} scored, "
                        f"single_asset={risk_dist.get('single_asset', 0)}, "
                        f"diversified={risk_dist.get('diversified', 0) + risk_dist.get('broad_portfolio', 0)}"
                    )

                except Exception as _exc:
                    logger.warning("Enhancement 'PipelineDiversity' failed: %s — skipping", _exc)

            # Step 9: Calculate competitive intensity scores (if available)
            competitive_intensity_result = None
            if HAS_COMPETITIVE_INTENSITY:
                try:
                    intensity_engine = CompetitiveIntensityEngine()

                    intensity_universe = [{"ticker": t.upper()} for t in active_tickers]
                    competitive_intensity_result = _safe_run_engine(
                        "CompetitiveIntensity",
                        intensity_engine.score_universe,
                        intensity_universe,
                        trial_records,
                        as_of_date_obj,
                    )

                    diag_ci = competitive_intensity_result.get("diagnostic_counts", {})
                    intensity_dist = diag_ci.get("intensity_distribution", {})
                    logger.info(
                        f"  Competitive intensity: {diag_ci.get('total_scored', 0)} scored, "
                        f"low={intensity_dist.get('low', 0)}, "
                        f"moderate={intensity_dist.get('moderate', 0)}, "
                        f"high={intensity_dist.get('high', 0)}, "
                        f"intense={intensity_dist.get('intense', 0)}"
                    )

                except Exception as _exc:
                    logger.warning("Enhancement 'CompetitiveIntensity' failed: %s — skipping", _exc)

            # Step 10: Calculate partnership validation scores (if available)
            partnership_result = None
            if HAS_PARTNERSHIP_ENGINE:
                try:
                    partnership_engine = PartnershipEngine()

                    # Load partnership data from file
                    partnerships_path = Path(data_dir) / "partnerships.json"
                    partnership_records = []
                    if partnerships_path.exists():
                        try:
                            with open(partnerships_path) as f:
                                partnership_data = json.load(f)
                            partnership_records = partnership_data.get("partnerships", [])
                            logger.info(f"  Loaded {len(partnership_records)} partnership records from file")
                        except Exception as e:
                            logger.warning(f"  Failed to load partnerships.json: {e}")

                    partnership_universe = [{"ticker": t.upper()} for t in active_tickers]
                    partnership_result = _safe_run_engine(
                        "Partnership",
                        partnership_engine.score_universe,
                        partnership_universe,
                        partnership_records,
                        as_of_date_obj,
                    )

                    diag_ps = partnership_result.get("diagnostic_counts", {})
                    strength_dist = diag_ps.get("strength_distribution", {})
                    logger.info(
                        f"  Partnership validation: {diag_ps.get('total_scored', 0)} scored, "
                        f"with_partnerships={diag_ps.get('with_partnerships', 0)}, "
                        f"with_top_tier={diag_ps.get('with_top_tier', 0)}"
                    )
                    logger.info(
                        f"    Strength distribution: exceptional={strength_dist.get('exceptional', 0)}, "
                        f"strong={strength_dist.get('strong', 0)}, "
                        f"moderate={strength_dist.get('moderate', 0)}, "
                        f"weak={strength_dist.get('weak', 0)}"
                    )

                except Exception as _exc:
                    logger.warning("Enhancement 'Partnership' failed: %s — skipping", _exc)

            # Step 11: Calculate cash burn trajectory scores (if available)
            cash_burn_result = None
            if HAS_CASH_BURN_ENGINE:
                try:
                    cash_burn_engine = CashBurnEngine()

                    # Load quarterly burn history data (if available)
                    quarterly_burn_path = Path(data_dir) / "quarterly_burn_history.json"
                    quarterly_burn_by_ticker = {}
                    if quarterly_burn_path.exists():
                        try:
                            with open(quarterly_burn_path) as f:
                                quarterly_data = json.load(f)
                            burn_history = quarterly_data.get("burn_history", {})
                            for ticker, data in burn_history.items():
                                quarterly_burn_by_ticker[ticker.upper()] = data.get("quarterly_burn", [])
                            logger.info(f"  Loaded quarterly burn history for {len(quarterly_burn_by_ticker)} tickers")
                        except Exception as e:
                            logger.warning(f"  Failed to load quarterly_burn_history.json: {e}")

                    # Build financial data map for cash burn engine
                    # Uses runway_months from Module 2 and quarterly burn data
                    cash_burn_financial = {}
                    for score in m2_result.get("scores", []):
                        ticker = score.get("ticker", "").upper()
                        if ticker:
                            cash_burn_financial[ticker] = {
                                "runway_months": score.get("runway_months"),
                                "cash_position": score.get("liquid_assets") or score.get("cash"),
                                "quarterly_burn": quarterly_burn_by_ticker.get(ticker, []),
                            }

                    # Build clinical data map for Phase 3 detection
                    cash_burn_clinical = {}
                    if m4_result:
                        for score in m4_result.get("scores", []):
                            ticker = score.get("ticker", "").upper()
                            if ticker:
                                cash_burn_clinical[ticker] = {
                                    "lead_phase": score.get("lead_phase"),
                                }

                    cash_burn_universe = [{"ticker": t.upper()} for t in active_tickers]
                    cash_burn_result = _safe_run_engine(
                        "CashBurn",
                        cash_burn_engine.score_universe,
                        cash_burn_universe,
                        cash_burn_financial,
                        cash_burn_clinical,
                        as_of_date_obj,
                    )

                    diag_cb = cash_burn_result.get("diagnostic_counts", {})
                    trajectory_dist = diag_cb.get("trajectory_distribution", {})
                    risk_dist = diag_cb.get("risk_distribution", {})
                    logger.info(f"  Cash burn trajectory: {diag_cb.get('total_scored', 0)} scored")
                    logger.info(
                        f"    Trajectory: decel={trajectory_dist.get('decelerating', 0)}, "
                        f"stable={trajectory_dist.get('stable', 0)}, "
                        f"accel={trajectory_dist.get('accelerating', 0)}, "
                        f"justified={trajectory_dist.get('accelerating_justified', 0)}, "
                        f"unknown={trajectory_dist.get('unknown', 0)}"
                    )
                    logger.info(
                        f"    Risk: low={risk_dist.get('low', 0)}, "
                        f"moderate={risk_dist.get('moderate', 0)}, "
                        f"high={risk_dist.get('high', 0)}, "
                        f"critical={risk_dist.get('critical', 0)}"
                    )

                except Exception as _exc:
                    logger.warning("Enhancement 'CashBurn' failed: %s — skipping", _exc)

            # Step 12: Calculate phase transition momentum scores (if available)
            phase_momentum_result = None
            if HAS_PHASE_MOMENTUM_ENGINE:
                try:
                    phase_momentum_engine = PhaseTransitionEngine()

                    # Build trials by ticker map
                    trials_by_ticker = {}
                    for trial in trial_records:
                        ticker = trial.get("ticker", "").upper()
                        if ticker:
                            if ticker not in trials_by_ticker:
                                trials_by_ticker[ticker] = []
                            trials_by_ticker[ticker].append(trial)

                    phase_momentum_universe = [{"ticker": t.upper()} for t in active_tickers]
                    phase_momentum_result = _safe_run_engine(
                        "PhaseMomentum",
                        phase_momentum_engine.score_universe,
                        phase_momentum_universe,
                        trials_by_ticker,
                        None,
                        as_of_date_obj,
                    )

                    diag_pm = phase_momentum_result.get("diagnostic_counts", {})
                    momentum_dist = diag_pm.get("momentum_distribution", {})
                    logger.info(f"  Phase momentum: {diag_pm.get('total_scored', 0)} scored")
                    logger.info(
                        f"    Momentum: strong_pos={momentum_dist.get('strong_positive', 0)}, "
                        f"pos={momentum_dist.get('positive', 0)}, "
                        f"neutral={momentum_dist.get('neutral', 0)}, "
                        f"neg={momentum_dist.get('negative', 0)}, "
                        f"strong_neg={momentum_dist.get('strong_negative', 0)}"
                    )

                except Exception as _exc:
                    logger.warning("Enhancement 'PhaseMomentum' failed: %s — skipping", _exc)

            # Step 13: Calculate Morningstar quantitative signal scores (if available)
            morningstar_result = None
            if HAS_MORNINGSTAR and morningstar_mode != "off":
                ms_engine = MorningstarSignalEngine()
                ms_loaded = ms_engine.load_data(Path(data_dir))
                if ms_loaded > 0:
                    ms_universe = [{"ticker": t.upper()} for t in active_tickers]
                    morningstar_result = _safe_run_engine(
                        "Morningstar",
                        ms_engine.score_universe,
                        ms_universe,
                        market_data_by_ticker,
                        as_of_date_obj,
                    )
                    ms_diag = morningstar_result.get("diagnostic_counts", {})
                    logger.info(
                        f"  Morningstar signals: {ms_diag.get('total_scored', 0)} scored, "
                        f"FV coverage: {ms_diag.get('fair_value_coverage_pct', 'N/A')}"
                    )

            # Assemble enhancement result (use empty dicts for None values to avoid downstream .get() errors)
            enhancement_result = {
                "regime": regime_result or {"regime": "UNKNOWN", "signal_adjustments": {}},
                "macro_snapshot": macro_snapshot.to_dict() if macro_snapshot else None,
                "pos_scores": pos_result or {},
                "short_interest_scores": si_result,
                "accuracy_enhancements": accuracy_result,
                "dilution_risk_scores": dilution_risk_result,
                "timeline_slippage_scores": timeline_slippage_result,
                "fda_designation_scores": fda_designation_result,
                "pipeline_diversity_scores": pipeline_diversity_result,
                "competitive_intensity_scores": competitive_intensity_result,
                "partnership_scores": partnership_result,
                "cash_burn_scores": cash_burn_result,
                "phase_momentum_scores": phase_momentum_result,
                "morningstar_scores": morningstar_result,
                "provenance": {
                    "module": "enhancements",
                    "version": "1.9.0",  # Bumped for Morningstar signal integration
                    "as_of_date": as_of_date,
                    "pos_engine_version": pos_engine.VERSION if pos_engine else None,
                    "regime_engine_version": regime_engine.VERSION if regime_engine else None,
                    "macro_collector_version": "1.0.0" if HAS_MACRO_COLLECTOR else None,
                    "accuracy_adapter_version": "1.0.0" if HAS_ACCURACY_ENHANCEMENTS else None,
                    "dilution_risk_engine_version": "1.0.0" if HAS_DILUTION_RISK else None,
                    "timeline_slippage_engine_version": "1.0.0" if HAS_TIMELINE_SLIPPAGE else None,
                    "fda_designation_engine_version": FDADesignationEngine.VERSION if HAS_FDA_DESIGNATIONS else None,
                    "pipeline_diversity_engine_version": (
                        PipelineDiversityEngine.VERSION if HAS_PIPELINE_DIVERSITY else None
                    ),
                    "competitive_intensity_engine_version": (
                        CompetitiveIntensityEngine.VERSION if HAS_COMPETITIVE_INTENSITY else None
                    ),
                    "partnership_engine_version": PartnershipEngine.VERSION if HAS_PARTNERSHIP_ENGINE else None,
                    "cash_burn_engine_version": CashBurnEngine.VERSION if HAS_CASH_BURN_ENGINE else None,
                    "phase_momentum_engine_version": (
                        PhaseTransitionEngine.VERSION if HAS_PHASE_MOMENTUM_ENGINE else None
                    ),
                    "morningstar_engine_version": MorningstarSignalEngine.VERSION if HAS_MORNINGSTAR else None,
                },
            }

            if checkpoint_dir:
                save_checkpoint(checkpoint_dir, "enhancements", as_of_date, enhancement_result)
    else:
        logger.info("[5.5/7] Enhancement Layer: SKIPPED (not enabled)")

    # ========================================================================
    # End Enhancement Layer
    # ========================================================================

    # Module 5: Composite ranking
    logger.info("[6/7] Module 5: Composite ranking...")
    m5_result = None
    if resume_index > 5 and checkpoint_dir:
        m5_result = load_checkpoint(checkpoint_dir, "module_5", as_of_date)

    if m5_result is None:
        # Apply clinical activity filter exclusions to module results
        # This ensures excluded tickers don't get ranked in Module 5
        clinical_excluded_set = set(t["ticker"] for t in clinical_exclusions) if clinical_exclusions else set()

        if clinical_excluded_set:
            # Create filtered copies of results (don't modify originals)
            m1_filtered = {
                **m1_result,
                "active_securities": [
                    s for s in m1_result.get("active_securities", []) if s.get("ticker") not in clinical_excluded_set
                ],
            }
            m2_filtered = {
                **m2_result,
                "scores": [s for s in m2_result.get("scores", []) if s.get("ticker") not in clinical_excluded_set],
            }
            m4_filtered = {
                **m4_result,
                "scores": [s for s in m4_result.get("scores", []) if s.get("ticker") not in clinical_excluded_set],
            }
        else:
            m1_filtered = m1_result
            m2_filtered = m2_result
            m4_filtered = m4_result

        # Load calibrated weights if provided
        m5_weights = None
        if module5_weights_path is not None:
            m5_bundle = _load_module5_weights_bundle(module5_weights_path)
            if m5_bundle is not None:
                current_regime = _normalize_m3_regime(
                    (regime_result or {}).get("regime"),
                )
                m5_weights = _select_module5_weights(
                    m5_bundle,
                    current_regime,
                    mode=module5_weights_mode,
                )
            # else: m5_weights stays None → defaults

        _t_m5 = __import__("time").time()
        m5_result = compute_module_5_composite_with_defensive(
            universe_result=m1_filtered,
            financial_result=m2_filtered,
            catalyst_result=m3_result,
            clinical_result=m4_filtered,
            as_of_date=as_of_date,  # Explicit threading
            weights=m5_weights,
            normalization="rank",
            cohort_mode="stage_only",
            coinvest_signals=coinvest_signals,
            validate=True,
            enhancement_result=enhancement_result,
            market_data_by_ticker=market_data_by_ticker,  # Enable volatility/momentum signals
            raw_financial_data=financial_records,  # Raw financial data for survivability scoring
            enable_sanity_override=False,  # Disabled: mixed v2/v3 scores cause rank artifacts
            enable_clustering=enable_clustering,
            cluster_method=cluster_method,
            cluster_threshold=cluster_threshold,
            # Defensive overlay parameters
            apply_defensive_multiplier=apply_defensive_multiplier,
            apply_position_sizing=enable_position_sizing,
            defensive_config=defensive_config,
            defensive_cache_path=defensive_cache,
            # Scoring mode (Baker-style or default)
            scoring_mode=scoring_mode,
        )
        if checkpoint_dir:
            save_checkpoint(checkpoint_dir, "module_5", as_of_date, m5_result)

    # Validate Module 5 output schema
    validate_module_5_output(m5_result)
    print(f"[timing] module_5: {__import__('time').time() - _t_m5:.2f}s", flush=True)

    diag = m5_result.get("diagnostic_counts", {})
    logger.info(
        f"  Rankable: {diag.get('rankable', len(m5_result.get('ranked_securities', [])))}, "
        f"Excluded: {diag.get('excluded', len(m5_result.get('excluded_securities', [])))}"
    )

    # Log pipeline health status
    run_status = m5_result.get("run_status", "UNKNOWN")
    degraded = m5_result.get("degraded_components", [])
    coverage = m5_result.get("component_coverage", {})
    gated = m5_result.get("gated_component_counts", {})

    if run_status == "FAIL":
        logger.error(f"  RUN STATUS: {run_status} - Pipeline health check FAILED")
        for err in m5_result.get("health_errors", []):
            logger.error(f"    {err}")
    elif run_status == "DEGRADED":
        logger.warning(f"  RUN STATUS: {run_status} - Some components degraded: {degraded}")
        for warn in m5_result.get("health_warnings", []):
            logger.warning(f"    {warn}")
    else:
        logger.info(f"  RUN STATUS: {run_status}")

    # Log component coverage
    logger.info(
        f"  Component coverage: catalyst={coverage.get('catalyst', '?')}, "
        f"momentum={coverage.get('momentum', '?')}, smart_money={coverage.get('smart_money', '?')}"
    )
    if gated:
        logger.warning(f"  Confidence-gated components: {gated}")

    # Dev-catalyst guardrail activation summary
    _gr_summary = summarize_dev_catalyst_guardrail(m5_result.get("ranked_securities", []))
    if _gr_summary["total_impacted"] > 0:
        logger.info(
            f"  Dev-catalyst guardrail: {_gr_summary['total_impacted']}/{_gr_summary['total_evaluated']} "
            f"({_gr_summary['pct_impacted']}%) impacted  "
            f"[Rule1={_gr_summary['rule1_count']} Rule2={_gr_summary['rule2_count']}]"
        )
        for _gi in _gr_summary["top_5_impacted"]:
            logger.info(f"    #{_gi['rank']:>3d}  {_gi['ticker']:<6s}  {' '.join(_gi['flags'])}")
    else:
        logger.info("  Dev-catalyst guardrail: 0 tickers impacted")
    # Stash for downstream / JSON output
    m5_result.setdefault("diagnostic_counts", {})["dev_catalyst_guardrail"] = _gr_summary

    # Final defensive overlay and top-N selection
    logger.info("[7/7] Defensive overlay & top-N selection...")

    # Log v2 risk diagnostics if present
    risk_v2 = diag.get("risk_v2_summary")
    if risk_v2 and risk_v2.get("v2_detected"):
        logger.info("  V2 risk metrics detected in defensive cache")
        for metric, counts in risk_v2.get("state_counts", {}).items():
            logger.info(
                f"    {metric}: FULL={counts.get('FULL', 0)} PARTIAL={counts.get('PARTIAL', 0)} NONE={counts.get('NONE', 0)}"
            )
        priors = risk_v2.get("priors_used", {})
        logger.info(
            f"    Priors used: vol={priors.get('vol', 0)} dd={priors.get('drawdown', 0)} beta={priors.get('beta', 0)}"
        )
        lc = risk_v2.get("low_confidence_count", 0)
        lc_pct = risk_v2.get("low_confidence_pct", 0)
        if risk_v2.get("low_confidence_warning"):
            logger.warning(f"    LOW CONFIDENCE WARNING: {lc} tickers ({lc_pct}%) have confidence_risk < 0.65")
        else:
            logger.info(f"    Low confidence: {lc} tickers ({lc_pct}%)")

    # Serialize module 3 result (contains TickerCatalystSummary dataclasses)
    def _serialize_m3_result(m3):
        """Convert TickerCatalystSummary objects to dicts for JSON serialization."""
        result = dict(m3)
        if "summaries" in result:
            serialized_summaries = {}
            for ticker, summary in result["summaries"].items():
                if hasattr(summary, "to_dict"):
                    serialized_summaries[ticker] = summary.to_dict()
                else:
                    serialized_summaries[ticker] = summary
            result["summaries"] = serialized_summaries
        return result

    m3_serialized = _serialize_m3_result(m3_result)

    # Assemble results
    run_metadata = {
        "as_of_date": as_of_date,
        "version": VERSION,
        "deterministic_timestamp": _deterministic_timestamp(as_of_date),
        "input_hashes": dict(sorted(content_hashes.items())),
        "enhancements_enabled": enable_enhancements,
        # New governance-friendly parameters
        "catalyst_window": f"{catalyst_window[0]}-{catalyst_window[1]}" if catalyst_window else "15-45",
        "catalyst_decay": catalyst_decay,
        "catalyst_half_life_days": catalyst_half_life_days if catalyst_decay == "exp" else None,
        "enable_smart_money": enable_smart_money,
        "smart_money_source": smart_money_source,
        "min_component_coverage": min_component_coverage,
        "min_component_coverage_mode": min_component_coverage_mode,
        "defensive_multiplier_enabled": apply_defensive_multiplier,
        "defensive_config": defensive_config if apply_defensive_multiplier else None,
        "position_sizing_enabled": enable_position_sizing,
        "pit_mode": pit_mode,
        "pit_violation_sources": [pit_violation] if pit_violation else [],
        "catalyst_mode": m3_result.get("catalyst_mode", "full"),
        # PIT audit + holdings schema for reproducibility
        "coinvest_audit": coinvest_audit_data if coinvest_audit_data else None,
        "holdings_detailed_schema": holdings_schema_data,
        # Conviction horizon overlay audit (Enhancement 12)
        "conviction_horizon_overlay_applied": (
            m5_result.get("diagnostic_counts", {}).get("conviction_horizon_overlay_applied", 0) if m5_result else 0
        ),
        "inputs_manifest": _inputs_manifest,
    }
    results = {
        "_governance": _build_governance_envelope(as_of_date=as_of_date, run_metadata=run_metadata),
        "run_metadata": run_metadata,
        "module_1_universe": m1_result,
        "module_2_financial": m2_result,
        "module_3_catalyst": m3_serialized,
        "module_4_clinical": m4_result,
        "module_5_composite": m5_result,
        "summary": {
            "total_evaluated": len(raw_universe),
            "active_universe": len(active_tickers),
            "excluded": len(m1_result.get("excluded_securities", [])),
            "clinical_activity_filter": {
                "enabled": not no_clinical_filter,
                "min_trials": min_trials,
                "min_phase": min_phase,
                "excluded_count": len(clinical_exclusions),
                "exempted_count": len(clinical_exemptions),
            },
            "final_ranked": len(m5_result.get("ranked_securities", [])),
            "catalyst_events": diag3.get("events_detected_total", 0),
            "severe_negatives": diag3.get("tickers_with_severe_negative", 0),
        },
        "clinical_exclusions": clinical_exclusions,
        "clinical_exemptions": clinical_exemptions,
        "company_archetypes": company_archetypes,
        "industry_groups": industry_groups,
    }

    # Add enhancement results if enabled
    if enhancement_result:
        results["enhancements"] = enhancement_result
        # Add enhancement summary info
        regime = enhancement_result.get("regime", {})
        results["summary"]["regime"] = regime.get("regime", "UNKNOWN")
        results["summary"]["regime_confidence"] = str(regime.get("confidence", "0"))
        if enhancement_result.get("pos_scores"):
            pos_diag = enhancement_result["pos_scores"].get("diagnostic_counts", {})
            results["summary"]["pos_indication_coverage"] = pos_diag.get("indication_coverage_pct", "N/A")
            results["summary"]["pos_effective_coverage"] = pos_diag.get("effective_coverage_pct", "N/A")
            results["summary"]["pos_confidence_distribution"] = pos_diag.get("confidence_distribution", {})
        if enhancement_result.get("short_interest_scores"):
            si_diag = enhancement_result["short_interest_scores"].get("diagnostic_counts", {})
            results["summary"]["short_interest_coverage"] = si_diag.get("data_coverage_pct", "N/A")
        if enhancement_result.get("morningstar_scores"):
            ms_diag = enhancement_result["morningstar_scores"].get("diagnostic_counts", {})
            results["summary"]["morningstar_coverage"] = ms_diag.get("total_scored", 0)
            results["summary"]["morningstar_fv_coverage"] = ms_diag.get("fair_value_coverage_pct", "N/A")

    # =========================================================================
    # BAKER OVERLAY (post-processing: CIK attribution, IC review queues)
    # =========================================================================
    if HAS_BAKER_OVERLAY and raw_holdings_for_baker and enable_smart_money:
        try:
            logger.info("[Baker] Computing Baker overlay...")
            # Get catalyst summaries for Queue 2 (may be serialized dicts or dataclasses)
            m3_summaries = m3_result.get("summaries", {}) if m3_result else {}
            # Serialize if needed (dataclass → dict)
            if m3_summaries:
                sample = next(iter(m3_summaries.values()), None)
                if sample and hasattr(sample, "to_dict"):
                    m3_summaries = {k: v.to_dict() for k, v in m3_summaries.items()}

            baker_result = compute_baker_overlay(
                ranked_securities=m5_result.get("ranked_securities", []),
                holdings_snapshots=raw_holdings_for_baker,
                catalyst_summaries=m3_summaries,
            )

            # Enrich each ranked security with baker signals
            for sec, baker_data in zip(
                m5_result.get("ranked_securities", []),
                baker_result.get("securities", []),
            ):
                sec["baker"] = baker_data

            # Add baker overlay to results
            results["baker_overlay"] = {
                "queue_1_core_divergence": baker_result["queue_1_core_divergence"],
                "queue_2_near_term_catalyst": baker_result["queue_2_near_term_catalyst"],
                "metrics": baker_result["metrics"],
                "diagnostics": baker_result["diagnostics"],
            }

            q1_count = len(baker_result["queue_1_core_divergence"])
            q2_count = len(baker_result["queue_2_near_term_catalyst"])
            held = baker_result["metrics"].get("baker_held_count", 0)
            logger.info(
                f"  Baker overlay: {held} held, Queue 1: {q1_count} divergences, "
                f"Queue 2: {q2_count} near-term catalysts"
            )
        except Exception as e:
            logger.warning(f"  Baker overlay failed (non-blocking): {type(e).__name__}: {e}")
            results["baker_overlay"] = {"error": f"{type(e).__name__}: {e}"}

    # Force deterministic timestamps for byte-identical outputs
    deterministic_ts = _deterministic_timestamp(as_of_date)
    _force_deterministic_generated_at(results, deterministic_ts)

    return results


def _attach_phase2_decision_ruleset_manifest(
    results: Dict[str, Any],
    ruleset_path: Path,
) -> Optional[Dict[str, Any]]:
    """Attach the Phase-2 decision ruleset to the run inputs manifest.

    The pipeline builds its manifest before the CLI resolves the Phase-2 ruleset.
    Patch it before writing outputs so manifest verify/replay covers ruleset drift.
    """
    manifest = (results.get("run_metadata") or {}).get("inputs_manifest")
    if not isinstance(manifest, dict):
        return None

    entry: Dict[str, Any] = {
        "key": "decision_ruleset",
        "path": str(ruleset_path),
        "required": True,
        "exists": ruleset_path.exists(),
        "resolved_from": "production_data",
        "load_site": "run_screen.py:phase2_cli",
        "sha256": sha256_file(ruleset_path) if ruleset_path.exists() else None,
        "record_count": None,
    }

    dependencies = [d for d in manifest.get("dependencies", []) if d.get("key") != "decision_ruleset"]
    dependencies.append(entry)
    dependencies.sort(key=lambda d: d["key"])
    manifest["dependencies"] = dependencies

    validation = manifest.setdefault("validation", {})
    errors = [e for e in validation.get("errors", []) if not str(e).startswith("decision_ruleset:")]
    if not entry["exists"]:
        errors.append(f"decision_ruleset: required file missing ({ruleset_path})")
    validation["errors"] = errors
    validation["all_required_present"] = len(errors) == 0
    validation.setdefault("warnings", [])
    return entry


def _verify_inputs_manifest_after_cli_patch(
    results: Dict[str, Any],
    verify_path: Optional[Path],
    from_replay_bundle: bool = False,
) -> None:
    """Re-run manifest verification after CLI-only dependencies are patched in."""
    manifest = (results.get("run_metadata") or {}).get("inputs_manifest")
    if not isinstance(manifest, dict):
        raise FileNotFoundError("Input manifest verify failed: manifest missing from results")
    if not verify_inputs_manifest(manifest):
        raise FileNotFoundError("Input manifest verify failed: " + "; ".join(manifest["validation"]["errors"]))
    if not verify_path:
        raise FileNotFoundError("--inputs-manifest verify requires --inputs-manifest-path")
    if not verify_path.exists():
        raise FileNotFoundError(f"Prior inputs_manifest not found: {verify_path}")
    with open(verify_path, "r", encoding="utf-8") as _mf:
        prior_manifest = json.load(_mf)
    drift_errors = verify_against_prior_manifest(manifest, prior_manifest, allow_new_required_deps=from_replay_bundle)
    if drift_errors:
        for drift_error in drift_errors:
            logger.error(f"[INPUT MANIFEST DRIFT] {drift_error}")
        raise FileNotFoundError("Input manifest drift vs prior: " + "; ".join(drift_errors))


def _load_decision_ruleset_for_cli(args: argparse.Namespace) -> DecisionRuleset:
    """Resolve and validate the CLI decision ruleset before any outputs are written."""
    if args.ruleset:
        de_ruleset = DecisionRuleset.from_json(str(args.ruleset))
        if args.decision_mode == "phase2" and de_ruleset.ruleset_id != PHASE2_PINNED_RULESET_ID:
            logger.warning(
                f"Phase-2 ruleset override: {de_ruleset.ruleset_id} != "
                f"pinned {PHASE2_PINNED_RULESET_ID} (explicit --ruleset)"
            )
        return de_ruleset

    if args.decision_mode == "phase2":
        if not PHASE2_DEFAULT_RULESET_PATH.exists():
            raise FileNotFoundError(f"Phase-2 pinned ruleset not found: {PHASE2_DEFAULT_RULESET_PATH}")
        de_ruleset = DecisionRuleset.from_json(str(PHASE2_DEFAULT_RULESET_PATH))
        logger.info(
            f"Phase-2 mode: loaded pinned ruleset {de_ruleset.ruleset_id} " f"from {PHASE2_DEFAULT_RULESET_PATH.name}"
        )
        if de_ruleset.ruleset_id != PHASE2_PINNED_RULESET_ID:
            raise ValueError(
                f"PHASE2 GUARDRAIL: loaded ruleset {de_ruleset.ruleset_id} != "
                f"pinned {PHASE2_PINNED_RULESET_ID}. File may be corrupted: {PHASE2_DEFAULT_RULESET_PATH}"
            )
        return de_ruleset

    return DEFAULT_RULESET


def compute_data_hash(data_dir: Path) -> str:
    """Compute an input-only hash for deterministic bootstrap seed derivation."""
    payload = _compute_run_input_hashes(data_dir)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def add_bootstrap_analysis(
    results: dict,
    as_of_date: str,
    data_dir: Path,
    n_bootstrap: int,
    run_id: str = None,
) -> dict:
    """Add bootstrap confidence intervals to results."""
    from common.random_state import DeterministicRNG, derive_base_seed
    from extensions.bootstrap_scoring import compute_bootstrap_ci_decimal

    # Derive deterministic seed
    data_hash = compute_data_hash(data_dir)
    if run_id is None:
        run_id = f"screen_{as_of_date}"
    base_seed = derive_base_seed(as_of_date, run_id, data_hash)

    # Create RNG
    rng = DeterministicRNG(base_seed, "bootstrap_ci")

    # Get composite scores
    ranked = results["module_5_composite"]["ranked_securities"]
    if not ranked:
        results["bootstrap_analysis"] = {"error": "no_securities_to_bootstrap"}
        return results

    # Guard: skip if composite_score is absent/empty (e.g. DE-only mode)
    if not any(s.get("composite_score") not in (None, "") for s in ranked):
        results["bootstrap_analysis"] = {"error": "no_composite_scores"}
        return results

    scores = [Decimal(s["composite_score"]) for s in ranked if s.get("composite_score") not in (None, "")]

    # Compute bootstrap CI
    bootstrap_result = compute_bootstrap_ci_decimal(
        scores=scores,
        rng=rng,
        n_bootstrap=n_bootstrap,
    )

    results["bootstrap_analysis"] = bootstrap_result
    return results


def _run_phase2_delta(snap_path, snapshot_dir, args, logger):
    """Run Phase-2 delta report as a post-snapshot hook.

    Writes delta artifacts into the snapshot directory. Warns on operational
    issues but only hard-fails on ruleset mismatch (already caught upstream).
    """
    from run_phase2_snapshot_delta import DEFAULT_HEALTH_THRESHOLDS
    from run_phase2_snapshot_delta import PHASE2_PINNED_RULESET_ID as DELTA_PINNED_ID
    from run_phase2_snapshot_delta import (
        Phase2HealthThresholds,
        compute_delta,
        compute_health_gate,
        compute_single_snapshot_summary,
        find_snapshots,
        generate_delta_csv,
        generate_details_json,
        generate_health_json,
        generate_report,
        is_snapshot_degraded,
        load_snapshot,
    )

    logger.info("--- Phase-2 Delta Report ---")

    # Load health thresholds
    ht_override = getattr(args, "health_thresholds", None)
    if ht_override and Path(ht_override).exists():
        health_thresholds = Phase2HealthThresholds.from_json(str(ht_override))
        logger.info(f"Health thresholds: {Path(ht_override).name} (id={health_thresholds.thresholds_id})")
    elif PHASE2_DEFAULT_HEALTH_THRESHOLDS_PATH.exists():
        health_thresholds = Phase2HealthThresholds.from_json(str(PHASE2_DEFAULT_HEALTH_THRESHOLDS_PATH))
        if health_thresholds.thresholds_id != PHASE2_PINNED_THRESHOLDS_ID:
            logger.warning(
                f"Health thresholds id {health_thresholds.thresholds_id} != " f"pinned {PHASE2_PINNED_THRESHOLDS_ID}"
            )
        logger.info(f"Health thresholds: pinned v1 (id={health_thresholds.thresholds_id})")
    else:
        logger.warning(
            f"Pinned health thresholds not found: {PHASE2_DEFAULT_HEALTH_THRESHOLDS_PATH}, " f"using built-in defaults"
        )
        health_thresholds = DEFAULT_HEALTH_THRESHOLDS

    current_date = snap_path.name
    # Defensive: argparse default is "auto", but downstream callers that set
    # args.delta_prior = None/"" would crash on .lower() below. Normalize here.
    delta_prior = getattr(args, "delta_prior", "auto") or "auto"

    # Resolve prior
    if delta_prior == "auto":
        _, prior_path = find_snapshots(snapshot_dir, current_date, None)
        # Silent auto-prior misses left no trail in 2026-04-24's delta report;
        # emit a candidate-by-candidate diagnostic whenever auto finds nothing.
        if prior_path is None:
            try:
                from run_phase2_snapshot_delta import diagnose_missing_prior

                logger.warning(
                    "Phase-2 delta auto-prior returned None; diagnostics:\n%s",
                    diagnose_missing_prior(snapshot_dir, current_date),
                )
            except Exception as _diag_exc:
                logger.debug("Auto-prior diagnostic failed: %s", _diag_exc)
    elif delta_prior.lower() == "none":
        prior_path = None
    else:
        _, prior_path = find_snapshots(snapshot_dir, current_date, delta_prior)

    current = load_snapshot(snap_path)
    if current is None:
        logger.warning("Delta: could not load current snapshot, skipping")
        return

    # Check if current snapshot is degraded → suppress churn
    _churn_suppressed = is_snapshot_degraded(snap_path)
    if _churn_suppressed:
        logger.warning("CACHE DEGRADED — churn comparisons suppressed; baselines not updated")
        prior = None  # force single-snapshot mode
        prior_path = None

    prior = None
    if prior_path is not None:
        prior = load_snapshot(prior_path)
        if prior is None:
            logger.warning(f"Delta: prior {prior_path.name} incompatible, running single-snapshot mode")

    if prior is not None:
        logger.info(f"Delta: {current_date} vs {prior.date}")
        result = compute_delta(current, prior)
    else:
        logger.info(f"Delta: {current_date} (no comparable prior)")
        result = compute_single_snapshot_summary(current)

    # Health gate — when --ruleset is explicit, compare against the requested
    # ruleset rather than the pinned default to avoid spurious FAIL.
    _explicit_id = None
    if getattr(args, "ruleset", None):
        _explicit_id = current.ruleset_id  # snapshot was built with this ruleset
    health = compute_health_gate(
        current,
        prior,
        result,
        thresholds=health_thresholds,
        expected_ruleset_id=_explicit_id,
    )
    health_json = generate_health_json(health, thresholds=health_thresholds)

    # Write artifacts into snapshot dir
    output_dir = snap_path
    report_txt = generate_report(current, prior, result, health=health, churn_suppressed=_churn_suppressed)
    details = generate_details_json(current, prior, result, churn_suppressed=_churn_suppressed)

    report_path = output_dir / "phase2_run_delta_report.txt"
    details_path = output_dir / "phase2_run_delta_details.json"
    csv_path = output_dir / "phase2_run_delta.csv"
    health_path = output_dir / "phase2_health.json"

    try:
        with open(report_path, "w") as f:
            f.write(report_txt)
        with open(details_path, "w") as f:
            json.dump(details, f, indent=2)
        generate_delta_csv(current, prior, csv_path)
        with open(health_path, "w") as f:
            json.dump(health_json, f, indent=2)
        # Exposure sidecar — per-check PASS/WARN/FAIL detail
        exposure_raw = health.metrics.get("exposure")
        if exposure_raw:
            from run_phase2_snapshot_delta import _EXPOSURE_CHECKS

            exposure_sidecar = {"metrics": exposure_raw, "checks": {}}
            for metric_key, warn_key, fail_key, reason in _EXPOSURE_CHECKS:
                val = exposure_raw.get(metric_key, 0)
                w_th = getattr(health_thresholds, warn_key)
                f_th = getattr(health_thresholds, fail_key)
                if val >= f_th:
                    chk = "FAIL"
                elif val >= w_th:
                    chk = "WARN"
                else:
                    chk = "PASS"
                exposure_sidecar["checks"][reason] = {
                    "value": val,
                    "warn_threshold": w_th,
                    "fail_threshold": f_th,
                    "status": chk,
                }
            exp_path = output_dir / "health_exposure_metrics.json"
            with open(exp_path, "w") as f:
                json.dump(exposure_sidecar, f, indent=2)
        logger.info(f"Delta artifacts: {report_path.name}, {details_path.name}, {csv_path.name}, {health_path.name}")
    except OSError as e:
        logger.warning(f"Delta: could not write artifacts: {e}")
        return health

    # Surface warnings
    warnings = result.warnings if hasattr(result, "warnings") else []
    if warnings:
        for w in warnings:
            logger.warning(f"Delta: {w}")

    # Print health headline
    m = health.metrics
    metric_parts = []
    if "name_turnover_pct" in m:
        metric_parts.append(f"turnover={m['name_turnover_pct']}%")
    metric_parts.append(f"catalyst={m['catalyst_coverage_pct']}%")
    metric_parts.append(f"A={m['a_count']}")
    reason_tag = f" [{', '.join(health.reasons)}]" if health.reasons else ""
    logger.info(f"PHASE2 HEALTH: {health.status} ({', '.join(metric_parts)}){reason_tag}")

    # Print delta headline summary
    is_delta = hasattr(result, "entrants")
    if is_delta:
        n_ent = len(result.entrants)
        n_exit = len(result.exits)
        logger.info(
            f"Delta headline: +{n_ent}/-{n_exit} names, "
            f"turnover={result.name_turnover_pct}%, "
            f"L1={result.weight_l1_delta}%"
        )
    logger.info("--- End Delta Report ---")

    return health


def _default_output_path(*, as_of_date: str, data_dir: Path, snapshot_dir: Optional[Path]) -> Path:
    """Derive a deterministic output path when --output is omitted."""
    if snapshot_dir is not None:
        base = snapshot_dir
    else:
        base = data_dir.parent / "data" / "snapshots"
    return base / as_of_date / "screen_output.json"


def _managed_snapshot_dir(*, as_of_date: str, data_dir: Path, snapshot_dir: Optional[Path]) -> Path:
    """Return the dated snapshot directory managed by run_screen.py."""
    if snapshot_dir is not None:
        base = snapshot_dir
    else:
        base = data_dir.parent / "data" / "snapshots"
    return base / as_of_date


def _same_filesystem_path(left: Path, right: Path) -> bool:
    """Compare paths without requiring them to exist."""
    try:
        return left.resolve(strict=False) == right.resolve(strict=False)
    except OSError:
        return left.absolute() == right.absolute()


def _output_targets_managed_snapshot_dir(
    output_path: Path,
    *,
    as_of_date: str,
    data_dir: Path,
    snapshot_dir: Optional[Path],
) -> bool:
    """Return True when an output path writes directly into the managed snapshot dir."""
    managed_dir = _managed_snapshot_dir(as_of_date=as_of_date, data_dir=data_dir, snapshot_dir=snapshot_dir)
    return _same_filesystem_path(output_path.parent, managed_dir)


def _enforce_snapshot_output_overwrite_policy(
    output_path: Path,
    *,
    as_of_date: str,
    data_dir: Path,
    snapshot_dir: Optional[Path],
    force_overwrite: bool,
) -> None:
    """Apply CCFT frozen-snapshot policy before writing managed snapshot outputs."""
    managed_dir = _managed_snapshot_dir(as_of_date=as_of_date, data_dir=data_dir, snapshot_dir=snapshot_dir)
    if force_overwrite or not _same_filesystem_path(output_path.parent, managed_dir):
        return
    if managed_dir.exists():
        raise FileExistsError(
            f"Snapshot already exists at {managed_dir}; refusing to write {output_path.name} "
            "without --force-overwrite."
        )


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Wake Robin Biotech Screening Pipeline (Deterministic)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic screening
  python run_screen.py --as-of-date 2024-12-15 --data-dir ./data --output results.json

  # Dry-run (validate inputs without running)
  python run_screen.py --as-of-date 2024-12-15 --data-dir ./data --dry-run

  # With checkpointing (save intermediate results)
  python run_screen.py --as-of-date 2024-12-15 --data-dir ./data --output results.json --checkpoint-dir ./checkpoints

  # Resume from module 3 (load prior checkpoints)
  python run_screen.py --as-of-date 2024-12-15 --data-dir ./data --output results.json --checkpoint-dir ./checkpoints --resume-from module_3

  # With audit trail
  python run_screen.py --as-of-date 2024-12-15 --data-dir ./data --output results.json --audit-log ./audit.jsonl

  # With co-invest overlay
  python run_screen.py --as-of-date 2024-12-15 --data-dir ./data --output results.json --enable-coinvest

  # Custom universe
  python run_screen.py --as-of-date 2024-12-15 --data-dir ./data --output results.json --tickers REGN VRTX ALNY

Determinism guarantees:
  - Identical inputs + as_of_date → identical outputs
  - No today() defaults (as_of_date is required)
  - PIT discipline enforced throughout
  - Stable ordering on all outputs

Orchestration features:
  - Dry-run: Validate inputs and compute hashes without running pipeline
  - Checkpointing: Save/load intermediate results for each module
  - Audit trail: Log parameter snapshots and content hashes to JSONL

Module 3 Catalyst Detection:
  - Delta-based event detection (compares current vs prior CT.gov state)
  - First run: 0 events (no prior state)
  - Subsequent runs: Events detected if trial data changed
  - IMPORTANT: Refresh trial_records.json from CT.gov before each run!
        """,
    )

    parser.add_argument(
        "--as-of-date",
        required=True,
        help="Analysis date (YYYY-MM-DD). REQUIRED - no defaults to prevent nondeterminism.",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing input data files (universe.json, financial.json, trial_records.json)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file path (not required for --dry-run)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for Module 3 side-outputs (catalyst_events, run_log, ctgov_state). Defaults to --data-dir.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs without running pipeline. Prints content hashes and exits.",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for saving/loading checkpoints between modules.",
    )

    parser.add_argument(
        "--resume-from",
        type=str,
        choices=["module_1", "module_2", "module_3", "module_4", "enhancements", "module_5"],
        default=None,
        help="Resume pipeline from specified module (requires --checkpoint-dir).",
    )

    parser.add_argument(
        "--audit-log",
        type=Path,
        default=None,
        help="Path to audit log file (JSONL format) for parameter and hash tracking.",
    )

    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="Directory for saving validation snapshots (rankings + metadata per run). "
        "Defaults to data-dir/../data/snapshots. Use --no-snapshot to disable.",
    )

    parser.add_argument(
        "--prior-snapshot-dir",
        type=Path,
        default=None,
        help="Directory to search for prior snapshots (for delta/shadow metrics). "
        "Defaults to --snapshot-dir. Useful when snapshot-dir is a staging dir.",
    )

    parser.add_argument(
        "--no-snapshot",
        action="store_true",
        default=False,
        help="Disable automatic validation snapshot saving.",
    )
    parser.add_argument(
        "--no-long-call-report",
        action="store_true",
        default=False,
        help="Skip automatic long-call candidate report after snapshot.",
    )
    parser.add_argument(
        "--no-data-collection-health",
        action="store_true",
        default=False,
        help="Skip automatic data collection health report after snapshot.",
    )

    parser.add_argument(
        "--decision-mode",
        type=str,
        default="phase2",
        choices=["observe", "actionable", "phase2"],
        help="Decision output mode: observe (default, sort by composite_rank), "
        "actionable (sort by tier-based deterministic ordering with target weights), "
        "or phase2 (observe + emit filtered decision_portfolio.csv/json with "
        "A+B tier filter, top-K=20, size-band weights).",
    )

    parser.add_argument(
        "--ranking-mode",
        type=str,
        default="decision",
        choices=["decision", "composite"],
        help="CSV sort order: 'decision' (default, rows sorted by DE actionable "
        "sort key) or 'composite' (legacy, re-sort by composite_rank).",
    )

    parser.add_argument(
        "--ranker-mode",
        type=str,
        default="pairwise_minimal",
        choices=["clinical_50", "pairwise_minimal"],
        help="Production ranker: 'clinical_50' (bounded adjustment, legacy) or "
        "'pairwise_minimal' (v2 pairwise logistic, Spec 051 winner). "
        "Default: pairwise_minimal. Rollback: --ranker-mode clinical_50.",
    )

    parser.add_argument(
        "--ruleset",
        type=Path,
        default=None,
        help="Path to a decision engine ruleset JSON file. " "If omitted, built-in defaults are used.",
    )

    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing snapshot for the same as_of_date. "
        "Without this flag, the pipeline refuses to overwrite (CCFT Frozen).",
    )

    parser.add_argument(
        "--no-delta",
        action="store_true",
        default=False,
        help="Skip Phase-2 delta report generation after snapshot.",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Exit non-zero on Phase-2 health WARN (2) or FAIL (1).",
    )

    parser.add_argument(
        "--alpha-schema-mode",
        type=str,
        default="fail",
        choices=["warn", "fail"],
        help="Alpha signal contract enforcement: 'fail' (raise on missing required "
        "fields, default) or 'warn' (log + continue, for historical replay).",
    )

    parser.add_argument(
        "--inputs-manifest",
        type=str,
        default="off",
        choices=["off", "write", "verify"],
        dest="inputs_manifest",
        help="Input manifest mode: 'off' (default), 'write' (emit sidecar), "
        "'verify' (drift-check against prior manifest, fail on hash mismatch).",
    )

    parser.add_argument(
        "--inputs-manifest-path",
        type=Path,
        default=None,
        help="Path to an existing inputs_manifest.json to verify against " "(required when --inputs-manifest=verify).",
    )

    # --alpha-ab-diagnostics removed (alpha_modifier deprecated & inert)

    parser.add_argument(
        "--replay-bundle-out",
        type=Path,
        default=None,
        help="Write a replay_bundle.tgz after the run using the inputs manifest. " "Implies --inputs-manifest=write.",
    )

    parser.add_argument(
        "--replay-bundle",
        type=Path,
        default=None,
        help="Run from a replay bundle (extracts data, auto-wires verify + ruleset).",
    )

    parser.add_argument(
        "--health-thresholds",
        type=Path,
        default=None,
        help="Path to Phase2HealthThresholds JSON override. " "If omitted, uses pinned production defaults.",
    )

    parser.add_argument(
        "--delta-prior",
        type=str,
        default="auto",
        help="Prior snapshot for delta report: 'auto' (default, find most recent "
        "compatible), 'none' (single-snapshot mode), or YYYY-MM-DD.",
    )

    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Optional: Whitelist specific tickers (default: use all from universe.json)",
    )

    parser.add_argument(
        "--enable-coinvest",
        action="store_true",
        default=True,
        help="Enable co-invest overlay (requires coinvest_signals.json or holdings_snapshots.json in data-dir). "
        "Enabled by default.",
    )
    parser.add_argument(
        "--no-coinvest",
        action="store_false",
        dest="enable_coinvest",
        help="Disable co-invest overlay.",
    )

    parser.add_argument(
        "--enable-enhancements",
        action="store_true",
        default=True,
        dest="enable_enhancements",
        help="Enable enhancement modules (PoS, Short Interest, Regime Detection). "
        "Enabled by default. Requires market_snapshot.json for regime detection.",
    )
    parser.add_argument(
        "--no-enhancements",
        action="store_false",
        dest="enable_enhancements",
        help="Disable enhancement modules (PoS, Short Interest, Regime Detection).",
    )

    parser.add_argument(
        "--enable-short-interest",
        action="store_true",
        default=True,
        dest="enable_short_interest",
        help="Enable short interest signals (requires short_interest.json in data-dir). " "Enabled by default.",
    )
    parser.add_argument(
        "--no-short-interest",
        action="store_false",
        dest="enable_short_interest",
        help="Disable short interest signals.",
    )

    parser.add_argument(
        "--enable-bootstrap",
        action="store_true",
        help="Enable bootstrap confidence intervals (deterministic)",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of bootstrap samples (default: 1000)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID for seed derivation (default: auto)",
    )

    # =========================================================================
    # Clinical Activity Filter
    # =========================================================================
    parser.add_argument(
        "--min-trials",
        type=int,
        default=1,
        help="Minimum number of clinical trials required for ranking (default: 1). "
        "Tickers with no trials are excluded from final rankings.",
    )
    parser.add_argument(
        "--min-phase",
        type=str,
        choices=["preclinical", "phase1", "phase2", "phase3", "approved"],
        default="preclinical",
        help="Minimum lead phase required for ranking (default: preclinical). "
        "Tickers with earlier-phase pipelines are excluded.",
    )
    parser.add_argument(
        "--no-clinical-filter",
        action="store_true",
        help="Disable clinical activity filter (include all tickers regardless of trials/phase).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )

    # =========================================================================
    # NEW: Governance-friendly diagnostic and tuning arguments
    # =========================================================================

    # Tier 0: Observability
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity level. DEBUG prints feature coverage tables, gating reasons, "
        "file paths, and per-component 'applied vs present' diagnostics. (default: INFO)",
    )

    parser.add_argument(
        "--diagnostics",
        type=str,
        choices=["none", "summary", "full"],
        default="summary",
        help="Diagnostics output mode. 'summary': one-line coverage + gating counts (current behavior). "
        "'full': dumps structured diagnostics to JSON sidecar. 'none': minimal output. (default: summary)",
    )

    parser.add_argument(
        "--diagnostics-out",
        type=Path,
        default=None,
        help="Path for diagnostics JSON output. Only used when --diagnostics=full. "
        "(default: <data-dir>/diagnostics_<run_id>.json)",
    )

    # Catalyst tuning
    parser.add_argument(
        "--catalyst-window",
        type=str,
        default=None,
        metavar="START-END",
        help="Catalyst event window as 'START-END' days (e.g., '15-45'). "
        "Events within this window are weighted highest. Overrides --catalyst-window-preset if both specified.",
    )

    parser.add_argument(
        "--catalyst-window-preset",
        type=str,
        choices=["tight", "standard", "wide"],
        default="standard",
        help="Catalyst window preset. tight=7-30, standard=15-45, wide=15-90. (default: standard)",
    )

    parser.add_argument(
        "--catalyst-decay",
        type=str,
        choices=["step", "linear", "exp"],
        default="step",
        help="Catalyst score decay function outside the window. "
        "'step': binary on/off (current behavior), 'linear': gradual taper, "
        "'exp': exponential decay with configurable half-life. (default: step)",
    )

    parser.add_argument(
        "--catalyst-half-life-days",
        type=int,
        default=30,
        metavar="DAYS",
        help="Half-life in days for exponential decay (only used with --catalyst-decay=exp). (default: 30)",
    )

    # Smart money controls
    parser.add_argument(
        "--enable-smart-money",
        action="store_true",
        help="Enable smart money signal in composite scoring. When enabled, includes 13F institutional "
        "momentum in coverage and gating reports even if it contributes zero.",
    )

    parser.add_argument(
        "--smart-money-source",
        type=str,
        choices=["13f", "internal", "auto"],
        default="auto",
        help="Smart money data source. '13f': force 13F file paths, 'internal': alternative feed, "
        "'auto': current behavior (search multiple locations). (default: auto)",
    )

    parser.add_argument(
        "--debug-smart-money",
        action="store_true",
        help="Print detailed smart money diagnostics: file paths resolved, rows loaded, "
        "ticker universe overlap, and gating reason counts.",
    )

    # Scoring mode
    parser.add_argument(
        "--scoring-mode",
        type=str,
        choices=["baker_style", "legacy", "enhanced", "default"],
        default="baker_style",
        help="Scoring mode. 'baker_style' (DEFAULT): fundamental-concentrated mode with thesis-first "
        "weighting, conviction×timing reinforcement, and thesis gating. "
        "'legacy': pre-baker mode with PoS but no thesis gating (alias: 'enhanced'). "
        "'default': auto-select based on data availability.",
    )

    # SEC 8-K live fetch
    parser.add_argument(
        "--sec-8k-live",
        action="store_true",
        help="Run SEC 8-K catalyst collector in live mode (fetches from EDGAR). "
        "Only allowed when --as-of-date equals today (PIT safety).",
    )
    parser.add_argument(
        "--refresh-price-history-live",
        action="store_true",
        default=False,
        help="Explicitly refresh price_history.csv from live yfinance before scoring. "
        "Only allowed when --as-of-date equals today (PIT safety).",
    )
    parser.add_argument(
        "--phase-scores-v1",
        action="store_true",
        default=False,
        help="Rollback to Wong reference phase scores (v1). " "Overrides the default v3 scores.",
    )
    parser.add_argument(
        "--phase-scores-v2",
        action="store_true",
        default=False,
        help="Use empirically calibrated phase scores (Spec 023). "
        "Opt-in only. ±2 capped, 90%% top-60 overlap gate passed.",
    )
    parser.add_argument(
        "--phase-scores-v3",
        action="store_true",
        default=True,
        help="Use survivorship-adjusted phase scores (Spec 024). "
        "DEFAULT-ON. Bounded caps (down=-2, up=+0), Phase 4 Wong fallback. "
        "9-date aggregate: mean top-60 overlap 99.43%%, max shift 13.",
    )
    parser.add_argument(
        "--sec-multi-form-mode",
        type=str,
        default="off",
        choices=["off", "cache_only", "live"],
        help="SEC multi-form (10-Q, 10-K, 6-K) catalyst mode: off (default), cache_only, or live.",
    )
    parser.add_argument(
        "--fda-regulatory-mode",
        type=str,
        default="off",
        choices=["off", "cache_only", "live"],
        help="FDA regulatory notices (Federal Register) mode: off (default), cache_only, or live.",
    )
    parser.add_argument(
        "--morningstar-mode",
        type=str,
        default="on",
        choices=["off", "on"],
        help="Morningstar data mode: off (skip vol enrichment + signal engine), on (default).",
    )

    # Clustering controls
    parser.add_argument(
        "--enable-clustering",
        action="store_true",
        help="Enable correlation/indication clustering. Adds cluster_id and cluster_size "
        "to output (does not affect ranks/scores).",
    )

    parser.add_argument(
        "--cluster-method",
        type=str,
        choices=["indication", "returns", "auto"],
        default="indication",
        help="Clustering method. 'indication': group by therapeutic indication (fast, safe). "
        "'returns': correlation-based (O(N^2), requires price history). "
        "'auto': try returns, fallback to indication. (default: indication)",
    )

    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.70,
        help="Correlation threshold for returns-based clustering. (default: 0.70)",
    )

    # Defensive overlay controls (OFF by default)
    parser.add_argument(
        "--no-defensive-multiplier",
        action="store_true",
        dest="no_defensive_multiplier",
        help="(Deprecated, now default) Disable defensive multiplier.",
    )
    parser.add_argument(
        "--apply-defensive-multiplier",
        action="store_true",
        help="Enable defensive multiplier (disabled by default).",
    )
    parser.add_argument(
        "--defensive-config",
        type=str,
        choices=["default", "aggressive"],
        default="default",
        help="Defensive overlay config when multiplier is applied. (default: default)",
    )
    parser.add_argument(
        "--defensive-cache",
        type=str,
        default=None,
        help="Optional PIT defensive features cache file (defensive_features_<AS_OF>.json). "
        "If provided, fills missing defensive_features only (no recompute).",
    )
    parser.add_argument(
        "--enable-position-sizing",
        action="store_true",
        default=False,
        help="Enable v2 position sizing using multi-horizon blended risk metrics. "
        "Requires v2 defensive cache (cache_version 2.0.0).",
    )

    parser.add_argument(
        "--pit-mode",
        type=str,
        choices=["strict", "degrade"],
        default="strict",
        help="PIT enforcement mode for trial_records. "
        "strict (default): fail if data is from the future. "
        "degrade: fall back to calendar-only catalysts with confidence downgrade.",
    )

    # Module 5 calibrated weights
    parser.add_argument(
        "--module5-weights",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to calibrated Module 5 weights JSON (e.g., data/module5_weights_m3.json). "
        "When provided, overrides default component weights. Allows negative weights. "
        "If file is missing or invalid, falls back to defaults with a warning.",
    )
    parser.add_argument(
        "--module5-weights-mode",
        choices=["global", "regime", "blended"],
        default="global",
        help="Weight selection mode: 'global' uses global weights (default, safe), "
        "'regime' uses pure regime-specific weights with global fallback, "
        "'blended' uses alpha-blended regime+global weights. "
        "Requires schema_version=2 bundle JSON for regime/blended modes.",
    )

    # Snapshot sanity checks
    parser.add_argument(
        "--compare-snapshots",
        action="store_true",
        help="Run hash + field-diff summary between current and prior CT.gov snapshots, "
        "even if diff-events are zero. Outputs file paths, line counts, SHA256, and field changes.",
    )

    parser.add_argument(
        "--snapshot-date-prior",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Override prior snapshot date for comparison (useful on weekends/missed runs). "
        "Default: day before --as-of-date.",
    )

    # Input freshness verification
    parser.add_argument(
        "--verify-against",
        type=Path,
        default=None,
        metavar="PREV_SCREEN.json",
        help="Compare current input file hashes against a previous screen result JSON. "
        "Emits warnings for any changed or new input files.",
    )

    # Coverage / ranking controls
    parser.add_argument(
        "--min-component-coverage",
        type=float,
        default=None,
        metavar="0..1",
        help="Minimum fraction of components required for a ticker to be rankable. "
        "E.g., 0.6 means ticker must have >=60%% of components applied.",
    )

    parser.add_argument(
        "--min-component-coverage-mode",
        type=str,
        choices=["applied", "present"],
        default="applied",
        help="Coverage counting mode. 'applied': respects confidence gating (recommended), "
        "'present': counts all present components regardless of gating. (default: applied)",
    )

    parser.add_argument(
        "--allow-weekend",
        action="store_true",
        default=False,
        help="Allow running on weekends/non-trading days (for research/testing only).",
    )

    parser.add_argument(
        "--validation-daily",
        action="store_true",
        default=False,
        help="Fast daily validation mode: degrade gracefully on missing PIT cache, "
        "skip live SEC/clinical refresh (use cached data only). Preserves PIT integrity, "
        "model hash, and data-quality checks. Use for forward-evidence capture, not model updates.",
    )

    args = parser.parse_args()

    # =========================================================================
    # Normalize aliases (canonicalize for logs + downstream logic)
    # =========================================================================
    if getattr(args, "scoring_mode", None) == "enhanced":
        args.scoring_mode = "legacy"

    # =========================================================================
    # Propagate --validation-daily implications (before logging level is set)
    # --validation-daily does NOT change the model, ranker, selector, or PIT
    # integrity checks. It only disables live fetches and degrades gracefully
    # when the PIT cache is missing instead of aborting.
    # =========================================================================
    if getattr(args, "validation_daily", False):
        if args.pit_mode == "strict":
            args.pit_mode = "degrade"
        args.sec_8k_live = False
        args.refresh_price_history_live = False
        # Print before logger is configured — this is intentional for visibility.
        print("[VALIDATION-DAILY] Fast validation mode: live fetches disabled, PIT degrades gracefully", flush=True)

    # =========================================================================
    # Configure logging level FIRST (before any logging calls)
    # =========================================================================
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    logger.setLevel(log_level)

    # =========================================================================
    # Validate argument combinations
    # =========================================================================
    if not args.dry_run:
        if args.output is None:
            args.output = _default_output_path(
                as_of_date=args.as_of_date,
                data_dir=Path(args.data_dir),
                snapshot_dir=getattr(args, "snapshot_dir", None),
            )
            logger.info(f"[OUTPUT] --output not provided; defaulting to {args.output}")
        try:
            _enforce_snapshot_output_overwrite_policy(
                args.output,
                as_of_date=args.as_of_date,
                data_dir=Path(args.data_dir),
                snapshot_dir=getattr(args, "snapshot_dir", None),
                force_overwrite=getattr(args, "force_overwrite", False),
            )
        except FileExistsError as exc:
            parser.error(str(exc))
        args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.resume_from and not args.checkpoint_dir:
        parser.error("--resume-from requires --checkpoint-dir")

    _ps_explicit = sum(
        [
            getattr(args, "phase_scores_v1", False),
            getattr(args, "phase_scores_v2", False),
            # v3 is default-on; only count as explicit if v1 or v2 also set
        ]
    )
    if _ps_explicit > 1:
        parser.error("--phase-scores-v1, --phase-scores-v2, and --phase-scores-v3 are mutually exclusive")
    if getattr(args, "phase_scores_v1", False) or getattr(args, "phase_scores_v2", False):
        args.phase_scores_v3 = False  # explicit v1/v2 overrides default v3

    # Validate catalyst window format if provided
    catalyst_window = None
    if args.catalyst_window:
        try:
            catalyst_window = parse_catalyst_window(args.catalyst_window)
            logger.info(f"Using explicit catalyst window: {catalyst_window[0]}-{catalyst_window[1]} days")
        except ValueError as e:
            parser.error(str(e))
    else:
        catalyst_window = CATALYST_WINDOW_PRESETS[args.catalyst_window_preset]
        logger.debug(
            f"Using catalyst window preset '{args.catalyst_window_preset}': {catalyst_window[0]}-{catalyst_window[1]} days"
        )

    # Log if explicit window overrides preset
    if args.catalyst_window and args.catalyst_window_preset != "standard":
        logger.info(f"Note: --catalyst-window overrides --catalyst-window-preset={args.catalyst_window_preset}")

    # Validate catalyst half-life
    if args.catalyst_decay == "exp" and args.catalyst_half_life_days <= 0:
        parser.error("--catalyst-half-life-days must be positive when using --catalyst-decay=exp")

    # Validate min-component-coverage
    if args.min_component_coverage is not None:
        if not (0.0 <= args.min_component_coverage <= 1.0):
            parser.error("--min-component-coverage must be between 0.0 and 1.0")

    # Validate snapshot-date-prior format if provided
    if args.snapshot_date_prior:
        try:
            validate_date_format(args.snapshot_date_prior, "snapshot-date-prior")
        except ValueError as e:
            parser.error(f"Invalid --snapshot-date-prior: {e}")

    # Validate live-source PIT safety: live fetches are current-date only.
    try:
        _require_current_as_of_date_for_live_modes(
            args.as_of_date,
            {
                "--sec-8k-live": args.sec_8k_live,
                "--sec-multi-form-mode=live": args.sec_multi_form_mode == "live",
                "--fda-regulatory-mode=live": args.fda_regulatory_mode == "live",
                "--refresh-price-history-live": args.refresh_price_history_live,
            },
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.verify_against and args.refresh_price_history_live:
        parser.error("--refresh-price-history-live cannot be combined with --verify-against")

    # Set diagnostics output path default if full diagnostics enabled
    diagnostics_out = args.diagnostics_out
    if args.diagnostics == "full" and diagnostics_out is None:
        run_id = args.run_id or f"screen_{args.as_of_date}"
        diagnostics_out = args.data_dir / f"diagnostics_{run_id}.json"
        logger.debug(f"Diagnostics output defaulting to: {diagnostics_out}")

    try:
        # Replay bundle: extract and override args before pipeline
        _replay_tmp_dir = None
        if args.replay_bundle:
            if not args.replay_bundle.exists():
                logger.error(f"Replay bundle not found: {args.replay_bundle}")
                return 1
            logger.info(f"[REPLAY BUNDLE] Extracting: {args.replay_bundle}")
            _rb = extract_replay_bundle(args.replay_bundle)
            _replay_tmp_dir = _rb["tmp_dir"]
            args.data_dir = _rb["data_dir"]
            # Auto-wire verify mode
            args.inputs_manifest = "verify"
            args.inputs_manifest_path = _rb["manifest_path"]
            # Auto-wire ruleset if present
            if _rb["ruleset_path"] and not args.ruleset:
                args.ruleset = _rb["ruleset_path"]
                logger.info(f"[REPLAY BUNDLE] Auto-wired ruleset: {_rb['ruleset_path']}")
            # Auto-wire cache dirs: set args AND symlink to default locations
            # (module_3_catalyst + event_ledger resolve via Path(__file__).parent / "cache")
            _repo_cache = Path(__file__).parent / "cache"
            for cache_key in ("ctgov_cache_dir", "sec_cache_dir", "fda_cache_dir"):
                bundle_dir = _rb.get(cache_key)
                if not bundle_dir:
                    continue
                setattr(args, cache_key, bundle_dir)
                logger.info(f"[REPLAY BUNDLE] Auto-wired {cache_key}: {bundle_dir}")
                # Symlink to default location so hardcoded resolvers find it
                subdir = cache_key.replace("_cache_dir", "")  # ctgov, sec, fda
                link_target = _repo_cache / subdir
                if not link_target.exists():
                    link_target.parent.mkdir(parents=True, exist_ok=True)
                    os.symlink(str(bundle_dir), str(link_target))
                    logger.info(f"[REPLAY BUNDLE] Symlinked {link_target} → {bundle_dir}")
            logger.info(f"[REPLAY BUNDLE] data_dir → {_rb['data_dir']}")

        # SECURITY: Validate data_dir (after replay-bundle auto-wiring)
        if not args.data_dir.exists():
            logger.error(f"Data directory not found: {args.data_dir}")
            return 1
        if args.data_dir.is_symlink():
            logger.error(f"Data directory is a symbolic link (security risk): {args.data_dir}")
            return 1

        # Handle dry-run mode
        if args.dry_run:
            logger.info(f"[DRY-RUN] Validating inputs for {args.as_of_date}...")
            validation = validate_inputs_dry_run(
                data_dir=args.data_dir,
                enable_coinvest=args.enable_coinvest,
            )

            logger.info("=" * 60)
            logger.info("DRY-RUN VALIDATION RESULTS")
            logger.info("=" * 60)
            logger.info(f"Data directory: {validation['data_dir']}")
            logger.info(f"Valid: {validation['valid']}")
            logger.info("")

            logger.info("Required files:")
            for filename, info in validation["required_files"].items():
                status = "OK" if info["exists"] else "MISSING"
                count = f" ({info.get('record_count', '?')} records)" if info["exists"] else ""
                logger.info(f"  {filename}: {status}{count}")

            logger.info("")
            logger.info("Content hashes:")
            for filename, hash_val in validation["content_hashes"].items():
                logger.info(f"  {filename}: {hash_val}")

            if validation["errors"]:
                logger.info("")
                logger.info("Errors:")
                for error in validation["errors"]:
                    logger.info(f"  - {error}")

            logger.info("=" * 60)

            return 0 if validation["valid"] else 1

        # Replay bundle out: force manifest write mode
        if args.replay_bundle_out and args.inputs_manifest == "off":
            args.inputs_manifest = "write"

        # Verify input freshness against previous screen if requested
        if args.verify_against:
            if not args.verify_against.exists():
                logger.error(f"--verify-against file not found: {args.verify_against}")
                return 1
            freshness_warnings = verify_input_freshness(args.verify_against, args.data_dir)
            if freshness_warnings:
                logger.warning(f"Input freshness check: {len(freshness_warnings)} warning(s)")
                for w in freshness_warnings:
                    logger.warning(f"  {w}")
            else:
                logger.info("Input freshness check: all hashes match previous run")

        # Trading day guard — prevent snapshots on weekends/holidays
        # (run_daily_production.py has its own gate, but run_screen.py can be called directly)
        from datetime import date as _date_cls

        _as_of = _date_cls.fromisoformat(args.as_of_date)
        if _as_of.weekday() >= 5 and not getattr(args, "allow_weekend", False):
            logger.error(
                f"NOT A TRADING DAY: {args.as_of_date} is {_as_of.strftime('%A')}. " f"Use --allow-weekend to override."
            )
            return 1

        _maybe_refresh_price_history_live(
            as_of_date=args.as_of_date,
            data_dir=args.data_dir,
            enabled=args.refresh_price_history_live,
        )

        # Run pipeline
        results = run_screening_pipeline(
            as_of_date=args.as_of_date,
            data_dir=args.data_dir,
            universe_tickers=args.tickers,
            enable_coinvest=args.enable_coinvest,
            enable_enhancements=args.enable_enhancements,
            enable_short_interest=args.enable_short_interest,
            checkpoint_dir=args.checkpoint_dir,
            resume_from=args.resume_from,
            audit_log_path=args.audit_log,
            # New governance-friendly parameters
            catalyst_window=catalyst_window,
            catalyst_decay=args.catalyst_decay,
            catalyst_half_life_days=args.catalyst_half_life_days,
            enable_smart_money=args.enable_smart_money,
            smart_money_source=args.smart_money_source,
            debug_smart_money=args.debug_smart_money,
            compare_snapshots=args.compare_snapshots,
            snapshot_date_prior=args.snapshot_date_prior,
            min_component_coverage=args.min_component_coverage,
            min_component_coverage_mode=args.min_component_coverage_mode,
            # Clinical activity filter parameters
            min_trials=args.min_trials,
            min_phase=args.min_phase,
            no_clinical_filter=args.no_clinical_filter,
            # Clustering parameters
            enable_clustering=args.enable_clustering,
            cluster_method=args.cluster_method,
            cluster_threshold=args.cluster_threshold,
            # Defensive overlay parameters (OFF by default unless --apply-defensive-multiplier)
            apply_defensive_multiplier=getattr(args, "apply_defensive_multiplier", False),
            defensive_config=args.defensive_config,
            defensive_cache=args.defensive_cache,
            enable_position_sizing=getattr(args, "enable_position_sizing", False),
            pit_mode=args.pit_mode,
            output_dir=args.output_dir if args.output_dir else args.data_dir,
            sec_8k_mode=("live" if args.sec_8k_live else "cache_only"),
            sec_multi_form_mode=args.sec_multi_form_mode,
            fda_regulatory_mode=args.fda_regulatory_mode,
            morningstar_mode=args.morningstar_mode,
            module5_weights_path=(Path(args.module5_weights) if getattr(args, "module5_weights", None) else None),
            module5_weights_mode=getattr(args, "module5_weights_mode", "global"),
            scoring_mode=getattr(args, "scoring_mode", None),
            inputs_manifest_mode=args.inputs_manifest,
            inputs_manifest_verify_path=getattr(args, "inputs_manifest_path", None),
            from_replay_bundle=bool(getattr(args, "replay_bundle", None)),
            ctgov_cache_dir=getattr(args, "ctgov_cache_dir", None),
            phase_scores_v1=getattr(args, "phase_scores_v1", False),
            phase_scores_v2=getattr(args, "phase_scores_v2", False),
            phase_scores_v3=getattr(args, "phase_scores_v3", True),
        )

        # Add bootstrap analysis if requested
        if args.enable_bootstrap:
            logger.info("[BOOTSTRAP] Computing confidence intervals...")
            results = add_bootstrap_analysis(
                results=results,
                as_of_date=args.as_of_date,
                data_dir=args.data_dir,
                n_bootstrap=args.bootstrap_samples,
                run_id=args.run_id,
            )
            if "error" not in results.get("bootstrap_analysis", {}):
                ba = results["bootstrap_analysis"]
                logger.info(f"  Mean score: {ba['mean']}")
                logger.info(f"  95% CI: [{ba['ci_lower']}, {ba['ci_upper']}]")
                logger.info(f"  Bootstrap samples: {ba['bootstrap_samples']}")

        # Production validation (non-blocking, warnings only)
        try:
            validation_config = {
                "cash_target": str(getattr(args, "cash_target", "0.10")),
                "top_n": getattr(args, "top_n", None),
            }
            validation_result = validate_screening_output(results, args.as_of_date, validation_config)
            results["production_validation"] = {
                "passed": validation_result["passed"],
                "failure_reasons": validation_result.get("failure_reasons", []),
                "config": validation_config,
            }
            if not validation_result["passed"]:
                logger.warning(
                    f"[VALIDATION] Some production validation checks failed: {validation_result.get('failure_reasons', [])}"
                )
        except Exception as e:
            logger.warning(f"[VALIDATION] Production validation error (non-blocking): {e}")
            results["production_validation"] = {
                "passed": None,
                "failure_reasons": [],
                "error": str(e),
            }

        # Load decision ruleset and patch CLI-only manifest inputs before any
        # output is written, so failed guardrails cannot leave partial outputs.
        de_ruleset = _load_decision_ruleset_for_cli(args)
        if args.decision_mode == "phase2" and args.inputs_manifest != "off":
            _rs_path = Path(args.ruleset) if args.ruleset else PHASE2_DEFAULT_RULESET_PATH
            _attach_phase2_decision_ruleset_manifest(results, _rs_path)
            if args.inputs_manifest == "verify":
                _verify_inputs_manifest_after_cli_patch(
                    results,
                    getattr(args, "inputs_manifest_path", None),
                    from_replay_bundle=bool(getattr(args, "replay_bundle", None)),
                )

        # Write output
        logger.info(f"[OUTPUT] Writing results to {args.output}")
        write_json_output(args.output, results)

        # Write diagnostics output if requested
        if args.diagnostics == "full":
            write_diagnostics_output(
                diagnostics_path=diagnostics_out,
                results=results,
                args=args,
                catalyst_window=catalyst_window,
            )

        # Print summary
        summary = results["summary"]
        logger.info("=" * 60)
        logger.info(f"SCREENING SUMMARY ({args.as_of_date})")
        logger.info("=" * 60)
        logger.info(f"Total evaluated:    {summary['total_evaluated']}")
        logger.info(f"Active universe:    {summary['active_universe']}")
        logger.info(f"Excluded:           {summary['excluded']}")
        logger.info(f"Final ranked:       {summary['final_ranked']}")
        logger.info(f"Catalyst events:    {summary.get('catalyst_events', 0)}")
        logger.info(f"Severe negatives:   {summary.get('severe_negatives', 0)}")
        if args.enable_enhancements:
            logger.info(f"Regime:             {summary.get('regime', 'N/A')}")
            logger.info(f"Regime confidence:  {summary.get('regime_confidence', 'N/A')}")
            logger.info(f"PoS mapped:         {summary.get('pos_indication_coverage', 'N/A')}")
            logger.info(f"PoS effective:      {summary.get('pos_effective_coverage', 'N/A')}")
            if args.enable_short_interest or summary.get("short_interest_coverage"):
                logger.info(f"SI coverage:        {summary.get('short_interest_coverage', 'N/A')}")
        if args.checkpoint_dir:
            logger.info(f"Checkpoints:        {args.checkpoint_dir}")
        if args.audit_log:
            logger.info(f"Audit log:          {args.audit_log}")
        logger.info("=" * 60)

        # Save validation snapshot for future forward-looking backtests
        if not args.no_snapshot:
            snapshot_dir = args.snapshot_dir or (args.data_dir.parent / "data" / "snapshots")
            prior_snapshot_dir = getattr(args, "prior_snapshot_dir", None) or snapshot_dir

            # Load trial_records for clinical_alpha_z computation
            # Use the same ctgov_cache_dir as the pipeline to avoid divergence
            _ctgov_dir = (
                Path(args.ctgov_cache_dir)
                if getattr(args, "ctgov_cache_dir", None)
                else Path(__file__).parent / "cache" / "ctgov"
            )
            _ctgov_snap = _ctgov_dir / f"trial_records_{args.as_of_date}.json"
            _ctgov_cache_date: Optional[str] = None
            if _ctgov_snap.exists():
                trial_records = json.loads(_ctgov_snap.read_text())
                _ctgov_cache_date = args.as_of_date
            else:
                _tr_path = args.data_dir / "trial_records.json"
                trial_records = json.loads(_tr_path.read_text()) if _tr_path.exists() else None

            # Build market_data lookup for cost-aware sizing in snapshot
            _mkt_data_for_snapshot = None
            if de_ruleset and de_ruleset.enable_cost_haircut:
                _mkt_path = args.data_dir / "market_data.json"
                if _mkt_path.exists():
                    try:
                        _mkt_records = json.loads(_mkt_path.read_text())
                        _mkt_data_for_snapshot = {
                            r["ticker"].upper(): r for r in _mkt_records if isinstance(r, dict) and "ticker" in r
                        }
                    except Exception as e:
                        logger.warning(f"Could not load market_data for cost sizing: {e}")

            snap_result = save_validation_snapshot(
                snapshot_dir=snapshot_dir,
                as_of_date=args.as_of_date,
                results=results,
                version=VERSION,
                data_dir=args.data_dir,
                decision_mode=args.decision_mode,
                ruleset=de_ruleset,
                price_history_path=args.data_dir / "price_history.csv",
                market_data_by_ticker=_mkt_data_for_snapshot,
                trial_records=trial_records,
                alpha_schema_mode=args.alpha_schema_mode,
                inputs_manifest_mode=args.inputs_manifest,
                ranking_mode=args.ranking_mode,
                prior_snapshot_dir=prior_snapshot_dir,
                ctgov_cache_date=_ctgov_cache_date,
                regime_result=results.get("regime_result"),
                phase_scores_version=(
                    "v1"
                    if getattr(args, "phase_scores_v1", False)
                    else ("v2" if getattr(args, "phase_scores_v2", False) else "v3")
                ),
                force_overwrite=getattr(args, "force_overwrite", False),
                ranker_mode=getattr(args, "ranker_mode", "pairwise_minimal"),
            )
            if snap_result:
                logger.info(f"Snapshot dir:       {snap_result}")

                # Create replay bundle if requested
                if args.replay_bundle_out:
                    _manifest = results.get("run_metadata", {}).get("inputs_manifest")
                    if _manifest:
                        create_replay_bundle(_manifest, args.replay_bundle_out)
                    else:
                        logger.warning("[REPLAY BUNDLE] No inputs_manifest in results — skipping bundle creation")

                # Long-call candidate report (Spec 025)
                if not getattr(args, "no_long_call_report", False):
                    try:
                        from scripts.research.build_long_call_candidates import run_from_screen as _run_long_call_report

                        _run_long_call_report(
                            snapshot_dir=snap_result,
                            data_dir=Path(args.data_dir),
                            as_of_date=args.as_of_date,
                        )
                    except Exception as _lc_exc:
                        logger.debug("[LONG_CALL] Skipped: %s", _lc_exc)

                # Data collection health report (Spec 026)
                if not getattr(args, "no_data_collection_health", False):
                    try:
                        from tools.build_data_collection_health import run_from_screen as _run_collection_health

                        _run_collection_health(
                            snapshot_dir=snap_result,
                            data_dir=Path(args.data_dir),
                            as_of_date=args.as_of_date,
                        )
                    except Exception as _ch_exc:
                        logger.debug("[COLLECTION_HEALTH] Skipped: %s", _ch_exc)

                # Phase-2 delta report hook
                if args.decision_mode == "phase2" and not getattr(args, "no_delta", False):
                    health = _run_phase2_delta(snap_result, snapshot_dir, args, logger)
                    if health is not None and getattr(args, "strict", False):
                        if health.status == "FAIL":
                            logger.error("--strict: Phase-2 health FAIL, exiting 1")
                            return 1
                        if health.status == "WARN":
                            logger.warning("--strict: Phase-2 health WARN, exiting 2")
                            return 2

                # Weekly readiness scorecard — policy-controlled gate
                try:
                    from tools.weekly_readiness_scorecard import (
                        DEFAULT_READINESS_POLICY,
                        ReadinessPolicy,
                        append_history,
                        build_scorecard,
                        evaluate_readiness_gate,
                        format_scorecard_md,
                        load_history,
                    )

                    _readiness_dir = Path(getattr(args, "data_dir", ".")).parent / "artifacts" / "readiness"
                    _sc = build_scorecard(
                        as_of_date=args.as_of_date,
                        snapshots_dir=snapshot_dir,
                        artifacts_dir=Path(getattr(args, "data_dir", ".")).parent / "artifacts" / "live_shadow",
                        policy_path=Path(getattr(args, "data_dir", ".")) / "portfolio_policy.json",
                        ruleset_id=getattr(de_ruleset, "id", ""),
                    )
                    _readiness_dir.mkdir(parents=True, exist_ok=True)
                    with open(_readiness_dir / f"scorecard_{args.as_of_date}.json", "w") as _f:
                        json.dump(_sc, _f, indent=2, default=str)
                        _f.write("\n")
                    with open(_readiness_dir / f"scorecard_{args.as_of_date}.md", "w") as _f:
                        _f.write(format_scorecard_md(_sc))

                    append_history(_readiness_dir / "history.jsonl", _sc)

                    # Evaluate gate verdict against policy
                    _rp = (
                        ReadinessPolicy.from_json(DEFAULT_READINESS_POLICY)
                        if DEFAULT_READINESS_POLICY.exists()
                        else ReadinessPolicy.default()
                    )
                    _hist = load_history(_readiness_dir / "history.jsonl")
                    _gate = evaluate_readiness_gate(_sc, _rp, _hist)

                    logger.info(
                        "[READINESS] %s (gate=%s) — %s",
                        _sc["verdict"],
                        _gate["gate_status"],
                        _readiness_dir / f"scorecard_{args.as_of_date}.md",
                    )

                    # In --strict mode, HOLD blocks the screen run
                    if getattr(args, "strict", False) and _gate["gate_status"] == "FAIL":
                        logger.error("--strict: readiness HOLD, exiting 1")
                        return 1
                except Exception as _exc:
                    logger.debug("Readiness scorecard skipped: %s", _exc)

            # --- Bioshort watch (read-only hedge monitor) — Spec 087 B0 freshness gate ---
            try:
                from common.bioshort_freshness import check_upstream_freshness, write_status_artifact

                _bw_repo_root = Path(__file__).resolve().parent
                _bw_report_dir = _bw_repo_root / "output" / "hedge_report"
                _bw_artifacts_dir = _bw_repo_root / "artifacts" / "bioshort_watch"
                _bw_freshness = check_upstream_freshness(_bw_report_dir)
                write_status_artifact(_bw_artifacts_dir, _bw_freshness)

                if _bw_freshness.status == "ORPHANED":
                    logger.info(
                        "[BIOSHORT_WATCH] SKIPPED_ORPHANED_UPSTREAM — no hedge_report_*.json in %s",
                        _bw_report_dir,
                    )
                elif _bw_freshness.status == "STALE":
                    logger.info(
                        "[BIOSHORT_WATCH] SKIPPED_STALE_UPSTREAM — latest=%s age=%dd threshold=%dd",
                        _bw_freshness.latest_as_of_date,
                        _bw_freshness.age_days,
                        _bw_freshness.threshold_days,
                    )
                    # Alert: hedge_report stale beyond threshold (producer restored in B1b)
                    try:
                        from common.alerts import send_operator_alert

                        send_operator_alert(
                            severity="WARN",
                            system="bioshort",
                            message=(
                                f"hedge_report stale: latest={_bw_freshness.latest_as_of_date} "
                                f"age={_bw_freshness.age_days}d threshold={_bw_freshness.threshold_days}d. "
                                f"bioshort_watch suppressed. Check hedge_report producer cron."
                            ),
                            dedupe_key=(f"bioshort:hedge_report_stale:{_bw_freshness.latest_as_of_date}"),
                        )
                    except Exception as _alert_exc:
                        logger.debug("Operator alert skipped: %s", _alert_exc)
                else:
                    from tools.build_bioshort_watch import build_bioshort_watch

                    _bw = build_bioshort_watch(as_of_date=args.as_of_date)
                    if "error" not in _bw:
                        logger.info(
                            "[BIOSHORT_WATCH] %s — %d alerts (level=%s)",
                            _bw.get("current_date", "?"),
                            _bw.get("n_alerts", 0),
                            _bw.get("alert_level", "?"),
                        )
                        for _alert in _bw.get("alerts", []):
                            logger.info("[BIOSHORT_WATCH]   %s", _alert)
                    else:
                        logger.debug("[BIOSHORT_WATCH] skipped: %s", _bw["error"])
            except Exception as _bw_exc:
                logger.debug("[BIOSHORT_WATCH] skipped: %s", _bw_exc)

            # --- Policy shadow compare (Spec 035) ---
            try:
                from tools.build_policy_shadow_compare import build_policy_shadow_compare

                _ps = build_policy_shadow_compare(as_of_date=args.as_of_date)
                if "error" not in _ps:
                    _pnl = _ps.get("daily_pnl_pct", {})
                    logger.info(
                        "[POLICY_SHADOW] current=%.2f%% tiered=%.2f%% exit=%.2f%% | excluded=%s",
                        _pnl.get("current", 0),
                        _pnl.get("tiered", 0),
                        _pnl.get("tiered_exit", 0),
                        _ps.get("excluded_by_exit") or "none",
                    )
                else:
                    logger.warning("[POLICY_SHADOW] skipped: %s", _ps["error"])
            except Exception as _ps_exc:
                logger.warning("[POLICY_SHADOW] skipped: %s", _ps_exc)

            # --- Policy candidate evaluation (Spec 035) ---
            try:
                from tools.eval_policy_candidate import evaluate_candidate, load_history

                _hist_path = Path("artifacts/policy_shadow/tier_weighted/history.jsonl")
                if _hist_path.exists():
                    _hist = load_history(_hist_path)
                    _eval = evaluate_candidate(_hist)
                    logger.info(
                        "[POLICY_CANDIDATE] %s — %s (%d days)",
                        _eval.get("verdict", "?"),
                        _eval.get("reason", ""),
                        _eval.get("n_trading_days", 0),
                    )
            except Exception as _pc_exc:
                logger.debug("[POLICY_CANDIDATE] skipped: %s", _pc_exc)

        return 0

    except (PathTraversalError, SymlinkError) as e:
        # SECURITY: Log security-related errors without exposing details
        logger.error(f"SECURITY ERROR: Path validation failed - {type(e).__name__}")
        logger.debug(f"Security error details: {e}")  # Only in debug mode
        return 3
    except IntegrityError as e:
        logger.error(f"INTEGRITY ERROR: Data integrity check failed")
        logger.debug(f"Integrity error details: {e}")
        return 4
    except FileSizeError as e:
        logger.error(f"FILE SIZE ERROR: {e}")
        return 5
    except OperationTimeoutError as e:
        logger.error(f"TIMEOUT ERROR: {e}")
        return 6
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        # Sanitize error message to avoid logging sensitive data
        error_msg = sanitize_for_logging(str(e), max_string_length=200)
        logger.error(f"ERROR: {error_msg}")
        return 1
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        # Log sanitized error to avoid exposing sensitive information
        error_type = type(e).__name__
        error_msg = sanitize_for_logging(str(e), max_string_length=200)
        logger.exception(f"UNEXPECTED ERROR ({error_type}): {error_msg}")
        return 2
    finally:
        # Clean up replay bundle temp dir if it was created
        _tmp = locals().get("_replay_tmp_dir")
        if _tmp and Path(_tmp).is_dir():
            shutil.rmtree(_tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
