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
import json
import logging
import sys
from datetime import date
import hashlib
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging with rotation support
from logging.handlers import RotatingFileHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Production hardening utilities
from common.production_hardening import (
    # Path security
    validate_path_within_base,
    safe_join_path,
    validate_checkpoint_path,
    PathTraversalError,
    SymlinkError,
    # File operations
    validate_file_size,
    safe_read_json,
    safe_write_json,
    safe_mkdir,
    FileSizeError,
    # Integrity
    save_with_integrity,
    verify_integrity,
    load_with_integrity_check,
    IntegrityError,
    compute_content_hash,
    # Timeouts
    operation_timeout,
    OperationTimeoutError,
    # Logging
    sanitize_for_logging,
    # Validation
    validate_date_format,
    validate_numeric_bounds,
    # Resources
    require_minimum_memory,
    # Constants
    MAX_JSON_FILE_SIZE_MB,
    DEFAULT_MODULE_EXECUTION_TIMEOUT,
    DEFAULT_PIPELINE_TIMEOUT,
)

# Common utilities
from common.date_utils import normalize_date, to_date_string, to_date_object
from common.integration_contracts import (
    validate_pipeline_handoff,
    validate_module_5_output,
    SchemaValidationError,
)
from common.data_integration_contracts import (
    safe_numeric_check,
    validate_market_data_schema,
    validate_financial_records_schema,
    normalize_ticker_set,
)

# Module imports
from module_1_universe import compute_module_1_universe
from module_2_financial import compute_module_2_financial
from module_3_catalyst import compute_module_3_catalyst, Module3Config
from module_4_clinical_dev import compute_module_4_clinical_dev
from module_5_composite_with_defensive import compute_module_5_composite_with_defensive

# Module 3A specific imports
from event_detector import SimpleMarketCalendar

# Enhancement modules (optional)
try:
    from pos_engine import ProbabilityOfSuccessEngine
    from short_interest_engine import ShortInterestSignalEngine
    from regime_engine import RegimeDetectionEngine
    from indication_mapper import IndicationMapper
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

# Optional: Risk gates for audit trail
try:
    from risk_gates import get_parameters_snapshot as get_risk_params, compute_parameters_hash as risk_params_hash
    HAS_RISK_GATES = True
except ImportError as e:
    HAS_RISK_GATES = False
    logger.info(f"Risk gates not available: {e}")

try:
    from liquidity_scoring import get_parameters_snapshot as get_liq_params, compute_parameters_hash as liq_params_hash
    HAS_LIQUIDITY_SCORING = True
except ImportError as e:
    HAS_LIQUIDITY_SCORING = False
    logger.info(f"Liquidity scoring not available: {e}")

# Ticker validation for fail-loud data quality
try:
    from src.validators.ticker_validator import validate_ticker_list
    HAS_TICKER_VALIDATION = True
except ImportError as e:
    HAS_TICKER_VALIDATION = False
    logger.warning(f"Ticker validation module not available - skipping validation: {e}")

VERSION = "1.6.0"  # Bumped for governance-friendly CLI enhancements
DETERMINISTIC_TIMESTAMP_SUFFIX = "T00:00:00Z"

# =============================================================================
# CATALYST WINDOW PRESETS AND DECAY FUNCTIONS
# =============================================================================

CATALYST_WINDOW_PRESETS = {
    "tight": (7, 30),
    "standard": (15, 45),
    "wide": (15, 90),
}

# =============================================================================
# CLINICAL ACTIVITY FILTER
# =============================================================================

# Phase ordering for minimum phase filter (lower index = earlier phase)
PHASE_ORDER = {
    "preclinical": 0,
    "phase1": 1,
    "phase 1": 1,
    "phase1/phase2": 1,
    "phase2": 2,
    "phase 2": 2,
    "phase2/phase3": 2,
    "phase3": 3,
    "phase 3": 3,
    "approved": 4,
    "nda/bla": 4,
}


def apply_clinical_activity_filter(
    m4_scores: List[Dict[str, Any]],
    min_trials: int = 5,
    min_phase: str = "phase1",
) -> tuple[List[str], List[Dict[str, Any]]]:
    """
    Filter tickers based on clinical activity thresholds.

    Args:
        m4_scores: Module 4 clinical scores list
        min_trials: Minimum number of trials required
        min_phase: Minimum lead phase required (preclinical, phase1, phase2, phase3, approved)

    Returns:
        Tuple of (excluded_tickers, exclusion_details)
    """
    min_phase_order = PHASE_ORDER.get(min_phase.lower(), 1)

    excluded = []
    exclusion_details = []

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
            excluded.append(ticker)
            exclusion_details.append({
                "ticker": ticker,
                "reason": "clinical_activity_filter",
                "details": ", ".join(reasons),
                "trial_count": trial_count,
                "lead_phase": lead_phase,
            })

    return excluded, exclusion_details


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
    if not window_str or '-' not in window_str:
        raise ValueError(f"Invalid catalyst window format: '{window_str}'. Expected 'START-END' (e.g., '15-45')")

    parts = window_str.split('-')
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
        if days_to_event < window_start:
            # Too soon - exponential approach
            distance = window_start - days_to_event
            return math.exp(-0.693 * distance / half_life_days)  # 0.693 = ln(2)
        else:
            # Beyond window_end - exponential decay
            distance = days_to_event - window_end
            return math.exp(-0.693 * distance / half_life_days)

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


def load_json_data(
    filepath: Path,
    description: str,
    max_size_mb: float = MAX_JSON_FILE_SIZE_MB,
    base_dir: Optional[Path] = None
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
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
    except UnicodeDecodeError as e:
        raise ValueError(f"{description} file is not valid UTF-8: {filepath}") from e

    if not isinstance(data, list):
        raise ValueError(f"{description} must be a JSON array, got {type(data)}")

    logger.debug(f"Loaded {description}: {len(data)} records from {filepath}")
    return data


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
                        manager_map[cik] = {
                            "name": mgr.get("name", f"Manager_{cik[-4:]}"),
                            "tier": 1
                        }
                # Conditional managers are Tier 2
                for mgr in registry.get("conditional", []):
                    cik = mgr.get("cik", "").zfill(10)
                    if cik:
                        manager_map[cik] = {
                            "name": mgr.get("name", f"Manager_{cik[-4:]}"),
                            "tier": 2
                        }
                return manager_map
            except Exception as e:
                logger.warning(f"Failed to load manager registry from {registry_path}: {e}")

    return {}


def _convert_holdings_to_coinvest(holdings_snapshots: Dict[str, Any], data_dir: Path = None) -> Dict[str, Dict]:
    """
    Convert holdings_snapshots.json format to coinvest_signals format for Module 5.

    holdings_snapshots format:
    {
        "TICKER": {
            "holdings": {
                "current": {
                    "MANAGER_CIK": {"value_kusd": 123456},
                    ...
                },
                "prior": {
                    "MANAGER_CIK": {"value_kusd": 100000},
                    ...
                }
            }
        }
    }

    coinvest_signals format expected by Module 5:
    {
        "TICKER": {
            "coinvest_overlap_count": int,
            "coinvest_holders": [str, ...],
            "position_changes": {"CIK": "NEW"|"INCREASE"|"DECREASE"|"EXIT"|"HOLD"},
            "holder_tiers": {...}
        }
    }

    Position change types:
    - NEW: Manager in current but not in prior (new position initiated)
    - INCREASE: Manager increased position value by >10%
    - DECREASE: Manager decreased position value by >10%
    - EXIT: Manager in prior but not in current (position closed)
    - HOLD: Position relatively unchanged (±10%)
    """
    # Load manager registry for name resolution
    MANAGER_REGISTRY = _load_manager_registry(data_dir)
    if MANAGER_REGISTRY:
        logger.info(f"  Loaded {len(MANAGER_REGISTRY)} managers from registry for name resolution")

    # Threshold for detecting meaningful position changes (10%)
    CHANGE_THRESHOLD = 0.10

    coinvest_signals = {}

    for ticker, ticker_data in holdings_snapshots.items():
        if not isinstance(ticker_data, dict):
            continue

        holdings = ticker_data.get("holdings", {})
        current = holdings.get("current", {})
        prior = holdings.get("prior", {})

        # Normalize ticker to uppercase for consistent lookups
        ticker = ticker.upper()

        if not current:
            continue

        # Count managers and identify tiers
        # FIX: Use holder names (not CIKs) and proper Dict[name -> int] format
        # for compatibility with compute_smart_money_signal()
        holder_names = []  # Current holders by name
        holder_tiers = {}  # Dict[name -> tier_int] (not Dict[cik -> dict])
        position_changes = {}  # Dict[name -> change_type]

        # All CIKs that appear in either current or prior
        all_ciks = set(current.keys()) | set(prior.keys())

        for cik in all_ciks:
            # Determine tier and resolve holder name from registry
            if cik in MANAGER_REGISTRY:
                holder_name = MANAGER_REGISTRY[cik]["name"]
                tier = MANAGER_REGISTRY[cik]["tier"]
            else:
                holder_name = f"Manager_{cik[-4:]}"
                tier = 2

            # Track current holders by name (not CIK)
            if cik in current:
                holder_names.append(holder_name)

            # Store tier by holder name (int, not dict)
            holder_tiers[holder_name] = tier

            # Calculate position change
            current_val = current.get(cik, {}).get("value_kusd", 0) or 0
            prior_val = prior.get(cik, {}).get("value_kusd", 0) or 0

            if cik in current and cik not in prior:
                # New position
                position_changes[holder_name] = "NEW"
            elif cik not in current and cik in prior:
                # Exited position
                position_changes[holder_name] = "EXIT"
            elif prior_val > 0:
                # Calculate percentage change
                change_pct = (current_val - prior_val) / prior_val

                if change_pct > CHANGE_THRESHOLD:
                    position_changes[holder_name] = "INCREASE"
                elif change_pct < -CHANGE_THRESHOLD:
                    position_changes[holder_name] = "DECREASE"
                else:
                    position_changes[holder_name] = "HOLD"
            else:
                # Prior was zero but current exists (edge case)
                if current_val > 0:
                    position_changes[holder_name] = "NEW"
                else:
                    position_changes[holder_name] = "HOLD"

        coinvest_signals[ticker] = {
            "coinvest_overlap_count": len(holder_names),
            "coinvest_holders": holder_names,
            "position_changes": position_changes,
            "holder_tiers": holder_tiers,
        }

    return coinvest_signals


def compute_momentum_from_price_history(
    price_history_path: Path,
    as_of_date: str,
    market_data_by_ticker: Dict[str, Dict],
    xbi_ticker: str = "XBI"
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

    with open(price_history_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row.get('ticker', '').upper()
            date_str = row.get('date', '')
            close_str = row.get('close', '')

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
        logger.info(f"XBI benchmark returns: 20d={xbi_return_20d:.2%}, 60d={xbi_return_60d:.2%}, 120d={xbi_return_120d:.2%}")
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
            if hasattr(obj, 'to_dict'):
                return obj.to_dict()
            if hasattr(obj, '__dataclass_fields__'):
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
    json_content = json.dumps(
        data,
        indent=2,
        sort_keys=True,  # Deterministic key ordering
        ensure_ascii=False,
        cls=DataclassEncoder,
    ) + '\n'  # Trailing newline for diff-friendliness

    if secure:
        # SECURITY: Write atomically with secure permissions
        import tempfile
        import os

        fd, tmp_path = tempfile.mkstemp(
            dir=filepath.parent,
            prefix='.tmp_output_',
            suffix='.json'
        )
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
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
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(json_content)


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
    result["current_line_count"] = current_bytes.count(b'\n')
    result["prior_line_count"] = prior_bytes.count(b'\n')

    # Check if files are identical
    if result["current_sha256"] == result["prior_sha256"]:
        result["files_identical"] = True
        result["comparison_status"] = "IDENTICAL"
        return result

    # Parse and compare records
    def parse_jsonl(file_path: Path) -> Dict[str, Dict]:
        records = {}
        with open(file_path, 'r', encoding='utf-8') as f:
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
# CHECKPOINTING
# =============================================================================

CHECKPOINT_MODULES = ["module_1", "module_2", "module_3", "module_4", "enhancements", "module_5"]


def save_checkpoint(
    checkpoint_dir: Path,
    module_name: str,
    as_of_date: str,
    data: Dict[str, Any]
) -> Path:
    """
    Save module checkpoint to disk with integrity metadata.

    Args:
        checkpoint_dir: Directory for checkpoints
        module_name: Module identifier (e.g., "module_1")
        as_of_date: Analysis date
        data: Module output data

    Returns:
        Path to checkpoint file

    Raises:
        PathTraversalError: If path components are invalid
    """
    # SECURITY: Validate checkpoint path to prevent directory traversal
    try:
        filepath = validate_checkpoint_path(checkpoint_dir, module_name, as_of_date)
    except (PathTraversalError, ValueError) as e:
        raise PathTraversalError(
            f"Invalid checkpoint path: module={module_name}, date={as_of_date}: {e}"
        ) from e

    # Create checkpoint directory with secure permissions
    safe_mkdir(checkpoint_dir, mode=0o700)

    checkpoint_data = {
        "module": module_name,
        "as_of_date": as_of_date,
        "version": VERSION,
        "data": data,
    }

    # INTEGRITY: Add content hash for verification on load
    data_json = json.dumps(data, sort_keys=True, default=str)
    checkpoint_data["_content_hash"] = compute_content_hash(data_json)

    # Write atomically with secure permissions
    safe_write_json(filepath, checkpoint_data, mode=0o600)
    logger.debug(f"Checkpoint saved with integrity hash: {filepath}")
    return filepath


def load_checkpoint(
    checkpoint_dir: Path,
    module_name: str,
    as_of_date: str,
    verify_integrity: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Load module checkpoint from disk with integrity verification.

    Args:
        checkpoint_dir: Directory for checkpoints
        module_name: Module identifier
        as_of_date: Analysis date
        verify_integrity: Whether to verify content hash (default True)

    Returns:
        Module output data, or None if checkpoint not found

    Raises:
        IntegrityError: If integrity verification fails
        PathTraversalError: If path components are invalid
    """
    # SECURITY: Validate checkpoint path to prevent directory traversal
    try:
        filepath = validate_checkpoint_path(checkpoint_dir, module_name, as_of_date)
    except (PathTraversalError, ValueError) as e:
        logger.warning(f"Invalid checkpoint path: {e}")
        return None

    if not filepath.exists():
        return None

    # SECURITY: Check for symlinks
    if filepath.is_symlink():
        logger.warning(f"Checkpoint is a symlink (security risk), ignoring: {filepath}")
        return None

    # SECURITY: Validate file size
    try:
        validate_file_size(filepath, MAX_JSON_FILE_SIZE_MB)
    except FileSizeError as e:
        logger.warning(f"Checkpoint file too large: {e}")
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        checkpoint_data = json.load(f)

    # Validate checkpoint version compatibility
    checkpoint_version = checkpoint_data.get("version", "0.0.0")
    if checkpoint_version.split(".")[0] != VERSION.split(".")[0]:
        logger.warning(
            f"Checkpoint version mismatch: {checkpoint_version} vs {VERSION}. "
            "Ignoring checkpoint."
        )
        return None

    # INTEGRITY: Verify content hash if present
    if verify_integrity and "_content_hash" in checkpoint_data:
        data = checkpoint_data.get("data", {})
        data_json = json.dumps(data, sort_keys=True, default=str)
        computed_hash = compute_content_hash(data_json)
        stored_hash = checkpoint_data["_content_hash"]

        if computed_hash != stored_hash:
            logger.error(
                f"Checkpoint integrity check FAILED: {filepath}. "
                f"Expected {stored_hash}, got {computed_hash}"
            )
            raise IntegrityError(f"Checkpoint corrupted or tampered: {filepath}")

        logger.debug(f"Checkpoint integrity verified: {filepath}")

    logger.info(f"Loaded checkpoint: {filepath}")
    return checkpoint_data.get("data")


def get_resume_module_index(resume_from: Optional[str]) -> int:
    """
    Get the index of the module to resume from.

    Args:
        resume_from: Module name to resume from (e.g., "module_3")

    Returns:
        Index in CHECKPOINT_MODULES (0 = start from beginning)
    """
    if resume_from is None:
        return 0

    try:
        return CHECKPOINT_MODULES.index(resume_from)
    except ValueError:
        logger.warning(f"Unknown module '{resume_from}', starting from beginning")
        return 0


# =============================================================================
# DRY-RUN VALIDATION
# =============================================================================

def validate_inputs_dry_run(data_dir: Path, enable_coinvest: bool = False) -> Dict[str, Any]:
    """
    Validate all required input files exist without running pipeline.

    Args:
        data_dir: Directory containing input data files
        enable_coinvest: Whether co-invest signals are required

    Returns:
        Dict with validation results
    """
    required_files = [
        ("universe.json", "Universe data"),
        ("financial_records.json", "Financial records"),
        ("trial_records.json", "Clinical trial records"),
        ("market_data.json", "Market data"),
    ]

    optional_files = [
        ("coinvest_signals.json", "Co-invest signals"),
    ]

    results = {
        "valid": True,
        "data_dir": str(data_dir),
        "required_files": {},
        "optional_files": {},
        "content_hashes": {},
        "errors": [],
    }

    # Check required files
    for filename, description in required_files:
        filepath = data_dir / filename
        exists = filepath.exists()
        results["required_files"][filename] = {
            "exists": exists,
            "description": description,
        }

        if exists:
            # Compute content hash
            content_hash = hashlib.sha256(filepath.read_bytes()).hexdigest()[:16]
            results["content_hashes"][filename] = content_hash

            # Try to load and count records
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    results["required_files"][filename]["record_count"] = len(data)
            except (json.JSONDecodeError, Exception) as e:
                results["required_files"][filename]["error"] = str(e)
                results["errors"].append(f"{filename}: {e}")
                results["valid"] = False
        else:
            results["errors"].append(f"Required file missing: {filename}")
            results["valid"] = False

    # Check optional files
    for filename, description in optional_files:
        filepath = data_dir / filename
        exists = filepath.exists()
        results["optional_files"][filename] = {
            "exists": exists,
            "description": description,
            "required": (filename == "coinvest_signals.json" and enable_coinvest),
        }

        if exists:
            content_hash = hashlib.sha256(filepath.read_bytes()).hexdigest()[:16]
            results["content_hashes"][filename] = content_hash
        elif filename == "coinvest_signals.json" and enable_coinvest:
            results["errors"].append(f"Co-invest enabled but {filename} missing")
            results["valid"] = False

    return results


# =============================================================================
# AUDIT TRAIL
# =============================================================================

def create_audit_record(
    as_of_date: str,
    data_dir: Path,
    content_hashes: Dict[str, str],
) -> Dict[str, Any]:
    """
    Create comprehensive audit record for the run.

    Args:
        as_of_date: Analysis date
        data_dir: Data directory path
        content_hashes: Dict of filename -> content hash

    Returns:
        Audit record dict
    """
    audit = {
        "as_of_date": as_of_date,
        "orchestrator_version": VERSION,
        "data_dir": str(data_dir),
        "input_hashes": dict(sorted(content_hashes.items())),
        "parameter_snapshots": {},
        "parameter_hashes": {},
    }

    # Add risk gates parameters if available
    if HAS_RISK_GATES:
        audit["parameter_snapshots"]["risk_gates"] = get_risk_params()
        audit["parameter_hashes"]["risk_gates"] = risk_params_hash()

    # Add liquidity scoring parameters if available
    if HAS_LIQUIDITY_SCORING:
        audit["parameter_snapshots"]["liquidity_scoring"] = get_liq_params()
        audit["parameter_hashes"]["liquidity_scoring"] = liq_params_hash()

    return audit


def append_audit_log(audit_log_path: Path, record: Dict[str, Any]) -> None:
    """
    Append audit record to JSONL log file.

    Args:
        audit_log_path: Path to audit log file
        record: Audit record to append
    """
    audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    line = json.dumps(record, sort_keys=True, separators=(',', ':'))
    with open(audit_log_path, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


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

    logger.info(f"[{as_of_date}] Starting screening pipeline...")
    logger.info(f"  Data directory: {data_dir}")
    if checkpoint_dir:
        logger.info(f"  Checkpoint directory: {checkpoint_dir}")
    if resume_from:
        logger.info(f"  Resuming from: {resume_from}")

    # Determine resume index
    resume_index = get_resume_module_index(resume_from)

    # Compute content hashes for audit trail
    # DETERMINISM: Exclude run_log files (they contain timestamps that change each run)
    content_hashes = {}
    for json_file in sorted(data_dir.glob("*.json")):
        if json_file.name.startswith("run_log"):
            continue  # Skip run logs - they're outputs, not inputs
        content_hashes[json_file.name] = hashlib.sha256(
            json_file.read_bytes()
        ).hexdigest()[:16]

    # Create audit record if audit log path provided
    if audit_log_path:
        audit_record = create_audit_record(as_of_date, data_dir, content_hashes)
        append_audit_log(audit_log_path, audit_record)
        logger.info(f"  Audit record written to: {audit_log_path}")

    # Load input data
    logger.info("[1/7] Loading input data...")
    raw_universe = load_json_data(data_dir / "universe.json", "Universe")

    # Extract full universe tickers BEFORE any filtering (for Module 3 stability)
    # This ensures Module 3 always processes the same population regardless of Module 1 filtering
    full_universe_tickers = frozenset(
        r.get('ticker') for r in raw_universe if r.get('ticker')
    )
    # Compute universe hash for state file namespacing (prevents churn on population changes)
    universe_hash = hashlib.sha256(
        json.dumps(sorted(full_universe_tickers)).encode()
    ).hexdigest()[:8]
    logger.info(f"  Full universe: {len(full_universe_tickers)} tickers (hash: {universe_hash})")

    # Validate tickers in universe (fail-loud on contamination)
    if HAS_TICKER_VALIDATION:
        universe_tickers_to_validate = [r.get('ticker') for r in raw_universe if r.get('ticker')]
        validation_result = validate_ticker_list(universe_tickers_to_validate)
        if validation_result['invalid']:
            invalid_sample = list(validation_result['invalid'].items())[:5]
            raise ValueError(
                f"Universe contains {len(validation_result['invalid'])} invalid tickers. "
                f"Examples: {invalid_sample}. "
                f"Run: python src/scripts/clean_universe.py to fix."
            )
        logger.info(f"  Universe validation passed: {validation_result['stats']['valid_count']} valid tickers")

    financial_records = load_json_data(data_dir / "financial_records.json", "Financial")
    trial_records = load_json_data(data_dir / "trial_records.json", "Trials")
    market_records = load_json_data(data_dir / "market_data.json", "Market data")

    # Convert market_records list to dict keyed by ticker for Module 5
    # This enables volatility adjustment, momentum signal, and other enhancements
    # IMPORTANT: Normalize ticker keys to uppercase for consistent lookups
    # (Module 5 uses .upper() for ticker lookups)
    market_data_by_ticker = {}
    if market_records:
        for record in market_records:
            if isinstance(record, dict) and 'ticker' in record:
                ticker = record['ticker'].upper()  # Normalize to uppercase for consistent lookups
                market_data_by_ticker[ticker] = record
        logger.info(f"  Market data indexed for {len(market_data_by_ticker)} tickers")

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

    coinvest_signals = None
    if enable_coinvest:
        # Try loading coinvest_signals.json first
        coinvest_file = data_dir / "coinvest_signals.json"
        if coinvest_file.exists():
            logger.info("  Loading co-invest signals...")
            coinvest_signals = load_json_data(coinvest_file, "Co-invest signals")
        else:
            # Fallback: try loading holdings_snapshots.json and convert to coinvest format
            holdings_file = data_dir / "holdings_snapshots.json"
            if holdings_file.exists():
                logger.info("  Loading holdings_snapshots.json for smart money signals...")
                # Holdings snapshots is a dict keyed by ticker, not an array
                with open(holdings_file, 'r', encoding='utf-8') as f:
                    holdings_snapshots = json.load(f)
                if holdings_snapshots and isinstance(holdings_snapshots, dict):
                    # Convert holdings format to coinvest_signals format
                    coinvest_signals = _convert_holdings_to_coinvest(holdings_snapshots, data_dir)
                    logger.info(f"  Converted holdings to coinvest signals for {len(coinvest_signals)} tickers")
            else:
                logger.warning(f"  --enable-coinvest specified but neither coinvest_signals.json nor holdings_snapshots.json found")

    # Enhancement data (optional)
    market_snapshot = None
    short_interest_data = None

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
            with open(snapshot_file, 'r', encoding='utf-8') as f:
                market_snapshot = json.load(f)
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
        with open(momentum_file, 'r', encoding='utf-8') as f:
            institutional_momentum = json.load(f)
        logger.info(f"  Institutional momentum: {len(institutional_momentum.get('rankings', []))} tickers scored")
    else:
        # Also check root directory
        root_momentum_file = Path("momentum_results.json")
        if root_momentum_file.exists():
            logger.info("  Loading 13F institutional momentum data from root...")
            with open(root_momentum_file, 'r', encoding='utf-8') as f:
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

    # Module 2: Financial health
    logger.info("[3/7] Module 2: Financial health...")
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

    diag = m2_result.get('diagnostic_counts', {})
    logger.info(f"  Scored: {diag.get('scored', len(m2_result.get('scores', [])))}, "
                f"Missing: {diag.get('missing', 'N/A')}")

    # ========================================================================
    # Module 3: Catalyst Detection (NEW: Delta-based event detection)
    # ========================================================================

    logger.info("[4/7] Module 3: Catalyst detection...")
    m3_result = None
    if resume_index > 2 and checkpoint_dir:
        m3_result = load_checkpoint(checkpoint_dir, "module_3", as_of_date)

    if m3_result is None:
        # Convert as_of_date string to date object for Module 3
        as_of_date_obj = to_date_object(as_of_date)

        # Create state directory namespaced by universe hash
        # This prevents churn gate from firing when Module 1 filtering changes
        # because snapshots from different universe populations are kept separate
        state_dir = data_dir / "ctgov_state" / universe_hash
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

            if snapshot_comparison['comparison_status'] in ("OK", "IDENTICAL"):
                logger.info(f"  Current SHA256: {snapshot_comparison['current_sha256'][:16]}...")
                logger.info(f"  Prior SHA256:   {snapshot_comparison['prior_sha256'][:16]}...")
                logger.info(f"  Current lines:  {snapshot_comparison['current_line_count']}")
                logger.info(f"  Prior lines:    {snapshot_comparison['prior_line_count']}")

                if snapshot_comparison['files_identical']:
                    logger.warning("  ⚠️  Files are IDENTICAL (same SHA256)")
                else:
                    logger.info(f"  Records added:   {snapshot_comparison['records_added']}")
                    logger.info(f"  Records removed: {snapshot_comparison['records_removed']}")
                    logger.info(f"  Field changes:   {snapshot_comparison['field_changes_detected']}")
                    if snapshot_comparison['top_changed_fields']:
                        logger.info(f"  Top changed fields: {snapshot_comparison['top_changed_fields']}")
            logger.info("  " + "=" * 56)

        # Run Module 3A with FULL UNIVERSE tickers (not post-Module-1 filtered)
        # This ensures stable delta comparisons regardless of Module 1 filtering
        m3_result = compute_module_3_catalyst(
            trial_records_path=data_dir / "trial_records.json",  # Path, not list!
            state_dir=state_dir,  # State directory for snapshots (namespaced by universe)
            active_tickers=full_universe_tickers,  # FULL universe, not filtered active_tickers
            as_of_date=as_of_date_obj,  # Date object
            market_calendar=SimpleMarketCalendar(),  # Market calendar for weekends
            config=Module3Config(),  # Default configuration
            output_dir=data_dir  # Output directory for catalyst_events_*.json
        )
        if checkpoint_dir:
            save_checkpoint(checkpoint_dir, "module_3", as_of_date, m3_result)

    # Validate Module 3 output schema
    validate_pipeline_handoff("module_3", "module_5", m3_result)

    # Extract results (summaries is already a dict keyed by ticker)
    catalyst_summaries = m3_result["summaries"]
    diag3 = m3_result.get("diagnostic_counts", {})

    # Print diagnostics
    logger.info(f"  Events detected: {diag3.get('events_detected_total', 0)}, "
                f"Tickers with events: {diag3.get('tickers_with_events', 0)}/{diag3.get('tickers_analyzed', 0)}, "
                f"Severe negatives: {diag3.get('tickers_with_severe_negative', 0)}")

    # ========================================================================
    # End Module 3
    # ========================================================================

    # Module 4: Clinical development
    logger.info("[5/7] Module 4: Clinical development...")
    m4_result = None
    if resume_index > 3 and checkpoint_dir:
        m4_result = load_checkpoint(checkpoint_dir, "module_4", as_of_date)

    if m4_result is None:
        m4_result = compute_module_4_clinical_dev(
            trial_records=trial_records,
            active_tickers=active_tickers,
            as_of_date=as_of_date,  # Explicit threading
        )
        if checkpoint_dir:
            save_checkpoint(checkpoint_dir, "module_4", as_of_date, m4_result)

    # Validate Module 4 output schema
    validate_pipeline_handoff("module_4", "module_5", m4_result)

    diag = m4_result.get('diagnostic_counts', {})
    logger.info(f"  Scored: {diag.get('scored', len(m4_result.get('scores', [])))}, "
                f"Trials evaluated: {diag.get('total_trials', 'N/A')}, "
                f"PIT filtered: {diag.get('pit_filtered', 'N/A')}")

    # ========================================================================
    # Clinical Activity Filter (excludes low-activity tickers from ranking)
    # ========================================================================
    clinical_exclusions = []
    if not no_clinical_filter:
        excluded_tickers, exclusion_details = apply_clinical_activity_filter(
            m4_scores=m4_result.get("scores", []),
            min_trials=min_trials,
            min_phase=min_phase,
        )
        clinical_exclusions = exclusion_details

        if excluded_tickers:
            logger.info(f"[5.1/7] Clinical activity filter: excluding {len(excluded_tickers)} tickers "
                       f"(min_trials={min_trials}, min_phase={min_phase})")

            # Remove excluded tickers from active_tickers
            active_tickers = [t for t in active_tickers if t not in set(excluded_tickers)]

            # Log sample exclusions
            for ex in exclusion_details[:5]:
                logger.info(f"    Excluded {ex['ticker']}: {ex['details']}")
            if len(exclusion_details) > 5:
                logger.info(f"    ... and {len(exclusion_details) - 5} more")
    else:
        logger.info("[5.1/7] Clinical activity filter: DISABLED (--no-clinical-filter)")

    # ========================================================================
    # Enhancement Layer: PoS, Short Interest, Regime Detection
    # ========================================================================
    enhancement_result = None

    # Run enhancement layer if enhancements enabled OR if short interest enabled with data
    should_run_enhancements = (enable_enhancements and HAS_ENHANCEMENTS) or (enable_short_interest and short_interest_data)

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
            pos_engine = ProbabilityOfSuccessEngine() if (enable_enhancements and HAS_ENHANCEMENTS) else None
            regime_engine = RegimeDetectionEngine() if (enable_enhancements and HAS_ENHANCEMENTS) else None
            si_engine = ShortInterestSignalEngine() if short_interest_data else None

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
                        fed_rate_change_3m=Decimal(str(market_snapshot.get("fed_rate_change_3m", "0")))
                            if market_snapshot.get("fed_rate_change_3m") is not None else None,
                        as_of_date=as_of_date_obj,
                        data_as_of_date=data_as_of_date,
                        **macro_params  # Include yield curve, HY spread, fund flows
                    )

                    # Log regime with staleness info if applicable
                    staleness = regime_result.get('staleness')
                    indicators = regime_result.get('indicators', {})
                    if staleness and staleness.get('is_stale'):
                        logger.warning(f"  Regime: UNKNOWN (data stale: {staleness['age_days']} days old)")
                    elif staleness and staleness.get('action') == 'HAIRCUT_APPLIED':
                        logger.info(f"  Regime: {regime_result['regime']} (confidence: {regime_result['confidence']}, {staleness['age_days']}d stale → {staleness['haircut_multiplier']}x haircut)")
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
                    tickers=active_tickers,
                    trial_records=trial_records,
                    as_of_date=as_of_date_obj
                )
                logger.info(f"  Indication mapping: {len(indication_map)} tickers mapped")

                for ticker in active_tickers:
                    stage_info = ticker_stage_map.get(ticker, {"base_stage": "phase_2"})
                    # Get indication from mapper (auto-detected from conditions)
                    mapped_indication = indication_map.get(ticker.upper(), {}).get("indication")
                    # Fall back to clinical data if mapper returns None
                    final_indication = mapped_indication or stage_info.get("indication")
                    pos_universe.append({
                        "ticker": ticker,
                        "base_stage": stage_info.get("base_stage", "phase_2"),
                        "indication": final_indication,
                        "trial_design_quality": stage_info.get("trial_design_quality"),
                        "pipeline_trial_count": stage_info.get("pipeline_trial_count", 0),
                        "pipeline_phase_diversity": stage_info.get("pipeline_phase_diversity", 1),
                    })

                pos_result = pos_engine.score_universe(pos_universe, as_of_date_obj)
                pos_diag = pos_result['diagnostic_counts']
                conf_dist = pos_diag.get('confidence_distribution', {})
                logger.info(f"  PoS: mapped={pos_diag['indication_coverage_pct']} | "
                            f"effective(>=0.40)={pos_diag.get('effective_coverage_pct', 'N/A')} | "
                            f"conf(H/M/L)={conf_dist.get('high', 0)}/{conf_dist.get('medium', 0)}/{conf_dist.get('low', 0)}")

            # Step 3: Calculate short interest signals (if data available)
            si_result = None
            if si_engine and short_interest_data:
                # Build SI universe from short interest data
                si_map = {r.get("ticker"): r for r in short_interest_data}
                si_universe = []
                for ticker in active_tickers:
                    si_info = si_map.get(ticker, {})
                    si_universe.append({
                        "ticker": ticker,
                        "short_interest_pct": si_info.get("short_interest_pct"),
                        "days_to_cover": si_info.get("days_to_cover"),
                        "short_interest_change_pct": si_info.get("short_interest_change_pct"),
                        "institutional_long_pct": si_info.get("institutional_long_pct"),
                    })

                si_result = si_engine.score_universe(si_universe, as_of_date_obj)
                logger.info(f"  SI scored: {si_result['diagnostic_counts']['total_scored']}, "
                            f"Coverage: {si_result['diagnostic_counts']['data_coverage_pct']}")

            # Step 4: Calculate accuracy enhancements (if available)
            accuracy_result = None
            if HAS_ACCURACY_ENHANCEMENTS:
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
                        elif trial.get("last_update_posted", "") > trial_data_map[ticker].get("last_update_posted", ""):
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
                accuracy_adjustments = accuracy_adapter.compute_universe_adjustments(
                    universe=universe_for_accuracy,
                    trial_data_map=trial_data_map,
                    financial_data_map=financial_data_map,
                    as_of_date=as_of_date_obj,
                    vix_current=vix_current,
                    market_regime=regime_result.get("regime") if regime_result else None,
                )

                accuracy_diag = accuracy_adapter.get_diagnostic_counts()
                logger.info(f"  Accuracy enhancements: {accuracy_diag['total']} tickers, "
                           f"staleness={accuracy_diag['staleness_coverage_pct']}%, "
                           f"regulatory={accuracy_diag['regulatory_coverage_pct']}%")

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

            # Step 5: Calculate dilution risk scores (if available)
            dilution_risk_result = None
            if HAS_DILUTION_RISK:
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

                    dilution_universe.append({
                        "ticker": ticker,
                        "quarterly_cash": Decimal(str(fin_data.get("Cash"))) if fin_data.get("Cash") else None,
                        "quarterly_burn": quarterly_burn,
                        "next_catalyst_date": next_catalyst_date,
                        "market_cap": Decimal(str(mkt_data.get("market_cap"))) if mkt_data.get("market_cap") else None,
                        "avg_daily_volume_90d": int(mkt_data.get("avg_volume_90d", 0)) if mkt_data.get("avg_volume_90d") else None,
                    })

                dilution_risk_result = dilution_engine.score_universe(dilution_universe, as_of_date_obj)
                diag_dr = dilution_risk_result.get("diagnostic_counts", {})
                risk_dist = diag_dr.get("risk_distribution", {})
                logger.info(f"  Dilution risk: {diag_dr.get('total_scored', 0)} scored, "
                           f"HIGH={risk_dist.get('HIGH_RISK', 0)}, "
                           f"MED={risk_dist.get('MEDIUM_RISK', 0)}, "
                           f"LOW={risk_dist.get('LOW_RISK', 0)}")

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
                            as_of_date=as_of_date_obj
                        )
                        diag_ts = timeline_slippage_result.get("diagnostic_counts", {})
                        logger.info(f"  Timeline slippage: {diag_ts.get('total_scored', 0)} scored, "
                                   f"repeat_offenders={diag_ts.get('repeat_offenders', 0)}")
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
                logger.info(f"  FDA designations: {diag_fda.get('total_scored', 0)} scored, "
                           f"with_designations={diag_fda.get('with_designations', 0)}")

            # Step 8: Calculate pipeline diversity scores (if available)
            pipeline_diversity_result = None
            if HAS_PIPELINE_DIVERSITY:
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
                    diversity_universe.append({
                        "ticker": ticker_upper,
                        "clinical_data": clinical_data_map.get(ticker_upper, {}),
                    })

                pipeline_diversity_result = diversity_engine.score_universe(
                    diversity_universe, trial_records, as_of_date_obj
                )

                diag_pd = pipeline_diversity_result.get("diagnostic_counts", {})
                risk_dist = diag_pd.get("risk_distribution", {})
                logger.info(f"  Pipeline diversity: {diag_pd.get('total_scored', 0)} scored, "
                           f"single_asset={risk_dist.get('single_asset', 0)}, "
                           f"diversified={risk_dist.get('diversified', 0) + risk_dist.get('broad_portfolio', 0)}")

            # Step 9: Calculate competitive intensity scores (if available)
            competitive_intensity_result = None
            if HAS_COMPETITIVE_INTENSITY:
                intensity_engine = CompetitiveIntensityEngine()

                intensity_universe = [{"ticker": t.upper()} for t in active_tickers]
                competitive_intensity_result = intensity_engine.score_universe(
                    intensity_universe, trial_records, as_of_date_obj
                )

                diag_ci = competitive_intensity_result.get("diagnostic_counts", {})
                intensity_dist = diag_ci.get("intensity_distribution", {})
                logger.info(f"  Competitive intensity: {diag_ci.get('total_scored', 0)} scored, "
                           f"low={intensity_dist.get('low', 0)}, "
                           f"moderate={intensity_dist.get('moderate', 0)}, "
                           f"high={intensity_dist.get('high', 0)}, "
                           f"intense={intensity_dist.get('intense', 0)}")

            # Step 10: Calculate partnership validation scores (if available)
            partnership_result = None
            if HAS_PARTNERSHIP_ENGINE:
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
                partnership_result = partnership_engine.score_universe(
                    partnership_universe, partnership_records, as_of_date_obj
                )

                diag_ps = partnership_result.get("diagnostic_counts", {})
                strength_dist = diag_ps.get("strength_distribution", {})
                logger.info(f"  Partnership validation: {diag_ps.get('total_scored', 0)} scored, "
                           f"with_partnerships={diag_ps.get('with_partnerships', 0)}, "
                           f"with_top_tier={diag_ps.get('with_top_tier', 0)}")
                logger.info(f"    Strength distribution: exceptional={strength_dist.get('exceptional', 0)}, "
                           f"strong={strength_dist.get('strong', 0)}, "
                           f"moderate={strength_dist.get('moderate', 0)}, "
                           f"weak={strength_dist.get('weak', 0)}")

            # Step 11: Calculate cash burn trajectory scores (if available)
            cash_burn_result = None
            if HAS_CASH_BURN_ENGINE:
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
                cash_burn_result = cash_burn_engine.score_universe(
                    cash_burn_universe, cash_burn_financial, cash_burn_clinical, as_of_date_obj
                )

                diag_cb = cash_burn_result.get("diagnostic_counts", {})
                trajectory_dist = diag_cb.get("trajectory_distribution", {})
                risk_dist = diag_cb.get("risk_distribution", {})
                logger.info(f"  Cash burn trajectory: {diag_cb.get('total_scored', 0)} scored")
                logger.info(f"    Trajectory: decel={trajectory_dist.get('decelerating', 0)}, "
                           f"stable={trajectory_dist.get('stable', 0)}, "
                           f"accel={trajectory_dist.get('accelerating', 0)}, "
                           f"justified={trajectory_dist.get('accelerating_justified', 0)}, "
                           f"unknown={trajectory_dist.get('unknown', 0)}")
                logger.info(f"    Risk: low={risk_dist.get('low', 0)}, "
                           f"moderate={risk_dist.get('moderate', 0)}, "
                           f"high={risk_dist.get('high', 0)}, "
                           f"critical={risk_dist.get('critical', 0)}")

            # Step 12: Calculate phase transition momentum scores (if available)
            phase_momentum_result = None
            if HAS_PHASE_MOMENTUM_ENGINE:
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
                phase_momentum_result = phase_momentum_engine.score_universe(
                    phase_momentum_universe, trials_by_ticker, None, as_of_date_obj
                )

                diag_pm = phase_momentum_result.get("diagnostic_counts", {})
                momentum_dist = diag_pm.get("momentum_distribution", {})
                logger.info(f"  Phase momentum: {diag_pm.get('total_scored', 0)} scored")
                logger.info(f"    Momentum: strong_pos={momentum_dist.get('strong_positive', 0)}, "
                           f"pos={momentum_dist.get('positive', 0)}, "
                           f"neutral={momentum_dist.get('neutral', 0)}, "
                           f"neg={momentum_dist.get('negative', 0)}, "
                           f"strong_neg={momentum_dist.get('strong_negative', 0)}")

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
                "provenance": {
                    "module": "enhancements",
                    "version": "1.8.0",  # Bumped for phase transition momentum
                    "as_of_date": as_of_date,
                    "pos_engine_version": pos_engine.VERSION if pos_engine else None,
                    "regime_engine_version": regime_engine.VERSION if regime_engine else None,
                    "macro_collector_version": "1.0.0" if HAS_MACRO_COLLECTOR else None,
                    "accuracy_adapter_version": "1.0.0" if HAS_ACCURACY_ENHANCEMENTS else None,
                    "dilution_risk_engine_version": "1.0.0" if HAS_DILUTION_RISK else None,
                    "timeline_slippage_engine_version": "1.0.0" if HAS_TIMELINE_SLIPPAGE else None,
                    "fda_designation_engine_version": FDADesignationEngine.VERSION if HAS_FDA_DESIGNATIONS else None,
                    "pipeline_diversity_engine_version": PipelineDiversityEngine.VERSION if HAS_PIPELINE_DIVERSITY else None,
                    "competitive_intensity_engine_version": CompetitiveIntensityEngine.VERSION if HAS_COMPETITIVE_INTENSITY else None,
                    "partnership_engine_version": PartnershipEngine.VERSION if HAS_PARTNERSHIP_ENGINE else None,
                    "cash_burn_engine_version": CashBurnEngine.VERSION if HAS_CASH_BURN_ENGINE else None,
                    "phase_momentum_engine_version": PhaseTransitionEngine.VERSION if HAS_PHASE_MOMENTUM_ENGINE else None,
                }
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
        clinical_excluded_set = set(t['ticker'] for t in clinical_exclusions) if clinical_exclusions else set()

        if clinical_excluded_set:
            # Create filtered copies of results (don't modify originals)
            m1_filtered = {
                **m1_result,
                "active_securities": [s for s in m1_result.get("active_securities", [])
                                      if s.get("ticker") not in clinical_excluded_set],
            }
            m2_filtered = {
                **m2_result,
                "scores": [s for s in m2_result.get("scores", [])
                           if s.get("ticker") not in clinical_excluded_set],
            }
            m4_filtered = {
                **m4_result,
                "scores": [s for s in m4_result.get("scores", [])
                           if s.get("ticker") not in clinical_excluded_set],
            }
        else:
            m1_filtered = m1_result
            m2_filtered = m2_result
            m4_filtered = m4_result

        m5_result = compute_module_5_composite_with_defensive(
            universe_result=m1_filtered,
            financial_result=m2_filtered,
            catalyst_result=m3_result,
            clinical_result=m4_filtered,
            as_of_date=as_of_date,  # Explicit threading
            normalization="rank",
            cohort_mode="stage_only",
            coinvest_signals=coinvest_signals,
            validate=True,
            enhancement_result=enhancement_result,
            market_data_by_ticker=market_data_by_ticker,  # Enable volatility/momentum signals
            raw_financial_data=financial_records,  # Raw financial data for survivability scoring
            enable_sanity_override=False,  # Disabled: mixed v2/v3 scores cause rank artifacts
        )
        if checkpoint_dir:
            save_checkpoint(checkpoint_dir, "module_5", as_of_date, m5_result)

    # Validate Module 5 output schema
    validate_module_5_output(m5_result)

    diag = m5_result.get('diagnostic_counts', {})
    logger.info(f"  Rankable: {diag.get('rankable', len(m5_result.get('ranked_securities', [])))}, "
                f"Excluded: {diag.get('excluded', len(m5_result.get('excluded_securities', [])))}")

    # Log pipeline health status
    run_status = m5_result.get('run_status', 'UNKNOWN')
    degraded = m5_result.get('degraded_components', [])
    coverage = m5_result.get('component_coverage', {})
    gated = m5_result.get('gated_component_counts', {})

    if run_status == 'FAIL':
        logger.error(f"  RUN STATUS: {run_status} - Pipeline health check FAILED")
        for err in m5_result.get('health_errors', []):
            logger.error(f"    {err}")
    elif run_status == 'DEGRADED':
        logger.warning(f"  RUN STATUS: {run_status} - Some components degraded: {degraded}")
        for warn in m5_result.get('health_warnings', []):
            logger.warning(f"    {warn}")
    else:
        logger.info(f"  RUN STATUS: {run_status}")

    # Log component coverage
    logger.info(f"  Component coverage: catalyst={coverage.get('catalyst', '?')}, "
                f"momentum={coverage.get('momentum', '?')}, smart_money={coverage.get('smart_money', '?')}")
    if gated:
        logger.warning(f"  Confidence-gated components: {gated}")

    # Final defensive overlay and top-N selection
    logger.info("[7/7] Defensive overlay & top-N selection...")
    # (Assuming this is handled in Module 5 or separately)

    # Serialize module 3 result (contains TickerCatalystSummary dataclasses)
    def _serialize_m3_result(m3):
        """Convert TickerCatalystSummary objects to dicts for JSON serialization."""
        result = dict(m3)
        if "summaries" in result:
            serialized_summaries = {}
            for ticker, summary in result["summaries"].items():
                if hasattr(summary, 'to_dict'):
                    serialized_summaries[ticker] = summary.to_dict()
                else:
                    serialized_summaries[ticker] = summary
            result["summaries"] = serialized_summaries
        return result

    m3_serialized = _serialize_m3_result(m3_result)

    # Assemble results
    results = {
        "run_metadata": {
            "as_of_date": as_of_date,
            "version": VERSION,
            "deterministic_timestamp": as_of_date + DETERMINISTIC_TIMESTAMP_SUFFIX,
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
        },
        "module_1_universe": m1_result,
        "module_2_financial": m2_result,
        "module_3_catalyst": m3_serialized,
        "module_4_clinical": m4_result,
        "module_5_composite": m5_result,
        "summary": {
            "total_evaluated": len(raw_universe),
            "active_universe": len(active_tickers),
            "excluded": len(m1_result.get('excluded_securities', [])),
            "clinical_activity_filter": {
                "enabled": not no_clinical_filter,
                "min_trials": min_trials,
                "min_phase": min_phase,
                "excluded_count": len(clinical_exclusions),
            },
            "final_ranked": len(m5_result.get('ranked_securities', [])),
            "catalyst_events": diag3.get('events_detected_total', 0),
            "severe_negatives": diag3.get('tickers_with_severe_negative', 0),
        },
        "clinical_exclusions": clinical_exclusions,
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
    
    # Force deterministic timestamps for byte-identical outputs
    deterministic_ts = as_of_date + DETERMINISTIC_TIMESTAMP_SUFFIX
    _force_deterministic_generated_at(results, deterministic_ts)
    
    return results


def compute_data_hash(data_dir: Path) -> str:
    """Compute hash of input data files for seed derivation."""
    h = hashlib.sha256()
    for json_file in sorted(data_dir.glob("*.json")):
        h.update(json_file.name.encode('utf-8'))
        h.update(json_file.read_bytes())
    return h.hexdigest()[:16]


def add_bootstrap_analysis(
    results: dict,
    as_of_date: str,
    data_dir: Path,
    n_bootstrap: int,
    run_id: str = None,
) -> dict:
    """Add bootstrap confidence intervals to results."""
    from common.random_state import derive_base_seed, DeterministicRNG
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
    
    scores = [Decimal(s["composite_score"]) for s in ranked]
    
    # Compute bootstrap CI
    bootstrap_result = compute_bootstrap_ci_decimal(
        scores=scores,
        rng=rng,
        n_bootstrap=n_bootstrap,
    )
    
    results["bootstrap_analysis"] = bootstrap_result
    return results


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
        help="Enable short interest signals (requires short_interest.json in data-dir). "
             "Enabled by default.",
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

    args = parser.parse_args()

    # =========================================================================
    # Configure logging level FIRST (before any logging calls)
    # =========================================================================
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    logger.setLevel(log_level)

    # =========================================================================
    # Validate argument combinations
    # =========================================================================
    if not args.dry_run and args.output is None:
        parser.error("--output is required unless --dry-run is specified")

    if args.resume_from and not args.checkpoint_dir:
        parser.error("--resume-from requires --checkpoint-dir")

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
        logger.debug(f"Using catalyst window preset '{args.catalyst_window_preset}': {catalyst_window[0]}-{catalyst_window[1]} days")

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

    # Set diagnostics output path default if full diagnostics enabled
    diagnostics_out = args.diagnostics_out
    if args.diagnostics == "full" and diagnostics_out is None:
        run_id = args.run_id or f"screen_{args.as_of_date}"
        diagnostics_out = args.data_dir / f"diagnostics_{run_id}.json"
        logger.debug(f"Diagnostics output defaulting to: {diagnostics_out}")

    try:
        # SECURITY: Validate data_dir early
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
            if args.enable_short_interest or summary.get('short_interest_coverage'):
                logger.info(f"SI coverage:        {summary.get('short_interest_coverage', 'N/A')}")
        if args.checkpoint_dir:
            logger.info(f"Checkpoints:        {args.checkpoint_dir}")
        if args.audit_log:
            logger.info(f"Audit log:          {args.audit_log}")
        logger.info("=" * 60)

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


if __name__ == "__main__":
    sys.exit(main())
