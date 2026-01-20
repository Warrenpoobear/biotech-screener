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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Common utilities
from common.date_utils import normalize_date, to_date_string, to_date_object
from common.integration_contracts import (
    validate_pipeline_handoff,
    validate_module_5_output,
    SchemaValidationError,
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
except ImportError:
    HAS_ENHANCEMENTS = False

# Accuracy enhancements adapter (optional)
try:
    from accuracy_enhancements_adapter import AccuracyEnhancementsAdapter
    HAS_ACCURACY_ENHANCEMENTS = True
except ImportError:
    HAS_ACCURACY_ENHANCEMENTS = False

# Optional: Risk gates for audit trail
try:
    from risk_gates import get_parameters_snapshot as get_risk_params, compute_parameters_hash as risk_params_hash
    HAS_RISK_GATES = True
except ImportError:
    HAS_RISK_GATES = False

try:
    from liquidity_scoring import get_parameters_snapshot as get_liq_params, compute_parameters_hash as liq_params_hash
    HAS_LIQUIDITY_SCORING = True
except ImportError:
    HAS_LIQUIDITY_SCORING = False

# Ticker validation for fail-loud data quality
try:
    from src.validators.ticker_validator import validate_ticker_list
    HAS_TICKER_VALIDATION = True
except ImportError:
    HAS_TICKER_VALIDATION = False
    logger.warning("Ticker validation module not available - skipping validation")

VERSION = "1.4.0"  # Bumped for ticker validation integration
DETERMINISTIC_TIMESTAMP_SUFFIX = "T00:00:00Z"


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
    Validate as_of_date format.

    Raises:
        ValueError: If date format is invalid

    Note:
        Does not compare to date.today() to maintain time-invariance.
        Lookahead protection is enforced via PIT filters in modules.
    """
    try:
        normalize_date(as_of_date)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid as_of_date format '{as_of_date}': must be YYYY-MM-DD") from e

    # NOTE: Do not compare to date.today() here (wall-clock dependency breaks time-invariance).
    # Lookahead protection should be enforced via PIT filters and/or input snapshot dating.


def load_json_data(filepath: Path, description: str) -> List[Dict[str, Any]]:
    """
    Load JSON data file with validation.
    
    Args:
        filepath: Path to JSON file
        description: Human-readable description for error messages
    
    Returns:
        List of records
    
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    if not filepath.exists():
        raise FileNotFoundError(f"{description} file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"{description} must be a JSON array, got {type(data)}")
    
    return data


def _convert_holdings_to_coinvest(holdings_snapshots: Dict[str, Any]) -> Dict[str, Dict]:
    """
    Convert holdings_snapshots.json format to coinvest_signals format for Module 5.

    holdings_snapshots format:
    {
        "TICKER": {
            "holdings": {
                "current": {
                    "MANAGER_CIK": {"value_kusd": 123456},
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
            "position_changes": {...},
            "holder_tiers": {...}
        }
    }
    """
    # Known elite managers (Tier 1) by CIK
    TIER1_MANAGERS = {
        "0001263508": "Baker Bros",
        "0001346824": "RA Capital",
        "0000909661": "Perceptive",
        "0001167483": "Deerfield",
        "0001390295": "RTW Investments",
        "0001495385": "Orbimed",
        "0001043233": "Viking Global",
        "0001535472": "Citadel",
    }

    coinvest_signals = {}

    for ticker, ticker_data in holdings_snapshots.items():
        if not isinstance(ticker_data, dict):
            continue

        holdings = ticker_data.get("holdings", {})
        current = holdings.get("current", {})

        if not current:
            continue

        # Count managers and identify tiers
        holder_ciks = list(current.keys())
        holder_tiers = {}

        for cik in holder_ciks:
            if cik in TIER1_MANAGERS:
                holder_tiers[cik] = {"tier": 1, "name": TIER1_MANAGERS[cik]}
            else:
                holder_tiers[cik] = {"tier": 2, "name": f"Manager_{cik[-4:]}"}

        coinvest_signals[ticker] = {
            "coinvest_overlap_count": len(holder_ciks),
            "coinvest_holders": holder_ciks,
            "position_changes": {},  # Would need historical data to compute
            "holder_tiers": holder_tiers,
        }

    return coinvest_signals


def write_json_output(filepath: Path, data: Dict[str, Any]) -> None:
    """
    Write JSON output with deterministic formatting.

    Args:
        filepath: Output path
        data: Data to serialize
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

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
            return super().default(obj)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(
            data,
            f,
            indent=2,
            sort_keys=True,  # Deterministic key ordering
            ensure_ascii=False,
            cls=DataclassEncoder,
        )
        f.write('\n')  # Trailing newline for diff-friendliness


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
    Save module checkpoint to disk.

    Args:
        checkpoint_dir: Directory for checkpoints
        module_name: Module identifier (e.g., "module_1")
        as_of_date: Analysis date
        data: Module output data

    Returns:
        Path to checkpoint file
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{module_name}_{as_of_date}.json"
    filepath = checkpoint_dir / filename

    checkpoint_data = {
        "module": module_name,
        "as_of_date": as_of_date,
        "version": VERSION,
        "data": data,
    }

    write_json_output(filepath, checkpoint_data)
    logger.debug(f"Checkpoint saved: {filepath}")
    return filepath


def load_checkpoint(
    checkpoint_dir: Path,
    module_name: str,
    as_of_date: str
) -> Optional[Dict[str, Any]]:
    """
    Load module checkpoint from disk.

    Args:
        checkpoint_dir: Directory for checkpoints
        module_name: Module identifier
        as_of_date: Analysis date

    Returns:
        Module output data, or None if checkpoint not found
    """
    filename = f"{module_name}_{as_of_date}.json"
    filepath = checkpoint_dir / filename

    if not filepath.exists():
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
    enable_enhancements: bool = False,
    enable_short_interest: bool = False,
    checkpoint_dir: Optional[Path] = None,
    resume_from: Optional[str] = None,
    audit_log_path: Optional[Path] = None,
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

    Returns:
        Complete screening results with provenance

    Raises:
        ValueError: If as_of_date is invalid
        FileNotFoundError: If required data files missing
    """
    # CRITICAL: Validate as_of_date FIRST (no implicit defaults)
    validate_as_of_date_param(as_of_date)

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
    market_data_by_ticker = {}
    if market_records:
        for record in market_records:
            if isinstance(record, dict) and 'ticker' in record:
                ticker = record['ticker']
                market_data_by_ticker[ticker] = record
        logger.info(f"  Market data indexed for {len(market_data_by_ticker)} tickers")

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
                holdings_snapshots = load_json_data(holdings_file, "Holdings snapshots")
                if holdings_snapshots:
                    # Convert holdings format to coinvest_signals format
                    coinvest_signals = _convert_holdings_to_coinvest(holdings_snapshots)
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

        # Create state directory if it doesn't exist
        state_dir = data_dir / "ctgov_state"
        state_dir.mkdir(parents=True, exist_ok=True)

        # Run Module 3A with correct signature
        m3_result = compute_module_3_catalyst(
            trial_records_path=data_dir / "trial_records.json",  # Path, not list!
            state_dir=state_dir,  # State directory for snapshots
            active_tickers=set(active_tickers),  # Set of active tickers
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
    # Enhancement Layer: PoS, Short Interest, Regime Detection
    # ========================================================================
    enhancement_result = None

    if enable_enhancements and HAS_ENHANCEMENTS:
        logger.info("[5.5/7] Enhancement Layer: PoS + Regime + Short Interest...")

        # Check for checkpoint
        if resume_index > 4 and checkpoint_dir:
            enhancement_result = load_checkpoint(checkpoint_dir, "enhancements", as_of_date)

        if enhancement_result is None:
            as_of_date_obj = to_date_object(as_of_date)

            # Initialize engines
            pos_engine = ProbabilityOfSuccessEngine()
            regime_engine = RegimeDetectionEngine()
            si_engine = ShortInterestSignalEngine() if short_interest_data else None

            # Step 1: Detect market regime
            regime_result = None
            if market_snapshot:
                try:
                    # Extract data date from provenance for staleness check
                    data_date_str = None
                    if "provenance" in market_snapshot:
                        data_date_str = market_snapshot["provenance"].get("as_of_date")
                    data_as_of_date = date.fromisoformat(data_date_str) if data_date_str else None

                    regime_result = regime_engine.detect_regime(
                        vix_current=Decimal(str(market_snapshot.get("vix", "20"))),
                        xbi_vs_spy_30d=Decimal(str(market_snapshot.get("xbi_vs_spy_30d", "0"))),
                        fed_rate_change_3m=Decimal(str(market_snapshot.get("fed_rate_change_3m", "0")))
                            if market_snapshot.get("fed_rate_change_3m") is not None else None,
                        as_of_date=as_of_date_obj,
                        data_as_of_date=data_as_of_date
                    )

                    # Log regime with staleness info if applicable
                    staleness = regime_result.get('staleness')
                    if staleness and staleness.get('is_stale'):
                        logger.warning(f"  Regime: UNKNOWN (data stale: {staleness['age_days']} days old)")
                    elif staleness and staleness.get('action') == 'HAIRCUT_APPLIED':
                        logger.info(f"  Regime: {regime_result['regime']} (confidence: {regime_result['confidence']}, {staleness['age_days']}d stale → {staleness['haircut_multiplier']}x haircut)")
                    else:
                        logger.info(f"  Regime: {regime_result['regime']} (confidence: {regime_result['confidence']})")
                except Exception as e:
                    logger.warning(f"  Regime detection failed: {e}")
                    regime_result = {"regime": "UNKNOWN", "signal_adjustments": {}}
            else:
                regime_result = {"regime": "UNKNOWN", "signal_adjustments": {}}
                logger.info("  Regime: UNKNOWN (no market snapshot)")

            # Step 2: Calculate PoS scores for each ticker
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

                    ticker_stage_map[ticker] = {
                        "base_stage": clinical_score.get("lead_phase", "phase_2"),
                        "indication": clinical_score.get("lead_indication"),
                        "trial_design_quality": design_quality,
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

            # Assemble enhancement result
            enhancement_result = {
                "regime": regime_result,
                "pos_scores": pos_result,
                "short_interest_scores": si_result,
                "accuracy_enhancements": accuracy_result,
                "provenance": {
                    "module": "enhancements",
                    "version": "1.1.0",
                    "as_of_date": as_of_date,
                    "pos_engine_version": pos_engine.VERSION,
                    "regime_engine_version": regime_engine.VERSION,
                    "accuracy_adapter_version": "1.0.0" if HAS_ACCURACY_ENHANCEMENTS else None,
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
        m5_result = compute_module_5_composite_with_defensive(
            universe_result=m1_result,
            financial_result=m2_result,
            catalyst_result=m3_result,
            clinical_result=m4_result,
            as_of_date=as_of_date,  # Explicit threading
            normalization="rank",
            cohort_mode="stage_only",
            coinvest_signals=coinvest_signals,
            validate=True,
            enhancement_result=enhancement_result,
            market_data_by_ticker=market_data_by_ticker,  # Enable volatility/momentum signals
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
            "final_ranked": len(m5_result.get('ranked_securities', [])),
            "catalyst_events": diag3.get('events_detected_total', 0),
            "severe_negatives": diag3.get('tickers_with_severe_negative', 0),
        }
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
        help="Enable co-invest overlay (requires coinvest_signals.json in data-dir)",
    )

    parser.add_argument(
        "--enable-enhancements",
        action="store_true",
        help="Enable enhancement modules (PoS, Short Interest, Regime Detection). "
             "Requires market_snapshot.json for regime detection.",
    )

    parser.add_argument(
        "--enable-short-interest",
        action="store_true",
        help="Enable short interest signals (requires short_interest.json in data-dir). "
             "Implied by --enable-enhancements if data is available.",
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
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )
    
    args = parser.parse_args()

    # Validate argument combinations
    if not args.dry_run and args.output is None:
        parser.error("--output is required unless --dry-run is specified")

    if args.resume_from and not args.checkpoint_dir:
        parser.error("--resume-from requires --checkpoint-dir")

    try:
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

    except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"ERROR: {e}")
        return 1
    except Exception as e:
        logger.exception(f"UNEXPECTED ERROR: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
