"""
Checkpoint, manifest, replay bundle, and audit trail functions.

Extracted from run_screen.py to reduce its size. All functions maintain
their original signatures and behavior.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional

from archive_snapshot import sha256_file
from common.production_hardening import (
    MAX_JSON_FILE_SIZE_MB,
    FileSizeError,
    IntegrityError,
    PathTraversalError,
    compute_content_hash,
    json_serializer,
    safe_mkdir,
    safe_write_json,
    validate_checkpoint_path,
    validate_file_size,
)

logger = logging.getLogger(__name__)

# Lazy imports for optional modules (risk_gates, liquidity_scoring)
_HAS_RISK_GATES = None
_HAS_LIQUIDITY_SCORING = None


def _ensure_optional_imports():
    global _HAS_RISK_GATES, _HAS_LIQUIDITY_SCORING
    if _HAS_RISK_GATES is not None:
        return
    try:
        from risk_gates import compute_parameters_hash as _rph  # noqa: F401
        from risk_gates import get_parameters_snapshot as _rps  # noqa: F401

        _HAS_RISK_GATES = True
    except ImportError:
        _HAS_RISK_GATES = False
    try:
        from liquidity_scoring import compute_parameters_hash as _lph  # noqa: F401
        from liquidity_scoring import get_parameters_snapshot as _lps  # noqa: F401

        _HAS_LIQUIDITY_SCORING = True
    except ImportError:
        _HAS_LIQUIDITY_SCORING = False


# ---------------------------------------------------------------------------
# Version (imported from run_screen at module level to avoid circular import)
# ---------------------------------------------------------------------------
_VERSION = None


def _get_version() -> str:
    global _VERSION
    if _VERSION is None:
        try:
            from run_screen import VERSION

            _VERSION = VERSION
        except ImportError:
            _VERSION = "0.0.0"
    return _VERSION


# =============================================================================
# CHECKPOINTING
# =============================================================================

CHECKPOINT_MODULES = ["module_1", "module_2", "module_3", "module_4", "enhancements", "module_5"]


def save_checkpoint(checkpoint_dir: Path, module_name: str, as_of_date: str, data: Dict[str, Any]) -> Path:
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
    try:
        filepath = validate_checkpoint_path(checkpoint_dir, module_name, as_of_date)
    except (PathTraversalError, ValueError) as e:
        raise PathTraversalError(f"Invalid checkpoint path: module={module_name}, date={as_of_date}: {e}") from e

    safe_mkdir(checkpoint_dir, mode=0o700)

    checkpoint_data = {
        "module": module_name,
        "as_of_date": as_of_date,
        "version": _get_version(),
        "data": data,
    }

    data_json = json.dumps(data, sort_keys=True, default=json_serializer)
    checkpoint_data["_content_hash"] = compute_content_hash(data_json)

    safe_write_json(filepath, checkpoint_data, mode=0o600)
    logger.debug(f"Checkpoint saved with integrity hash: {filepath}")
    return filepath


def load_checkpoint(
    checkpoint_dir: Path, module_name: str, as_of_date: str, verify_integrity: bool = True
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
    try:
        filepath = validate_checkpoint_path(checkpoint_dir, module_name, as_of_date)
    except (PathTraversalError, ValueError) as e:
        logger.warning(f"Invalid checkpoint path: {e}")
        return None

    if not filepath.exists():
        return None

    if filepath.is_symlink():
        logger.warning(f"Checkpoint is a symlink (security risk), ignoring: {filepath}")
        return None

    try:
        validate_file_size(filepath, MAX_JSON_FILE_SIZE_MB)
    except FileSizeError as e:
        logger.warning(f"Checkpoint file too large: {e}")
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        checkpoint_data = json.load(f)

    checkpoint_version = checkpoint_data.get("version", "0.0.0")
    if checkpoint_version.split(".")[0] != _get_version().split(".")[0]:
        logger.warning(f"Checkpoint version mismatch: {checkpoint_version} vs {_get_version()}. Ignoring checkpoint.")
        return None

    if verify_integrity and "_content_hash" in checkpoint_data:
        data = checkpoint_data.get("data", {})
        data_json = json.dumps(data, sort_keys=True, default=json_serializer)
        computed_hash = compute_content_hash(data_json)
        stored_hash = checkpoint_data["_content_hash"]

        if computed_hash != stored_hash:
            logger.error(f"Checkpoint integrity check FAILED: {filepath}. Expected {stored_hash}, got {computed_hash}")
            raise IntegrityError(f"Checkpoint corrupted or tampered: {filepath}")

        logger.debug(f"Checkpoint integrity verified: {filepath}")

    logger.info(f"Loaded checkpoint: {filepath}")
    return checkpoint_data.get("data")


def get_resume_module_index(resume_from: Optional[str]) -> int:
    """Get the index of the module to resume from.

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
    """Validate all required input files exist without running pipeline."""
    required_files = [
        ("universe.json", "Universe data"),
        ("financial_records.json", "Financial records"),
        ("trial_records.json", "Clinical trial records"),
        ("market_data.json", "Market data"),
    ]

    optional_files = [
        ("coinvest_signals.json", "Co-invest signals"),
    ]

    results: Dict[str, Any] = {
        "valid": True,
        "data_dir": str(data_dir),
        "required_files": {},
        "optional_files": {},
        "content_hashes": {},
        "errors": [],
    }

    for filename, description in required_files:
        filepath = data_dir / filename
        exists = filepath.exists()
        results["required_files"][filename] = {
            "exists": exists,
            "description": description,
        }

        if exists:
            content_hash = hashlib.sha256(filepath.read_bytes()).hexdigest()[:16]
            results["content_hashes"][filename] = content_hash
            try:
                with open(filepath, "r", encoding="utf-8") as f:
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
            fallback_detailed = data_dir / "holdings_detailed.json"
            fallback_snapshots = data_dir / "holdings_snapshots.json"
            if fallback_detailed.exists() or fallback_snapshots.exists():
                fallback_name = fallback_detailed.name if fallback_detailed.exists() else fallback_snapshots.name
                results["optional_files"][filename]["fallback"] = fallback_name
            else:
                results["errors"].append(f"Co-invest enabled but {filename} missing (no holdings fallback found)")
                results["valid"] = False

    return results


# =============================================================================
# INPUT MANIFEST
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
    # --- Dynamic / cache paths ---
    InputDependency("ctgov_cache", "", False, "pit_mode", "cache/ctgov", "run_screen.py:4209"),
    InputDependency("sec_8k_cache", "", False, "sec_8k", "cache/sec", "run_screen.py:4026"),
    InputDependency("fda_adcom_cache", "", False, "fda_adcom", "cache/fda", "run_screen.py:4030"),
    InputDependency("decision_ruleset", "", False, "decision_phase2", "production_data", "run_screen.py:6544"),
]

MANIFEST_VERSION = "v1"


def _count_records(filepath: Path) -> Optional[int]:
    """Return record count for a data file."""
    try:
        if filepath.suffix == ".csv":
            with open(filepath, "r", encoding="utf-8") as f:
                n = sum(1 for _ in f) - 1
            return max(n, 0)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict):
            return len(data)
        return None
    except Exception:
        return None


def _resolve_cache_paths(
    as_of_date: str,
    data_dir: Path,
    ctgov_cache_dir: Optional[Any] = None,
) -> Dict[str, Path]:
    """Pre-resolve dynamic cache paths for the input manifest."""
    resolved: Dict[str, Path] = {}
    repo_root = Path(__file__).resolve().parent

    if ctgov_cache_dir is not False:
        _cache_root = Path(ctgov_cache_dir) if ctgov_cache_dir else repo_root / "cache" / "ctgov"
        candidate = _cache_root / f"trial_records_{as_of_date}.json"
        if candidate.exists():
            resolved["ctgov_cache"] = candidate

    sec_dir = repo_root / "cache" / "sec" / "8k_catalysts"
    if sec_dir.is_dir():
        candidates = sorted(sec_dir.glob(f"8k_catalysts_{as_of_date}*.json"))
        if candidates:
            resolved["sec_8k_cache"] = candidates[-1]

    fda_dir = repo_root / "cache" / "fda"
    if fda_dir.is_dir():
        candidate = fda_dir / f"adcom_calendar_{as_of_date}.json"
        if candidate.exists():
            resolved["fda_adcom_cache"] = candidate

    return resolved


def build_inputs_manifest(
    as_of_date: str,
    data_dir: Path,
    conditions: Dict[str, bool],
    resolved_paths: Optional[Dict[str, Path]] = None,
) -> Dict[str, Any]:
    """Build a deterministic manifest of every input file consumed by this run."""
    resolved_paths = resolved_paths or {}
    dependencies = []
    errors: List[str] = []
    warnings: List[str] = []

    for dep in DEPENDENCY_REGISTRY:
        if dep.condition and not conditions.get(dep.condition, False):
            continue

        if dep.key in resolved_paths:
            filepath = resolved_paths[dep.key]
        elif dep.path_template:
            filepath = data_dir / dep.path_template
        else:
            if dep.required:
                errors.append(f"{dep.key}: required but path not resolvable")
            else:
                warnings.append(f"{dep.key}: dynamic path not resolved (skipped)")
            continue

        entry: Dict[str, Any] = {
            "key": dep.key,
            "path": str(filepath),
            "required": dep.required,
            "exists": filepath.exists(),
            "resolved_from": dep.resolved_from,
            "load_site": dep.load_site,
            "sha256": None,
            "record_count": None,
        }

        if filepath.exists():
            entry["sha256"] = sha256_file(filepath)
            entry["record_count"] = _count_records(filepath)
        else:
            if dep.required:
                errors.append(f"{dep.key}: required file missing ({filepath})")
            else:
                warnings.append(f"{dep.key}: optional file missing ({filepath})")

        dependencies.append(entry)

    dependencies.sort(key=lambda d: d["key"])

    return {
        "manifest_version": MANIFEST_VERSION,
        "as_of_date": as_of_date,
        "generated_at": f"{as_of_date}T00:00:00Z",
        "data_dir": str(data_dir),
        "dependencies": dependencies,
        "validation": {
            "all_required_present": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        },
    }


def verify_inputs_manifest(manifest: Dict[str, Any]) -> bool:
    """Return True if all required inputs are present, logging each error."""
    validation = manifest.get("validation", {})
    for err in validation.get("errors", []):
        logger.error(f"[INPUT MANIFEST] {err}")
    for warn in validation.get("warnings", []):
        logger.warning(f"[INPUT MANIFEST] {warn}")
    return validation.get("all_required_present", False)


# Manifest deps that the CLI layer patches in after the pipeline's own drift
# check has already run. Only meaningful for the early (in-pipeline) check, and
# only under replay; the post-CLI check verifies these normally.
CLI_DEFERRED_MANIFEST_KEYS: FrozenSet[str] = frozenset({"decision_ruleset"})


def verify_against_prior_manifest(
    current: Dict[str, Any],
    prior: Dict[str, Any],
    *,
    allow_new_required_deps: bool = False,
    deferred_keys: FrozenSet[str] = frozenset(),
) -> List[str]:
    """Compare current manifest against a prior one. Returns drift errors.

    allow_new_required_deps: when True (replay against a frozen bundle whose
        manifest predates a now-required dep), a required dep present in
        ``current`` but absent from ``prior`` is treated as schema evolution
        (warn + continue) rather than a hard drift error. Defaults to False so
        day-over-day production still flags contract growth.

    deferred_keys: the mirror case — deps that are required in ``prior`` but are
        not in ``current`` *yet* because a later stage patches them in. Absence
        of such a key is warned and skipped instead of erroring. Defaults to
        empty, so no existing call site changes behaviour.

        This exists because ``decision_ruleset`` is attached by
        ``_attach_phase2_decision_ruleset_manifest`` at the CLI layer, which runs
        after the pipeline's own drift check. Replaying a bundle whose manifest
        already marks that dep required therefore aborted before producing any
        rankings — meaning no bundle generated by current code could be replayed,
        and the golden baseline could not be refreshed.

        Deferral is not a relaxation: the post-CLI check
        (``_verify_inputs_manifest_after_cli_patch``) re-runs this same
        comparison once the dep is present, and enforcement there is unchanged.
        Only absence is skipped — a deferred dep that is present but stale still
        raises sha256 drift, and one flagged ``exists=False`` still errors.
    """
    errs: List[str] = []
    if prior.get("manifest_version") != current.get("manifest_version"):
        return [
            f"manifest_version mismatch (prior={prior.get('manifest_version')} current={current.get('manifest_version')})"
        ]
    if prior.get("as_of_date") != current.get("as_of_date"):
        return [f"as_of_date mismatch (prior={prior.get('as_of_date')} current={current.get('as_of_date')})"]

    cur_map = {d["key"]: d for d in current.get("dependencies", []) if d.get("required")}
    prior_req = {d["key"] for d in prior.get("dependencies", []) if d.get("required")}
    for key in sorted(set(cur_map.keys()) - prior_req):
        if allow_new_required_deps:
            # Replay against a frozen bundle: a dep that is required now but has no
            # entry in the (older) bundle manifest is schema evolution, not input
            # drift — there is no prior baseline to sha-compare against, and the
            # dep's *presence* is already enforced by verify_inputs_manifest. A hard
            # error here aborts replay before any rankings are produced (e.g.
            # `decision_ruleset` vs the Feb golden bundle). Warn and continue.
            logger.warning(
                f"[INPUT MANIFEST] {key}: required dep absent from prior manifest — "
                "no baseline to drift-check (replay against pre-existing bundle)"
            )
        else:
            errs.append(f"{key}: required dep absent from prior manifest (cannot drift-check)")

    for p in prior.get("dependencies", []):
        if not p.get("required"):
            continue
        key = p["key"]
        c = cur_map.get(key)
        if c is None and key in deferred_keys:
            # Patched in by a later stage; the post-CLI check verifies it.
            logger.warning(
                f"[INPUT MANIFEST] {key}: required in prior but not yet in current — "
                "deferred to the post-CLI manifest check"
            )
            continue
        if not c or not c.get("exists"):
            errs.append(f"{key}: required dep missing (prior path={p.get('path')})")
            continue
        p_sha = p.get("sha256")
        c_sha = c.get("sha256")
        if p_sha and c_sha and p_sha != c_sha:
            errs.append(f"{key}: sha256 drift (prior={p_sha[:12]}.. current={c_sha[:12]}..)")
    return errs


# =============================================================================
# REPLAY BUNDLES
# =============================================================================

_CACHE_KEY_TO_SUBDIR = {
    "ctgov_cache": "ctgov",
    "ctgov_cache_dir": "ctgov",
    "sec_8k_cache": "sec",
    "sec_cache_dir": "sec",
    "fda_adcom_cache": "fda",
    "fda_cache_dir": "fda",
}


def _bundle_relpath(dep: Dict[str, Any]) -> Optional[str]:
    """Map a manifest dependency to its relative path inside a replay bundle."""
    key = dep.get("key", "")
    path = dep.get("path", "")
    if not path:
        return None
    p = Path(path)
    fname = p.name
    if key == "decision_ruleset":
        return f"rulesets/{fname}"
    subdir = _CACHE_KEY_TO_SUBDIR.get(key)
    if subdir:
        return f"cache/{subdir}" if p.is_dir() else f"cache/{subdir}/{fname}"
    return f"data/{fname}"


def create_replay_bundle(
    manifest: Dict[str, Any],
    output_path: Path,
    include_optional_present: bool = True,
) -> Path:
    """Create a replay_bundle.tgz from a manifest's dependency list."""
    deps = manifest.get("dependencies", [])
    index_entries: List[Dict[str, Any]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(output_path, "w:gz") as tar:
        manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        ti = tarfile.TarInfo(name="inputs_manifest.json")
        ti.size = len(manifest_bytes)
        tar.addfile(ti, io.BytesIO(manifest_bytes))

        for dep in deps:
            if not dep.get("exists"):
                continue
            if not dep.get("required") and not include_optional_present:
                continue
            src = Path(dep["path"])
            if not src.exists():
                continue
            relpath = _bundle_relpath(dep)
            if not relpath:
                continue
            arc = relpath if src.is_file() else relpath.rstrip("/")
            tar.add(str(src), arcname=arc)
            index_entries.append(
                {
                    "key": dep["key"],
                    "relpath": relpath,
                    "sha256": dep.get("sha256"),
                    "required": dep.get("required", False),
                }
            )

        bundle_index = {
            "bundle_version": "v1",
            "manifest_version": manifest.get("manifest_version"),
            "as_of_date": manifest.get("as_of_date"),
            "created_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "files": sorted(index_entries, key=lambda e: e["key"]),
        }
        idx_bytes = json.dumps(bundle_index, indent=2, sort_keys=True).encode("utf-8")
        ti2 = tarfile.TarInfo(name="bundle_index.json")
        ti2.size = len(idx_bytes)
        tar.addfile(ti2, io.BytesIO(idx_bytes))

    logger.info(f"[REPLAY BUNDLE] Created: {output_path} ({len(index_entries)} files)")
    return output_path


def extract_replay_bundle(bundle_path: Path) -> Dict[str, Any]:
    """Extract a replay bundle to a temp directory and return path mappings."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="replay_bundle_"))
    with tarfile.open(bundle_path, "r:gz") as tar:
        tar.extractall(tmp_dir, filter="data")

    data_dir = tmp_dir / "data"
    if not data_dir.is_dir():
        data_dir.mkdir()

    manifest_path = tmp_dir / "inputs_manifest.json"
    index_path = tmp_dir / "bundle_index.json"

    bundle_index = {}
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            bundle_index = json.load(f)

    rulesets_dir = tmp_dir / "rulesets"
    ruleset_path = None
    if rulesets_dir.is_dir():
        rulesets = list(rulesets_dir.glob("*.json"))
        if rulesets:
            ruleset_path = rulesets[0]

    ctgov_cache_dir = (tmp_dir / "cache" / "ctgov") if (tmp_dir / "cache" / "ctgov").is_dir() else None
    sec_cache_dir = (tmp_dir / "cache" / "sec") if (tmp_dir / "cache" / "sec").is_dir() else None
    fda_cache_dir = (tmp_dir / "cache" / "fda") if (tmp_dir / "cache" / "fda").is_dir() else None

    return {
        "tmp_dir": tmp_dir,
        "data_dir": data_dir,
        "manifest_path": manifest_path,
        "ruleset_path": ruleset_path,
        "ctgov_cache_dir": ctgov_cache_dir,
        "sec_cache_dir": sec_cache_dir,
        "fda_cache_dir": fda_cache_dir,
        "bundle_index": bundle_index,
    }


# =============================================================================
# AUDIT TRAIL
# =============================================================================


def create_audit_record(
    as_of_date: str,
    data_dir: Path,
    content_hashes: Dict[str, str],
) -> Dict[str, Any]:
    """Create comprehensive audit record for the run."""
    _ensure_optional_imports()

    audit: Dict[str, Any] = {
        "as_of_date": as_of_date,
        "orchestrator_version": _get_version(),
        "data_dir": str(data_dir),
        "input_hashes": dict(sorted(content_hashes.items())),
        "parameter_snapshots": {},
        "parameter_hashes": {},
    }

    if _HAS_RISK_GATES:
        from risk_gates import compute_parameters_hash as risk_params_hash
        from risk_gates import get_parameters_snapshot as get_risk_params

        audit["parameter_snapshots"]["risk_gates"] = get_risk_params()
        audit["parameter_hashes"]["risk_gates"] = risk_params_hash()

    if _HAS_LIQUIDITY_SCORING:
        from liquidity_scoring import compute_parameters_hash as liq_params_hash
        from liquidity_scoring import get_parameters_snapshot as get_liq_params

        audit["parameter_snapshots"]["liquidity_scoring"] = get_liq_params()
        audit["parameter_hashes"]["liquidity_scoring"] = liq_params_hash()

    return audit


def append_audit_log(audit_log_path: Path, record: Dict[str, Any]) -> None:
    """Append audit record to JSONL log file."""
    audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, separators=(",", ":"))
    with open(audit_log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def verify_input_freshness(previous_screen: Path, data_dir: Path) -> List[str]:
    """Compare current input hashes against a previous screen run."""
    with open(previous_screen, "r") as f:
        prev = json.load(f)
    prev_hashes = prev.get("run_metadata", {}).get("input_hashes", {})
    warnings = []
    for json_file in sorted(data_dir.glob("*.json")):
        if json_file.name.startswith("run_log") or json_file.name.startswith("screen_"):
            continue
        current_hash = hashlib.sha256(json_file.read_bytes()).hexdigest()[:16]
        prev_hash = prev_hashes.get(json_file.name)
        if prev_hash and current_hash != prev_hash:
            warnings.append(f"STALE: {json_file.name} changed since {previous_screen.name}")
        elif not prev_hash:
            warnings.append(f"NEW: {json_file.name} not in previous run")
    return warnings
