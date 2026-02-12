#!/usr/bin/env python3
"""Create a new candidate ruleset from the current active, with parameter overrides.

Reads manifest.json, loads the active ruleset, applies CLI overrides, writes
a new JSON file, adds a candidate entry to the manifest, and appends a
DRAFT changelog entry.

CLI:
  python scripts/bump_ruleset.py --tier-a-optionality-floor 0.50
  python scripts/bump_ruleset.py --catalyst-near-days 120 --file v1.4_candidate.json
  python scripts/bump_ruleset.py --from-json path/to/overrides.json
  python scripts/bump_ruleset.py --tier-a-optionality-floor 0.50 --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields as dc_fields
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from decision_engine import VERSION, DecisionRuleset

RULESETS_DIR = PROJECT_ROOT / "production_data" / "decision_rulesets"


def _parse_bool(v: str) -> bool:
    """Parse boolean CLI argument (argparse doesn't handle --flag true/false)."""
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")


def _parse_buckets(s: str) -> tuple:
    """Parse 'threshold:mult,...' into tuple-of-tuples.

    Example: '400:1.0,1000:0.85,2000:0.70' -> ((400.0,1.0),(1000.0,0.85),(2000.0,0.70))
    """
    pairs = []
    for token in s.split(","):
        token = token.strip()
        parts = token.split(":")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f"Expected 'threshold:mult' pair, got '{token}'"
            )
        try:
            raw_thresh = parts[0].strip()
            threshold = int(raw_thresh) if "." not in raw_thresh else float(raw_thresh)
            mult = float(parts[1])
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Non-numeric value in bucket pair '{token}'"
            )
        if not (0 < mult <= 1.0):
            raise argparse.ArgumentTypeError(
                f"Multiplier must be in (0, 1], got {mult} in '{token}'"
            )
        pairs.append((threshold, mult))
    # Validate ascending thresholds
    for i in range(1, len(pairs)):
        if pairs[i][0] <= pairs[i - 1][0]:
            raise argparse.ArgumentTypeError(
                f"Bucket thresholds must be strictly ascending: {pairs}"
            )
    return tuple(pairs)


_VALID_TILT_BANDS = {"NEAR", "MID", "FAR", "MISSING"}


def _parse_tilt_mults(s: str) -> tuple:
    """Parse 'BAND:mult,...' into tuple-of-tuples.

    Example: 'NEAR:1.10,MID:1.05,FAR:0.95,MISSING:0.90'
             -> (("NEAR",1.10),("MID",1.05),("FAR",0.95),("MISSING",0.90))
    """
    pairs = []
    for token in s.split(","):
        token = token.strip()
        parts = token.split(":")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f"Expected 'BAND:mult' pair, got '{token}'"
            )
        band = parts[0].strip().upper()
        if band not in _VALID_TILT_BANDS:
            raise argparse.ArgumentTypeError(
                f"Unknown band '{band}', expected one of {sorted(_VALID_TILT_BANDS)}"
            )
        try:
            mult = float(parts[1])
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Non-numeric multiplier in '{token}'"
            )
        if mult <= 0:
            raise argparse.ArgumentTypeError(
                f"Multiplier must be > 0, got {mult} for '{band}'"
            )
        pairs.append((band, mult))
    return tuple(pairs)


MANIFEST_PATH = RULESETS_DIR / "manifest.json"
CHANGELOG_PATH = PROJECT_ROOT / "RULESET_CHANGELOG.md"


def _load_manifest() -> Dict[str, Any]:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_manifest(manifest: Dict[str, Any]) -> None:
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def _find_active(manifest: Dict[str, Any]) -> Dict[str, Any] | None:
    for entry in manifest["rulesets"]:
        if entry["status"] == "active":
            return entry
    return None


def _build_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Collect CLI flags into a dict of field_name -> value overrides."""
    overrides: Dict[str, Any] = {}

    # Map CLI flag names to dataclass field names
    flag_map = {
        "drawdown_gate": "drawdown_gate",
        "drawdown_gate_mode": "drawdown_gate_mode",
        "drawdown_size_penalty": "drawdown_size_penalty",
        "drawdown_hard_floor": "drawdown_hard_floor",
        "vol_high_threshold": "vol_high_threshold",
        "beta_high_threshold": "beta_high_threshold",
        "drawdown_flag_threshold": "drawdown_flag_threshold",
        "rsi_overbought": "rsi_overbought",
        "confidence_low_threshold": "confidence_low_threshold",
        "alpha_tailwind": "alpha_tailwind",
        "alpha_headwind": "alpha_headwind",
        "tier_a_optionality_floor": "tier_a_optionality_floor",
        "tier_b_optionality_floor": "tier_b_optionality_floor",
        "catalyst_near_days": "catalyst_near_days",
        "sponsor_confirm_threshold": "sponsor_confirm_threshold",
        "enable_cost_haircut": "enable_cost_haircut",
        "cost_impact_cap_bps": "cost_impact_cap_bps",
        "cost_haircut_floor_mult": "cost_haircut_floor_mult",
        "cost_haircut_buckets": "cost_haircut_buckets",
        "enable_catalyst_tilt": "enable_catalyst_tilt",
        "catalyst_tilt_mults": "catalyst_tilt_mults",
        "enable_dd_rel_margin_rescue": "enable_dd_rel_margin_rescue",
        "dd_rel_margin_rescue_threshold": "dd_rel_margin_rescue_threshold",
    }

    for cli_name, field_name in flag_map.items():
        val = getattr(args, cli_name, None)
        if val is not None:
            overrides[field_name] = val

    # --from-json: merge arbitrary overrides (can include sizing_weights)
    if args.from_json:
        with open(args.from_json, "r", encoding="utf-8") as f:
            extra = json.load(f)
        overrides.update(extra)

    return overrides


def _apply_overrides(base: DecisionRuleset, overrides: Dict[str, Any]) -> DecisionRuleset:
    """Create a new DecisionRuleset with overrides applied to base."""
    params: Dict[str, Any] = {}
    for f in dc_fields(base):
        val = getattr(base, f.name)
        if f.name == "sizing_weights":
            params[f.name] = dict(val)
        else:
            params[f.name] = val

    # Apply overrides
    for key, val in overrides.items():
        if key not in params:
            print(f"WARNING: unknown field '{key}', skipping.", file=sys.stderr)
            continue
        params[key] = val

    # Convert sizing_weights back to tuple-of-tuples
    if isinstance(params.get("sizing_weights"), dict):
        params["sizing_weights"] = tuple(
            (k, v) for k, v in sorted(params["sizing_weights"].items())
        )

    # Convert cost_haircut_buckets list-of-lists back to tuple-of-tuples
    if isinstance(params.get("cost_haircut_buckets"), list):
        params["cost_haircut_buckets"] = tuple(
            tuple(pair) for pair in params["cost_haircut_buckets"]
        )

    # Convert catalyst_tilt_mults list-of-lists back to tuple-of-tuples
    if isinstance(params.get("catalyst_tilt_mults"), list):
        params["catalyst_tilt_mults"] = tuple(
            tuple(pair) for pair in params["catalyst_tilt_mults"]
        )

    return DecisionRuleset(**params)


def _param_diff(old: DecisionRuleset, new: DecisionRuleset) -> list[str]:
    """Return human-readable lines showing changed fields."""
    diffs = []
    for f in dc_fields(old):
        old_val = getattr(old, f.name)
        new_val = getattr(new, f.name)
        if old_val != new_val:
            diffs.append(f"  {f.name}: {old_val} -> {new_val}")
    return diffs


def _append_changelog_draft(new_id: str, diffs: list[str], active_id: str) -> None:
    """Append a DRAFT changelog entry."""
    today = date.today().isoformat()
    lines = [
        "",
        "---",
        "",
        f"## [DRAFT] [{VERSION}] {new_id} — {today} — <description>",
        "",
        f"**Parameters changed** (vs {active_id}):",
    ]
    for d in diffs:
        lines.append(f"-{d}")
    lines.append("")
    lines.append("**Rationale**: <fill in>")
    lines.append("")
    lines.append("**Governance**: <fill in>")
    lines.append("")

    with open(CHANGELOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a new candidate ruleset from the active ruleset."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be created without writing.",
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Custom filename for the new ruleset (default: v{version}_candidate.json).",
    )
    parser.add_argument(
        "--from-json", type=str, default=None,
        help="JSON file with arbitrary parameter overrides.",
    )

    # One flag per DecisionRuleset scalar field
    parser.add_argument("--drawdown-gate", type=float)
    parser.add_argument("--drawdown-gate-mode", type=str)
    parser.add_argument("--drawdown-size-penalty", type=int)
    parser.add_argument("--drawdown-hard-floor", type=float)
    parser.add_argument("--vol-high-threshold", type=float)
    parser.add_argument("--beta-high-threshold", type=float)
    parser.add_argument("--drawdown-flag-threshold", type=float)
    parser.add_argument("--rsi-overbought", type=float)
    parser.add_argument("--confidence-low-threshold", type=float)
    parser.add_argument("--alpha-tailwind", type=float)
    parser.add_argument("--alpha-headwind", type=float)
    parser.add_argument("--tier-a-optionality-floor", type=float)
    parser.add_argument("--tier-b-optionality-floor", type=float)
    parser.add_argument("--catalyst-near-days", type=int)
    parser.add_argument("--sponsor-confirm-threshold", type=int)
    parser.add_argument("--enable-cost-haircut", type=_parse_bool)
    parser.add_argument("--cost-impact-cap-bps", type=float)
    parser.add_argument("--cost-haircut-floor-mult", type=float)
    parser.add_argument(
        "--cost-haircut-buckets", type=_parse_buckets,
        help="Bucket boundaries as 'threshold:mult,...' (e.g. '400:1.0,1000:0.85,2000:0.70')",
    )
    parser.add_argument("--enable-catalyst-tilt", type=_parse_bool)
    parser.add_argument(
        "--catalyst-tilt-mults", type=_parse_tilt_mults,
        help="Tilt multipliers as 'BAND:mult,...' (e.g. 'NEAR:1.10,MID:1.05,FAR:0.95,MISSING:0.90')",
    )
    parser.add_argument("--enable-dd-rel-margin-rescue", type=_parse_bool)
    parser.add_argument("--dd-rel-margin-rescue-threshold", type=float)

    args = parser.parse_args()

    # Load manifest and find active
    if not MANIFEST_PATH.exists():
        print("ERROR: manifest.json not found.", file=sys.stderr)
        return 1

    manifest = _load_manifest()
    active_entry = _find_active(manifest)
    if not active_entry:
        print("ERROR: No active ruleset in manifest.", file=sys.stderr)
        return 1

    active_path = RULESETS_DIR / active_entry["file"]
    active_rs = DecisionRuleset.from_json(str(active_path))
    active_id = active_entry["id"]

    # Build and apply overrides
    overrides = _build_overrides(args)
    if not overrides:
        print("ERROR: No parameter overrides specified.", file=sys.stderr)
        print("Use flags (e.g. --tier-a-optionality-floor 0.50) or --from-json.", file=sys.stderr)
        return 1

    new_rs = _apply_overrides(active_rs, overrides)
    new_id = new_rs.ruleset_id

    if new_id == active_id:
        print("WARNING: Overrides produce the same ruleset_id — no effective change.", file=sys.stderr)
        return 1

    # Determine output filename
    version_short = VERSION.lstrip("v")
    default_filename = f"v{version_short}_candidate.json"
    filename = args.file or default_filename
    out_path = RULESETS_DIR / filename

    # Compute diff
    diffs = _param_diff(active_rs, new_rs)

    # Print summary
    print(f"New candidate ruleset:")
    print(f"  ID:   {new_id}")
    print(f"  File: {out_path}")
    print(f"  Base: {active_id} ({active_entry['file']})")
    print()
    print("Parameter changes:")
    for d in diffs:
        print(d)
    print()

    if args.dry_run:
        print("[dry-run] Would write:")
        print(f"  Ruleset JSON: {out_path}")
        print(f"  Manifest entry: {MANIFEST_PATH}")
        print(f"  Changelog draft: {CHANGELOG_PATH}")
        return 0

    # Write ruleset JSON
    new_rs.to_json(str(out_path))
    print(f"Wrote ruleset: {out_path}")

    # Add manifest entry
    new_entry = {
        "id": new_id,
        "file": filename,
        "created_at": date.today().isoformat(),
        "engine_version": VERSION,
        "description": f"Candidate ({', '.join(f'{k}={v}' for k, v in overrides.items())})",
        "status": "candidate",
        "requires_replay": False,
        "notes": f"Generated by bump_ruleset.py from {active_id}.",
        "updated_by": "bump_ruleset.py",
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    # Insert after active entry
    idx = manifest["rulesets"].index(active_entry)
    manifest["rulesets"].insert(idx + 1, new_entry)
    _save_manifest(manifest)
    print(f"Added candidate entry to manifest.")

    # Append changelog draft
    _append_changelog_draft(new_id, diffs, active_id)
    print(f"Appended [DRAFT] entry to {CHANGELOG_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
