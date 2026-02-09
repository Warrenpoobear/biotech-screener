#!/usr/bin/env python3
"""Phase-2 daily runner — wraps run_screen.py with --decision-mode phase2 --strict,
captures log, prints one summary line, and exits with the health gate code.

Exit codes:
    0  — health OK
    1  — health FAIL  (or pipeline crash)
    2  — health WARN
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def build_command(args: argparse.Namespace, extra: list[str]) -> list[str]:
    """Assemble the run_screen.py command."""
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_screen.py"),
        "--as-of-date", args.as_of_date,
        "--data-dir", str(args.data_dir),
        "--decision-mode", "phase2",
        "--strict",
        "--snapshot-dir", str(args.snapshot_dir),
    ]
    if args.health_thresholds:
        cmd.extend(["--health-thresholds", str(args.health_thresholds)])
    if args.dry_run:
        cmd.append("--dry-run")
    if extra:
        cmd.extend(extra)
    return cmd


def read_health_json(snapshot_dir: Path, as_of_date: str) -> dict | None:
    """Load phase2_health.json from the snapshot, or None if missing."""
    health_path = snapshot_dir / as_of_date / "phase2_health.json"
    if not health_path.exists():
        return None
    try:
        return json.loads(health_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def format_summary(as_of_date: str, status: str, reasons: list[str],
                   snapshot_dir: Path) -> str:
    """Build the one-line summary."""
    reason_str = f" [{', '.join(reasons)}]" if reasons else ""
    snap_path = snapshot_dir / as_of_date
    return f"[PHASE2] {as_of_date}  {status:<6}{reason_str:<30} snapshot={snap_path}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase-2 daily runner: screen + health gate + summary",
    )
    parser.add_argument(
        "--as-of-date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Screen date (default: today)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=SCRIPT_DIR / "production_data",
        help="Production data directory (default: production_data/)",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=SCRIPT_DIR / "data" / "snapshots",
        help="Snapshot output directory (default: data/snapshots)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=SCRIPT_DIR / "output" / "phase2_daily.log",
        help="Pipeline log file (default: output/phase2_daily.log)",
    )
    parser.add_argument(
        "--health-thresholds",
        type=Path,
        default=None,
        help="Override Phase2HealthThresholds JSON path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Pass --dry-run to run_screen.py",
    )

    args, extra = parser.parse_known_args(argv)

    # Ensure log directory exists
    args.log_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_command(args, extra)
    print(f"[PHASE2] Running: {' '.join(cmd)}", file=sys.stderr)

    # Run pipeline — all output captured to log
    with open(args.log_file, "w") as log_fh:
        result = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT)

    rc = result.returncode

    # Dry-run: pass through exit code without health parsing
    if args.dry_run:
        print(f"[PHASE2] dry-run finished, exit code={rc}")
        return rc

    # Read health JSON
    health = read_health_json(args.snapshot_dir, args.as_of_date)

    if health is None:
        # Pipeline crashed before producing health JSON
        status = "UNKNOWN"
        reasons = ["pipeline_error"]
        summary = format_summary(args.as_of_date, status, reasons, args.snapshot_dir)
        print(summary)
        print(f"[PHASE2] Pipeline exited {rc} but no health JSON found. "
              f"Log: {args.log_file}", file=sys.stderr)
        return 1

    status = health.get("status", "UNKNOWN")
    reasons = health.get("reasons", [])
    summary = format_summary(args.as_of_date, status, reasons, args.snapshot_dir)
    print(summary)

    if rc not in (0, 1, 2):
        # Unexpected exit code — treat as FAIL
        print(f"[PHASE2] Unexpected exit code {rc}. Log: {args.log_file}",
              file=sys.stderr)
        return 1

    if rc != 0:
        print(f"[PHASE2] Health {status}: {', '.join(reasons)}. "
              f"Log: {args.log_file}", file=sys.stderr)

    return rc


if __name__ == "__main__":
    sys.exit(main())
