#!/usr/bin/env python3
"""Cache-complete daily screen: warm → phase2 → rollup.

One command, every row comparable.

Single date:
    python3 run_daily.py --as-of-date 2026-02-15

Backfill range (business days):
    python3 run_daily.py --from-date 2026-01-28 --to-date 2026-02-15

Skip cache warming (if already warm):
    python3 run_daily.py --as-of-date 2026-02-15 --no-warm

Dry-run (print commands without executing):
    python3 run_daily.py --as-of-date 2026-02-15 --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# Exit-code semantics from run_phase2_daily.py
_EXIT_OK = 0
_EXIT_FAIL = 1
_EXIT_WARN = 2

_STATUS_LABELS = {_EXIT_OK: "OK", _EXIT_FAIL: "FAIL", _EXIT_WARN: "WARN"}


# ── date helpers ────────────────────────────────────────────────────────


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def resolve_dates(args: argparse.Namespace) -> list[str]:
    """Return ordered list of ISO date strings to process."""
    if args.from_date is not None:
        start = _parse_date(args.from_date)
        end = _parse_date(args.to_date)
        dates: list[str] = []
        cur = start
        while cur <= end:
            if cur.weekday() < 5:  # Mon-Fri
                dates.append(cur.isoformat())
            cur += timedelta(days=1)
        return dates
    return [args.as_of_date]


# ── command builders ────────────────────────────────────────────────────


def build_warm_cmd(date_str: str, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT_DIR / "warm_caches.py"),
        "--as-of-date", date_str,
        "--sources", args.sources,
        "--data-dir", str(args.data_dir),
    ]


def build_screen_cmd(
    date_str: str,
    args: argparse.Namespace,
    extra: list[str] | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_phase2_daily.py"),
        "--as-of-date", date_str,
        "--data-dir", str(args.data_dir),
        "--snapshot-dir", str(args.snapshot_dir),
    ]
    if args.health_thresholds is not None:
        cmd += ["--health-thresholds", str(args.health_thresholds)]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.pit_mode != "strict":
        cmd += ["--pit-mode", args.pit_mode]
    if extra:
        cmd += extra
    return cmd


def build_rollup_cmd(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT_DIR / "scripts" / "rollup_shadow_metrics.py"),
        "--snapshot-dir", str(args.snapshot_dir),
    ]


# ── execution ───────────────────────────────────────────────────────────


def run_step(cmd: list[str], label: str, *, dry_run: bool = False) -> int:
    """Run a subprocess step. Returns exit code (0 on dry-run)."""
    if dry_run:
        print(f"[DAILY] {label}: {' '.join(cmd)}")
        return 0
    print(f"[DAILY] {label} ...")
    result = subprocess.run(cmd)
    return result.returncode


# ── CLI ─────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cache-complete daily screen orchestrator.",
    )
    # Date selection (mutually exclusive groups)
    single = p.add_argument_group("single date")
    single.add_argument(
        "--as-of-date",
        default=date.today().isoformat(),
        help="Single screen date (default: today)",
    )
    rng = p.add_argument_group("date range (backfill)")
    rng.add_argument("--from-date", default=None, help="Start date (inclusive)")
    rng.add_argument("--to-date", default=None, help="End date (inclusive)")

    # Step control
    p.add_argument("--no-warm", action="store_true", help="Skip cache warming")
    p.add_argument("--no-rollup", action="store_true", help="Skip rollup step")
    p.add_argument("--dry-run", action="store_true", help="Print commands only")
    p.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="Halt backfill on first FAIL (exit 1)",
    )

    # Pass-through
    p.add_argument("--sources", default="fda_adcom,sec_8k", help="warm_caches sources")
    p.add_argument("--pit-mode", default="strict", help="PIT mode for run_screen")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=SCRIPT_DIR / "production_data",
        help="Production data directory",
    )
    p.add_argument(
        "--snapshot-dir",
        type=Path,
        default=SCRIPT_DIR / "data" / "snapshots",
        help="Snapshot output directory",
    )
    p.add_argument(
        "--health-thresholds",
        type=Path,
        default=None,
        help="Override health thresholds JSON",
    )
    return p


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate mutually-exclusive date options."""
    has_range = args.from_date is not None or args.to_date is not None
    if has_range:
        if args.from_date is None or args.to_date is None:
            parser.error("--from-date and --to-date must both be specified")
        start = _parse_date(args.from_date)
        end = _parse_date(args.to_date)
        if start > end:
            parser.error("--from-date must be <= --to-date")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    validate_args(args, parser)

    dates = resolve_dates(args)
    if not dates:
        print("[DAILY] No business days in range.")
        return 0

    results: list[tuple[str, int, int]] = []  # (date, warm_rc, screen_rc)

    for d in dates:
        # ── warm ──
        warm_rc = 0
        if not args.no_warm:
            warm_cmd = build_warm_cmd(d, args)
            warm_rc = run_step(warm_cmd, f"{d}  warm", dry_run=args.dry_run)

        # ── screen ──
        screen_cmd = build_screen_cmd(d, args, extra=extra or None)
        screen_rc = run_step(screen_cmd, f"{d}  screen", dry_run=args.dry_run)

        warm_lbl = "SKIP" if args.no_warm else _STATUS_LABELS.get(warm_rc, f"ERR({warm_rc})")
        screen_lbl = _STATUS_LABELS.get(screen_rc, f"ERR({screen_rc})")
        print(f"[DAILY] {d}  warm={warm_lbl}  screen={screen_lbl}")
        results.append((d, warm_rc, screen_rc))

        if args.stop_on_fail and screen_rc == _EXIT_FAIL:
            print(f"[DAILY] Stopping: FAIL on {d}")
            break

    # ── rollup (once at end) ──
    rollup_rc = 0
    if not args.no_rollup:
        rollup_cmd = build_rollup_cmd(args)
        rollup_rc = run_step(rollup_cmd, "rollup", dry_run=args.dry_run)

    # ── summary ──
    n_ok = sum(1 for _, _, rc in results if rc == _EXIT_OK)
    n_warn = sum(1 for _, _, rc in results if rc == _EXIT_WARN)
    n_fail = sum(1 for _, _, rc in results if rc == _EXIT_FAIL)
    n_err = sum(1 for _, _, rc in results if rc not in (_EXIT_OK, _EXIT_WARN, _EXIT_FAIL))
    print(
        f"\n[DAILY] Summary: {len(results)} dates — "
        f"{n_ok} OK, {n_warn} WARN, {n_fail} FAIL"
        + (f", {n_err} ERR" if n_err else "")
        + (f", rollup={'OK' if rollup_rc == 0 else f'ERR({rollup_rc})'}" if not args.no_rollup else "")
    )

    if n_fail > 0 or n_err > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
