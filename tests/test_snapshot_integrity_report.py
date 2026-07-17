"""Tests for the read-only snapshot-integrity validator."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_snapshot_integrity_report import (
    EXPECTED_V2_COHORT_SIZE,
    REQUIRED_COLUMNS,
    TOP_N,
    build_integrity_report,
    check_columns,
    check_decision_engine_consistency,
    check_eligible_count,
    check_provenance,
    check_rank_space,
    check_ticker_uniqueness,
    check_top_n,
    check_v2_cohort,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_row(**kwargs) -> dict[str, str]:
    base = {c: "" for c in REQUIRED_COLUMNS}
    base.update({k: ("" if v is None else str(v)) for k, v in kwargs.items()})
    return base


def write_snapshot(snap_root: Path, date: str, rows: list[dict[str, str]]) -> Path:
    target = snap_root / date
    target.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys()) if rows else REQUIRED_COLUMNS
    with open(target / "rankings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return target


# ---------------------------------------------------------------------------
# Per-check tests
# ---------------------------------------------------------------------------


def test_columns_check_passes_when_all_required_present():
    cols = list(REQUIRED_COLUMNS) + ["extra1", "extra2"]
    r = check_columns(cols)
    assert r["severity"] == "PASS"
    assert r["missing"] == []


def test_columns_check_fails_when_required_missing():
    cols = [c for c in REQUIRED_COLUMNS if c != "ranker_v2_score"]
    r = check_columns(cols)
    assert r["severity"] == "FAIL"
    assert "ranker_v2_score" in r["missing"]


def test_ticker_uniqueness_detects_dup_and_blank():
    rows = [
        make_row(ticker="AAA"),
        make_row(ticker="AAA"),
        make_row(ticker=""),
    ]
    r = check_ticker_uniqueness(rows)
    assert r["severity"] == "FAIL"
    assert r["duplicates"] == {"AAA": 2}
    assert r["n_blank"] == 1


def test_rank_space_dense_passes():
    rows = [make_row(ticker=f"T{i}", actionable_rank=i, eligible="1") for i in range(1, 5)]
    r = check_rank_space(rows)
    assert r["severity"] == "PASS"


def test_rank_space_with_gap_fails():
    rows = [
        make_row(ticker="A", actionable_rank=1, eligible="1"),
        make_row(ticker="B", actionable_rank=3, eligible="1"),
    ]
    r = check_rank_space(rows)
    assert r["severity"] == "FAIL"
    assert 2 in r["missing_ranks"]


def test_rank_space_eligible_without_rank_warns_only():
    """Eligible row with no rank is a soft anomaly, not a structural break."""
    rows = [
        make_row(ticker="A", actionable_rank=1, eligible="1"),
        make_row(ticker="B", actionable_rank=2, eligible="1"),
        make_row(ticker="C", actionable_rank="", eligible="1"),  # eligible but no rank
    ]
    r = check_rank_space(rows)
    assert r["severity"] == "WARN"
    assert "C" in r["eligible_without_rank"]


def test_rank_space_does_not_start_at_one_fails():
    rows = [
        make_row(ticker="A", actionable_rank=2, eligible="1"),
        make_row(ticker="B", actionable_rank=3, eligible="1"),
    ]
    r = check_rank_space(rows)
    assert r["severity"] == "FAIL"
    assert r["starts_at_one"] is False


def test_top_n_size_passes_at_exact_count():
    rows = [make_row(ticker=f"T{i}", actionable_rank=i) for i in range(1, TOP_N + 1)]
    r = check_top_n(rows, TOP_N)
    assert r["severity"] == "PASS"
    assert r["actual"] == TOP_N


def test_top_n_size_fails_when_short():
    rows = [make_row(ticker=f"T{i}", actionable_rank=i) for i in range(1, TOP_N - 1)]
    r = check_top_n(rows, TOP_N)
    assert r["severity"] == "FAIL"
    assert r["actual"] < TOP_N


def test_v2_cohort_size_passes_at_60():
    rows = [make_row(ticker=f"T{i}", ranker_v2_score=0.6) for i in range(EXPECTED_V2_COHORT_SIZE)]
    rows.extend(make_row(ticker=f"X{i}") for i in range(20))  # non-cohort
    r = check_v2_cohort(rows)
    assert r["severity"] == "PASS"
    assert r["actual"] == EXPECTED_V2_COHORT_SIZE


def test_v2_cohort_size_fails_when_off():
    rows = [make_row(ticker=f"T{i}", ranker_v2_score=0.6) for i in range(EXPECTED_V2_COHORT_SIZE - 1)]
    r = check_v2_cohort(rows)
    assert r["severity"] == "FAIL"


def test_eligible_count_zero_fails():
    rows = [make_row(ticker="A", eligible="0"), make_row(ticker="B", eligible="0")]
    r = check_eligible_count(rows)
    assert r["severity"] == "FAIL"


def test_decision_engine_drift_fails():
    rows = [
        make_row(ticker="A", decision_engine_version="v1.13.0", decision_engine_ruleset_id="2a3e79eb"),
        make_row(ticker="B", decision_engine_version="v1.14.0", decision_engine_ruleset_id="2a3e79eb"),
    ]
    r = check_decision_engine_consistency(rows)
    assert r["severity"] == "FAIL"
    assert len(r["versions"]) == 2


# ---------------------------------------------------------------------------
# Provenance check
# ---------------------------------------------------------------------------


def _write_manifest(snap_dir: Path, **manifest) -> None:
    snap_dir.mkdir(parents=True, exist_ok=True)
    (snap_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_provenance_pass_when_no_manifest(tmp_path):
    # No run_manifest.json -> nothing to assess -> PASS (must not raise overall)
    r = check_provenance(tmp_path)
    assert r["severity"] == "PASS"
    assert r["manifest_present"] is False


def test_provenance_pass_when_clean(tmp_path):
    _write_manifest(tmp_path, git={"commit_sha": "abc123", "dirty": False}, screen_exit_code=0)
    r = check_provenance(tmp_path)
    assert r["severity"] == "PASS"
    assert r["dirty"] is False
    assert r["reasons"] == []


def test_provenance_info_when_dirty(tmp_path):
    _write_manifest(tmp_path, git={"commit_sha": "abc123", "dirty": True}, screen_exit_code=0)
    r = check_provenance(tmp_path)
    assert r["severity"] == "INFO"
    assert "dirty_working_tree" in r["reasons"]


def test_provenance_info_when_nonzero_screen_exit(tmp_path):
    _write_manifest(tmp_path, git={"commit_sha": "abc123", "dirty": False}, screen_exit_code=1)
    r = check_provenance(tmp_path)
    assert r["severity"] == "INFO"
    assert any("screen_exit_code=1" in x for x in r["reasons"])


# Non-hex placeholder tokens below are deliberate: the pin logic is plain
# string-prefix matching, so tokens test it identically while avoiding
# detect-secrets false positives on high-entropy hex literals.
def test_provenance_warn_on_pin_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("BIOTECH_PRODUCTION_PINNED_SHA", "PIN_REF_499")
    _write_manifest(tmp_path, git={"commit_sha": "COMMIT_500_DIFFERENT", "dirty": True}, screen_exit_code=1)
    r = check_provenance(tmp_path)
    assert r["severity"] == "WARN"
    assert any("pinned" in x for x in r["reasons"])


def test_provenance_pass_when_pin_matches_short_sha(tmp_path, monkeypatch):
    # Short pinned token is a prefix of the full commit token -> match, no WARN
    monkeypatch.setenv("BIOTECH_PRODUCTION_PINNED_SHA", "COMMIT_PREFIX")
    _write_manifest(tmp_path, git={"commit_sha": "COMMIT_PREFIX_AND_MORE", "dirty": False}, screen_exit_code=0)
    r = check_provenance(tmp_path)
    assert r["severity"] == "PASS"


def _clean_rows() -> list[dict[str, str]]:
    """Structurally-invariant-clean rows (mirrors the clean end-to-end fixture)."""
    rows = []
    for i in range(1, 31):
        rows.append(
            make_row(
                ticker=f"T{i:03d}",
                company_name=f"Company {i}",
                actionable_rank=i,
                eligible="1",
                ranker_v2_score=0.6,
                decision_engine_version="v1.13.0",
                decision_engine_ruleset_id="2a3e79eb",
            )
        )
    for i in range(31, 61):
        rows.append(
            make_row(
                ticker=f"V{i:03d}",
                company_name=f"V{i}",
                ranker_v2_score=0.6,
                eligible="0",
                decision_engine_version="v1.13.0",
                decision_engine_ruleset_id="2a3e79eb",
            )
        )
    for i in range(61, 66):
        rows.append(
            make_row(
                ticker=f"X{i:03d}",
                company_name=f"X{i}",
                eligible="0",
                decision_engine_version="v1.13.0",
                decision_engine_ruleset_id="2a3e79eb",
            )
        )
    return rows


def test_provenance_check_included_in_report(tmp_path):
    snap_dir = write_snapshot(
        tmp_path / "snap",
        "2026-04-27",
        [make_row(ticker="A", actionable_rank=1, eligible="1")],
    )
    _write_manifest(snap_dir, git={"commit_sha": "abc123", "dirty": True}, screen_exit_code=2)
    report = build_integrity_report(snap_dir)
    by_name = {c["name"]: c for c in report["checks"]}
    assert "run_provenance" in by_name
    assert by_name["run_provenance"]["severity"] == "INFO"


def test_provenance_info_keeps_report_ok_true_on_clean_snapshot(tmp_path):
    # Structurally clean snapshot but produced on a dirty tree -> provenance INFO,
    # overall INFO, and `ok` stays True (non-blocking on tolerated dirty runs).
    snap_dir = write_snapshot(tmp_path / "snap", "2026-04-27", _clean_rows())
    _write_manifest(snap_dir, git={"commit_sha": "abc123", "dirty": True}, screen_exit_code=2)
    report = build_integrity_report(snap_dir)
    by_name = {c["name"]: c for c in report["checks"]}
    assert by_name["run_provenance"]["severity"] == "INFO"
    assert report["overall_severity"] == "INFO"
    assert report["ok"] is True


# ---------------------------------------------------------------------------
# End-to-end on synthetic snapshot
# ---------------------------------------------------------------------------


def test_end_to_end_clean_snapshot_overall_pass(tmp_path):
    # Build a minimal but invariant-clean snapshot:
    #   - 65 rows (universe)
    #   - 30 eligible+ranked rows (top-30 size)
    #   - exactly 60 rows with non-empty ranker_v2_score
    #   - dense actionable_rank space [1..30]
    rows = []
    for i in range(1, 31):  # 30 ranked, eligible, in v2 cohort
        rows.append(
            make_row(
                ticker=f"T{i:03d}",
                company_name=f"Company {i}",
                actionable_rank=i,
                eligible="1",
                ranker_v2_score=0.6,
                decision_engine_version="v1.13.0",
                decision_engine_ruleset_id="2a3e79eb",
            )
        )
    for i in range(31, 61):  # 30 v2-cohort but unranked + ineligible (cohort size = 60 total)
        rows.append(
            make_row(
                ticker=f"V{i:03d}",
                company_name=f"V{i}",
                ranker_v2_score=0.6,
                eligible="0",
                decision_engine_version="v1.13.0",
                decision_engine_ruleset_id="2a3e79eb",
            )
        )
    for i in range(61, 66):  # 5 non-cohort, ineligible
        rows.append(
            make_row(
                ticker=f"X{i:03d}",
                company_name=f"X{i}",
                eligible="0",
                decision_engine_version="v1.13.0",
                decision_engine_ruleset_id="2a3e79eb",
            )
        )
    snap_dir = write_snapshot(tmp_path / "snap", "2026-04-27", rows)
    report = build_integrity_report(snap_dir)
    by_name = {c["name"]: c for c in report["checks"]}
    assert by_name["top_30_size"]["severity"] == "PASS"
    assert by_name["v2_cohort_size"]["severity"] == "PASS"
    assert by_name["rank_space_integrity"]["severity"] == "PASS"
    assert report["overall_severity"] == "PASS"


def test_end_to_end_corrupted_snapshot_fails(tmp_path):
    snap_dir = write_snapshot(
        tmp_path / "snap",
        "2026-04-27",
        [
            make_row(ticker="A", actionable_rank=1, eligible="1"),
            make_row(ticker="A", actionable_rank=1, eligible="1"),  # duplicate ticker AND rank
        ],
    )
    report = build_integrity_report(snap_dir)
    assert report["overall_severity"] == "FAIL"
    by_name = {c["name"]: c for c in report["checks"]}
    assert by_name["ticker_uniqueness"]["severity"] == "FAIL"
    assert by_name["rank_space_integrity"]["severity"] == "FAIL"


def test_end_to_end_writes_both_artifacts(tmp_path):
    snap_dir = write_snapshot(
        tmp_path / "snap",
        "2026-04-27",
        [make_row(ticker="A", actionable_rank=1, eligible="1")],
    )
    from tools.build_snapshot_integrity_report import main

    rc = main(["--as-of-date", "2026-04-27", "--snapshots-dir", str(tmp_path / "snap"), "--quiet"])
    assert rc == 0
    assert (snap_dir / "snapshot_integrity_report.json").exists()
    assert (snap_dir / "snapshot_integrity_report.md").exists()
    # JSON is parseable and has overall_severity
    payload = json.loads((snap_dir / "snapshot_integrity_report.json").read_text())
    assert "overall_severity" in payload
    assert payload["schema_version"].startswith("snapshot_integrity_report")


def test_strict_flag_returns_1_on_fail(tmp_path):
    write_snapshot(
        tmp_path / "snap",
        "2026-04-27",
        [
            make_row(ticker="A", actionable_rank=1, eligible="1"),
            make_row(ticker="A", actionable_rank=1, eligible="1"),
        ],
    )
    from tools.build_snapshot_integrity_report import main

    rc = main(
        [
            "--as-of-date",
            "2026-04-27",
            "--snapshots-dir",
            str(tmp_path / "snap"),
            "--quiet",
            "--strict",
        ]
    )
    assert rc == 1
