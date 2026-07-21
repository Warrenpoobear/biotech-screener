#!/usr/bin/env python3
"""Governed recursive self-improvement diagnostics.

This module is observability-only. It reads already-produced daily artifacts,
classifies bounded failure modes, and writes diagnosis/remediation artifacts.
It must not mutate scoring inputs, selector/ranker behavior, portfolio sizing,
or production snapshots.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "self_improvement"

EXPECTED_EXPORT_FIELDS: tuple[str, ...] = (
    "short_interest_pct",
    "close_price",
    "market_cap_mm",
    "priced_move_pct",
)

CORE_PROBABILISTIC_FEATURE_FIELDS: tuple[str, ...] = (
    "confidence_overall",
    "confidence_financial",
    "confidence_clinical",
    "confidence_catalyst",
    "confidence_pos",
)

OPTIONAL_PROBABILISTIC_FEATURE_FIELDS: tuple[str, ...] = (
    "p_move_gt_implied",
    "p_iv_crush",
    "p_false_positive",
)

PROBABILISTIC_FEATURE_FIELDS: tuple[str, ...] = (
    *CORE_PROBABILISTIC_FEATURE_FIELDS,
    *OPTIONAL_PROBABILISTIC_FEATURE_FIELDS,
)

DO_NOT_CHANGE: tuple[str, ...] = (
    "No ranker, selector, sizing, final_score, or alpha changes from this artifact.",
    "Do not invent insider signal or treat unwired insider buying as validated alpha.",
    "Do not change catalyst taxonomy or event timing without explicit governance approval.",
    "Do not promote model/policy changes without fresh snapshot evidence and operator approval.",
    "Do not mutate production data, caches, ledgers, CRT rows, or shadow evidence from this harness.",
)

QUEUE_CLASSIFICATIONS: tuple[str, ...] = (
    "AUTO_SAFE_DIAGNOSTIC",
    "OPERATOR_APPROVAL_REQUIRED",
    "RESEARCH_ONLY",
    "BLOCKED_BY_GOVERNANCE",
    "REJECTED_SCOPE_CREEP",
)

SEVERITY_RANK: dict[str, int] = {
    "CRITICAL": 0,
    "HIGH": 1,
    "MEDIUM": 2,
    "LOW": 3,
    "INFO": 4,
}


def _normal_date(as_of_date: str) -> str:
    return date.fromisoformat(as_of_date).isoformat()


def _date_token(as_of_date: str) -> str:
    return _normal_date(as_of_date).replace("-", "_")


def _iso_week(as_of_date: str) -> str:
    iso = date.fromisoformat(as_of_date).isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def week_for_as_of_date(as_of_date: str) -> str:
    """Return the ISO week label used by weekly remediation artifacts."""
    return _iso_week(as_of_date)


def _week_token(week: str) -> str:
    return week.replace("-", "_")


def _read_rankings(rankings_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not rankings_path.exists():
        return [], []
    with open(rankings_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
        return list(reader.fieldnames or []), rows


def _non_null(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and text.lower() not in {"nan", "none", "null"}


def _parse_probability(value: Any) -> float | None:
    if not _non_null(value):
        return None
    text = str(value).strip().lower()
    confidence_labels = {
        "high": 0.9,
        "med": 0.6,
        "medium": 0.6,
        "low": 0.3,
    }
    if text in confidence_labels:
        return confidence_labels[text]
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _is_probabilistic_feature_field(fieldname: str) -> bool:
    name = fieldname.lower()
    if name in PROBABILISTIC_FEATURE_FIELDS:
        return True
    if name.startswith("confidence_") or name.endswith("_confidence"):
        return True
    if name.startswith("p_"):
        return True
    return "probability" in name or name.endswith("_prob") or "_prob_" in name or name.endswith("_probability")


def _gate_by_name(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    gates = manifest.get("gates") or []
    return {str(gate.get("name", "")): dict(gate) for gate in gates if isinstance(gate, dict)}


def _gate_status(gate: dict[str, Any] | None) -> str:
    if not gate:
        return "MISSING"
    return str(gate.get("status") or "UNKNOWN").upper()


def _gate_evidence(gate: dict[str, Any]) -> str:
    name = gate.get("name", "unknown_gate")
    status = _gate_status(gate)
    detail = gate.get("detail") or ""
    value = gate.get("value")
    if value not in (None, "", {}):
        return f"{name}: {status} - {detail} value={value}"
    return f"{name}: {status} - {detail}"


def _field_coverage(fieldnames: list[str], rows: list[dict[str, str]]) -> dict[str, Any]:
    row_count = len(rows)
    fields: dict[str, dict[str, Any]] = {}
    for field in EXPECTED_EXPORT_FIELDS:
        if field not in fieldnames:
            fields[field] = {
                "status": "MISSING_COLUMN",
                "present": False,
                "non_null_count": 0,
                "coverage_pct": 0.0,
            }
            continue
        non_null_count = sum(1 for row in rows if _non_null(row.get(field)))
        coverage_pct = round((non_null_count / row_count) * 100, 2) if row_count else 0.0
        if non_null_count == 0:
            status = "EMPTY"
        elif non_null_count == row_count:
            status = "COMPLETE"
        else:
            status = "PARTIAL"
        fields[field] = {
            "status": status,
            "present": True,
            "non_null_count": non_null_count,
            "coverage_pct": coverage_pct,
        }
    return {
        "rankings_present": bool(fieldnames),
        "row_count": row_count,
        "eligible_count": sum(1 for row in rows if str(row.get("eligible", "")).strip() == "1"),
        "fields": fields,
    }


def _probabilistic_feature_feedback(fieldnames: list[str], rows: list[dict[str, str]]) -> dict[str, Any]:
    row_count = len(rows)
    observed_fields = [field for field in fieldnames if _is_probabilistic_feature_field(field)]
    all_fields = [
        *CORE_PROBABILISTIC_FEATURE_FIELDS,
        *[field for field in observed_fields if field not in CORE_PROBABILISTIC_FEATURE_FIELDS],
    ]
    fields: dict[str, dict[str, Any]] = {}
    for field in all_fields:
        if field not in fieldnames:
            fields[field] = {
                "status": "MISSING_COLUMN",
                "present": False,
                "non_null_count": 0,
                "numeric_count": 0,
                "parse_error_count": 0,
                "coverage_pct": 0.0,
                "out_of_bounds_count": 0,
                "extreme_count": 0,
                "extreme_rate_pct": 0.0,
                "min": None,
                "max": None,
            }
            continue

        non_null_values = [row.get(field) for row in rows if _non_null(row.get(field))]
        parsed_values = [
            parsed for parsed in (_parse_probability(value) for value in non_null_values) if parsed is not None
        ]
        out_of_bounds = [value for value in parsed_values if value < 0.0 or value > 1.0]
        bounded_values = [value for value in parsed_values if 0.0 <= value <= 1.0]
        extreme_count = sum(1 for value in bounded_values if value <= 0.05 or value >= 0.95)
        non_null_count = len(non_null_values)
        numeric_count = len(parsed_values)
        parse_error_count = non_null_count - numeric_count
        coverage_pct = round((non_null_count / row_count) * 100, 2) if row_count else 0.0
        extreme_rate_pct = round((extreme_count / len(bounded_values)) * 100, 2) if bounded_values else 0.0
        if non_null_count == 0:
            status = "EMPTY"
        elif parse_error_count:
            status = "NON_NUMERIC"
        elif out_of_bounds:
            status = "OUT_OF_BOUNDS"
        elif non_null_count < row_count:
            status = "PARTIAL"
        else:
            status = "COMPLETE"
        fields[field] = {
            "status": status,
            "present": True,
            "non_null_count": non_null_count,
            "numeric_count": numeric_count,
            "parse_error_count": parse_error_count,
            "coverage_pct": coverage_pct,
            "out_of_bounds_count": len(out_of_bounds),
            "extreme_count": extreme_count,
            "extreme_rate_pct": extreme_rate_pct,
            "min": round(min(parsed_values), 4) if parsed_values else None,
            "max": round(max(parsed_values), 4) if parsed_values else None,
        }

    return {
        "rankings_present": bool(fieldnames),
        "row_count": row_count,
        "core_fields": list(CORE_PROBABILISTIC_FEATURE_FIELDS),
        "optional_fields": list(OPTIONAL_PROBABILISTIC_FEATURE_FIELDS),
        "observed_fields": observed_fields,
        "fields": fields,
    }


def _finding(
    *,
    finding_id: str,
    title: str,
    area: str,
    severity: str,
    classification: str,
    allowed_hypothesis: str,
    proposal_type: str,
    risk_level: str,
    governance_classification: str,
    evidence: list[str],
    likely_cause: str,
    suggested_owner: str,
    forbidden_changes: list[str],
    verification: str,
    promotion_status: str,
) -> dict[str, Any]:
    return {
        "id": finding_id,
        "title": title,
        "area": area,
        "severity": severity,
        "classification": classification,
        "allowed_hypothesis": allowed_hypothesis,
        "proposal_type": proposal_type,
        "risk_level": risk_level,
        "governance_classification": governance_classification,
        "evidence": evidence,
        "likely_cause": likely_cause,
        "suggested_owner": suggested_owner,
        "forbidden_changes": forbidden_changes,
        "verification": verification,
        "promotion_status": promotion_status,
    }


def _sort_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        findings,
        key=lambda item: (
            SEVERITY_RANK.get(str(item.get("severity", "INFO")), 99),
            str(item.get("area", "")),
            str(item.get("id", "")),
        ),
    )


def _build_data_coverage_findings(
    data_coverage: dict[str, Any],
) -> list[dict[str, Any]]:
    fields = data_coverage["fields"]
    missing = sorted(name for name, stats in fields.items() if stats["status"] == "MISSING_COLUMN")
    empty = sorted(name for name, stats in fields.items() if stats["status"] == "EMPTY")
    partial = sorted(
        name for name, stats in fields.items() if stats["status"] == "PARTIAL" and float(stats["coverage_pct"]) < 95.0
    )
    if not missing and not empty and not partial:
        return []

    evidence = []
    if missing:
        evidence.append(f"Missing expected exported fields: {', '.join(missing)}")
    if empty:
        evidence.append(f"Expected exported fields are present but empty: {', '.join(empty)}")
    if partial:
        details = ", ".join(f"{name}={fields[name]['coverage_pct']}%" for name in partial)
        evidence.append(f"Expected exported fields below 95% coverage: {details}")

    return [
        _finding(
            finding_id="data_coverage_expectation_exports",
            title="Expectation model feature coverage gap",
            area="Data coverage",
            severity="MEDIUM" if missing or empty else "LOW",
            classification="Plumbing/export gap",
            allowed_hypothesis="Missing fields",
            proposal_type="Plumbing fix",
            risk_level="Low",
            governance_classification="AUTO_SAFE_DIAGNOSTIC",
            evidence=evidence,
            likely_cause="Existing snapshot fields are not fully exported or populated for downstream diagnostics.",
            suggested_owner="Data plumbing / daily production",
            forbidden_changes=[
                "Do not invent insider signal",
                "Do not alter final_score, ranker weights, selector policy, or sizing.",
            ],
            verification="Check the next production rankings.csv for non-null coverage and expectation-model consumption.",
            promotion_status="Pending fresh snapshot evidence.",
        )
    ]


def _build_probabilistic_feature_findings(
    feedback: dict[str, Any],
) -> list[dict[str, Any]]:
    fields = feedback["fields"]
    core_missing = sorted(
        field for field in CORE_PROBABILISTIC_FEATURE_FIELDS if fields[field]["status"] == "MISSING_COLUMN"
    )
    empty = sorted(name for name, stats in fields.items() if stats["status"] == "EMPTY")
    partial = sorted(
        name for name, stats in fields.items() if stats["status"] == "PARTIAL" and float(stats["coverage_pct"]) < 95.0
    )
    non_numeric = sorted(name for name, stats in fields.items() if stats["status"] == "NON_NUMERIC")
    out_of_bounds = sorted(name for name, stats in fields.items() if stats["status"] == "OUT_OF_BOUNDS")
    degenerate = sorted(
        name
        for name, stats in fields.items()
        if int(stats["numeric_count"]) >= 5 and float(stats["extreme_rate_pct"]) >= 80.0
    )

    findings: list[dict[str, Any]] = []
    if core_missing or empty or partial or non_numeric:
        evidence = []
        if not feedback["observed_fields"]:
            evidence.append("No probabilistic feature columns were found in rankings.csv")
        if core_missing:
            evidence.append(f"Missing core probabilistic fields: {', '.join(core_missing)}")
        if empty:
            evidence.append(f"Probabilistic fields are present but empty: {', '.join(empty)}")
        if partial:
            details = ", ".join(f"{name}={fields[name]['coverage_pct']}%" for name in partial)
            evidence.append(f"Probabilistic fields below 95% coverage: {details}")
        if non_numeric:
            details = ", ".join(f"{name} parse_errors={fields[name]['parse_error_count']}" for name in non_numeric)
            evidence.append(f"Probabilistic fields contain non-numeric values: {details}")
        findings.append(
            _finding(
                finding_id="probabilistic_feature_feedback_gap",
                title="Probabilistic feature feedback loop has observability gaps",
                area="Probabilistic features",
                severity="MEDIUM" if core_missing or empty else "LOW",
                classification="Probability feature export/feedback gap",
                allowed_hypothesis="Missing or partial probabilistic feature telemetry",
                proposal_type="Diagnostic fix",
                risk_level="Low",
                governance_classification="AUTO_SAFE_DIAGNOSTIC",
                evidence=evidence,
                likely_cause="Probability-like model outputs are not fully exported for recursive diagnosis and calibration review.",
                suggested_owner="Diagnostics / daily production",
                forbidden_changes=[
                    "Do not change probability model weights from this diagnostic.",
                    "Do not rescale confidence or probability outputs without governance approval.",
                ],
                verification="Confirm the next rankings.csv exports non-null bounded probabilistic feature columns.",
                promotion_status="Diagnostic-only; no production effect.",
            )
        )

    if out_of_bounds:
        evidence = [
            ", ".join(
                f"{name} out_of_bounds={fields[name]['out_of_bounds_count']} "
                f"min={fields[name]['min']} max={fields[name]['max']}"
                for name in out_of_bounds
            )
        ]
        findings.append(
            _finding(
                finding_id="probabilistic_feature_contract_violation",
                title="Probabilistic features contain values outside [0, 1]",
                area="Probabilistic features",
                severity="HIGH",
                classification="Probability feature export contract issue",
                allowed_hypothesis="Malformed probability or confidence export",
                proposal_type="Plumbing fix",
                risk_level="Low",
                governance_classification="AUTO_SAFE_DIAGNOSTIC",
                evidence=evidence,
                likely_cause="A probability-like field is emitted on the wrong scale or without validation before export.",
                suggested_owner="Data plumbing / diagnostics",
                forbidden_changes=[
                    "Do not silently clip probabilities in scoring paths.",
                    "Do not alter ranker, selector, sizing, or final_score behavior.",
                ],
                verification="Re-run the daily diagnosis and require all probability-like fields to be bounded in [0, 1].",
                promotion_status="Diagnostic-only until fixed by an explicit export contract change.",
            )
        )

    if degenerate:
        evidence = [
            ", ".join(
                f"{name} extreme_rate={fields[name]['extreme_rate_pct']}% " f"n={fields[name]['numeric_count']}"
                for name in degenerate
            )
        ]
        findings.append(
            _finding(
                finding_id="probabilistic_feature_degenerate_distribution",
                title="Probabilistic features are concentrated at extreme values",
                area="Probabilistic features",
                severity="MEDIUM",
                classification="Calibration or cohort-mix diagnostic",
                allowed_hypothesis="Degenerate probability distribution",
                proposal_type="Model fix",
                risk_level="High",
                governance_classification="RESEARCH_ONLY",
                evidence=evidence,
                likely_cause="A cohort, calibration table, or fallback path may be collapsing probabilities near zero or one.",
                suggested_owner="Research / calibration",
                forbidden_changes=[
                    "Do not recalibrate live probabilities from this artifact alone.",
                    "Do not promote a model change without out-of-sample evidence.",
                ],
                verification="Accumulate repeated snapshots and compare against resolved-event calibration before any model proposal.",
                promotion_status="Research-only; no production effect.",
            )
        )

    return findings


def _build_gate_findings(gates: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    catalyst_gates = [
        gates[name]
        for name in (
            "hard_catalyst_supply",
            "hard_queue_actionability",
            "hard_carry_state",
            "regulatory_calendar",
            "options_coverage",
        )
        if _gate_status(gates.get(name)) in {"WARN", "FAIL"}
    ]
    if catalyst_gates:
        findings.append(
            _finding(
                finding_id="catalyst_attribution_anomaly",
                title="Catalyst attribution or hard-catalyst supply anomaly",
                area="Catalyst attribution",
                severity="HIGH" if any(_gate_status(g) == "FAIL" for g in catalyst_gates) else "MEDIUM",
                classification="Event extraction/classification issue",
                allowed_hypothesis="Wrong catalyst timing",
                proposal_type="Data-source fix",
                risk_level="Medium",
                governance_classification="OPERATOR_APPROVAL_REQUIRED",
                evidence=[_gate_evidence(g) for g in catalyst_gates],
                likely_cause="Catalyst extraction, classification, or enrichment artifacts need review.",
                suggested_owner="Catalyst data owner",
                forbidden_changes=[
                    "Do not change catalyst taxonomy without governance approval.",
                    "Do not change ranker response to catalysts from this diagnosis.",
                ],
                verification="Validate hard-catalyst artifacts and compare the next fresh snapshot sidecars.",
                promotion_status="Requires operator review before production policy changes.",
            )
        )

    drift_gate = gates.get("drift_monitoring")
    if _gate_status(drift_gate) in {"WARN", "FAIL"} and drift_gate:
        findings.append(
            _finding(
                finding_id="ranking_stability_churn",
                title="Top-rank churn needs artifact-vs-real-cause explanation",
                area="Ranking stability",
                severity="HIGH" if _gate_status(drift_gate) == "FAIL" else "MEDIUM",
                classification="Data freshness, normalization, or ranker instability",
                allowed_hypothesis="Top-rank churn",
                proposal_type="Diagnostic fix",
                risk_level="Low",
                governance_classification="AUTO_SAFE_DIAGNOSTIC",
                evidence=[_gate_evidence(drift_gate)],
                likely_cause="Freshness, normalization, or export coverage may explain churn before model behavior is suspect.",
                suggested_owner="Daily production / diagnostics",
                forbidden_changes=[
                    "Do not change ranker weights.",
                    "Do not change Top-30 membership policy from this artifact.",
                ],
                verification="Compare drift_report.json/md with source freshness and feature coverage in the next snapshot.",
                promotion_status="Diagnostic-only until repeated evidence accumulates.",
            )
        )

    risk_gates = [
        gates[name]
        for name in ("risk_concentration", "portfolio_weights", "exposure_missingness")
        if _gate_status(gates.get(name)) in {"WARN", "FAIL"}
    ]
    if risk_gates:
        findings.append(
            _finding(
                finding_id="portfolio_policy_risk_breach",
                title="Portfolio risk or policy limit breach",
                area="Portfolio risk",
                severity="HIGH" if any(_gate_status(g) == "FAIL" for g in risk_gates) else "MEDIUM",
                classification="Portfolio construction policy mismatch",
                allowed_hypothesis="Excess 0-7d exposure",
                proposal_type="Policy fix",
                risk_level="Medium",
                governance_classification="OPERATOR_APPROVAL_REQUIRED",
                evidence=[_gate_evidence(g) for g in risk_gates],
                likely_cause="Portfolio construction exposure or diagnostic coverage may be outside policy tolerance.",
                suggested_owner="Portfolio construction / risk",
                forbidden_changes=[
                    "Do not change sizing or exposure policy without explicit approval.",
                    "Do not relabel policy changes as model improvements.",
                ],
                verification="Run a fresh snapshot and confirm risk_concentration/portfolio_weights gates after approved policy review.",
                promotion_status="Operator approval required.",
            )
        )

    forward_gate = gates.get("forward_eval")
    if _gate_status(forward_gate) in {"WARN", "FAIL"} and forward_gate:
        value = forward_gate.get("value") or {}
        has_ic = isinstance(value, dict) and value.get("mean_ic") is not None
        if has_ic:
            findings.append(
                _finding(
                    finding_id="forward_eval_ic_degradation",
                    title="Forward-eval IC contradicted model expectations",
                    area="Forward eval",
                    severity="HIGH",
                    classification="Signal degradation or regime mismatch",
                    allowed_hypothesis="Poor IC",
                    proposal_type="Model fix",
                    risk_level="High",
                    governance_classification="RESEARCH_ONLY",
                    evidence=[_gate_evidence(forward_gate)],
                    likely_cause="Cohort mix, binary catalyst timing, regime mismatch, or horizon observability may explain IC weakness.",
                    suggested_owner="Research / governance",
                    forbidden_changes=[
                        "Do not change final_score weights.",
                        "Do not change ranker, selector, or alpha admission from a single IC warning.",
                    ],
                    verification="Persist IC evidence and run out-of-sample/shadow evaluation before any governance proposal.",
                    promotion_status="Research-only; no production effect.",
                )
            )
        else:
            findings.append(
                _finding(
                    finding_id="forward_eval_observability_gap",
                    title="Forward-eval observability is incomplete",
                    area="Forward eval",
                    severity="LOW",
                    classification="Forward-eval observability gap",
                    allowed_hypothesis="PIT return horizon not yet observable",
                    proposal_type="Diagnostic fix",
                    risk_level="Low",
                    governance_classification="AUTO_SAFE_DIAGNOSTIC",
                    evidence=[_gate_evidence(forward_gate)],
                    likely_cause="The forward horizon or IC ledger may not have enough observable data yet.",
                    suggested_owner="Diagnostics / evidence",
                    forbidden_changes=["Do not infer model weakness until the forward horizon is observable."],
                    verification="Confirm forward_eval IC ledger persistence once horizon data matures.",
                    promotion_status="Pending observable forward-return evidence.",
                )
            )

    ops_gates = [
        gates[name]
        for name in (
            "cache_health",
            "ruleset_health",
            "phase2_health",
            "ctgov_cache",
            "sec_13f_cache",
            "institutional_summary",
            "institutional_delta",
            "pit_bundle_health",
            "price_pit_cache",
        )
        if _gate_status(gates.get(name)) in {"WARN", "FAIL"}
    ]
    if ops_gates:
        findings.append(
            _finding(
                finding_id="agent_ops_or_ledger_health",
                title="Agent ops, cache, or ledger health warning",
                area="Agent ops",
                severity="HIGH" if any(_gate_status(g) == "FAIL" for g in ops_gates) else "LOW",
                classification="Agent ops or freshness check failure",
                allowed_hypothesis="Agent ops",
                proposal_type="Diagnostic fix",
                risk_level="Low",
                governance_classification="AUTO_SAFE_DIAGNOSTIC",
                evidence=[_gate_evidence(g) for g in ops_gates],
                likely_cause="Hermes/cron/cache/ledger freshness or validation needs operational follow-up.",
                suggested_owner="Ops / Hermes",
                forbidden_changes=["Do not mutate caches or ledgers from this diagnosis artifact."],
                verification="Inspect the named ledger/cache gate and confirm the next daily run clears or repeats the warning.",
                promotion_status="Diagnostic-only.",
            )
        )

    return findings


def build_daily_model_diagnosis(
    snapshot_date_dir: Path,
    manifest: dict[str, Any],
    as_of_date: str | None = None,
) -> dict[str, Any]:
    """Build the governed daily self-improvement diagnosis payload."""
    effective_date = _normal_date(as_of_date or manifest.get("effective_as_of_date") or manifest.get("as_of_date"))
    fieldnames, rows = _read_rankings(snapshot_date_dir / "rankings.csv")
    data_coverage = _field_coverage(fieldnames, rows)
    probabilistic_feature_feedback = _probabilistic_feature_feedback(fieldnames, rows)
    gates = _gate_by_name(manifest)

    findings = _sort_findings(
        _build_data_coverage_findings(data_coverage)
        + _build_probabilistic_feature_findings(probabilistic_feature_feedback)
        + _build_gate_findings(gates)
    )

    return {
        "artifact_version": "1.0",
        "as_of_date": effective_date,
        "governance_boundary": (
            "Governed learning loop: diagnose, propose bounded fixes, test out-of-sample, "
            "and promote only through explicit governance gates."
        ),
        "source_artifacts": {
            "snapshot_dir": str(snapshot_date_dir),
            "rankings_csv": str(snapshot_date_dir / "rankings.csv"),
            "run_manifest": str(snapshot_date_dir / "run_manifest.json"),
        },
        "overall_status": manifest.get("overall_status"),
        "data_coverage": data_coverage,
        "probabilistic_feature_feedback": probabilistic_feature_feedback,
        "findings": findings,
        "top_failure_modes": findings[:5],
        "do_not_change": list(DO_NOT_CHANGE),
    }


def _finding_lines(findings: list[dict[str, Any]]) -> list[str]:
    if not findings:
        return ["- No findings in this section."]
    lines: list[str] = []
    for finding in findings:
        lines.append(
            f"- **{finding['severity']} | {finding['title']}** "
            f"({finding['governance_classification']}): {finding['classification']}."
        )
        for evidence in finding.get("evidence", []):
            lines.append(f"  - Evidence: {evidence}")
        lines.append(f"  - Allowed action: {finding['proposal_type']} ({finding['risk_level']} risk).")
        lines.append(f"  - Forbidden: {'; '.join(finding.get('forbidden_changes', []))}")
        lines.append(f"  - Verification: {finding['verification']}")
    return lines


def render_daily_model_diagnosis_markdown(diagnosis: dict[str, Any]) -> str:
    """Render the daily diagnosis markdown artifact."""
    findings = list(diagnosis.get("findings", []))

    def by_area(area: str) -> list[dict[str, Any]]:
        return [finding for finding in findings if finding.get("area") == area]

    lines = [
        f"# Daily Model Diagnosis: {diagnosis['as_of_date']}",
        "",
        diagnosis["governance_boundary"],
        "",
        "**Governance rule**: No ranker, selector, sizing, final_score, or alpha changes may be made from this artifact.",
        "",
        "## Top 5 failure modes",
        "",
    ]
    lines.extend(_finding_lines(list(diagnosis.get("top_failure_modes", []))))
    lines.extend(
        [
            "",
            "## 1. Data coverage regressions",
            "",
            "| Field | Status | Non-null count | Coverage % |",
            "|---|---:|---:|---:|",
        ]
    )
    fields = diagnosis["data_coverage"]["fields"]
    for field in EXPECTED_EXPORT_FIELDS:
        stats = fields[field]
        lines.append(f"| {field} | {stats['status']} | {stats['non_null_count']} | {stats['coverage_pct']} |")
    lines.extend(["", *_finding_lines(by_area("Data coverage"))])
    lines.extend(
        [
            "",
            "### Probabilistic feature feedback",
            "",
            "| Field | Status | Non-null count | Coverage % | Out-of-bounds | Extreme rate % |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    prob_feedback = diagnosis["probabilistic_feature_feedback"]
    for field, stats in prob_feedback["fields"].items():
        lines.append(
            f"| {field} | {stats['status']} | {stats['non_null_count']} | "
            f"{stats['coverage_pct']} | {stats['out_of_bounds_count']} | "
            f"{stats['extreme_rate_pct']} |"
        )
    lines.extend(["", *_finding_lines(by_area("Probabilistic features"))])

    section_map = [
        ("2. Catalyst attribution anomalies", "Catalyst attribution"),
        ("3. Rank/portfolio churn explanation", "Ranking stability"),
        ("4. Forward-eval observability status", "Forward eval"),
        ("5. Policy/risk breaches", "Portfolio risk"),
    ]
    for title, area in section_map:
        lines.extend(["", f"## {title}", ""])
        lines.extend(_finding_lines(by_area(area)))

    low_risk = [f for f in findings if f.get("governance_classification") == "AUTO_SAFE_DIAGNOSTIC"]
    high_risk = [f for f in findings if f.get("governance_classification") != "AUTO_SAFE_DIAGNOSTIC"]
    lines.extend(["", "## Agent ops", ""])
    lines.extend(_finding_lines(by_area("Agent ops")))
    lines.extend(["", "## 6. Suggested low-risk fixes", ""])
    lines.extend(_finding_lines(low_risk))
    lines.extend(["", "## 7. Suggested high-risk fixes requiring governance", ""])
    lines.extend(_finding_lines(high_risk))
    lines.extend(["", "## 8. Do-not-change list", ""])
    for item in diagnosis["do_not_change"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def write_daily_model_diagnosis(
    snapshot_date_dir: Path,
    manifest: dict[str, Any],
    as_of_date: str | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Write daily diagnosis JSON and Markdown artifacts."""
    diagnosis = build_daily_model_diagnosis(snapshot_date_dir, manifest, as_of_date)
    output_dir.mkdir(parents=True, exist_ok=True)
    token = _date_token(diagnosis["as_of_date"])
    json_path = output_dir / f"DAILY_MODEL_DIAGNOSIS_{token}.json"
    md_path = output_dir / f"DAILY_MODEL_DIAGNOSIS_{token}.md"
    json_path.write_text(json.dumps(diagnosis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_daily_model_diagnosis_markdown(diagnosis), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def _queue_classification(finding: dict[str, Any]) -> str:
    explicit = str(finding.get("governance_classification") or "")
    if explicit in QUEUE_CLASSIFICATIONS:
        return explicit
    proposal_type = str(finding.get("proposal_type") or "")
    risk_level = str(finding.get("risk_level") or "")
    promotion_status = str(finding.get("promotion_status") or "").lower()
    if "rejected" in promotion_status or "scope creep" in promotion_status:
        return "REJECTED_SCOPE_CREEP"
    if proposal_type == "New alpha signal" or risk_level == "Highest":
        return "BLOCKED_BY_GOVERNANCE"
    if proposal_type == "Model fix" or risk_level == "High":
        return "RESEARCH_ONLY"
    if proposal_type in {"Policy fix", "Data-source fix"} or risk_level == "Medium":
        return "OPERATOR_APPROVAL_REQUIRED"
    return "AUTO_SAFE_DIAGNOSTIC"


def build_weekly_remediation_queue(
    diagnoses: list[dict[str, Any]],
    week: str | None = None,
) -> dict[str, Any]:
    """Build a weekly remediation queue from daily diagnoses."""
    if week is None:
        dated = sorted(d.get("as_of_date") for d in diagnoses if d.get("as_of_date"))
        week = _iso_week(dated[-1]) if dated else "unknown-week"

    items: list[dict[str, Any]] = []
    for diagnosis in sorted(diagnoses, key=lambda d: str(d.get("as_of_date", ""))):
        as_of_date = diagnosis.get("as_of_date")
        for finding in diagnosis.get("findings", []):
            classification = _queue_classification(finding)
            items.append(
                {
                    "week": week,
                    "as_of_date": as_of_date,
                    "classification": classification,
                    "finding_id": finding.get("id"),
                    "title": finding.get("title"),
                    "severity": finding.get("severity"),
                    "proposal_type": finding.get("proposal_type"),
                    "risk_level": finding.get("risk_level"),
                    "evidence": finding.get("evidence", []),
                    "forbidden_changes": finding.get("forbidden_changes", []),
                    "verification": finding.get("verification"),
                    "promotion_status": finding.get("promotion_status"),
                }
            )
    items = sorted(
        items,
        key=lambda item: (
            (
                QUEUE_CLASSIFICATIONS.index(item["classification"])
                if item["classification"] in QUEUE_CLASSIFICATIONS
                else 999
            ),
            SEVERITY_RANK.get(str(item.get("severity", "INFO")), 99),
            str(item.get("as_of_date", "")),
            str(item.get("finding_id", "")),
        ),
    )
    return {
        "artifact_version": "1.0",
        "week": week,
        "source_dates": sorted(d.get("as_of_date") for d in diagnoses if d.get("as_of_date")),
        "items": items,
        "do_not_change": list(DO_NOT_CHANGE),
    }


def render_weekly_remediation_queue_markdown(queue: dict[str, Any]) -> str:
    """Render the weekly remediation queue markdown artifact."""
    lines = [
        f"# Weekly Remediation Queue: {queue['week']}",
        "",
        "This queue is a governed RSI memory artifact. It proposes bounded remediation only; it does not authorize production behavior changes.",
        "",
    ]
    items = queue.get("items", [])
    for classification in QUEUE_CLASSIFICATIONS:
        lines.extend([f"## {classification}", ""])
        classified = [item for item in items if item["classification"] == classification]
        if not classified:
            lines.append("- No items.")
            lines.append("")
            continue
        for item in classified:
            lines.append(
                f"- **{item['severity']} | {item['title']}** "
                f"({item['as_of_date']}, {item['proposal_type']}, {item['risk_level']} risk)"
            )
            for evidence in item.get("evidence", []):
                lines.append(f"  - Evidence: {evidence}")
            lines.append(f"  - Verification: {item.get('verification')}")
            lines.append(f"  - Promotion status: {item.get('promotion_status')}")
            if item.get("forbidden_changes"):
                lines.append(f"  - Forbidden: {'; '.join(item['forbidden_changes'])}")
        lines.append("")
    lines.extend(["## Do-not-change list", ""])
    for item in queue["do_not_change"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def write_weekly_remediation_queue(queue: dict[str, Any], output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    """Write weekly remediation queue JSON and Markdown artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    token = _week_token(queue["week"])
    json_path = output_dir / f"WEEKLY_REMEDIATION_QUEUE_{token}.json"
    md_path = output_dir / f"WEEKLY_REMEDIATION_QUEUE_{token}.md"
    json_path.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_weekly_remediation_queue_markdown(queue), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def write_weekly_remediation_queue_from_dir(
    diagnosis_dir: Path = DEFAULT_OUTPUT_DIR,
    week: str | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Load daily diagnoses for a week and write the weekly remediation queue."""
    diagnoses = _load_week_diagnoses(diagnosis_dir, week)
    queue = build_weekly_remediation_queue(diagnoses, week)
    return write_weekly_remediation_queue(queue, output_dir)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_week_diagnoses(diagnosis_dir: Path, week: str | None) -> list[dict[str, Any]]:
    diagnoses: list[dict[str, Any]] = []
    for path in sorted(diagnosis_dir.glob("DAILY_MODEL_DIAGNOSIS_*.json")):
        diagnosis = _load_json(path)
        as_of_date = diagnosis.get("as_of_date")
        if week and as_of_date and _iso_week(as_of_date) != week:
            continue
        diagnoses.append(diagnosis)
    return diagnoses


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write governed self-improvement diagnosis artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    daily = subparsers.add_parser("daily", help="Write daily model diagnosis artifacts")
    daily.add_argument("--snapshot-dir", type=Path, required=True)
    daily.add_argument("--manifest", type=Path, required=True)
    daily.add_argument("--as-of-date", required=True)
    daily.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    weekly = subparsers.add_parser("weekly", help="Write weekly remediation queue artifacts")
    weekly.add_argument("--diagnosis-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    weekly.add_argument("--week", help="ISO week label, e.g. 2026-W23")
    weekly.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    args = parser.parse_args(argv)
    if args.command == "daily":
        manifest = _load_json(args.manifest)
        paths = write_daily_model_diagnosis(args.snapshot_dir, manifest, args.as_of_date, args.output_dir)
    else:
        paths = write_weekly_remediation_queue_from_dir(args.diagnosis_dir, args.week, args.output_dir)

    for path in paths.values():
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
