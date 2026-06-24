#!/usr/bin/env python3
"""ops_supervisor sentinel — verifies the supervisor agent ran and produced
a sane artifact. Does NOT interpret model state or classify failures.

Per feedback_no_recursive_supervision.md: this is the terminus of the
monitoring chain. Do NOT propose another layer above this.

Outputs:
  artifacts/ops_supervisor/{as_of_date}_sentinel.json
  artifacts/ops_supervisor/{as_of_date}_sentinel.md

Severity:
  GREEN  — supervisor artifact present, sane, fresh.
  YELLOW — supervisor ran but fail-closed (input missing); artifact still valid.
  RED    — supervisor missing, malformed, stale, or empty.

Exit codes: 0=GREEN, 1=YELLOW, 2=RED.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SUP_DIR = REPO / "artifacts" / "ops_supervisor"

from common.operator_delivery import send_operator_event

VALID_SEVERITY = {"GREEN", "YELLOW", "ORANGE", "RED"}
VALID_ACTION = {"no_action", "watch", "investigate", "fix_now"}
REQUIRED_FIELDS = {
    "schema",
    "as_of_date",
    "generated_at",
    "input_status",
    "final_severity",
    "final_action",
    "summary_one_line",
    "anomalies",
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--as-of", default=None)
    args = p.parse_args()
    as_of = args.as_of or datetime.now().date().isoformat()

    json_path = SUP_DIR / f"{as_of}_supervisor.json"
    md_path = SUP_DIR / f"{as_of}_supervisor.md"

    failures: list[str] = []
    sentinel_state = "GREEN"  # default; downgrade on detection

    if not json_path.exists():
        failures.append(f"supervisor JSON missing: {json_path}")
        sentinel_state = "RED"
        sup = {}
    else:
        try:
            sup = json.load(open(json_path))
        except Exception as e:
            failures.append(f"supervisor JSON malformed: {e}")
            sentinel_state = "RED"
            sup = {}

    # If JSON parsed, run schema/sanity checks
    if sup:
        missing_fields = REQUIRED_FIELDS - set(sup.keys())
        if missing_fields:
            failures.append(f"missing required fields: {sorted(missing_fields)}")
            sentinel_state = "RED"

        if sup.get("final_severity") not in VALID_SEVERITY:
            failures.append(f"final_severity invalid: {sup.get('final_severity')}")
            sentinel_state = "RED"
        if sup.get("final_action") not in VALID_ACTION:
            failures.append(f"final_action invalid: {sup.get('final_action')}")
            sentinel_state = "RED"

        # Check generated_at freshness — must be as_of or as_of+1 (UTC may be next day after ET evening run)
        gen = sup.get("generated_at", "")
        from datetime import date as _date
        from datetime import timedelta as _td

        try:
            as_of_d = _date.fromisoformat(as_of)
            valid_prefixes = {as_of_d.isoformat(), (as_of_d + _td(days=1)).isoformat()}
        except Exception:
            valid_prefixes = {as_of}
        if not any(gen.startswith(p) for p in valid_prefixes):
            failures.append(f"generated_at not within as_of window ({gen}, expected {sorted(valid_prefixes)})")
            sentinel_state = "RED"

        # agent_count or checked_items_count > 0
        ac = sup.get("agent_count", 0)
        ci = sup.get("checked_items_count", 0)
        if not (isinstance(ac, int) and isinstance(ci, int) and (ac > 0 or ci > 0)):
            failures.append(f"agent_count={ac} checked_items_count={ci} both nonpositive")
            sentinel_state = "RED"

        # Detect supervisor's own fail-closed posture
        # If RED and the reason is missing-monitor-input, that's a YELLOW for the sentinel
        # (supervisor did its job; upstream is broken).
        if sup.get("final_severity") == "RED" and "monitor" in (sup.get("summary_one_line") or "").lower():
            if sentinel_state == "GREEN":
                sentinel_state = "YELLOW"
                failures.append(
                    "supervisor reported RED due to missing upstream monitor input "
                    "(fail-closed); sentinel accepts artifact but flags YELLOW"
                )

        # Check markdown companion
        if not md_path.exists():
            failures.append(f"supervisor markdown missing: {md_path}")
            if sentinel_state == "GREEN":
                sentinel_state = "RED"

    out = {
        "schema": "ops_supervisor_sentinel.v1",
        "as_of_date": as_of,
        "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "supervisor_json_path": str(json_path.relative_to(REPO)),
        "supervisor_md_path": str(md_path.relative_to(REPO)),
        "sentinel_state": sentinel_state,
        "failures": failures,
        "supervisor_final_severity": sup.get("final_severity") if sup else None,
        "supervisor_final_action": sup.get("final_action") if sup else None,
        "supervisor_summary": sup.get("summary_one_line") if sup else None,
    }

    SUP_DIR.mkdir(parents=True, exist_ok=True)
    out_json = SUP_DIR / f"{as_of}_sentinel.json"
    out_md = SUP_DIR / f"{as_of}_sentinel.md"
    with open(out_json, "w") as fh:
        json.dump(out, fh, indent=2)
    with open(out_md, "w") as fh:
        fh.write(f"# Supervisor Sentinel — {as_of}\n\n")
        fh.write(f"**Sentinel state**: `{sentinel_state}`\n\n")
        if failures:
            fh.write("## Findings\n\n")
            for f_ in failures:
                fh.write(f"- {f_}\n")
            fh.write("\n")
        if out["supervisor_final_severity"]:
            fh.write(
                f"Supervisor verdict: `{out['supervisor_final_severity']}` "
                f"({out['supervisor_final_action']}) — {out['supervisor_summary']}\n"
            )
        if sentinel_state == "RED":
            fh.write(
                "\n**ACTION**: supervisor did not produce a valid report. "
                "Check `agents/ops_supervisor/supervisor.py` and the upstream heartbeat artifact.\n"
            )

    print(f"[sentinel] {as_of} → {sentinel_state}")
    if failures:
        for f_ in failures:
            print(f"  - {f_}")

    # Route snapshot_missing FAIL event to Town if sentinel is RED (Spec 090 Phase B)
    if sentinel_state == "RED":
        try:
            send_operator_event(
                channel="town",
                severity="FAIL",
                event_type="snapshot_missing",
                title=f"Production snapshot missing: supervisor sentinel RED ({as_of})",
                summary=(
                    f"ops_supervisor artifact missing or invalid as of {as_of}. "
                    f"Production snapshot did not complete. See sentinel artifact for details."
                ),
                artifact=str(out_json.relative_to(REPO)),
                next_operator_action="investigate",
                extra={
                    "as_of_date": as_of,
                    "sentinel_state": sentinel_state,
                    "supervisor_final_severity": sup.get("final_severity") if sup else None,
                    "failure_count": len(failures),
                }
            )
        except Exception as e:
            print(f"[sentinel] WARNING: Failed to route snapshot_missing event to Town: {e}")

    try:
        from tools.agent_skill_telemetry import log_agent_run
        from tools.record_skill_feedback import attach_outcome_verdict

        exec_id = log_agent_run(
            "supervisor_sentinel",
            f"Supervisor sentinel for {as_of}",
            inputs={"as_of_date": as_of},
            outputs={"sentinel_state": sentinel_state, "failure_count": len(failures)},
            success=sentinel_state in ("GREEN", "YELLOW"),
            error=None if sentinel_state in ("GREEN", "YELLOW") else sentinel_state,
        )
        if exec_id:
            attach_outcome_verdict(
                exec_id,
                was_correct=sentinel_state == "GREEN",
                evidence=f"state={sentinel_state} failures={len(failures)}",
            )
    except Exception:
        pass

    return {"GREEN": 0, "YELLOW": 1, "RED": 2}[sentinel_state]


if __name__ == "__main__":
    sys.exit(main())
