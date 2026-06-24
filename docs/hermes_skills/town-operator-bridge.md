# Town-Hermes Bridge (Spec 090)

**Status:** Phase B wired (live delivery pending operator sign-off)  
**Architecture:** Email-based delivery (Phase A), webhook-ready structure  
**Governance:** Read-only ops layer; Town has no production mutation authority

---

## Overview

The Town-Hermes bridge routes Hermes job outputs (governance decisions, hard failures, catalyst discoveries) to Town as operator-facing tasks and alerts, without giving Town control over Hermes or production systems.

```
Hermes job completes
  → write artifact (repo)
  → send_operator_event(channel="town", ...)
      → structured email to TOWN_EMAIL (default: djschulz@gmail.com)
          → Town routine parses [Hermes] prefix
          → Town creates/updates task
          → Town DMs operator (Slack/email/future webhooks)
```

---

## Architecture

### Data Flow (Phase A: Email)

1. **Event trigger:** Hermes job completes, validator passes/fails, snapshot missing, contradiction detected
2. **Event construction:** Job script calls `send_operator_event()`
3. **Payload generation:** `common/operator_delivery.py` builds JSON envelope with event metadata
4. **Email delivery:** `common/alert_email.py` sends structured email to `TOWN_EMAIL`
5. **Town consumption:** Email parser (future: webhook handler) routes to operator task system
6. **Ledger recording:** Event logged to `artifacts/ops/held_spec_ledger/` or `contradiction_ledger/` for audit trail

### What Town is NOT

- NOT a scheduler (Hermes owns cron)
- NOT a repo mutator (no git push from Town)
- NOT a ranker/selector authority (no alpha decisions)
- NOT allowed to reactivate suppressed signals
- NOT the source of truth for production state

---

## Event Types (Spec 090 Table)

| event_type | severity | trigger | next_operator_action |
|---|---|---|---|
| `held_spec_ledger` | INFO | hermes-held-spec-ledger job | review |
| `first_fire_pass` | INFO | first-fire validator PASS | none |
| `first_fire_fail` | FAIL | first-fire validator FAIL | investigate |
| `snapshot_missing` | FAIL | production snapshot absent | investigate |
| `ruleset_mismatch` | FAIL | CLAUDE.md ruleset vs code | investigate |
| `stale_artifact` | WARN | artifact past staleness threshold | review |
| `cron_missed` | FAIL | cron job did not fire | investigate |
| `contradiction_detected` | WARN | hard contradiction in ledger | review |

**Phase C (future):** `first_fire_fail`, `snapshot_missing`, `ruleset_mismatch`, `cron_missed`, `contradiction_detected` → dual Telegram + Town routing for hard failures.

---

## API Reference

### send_operator_event()

```python
from common.operator_delivery import send_operator_event

success: bool = send_operator_event(
    channel: str,                           # "town" | "telegram"
    severity: str,                          # "INFO" | "WARN" | "FAIL"
    event_type: str,                        # from table above
    title: str,                             # short title (e.g., "Held Spec: dossier-gen")
    summary: str = "",                      # 1–2 line summary of result
    artifact: str = "",                     # path to artifact (ledger file, spectrum doc)
    next_operator_action: str = "none",     # "none" | "review" | "approve" | "investigate"
    extra: Optional[dict[str, Any]] = None, # arbitrary metadata (job duration, row count, etc.)
    dry_run: Optional[bool] = None,         # override env default
    skip_dedupe: bool = False,              # skip dedup (for retries)
) -> bool
```

**Returns:** `True` if event logged/sent successfully, `False` on error

**Dedupe windows (default):**
- FAIL: 15 minutes (no duplicate FAIL events within 15m)
- WARN: 30 minutes
- INFO: 60 minutes

**Dry-run mode:** When `OPERATOR_DELIVERY_DRY_RUN=1` (default), event is logged but email is NOT sent. For testing and Phase A validation before going live.

**Examples:**

```python
# Held spec notification
send_operator_event(
    channel="town",
    severity="INFO",
    event_type="held_spec_ledger",
    title="Spec 089 KG: pending 13F cohort stability clearance",
    summary="13F Jaccard 0.875 ≥ 0.70 threshold met. Awaiting final ops approval.",
    artifact="artifacts/ops/held_spec_ledger/2026-05-27_held_specs.json",
    next_operator_action="review"
)

# First-fire failure
send_operator_event(
    channel="town",
    severity="FAIL",
    event_type="first_fire_fail",
    title="First-fire validator: clinical_score sanity check FAIL",
    summary="3 tickers (RVMD, CRSP, BIIB) exceeded ceiling of 100.0. Snapshot aborted.",
    artifact="artifacts/qa/first_fire_2026-05-27.json",
    next_operator_action="investigate",
    extra={"failed_count": 3, "max_clinical_score": 101.5}
)

# Snapshot missing (hard failure)
send_operator_event(
    channel="town",
    severity="FAIL",
    event_type="snapshot_missing",
    title="Production snapshot 2026-05-27 missing",
    summary="Expected data/snapshots_pit/2026-05-27/ not created. Screen run aborted.",
    next_operator_action="investigate",
    extra={"as_of_date": "2026-05-27", "expected_path": "data/snapshots_pit/2026-05-27/screen.pkl"}
)
```

---

## Integration Pattern

### Step 1: Import

```python
from common.operator_delivery import send_operator_event
```

### Step 2: Wire at Job Completion

```python
# Example: hermes-held-spec-ledger job
def main():
    # ... build held_spec_ledger ...
    ledger_path = "artifacts/ops/held_spec_ledger/2026-05-27_held_specs.json"
    
    # On success:
    send_operator_event(
        channel="town",
        severity="INFO",
        event_type="held_spec_ledger",
        title=f"Held specs ledger: {len(held_specs)} specs",
        summary=f"{len(approved)} approved, {len(blocked)} blocked, {len(waiting)} awaiting clearance",
        artifact=ledger_path,
        next_operator_action="review"
    )
    
    # On failure:
    if error:
        send_operator_event(
            channel="town",
            severity="FAIL",
            event_type="held_spec_ledger",
            title="Held spec ledger: job FAILED",
            summary=str(error),
            next_operator_action="investigate"
        )

if __name__ == "__main__":
    main()
```

### Step 3: Test with Dry-Run

```bash
# Phase A: dry-run (logs event, does not send email)
export OPERATOR_DELIVERY_DRY_RUN=1
python3 path/to/job.py

# Check log output for event construction
# Once verified, proceed to Phase B live test
```

---

## Phase A (Current): Email Delivery

**Status:** Complete and tested

**Env vars:**
- `TOWN_EMAIL` (default: `djschulz@gmail.com`) — recipient email
- `OPERATOR_DELIVERY_DRY_RUN` (default: `1`) — log-only mode (no email)

**Subject format:**
```
[Hermes] {SEVERITY} | {event_type} | {title}
```

Example:
```
[Hermes] FAIL | snapshot_missing | Production snapshot 2026-05-27 missing
```

**Body:** JSON envelope with event metadata, artifact links, next action

**Implementation:** `common/operator_delivery.py` → `common/alert_email.py` (Sendgrid or similar)

**Test smoke script:** `tools/smoke_operator_delivery.py` (dry_run=True hardcoded)

---

## Phase B: Hard Failures to Town

**Scope:** Wire `first_fire_fail`, `snapshot_missing`, `ruleset_mismatch`, `cron_missed`, `contradiction_detected` events

**Code status (2026-05-30):** All Phase B event types wired in repo:
- `build_hermes_knowledge_layer.py` + `hermes-contradiction-detector` → `contradiction_detected`
- `ops_supervisor` runtime health + `cron_watchdog.sh` → `cron_missed`
- Operator action remaining: set `OPERATOR_DELIVERY_DRY_RUN=0` in `.env` after smoke tests

**Call sites:**
- `hermes-held-spec-ledger` job (entry point)
- `first-fire-validator` → `first_fire_pass` / `first_fire_fail`
- Snapshot creation watchdog → `snapshot_missing`
- `hermes-ruleset-integrity` → `ruleset_mismatch`
- Morning watchdog → `cron_missed`
- `hermes-contradiction-detector` → `contradiction_detected`

**Deliverables:**
1. All call sites wired (see Call Sites section below)
2. `OPERATOR_DELIVERY_DRY_RUN=1` validation on all target jobs
3. Operator sign-off to set `OPERATOR_DELIVERY_DRY_RUN=0` in `.env`

---

## Phase C (Future): Telegram + Town Dual Routing

**Scope:** Hard failures (`FAIL` severity) route to BOTH Telegram (hard interrupt) AND Town (task creation)

**Design:** Modify `send_operator_event()` to accept `channels=["town", "telegram"]` and dispatch to both

**Motivation:** Critical failures (snapshot missing, ruleset mismatch) need immediate attention; email alone may not wake the operator in time

**Implementation deferred:** Awaiting Phase B completion + operator feedback on email routing effectiveness

---

## Call Sites (Phase B)

Listed in order of implementation:

| Job | event_type | Artifact | Status |
|---|---|---|---|
| `hermes-held-spec-ledger` | `held_spec_ledger` | `artifacts/ops/held_spec_ledger/` | DONE (2026-05-27) |
| `hermes-first-fire-validator` | `first_fire_pass` / `first_fire_fail` | `artifacts/qa/first_fire_{date}.json` | DONE (2026-05-27) |
| `agent_supervisor_sentinel` | `snapshot_missing` | (none, metadata only) | DONE (2026-05-27) |
| `hermes-ruleset-integrity` | `ruleset_mismatch` | `artifacts/ruleset_audit/` | DONE (2026-05-27) |
| `cron_watchdog.sh` + `ops_supervisor` | `cron_missed` | `logs/watchdog.log` / supervisor JSON | DONE (2026-05-30) |
| `hermes-contradiction-detector` | `contradiction_detected` | `artifacts/ops/contradiction_ledger/` | DONE (2026-05-30) |

---

## Operator triage (Town inbox)

When Town receives a `[Hermes]` email, map `event_type` → likely root cause before acting:

| event_type | First check | Cross-ref skill |
|---|---|---|
| `cron_missed` | `agents.log` for `ModuleNotFoundError: No module named 'tools'` | `openclaw-data-pipeline-debug` Class P; `openclaw-cron-scheduler-debug` |
| `snapshot_missing` | Recent production run logs; cache-warm step timeout | `openclaw-data-pipeline-debug` Class O |
| `first_fire_fail` | Hedge artifact path + validator JSON | `hermeslink-state-capture` |
| `ruleset_mismatch` | `CLAUDE.md` ruleset ID vs `run_screen.py` pin | `screener-ops` Active Ruleset |
| `contradiction_detected` | `artifacts/ops/contradiction_ledger/latest.md` | `hermeslink-state-capture` |
| `held_spec_ledger` | `artifacts/ops/held_spec_ledger/latest.md` | `governance-spec-enforcement` |

**2026-06-24 pipeline recovery patterns** (Classes M–P in `openclaw-data-pipeline-debug`):

- **Class M** — yfinance date parse (`datetime.isoformat()` → use `strftime("%Y-%m-%d")`)
- **Class N** — delisted ticker still in screen after one-loader fix (audit all universe consumers)
- **Class O** — production cache-warm timeout (`--warm-sources` CLI default masked function default)
- **Class P** — cron `sys.path` isolation (`from tools.*` fails without `PROJECT_ROOT` on path)

Town creates tasks and context only — operator executes fixes on WSL; Town does not mutate repo or cron.

---

## Verification Checklist

- [ ] **Phase A smoke test:** `python3 tools/smoke_operator_delivery.py` — dry_run=True, verify log output
- [ ] **Phase B unit test:** `pytest tests/test_operator_delivery.py` — all event types, dedupe logic
- [ ] **Phase B integration test:** Set `OPERATOR_DELIVERY_DRY_RUN=1`, run each target job, verify logged events
- [ ] **Phase B live test:** Set `OPERATOR_DELIVERY_DRY_RUN=0` (after operator approval), run one job, verify email received

### Enable live Town email (operator host)

```bash
# 1. Smoke all event types in dry-run (default)
export OPERATOR_DELIVERY_DRY_RUN=1
python3 tools/build_hermes_knowledge_layer.py
python3 agents/hermes-contradiction-detector/run_job.py
python3 agents/hermes-held-spec-ledger/run_job.py
python3 agents/ops_supervisor/supervisor.py --as-of $(date +%F)

# 2. After operator approval — add to .env (never commit secrets):
#    OPERATOR_DELIVERY_DRY_RUN=0
#    TOWN_EMAIL=your@email.com
#    SMTP_USER / SMTP_PASSWORD (see .env.example)

# 3. Re-run one job; confirm [Hermes] email in TOWN_EMAIL inbox
```
- [ ] **Cursor discovery:** `hermes -s town-operator-bridge` in Cursor, verify skill loads
- [ ] **Ledger audit:** `hermes knowledge_read artifact=held_spec_ledger` — verify delivery timestamps

---

## References

- **Spec 090:** `specs/changes/spec_090_town_hermes_bridge.md`
- **Implementation:** `common/operator_delivery.py` (event construction), `common/alert_email.py` (delivery)
- **Tests:** `tests/test_operator_delivery.py` (unit tests), `tools/smoke_operator_delivery.py` (manual smoke)
- **Env:** `.env` and `.env.example` (`TOWN_EMAIL`, `OPERATOR_DELIVERY_DRY_RUN`)
- **Related skills:** `screener-ops.md` (ops overview)

---

## FAQ

**Q: What if Town is down or unreachable?**  
A: Email delivery is async and fault-tolerant. If email service is down, `send_operator_event()` returns False and logs the error. The job continues; operator reviews the error in Hermes logs.

**Q: Can Town mutate production state?**  
A: No. Town has read-only permissions on the repo. Any Town-initiated action (e.g., "approve this spec") must be manually executed by the operator, not by Town scripts.

**Q: How do I test this without sending real emails?**  
A: Set `OPERATOR_DELIVERY_DRY_RUN=1` (the default). Events are logged to stdout/logs but email is NOT sent. Verified Phase A behavior before toggling to dry_run=0.

**Q: When will webhooks be supported?**  
A: Phase C. The email path is stable and production-ready; webhooks are a future enhancement to enable tighter Town integration (native task creation, no email parsing).

**Q: What's the dedup window?**  
A: FAIL=15 min, WARN=30 min, INFO=60 min. Prevents duplicate events if a job re-runs frequently. Override with `skip_dedupe=True` for retries.
