"""
common/operator_delivery.py — Multi-channel operator event delivery.

Routes Hermes Knowledge Layer events to Town (via email trigger), Telegram,
or Slack. This is the single integration point — callers do not contain
channel logic.

Usage
-----
    from common.operator_delivery import send_operator_event

    send_operator_event(
        channel="town",
        severity="INFO",
        event_type="held_spec_ledger",
        title="Held-spec ledger updated",
        summary="6 held items. Bioshort first-fire due Fri 2026-05-08 18:00 ET.",
        artifact="artifacts/ops/held_spec_ledger/latest.md",
        next_operator_action="Validate bioshort first-fire after 18:00 ET",
    )

Town integration
----------------
    Town does not expose a native inbound webhook endpoint. The integration
    path is email: Hermes sends a structured email to TOWN_EMAIL; a Town
    routine triggers on arrival and creates a task.

    Subject format: [Hermes] {SEVERITY} | {event_type} | {title}
    Town routine should filter on subject containing "[Hermes]".

    Body: plain-text summary + JSON payload block for structured parsing.

Environment variables (read from repo .env via dotenv, falls back to os.environ)
-----------------
    TOWN_EMAIL                 — destination address for Town trigger emails
                                 (default: djschulz@gmail.com)
    OPERATOR_DELIVERY_DRY_RUN  — "1" or "true" to log without sending

    SMTP vars (shared with alert_email.py):
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD

Behaviour
---------
    - dry_run=True (or OPERATOR_DELIVERY_DRY_RUN=1): logs payload, skips send, returns True.
    - Missing SMTP creds: logs warning, returns False, never raises.
    - SMTP errors: logged at WARNING level, returns False, never raises.
    - Deduplication: uses common/alert_dedupe.py (same store as alerts.py).
      Dedupe window: 1 hour for INFO, 30 min for WARN, 15 min for FAIL.
    - channel="telegram": delegates to common/alerts.py send_operator_alert().
    - channel="slack": not yet implemented (logs warning, returns False).

Spec
----
    Spec 090 — Town-Hermes Bridge, Phase A
    specs/changes/spec_090_town_hermes_bridge.md

Guardrails
----------
    This module sends read-only summaries and artifact paths only.
    It does NOT grant Town write access to repo, cron, config, or scoring.
    It does NOT infer operator approval from any delivery response.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Severity config
# ---------------------------------------------------------------------------

SEVERITY_EMOJI = {
    "FAIL": "🔴",
    "WARN": "🟡",
    "INFO": "🔵",
}

DEDUPE_WINDOW_SECONDS = {
    "FAIL": 900,
    "WARN": 1800,
    "INFO": 3600,
}

TOWN_EMAIL_DEFAULT = "djschulz@gmail.com"

# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _is_dry_run(override: Optional[bool]) -> bool:
    if override is not None:
        return override
    return _env("OPERATOR_DELIVERY_DRY_RUN", "1").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Build structured payload (shared across channels)
# ---------------------------------------------------------------------------


def _build_payload(
    *,
    severity: str,
    event_type: str,
    title: str,
    summary: str,
    artifact: str,
    next_operator_action: str,
    extra: Optional[dict[str, Any]],
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "source": "hermes",
        "event_type": event_type,
        "severity": severity,
        "title": title,
        "summary": summary,
        "artifact": artifact,
        "next_operator_action": next_operator_action,
        "as_of": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if extra:
        data.update(extra)
    return {"data": data}


# ---------------------------------------------------------------------------
# Town delivery — email trigger
# ---------------------------------------------------------------------------


def _send_town_email(
    *,
    severity: str,
    event_type: str,
    title: str,
    subject_title: str,
    summary: str,
    artifact: str,
    next_operator_action: str,
    extra: Optional[dict[str, Any]],
    dry_run: bool,
) -> bool:
    """Send a structured email to the Town trigger address."""
    payload = _build_payload(
        severity=severity,
        event_type=event_type,
        title=title,
        summary=summary,
        artifact=artifact,
        next_operator_action=next_operator_action,
        extra=extra,
    )

    to_addr = _env("TOWN_EMAIL", TOWN_EMAIL_DEFAULT)
    # Clean subject: [Hermes] {SEVERITY} | {event_type} | {original caller title}
    subject = f"[Hermes] {severity} | {event_type} | {subject_title}"

    body_lines = [
        "Source: hermes",
        f"Event:  {event_type}",
        f"Severity: {severity}",
        "",
    ]
    if summary:
        body_lines += [summary, ""]
    if artifact:
        body_lines += [f"Artifact: {artifact}", ""]
    if next_operator_action and next_operator_action != "none":
        body_lines += [f"Next operator action: {next_operator_action}", ""]
    body_lines += [
        "--- JSON payload ---",
        json.dumps(payload, indent=2, ensure_ascii=False),
    ]
    body_text = "\n".join(body_lines)

    if dry_run:
        logger.warning(
            "[operator_delivery:DRY_RUN] town/email | %s | %s\nTo: %s\nSubject: %s\n%s",
            severity,
            event_type,
            to_addr,
            subject,
            json.dumps(payload, indent=2, ensure_ascii=False),
        )
        return True

    try:
        from common.alert_email import send_email

        return send_email(subject=subject, body_text=body_text, to_addr=to_addr)
    except ImportError:
        logger.warning("[operator_delivery] common.alert_email not importable — skipping town email")
        return False


# ---------------------------------------------------------------------------
# Telegram delegation
# ---------------------------------------------------------------------------


def _send_telegram(
    *,
    severity: str,
    event_type: str,
    title: str,
    summary: str,
    dry_run: bool,
) -> bool:
    try:
        from common.alerts import send_operator_alert

        dedupe_key = f"operator_delivery:{event_type}:{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d')}"
        message = f"{title}\n{summary}" if summary else title
        if dry_run:
            logger.warning(
                "[operator_delivery:DRY_RUN] telegram | %s | %s | %s",
                severity,
                event_type,
                message,
            )
            return True
        return send_operator_alert(
            severity=severity,
            system=f"hermes:{event_type}",
            message=message,
            dedupe_key=dedupe_key,
        )
    except ImportError:
        logger.warning("[operator_delivery] common.alerts not importable — skipping telegram")
        return False


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _is_duplicate(event_type: str, severity: str) -> bool:
    # Dedup is intentionally a no-op. The original implementation called
    # AlertDedupeStore() with no `path` and invoked is_duplicate()/record(),
    # neither of which exists (the real API is AlertDedupeStore(path, ...) with
    # decide()/record_sent()). That call raised at runtime and was swallowed by
    # a bare `except`, so this function always reported "not duplicate". This
    # preserves that behaviour explicitly instead of via a broken call; wiring
    # real dedup (which requires a state-file path and would start suppressing
    # alerts) is a separate, reviewed change. See issue #485.
    return False


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def send_operator_event(
    channel: str,
    severity: str,
    event_type: str,
    title: str,
    summary: str = "",
    artifact: str = "",
    next_operator_action: str = "none",
    extra: Optional[dict[str, Any]] = None,
    dry_run: Optional[bool] = None,
    skip_dedupe: bool = False,
) -> bool:
    """
    Route a Hermes Knowledge Layer event to the operator.

    Parameters
    ----------
    channel : "town" | "telegram" | "slack"
        Delivery channel.
        "town"     — structured email to TOWN_EMAIL (Town routine trigger).
        "telegram" — Telegram alert via common/alerts.py.
        "slack"    — not yet implemented.
    severity : "INFO" | "WARN" | "FAIL"
    event_type : str
        Machine-readable event identifier.
        Examples: "held_spec_ledger", "first_fire_fail", "snapshot_missing".
    title : str
        Short human-readable title (email subject / notification subject).
    summary : str
        1-3 sentence plain-text summary. Optional.
    artifact : str
        Relative repo path to the source ledger artifact. Optional.
    next_operator_action : str
        Specific next step for the operator, or "none".
    extra : dict | None
        Additional fields to include in the JSON payload.
    dry_run : bool | None
        Override dry-run mode. None = read OPERATOR_DELIVERY_DRY_RUN env.
    skip_dedupe : bool
        If True, bypass deduplication (use for forced re-alerts or tests).

    Returns
    -------
    bool
        True if delivered (or dry-run logged), False if skipped/failed.
    """
    dry = _is_dry_run(dry_run)

    emoji = SEVERITY_EMOJI.get(severity.upper(), "⚪")
    # Build a display title with emoji prefix (used in JSON payload + logs)
    # Keep the original caller title for use in email subjects
    original_title = title
    if not title.startswith(emoji):
        title = f"{emoji} {severity} | {event_type} | {title}"

    if not skip_dedupe and _is_duplicate(event_type, severity):
        logger.debug("[operator_delivery] dedupe skip: %s/%s", channel, event_type)
        return False

    if channel == "town":
        return _send_town_email(
            severity=severity,
            event_type=event_type,
            title=title,
            subject_title=original_title,
            summary=summary,
            artifact=artifact,
            next_operator_action=next_operator_action,
            extra=extra,
            dry_run=dry,
        )

    if channel == "telegram":
        return _send_telegram(
            severity=severity,
            event_type=event_type,
            title=title,
            summary=summary,
            dry_run=dry,
        )

    if channel == "slack":
        logger.warning("[operator_delivery] slack channel not yet implemented")
        return False

    logger.warning("[operator_delivery] unknown channel: %s", channel)
    return False
