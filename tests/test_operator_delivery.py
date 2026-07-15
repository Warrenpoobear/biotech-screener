"""
tests/test_operator_delivery.py — Unit tests for common/operator_delivery.py

Spec 090 Phase A acceptance tests.
All tests run with OPERATOR_DELIVERY_DRY_RUN=1 (no live SMTP).

Town integration uses email trigger (not webhook):
- channel="town" sends to TOWN_EMAIL via common/alert_email.send_email()
- Subject: [Hermes] {SEVERITY} | {event_type} | {title}
- Body: plain-text summary + JSON payload block
"""

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def dry_run_env(monkeypatch):
    """Force dry-run for all tests. No live SMTP ever."""
    monkeypatch.setenv("OPERATOR_DELIVERY_DRY_RUN", "1")


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def test_import():
    from common.operator_delivery import send_operator_event

    assert callable(send_operator_event)


# ---------------------------------------------------------------------------
# Dry-run: town channel (email)
# ---------------------------------------------------------------------------


def test_town_dry_run_returns_true(caplog):
    from common.operator_delivery import send_operator_event

    with caplog.at_level(logging.WARNING):
        result = send_operator_event(
            channel="town",
            severity="INFO",
            event_type="held_spec_ledger",
            title="Held-spec ledger updated",
            summary="6 held items. Bioshort first-fire due Fri 18:00 ET.",
            artifact="artifacts/ops/held_spec_ledger/latest.md",
            next_operator_action="Validate bioshort first-fire after 18:00 ET",
            skip_dedupe=True,
        )

    assert result is True


def test_town_dry_run_logs_to_addr(caplog):
    from common.operator_delivery import TOWN_EMAIL_DEFAULT, send_operator_event

    with caplog.at_level(logging.WARNING):
        send_operator_event(
            channel="town",
            severity="WARN",
            event_type="first_fire_fail",
            title="First-fire FAILED",
            summary="Artifact missing.",
            skip_dedupe=True,
        )

    combined = " ".join(caplog.messages)
    assert TOWN_EMAIL_DEFAULT in combined


def test_town_dry_run_logs_subject_prefix(caplog):
    from common.operator_delivery import send_operator_event

    with caplog.at_level(logging.WARNING):
        send_operator_event(
            channel="town",
            severity="FAIL",
            event_type="snapshot_missing",
            title="Snapshot missing",
            skip_dedupe=True,
        )

    combined = " ".join(caplog.messages)
    assert "[Hermes]" in combined


def test_town_dry_run_logs_json_payload(caplog):
    from common.operator_delivery import send_operator_event

    with caplog.at_level(logging.WARNING):
        send_operator_event(
            channel="town",
            severity="FAIL",
            event_type="snapshot_missing",
            title="Production snapshot missing",
            summary="No snapshot for 2026-05-07 after 16:45 ET.",
            artifact="data/snapshots/2026-05-07/",
            next_operator_action="Check logs/cron.log",
            skip_dedupe=True,
        )

    json_log = next((m for m in caplog.messages if '"source": "hermes"' in m), None)
    assert json_log is not None, "Expected JSON payload in log"

    payload = json.loads(json_log[json_log.index("{") :])
    data = payload["data"]
    assert data["source"] == "hermes"
    assert data["event_type"] == "snapshot_missing"
    assert data["severity"] == "FAIL"
    assert data["artifact"] == "data/snapshots/2026-05-07/"
    assert data["next_operator_action"] == "Check logs/cron.log"
    assert "as_of" in data


# ---------------------------------------------------------------------------
# Severity emoji
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "severity,emoji",
    [
        ("FAIL", "🔴"),
        ("WARN", "🟡"),
        ("INFO", "🔵"),
    ],
)
def test_severity_emojis(severity, emoji, caplog):
    from common.operator_delivery import send_operator_event

    with caplog.at_level(logging.WARNING):
        send_operator_event(
            channel="town",
            severity=severity,
            event_type="test_event",
            title="Test",
            skip_dedupe=True,
        )

    combined = " ".join(caplog.messages)
    assert emoji in combined


# ---------------------------------------------------------------------------
# Extra fields
# ---------------------------------------------------------------------------


def test_extra_fields_in_payload(caplog):
    from common.operator_delivery import send_operator_event

    with caplog.at_level(logging.WARNING):
        send_operator_event(
            channel="town",
            severity="INFO",
            event_type="held_spec_ledger",
            title="Ledger updated",
            extra={"held_count": 6, "first_fire_status": "PENDING_NOT_YET_DUE"},
            skip_dedupe=True,
        )

    json_log = next((m for m in caplog.messages if '"source": "hermes"' in m), None)
    assert json_log is not None
    payload = json.loads(json_log[json_log.index("{") :])
    assert payload["data"]["held_count"] == 6
    assert payload["data"]["first_fire_status"] == "PENDING_NOT_YET_DUE"


# ---------------------------------------------------------------------------
# not_allowed list in extra
# ---------------------------------------------------------------------------


def test_not_allowed_in_payload(caplog):
    from common.operator_delivery import send_operator_event

    with caplog.at_level(logging.WARNING):
        send_operator_event(
            channel="town",
            severity="FAIL",
            event_type="first_fire_fail",
            title="First-fire FAILED",
            extra={
                "not_allowed": [
                    "reactivate bioshort_watch LLM",
                    "run extra producer manually",
                ]
            },
            skip_dedupe=True,
        )

    json_log = next((m for m in caplog.messages if '"source": "hermes"' in m), None)
    assert json_log is not None
    payload = json.loads(json_log[json_log.index("{") :])
    assert "not_allowed" in payload["data"]
    assert "reactivate bioshort_watch LLM" in payload["data"]["not_allowed"]


# ---------------------------------------------------------------------------
# Live send path: mock alert_email.send_email
# ---------------------------------------------------------------------------


def test_town_live_calls_send_email(monkeypatch):
    """Live path reaches common.alert_email.send_email."""
    from common.operator_delivery import send_operator_event

    monkeypatch.setenv("OPERATOR_DELIVERY_DRY_RUN", "0")
    monkeypatch.setenv("TOWN_EMAIL", "testrecipient@example.com")

    mock_send = MagicMock(return_value=True)
    with patch("common.alert_email.send_email", mock_send):
        result = send_operator_event(
            channel="town",
            severity="INFO",
            event_type="held_spec_ledger",
            title="Test live send",
            skip_dedupe=True,
            dry_run=False,
        )

    assert result is True
    assert mock_send.called


def test_town_live_subject_format(monkeypatch):
    """Verify [Hermes] subject prefix reaches send_email."""
    from common.operator_delivery import send_operator_event

    monkeypatch.setenv("OPERATOR_DELIVERY_DRY_RUN", "0")
    monkeypatch.setenv("TOWN_EMAIL", "test@example.com")

    captured_subjects = []

    def fake_send_email(subject, body_text, *, body_html=None, to_addr=None, smtp_cls=None):
        captured_subjects.append(subject)
        return True

    with patch("common.alert_email.send_email", fake_send_email):
        send_operator_event(
            channel="town",
            severity="WARN",
            event_type="stale_artifact",
            title="Artifact stale",
            skip_dedupe=True,
            dry_run=False,
        )

    assert len(captured_subjects) == 1
    assert captured_subjects[0].startswith("[Hermes]")
    assert "WARN" in captured_subjects[0]
    assert "stale_artifact" in captured_subjects[0]


def test_town_live_body_contains_json(monkeypatch):
    """Verify email body contains parseable JSON payload."""
    from common.operator_delivery import send_operator_event

    monkeypatch.setenv("OPERATOR_DELIVERY_DRY_RUN", "0")
    monkeypatch.setenv("TOWN_EMAIL", "test@example.com")

    captured_bodies = []

    def fake_send_email(subject, body_text, *, body_html=None, to_addr=None, smtp_cls=None):
        captured_bodies.append(body_text)
        return True

    with patch("common.alert_email.send_email", fake_send_email):
        send_operator_event(
            channel="town",
            severity="INFO",
            event_type="held_spec_ledger",
            title="Test",
            summary="Test summary.",
            artifact="artifacts/ops/held_spec_ledger/latest.md",
            next_operator_action="Check things",
            skip_dedupe=True,
            dry_run=False,
        )

    assert captured_bodies
    body = captured_bodies[0]
    assert "--- JSON payload ---" in body
    json_start = body.index("{", body.index("--- JSON payload ---"))
    payload = json.loads(body[json_start:])
    assert payload["data"]["source"] == "hermes"
    assert payload["data"]["event_type"] == "held_spec_ledger"


def test_town_live_to_addr_from_env(monkeypatch):
    """TOWN_EMAIL env var reaches send_email to_addr."""
    from common.operator_delivery import send_operator_event

    monkeypatch.setenv("OPERATOR_DELIVERY_DRY_RUN", "0")
    monkeypatch.setenv("TOWN_EMAIL", "custom@example.com")

    captured_addrs = []

    def fake_send_email(subject, body_text, *, body_html=None, to_addr=None, smtp_cls=None):
        captured_addrs.append(to_addr)
        return True

    with patch("common.alert_email.send_email", fake_send_email):
        send_operator_event(
            channel="town",
            severity="INFO",
            event_type="test",
            title="Test",
            skip_dedupe=True,
            dry_run=False,
        )

    assert captured_addrs == ["custom@example.com"]


# ---------------------------------------------------------------------------
# Telegram delegation
# ---------------------------------------------------------------------------


def test_telegram_dry_run(caplog):
    from common.operator_delivery import send_operator_event

    with caplog.at_level(logging.WARNING):
        result = send_operator_event(
            channel="telegram",
            severity="FAIL",
            event_type="first_fire_fail",
            title="First-fire FAIL",
            summary="Artifact missing.",
            skip_dedupe=True,
        )

    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Unknown / unimplemented channels
# ---------------------------------------------------------------------------


def test_unknown_channel_returns_false(caplog):
    from common.operator_delivery import send_operator_event

    with caplog.at_level(logging.WARNING):
        result = send_operator_event(
            channel="discord",
            severity="INFO",
            event_type="test",
            title="Test",
            skip_dedupe=True,
        )

    assert result is False
    assert any("unknown channel" in m for m in caplog.messages)


def test_slack_not_implemented_returns_false(caplog):
    from common.operator_delivery import send_operator_event

    with caplog.at_level(logging.WARNING):
        result = send_operator_event(
            channel="slack",
            severity="INFO",
            event_type="test",
            title="Test",
            skip_dedupe=True,
        )

    assert result is False
    assert any("slack" in m for m in caplog.messages)


# ---------------------------------------------------------------------------
# OPERATOR_DELIVERY_DRY_RUN env parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("val", ["1", "true", "True", "TRUE", "yes"])
def test_dry_run_env_truthy(val, monkeypatch):
    from common.operator_delivery import _is_dry_run

    monkeypatch.setenv("OPERATOR_DELIVERY_DRY_RUN", val)
    assert _is_dry_run(None) is True


@pytest.mark.parametrize("val", ["0", "false", "False", "no", ""])
def test_dry_run_env_falsy(val, monkeypatch):
    from common.operator_delivery import _is_dry_run

    monkeypatch.setenv("OPERATOR_DELIVERY_DRY_RUN", val)
    assert _is_dry_run(None) is False


def test_dry_run_override_true(monkeypatch):
    from common.operator_delivery import _is_dry_run

    monkeypatch.setenv("OPERATOR_DELIVERY_DRY_RUN", "0")
    assert _is_dry_run(True) is True


def test_dry_run_override_false(monkeypatch):
    from common.operator_delivery import _is_dry_run

    monkeypatch.setenv("OPERATOR_DELIVERY_DRY_RUN", "1")
    assert _is_dry_run(False) is False
