"""
Tests for tools/send_alert.py

8 tests covering:
  - Payload format (required fields present)
  - FAIL vs WARN level differences
  - No-webhook graceful exit (no error)
  - Health packet field extraction
  - Rollback flag surfaced
  - Live performance field in payload
  - Dry-run mode (no HTTP call)
  - Missing health packet (empty dict → graceful)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tools.send_alert import (
    LEVEL_COLOR,
    _extract_action_items,
    _extract_gate_summary,
    _extract_live_perf,
    build_payload,
    send_alert,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_packet(
    level: str = "FAIL",
    fail_names=None,
    warn_names=None,
    action_items=None,
    rollback: bool = False,
    ruleset_id: str = "abc12345",
) -> dict:
    fail_names = fail_names or []
    warn_names = warn_names or []
    action_items = action_items or []
    return {
        "provenance": {"ruleset_id": ruleset_id, "as_of_date": "2026-03-05"},
        "gates": {
            "pass_count": 8 - len(fail_names) - len(warn_names),
            "warn_count": len(warn_names),
            "fail_count": len(fail_names),
            "fail_names": fail_names,
            "warn_names": warn_names,
        },
        "action_items": action_items,
        "ruleset_health": {"recommend_rollback": rollback},
        "live_performance": {},
    }


def _make_live_summary(mean_net: float = 0.02, n: int = 10) -> dict:
    return {
        "last_4w": {"mean_net_return": mean_net, "n_dates": n},
        "last_13w": {"mean_net_return": 0.03, "n_dates": 30},
    }


# ---------------------------------------------------------------------------
# 1. Payload format
# ---------------------------------------------------------------------------


class TestPayloadFormat:

    def test_required_fields_present(self):
        """Payload has attachments array with required keys."""
        packet = _make_packet()
        payload = build_payload("FAIL", "2026-03-05", packet, {})
        assert "attachments" in payload
        att = payload["attachments"][0]
        assert "color" in att
        assert "title" in att
        assert "text" in att
        assert "fields" in att

    def test_fail_color_is_red(self):
        payload = build_payload("FAIL", "2026-03-05", _make_packet(), {})
        att = payload["attachments"][0]
        assert att["color"] == LEVEL_COLOR["FAIL"]

    def test_warn_color_is_yellow(self):
        payload = build_payload("WARN", "2026-03-05", _make_packet(), {})
        att = payload["attachments"][0]
        assert att["color"] == LEVEL_COLOR["WARN"]


# ---------------------------------------------------------------------------
# 2. FAIL vs WARN level differences
# ---------------------------------------------------------------------------


class TestLevelDifferences:

    def test_fail_level_in_title(self):
        payload = build_payload("FAIL", "2026-03-05", _make_packet(), {})
        title = payload["attachments"][0]["title"]
        assert "FAIL" in title

    def test_warn_level_in_title(self):
        payload = build_payload("WARN", "2026-03-05", _make_packet(), {})
        title = payload["attachments"][0]["title"]
        assert "WARN" in title

    def test_fail_names_appear_in_fields(self):
        packet = _make_packet(fail_names=["ctgov_cache", "audit"])
        payload = build_payload("FAIL", "2026-03-05", packet, {})
        fields_text = json.dumps(payload["attachments"][0]["fields"])
        assert "ctgov_cache" in fields_text
        assert "audit" in fields_text


# ---------------------------------------------------------------------------
# 3. No-webhook graceful exit
# ---------------------------------------------------------------------------


class TestNoWebhook:

    def test_no_webhook_no_exception(self):
        """send_alert with no webhook URL exits cleanly (no exception)."""
        # No webhook configured → returns None without raising
        assert send_alert("FAIL", "2026-03-05", webhook_url=None) is None

    def test_no_webhook_no_http_call(self):
        """No HTTP request is made when webhook URL is None."""
        with patch("tools.send_alert.post_webhook") as mock_post:
            send_alert("FAIL", "2026-03-05", webhook_url=None)
        mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# 4. Health packet field extraction
# ---------------------------------------------------------------------------


class TestPacketExtraction:

    def test_gate_summary_extracted(self):
        packet = _make_packet(fail_names=["gate_a"], warn_names=["gate_b"])
        summary = _extract_gate_summary(packet)
        assert summary["fail"] == 1
        assert summary["warn"] == 1
        assert "gate_a" in summary["fail_names"]

    def test_action_items_extracted(self):
        packet = _make_packet(
            action_items=[
                {"severity": "FAIL", "type": "gate_fail", "detail": "ctgov_cache: stale"},
            ]
        )
        items = _extract_action_items(packet)
        assert len(items) == 1
        assert "ctgov_cache" in items[0]

    def test_rollback_flag_in_text(self):
        packet = _make_packet(rollback=True)
        payload = build_payload("FAIL", "2026-03-05", packet, {})
        text = payload["attachments"][0]["text"]
        assert "ROLLBACK" in text

    def test_missing_packet_graceful(self):
        """build_payload with empty dict packet doesn't raise."""
        payload = build_payload("WARN", "2026-03-05", {}, {})
        assert "attachments" in payload


# ---------------------------------------------------------------------------
# 5. Live performance field
# ---------------------------------------------------------------------------


class TestLivePerformanceField:

    def test_positive_net_return_shown(self):
        live = _make_live_summary(mean_net=0.032)
        payload = build_payload("WARN", "2026-03-05", _make_packet(), live)
        fields_text = json.dumps(payload["attachments"][0]["fields"])
        assert "+0.0320" in fields_text

    def test_negative_net_return_shown(self):
        live = _make_live_summary(mean_net=-0.015)
        payload = build_payload("WARN", "2026-03-05", _make_packet(), live)
        fields_text = json.dumps(payload["attachments"][0]["fields"])
        assert "-0.0150" in fields_text

    def test_no_live_summary_no_field(self):
        """No live_perf field in payload when summary is empty."""
        payload = build_payload("FAIL", "2026-03-05", _make_packet(), {})
        fields = payload["attachments"][0]["fields"]
        field_titles = [f["title"] for f in fields]
        assert "4w net return" not in field_titles
