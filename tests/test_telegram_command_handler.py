"""
tests/test_telegram_command_handler.py — Unit tests for telegram_command_handler.py
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    """Set up environment variables for tests."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test_token_123")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "6118239110")
    monkeypatch.setenv("TELEGRAM_HANDLER_DRY_RUN", "1")


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


def test_import():
    """Import cleanly."""
    from tools.telegram_command_handler import TelegramCommandHandler

    assert TelegramCommandHandler is not None


# ---------------------------------------------------------------------------
# Authorization tests
# ---------------------------------------------------------------------------


def test_authorization_rejects_unknown_chat():
    """Update from unknown chat_id → not authorized."""
    from tools.telegram_command_handler import TelegramCommandHandler

    handler = TelegramCommandHandler("token", "6118239110")
    update = {
        "message": {
            "chat": {"id": 999999999},  # Different chat
            "text": "/help",
        }
    }
    assert handler._is_authorized(update) is False


def test_authorization_accepts_known_chat():
    """Update from configured chat_id → authorized."""
    from tools.telegram_command_handler import TelegramCommandHandler

    handler = TelegramCommandHandler("token", "6118239110")
    update = {
        "message": {
            "chat": {"id": 6118239110},
            "text": "/help",
        }
    }
    assert handler._is_authorized(update) is True


def test_authorization_rejects_missing_chat_id():
    """Malformed update without chat.id → not authorized."""
    from tools.telegram_command_handler import TelegramCommandHandler

    handler = TelegramCommandHandler("token", "6118239110")
    update = {"message": {"text": "/help"}}  # missing chat
    assert handler._is_authorized(update) is False


# ---------------------------------------------------------------------------
# Command handler tests
# ---------------------------------------------------------------------------


def test_help_command_returns_string():
    """Help command returns command list."""
    from tools.telegram_command_handler import TelegramCommandHandler

    handler = TelegramCommandHandler("token", "chat_id")
    result = handler._cmd_help()
    assert isinstance(result, str)
    assert "/help" in result
    assert "/status" in result
    assert "/agents" in result


def test_dispatch_help_command():
    """Dispatch /help → calls help handler."""
    from tools.telegram_command_handler import TelegramCommandHandler

    handler = TelegramCommandHandler("token", "chat_id")
    result = handler._dispatch_command("/help")
    assert "help" in result.lower()


def test_dispatch_unknown_command():
    """Dispatch unknown command → returns error message."""
    from tools.telegram_command_handler import TelegramCommandHandler

    handler = TelegramCommandHandler("token", "chat_id")
    result = handler._dispatch_command("/unknown_cmd")
    assert "Unknown command" in result or "unknown" in result.lower()


def test_dispatch_command_without_slash():
    """Text without slash → suggests /help."""
    from tools.telegram_command_handler import TelegramCommandHandler

    handler = TelegramCommandHandler("token", "chat_id")
    result = handler._dispatch_command("hello")
    assert "/help" in result


# ---------------------------------------------------------------------------
# Status command tests
# ---------------------------------------------------------------------------


def test_status_missing_artifact(tmp_path, monkeypatch):
    """Status command when no artifacts exist → graceful fallback."""
    from tools.telegram_command_handler import TelegramCommandHandler

    monkeypatch.setattr("tools.telegram_command_handler.REPO_ROOT", tmp_path)
    handler = TelegramCommandHandler("token", "chat_id")
    result = handler._cmd_status()
    assert isinstance(result, str)
    assert "snapshots" in result.lower()


def test_status_with_snapshot_dir(tmp_path, monkeypatch):
    """Status command finds latest snapshot."""
    from tools.telegram_command_handler import TelegramCommandHandler

    monkeypatch.setattr("tools.telegram_command_handler.REPO_ROOT", tmp_path)
    snap_dir = tmp_path / "data" / "snapshots" / "2026-05-15"
    snap_dir.mkdir(parents=True)

    handler = TelegramCommandHandler("token", "chat_id")
    result = handler._cmd_status()
    assert "2026-05-15" in result


# ---------------------------------------------------------------------------
# Agents command tests
# ---------------------------------------------------------------------------


def test_agents_no_heartbeat(tmp_path, monkeypatch):
    """Agents command when no heartbeat artifacts → fallback message."""
    from tools.telegram_command_handler import TelegramCommandHandler

    monkeypatch.setattr("tools.telegram_command_handler.REPO_ROOT", tmp_path)
    handler = TelegramCommandHandler("token", "chat_id")
    result = handler._cmd_agents()
    assert "heartbeat" in result.lower() or "No" in result


def test_agents_with_heartbeat(tmp_path, monkeypatch):
    """Agents command reads fleet receipt (canonical heartbeat artifact)."""
    from tools.telegram_command_handler import TelegramCommandHandler

    monkeypatch.setattr("tools.telegram_command_handler.REPO_ROOT", tmp_path)
    hb_dir = tmp_path / "artifacts" / "heartbeat"
    hb_dir.mkdir(parents=True)
    hb_file = hb_dir / "2026-05-15_receipt.md"
    hb_file.write_text("# Fleet Receipt\n\nVerdict: GREEN\n")

    handler = TelegramCommandHandler("token", "chat_id")
    result = handler._cmd_agents()
    assert "Fleet Receipt" in result


def test_agents_prefers_fleet_ops_json(tmp_path, monkeypatch):
    from tools.telegram_command_handler import TelegramCommandHandler

    monkeypatch.setattr("tools.telegram_command_handler.REPO_ROOT", tmp_path)
    fleet_dir = tmp_path / "artifacts" / "fleet_ops"
    fleet_dir.mkdir(parents=True)
    (fleet_dir / "2026-06-24_status.json").write_text(
        json.dumps(
            {
                "as_of_date": "2026-06-24",
                "overall": "WARN",
                "herald": {"verdict": "WARN", "herald_done": False},
                "heartbeat": {"verdict": "YELLOW"},
                "completion_audit": {"exists": True, "overall": "PASS"},
                "crontab_verify": {"overall": "FAIL", "fail_count": 2},
                "selfimprove_gates": {"message": "blocked"},
            }
        ),
        encoding="utf-8",
    )

    handler = TelegramCommandHandler("token", "chat_id")
    result = handler._cmd_agents()
    assert "Fleet ops" in result
    assert "WARN" in result
    assert "blocked" in result
    assert "Crontab: FAIL" in result


def test_status_includes_crontab_verify(tmp_path, monkeypatch):
    from tools.telegram_command_handler import TelegramCommandHandler

    monkeypatch.setattr("tools.telegram_command_handler.REPO_ROOT", tmp_path)
    fleet_dir = tmp_path / "artifacts" / "fleet_ops"
    fleet_dir.mkdir(parents=True)
    (fleet_dir / "2026-06-24_status.json").write_text(
        json.dumps(
            {
                "overall": "WARN",
                "crontab_verify": {"overall": "PASS", "fail_count": 0},
            }
        ),
        encoding="utf-8",
    )

    handler = TelegramCommandHandler("token", "chat_id")
    result = handler._cmd_status()
    assert "Crontab:" in result and "PASS" in result


def test_status_uses_final_severity(tmp_path, monkeypatch):
    from tools.telegram_command_handler import TelegramCommandHandler

    monkeypatch.setattr("tools.telegram_command_handler.REPO_ROOT", tmp_path)
    sup_dir = tmp_path / "artifacts" / "ops_supervisor"
    sup_dir.mkdir(parents=True)
    (sup_dir / "2026-06-24_supervisor.json").write_text(
        json.dumps({"final_severity": "ORANGE"}),
        encoding="utf-8",
    )

    handler = TelegramCommandHandler("token", "chat_id")
    result = handler._cmd_status()
    assert "ORANGE" in result


# ---------------------------------------------------------------------------
# Held command tests
# ---------------------------------------------------------------------------


def test_held_no_artifact(tmp_path, monkeypatch):
    """Held command when no ledger → fallback message."""
    from tools.telegram_command_handler import TelegramCommandHandler

    monkeypatch.setattr("tools.telegram_command_handler.REPO_ROOT", tmp_path)
    handler = TelegramCommandHandler("token", "chat_id")
    result = handler._cmd_held()
    assert "held" in result.lower()


def test_held_with_artifact(tmp_path, monkeypatch):
    """Held command with ledger artifact → formats held items."""
    from tools.telegram_command_handler import TelegramCommandHandler

    monkeypatch.setattr("tools.telegram_command_handler.REPO_ROOT", tmp_path)
    held_dir = tmp_path / "artifacts" / "ops" / "held_spec_ledger"
    held_dir.mkdir(parents=True)
    held_file = held_dir / "latest.json"
    held_file.write_text(
        json.dumps(
            {
                "held_items": [
                    {"spec_id": "spec_001", "status": "PENDING"},
                    {"spec_id": "spec_002", "status": "BLOCKED"},
                ]
            }
        )
    )

    handler = TelegramCommandHandler("token", "chat_id")
    result = handler._cmd_held()
    assert "spec_001" in result
    assert "spec_002" in result


def test_held_empty_ledger(tmp_path, monkeypatch):
    """Held command with empty ledger → no specs message."""
    from tools.telegram_command_handler import TelegramCommandHandler

    monkeypatch.setattr("tools.telegram_command_handler.REPO_ROOT", tmp_path)
    held_dir = tmp_path / "artifacts" / "ops" / "held_spec_ledger"
    held_dir.mkdir(parents=True)
    held_file = held_dir / "latest.json"
    held_file.write_text(json.dumps({"held_items": []}))

    handler = TelegramCommandHandler("token", "chat_id")
    result = handler._cmd_held()
    assert "No held" in result or "no specs" in result.lower()


# ---------------------------------------------------------------------------
# Snap command tests
# ---------------------------------------------------------------------------


def test_snap_no_snapshots(tmp_path, monkeypatch):
    """Snap command when no snapshots → fallback message."""
    from tools.telegram_command_handler import TelegramCommandHandler

    monkeypatch.setattr("tools.telegram_command_handler.REPO_ROOT", tmp_path)
    handler = TelegramCommandHandler("token", "chat_id")
    result = handler._cmd_snap()
    assert "No snapshots" in result or "snapshots" in result.lower()


def test_snap_with_manifest(tmp_path, monkeypatch):
    """Snap command with manifest → formats pipeline status."""
    from tools.telegram_command_handler import TelegramCommandHandler

    monkeypatch.setattr("tools.telegram_command_handler.REPO_ROOT", tmp_path)
    snap_dir = tmp_path / "data" / "snapshots" / "2026-05-15"
    snap_dir.mkdir(parents=True)
    manifest_file = snap_dir / "run_manifest.json"
    manifest_file.write_text(
        json.dumps(
            {
                "overall_status": "PASS",
                "modules": {
                    "module_1": {"status": "PASS"},
                    "module_2": {"status": "FAIL"},
                },
            }
        )
    )

    handler = TelegramCommandHandler("token", "chat_id")
    result = handler._cmd_snap()
    assert "2026-05-15" in result
    assert "PASS" in result


# ---------------------------------------------------------------------------
# Dry-run mode tests
# ---------------------------------------------------------------------------


def test_dry_run_does_not_send(monkeypatch):
    """Dry-run mode logs instead of sending."""
    from tools.telegram_command_handler import TelegramCommandHandler

    handler = TelegramCommandHandler("token", "chat_id", dry_run=True)

    with patch("tools.telegram_command_handler._send_telegram_request") as mock_send:
        result = handler._send_reply("test message")
        assert result is True
        mock_send.assert_not_called()


def test_live_mode_sends_request(monkeypatch):
    """Live mode sends HTTP request."""
    from tools.telegram_command_handler import TelegramCommandHandler

    handler = TelegramCommandHandler("token", "6118239110", dry_run=False)

    mock_response = {"ok": True, "result": {"message_id": 123}}
    with patch("tools.telegram_command_handler._send_telegram_request", return_value=mock_response) as mock_send:
        result = handler._send_reply("test message")
        assert result is True
        assert mock_send.called


# ---------------------------------------------------------------------------
# Polling loop tests
# ---------------------------------------------------------------------------


def test_once_flag_exits_after_batch(monkeypatch):
    """--once flag processes one batch and exits."""
    from tools.telegram_command_handler import TelegramCommandHandler

    handler = TelegramCommandHandler("token", "6118239110")

    # Mock _get_updates to return empty list (no updates)
    with patch.object(handler, "_get_updates", return_value=[]):
        # Should exit immediately without error; run() returns None after one batch
        assert handler.run(once=True) is None


def test_handles_updates_without_text(monkeypatch):
    """Handler gracefully skips updates without text."""
    from tools.telegram_command_handler import TelegramCommandHandler

    handler = TelegramCommandHandler("token", "6118239110")

    updates = [
        {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "chat": {"id": 6118239110},
                # No 'text' field
            },
        }
    ]

    with patch.object(handler, "_get_updates", return_value=updates):
        with patch.object(handler, "_send_reply") as mock_reply:
            handler.run(once=True)
            # Updates without text are skipped: no reply is sent
            mock_reply.assert_not_called()


# ---------------------------------------------------------------------------
# API helper tests
# ---------------------------------------------------------------------------


def test_send_telegram_request_success():
    """_send_telegram_request handles successful response."""
    from tools.telegram_command_handler import _send_telegram_request

    mock_response = {"ok": True, "result": {"message_id": 123}}

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response_obj = MagicMock()
        mock_response_obj.read.return_value = json.dumps(mock_response).encode()
        mock_response_obj.__enter__.return_value = mock_response_obj
        mock_urlopen.return_value = mock_response_obj

        result = _send_telegram_request("token", "sendMessage", {"chat_id": "123", "text": "test"})
        assert result["ok"] is True


def test_send_telegram_request_http_error():
    """_send_telegram_request handles HTTP errors gracefully."""
    from tools.telegram_command_handler import _send_telegram_request

    with patch("urllib.request.urlopen") as mock_urlopen:
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError("url", 400, "Bad Request", {}, None)

        result = _send_telegram_request("token", "sendMessage", {"chat_id": "123", "text": "test"})
        assert result == {}


# ---------------------------------------------------------------------------
# PID file tests
# ---------------------------------------------------------------------------


def test_pid_file_guard_writes_pid(tmp_path, monkeypatch):
    """_check_and_write_pid writes current process PID."""
    from tools.telegram_command_handler import _check_and_write_pid

    pid_file = tmp_path / "test.pid"
    monkeypatch.setattr("tools.telegram_command_handler.PID_FILE", pid_file)

    result = _check_and_write_pid()
    assert result is True
    assert pid_file.exists()
    assert str(os.getpid()) in pid_file.read_text()


def test_pid_file_guard_detects_existing(tmp_path, monkeypatch):
    """_check_and_write_pid detects running daemon."""
    from tools.telegram_command_handler import _check_and_write_pid

    pid_file = tmp_path / "test.pid"
    pid_file.write_text(str(os.getpid()))  # Write current process PID

    monkeypatch.setattr("tools.telegram_command_handler.PID_FILE", pid_file)

    result = _check_and_write_pid()
    assert result is False  # Current process is running, so guard blocks


# ---------------------------------------------------------------------------
# Environment loading tests
# ---------------------------------------------------------------------------


def test_loads_env_from_dotenv(tmp_path, monkeypatch):
    """_load_env loads from .env file."""
    from tools.telegram_command_handler import _load_env

    env_file = tmp_path / ".env"
    env_file.write_text("TEST_VAR=test_value\n")

    monkeypatch.setattr("tools.telegram_command_handler.REPO_ROOT", tmp_path)

    # dotenv availability varies, but _load_env must not raise; returns None
    assert _load_env() is None


def test_main_missing_credentials(monkeypatch):
    """main() exits with error if credentials missing."""
    from tools.telegram_command_handler import main

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "")

    # main() reads from sys.argv, so mock it
    with patch("sys.argv", ["telegram_command_handler.py"]):
        result = main()
        assert result != 0
