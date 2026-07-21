#!/usr/bin/env python3
"""
tools/telegram_command_handler.py — Hermes Telegram command handler.

Long-polling daemon that listens for operator commands in the Telegram chat
and responds with read-only system state queries.

Commands:
    /help       List available commands
    /status     System health overview (snapshot, pipeline, alerts)
    /agents     Fleet agent health summary
    /held       Held spec ledger
    /snap       Latest snapshot pipeline status

Environment variables (from .env):
    TELEGRAM_BOT_TOKEN    — bot token from @BotFather (required)
    TELEGRAM_CHAT_ID      — destination chat ID (required; only this chat responds)
    TELEGRAM_HANDLER_DRY_RUN — "1" or "true" to log without sending (default: "0")

Usage:
    # Run as daemon (long-polling loop)
    python3 tools/telegram_command_handler.py

    # Dry-run: process one batch and log replies without sending
    python3 tools/telegram_command_handler.py --dry-run --once

    # Process one batch and exit
    python3 tools/telegram_command_handler.py --once

Cron integration (manual — not auto-wired):
    @reboot sleep 60 && cd /mnt/c/Projects/biotech_screener/biotech-screener && \
        python3 tools/telegram_command_handler.py >> logs/telegram_handler.log 2>&1

PID file guard:
    artifacts/ops/telegram_handler.pid — prevents duplicate daemon instances
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
PID_FILE = REPO_ROOT / "artifacts" / "ops" / "telegram_handler.pid"
POLL_TIMEOUT = 30  # seconds — long-polling timeout

TELEGRAM_API_BASE = "https://api.telegram.org/bot{token}"

SNAPSHOT_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _list_snapshot_dates(snap_dir: Path) -> list[str]:
    """Return sorted YYYY-MM-DD subdirs of snap_dir; skips non-date entries like 'resolutions'."""
    if not snap_dir.exists():
        return []
    return sorted(d.name for d in snap_dir.iterdir() if d.is_dir() and SNAPSHOT_DATE_RE.match(d.name))


# ---------------------------------------------------------------------------
# Environment loading
# ---------------------------------------------------------------------------


def _load_env() -> None:
    """Load repo .env into os.environ if dotenv is available."""
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(REPO_ROOT / ".env", override=False)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# PID file guard
# ---------------------------------------------------------------------------


def _check_and_write_pid() -> bool:
    """Check if daemon already running; write PID if not. Returns False if already running."""
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)

    if PID_FILE.exists():
        try:
            existing_pid = int(PID_FILE.read_text().strip())
            # Check if process with this PID exists
            if _process_exists(existing_pid):
                logger.warning("Daemon already running (PID %d)", existing_pid)
                return False
        except (ValueError, OSError):
            pass

    # Write current PID
    PID_FILE.write_text(str(os.getpid()))
    return True


def _process_exists(pid: int) -> bool:
    """Check if process with given PID is alive."""
    try:
        os.kill(pid, 0)  # signal 0 = check only, don't send
        return True
    except (OSError, ProcessLookupError):
        return False


def _remove_pid_file() -> None:
    """Remove PID file on exit."""
    try:
        PID_FILE.unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Telegram API helpers
# ---------------------------------------------------------------------------


def _send_telegram_request(
    token: str,
    method: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Send HTTP request to Telegram API. Returns response dict or empty dict on error."""
    url = f"{TELEGRAM_API_BASE.format(token=token)}/{method}"
    payload = json.dumps(params).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        logger.warning("Telegram HTTP %s: %s", exc.code, exc.read()[:200])
        return {}
    except Exception as exc:
        logger.warning("Telegram request failed: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Command handler class
# ---------------------------------------------------------------------------


class TelegramCommandHandler:
    def __init__(self, token: str, chat_id: str, dry_run: bool = False):
        self.token = token
        self.chat_id = chat_id
        self.dry_run = dry_run
        self.last_update_id = 0
        self.command_handlers = {
            "help": self._cmd_help,
            "status": self._cmd_status,
            "agents": self._cmd_agents,
            "held": self._cmd_held,
            "snap": self._cmd_snap,
        }

    def _is_authorized(self, update: dict[str, Any]) -> bool:
        """Check if update is from the authorized chat."""
        try:
            message = update.get("message", {})
            chat = message.get("chat", {})
            return str(chat.get("id")) == str(self.chat_id)
        except (KeyError, TypeError):
            return False

    def _get_updates(self) -> list[dict[str, Any]]:
        """Fetch pending updates from Telegram API."""
        resp = _send_telegram_request(
            self.token,
            "getUpdates",
            {
                "offset": self.last_update_id + 1,
                "timeout": POLL_TIMEOUT,
                "allowed_updates": ["message"],
            },
        )
        if resp.get("ok"):
            updates = resp.get("result", [])
            if updates:
                self.last_update_id = max(u.get("update_id", 0) for u in updates)
            return updates
        return []

    def _send_reply(self, text: str) -> bool:
        """Send a reply message to the chat."""
        if self.dry_run:
            logger.warning("[DRY_RUN] telegram reply: %s", text)
            return True

        resp = _send_telegram_request(
            self.token,
            "sendMessage",
            {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "HTML",
            },
        )
        return resp.get("ok", False)

    def _dispatch_command(self, text: str) -> str:
        """Parse and dispatch a command. Returns response text."""
        text = text.strip()
        if not text.startswith("/"):
            return "Send /help for available commands."

        parts = text.split(None, 1)
        cmd = parts[0][1:].lower()  # remove '/', lowercase
        handler = self.command_handlers.get(cmd)
        if handler:
            try:
                return handler()
            except Exception as exc:
                logger.exception("Command handler failed: %s", exc)
                return f"Error: {exc}"
        return f"Unknown command: /{cmd}\nSend /help for available commands."

    # -----------------------------------------------------------------------
    # Command implementations
    # -----------------------------------------------------------------------

    def _cmd_help(self) -> str:
        """List available commands."""
        return (
            "<b>Hermes Telegram Commands</b>\n\n"
            "<code>/help</code> — This message\n"
            "<code>/status</code> — System health overview\n"
            "<code>/agents</code> — Fleet agent health\n"
            "<code>/held</code> — Held spec ledger\n"
            "<code>/snap</code> — Latest snapshot status"
        )

    def _cmd_status(self) -> str:
        """System health overview."""
        lines = ["<b>System Status</b>\n"]

        # Latest snapshot date
        snap_dir = REPO_ROOT / "data" / "snapshots"
        dates = _list_snapshot_dates(snap_dir)
        if dates:
            lines.append(f"<b>Latest snapshot:</b> {dates[-1]}")
        else:
            lines.append("<b>Latest snapshot:</b> no snapshots")

        # Git state
        state_file = REPO_ROOT / "artifacts" / "ops" / "knowledge_layer" / "latest_state.json"
        if state_file.exists():
            try:
                state = json.loads(state_file.read_text())
                git_info = state.get("git", {})
                head = git_info.get("head", "?")
                lines.append(f"<b>Git HEAD:</b> {head[:40]}")
            except (json.JSONDecodeError, OSError):
                pass

        # Ops supervisor severity (final_severity is canonical)
        ops_sup_dir = REPO_ROOT / "artifacts" / "ops_supervisor"
        if ops_sup_dir.exists():
            sup_files = sorted(ops_sup_dir.glob("*_supervisor.json"), reverse=True)
            if sup_files:
                try:
                    sup_data = json.loads(sup_files[0].read_text())
                    verdict = sup_data.get("final_severity") or sup_data.get("overall_verdict", "?")
                    lines.append(f"<b>Ops verdict:</b> {verdict}")
                except (json.JSONDecodeError, OSError):
                    pass

        # Fleet ops snapshot when available
        fleet_ops_dir = REPO_ROOT / "artifacts" / "fleet_ops"
        if fleet_ops_dir.exists():
            status_files = sorted(fleet_ops_dir.glob("*_status.json"), reverse=True)
            if status_files:
                try:
                    fleet = json.loads(status_files[0].read_text())
                    lines.append(f"<b>Fleet overall:</b> {fleet.get('overall', '?')}")
                    cron = fleet.get("crontab_verify") or {}
                    if cron.get("overall"):
                        lines.append(f"<b>Crontab:</b> {cron.get('overall')} " f"(fail={cron.get('fail_count', 0)})")
                except (json.JSONDecodeError, OSError):
                    pass

        return "\n".join(lines)

    def _cmd_agents(self) -> str:
        """Fleet agent health summary."""
        fleet_ops_dir = REPO_ROOT / "artifacts" / "fleet_ops"
        if fleet_ops_dir.exists():
            status_files = sorted(fleet_ops_dir.glob("*_status.json"), reverse=True)
            if status_files:
                try:
                    data = json.loads(status_files[0].read_text())
                    herald = data.get("herald") or {}
                    heartbeat = data.get("heartbeat") or {}
                    lines = [
                        f"<b>Fleet ops</b> ({data.get('as_of_date', '?')})",
                        f"Overall: {data.get('overall', '?')}",
                        f"Herald: {herald.get('verdict', '?')} (done={herald.get('herald_done')})",
                        f"Heartbeat: {heartbeat.get('verdict', '?')}",
                    ]
                    audit = data.get("completion_audit") or {}
                    if audit.get("exists"):
                        lines.append(f"Wiring audit: {audit.get('overall', '?')}")
                    cron = data.get("crontab_verify") or {}
                    if cron.get("overall"):
                        lines.append(f"Crontab: {cron.get('overall')} " f"(fail={cron.get('fail_count', 0)})")
                    gates = data.get("selfimprove_gates") or {}
                    if gates.get("message"):
                        lines.append(gates["message"][:240])
                    return "\n".join(lines)
                except (json.JSONDecodeError, OSError):
                    pass

        hb_dir = REPO_ROOT / "artifacts" / "heartbeat"
        if not hb_dir.exists():
            return "No fleet artifacts found. Run: python3 tools/fleet_ops_status.py --write"

        receipt_files = sorted(hb_dir.glob("*_receipt.md"), reverse=True)
        if not receipt_files:
            return "No fleet receipt found. Run: python3 tools/agent_heartbeat_checks.py"

        try:
            content = receipt_files[0].read_text()
            lines_out = content.split("\n")[:30]
            summary = "\n".join(lines_out)
            if len(content.split("\n")) > 30:
                summary += "\n...(truncated)"
            return summary if summary.strip() else "Fleet receipt empty."
        except OSError:
            return "Failed to read fleet receipt."

    def _cmd_held(self) -> str:
        """Held spec ledger."""
        held_file = REPO_ROOT / "artifacts" / "ops" / "held_spec_ledger" / "latest.json"
        if not held_file.exists():
            return "No held spec ledger found."

        try:
            data = json.loads(held_file.read_text())
            items = data.get("held_items", [])
            if not items:
                return "No held specs."

            lines = ["<b>Held Specs:</b>\n"]
            for item in items[:10]:  # Limit to 10 items
                spec_id = item.get("spec_id", "?")
                status = item.get("status", "?")
                lines.append(f"• <code>{spec_id}</code> — {status}")

            if len(items) > 10:
                lines.append(f"...and {len(items) - 10} more")

            return "\n".join(lines)
        except (json.JSONDecodeError, OSError):
            return "Failed to read held spec ledger."

    def _cmd_snap(self) -> str:
        """Latest snapshot pipeline status."""
        snap_dir = REPO_ROOT / "data" / "snapshots"
        if not snap_dir.exists():
            return "No snapshots directory found."

        dates = _list_snapshot_dates(snap_dir)
        if not dates:
            return "No snapshots found."

        latest_date = dates[-1]
        manifest_file = snap_dir / latest_date / "run_manifest.json"
        if not manifest_file.exists():
            return f"Snapshot {latest_date}: no run_manifest.json"

        try:
            manifest = json.loads(manifest_file.read_text())
            status = manifest.get("overall_status", "?")
            modules = manifest.get("modules", {})

            lines = [f"<b>Snapshot {latest_date}</b>", f"Status: <b>{status}</b>\n"]

            if modules:
                for module, result in sorted(modules.items())[:10]:
                    mod_status = result.get("status", "?")
                    emoji = "✓" if mod_status == "PASS" else "✗"
                    lines.append(f"{emoji} {module}: {mod_status}")

            return "\n".join(lines)
        except (json.JSONDecodeError, OSError):
            return f"Failed to read snapshot {latest_date} manifest."

    # -----------------------------------------------------------------------
    # Main polling loop
    # -----------------------------------------------------------------------

    def run(self, once: bool = False) -> None:
        """Start polling loop. If once=True, process one batch and exit."""
        logger.info("Telegram command handler starting (dry_run=%s)", self.dry_run)

        while True:
            try:
                updates = self._get_updates()
                if not updates:
                    if once:
                        logger.info("No updates; exiting (--once)")
                        break
                    continue

                for update in updates:
                    if not self._is_authorized(update):
                        logger.debug(
                            "Unauthorized update from chat %s", update.get("message", {}).get("chat", {}).get("id")
                        )
                        continue

                    message = update.get("message", {})
                    text = message.get("text", "").strip()
                    if text:
                        logger.info("Command from chat: %s", text)
                        reply = self._dispatch_command(text)
                        self._send_reply(reply)

                if once:
                    logger.info("Processed one batch; exiting (--once)")
                    break
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except Exception as exc:
                logger.exception("Polling error: %s", exc)
                if once:
                    break
                time.sleep(5)  # backoff on error


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Hermes Telegram command handler daemon")
    parser.add_argument("--dry-run", action="store_true", help="Log replies without sending")
    parser.add_argument("--once", action="store_true", help="Process one batch and exit")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    )

    # Load environment
    _load_env()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

    if not token or not chat_id:
        logger.error("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set in .env")
        return 1

    # Check dry-run env
    dry_run = args.dry_run or os.environ.get("TELEGRAM_HANDLER_DRY_RUN", "0").lower() in ("1", "true")

    # Check PID file (unless --dry-run)
    if not dry_run and not args.once:
        if not _check_and_write_pid():
            return 1

    try:
        handler = TelegramCommandHandler(token, chat_id, dry_run=dry_run)
        handler.run(once=args.once)
        return 0
    finally:
        if not dry_run and not args.once:
            _remove_pid_file()


if __name__ == "__main__":
    sys.exit(main())
