"""
Send Alert — tools/send_alert.py

Posts a notification to a Slack incoming webhook (or other HTTP endpoint)
when a production health degradation is detected.

Reads output/live_performance_summary.json and output/health_packets/health_*.json
for context. Exits 0 on success or when no webhook is configured.

Usage:
    python3 tools/send_alert.py --level FAIL --date 2026-03-05 --webhook "$SLACK_WEBHOOK_URL"
    python3 tools/send_alert.py --level WARN --date 2026-03-05  # no webhook → silent exit 0
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_ROOT = PROJECT_ROOT / "output"
HEALTH_PACKETS_DIR = OUTPUT_ROOT / "health_packets"
LIVE_PERFORMANCE_SUMMARY = OUTPUT_ROOT / "live_performance_summary.json"

LEVEL_EMOJI = {
    "FAIL": ":red_circle:",
    "WARN": ":large_yellow_circle:",
    "OK": ":large_green_circle:",
}
LEVEL_COLOR = {
    "FAIL": "#d73a49",
    "WARN": "#e36209",
    "OK": "#22863a",
}


def _load_json(path: Path) -> Dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _find_health_packet(date_str: str) -> Dict:
    """Find health packet JSON for a given date."""
    path = HEALTH_PACKETS_DIR / f"health_{date_str}.json"
    return _load_json(path)


def _extract_gate_summary(packet: Dict) -> Dict:
    """Extract gate pass/warn/fail counts from health packet."""
    gates = packet.get("gates", {})
    return {
        "pass": gates.get("pass_count", 0),
        "warn": gates.get("warn_count", 0),
        "fail": gates.get("fail_count", 0),
        "fail_names": gates.get("fail_names", []),
        "warn_names": gates.get("warn_names", []),
    }


def _extract_action_items(packet: Dict) -> List[str]:
    """Extract action item descriptions from health packet."""
    items = packet.get("action_items", [])
    return [f"[{i.get('severity','?')}] {i.get('type','?')}: {i.get('detail','')}"
            for i in items]


def _extract_live_perf(summary: Dict) -> Optional[str]:
    """Extract 4w mean net return from live performance summary."""
    last_4w = summary.get("last_4w", {})
    mnr = last_4w.get("mean_net_return")
    n = last_4w.get("n_dates", 0)
    if mnr is None:
        return None
    sign = "+" if mnr >= 0 else ""
    return f"{sign}{mnr:.4f} ({n} dates)"


def build_payload(
    level: str,
    date_str: str,
    packet: Dict,
    live_summary: Dict,
) -> Dict:
    """Build Slack webhook JSON payload.

    Returns a dict ready to POST as JSON.
    """
    level = level.upper()
    emoji = LEVEL_EMOJI.get(level, ":white_circle:")
    color = LEVEL_COLOR.get(level, "#888888")

    gate_summary = _extract_gate_summary(packet)
    action_items = _extract_action_items(packet)
    live_perf = _extract_live_perf(live_summary)
    rollback_rec = packet.get("ruleset_health", {}).get("recommend_rollback", False)
    prov = packet.get("provenance", {})
    ruleset_id = prov.get("ruleset_id", "unknown")

    title = f"{emoji} Biotech Screener {level} — {date_str}"
    summary_text = f"Ruleset: `{ruleset_id}` | Gates: {gate_summary['pass']}P {gate_summary['warn']}W {gate_summary['fail']}F"
    if rollback_rec:
        summary_text += " | :rotating_light: ROLLBACK RECOMMENDED"

    fields = []
    if gate_summary["fail_names"]:
        fields.append({"title": "FAIL gates", "value": ", ".join(gate_summary["fail_names"]), "short": False})
    if gate_summary["warn_names"]:
        fields.append({"title": "WARN gates", "value": ", ".join(gate_summary["warn_names"]), "short": False})
    if action_items:
        fields.append({"title": "Action items", "value": "\n".join(action_items[:5]), "short": False})
    if live_perf:
        fields.append({"title": "4w net return", "value": live_perf, "short": True})

    attachment = {
        "color": color,
        "title": title,
        "text": summary_text,
        "fields": fields,
        "footer": "biotech-screener",
        "ts": int(datetime.utcnow().timestamp()),
    }

    return {"attachments": [attachment]}


def post_webhook(payload: Dict, webhook_url: str) -> None:
    """POST JSON payload to webhook URL. Raises on HTTP error."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode("utf-8")
        if resp.status not in (200, 204):
            raise RuntimeError(f"Webhook returned HTTP {resp.status}: {body}")


def send_alert(
    level: str,
    date_str: str,
    webhook_url: Optional[str],
    *,
    dry_run: bool = False,
) -> None:
    """Main alert sending logic.

    Exits 0 silently if no webhook URL is configured (optional feature).
    """
    if not webhook_url:
        print("[send_alert] No webhook URL configured — nothing to do.", file=sys.stderr)
        return

    packet = _find_health_packet(date_str)
    live_summary = _load_json(LIVE_PERFORMANCE_SUMMARY)

    payload = build_payload(level, date_str, packet, live_summary)

    if dry_run:
        print(f"[send_alert] DRY RUN — would POST to webhook:")
        print(json.dumps(payload, indent=2))
        return

    try:
        post_webhook(payload, webhook_url)
        print(f"[send_alert] Alert posted: {level} for {date_str}")
    except Exception as exc:
        # Non-fatal: alert failure should never block the CI pipeline
        print(f"[send_alert] WARNING: webhook POST failed: {exc}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a health degradation alert to a Slack webhook."
    )
    parser.add_argument("--level", required=True, choices=["FAIL", "WARN", "OK"],
                        help="Alert severity level")
    parser.add_argument("--date", required=True, metavar="YYYY-MM-DD",
                        help="Production run date")
    parser.add_argument("--webhook", default=None, metavar="URL",
                        help="Slack incoming webhook URL (omit to skip)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print payload without POSTing")
    args = parser.parse_args()

    send_alert(
        level=args.level,
        date_str=args.date,
        webhook_url=args.webhook,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
