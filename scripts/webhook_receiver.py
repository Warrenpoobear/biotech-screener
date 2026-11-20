#!/usr/bin/env python
"""Webhook receiver for FDA and ClinicalTrials.gov alerts.

Runs a lightweight HTTP server that receives webhook notifications from:
- FDA API alerts (drug approvals, PDUFA dates)
- ClinicalTrials.gov (trial status changes, new enrollments)

When a webhook fires, it writes the event to a JSONL log and optionally
triggers the catalyst pipeline for re-analysis.

Usage:
    python scripts/webhook_receiver.py --port 9099

Configure with a reverse proxy (ngrok, cloudflare tunnel) to receive
external webhooks, or use locally with curl for testing:

    curl -X POST http://localhost:9099/webhook/fda \
      -H "Content-Type: application/json" \
      -d '{"event":"approval","drug":"Drug X","company":"MRNA","date":"2024-06-01"}'

    curl -X POST http://localhost:9099/webhook/clinicaltrials \
      -H "Content-Type: application/json" \
      -d '{"event":"status_change","nct_id":"NCT04470427","new_status":"Recruiting"}'

Integration with Hermes:
    The webhook receiver writes events to output/webhook_events.jsonl.
    A cron job can poll this file and trigger agent analysis when new
    events arrive. See scripts/check_webhook_events.py for the poller.

Alternatively, configure Hermes webhook subscriptions directly:
    hermes webhook add --url http://your-tunnel:9099/webhook/fda --event fda
    hermes webhook add --url http://your-tunnel:9099/webhook/ctgov --event clinicaltrials
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

PROJECT_DIR = Path(os.environ.get("BIOTECH_PROJECT_DIR", Path(__file__).resolve().parent.parent))
EVENTS_FILE = PROJECT_DIR / "output" / "webhook_events.jsonl"
WEBHOOK_SECRET_ENV = "WEBHOOK_SHARED_SECRET"  # pragma: allowlist secret
WEBHOOK_SECRET_HEADER = "X-Webhook-Secret"  # pragma: allowlist secret
DEFAULT_BIND_HOST = "127.0.0.1"


def webhook_secret() -> str:
    """Shared secret from the environment. Empty means the receiver must reject."""
    return os.environ.get(WEBHOOK_SECRET_ENV, "").strip()


def webhook_authorized(header_value: str | None) -> bool:
    """True only when the env secret is set and the header matches it.

    An unset secret is a rejection, not an open receiver. Comparison is
    constant-time and does not treat a missing header as a match.
    """
    secret = webhook_secret()
    if not secret or header_value is None:
        return False
    return hmac.compare_digest(header_value, secret)


class WebhookHandler(BaseHTTPRequestHandler):
    """HTTP handler for FDA and ClinicalTrials.gov webhooks."""

    def do_POST(self):
        """Handle incoming webhook POST requests."""
        path = urlparse(self.path).path
        if not webhook_authorized(self.headers.get(WEBHOOK_SECRET_HEADER)):
            if not webhook_secret():
                self.send_error(401, "WEBHOOK_SHARED_SECRET is unset")
            else:
                self.send_error(401, "Invalid webhook secret")
            return
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length else ""

        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        # Route based on path
        if path == "/webhook/fda":
            event = self._process_fda(payload)
        elif path == "/webhook/clinicaltrials":
            event = self._process_ctgov(payload)
        elif path == "/webhook/generic":
            event = self._process_generic(payload)
        else:
            self.send_error(404, f"Unknown webhook path: {path}")
            return

        # Log the event
        self._log_event(event)

        # Send response
        response = json.dumps({"status": "received", "event_id": event["event_id"]})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(response.encode())

        # Print to stdout for cron delivery
        print(f"📡 Webhook received: {event['source']} → {event['event_type']}")
        if event.get("ticker"):
            print(f"   Ticker: {event['ticker']}")
        if event.get("description"):
            print(f"   {event['description']}")

    def do_GET(self):
        """Health check endpoint."""
        path = urlparse(self.path).path
        if path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "status": "ok",
                        "events_file": str(EVENTS_FILE),
                        "events_logged": self._count_events(),
                    }
                ).encode()
            )
        else:
            self.send_error(404, "Not found. Use POST /webhook/fda or /webhook/clinicaltrials")

    def _process_fda(self, payload: dict) -> dict:
        """Process FDA webhook payload."""
        event_id = f"fda_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{payload.get('drug', 'unknown')}"
        return {
            "event_id": event_id,
            "source": "FDA",
            "event_type": payload.get("event", "unknown"),
            "drug": payload.get("drug", ""),
            "ticker": payload.get("company", "").upper() if payload.get("company") else None,
            "date": payload.get("date", datetime.utcnow().strftime("%Y-%m-%d")),
            "description": payload.get(
                "description", f"FDA event: {payload.get('event', 'unknown')} — {payload.get('drug', 'N/A')}"
            ),
            "raw": payload,
            "received_at": datetime.utcnow().isoformat() + "Z",
        }

    def _process_ctgov(self, payload: dict) -> dict:
        """Process ClinicalTrials.gov webhook payload."""
        nct_id = payload.get("nct_id", "")
        event_id = f"ctgov_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{nct_id}"
        return {
            "event_id": event_id,
            "source": "ClinicalTrials.gov",
            "event_type": payload.get("event", "unknown"),
            "nct_id": nct_id,
            "ticker": payload.get("ticker", "").upper() if payload.get("ticker") else None,
            "new_status": payload.get("new_status", ""),
            "old_status": payload.get("old_status", ""),
            "description": payload.get("description", f"CT.gov event: {payload.get('event', 'unknown')} — {nct_id}"),
            "raw": payload,
            "received_at": datetime.utcnow().isoformat() + "Z",
        }

    def _process_generic(self, payload: dict) -> dict:
        """Process generic webhook payload."""
        event_id = f"generic_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        return {
            "event_id": event_id,
            "source": payload.get("source", "unknown"),
            "event_type": payload.get("event", "unknown"),
            "ticker": payload.get("ticker", "").upper() if payload.get("ticker") else None,
            "description": payload.get("description", "Generic webhook event"),
            "raw": payload,
            "received_at": datetime.utcnow().isoformat() + "Z",
        }

    def _log_event(self, event: dict) -> None:
        """Append event to JSONL log file."""
        EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(EVENTS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def _count_events(self) -> int:
        """Count logged events."""
        if not EVENTS_FILE.exists():
            return 0
        with open(EVENTS_FILE, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)

    def log_message(self, format, *args):
        """Suppress default logging."""


def main():
    parser = argparse.ArgumentParser(description="Biotech webhook receiver")
    parser.add_argument("--port", type=int, default=9099, help="Port to listen on")
    parser.add_argument(
        "--host",
        default=DEFAULT_BIND_HOST,
        help="Host to bind (default 127.0.0.1; put a reverse proxy in front for external delivery)",
    )
    args = parser.parse_args()

    if not webhook_secret():
        print(
            f"{WEBHOOK_SECRET_ENV} is unset; refusing to start",
            file=sys.stderr,
        )
        return 2

    server = HTTPServer((args.host, args.port), WebhookHandler)
    print("🧬 Biotech Webhook Receiver")
    print(f"   Listening: http://{args.host}:{args.port}")
    print("   FDA:       POST /webhook/fda")
    print("   CT.gov:    POST /webhook/clinicaltrials")
    print("   Generic:   POST /webhook/generic")
    print("   Health:    GET  /health")
    print(f"   Events:    {EVENTS_FILE}")
    print("   Press Ctrl+C to stop")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Stopped")
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
