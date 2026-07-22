"""Unit tests for tools.build_intraday_mover_watch (Spec 063)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import pytest

from common.realtime_quote_client import HealthStatus, QuoteRecord
from tools.build_intraday_mover_watch import (
    THRESHOLDS,
    build_daily_digest,
    build_intraday_mover_watch,
    classify_intraday_alerts,
    compute_intraday_metrics,
    compute_severity,
    derive_potential_drivers,
    format_daily_digest_email,
    format_immediate_alert,
    lookup_same_day_news,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeQuoteClient:
    def __init__(self, quotes: Dict[str, QuoteRecord], mode: str = "live"):
        self._quotes = quotes
        self._mode = mode

    def get_quotes(self, tickers: Iterable[str]) -> Dict[str, QuoteRecord]:
        return {t: self._quotes[t] for t in tickers if t in self._quotes}

    def health(self) -> HealthStatus:
        return HealthStatus(ok=(self._mode == "live"), mode=self._mode, detail="fake")


def _mkq(ticker: str, last: float, prev: float, *, vol: int = 100_000, avg20: int = 40_000):
    return QuoteRecord(
        ticker=ticker,
        last=last,
        prev_close=prev,
        open=prev,
        high=max(last, prev),
        low=min(last, prev),
        volume=vol,
        avg_volume_20d=avg20,
        quote_ts="2026-04-17T14:30:00Z",
        market_status="open",
        source="massive",
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_compute_intraday_metrics_basic():
    stock = _mkq("SRPT", last=107.5, prev=100.0)
    xbi = _mkq("XBI", last=99.0, prev=100.0)
    row = compute_intraday_metrics(stock, xbi, {"tier_dev": "A", "catalyst_days": "11"})
    assert row["stock_abs_move_pct"] == 7.5
    assert row["rel_move_vs_xbi_pct"] == pytest.approx(8.5, abs=0.01)
    assert row["rvol"] == 2.5
    assert row["tier"] == "A"


def test_compute_intraday_metrics_without_xbi():
    stock = _mkq("KROS", last=91.0, prev=100.0)
    row = compute_intraday_metrics(stock, None, {})
    assert row["stock_abs_move_pct"] == -9.0
    assert row["rel_move_vs_xbi_pct"] is None


def test_compute_intraday_metrics_rvol_none_when_avg_missing():
    stock = QuoteRecord(
        ticker="X",
        last=11.0,
        prev_close=10.0,
        open=10.0,
        high=11.0,
        low=10.0,
        volume=100_000,
        avg_volume_20d=None,
        quote_ts="2026-04-17T14:30:00Z",
        market_status="open",
        source="massive",
    )
    row = compute_intraday_metrics(stock, None, {})
    assert row["rvol"] is None


def test_compute_intraday_metrics_skips_invalid_prev_close():
    stock = QuoteRecord(
        ticker="X",
        last=11.0,
        prev_close=0.0,
        open=0.0,
        high=11.0,
        low=0.0,
        volume=0,
        avg_volume_20d=0,
        quote_ts="2026-04-17T14:30:00Z",
        market_status="open",
        source="massive",
    )
    row = compute_intraday_metrics(stock, None, {})
    assert row.get("skip_reason") == "invalid_prev_close"


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def test_classify_abs_move_high_up():
    codes = classify_intraday_alerts({"stock_abs_move_pct": 12.0, "rel_move_vs_xbi_pct": 0, "rvol": None, "last": 50.0})
    assert "INTRADAY_ABS_MOVE_UP_HIGH" in codes


def test_classify_rel_move_high_up():
    codes = classify_intraday_alerts(
        {"stock_abs_move_pct": 3.0, "rel_move_vs_xbi_pct": 7.5, "rvol": None, "last": 50.0}
    )
    assert "INTRADAY_REL_MOVE_UP_HIGH" in codes


def test_classify_no_alert_when_small_move():
    codes = classify_intraday_alerts({"stock_abs_move_pct": 1.0, "rel_move_vs_xbi_pct": 0.5, "rvol": 1.0, "last": 50.0})
    assert codes == []


def test_classify_skipped_below_min_price():
    codes = classify_intraday_alerts(
        {"stock_abs_move_pct": 20.0, "rel_move_vs_xbi_pct": 20.0, "rvol": 5.0, "last": 0.5}
    )
    assert codes == []


def test_classify_rvol_only_when_volume_available():
    codes_none = classify_intraday_alerts({"stock_abs_move_pct": 0.1, "rvol": None, "last": 50.0})
    assert "INTRADAY_RVOL_SPIKE" not in codes_none
    codes_spike = classify_intraday_alerts(
        {"stock_abs_move_pct": 0.1, "rel_move_vs_xbi_pct": 0.1, "rvol": 3.0, "last": 50.0}
    )
    assert "INTRADAY_RVOL_SPIKE" in codes_spike


# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------
def test_severity_high_from_any_high_code():
    row = {"trigger_codes": ["INTRADAY_ABS_MOVE_UP_HIGH"], "news_status": "NONE"}
    assert compute_severity(row) == "HIGH"


def test_severity_high_on_medium_plus_official_news():
    row = {"trigger_codes": ["INTRADAY_ABS_MOVE_UP_MEDIUM"], "news_status": "OFFICIAL"}
    assert compute_severity(row) == "HIGH"


def test_severity_medium_when_medium_alone():
    row = {"trigger_codes": ["INTRADAY_ABS_MOVE_DOWN_MEDIUM"], "news_status": "NONE"}
    assert compute_severity(row) == "MEDIUM"


def test_severity_low_when_no_tiered_code():
    row = {"trigger_codes": [], "news_status": "NONE"}
    assert compute_severity(row) == "LOW"


# ---------------------------------------------------------------------------
# News lookup: Herald-first, Grok never official
# ---------------------------------------------------------------------------
def test_news_lookup_prefers_herald_classified(tmp_path: Path):
    herald_c = tmp_path / "herald" / "classified"
    herald_r = tmp_path / "herald" / "raw"
    herald_c.mkdir(parents=True)
    herald_r.mkdir(parents=True)

    (herald_c / "2026-04-17.json").write_text(
        json.dumps({"releases": [{"ticker": "SRPT", "headline": "Sarepta topline", "source_type": "company_ir"}]})
    )

    news = lookup_same_day_news(
        "SRPT",
        "2026-04-17",
        herald_classified_dir=herald_c,
        herald_raw_dir=herald_r,
    )
    assert news["news_status"] == "OFFICIAL"
    assert news["headline"] == "Sarepta topline"
    assert news["source_type"] == "company_ir"
    assert news["source_rank"] == 1


def test_news_lookup_no_herald_record_returns_none(tmp_path: Path):
    # Grok/xAI ingestion was removed (809da0ce): only herald-backed news is
    # recognized. A ticker with no herald classified/raw record yields NONE —
    # there is no longer a non-herald "SUPPORTING" fallback.
    herald_c = tmp_path / "herald" / "classified"
    herald_r = tmp_path / "herald" / "raw"
    herald_c.mkdir(parents=True)
    herald_r.mkdir(parents=True)

    news = lookup_same_day_news(
        "KROS",
        "2026-04-17",
        herald_classified_dir=herald_c,
        herald_raw_dir=herald_r,
    )
    assert news["news_status"] == "NONE"
    assert news["news_status"] != "OFFICIAL"


def test_news_lookup_returns_none_when_empty(tmp_path: Path):
    news = lookup_same_day_news(
        "XYZ",
        "2026-04-17",
        herald_classified_dir=tmp_path / "a",
        herald_raw_dir=tmp_path / "b",
    )
    assert news["news_status"] == "NONE"


# ---------------------------------------------------------------------------
# Builder end-to-end (scaffolded fixtures)
# ---------------------------------------------------------------------------
def _setup_min_repo(tmp_path: Path, tickers: list) -> Path:
    snap_dir = tmp_path / "snapshots" / "2026-04-17"
    snap_dir.mkdir(parents=True)
    import csv as _csv

    with open(snap_dir / "rankings.csv", "w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(
            f,
            fieldnames=["ticker", "tier_dev", "actionable_rank", "catalyst_days", "is_hard_catalyst"],
        )
        w.writeheader()
        for i, t in enumerate(tickers):
            w.writerow(
                {
                    "ticker": t,
                    "tier_dev": "A",
                    "actionable_rank": str(i + 1),
                    "catalyst_days": "15",
                    "is_hard_catalyst": "1",
                }
            )
    with open(snap_dir / "review_queue.csv", "w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["ticker"])
        w.writeheader()
        for t in tickers:
            w.writerow({"ticker": t})
    return snap_dir.parent.parent


def test_builder_no_data_when_client_empty(tmp_path: Path):
    _setup_min_repo(tmp_path, ["SRPT"])
    client = FakeQuoteClient(quotes={}, mode="no_credentials")
    result = build_intraday_mover_watch(
        "2026-04-17T14:30:00Z",
        quote_client=client,
        snapshots_dir=tmp_path / "snapshots",
        artifacts_dir=tmp_path / "artifacts",
    )
    assert result["status"] == "NO_DATA"
    assert result["n_triggered"] == 0


def test_builder_triggers_and_writes_artifact(tmp_path: Path):
    _setup_min_repo(tmp_path, ["SRPT"])
    quotes = {
        "SRPT": _mkq("SRPT", last=107.5, prev=100.0),
        "XBI": _mkq("XBI", last=99.0, prev=100.0),
    }
    client = FakeQuoteClient(quotes=quotes, mode="live")
    result = build_intraday_mover_watch(
        "2026-04-17T14:30:00Z",
        quote_client=client,
        snapshots_dir=tmp_path / "snapshots",
        artifacts_dir=tmp_path / "artifacts",
    )
    assert result["status"] == "OK"
    assert result["n_triggered"] == 1
    row = result["rows"][0]
    assert row["ticker"] == "SRPT"
    assert row["severity"] in ("HIGH", "MEDIUM")
    assert "INTRADAY_ABS_MOVE_UP_MEDIUM" in row["trigger_codes"] or "INTRADAY_ABS_MOVE_UP_HIGH" in row["trigger_codes"]
    assert result["xbi_abs_move_pct"] == -1.0
    # Artifact was written
    out = tmp_path / "artifacts" / "intraday_mover_watch" / "2026-04-17T14-30-00Z_poll.json"
    assert out.exists()


def test_builder_stamps_feed_provenance(tmp_path: Path):
    _setup_min_repo(tmp_path, ["SRPT"])
    quotes = {
        "SRPT": _mkq("SRPT", last=107.5, prev=100.0),
        "XBI": _mkq("XBI", last=99.0, prev=100.0),
    }
    client = FakeQuoteClient(quotes=quotes, mode="live")
    result = build_intraday_mover_watch(
        "2026-04-17T14:30:00Z",
        quote_client=client,
        snapshots_dir=tmp_path / "snapshots",
        artifacts_dir=tmp_path / "artifacts",
    )
    # Artifact header carries feed provenance
    assert result["feed_source"] == "massive"  # FakeQuoteClient's mkq uses source="massive"
    assert result["xbi_source"] == "massive"
    assert "feed_detail" in result
    # Each row carries source + xbi_source for audit
    row = result["rows"][0]
    assert row["source"] == "massive"
    assert row["xbi_source"] == "massive"


def test_builder_no_data_artifact_has_feed_fields(tmp_path: Path):
    _setup_min_repo(tmp_path, ["SRPT"])
    client = FakeQuoteClient(quotes={}, mode="no_credentials")
    result = build_intraday_mover_watch(
        "2026-04-17T14:30:00Z",
        quote_client=client,
        snapshots_dir=tmp_path / "snapshots",
        artifacts_dir=tmp_path / "artifacts",
    )
    assert result["status"] == "NO_DATA"
    assert result["feed_source"] == "none"
    assert result["feed_detail"] == "fake"


def test_builder_xbi_missing_keeps_abs_alerts(tmp_path: Path):
    _setup_min_repo(tmp_path, ["SRPT"])
    quotes = {"SRPT": _mkq("SRPT", last=112.0, prev=100.0)}
    client = FakeQuoteClient(quotes=quotes, mode="live")
    result = build_intraday_mover_watch(
        "2026-04-17T14:30:00Z",
        quote_client=client,
        snapshots_dir=tmp_path / "snapshots",
        artifacts_dir=tmp_path / "artifacts",
    )
    assert result["n_triggered"] == 1
    row = result["rows"][0]
    assert "INTRADAY_ABS_MOVE_UP_HIGH" in row["trigger_codes"]
    assert row["rel_move_vs_xbi_pct"] is None


# ---------------------------------------------------------------------------
# Digest
# ---------------------------------------------------------------------------
def test_digest_rollup_counts(tmp_path: Path):
    out_dir = tmp_path / "artifacts" / "intraday_mover_watch"
    out_dir.mkdir(parents=True)
    poll = {
        "rows": [
            {
                "ticker": "SRPT",
                "stock_abs_move_pct": 12.0,
                "rel_move_vs_xbi_pct": 9.0,
                "severity": "HIGH",
                "news_status": "OFFICIAL",
            },
            {
                "ticker": "KROS",
                "stock_abs_move_pct": -6.0,
                "rel_move_vs_xbi_pct": -4.5,
                "severity": "MEDIUM",
                "news_status": "NONE",
            },
        ]
    }
    (out_dir / "2026-04-17T14-30-00Z_poll.json").write_text(json.dumps(poll))
    (out_dir / "2026-04-17T15-00-00Z_poll.json").write_text(json.dumps(poll))

    d = build_daily_digest("2026-04-17", tmp_path / "artifacts")
    assert d["polls_seen"] == 2
    assert d["unique_movers"] == 2
    assert d["severity_counts"]["HIGH"] == 1
    assert d["severity_counts"]["MEDIUM"] == 1
    assert len(d["movers_with_official_news"]) == 1
    assert (tmp_path / "artifacts" / "intraday_mover_watch" / "2026-04-17_digest.md").exists()


# ---------------------------------------------------------------------------
# Thresholds sanity
# ---------------------------------------------------------------------------
def test_thresholds_shape():
    # Asymmetric threshold names must all exist with expected polarity
    assert THRESHOLDS["abs_move_high_up"] > THRESHOLDS["abs_move_medium_up"] > 0
    assert THRESHOLDS["abs_move_high_down"] < THRESHOLDS["abs_move_medium_down"] < 0
    assert THRESHOLDS["rel_move_high_up"] > THRESHOLDS["rel_move_medium_up"] > 0
    assert THRESHOLDS["rel_move_high_down"] < THRESHOLDS["rel_move_medium_down"] < 0


# ---------------------------------------------------------------------------
# Phase 2 — email formatting & send gates
# ---------------------------------------------------------------------------
def _high_row(**overrides):
    base = {
        "ticker": "CGON",
        "tier": "A",
        "actionable_rank": 3,
        "catalyst_days": 11,
        "last": 73.42,
        "prev_close": 66.35,
        "rvol": 3.12,
        "stock_abs_move_pct": 10.66,
        "rel_move_vs_xbi_pct": 8.33,
        "quote_ts": "2026-04-17T20:30:00Z",
        "severity": "HIGH",
        "trigger_codes": ["INTRADAY_ABS_MOVE_UP_HIGH", "INTRADAY_REL_MOVE_UP_HIGH"],
        "news_status": "NONE",
        "source_type": None,
        "headline": "",
        "source": "alpaca",
    }
    base.update(overrides)
    return base


def test_format_immediate_alert_no_news_label_is_honest():
    subj, body = format_immediate_alert(
        _high_row(),
        feed_source="alpaca",
        feed_detail="alpaca basic (15-min delayed REST snapshots)",
        artifact_path="/tmp/x.json",
    )
    # Subject must say no official news found
    assert "no official same-day source found" in subj
    # Body stamps feed provenance and news status explicitly
    assert "Feed source: alpaca" in body
    assert "News status: NONE" in body
    assert "no official same-day source found" in body
    assert "Read-only alert only" in body


def test_format_immediate_alert_official_news_label():
    subj, body = format_immediate_alert(
        _high_row(news_status="OFFICIAL", source_type="company_ir", headline="CGON topline"),
        feed_source="alpaca",
        feed_detail="alpaca basic (15-min delayed REST snapshots)",
        artifact_path="/tmp/x.json",
    )
    assert "official catalyst" in subj
    assert "company_ir" in subj
    assert "News status: OFFICIAL" in body
    assert "CGON topline" in body


def test_format_immediate_alert_grok_only_is_supporting_unverified():
    subj, body = format_immediate_alert(
        _high_row(news_status="SUPPORTING", source_type="grok", headline="chatter"),
        feed_source="alpaca",
        feed_detail="alpaca basic (15-min delayed REST snapshots)",
        artifact_path="/tmp/x.json",
    )
    # Grok/supporting must never be labeled as "official" in the outbound body
    assert "supporting context only" in subj
    assert "News status: SUPPORTING" in body
    assert "official catalyst" not in subj.lower() or "no official" in subj.lower()


def test_format_immediate_alert_handles_missing_xbi():
    subj, body = format_immediate_alert(
        _high_row(rel_move_vs_xbi_pct=None),
        feed_source="alpaca",
        feed_detail="alpaca basic",
        artifact_path="/tmp/x.json",
    )
    # No rel-vs-XBI in subject when XBI missing
    assert "vs XBI" not in subj
    assert "Relative vs XBI: N/A" in body


def test_format_daily_digest_email_stamps_feed_source():
    digest = {
        "as_of_date": "2026-04-17",
        "polls_seen": 7,
        "unique_movers": 3,
        "severity_counts": {"HIGH": 1, "MEDIUM": 2, "LOW": 0},
        "top_absolute_movers": [_high_row()],
        "top_relative_movers_vs_xbi": [_high_row()],
        "movers_with_official_news": [],
        "movers_without_official_news": [_high_row()],
    }
    subj, body = format_daily_digest_email(digest, feed_source="alpaca")
    assert "[Digest 2026-04-17]" in subj
    assert "1 HIGH" in subj
    assert "Feed source: alpaca" in body
    assert "no same-day official source" in body


def test_builder_send_email_skipped_when_not_live(tmp_path: Path, monkeypatch):
    """send_email=True must no-op when client mode != live, even if SMTP is configured."""
    _setup_min_repo(tmp_path, ["SRPT"])
    monkeypatch.setenv("SMTP_USER", "u@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "pw")
    monkeypatch.setenv("ALERT_EMAIL_TO", "to@example.com")

    # mode=dry_run must block the email path even when quotes are returned
    # and a HIGH-severity row would otherwise trigger a send.
    client = FakeQuoteClient(
        quotes={"SRPT": _mkq("SRPT", last=115.0, prev=100.0)},  # +15% → HIGH
        mode="dry_run",
    )

    import tools.build_intraday_mover_watch as mod

    called = []
    monkeypatch.setattr(mod, "send_email", lambda *a, **kw: called.append(a) or True)

    result = build_intraday_mover_watch(
        "2026-04-17T14:30:00Z",
        quote_client=client,
        snapshots_dir=tmp_path / "snapshots",
        artifacts_dir=tmp_path / "artifacts",
        send_email=True,
    )
    # Builder still produces the artifact, but the live-mode gate blocks the email
    assert result["status"] == "OK"
    assert called == []


def test_builder_send_email_skipped_when_smtp_unconfigured(tmp_path: Path, monkeypatch):
    """send_email=True with live quotes but no SMTP → logs warn, no send attempt."""
    _setup_min_repo(tmp_path, ["SRPT"])
    # No SMTP env vars
    for v in ("SMTP_USER", "SMTP_PASSWORD", "ALERT_EMAIL_TO", "ALERT_RECIPIENT"):
        monkeypatch.delenv(v, raising=False)

    client = FakeQuoteClient(
        quotes={"SRPT": _mkq("SRPT", 115.0, 100.0), "XBI": _mkq("XBI", 99.0, 100.0)},
        mode="live",
    )
    import tools.build_intraday_mover_watch as mod

    called = []
    monkeypatch.setattr(mod, "send_email", lambda *a, **kw: called.append(a))

    result = build_intraday_mover_watch(
        "2026-04-17T14:30:00Z",
        quote_client=client,
        snapshots_dir=tmp_path / "snapshots",
        artifacts_dir=tmp_path / "artifacts",
        send_email=True,
    )
    assert result["status"] == "OK"
    # With SMTP unconfigured, our guard runs first and send_email is never called
    assert called == []


def test_builder_send_email_fires_for_high_when_all_gates_pass(tmp_path: Path, monkeypatch):
    _setup_min_repo(tmp_path, ["CGON"])
    monkeypatch.setenv("SMTP_USER", "u@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "pw")
    monkeypatch.setenv("ALERT_EMAIL_TO", "to@example.com")

    client = FakeQuoteClient(
        quotes={
            # +10.66% with XBI flat → abs_HIGH + rel_HIGH → severity HIGH
            "CGON": _mkq("CGON", last=110.66, prev=100.0),
            "XBI": _mkq("XBI", last=100.0, prev=100.0),
        },
        mode="live",
    )
    import tools.build_intraday_mover_watch as mod

    captured = []

    def _fake_send(subject, body, **kw):
        captured.append((subject, body))
        return True

    monkeypatch.setattr(mod, "send_email", _fake_send)

    result = build_intraday_mover_watch(
        "2026-04-17T14:30:00Z",
        quote_client=client,
        snapshots_dir=tmp_path / "snapshots",
        artifacts_dir=tmp_path / "artifacts",
        send_email=True,
    )
    assert result["status"] == "OK"
    assert len(captured) == 1
    subject, body = captured[0]
    assert "[HIGH] CGON" in subject
    assert "Feed source: massive" in body  # _mkq uses source=massive
    assert "News status:" in body


def test_builder_persists_sent_alerts_and_suppresses_second_run(tmp_path: Path, monkeypatch):
    """First run sends; immediate second run finds state on disk and suppresses."""
    _setup_min_repo(tmp_path, ["CGON"])
    monkeypatch.setenv("SMTP_USER", "u@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "pw")
    monkeypatch.setenv("ALERT_EMAIL_TO", "to@example.com")

    quotes = {
        "CGON": _mkq("CGON", last=110.66, prev=100.0),
        "XBI": _mkq("XBI", last=100.0, prev=100.0),
    }

    import tools.build_intraday_mover_watch as mod

    captured = []
    monkeypatch.setattr(mod, "send_email", lambda s, b, **kw: captured.append((s, b)) or True)

    # First run: HIGH fires, email sent, dedupe state written
    client_1 = FakeQuoteClient(quotes=quotes, mode="live")
    build_intraday_mover_watch(
        "2026-04-17T14:30:00Z",
        quote_client=client_1,
        snapshots_dir=tmp_path / "snapshots",
        artifacts_dir=tmp_path / "artifacts",
        send_email=True,
    )
    assert len(captured) == 1
    state_path = tmp_path / "artifacts" / "intraday_mover_watch" / "sent_alerts.json"
    assert state_path.exists()

    # Second run 10 minutes later: same mover, same dedupe_key → suppressed
    client_2 = FakeQuoteClient(quotes=quotes, mode="live")
    build_intraday_mover_watch(
        "2026-04-17T14:40:00Z",
        quote_client=client_2,
        snapshots_dir=tmp_path / "snapshots",
        artifacts_dir=tmp_path / "artifacts",
        send_email=True,
    )
    assert len(captured) == 1  # still 1 — suppressed on second run


def test_builder_widened_move_resends_across_invocations(tmp_path: Path, monkeypatch):
    """If the move widens by ≥3pp on a subsequent run, re-send."""
    _setup_min_repo(tmp_path, ["CGON"])
    monkeypatch.setenv("SMTP_USER", "u@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "pw")
    monkeypatch.setenv("ALERT_EMAIL_TO", "to@example.com")

    import tools.build_intraday_mover_watch as mod

    captured = []
    monkeypatch.setattr(mod, "send_email", lambda s, b, **kw: captured.append((s, b)) or True)

    # Run 1: +11% HIGH
    c1 = FakeQuoteClient(
        quotes={
            "CGON": _mkq("CGON", last=111.0, prev=100.0),
            "XBI": _mkq("XBI", last=100.0, prev=100.0),
        },
        mode="live",
    )
    build_intraday_mover_watch(
        "2026-04-17T14:30:00Z",
        quote_client=c1,
        snapshots_dir=tmp_path / "snapshots",
        artifacts_dir=tmp_path / "artifacts",
        send_email=True,
    )
    assert len(captured) == 1

    # Run 2: +15% HIGH — widened by +4pp, should re-send
    c2 = FakeQuoteClient(
        quotes={
            "CGON": _mkq("CGON", last=115.0, prev=100.0),
            "XBI": _mkq("XBI", last=100.0, prev=100.0),
        },
        mode="live",
    )
    build_intraday_mover_watch(
        "2026-04-17T14:45:00Z",
        quote_client=c2,
        snapshots_dir=tmp_path / "snapshots",
        artifacts_dir=tmp_path / "artifacts",
        send_email=True,
    )
    assert len(captured) == 2


def test_digest_send_email_stamps_feed_and_sends(tmp_path: Path, monkeypatch):
    # Seed one poll artifact
    out_dir = tmp_path / "artifacts" / "intraday_mover_watch"
    out_dir.mkdir(parents=True)
    poll = {
        "feed_source": "alpaca",
        "rows": [
            {
                "ticker": "CGON",
                "stock_abs_move_pct": 10.66,
                "rel_move_vs_xbi_pct": 8.33,
                "severity": "HIGH",
                "news_status": "NONE",
            }
        ],
    }
    (out_dir / "2026-04-17T14-30-00Z_poll.json").write_text(json.dumps(poll))

    monkeypatch.setenv("SMTP_USER", "u@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "pw")
    monkeypatch.setenv("ALERT_EMAIL_TO", "to@example.com")

    import tools.build_intraday_mover_watch as mod

    captured = []
    monkeypatch.setattr(mod, "send_email", lambda s, b, **kw: captured.append((s, b)) or True)

    d = build_daily_digest("2026-04-17", tmp_path / "artifacts", send_email_flag=True)
    assert d["feed_source"] == "alpaca"
    assert len(captured) == 1
    subject, body = captured[0]
    assert "Digest 2026-04-17" in subject
    assert "Feed source: alpaca" in body


# ---------------------------------------------------------------------------
# Potential drivers (heuristic, unconfirmed) — Spec 063 Phase 3 extension
# ---------------------------------------------------------------------------
def test_derive_drivers_news_none_is_honest():
    lines = derive_potential_drivers(
        {"stock_abs_move_pct": -12.0, "news_status": "NONE", "catalyst_days": 8, "market_cap_mm": 300.0}
    )
    news = next(ln for ln in lines if ln.startswith("- News:"))
    assert "No primary or secondary same-day catalyst identified" in news


def test_derive_drivers_news_official_names_source():
    lines = derive_potential_drivers(
        {
            "stock_abs_move_pct": 12.0,
            "news_status": "OFFICIAL",
            "source_type": "company_ir",
            "headline": "Topline hits primary endpoint",
            "catalyst_days": 0,
        }
    )
    news = next(ln for ln in lines if ln.startswith("- News:"))
    assert "Official same-day catalyst" in news
    assert "company_ir" in news
    assert "Topline hits primary endpoint" in news


def test_derive_drivers_small_cap_air_pocket_flagged():
    lines = derive_potential_drivers({"stock_abs_move_pct": -12.0, "news_status": "NONE", "market_cap_mm": 300.0})
    tech = next(ln for ln in lines if ln.startswith("- Technical/liquidity:"))
    assert "Small-cap" in tech
    assert "air pocket" in tech


def test_derive_drivers_large_cap_not_flagged_as_liquidity():
    lines = derive_potential_drivers({"stock_abs_move_pct": -12.0, "news_status": "NONE", "market_cap_mm": 5000.0})
    tech = next(ln for ln in lines if ln.startswith("- Technical/liquidity:"))
    assert "Large-cap" in tech
    assert "air pocket" not in tech


def test_derive_drivers_event_proximity_near_term():
    lines = derive_potential_drivers({"stock_abs_move_pct": -8.0, "catalyst_days": 3})
    prox = next(ln for ln in lines if ln.startswith("- Event proximity:"))
    assert "Near-term catalyst" in prox
    assert "positioning unwind" in prox


def test_derive_drivers_event_proximity_far_out():
    lines = derive_potential_drivers({"stock_abs_move_pct": -8.0, "catalyst_days": 60})
    prox = next(ln for ln in lines if ln.startswith("- Event proximity:"))
    assert "move likely unrelated to scheduled catalyst" in prox


def test_derive_drivers_no_options_data_says_unavailable():
    lines = derive_potential_drivers({"stock_abs_move_pct": -8.0})
    opt = next(ln for ln in lines if ln.startswith("- Options flow:"))
    assert "No options data available" in opt


def test_derive_drivers_elevated_puts_on_downside():
    lines = derive_potential_drivers({"stock_abs_move_pct": -12.0, "pre_event_put_call_ratio": 2.1})
    opt = next(ln for ln in lines if ln.startswith("- Options flow:"))
    assert "Elevated put activity" in opt
    assert "2.10" in opt


def test_derive_drivers_short_squeeze_vs_reinforcement_direction_aware():
    down = derive_potential_drivers({"stock_abs_move_pct": -10.0, "short_interest_pct": 35.0})
    up = derive_potential_drivers({"stock_abs_move_pct": +10.0, "short_interest_pct": 35.0})
    assert any("short reinforcement" in ln for ln in down)
    assert any("squeeze dynamics" in ln for ln in up)


def test_derive_drivers_sector_sympathy_when_rel_small():
    lines = derive_potential_drivers({"stock_abs_move_pct": -8.0, "rel_move_vs_xbi_pct": -0.5})
    other = next(ln for ln in lines if ln.startswith("- Other:"))
    assert "sector sympathy" in other


def test_derive_drivers_fixed_section_order():
    lines = derive_potential_drivers({"stock_abs_move_pct": -8.0, "catalyst_days": 4, "market_cap_mm": 300.0})
    prefixes = [ln.split(":", 1)[0] for ln in lines]
    assert prefixes == [
        "- News",
        "- Options flow",
        "- Technical/liquidity",
        "- Event proximity",
        "- Other",
    ]


def test_format_immediate_alert_embeds_drivers_block_between_news_and_interpretation():
    row = _high_row(news_status="NONE")
    row["market_cap_mm"] = 300.0
    row["catalyst_days"] = 3
    subj, body = format_immediate_alert(
        row,
        feed_source="alpaca",
        feed_detail="alpaca basic",
        artifact_path="/tmp/x.json",
    )
    assert "Potential drivers (heuristic, unconfirmed):" in body
    # Block appears after News status line and before Interpretation
    idx_news = body.index("News status:")
    idx_drivers = body.index("Potential drivers (heuristic, unconfirmed):")
    idx_interp = body.index("Interpretation:")
    assert idx_news < idx_drivers < idx_interp
    # At least one driver line uses heuristic language
    assert "Near-term catalyst" in body
    assert "Small-cap" in body


def test_format_immediate_alert_drivers_block_graceful_when_fields_missing():
    # Row with no market_cap, no options, no catalyst_days, no short interest
    subj, body = format_immediate_alert(
        _high_row(news_status="NONE"),
        feed_source="alpaca",
        feed_detail="alpaca basic",
        artifact_path="/tmp/x.json",
    )
    assert "Potential drivers (heuristic, unconfirmed):" in body
    assert "No primary or secondary same-day catalyst identified" in body
    assert "Market-cap data unavailable" in body
    assert "No options data available" in body


def test_compute_intraday_metrics_plumbs_ranking_fields():
    q = _mkq("ALT", last=90.0, prev=100.0)
    ranking_row = {
        "tier_dev": "A",
        "actionable_rank": "5",
        "catalyst_days": "8",
        "market_cap_mm": "380.0",
        "short_interest_pct": "22.5",
        "pre_event_put_call_ratio": "1.8",
        "opt_put_call_skew": "0.3",
        "priced_move_pct": "15.0",
    }
    row = compute_intraday_metrics(q, None, ranking_row)
    assert row["market_cap_mm"] == 380.0
    assert row["short_interest_pct"] == 22.5
    assert row["pre_event_put_call_ratio"] == 1.8
    assert row["opt_put_call_skew"] == 0.3
    assert row["priced_move_pct"] == 15.0


def test_compute_intraday_metrics_missing_fields_are_none_not_nan():
    q = _mkq("XYZ", last=90.0, prev=100.0)
    row = compute_intraday_metrics(q, None, {})
    # Must be None (JSON-safe), not NaN
    assert row["market_cap_mm"] is None
    assert row["short_interest_pct"] is None
    assert row["pre_event_put_call_ratio"] is None
