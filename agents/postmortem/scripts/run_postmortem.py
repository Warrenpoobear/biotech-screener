#!/usr/bin/env python3
"""
Postmortem capture script — run after HEARTBEAT detects resolved catalysts.
Reads pre-event snapshots, price history, and writes JSON + MD artifacts.
"""

import csv
import json
import os
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
SNAPS_DIR = os.path.join(REPO, "data/snapshots")
RESOL_DIR = os.path.join(SNAPS_DIR, "resolutions")
PRICE_CSV = os.path.join(REPO, "production_data/price_history.csv")
PM_DIR = os.path.join(REPO, "artifacts/postmortem")
MEM_DIR = os.path.join(REPO, "agents/postmortem/memory")
RULESET_ID = "9f1f4587"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ── helpers ───────────────────────────────────────────────────────────────────


def dated_snapshots():
    """Return sorted list of clean YYYY-MM-DD snapshot dirs only (no suffixes)."""
    return sorted(
        d
        for d in os.listdir(SNAPS_DIR)
        if DATE_RE.match(d)
        and os.path.isdir(os.path.join(SNAPS_DIR, d))
        and os.path.exists(os.path.join(SNAPS_DIR, d, "rankings.csv"))
    )


def load_snapshot(snap_date):
    path = os.path.join(SNAPS_DIR, snap_date, "rankings.csv")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        rows = {}
        for r in csv.DictReader(f):
            t = r.get("ticker", "").strip()
            if t:
                rows[t] = r
        return rows


def load_prices():
    """Returns {ticker: {date_str: close_float}}"""
    prices = {}
    with open(PRICE_CSV) as f:
        for row in csv.DictReader(f):
            t, d, c = row["ticker"], row["date"], row.get("close", "")
            if c:
                prices.setdefault(t, {})[d] = float(c)
    return prices


def compute_returns(ticker, event_date_str, prices, xbi_prices):
    tdates = sorted(prices.get(ticker, {}).keys())
    pre = [d for d in tdates if d <= event_date_str]
    if not pre:
        return {}
    t0d = pre[-1]
    t0c = prices[ticker][t0d]
    post = [d for d in tdates if d > event_date_str]

    result = {"pre_close_date": t0d, "pre_close": round(t0c, 4)}
    for n, key in [(1, "t1"), (3, "t3"), (5, "t5")]:
        if len(post) >= n:
            tn_d = post[n - 1]
            tn_c = prices[ticker].get(tn_d)
            if tn_c and t0c:
                ret = round((tn_c - t0c) / t0c, 4)
                result[f"return_{key}"] = ret
                result[f"{key}_date"] = tn_d
                xbi_t0 = xbi_prices.get(t0d)
                xbi_tn = xbi_prices.get(tn_d)
                if xbi_t0 and xbi_tn:
                    xbi_ret = round((xbi_tn - xbi_t0) / xbi_t0, 4)
                    result[f"excess_vs_xbi_{key}"] = round(ret - xbi_ret, 4)
                else:
                    result[f"excess_vs_xbi_{key}"] = None
            else:
                result[f"return_{key}"] = None
                result[f"excess_vs_xbi_{key}"] = None
        else:
            result[f"return_{key}"] = None
            result[f"excess_vs_xbi_{key}"] = None
    return result


def existing_postmortems():
    """Return set of (ticker, event_date) tuples already postmorted.

    Keyed on (ticker, event_date) so a ticker with a prior postmortem
    does not silently filter out new resolutions on later dates.
    """
    existing = set()
    if not os.path.exists(PM_DIR):
        return existing
    for d in os.listdir(PM_DIR):
        dp = os.path.join(PM_DIR, d)
        if not (os.path.isdir(dp) and DATE_RE.match(d)):
            continue
        for f in os.listdir(dp):
            if f.endswith(".json"):
                existing.add((f.replace(".json", ""), d))
    return existing


def detect_snapshot_transitions(snaps, today_str, lookback_days=45):
    """Detect resolutions by tracking next_catalyst_date forward-transitions.

    The screener's `catalyst_days` resets to the *next* event the moment a
    catalyst date passes, so scanning for `catalyst_days <= 0` in the latest
    snapshot finds nothing. The reliable signal is: when a ticker's
    next_catalyst_date advances forward, the prior date was a resolved event.

    Returns list of (ticker, event_date, family, event_type), filtered to
    events with event_date <= today and within `lookback_days` of latest snap.
    """
    if len(snaps) < 2:
        return []
    latest_dt = datetime.strptime(snaps[-1], "%Y-%m-%d").date()
    cutoff = (latest_dt - timedelta(days=lookback_days)).isoformat()

    last_state = {}  # ticker -> (next_date, family, evt_type)
    detected = {}  # (ticker, event_date) -> (family, evt_type)

    for snap_date in snaps:
        snap = load_snapshot(snap_date)
        for ticker, row in snap.items():
            nxt = row.get("next_catalyst_date", "").strip()
            fam = row.get("catalyst_family", "").strip()
            evt = row.get("catalyst_event_type", "").strip()
            prev = last_state.get(ticker)
            if prev:
                prev_nxt, prev_fam, prev_evt = prev
                if prev_nxt and prev_nxt >= cutoff and prev_nxt <= today_str and (not nxt or nxt > prev_nxt):
                    detected.setdefault((ticker, prev_nxt), (prev_fam, prev_evt))
            last_state[ticker] = (nxt, fam, evt)

    return [(tk, ed, v[0], v[1]) for (tk, ed), v in sorted(detected.items())]


def load_resolution_files():
    """Scan resolutions/ sub-dirs for JSON files."""
    recs = []
    if not os.path.exists(RESOL_DIR):
        return recs
    for sub in sorted(os.listdir(RESOL_DIR)):
        subdir = os.path.join(RESOL_DIR, sub)
        if not os.path.isdir(subdir):
            continue
        for fname in sorted(os.listdir(subdir)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(subdir, fname)) as f:
                recs.append(json.load(f))
    return recs


def best_pre_snap(ticker, event_date_str, snaps):
    """Latest dated snapshot strictly before event_date that has ticker with cat_days <= 3."""
    candidates = [s for s in reversed(snaps) if s < event_date_str]
    for sd in candidates:
        snap = load_snapshot(sd)
        row = snap.get(ticker)
        if row is None:
            continue
        cd_raw = row.get("catalyst_days", "").strip()
        if not cd_raw:
            continue
        try:
            cd = int(float(cd_raw))
        except ValueError:
            continue
        if cd <= 3:
            return sd, row
    # fallback: any snapshot that has the ticker
    for sd in candidates:
        snap = load_snapshot(sd)
        if ticker in snap:
            return sd, snap[ticker]
    return None, None


def write_postmortem(ticker, event_date_str, pre_snap_date, pre_row, outcome, resolution_rec=None):
    out_dir = os.path.join(PM_DIR, event_date_str)
    os.makedirs(out_dir, exist_ok=True)

    def _f(key, default=None):
        v = pre_row.get(key, "").strip()
        return v if v else default

    def _fi(key, default=None):
        v = _f(key)
        try:
            return int(float(v)) if v else default
        except (ValueError, TypeError):
            return default

    def _ff(key, default=None):
        v = _f(key)
        try:
            return round(float(v), 6) if v else default
        except (ValueError, TypeError):
            return default

    # shadow / trade-plan / readiness
    shadow_path = os.path.join(REPO, f"artifacts/live_shadow/positions/{pre_snap_date}.json")
    in_shadow = False
    if os.path.exists(shadow_path):
        with open(shadow_path) as f:
            sh = json.load(f)
        positions = sh if isinstance(sh, list) else sh.get("positions", [])
        in_shadow = any((p if isinstance(p, str) else p.get("ticker", "")) == ticker for p in positions)

    tp_path = os.path.join(REPO, f"artifacts/live_shadow/trade_plan/{pre_snap_date}/trade_plan.csv")
    in_trade = False
    if os.path.exists(tp_path):
        with open(tp_path) as f:
            in_trade = any(r.get("ticker") == ticker for r in csv.DictReader(f))

    rd_path = os.path.join(REPO, f"artifacts/readiness/scorecard_{pre_snap_date}.json")
    readiness = None
    if os.path.exists(rd_path):
        with open(rd_path) as f:
            rd = json.load(f)
        readiness = rd.get(ticker, {}).get("verdict")

    record = {
        "schema": "postmortem.v1",
        "ticker": ticker,
        "event_date": event_date_str,
        "captured_at": datetime.utcnow().isoformat() + "Z",
        "pre_event": {
            "snapshot_date": pre_snap_date,
            "actionable_rank": _fi("actionable_rank"),
            "composite_score": _ff("composite_score"),
            "tier_dev": _f("tier_dev"),
            "size_band": _f("size_band"),
            "target_weight_pct": _ff("target_weight_pct"),
            "catalyst_days": _fi("catalyst_days"),
            "catalyst_mode": _f("catalyst_mode"),
            "catalyst_family": _f("catalyst_family"),
            "catalyst_event_type": _f("catalyst_event_type"),
            "catalyst_source": _f("catalyst_source"),
            "is_hard_catalyst": _f("is_hard_catalyst") == "1",
            "confidence_overall": _ff("confidence_overall"),
            "mom_state": _f("mom_state"),
            "eligible": _f("eligible") == "1",
            "ineligible_reasons": _f("ineligible_reasons") or None,
            "in_shadow": in_shadow,
            "in_trade_plan": in_trade,
            "readiness_verdict": readiness,
            "ruleset_id": _f("decision_engine_ruleset_id") or RULESET_ID,
        },
        "outcome": outcome,
    }
    if resolution_rec:
        record["resolution_source"] = {
            "catalyst_type": resolution_rec.get("catalyst_type"),
            "catalyst_description": resolution_rec.get("catalyst_description"),
            "outcome_label": resolution_rec.get("outcome"),
            "outcome_detail": resolution_rec.get("outcome_detail"),
            "days_from_expected": resolution_rec.get("days_from_expected"),
            "source_type": resolution_rec.get("source_type"),
            "source_id": resolution_rec.get("source_id"),
            "prediction_composite_score": resolution_rec.get("prediction_composite_score"),
            "prediction_snapshot_date": resolution_rec.get("prediction_snapshot_date"),
            "prediction_dem_rank": resolution_rec.get("prediction_dem_rank"),
            "prediction_match_type": resolution_rec.get("prediction_match_type"),
            "event_ev_p_hit": resolution_rec.get("event_ev_p_hit"),
            "event_ev_p_miss": resolution_rec.get("event_ev_p_miss"),
            "event_ev_confidence": resolution_rec.get("event_ev_confidence"),
            "event_ev_asof_date": resolution_rec.get("event_ev_asof_date"),
            "event_ev_match_type": resolution_rec.get("event_ev_match_type"),
        }

    json_path = os.path.join(out_dir, f"{ticker}.json")
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2)

    pre = record["pre_event"]
    out = record["outcome"]
    rs = record.get("resolution_source", {})
    md = [
        f"# Postmortem: {ticker} — {event_date_str}",
        "",
        f"**Captured:** {record['captured_at']}  ",
        f"**Ruleset:** {pre['ruleset_id']}",
        "",
        "## Pre-Event State",
        "| Field | Value |",
        "|-------|-------|",
        f"| Snapshot date | {pre['snapshot_date']} |",
        f"| Actionable rank | {pre['actionable_rank']} |",
        f"| Tier (dev) | {pre['tier_dev']} |",
        f"| Size band | {pre['size_band']} |",
        f"| Target weight % | {pre['target_weight_pct']} |",
        f"| Catalyst days | {pre['catalyst_days']} |",
        f"| Catalyst type | {pre['catalyst_event_type']} |",
        f"| Catalyst family | {pre['catalyst_family']} |",
        f"| Is hard catalyst | {pre['is_hard_catalyst']} |",
        f"| Confidence overall | {pre['confidence_overall']} |",
        f"| Momentum state | {pre['mom_state']} |",
        f"| Eligible | {pre['eligible']} |",
        f"| Ineligible reasons | {pre['ineligible_reasons']} |",
        f"| In shadow portfolio | {pre['in_shadow']} |",
        f"| In trade plan | {pre['in_trade_plan']} |",
        f"| Readiness verdict | {pre['readiness_verdict']} |",
        "",
        "## Resolution",
        "| Field | Value |",
        "|-------|-------|",
        f"| Catalyst type | {rs.get('catalyst_type')} |",
        f"| Description | {rs.get('catalyst_description')} |",
        f"| Outcome label | {rs.get('outcome_label')} |",
        f"| Outcome detail | {rs.get('outcome_detail')} |",
        f"| Days from expected | {rs.get('days_from_expected')} |",
        f"| Source | {rs.get('source_type')} / {rs.get('source_id')} |",
        "",
        "## Outcome",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Pre-close date | {out.get('pre_close_date')} |",
        f"| Pre-close | {out.get('pre_close')} |",
        f"| Return T+1 ({out.get('t1_date', '?')}) | {out.get('return_t1')} |",
        f"| Return T+3 ({out.get('t3_date', '?')}) | {out.get('return_t3')} |",
        f"| Return T+5 ({out.get('t5_date', '?')}) | {out.get('return_t5')} |",
        f"| Excess vs XBI T+1 | {out.get('excess_vs_xbi_t1')} |",
        f"| Excess vs XBI T+3 | {out.get('excess_vs_xbi_t3')} |",
        f"| Excess vs XBI T+5 | {out.get('excess_vs_xbi_t5')} |",
        "",
        "_Facts only. No causal inference._",
    ]
    md_path = os.path.join(out_dir, f"{ticker}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"  ✓ {ticker}: {json_path}")
    return record


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    today = date.today().isoformat()
    snaps = dated_snapshots()
    latest = snaps[-1] if snaps else None
    print(f"[postmortem] {today}  latest_snap={latest}")

    already = existing_postmortems()
    print(f"[postmortem] existing postmortems on file: {len(already)}")

    # Primary source: explicit resolution records dropped into snapshots/resolutions/
    resol_recs = load_resolution_files()
    print(f"[postmortem] resolution records found: {len(resol_recs)}")

    # Build candidate map keyed on (ticker, event_date) so multiple events
    # per ticker are preserved and prior-month postmortems do not mask new ones.
    candidates = {}  # (ticker, event_date) -> resolution_rec_or_None

    for rec in resol_recs:
        t = rec.get("ticker")
        evt = rec.get("catalyst_date") or rec.get("resolution_date")
        if not (t and evt):
            continue
        if evt > today:
            continue  # event hasn't happened yet
        key = (t, evt)
        if key in already:
            continue
        candidates[key] = rec

    # Secondary source: snapshot transitions where next_catalyst_date advanced.
    # This catches resolutions for which no explicit resolution record was filed.
    transitions = detect_snapshot_transitions(snaps, today)
    for ticker, event_date, family, evt_type in transitions:
        key = (ticker, event_date)
        if key in already or key in candidates:
            continue
        candidates[key] = {
            "ticker": ticker,
            "catalyst_date": event_date,
            "catalyst_type": evt_type,
            "catalyst_family": family,
            "source_type": "SNAPSHOT_TRANSITION",
        }

    print(f"[postmortem] candidates: {len(candidates)}")

    if not candidates:
        print("[postmortem] HEARTBEAT_OK — no new resolutions")
        return

    print("[postmortem] loading price history...")
    prices = load_prices()
    xbi_prices = prices.get("XBI", {})

    written, gaps, skipped = [], [], []

    for (ticker, event_date), resol_rec in sorted(candidates.items()):
        print(f"\n[postmortem] {ticker}  event={event_date}")

        pre_snap_date, pre_row = best_pre_snap(ticker, event_date, snaps)
        if pre_snap_date is None:
            print("  ⚠ no pre-event snapshot found")
            gaps.append(f"{ticker}@{event_date}")
            continue

        outcome = compute_returns(ticker, event_date, prices, xbi_prices)
        if not outcome:
            print("  ⚠ no price data")
            gaps.append(f"{ticker}@{event_date}")
            continue

        if outcome.get("return_t3") is None:
            print("  ⚠ T+3 not yet available — skipping")
            skipped.append(f"{ticker}@{event_date}")
            continue

        write_postmortem(ticker, event_date, pre_snap_date, pre_row, outcome, resol_rec)
        written.append(f"{ticker}@{event_date}")

    # Write/update memory note
    os.makedirs(MEM_DIR, exist_ok=True)
    mem_path = os.path.join(MEM_DIR, f"{today}.md")
    with open(mem_path, "w") as f:
        f.write(f"# Memory: {today}\n\n")
        f.write("## Postmortem run\n\n")
        f.write(f"- Latest snapshot: {latest}\n")
        f.write(f"- Existing postmortems on file: {len(already)}\n")
        f.write(f"- Candidates: {sorted(candidates.keys())}\n")
        f.write(f"- Written: {written}\n")
        f.write(f"- Skipped (T+3 not ready): {skipped}\n")
        f.write(f"- Gaps (no data): {gaps}\n")
        f.write(f"- Total postmortems to date: {len(already) + len(written)}\n")
    Path(MEM_DIR).touch()
    print(f"\n[postmortem] written={written}  skipped={skipped}  gaps={gaps}")
    print(f"[postmortem] memory → {mem_path}")
    if skipped:
        print(f"[postmortem] PRICE_DATA_GAP (T+3 pending): {skipped}")
    if not written and not skipped:
        print("[postmortem] HEARTBEAT_OK")

    try:
        from tools.agent_skill_telemetry import log_agent_run
        from tools.record_skill_feedback import attach_outcome_verdict

        n_candidates = len(candidates)
        exec_id = log_agent_run(
            "postmortem",
            f"Postmortem capture for {today}",
            inputs={"as_of_date": today},
            outputs={"written": len(written), "skipped": len(skipped), "gaps": len(gaps)},
            success=True,
        )
        if exec_id and n_candidates:
            capture_rate = len(written) / n_candidates
            attach_outcome_verdict(
                exec_id,
                was_correct=capture_rate >= 0.5,
                evidence=f"captured {len(written)}/{n_candidates} resolution candidates",
            )
    except Exception:
        pass


if __name__ == "__main__":
    main()
