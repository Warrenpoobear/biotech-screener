"""resolve_cross_quarter_delta: no fake zeros when the day's 13F cache is missing.

Regression for 2026-09-18..09-30 / 10-08 / 10-09: the summary was built from an
older cache, the filing period was looked up for as_of_date (missing -> ""),
the cross-quarter guard switched off, the prior was a summary of the SAME cache,
and every net_elite_holders_delta was 0 while has_inst_delta read True.
"""

import json
from pathlib import Path

from institutional_summary import resolve_cross_quarter_delta


def _summary(as_of: str, cache: str, holders: dict) -> dict:
    return {
        "as_of_date": as_of,
        "cache_as_of_date": cache,
        "elite_managers_with_filing": 2,
        "tickers": {t: {"elite_holder_shares": {m: 100 for m in mgrs}} for t, mgrs in holders.items()},
    }


def _write_summary(snap_dir: Path, s: dict) -> None:
    d = snap_dir / s["as_of_date"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "institutional_summary.json").write_text(json.dumps(s))


def _write_cache(cache_base: Path, cache_date: str, period: str) -> None:
    d = cache_base / cache_date
    d.mkdir(parents=True, exist_ok=True)
    (d / "index.json").write_text(json.dumps({"managers": [{"cik": "1", "period_of_report": period}]}))


Q1 = {"AAA": ["m1"], "BBB": ["m1", "m2"]}
Q2 = {"AAA": ["m1", "m2"], "BBB": ["m1"]}


def _setup(tmp_path):
    snaps, cache = tmp_path / "snapshots", tmp_path / "caches" / "sec_13f" / "PIT"
    _write_cache(cache, "2026-08-13", "2026-03-31")
    _write_cache(cache, "2026-08-14", "2026-06-30")
    _write_summary(snaps, _summary("2026-08-13", "2026-08-13", Q1))
    _write_summary(snaps, _summary("2026-08-14", "2026-08-14", Q2))
    return snaps, cache


def test_missing_day_cache_uses_source_cache_period(tmp_path):
    """09-22 has no cache of its own; summary built from 08-14 (Q2).

    The prior must be the Q1 summary (08-13), giving a real delta.
    """
    snaps, cache = _setup(tmp_path)
    cur = _summary("2026-09-22", "2026-08-14", Q2)
    prior, delta, status = resolve_cross_quarter_delta(cur, snaps, "2026-09-22", cache)
    assert status == "ok"
    assert prior["cache_as_of_date"] == "2026-08-13"
    nets = {t: v["net_elite_holders_delta"] for t, v in delta["tickers"].items()}
    assert nets == {"AAA": 1, "BBB": -1}


def test_same_source_prior_yields_no_signal(tmp_path):
    """Only a same-cache prior exists -> no delta, not zeros."""
    snaps, cache = tmp_path / "snapshots", tmp_path / "caches" / "sec_13f" / "PIT"
    _write_cache(cache, "2026-10-07", "")  # period unknown: guard cannot skip it
    _write_summary(snaps, _summary("2026-10-07", "2026-10-07", Q2))
    cur = _summary("2026-10-08", "2026-10-07", Q2)
    prior, delta, status = resolve_cross_quarter_delta(cur, snaps, "2026-10-08", cache)
    assert (prior, delta, status) == (None, None, "same_source_cache")


def test_no_prior(tmp_path):
    snaps, cache = tmp_path / "snapshots", tmp_path / "caches" / "sec_13f" / "PIT"
    snaps.mkdir()
    _write_cache(cache, "2026-08-14", "2026-06-30")
    cur = _summary("2026-08-14", "2026-08-14", Q2)
    assert resolve_cross_quarter_delta(cur, snaps, "2026-08-14", cache) == (None, None, "no_prior")


def test_live_day_unchanged(tmp_path):
    """A day with its own cache behaves as before (cross-quarter vs 08-13)."""
    snaps, cache = _setup(tmp_path)
    _write_cache(cache, "2026-11-11", "2026-06-30")
    cur = _summary("2026-11-11", "2026-11-11", Q2)
    prior, delta, status = resolve_cross_quarter_delta(cur, snaps, "2026-11-11", cache)
    assert status == "ok" and prior["as_of_date"] == "2026-08-13"


def test_unknown_filing_period_is_not_a_zero_delta(tmp_path):
    """Different cache dates, neither index readable: do not emit a delta of zeros."""
    snaps, cache = tmp_path / "snapshots", tmp_path / "caches"
    _write_summary(snaps, _summary("2026-09-18", "2026-09-01", Q2))
    cur = _summary("2026-09-22", "2026-09-02", Q2)
    prior, delta, status = resolve_cross_quarter_delta(cur, snaps, "2026-09-22", cache)
    assert (prior, delta, status) == (None, None, "unknown_filing_period")


def test_filing_period_is_majority_not_first(tmp_path):
    from institutional_summary import _get_filing_period

    d = tmp_path / "2026-08-13"
    d.mkdir()
    mgrs = [{"period_of_report": "2026-06-30"}] + [{"period_of_report": "2026-03-31"}] * 3
    (d / "index.json").write_text(json.dumps({"managers": mgrs}))
    assert _get_filing_period(tmp_path, "2026-08-13") == "2026-03-31"


def test_filing_period_tie_goes_to_later(tmp_path):
    from institutional_summary import _get_filing_period

    d = tmp_path / "2026-11-14"
    d.mkdir()
    mgrs = [{"period_of_report": "2026-06-30"}, {"period_of_report": "2026-09-30"}]
    (d / "index.json").write_text(json.dumps({"managers": mgrs}))
    assert _get_filing_period(tmp_path, "2026-11-14") == "2026-09-30"
