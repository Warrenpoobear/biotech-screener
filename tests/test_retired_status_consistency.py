"""Tests for consistent handling of universe.json retirement statuses (2026-08-05).

Root cause: `tools/maintain_universe.py retire` infers the new status from the
reason text — an acquisition becomes "excluded_acquired", a delisting or
bankruptcy becomes "delisted", anything else "retired". But five call sites
hardcoded `status == "delisted"`:

    tools/run_daily_production.py  price-refresh ticker list
    tools/run_daily_production.py  coverage denominator (x2)
    run_screen.py                  raw_universe filter
    scripts/run_screen_from_bundle.py  universe filter

`module_1_universe._classify_status` already treated acquired/m&a/
excluded_acquired as exclusions, so the semantically-correct retirement of an
acquired company dropped it from scoring while the price refresh kept fetching
it from yfinance every run — forever, for a company that no longer trades.

Concretely, on 2026-08-05 five tickers had been retried every run for weeks:
CNTA (Lilly, closed 06-24), SGMO (Nasdaq delist -> OTCQB -> Ch.11, now SGMOQ),
ESPR (ARCHIMED, 07-13), CPRX (Angelini, 07-15), NUVL (GSK, 07-15). All produced
45 ERROR lines per run.

Fix: common.types.is_retired_status covers every retirement status, and
deliberately does NOT cover pending_coverage (not-ready-yet still wants a feed).
"""

from __future__ import annotations

import pytest

from common.types import RETIRED_UNIVERSE_STATUSES, is_retired_status


# --- the statuses maintain_universe.py can actually write -------------------
@pytest.mark.parametrize("status", ["delisted", "excluded_acquired", "retired"])
def test_every_retire_status_is_recognised(status):
    """These three are exactly the --status choices of `maintain_universe.py retire`."""
    assert is_retired_status(status) is True


def test_acquired_variants_recognised():
    """module_1_universe._classify_status also accepts these spellings."""
    for s in ("acquired", "m&a", "d"):
        assert is_retired_status(s) is True


def test_excluded_acquired_is_the_regression():
    """THE bug: an acquisition retired the natural way must be treated as retired.
    Before the fix, `status == "delisted"` was False here and the ticker kept
    getting fetched every run."""
    assert is_retired_status("excluded_acquired") is True


# --- what must NOT be treated as retired -----------------------------------
def test_active_is_not_retired():
    assert is_retired_status("active") is False
    assert is_retired_status("") is False


def test_pending_coverage_is_not_retired():
    """ "Not ready yet" is not "gone" — these still want a price feed so coverage
    can be built. Treating them as retired would strand them permanently."""
    assert is_retired_status("pending_coverage") is False
    assert is_retired_status("pending_data_collection") is False


def test_benchmark_is_not_retired():
    assert is_retired_status("benchmark") is False


# --- robustness ------------------------------------------------------------
def test_case_and_whitespace_insensitive():
    assert is_retired_status("  Delisted ") is True
    assert is_retired_status("EXCLUDED_ACQUIRED") is True


def test_non_string_is_not_retired():
    for junk in (None, 0, 1, [], {}, object()):
        assert is_retired_status(junk) is False


def test_status_set_is_immutable():
    assert isinstance(RETIRED_UNIVERSE_STATUSES, frozenset)


# --- the call sites actually use the helper --------------------------------
def test_no_call_site_still_hardcodes_the_delisted_literal():
    """Guard against the pattern creeping back in. Any new
    `status == "delisted"` comparison in these files should use
    is_retired_status() instead."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    offenders = []
    for rel in (
        "run_screen.py",
        "tools/run_daily_production.py",
        "scripts/run_screen_from_bundle.py",
    ):
        text = (root / rel).read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if 'status") == "delisted"' in line or "status') == 'delisted'" in line:
                offenders.append(f"{rel}:{i}")
    assert not offenders, f"hardcoded delisted comparison still present: {offenders}"
