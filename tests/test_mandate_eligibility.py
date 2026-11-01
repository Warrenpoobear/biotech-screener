"""Tests for the authoritative mandate-eligibility rule (SM-20260629-001).

A capture counts toward the 20-window gate ONLY if it is a genuine, complete,
live forward observation of the frozen candidate. See
docs/governance/2026-07-10-dem-candidate-hash-equivalence.md.
"""

from __future__ import annotations

from tools.run_forward_validation import capture_is_eligible_for_mandate, forward_return_realized

GOOD_FILL = {"xs_5d": 0.012}


def _capture(**overrides):
    cap = {
        "capture_mode": "LIVE",
        "quality_status": "PASS",
        "model_hash_match": True,
        "benchmark_available": True,
    }
    cap.update(overrides)
    return cap


def test_fully_eligible():
    assert capture_is_eligible_for_mandate(_capture(), GOOD_FILL) is True


def test_replay_is_ineligible():
    assert capture_is_eligible_for_mandate(_capture(capture_mode="REPLAY"), GOOD_FILL) is False


def test_non_pass_quality_ineligible():
    assert capture_is_eligible_for_mandate(_capture(quality_status="DEGRADED"), GOOD_FILL) is False
    assert capture_is_eligible_for_mandate(_capture(quality_status="FAIL"), GOOD_FILL) is False


def test_hash_mismatch_ineligible():
    assert capture_is_eligible_for_mandate(_capture(model_hash_match=False), GOOD_FILL) is False


def test_missing_benchmark_ineligible():
    assert capture_is_eligible_for_mandate(_capture(benchmark_available=False), GOOD_FILL) is False


def test_unrealized_return_ineligible():
    assert capture_is_eligible_for_mandate(_capture(), None) is False
    assert capture_is_eligible_for_mandate(_capture(), {}) is False
    assert capture_is_eligible_for_mandate(_capture(), {"xs_5d": None}) is False


def test_legacy_capture_without_schema_fields_ineligible():
    # A pre-v2 capture (the 10 backfilled rows) has none of the schema-v2 fields.
    legacy = {"date": "2026-06-26", "data_quality": "PASS"}
    assert capture_is_eligible_for_mandate(legacy, GOOD_FILL) is False


def test_capture_before_clock_epoch_ineligible():
    # An operator-authorized fix resets the out-of-sample clock; earlier
    # captures observed a different model and must not count.
    cap = _capture(date="2026-10-30")
    assert capture_is_eligible_for_mandate(cap, GOOD_FILL, clock_epoch="2026-11-02") is False


def test_capture_on_or_after_clock_epoch_eligible():
    assert capture_is_eligible_for_mandate(_capture(date="2026-11-02"), GOOD_FILL, clock_epoch="2026-11-02") is True
    assert capture_is_eligible_for_mandate(_capture(date="2026-11-09"), GOOD_FILL, clock_epoch="2026-11-02") is True


def test_no_clock_epoch_keeps_prior_rule():
    assert capture_is_eligible_for_mandate(_capture(date="2026-07-01"), GOOD_FILL, clock_epoch=None) is True


def test_clock_epoch_rejects_undated_capture():
    assert capture_is_eligible_for_mandate(_capture(), GOOD_FILL, clock_epoch="2026-11-02") is False


def test_mandate_clock_epoch_reads_candidate():
    from tools.run_forward_validation import mandate_clock_epoch

    assert mandate_clock_epoch({"oos_clock_start": "2026-11-02"}) == "2026-11-02"
    assert mandate_clock_epoch({}) is None


def test_forward_return_realized_helper():
    assert forward_return_realized({"xs_5d": 0.0}) is True
    assert forward_return_realized({"xs_5d": None}) is False
    assert forward_return_realized({}) is False
    assert forward_return_realized(None) is False
