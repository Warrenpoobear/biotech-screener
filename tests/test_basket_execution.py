#!/usr/bin/env python3
"""Tests for basket assembly, CSRF, and execution idempotency (PR 3b).

This is the layer between "user clicks Execute" and the subprocess that places orders.
Three things must hold, and each has cost a real trading system money somewhere:

* the basket the user approved is the basket that executes (no re-read between review
  and click, no silent substitution if the snapshot rolls over mid-session)
* one click executes once (a double-submitted form must not double the position)
* a state-changing POST cannot be driven cross-origin
"""

import json

import pytest

from dashboard.basket import (
    BasketAlreadyExecuted,
    BasketMismatch,
    CSRFError,
    ExecutionLedger,
    build_basket,
    issue_csrf,
    verify_csrf,
)

SECRET = b"basket-test-secret"


def _rankings():
    """Three ranked names, descending final_score, as rankings.csv rows."""
    return {
        "COGT": {"ticker": "COGT", "actionable_rank": "1", "final_score": "0.65"},
        "DNTH": {"ticker": "DNTH", "actionable_rank": "2", "final_score": "0.64"},
        "NRIX": {"ticker": "NRIX", "actionable_rank": "3", "final_score": "0.63"},
    }


class TestBasketAssembly:
    def test_basket_is_ordered_by_actionable_rank(self):
        b = build_basket("2026-08-03", _rankings(), top_n=3, equity_usd="300")
        assert [p["ticker"] for p in b.positions] == ["COGT", "DNTH", "NRIX"]

    def test_top_n_truncates(self):
        b = build_basket("2026-08-03", _rankings(), top_n=2, equity_usd="300")
        assert [p["ticker"] for p in b.positions] == ["COGT", "DNTH"]

    def test_equal_weight_notional_split(self):
        b = build_basket("2026-08-03", _rankings(), top_n=3, equity_usd="300")
        assert all(p["notional_usd"] == "100.00" for p in b.positions)

    def test_basket_id_is_stable_for_same_inputs(self):
        a = build_basket("2026-08-03", _rankings(), top_n=3, equity_usd="300")
        b = build_basket("2026-08-03", _rankings(), top_n=3, equity_usd="300")
        assert a.basket_id == b.basket_id

    def test_basket_id_changes_with_membership(self):
        a = build_basket("2026-08-03", _rankings(), top_n=3, equity_usd="300")
        b = build_basket("2026-08-03", _rankings(), top_n=2, equity_usd="300")
        assert a.basket_id != b.basket_id

    def test_basket_id_changes_with_date(self):
        a = build_basket("2026-08-03", _rankings(), top_n=3, equity_usd="300")
        b = build_basket("2026-08-04", _rankings(), top_n=3, equity_usd="300")
        assert a.basket_id != b.basket_id

    def test_unranked_rows_are_excluded(self):
        r = _rankings()
        r["ZZZZ"] = {"ticker": "ZZZZ", "actionable_rank": "", "final_score": ""}
        b = build_basket("2026-08-03", r, top_n=10, equity_usd="300")
        assert "ZZZZ" not in [p["ticker"] for p in b.positions]

    def test_empty_rankings_yields_empty_basket(self):
        b = build_basket("2026-08-03", {}, top_n=30, equity_usd="300")
        assert b.positions == []


class TestApprovedBasketIsTheExecutedBasket:
    def test_mismatched_basket_id_is_refused(self):
        b = build_basket("2026-08-03", _rankings(), top_n=3, equity_usd="300")
        with pytest.raises(BasketMismatch):
            b.assert_matches("some-other-basket-id")

    def test_matching_basket_id_passes(self):
        b = build_basket("2026-08-03", _rankings(), top_n=3, equity_usd="300")
        b.assert_matches(b.basket_id)

    def test_snapshot_rollover_invalidates_the_approved_basket(self):
        """User reviews Monday's basket; Tuesday's snapshot lands before they click."""
        approved = build_basket("2026-08-03", _rankings(), top_n=3, equity_usd="300")
        now_current = build_basket("2026-08-04", _rankings(), top_n=3, equity_usd="300")
        with pytest.raises(BasketMismatch):
            now_current.assert_matches(approved.basket_id)


class TestCSRF:
    def test_round_trip(self):
        tok = issue_csrf("scott", secret=SECRET)
        verify_csrf(tok, "scott", secret=SECRET)

    def test_token_for_another_user_is_refused(self):
        tok = issue_csrf("darren", secret=SECRET)
        with pytest.raises(CSRFError):
            verify_csrf(tok, "scott", secret=SECRET)

    def test_tampered_token_refused(self):
        tok = issue_csrf("scott", secret=SECRET)
        with pytest.raises(CSRFError):
            verify_csrf(tok[:-4] + "0000", "scott", secret=SECRET)

    def test_missing_token_refused(self):
        for junk in ("", None, "garbage"):
            with pytest.raises(CSRFError):
                verify_csrf(junk, "scott", secret=SECRET)

    def test_token_from_other_secret_refused(self):
        tok = issue_csrf("scott", secret=b"attacker")
        with pytest.raises(CSRFError):
            verify_csrf(tok, "scott", secret=SECRET)


class TestIdempotency:
    def test_first_reservation_is_allowed(self, tmp_path):
        led = ExecutionLedger(tmp_path / "exec.db")
        led.reserve("scott", "basket-abc")

    def test_second_reservation_of_same_basket_is_refused(self, tmp_path):
        led = ExecutionLedger(tmp_path / "exec.db")
        led.reserve("scott", "basket-abc")
        with pytest.raises(BasketAlreadyExecuted):
            led.reserve("scott", "basket-abc")

    def test_reservation_blocks_even_before_the_result_is_recorded(self, tmp_path):
        """The window the review identified: claimed, orders in flight, nothing recorded."""
        led = ExecutionLedger(tmp_path / "exec.db")
        led.reserve("scott", "basket-abc")
        # ...order loop would run here...
        with pytest.raises(BasketAlreadyExecuted):
            led.reserve("scott", "basket-abc")

    def test_ledger_is_per_tenant(self, tmp_path):
        """Scott executing his basket must not block Darren executing his."""
        led = ExecutionLedger(tmp_path / "exec.db")
        led.reserve("scott", "basket-abc")
        led.reserve("darren", "basket-abc")

    def test_different_basket_same_tenant_allowed(self, tmp_path):
        led = ExecutionLedger(tmp_path / "exec.db")
        led.reserve("scott", "basket-abc")
        led.reserve("scott", "basket-def")

    def test_reservation_survives_reopen(self, tmp_path):
        path = tmp_path / "exec.db"
        ExecutionLedger(path).reserve("scott", "basket-abc")
        with pytest.raises(BasketAlreadyExecuted):
            ExecutionLedger(path).reserve("scott", "basket-abc")

    def test_recorded_payload_is_readable_back(self, tmp_path):
        led = ExecutionLedger(tmp_path / "exec.db")
        led.reserve("scott", "basket-abc")
        led.record("scott", "basket-abc", {"placed": 2, "mode": "LIVE"})
        rec = led.get("scott", "basket-abc")
        assert rec["status"] == "completed"
        assert rec["result"]["mode"] == "LIVE"

    def test_status_is_reserved_until_recorded(self, tmp_path):
        led = ExecutionLedger(tmp_path / "exec.db")
        led.reserve("scott", "basket-abc")
        assert led.get("scott", "basket-abc")["status"] == "reserved"


class TestConcurrentExecutionIsSerialised:
    """The PR #14 review finding, exercised concurrently rather than sequentially.

    Previously the check (assert_not_executed) and the write (record) were separated by
    the entire order-placement loop with no lock, so two requests could both pass the
    check before either recorded. The old test only proved the sequential case. These
    launch real threads contending for the same key and assert exactly one wins.
    """

    @staticmethod
    def _race(path, user_id, basket_id, n_threads):
        import threading

        winners, losers = [], []
        lock = threading.Lock()
        barrier = threading.Barrier(n_threads, timeout=30)

        def attempt(i):
            led = ExecutionLedger(path)  # separate connection, as a separate worker would
            barrier.wait()  # maximise overlap
            try:
                led.reserve(user_id, basket_id)
            except BasketAlreadyExecuted:
                with lock:
                    losers.append(i)
            else:
                with lock:
                    winners.append(i)

        threads = [threading.Thread(target=attempt, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        return winners, losers

    def test_exactly_one_of_two_concurrent_requests_wins(self, tmp_path):
        winners, losers = self._race(tmp_path / "exec.db", "scott", "basket-abc", 2)
        assert len(winners) == 1, f"expected exactly one winner, got {winners}"
        assert len(losers) == 1

    def test_exactly_one_of_sixteen_concurrent_requests_wins(self, tmp_path):
        winners, losers = self._race(tmp_path / "exec.db", "scott", "basket-abc", 16)
        assert len(winners) == 1, f"expected exactly one winner, got {winners}"
        assert len(losers) == 15

    def test_concurrent_requests_for_different_tenants_all_win(self, tmp_path):
        """Serialisation must be per-key, not a global mutex on the whole ledger."""
        import threading

        path = tmp_path / "exec.db"
        results = []
        lock = threading.Lock()
        barrier = threading.Barrier(8, timeout=30)

        def attempt(uid):
            led = ExecutionLedger(path)
            barrier.wait()
            try:
                led.reserve(uid, "basket-abc")
            except BasketAlreadyExecuted:
                pass
            else:
                with lock:
                    results.append(uid)

        threads = [threading.Thread(target=attempt, args=(f"tenant{i:02d}",)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert len(results) == 8, "distinct tenants must not contend with each other"

    def test_concurrent_first_construction_never_raises_locked(self, tmp_path):
        """Many processes/threads may open a fresh ledger at once (first request after deploy).

        Switching a new database to WAL takes an exclusive lock and SQLite can return
        SQLITE_BUSY at once instead of waiting, so a bare open raised
        "database is locked" about 1 in 350 times with 16 threads; a thread that dies
        before the race barrier also hung the other tests forever.
        """
        import threading

        errors = []
        lock = threading.Lock()
        for trial in range(150):
            path = tmp_path / f"exec{trial}.db"

            def build():
                try:
                    ExecutionLedger(path)
                except Exception as exc:  # pragma: no cover - the failure being tested for
                    with lock:
                        errors.append(repr(exc))

            threads = [threading.Thread(target=build) for _ in range(16)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=60)
        assert errors == []


class TestCredentialRedaction:
    def test_top_level_credential_is_stripped(self, tmp_path):
        led = ExecutionLedger(tmp_path / "exec.db")
        led.reserve("scott", "basket-abc")
        led.record("scott", "basket-abc", {"placed": 1, "bearer": "LEAK_TOP"})
        assert "LEAK_TOP" not in json.dumps(led.get("scott", "basket-abc"))

    def test_nested_credential_is_stripped(self, tmp_path):
        """Review note: review/raw come straight from the broker response."""
        led = ExecutionLedger(tmp_path / "exec.db")
        led.reserve("scott", "basket-abc")
        led.record(
            "scott",
            "basket-abc",
            {"results": [{"ticker": "COGT", "raw": {"authorization": "LEAK_NESTED"}}]},
        )
        assert "LEAK_NESTED" not in json.dumps(led.get("scott", "basket-abc"))

    def test_deeply_nested_credential_is_stripped(self, tmp_path):
        led = ExecutionLedger(tmp_path / "exec.db")
        led.reserve("scott", "basket-abc")
        led.record("scott", "basket-abc", {"a": {"b": {"c": {"token": "LEAK_DEEP"}}}})
        assert "LEAK_DEEP" not in json.dumps(led.get("scott", "basket-abc"))

    def test_non_credential_nested_data_is_preserved(self, tmp_path):
        led = ExecutionLedger(tmp_path / "exec.db")
        led.reserve("scott", "basket-abc")
        led.record("scott", "basket-abc", {"results": [{"ticker": "COGT", "order_id": "ord-1"}]})
        assert "ord-1" in json.dumps(led.get("scott", "basket-abc"))
