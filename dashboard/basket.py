"""Basket assembly, CSRF, and execution idempotency for the approval flow.

See ``docs/design/MULTI_TENANCY_PR_PLAN.md`` §3 (PR 3b).

This sits between "user reviews the latest basket" and ``common.order_broker``. It owns
three properties that a review-then-click flow gets wrong by default:

**The approved basket is the executed basket.** Each basket carries a ``basket_id`` — a
digest over its date, membership, and sizing. The review page renders the id; the execute
request must send it back; the server rebuilds the basket from the current snapshot and
refuses if the id no longer matches. So if a new snapshot promotes overnight while the
review page is still open, the click fails loudly rather than silently trading a basket
nobody looked at.

**One click executes once.** The execution ledger is keyed on ``(user_id, basket_id)``.
A double-submitted form, a retried request, or an impatient second click is refused
rather than doubling the position.

**A state-changing POST cannot be driven cross-origin.** CSRF tokens are HMAC-bound to
the tenant, so a token minted for one user is useless for another. This is belt-and-braces
alongside ``SameSite=Strict`` on the session cookie.

Python 3.10 compatible.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import ROUND_DOWN, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Optional

from common.tenancy import validate_user_id

#: Keys never written to the execution ledger, whatever the caller passes.
_LEDGER_REDACT = frozenset({"bearer", "token", "authorization", "anthropic_api_key", "password"})


class BasketError(Exception):
    """Base for approval-flow refusals."""


class BasketMismatch(BasketError):
    """The approved basket is not the basket that would execute now."""


class BasketAlreadyExecuted(BasketError):
    """This tenant already executed this basket."""


class CSRFError(BasketError):
    """CSRF token missing, malformed, or not bound to this tenant."""


@dataclass(frozen=True)
class Basket:
    """A reviewable, executable set of positions."""

    as_of_date: str
    basket_id: str
    positions: "list[dict[str, Any]]" = field(default_factory=list)
    equity_usd: str = "0"

    def assert_matches(self, approved_basket_id: str) -> None:
        if not approved_basket_id or not hmac.compare_digest(self.basket_id, str(approved_basket_id)):
            raise BasketMismatch(
                "the basket under review is no longer current (approved "
                + repr(str(approved_basket_id)[:12])
                + ", current "
                + repr(self.basket_id[:12])
                + ") — re-review before executing"
            )

    def as_dict(self) -> "dict[str, Any]":
        return {
            "as_of_date": self.as_of_date,
            "basket_id": self.basket_id,
            "equity_usd": self.equity_usd,
            "positions": list(self.positions),
        }


def _rank_of(row: Mapping[str, Any]) -> Optional[int]:
    raw = str(row.get("actionable_rank", "")).strip()
    if not raw:
        return None
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def build_basket(
    as_of_date: str,
    rankings: Mapping[str, Mapping[str, Any]],
    *,
    top_n: int = 30,
    equity_usd: str = "0",
) -> Basket:
    """Assemble the equal-weight top-N basket from a snapshot's rankings.

    Rows without an ``actionable_rank`` are excluded — an unranked row is not a member of
    the cohort, and silently treating it as rank-0 would put an arbitrary name at the top.
    """
    ranked = []
    for row in rankings.values():
        rank = _rank_of(row)
        if rank is None:
            continue
        ranked.append((rank, row))
    ranked.sort(key=lambda pair: pair[0])
    ranked = ranked[: max(0, int(top_n))]

    try:
        equity = Decimal(str(equity_usd))
    except (InvalidOperation, ValueError):
        equity = Decimal("0")

    n = len(ranked)
    per = (equity / n).quantize(Decimal("0.01"), rounding=ROUND_DOWN) if n else Decimal("0")

    positions = [
        {
            "ticker": str(row.get("ticker", "")).upper(),
            "rank": rank,
            "notional_usd": str(per),
            "final_score": str(row.get("final_score", "")),
        }
        for rank, row in ranked
    ]

    digest = hashlib.sha256()
    digest.update(as_of_date.encode())
    digest.update(str(equity).encode())
    for p in positions:
        digest.update(("|" + p["ticker"] + ":" + str(p["rank"]) + ":" + p["notional_usd"]).encode())

    return Basket(
        as_of_date=as_of_date,
        basket_id=digest.hexdigest()[:32],
        positions=positions,
        equity_usd=str(equity),
    )


# ---------------------------------------------------------------------------------
# CSRF
# ---------------------------------------------------------------------------------


def _csrf_secret(explicit: bytes | str | None = None) -> bytes:
    raw = explicit if explicit is not None else os.environ.get("BIOTECH_SESSION_SECRET", "")
    if not raw:
        raise CSRFError("no session secret available to sign CSRF tokens")
    return raw.encode() if isinstance(raw, str) else raw


def issue_csrf(user_id: str, *, secret: bytes | str | None = None) -> str:
    """Mint a CSRF token bound to ``user_id``."""
    validate_user_id(user_id)
    return hmac.new(_csrf_secret(secret), ("csrf:" + user_id).encode(), hashlib.sha256).hexdigest()


def verify_csrf(token: Any, user_id: str, *, secret: bytes | str | None = None) -> None:
    """Raise unless ``token`` was minted for ``user_id``."""
    if not token or not isinstance(token, str):
        raise CSRFError("missing CSRF token")
    expected = issue_csrf(user_id, secret=secret)
    if not hmac.compare_digest(expected, token):
        raise CSRFError("CSRF token is not valid for this session")


# ---------------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------------


def _redact(value: Any) -> Any:
    """Recursively strip credential-shaped keys from a nested structure.

    The ``review``/``raw`` sub-objects come straight from the broker's MCP response, whose
    exact shape we do not control. Stripping only top-level keys meant "no credential
    reaches the ledger" held one level deep and nowhere else.
    """
    if isinstance(value, Mapping):
        return {k: _redact(v) for k, v in value.items() if str(k).lower() not in _LEDGER_REDACT}
    if isinstance(value, (list, tuple)):
        return [_redact(v) for v in value]
    return value


class ExecutionLedger:
    """Atomic execution claims, keyed on ``(user_id, basket_id)``.

    SQLite with a composite primary key, so a duplicate claim fails at the database level
    regardless of timing — the same pattern ``common/credstore.py`` already uses.

    This replaces an append-only JSONL file whose check (``assert_not_executed``) and write
    (``record``) were separated by the entire order-placement loop, with nothing locking
    between them. Two concurrent requests for the same key could both pass the check before
    either wrote, and both would place the full basket. Serialisation appeared to work only
    because the handler blocks the event loop in a single-worker deployment — an accident of
    deployment shape that breaks under multiple Uvicorn workers.

    Usage is two-phase, and the order matters:

    1. :meth:`reserve` — claim the key **before** placing anything. Atomic; the loser of a
       race raises :class:`BasketAlreadyExecuted`.
    2. :meth:`record` — attach the outcome once the loop finishes.

    A row left in ``reserved`` state means a run claimed the basket and did not report back
    (crash, kill, timeout). That deliberately keeps the basket un-runnable: the orders may
    have been placed, so the safe reading of an unfinished run is "assume it happened" and
    reconcile against the account.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS executions (
        user_id    TEXT NOT NULL,
        basket_id  TEXT NOT NULL,
        reserved_at TEXT NOT NULL,
        recorded_at TEXT,
        status     TEXT NOT NULL,
        result     TEXT,
        PRIMARY KEY (user_id, basket_id)
    );
    """

    # Bounded retry for first-time setup (see __init__).
    _INIT_DEADLINE_S = 30.0

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Switching a fresh database to WAL needs an exclusive lock, and SQLite can return
        # SQLITE_BUSY at once (no busy-handler wait) when several connections open the new
        # file together. The schema is idempotent, so retry the whole setup until a deadline.
        deadline = time.monotonic() + self._INIT_DEADLINE_S
        delay = 0.005
        while True:
            try:
                con = self._connect()
                try:
                    con.executescript(self._SCHEMA)
                    con.commit()
                finally:
                    con.close()
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc) or time.monotonic() >= deadline:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 0.25)

    def _connect(self) -> sqlite3.Connection:
        # timeout covers the brief writer lock; isolation_level=None keeps the INSERT its
        # own immediate transaction rather than deferring inside an implicit one.
        con = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=30000")
        return con

    def reserve(self, user_id: str, basket_id: str) -> None:
        """Atomically claim ``(user_id, basket_id)``. Raises if already claimed."""
        con = self._connect()
        try:
            con.execute(
                "INSERT INTO executions (user_id, basket_id, reserved_at, status) VALUES (?, ?, ?, 'reserved')",
                (str(user_id), str(basket_id), datetime.now(timezone.utc).isoformat()),
            )
        except sqlite3.IntegrityError as exc:
            raise BasketAlreadyExecuted(
                "tenant " + repr(user_id) + " already executed basket " + repr(str(basket_id)[:12])
            ) from exc
        finally:
            con.close()

    def record(self, user_id: str, basket_id: str, result: Mapping[str, Any]) -> None:
        """Attach the outcome to a previously reserved claim."""
        safe = _redact(dict(result))
        con = self._connect()
        try:
            con.execute(
                "UPDATE executions SET status='completed', recorded_at=?, result=? " "WHERE user_id=? AND basket_id=?",
                (
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(safe, sort_keys=True),
                    str(user_id),
                    str(basket_id),
                ),
            )
        finally:
            con.close()

    @staticmethod
    def is_safe_to_release(*, placed: int, failures: "list[Mapping[str, Any]]") -> bool:
        """Whether a claim may be dropped so the basket can be attempted again.

        This is the entire safety argument for :meth:`release`, kept as one small pure
        function so it can be read and tested on its own. A claim is releasable only when
        it is *provable* that nothing reached the broker:

        * ``placed == 0`` — no order was confirmed placed, and
        * there was at least one failure — a run with nothing placed and nothing failed
          is an empty basket that completed, not a failed attempt, and
        * **every** failure is unambiguous — none timed out, and none failed during the
          placement call itself, where the order may have landed before the error.

        A failure missing the ``ambiguous`` key counts as ambiguous. Failing closed on an
        unclassified failure is the only safe default: the cost of a wrong "safe" is a
        duplicated position.
        """
        if placed != 0:
            return False
        if not failures:
            return False
        return all(f.get("ambiguous") is False for f in failures)

    def release(self, user_id: str, basket_id: str, *, reason: str) -> None:
        """Drop a *reserved* claim so the basket can be attempted again.

        Refuses once the claim is ``completed``: a recorded outcome means the placement
        loop ran to the end, and releasing it would permit a genuine double-execution.
        Callers must gate on :meth:`is_safe_to_release` first — this method enforces only
        the state precondition, not the safety analysis.
        """
        rec = self.get(user_id, basket_id)
        if rec is None:
            return  # nothing claimed; nothing to release
        if rec.get("status") != "reserved":
            raise BasketAlreadyExecuted(
                "refusing to release basket "
                + repr(str(basket_id)[:12])
                + " for tenant "
                + repr(user_id)
                + ": it is "
                + str(rec.get("status"))
                + ", not merely reserved"
            )
        con = self._connect()
        try:
            con.execute(
                "DELETE FROM executions WHERE user_id=? AND basket_id=? AND status='reserved'",
                (str(user_id), str(basket_id)),
            )
        finally:
            con.close()

    def get(self, user_id: str, basket_id: str) -> "Optional[dict[str, Any]]":
        con = self._connect()
        try:
            row = con.execute(
                "SELECT reserved_at, recorded_at, status, result FROM executions " "WHERE user_id=? AND basket_id=?",
                (str(user_id), str(basket_id)),
            ).fetchone()
        finally:
            con.close()
        if row is None:
            return None
        reserved_at, recorded_at, status, result = row
        return {
            "user_id": str(user_id),
            "basket_id": str(basket_id),
            "reserved_at": reserved_at,
            "recorded_at": recorded_at,
            "status": status,
            "result": json.loads(result) if result else {},
        }
