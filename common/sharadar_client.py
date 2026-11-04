"""Direct Sharadar API client (https://api.sharadar.com/v1.0).

This is the *direct* Sharadar service, not Nasdaq Data Link. Keys issued at
sharadar.com do not work against ``data.nasdaq.com`` and vice versa.

Key handling
    The key is read from ``SHARADAR_API_KEY`` and sent as the ``x-api-key``
    header — never on the query string, so it cannot leak into logs, error
    messages or cached URLs.

Pagination
    Sharadar reports ``count == len(data)`` even when a response is truncated
    at ``limit`` (verified live 2026-11-04: a >10k-row query returned
    ``count=10000``). The vendor README's "compare len(data) to count" check
    therefore never fires. This client treats a full page as "maybe more" and
    requests the next ``offset`` until a short page arrives.

Governance
    Infrastructure only (Tier 2 ingestion plumbing). Nothing in the production
    scoring path imports this module; see tests/test_sync_sharadar_shadow.py
    for the isolation guard.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

BASE_URL = "https://api.sharadar.com/v1.0"
API_KEY_ENV_VAR = "SHARADAR_API_KEY"  # pragma: allowlist secret (env var name)
DEFAULT_PAGE_LIMIT = 10000
# Live API limits on the ticker param (both return HTTP 400; observed 2026-11-04).
MAX_TICKERS_PER_REQUEST = 30
MAX_TICKER_PARAM_CHARS = 200
DEFAULT_TICKER_BATCH = MAX_TICKERS_PER_REQUEST
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})

# fetch(url, params, headers, timeout) -> (http_status, body_bytes)
Fetcher = Callable[[str, Mapping[str, Any], Mapping[str, str], float], Tuple[int, bytes]]


class SharadarError(RuntimeError):
    """Any non-recoverable Sharadar API failure."""


class SharadarAuthError(SharadarError):
    """Missing key, invalid key, or request outside the subscription (401/403)."""


def _urllib_fetch(url: str, params: Mapping[str, Any], headers: Mapping[str, str], timeout: float) -> Tuple[int, bytes]:
    full_url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(full_url, headers=dict(headers))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _error_text(body: bytes) -> str:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return body[:200].decode("utf-8", "replace")
    if isinstance(payload, dict):
        parts = [str(payload.get(k)) for k in ("error", "description") if payload.get(k)]
        if parts:
            return ": ".join(parts)
    return str(payload)[:200]


def _ticker_batches(tickers: Sequence[str], max_count: int, max_chars: int) -> List[List[str]]:
    """Split tickers so each comma-joined batch has <= max_count names and <= max_chars chars."""
    batches: List[List[str]] = []
    cur: List[str] = []
    cur_len = 0
    for t in tickers:
        if len(t) > max_chars:
            raise SharadarError(f"ticker {t!r} exceeds {max_chars} chars")
        add = len(t) + (1 if cur else 0)
        if cur and (len(cur) >= max_count or cur_len + add > max_chars):
            batches.append(cur)
            cur, cur_len, add = [], 0, len(t)
        cur.append(t)
        cur_len += add
    if cur:
        batches.append(cur)
    return batches


class SharadarClient:
    """Thin, retrying, paginating client for ``GET /data/{table}``."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: str = BASE_URL,
        fetch: Optional[Fetcher] = None,
        page_limit: int = DEFAULT_PAGE_LIMIT,
        max_retries: int = 4,
        backoff_seconds: float = 2.0,
        timeout: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        key = (api_key or os.environ.get(API_KEY_ENV_VAR) or "").strip()
        if not key:
            raise SharadarAuthError(f"Sharadar API key not found: set {API_KEY_ENV_VAR}")
        if page_limit < 1:
            raise ValueError("page_limit must be >= 1")
        self._key = key
        self._base_url = base_url.rstrip("/")
        self._fetch = fetch or _urllib_fetch
        self._page_limit = page_limit
        self._max_retries = max_retries
        self._backoff = backoff_seconds
        self._timeout = timeout
        self._sleep = sleep

    def __repr__(self) -> str:
        return f"SharadarClient(base_url={self._base_url!r}, page_limit={self._page_limit})"

    # ------------------------------------------------------------------ public

    def get_table(
        self,
        table: str,
        filters: Optional[Mapping[str, Any]] = None,
        fields: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """All rows of ``table`` matching ``filters`` (e.g. ``{"date.gte": ...}``)."""
        base: Dict[str, Any] = {"format": "json"}
        if filters:
            base.update(filters)
        if fields:
            base["fields"] = ",".join(fields)

        rows: List[Dict[str, Any]] = []
        offset = 0
        while True:
            params = dict(base, limit=self._page_limit, offset=offset)
            page = self._request(table, params)
            rows.extend(page)
            if len(page) < self._page_limit:
                return rows
            offset += self._page_limit

    def get_table_for_tickers(
        self,
        table: str,
        tickers: Iterable[str],
        filters: Optional[Mapping[str, Any]] = None,
        fields: Optional[Sequence[str]] = None,
        batch_size: int = DEFAULT_TICKER_BATCH,
    ) -> List[Dict[str, Any]]:
        """``get_table`` over ``tickers`` in comma-joined batches (sorted, deduped)."""
        unique = sorted({t.strip().upper() for t in tickers if t and t.strip()})
        rows: List[Dict[str, Any]] = []
        max_count = max(1, min(batch_size, MAX_TICKERS_PER_REQUEST))
        for batch in _ticker_batches(unique, max_count, MAX_TICKER_PARAM_CHARS):
            batch_filters = dict(filters or {})
            batch_filters["ticker"] = ",".join(batch)
            rows.extend(self.get_table(table, filters=batch_filters, fields=fields))
        return rows

    # ----------------------------------------------------------------- private

    def _request(self, table: str, params: Mapping[str, Any]) -> List[Dict[str, Any]]:
        url = f"{self._base_url}/data/{table}"
        headers = {"x-api-key": self._key, "Accept": "application/json"}
        last_err = ""
        for attempt in range(self._max_retries + 1):
            if attempt:
                self._sleep(self._backoff * (2 ** (attempt - 1)))
            try:
                status, body = self._fetch(url, params, headers, self._timeout)
            except OSError as e:  # URLError, timeouts, connection resets
                last_err = f"network error: {e}"
                continue

            if status in (401, 403):
                raise SharadarAuthError(f"{table}: HTTP {status}: {_error_text(body)}")
            if status in RETRYABLE_STATUSES:
                last_err = f"HTTP {status}: {_error_text(body)}"
                continue
            if status != 200:
                raise SharadarError(f"{table}: HTTP {status}: {_error_text(body)}")

            try:
                payload = json.loads(body.decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as e:
                raise SharadarError(f"{table}: invalid JSON response: {e}") from e
            data = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(data, list):
                raise SharadarError(f"{table}: response missing 'data' list")
            return data

        raise SharadarError(f"{table}: gave up after {self._max_retries + 1} attempts ({last_err})")
