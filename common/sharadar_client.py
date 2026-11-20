"""Direct Sharadar API client (https://api.sharadar.com/v1.0).

This is the *direct* Sharadar service, not Nasdaq Data Link. Keys issued at
sharadar.com do not work against ``data.nasdaq.com`` and vice versa.

Key handling
    The key is read from ``SHARADAR_API_KEY`` and sent as the ``x-api-key``
    header — never on the query string, so it cannot leak into logs, error
    messages or cached URLs.

Truncation (no offset paging)
    Sharadar reports ``count == len(data)`` even when a response is truncated
    at ``limit`` (verified live 2026-11-04: a >10k-row query returned
    ``count=10000``), so the vendor README's "compare len(data) to count" check
    never fires. Offset paging is NOT safe either: ordering is unstable across
    pages, and a 2026-11-04 backfill both duplicated 107 rows and silently
    skipped rows (VKTX lost a trading day). So every request must fit in a
    single page: a full page raises ``SharadarTruncatedError``, and ``fetch``
    recursively splits the date range (then the ticker batch) until each
    piece fits.

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
from datetime import date, timedelta
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


class SharadarTruncatedError(SharadarError):
    """A response filled the page limit, so rows may be missing."""


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
    """Thin, retrying client for ``GET /data/{table}``; splits instead of paging."""

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
        """One request. Raises ``SharadarTruncatedError`` if the page is full."""
        params: Dict[str, Any] = {"format": "json"}
        if filters:
            params.update(filters)
        if fields:
            params["fields"] = ",".join(fields)
        params["limit"] = self._page_limit
        rows = self._request(table, params)
        if len(rows) >= self._page_limit:
            raise SharadarTruncatedError(f"{table}: response hit the {self._page_limit}-row limit; narrow the query")
        return rows

    def fetch(
        self,
        table: str,
        *,
        tickers: Optional[Iterable[str]] = None,
        filters: Optional[Mapping[str, Any]] = None,
        fields: Optional[Sequence[str]] = None,
        date_range: Optional[Tuple[str, str]] = None,
        date_field: str = "date",
        batch_size: int = DEFAULT_TICKER_BATCH,
    ) -> List[Dict[str, Any]]:
        """All matching rows, split into single-page requests.

        ``tickers=None`` queries the whole table. ``date_range`` is an
        inclusive ``(start, end)`` on ``date_field``; it is the axis used to
        split a truncated request. Without it (or once it is one day wide) the
        ticker batch is halved instead.
        """
        if tickers is None:
            batches: List[Optional[List[str]]] = [None]
        else:
            unique = sorted({t.strip().upper() for t in tickers if t and t.strip()})
            max_count = max(1, min(batch_size, MAX_TICKERS_PER_REQUEST))
            batches = list(_ticker_batches(unique, max_count, MAX_TICKER_PARAM_CHARS))
        rows: List[Dict[str, Any]] = []
        for batch in batches:
            rows.extend(self._fetch_split(table, batch, dict(filters or {}), fields, date_range, date_field))
        return rows

    def get_table_for_tickers(
        self,
        table: str,
        tickers: Iterable[str],
        filters: Optional[Mapping[str, Any]] = None,
        fields: Optional[Sequence[str]] = None,
        batch_size: int = DEFAULT_TICKER_BATCH,
        date_range: Optional[Tuple[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """``fetch`` restricted to ``tickers`` (kept for existing callers)."""
        return self.fetch(
            table, tickers=list(tickers), filters=filters, fields=fields, date_range=date_range, batch_size=batch_size
        )

    def _fetch_split(
        self,
        table: str,
        tickers: Optional[List[str]],
        filters: Dict[str, Any],
        fields: Optional[Sequence[str]],
        date_range: Optional[Tuple[str, str]],
        date_field: str,
    ) -> List[Dict[str, Any]]:
        params = dict(filters)
        if tickers is not None:
            if not tickers:
                return []
            params["ticker"] = ",".join(tickers)
        if date_range is not None:
            params[f"{date_field}.gte"], params[f"{date_field}.lte"] = date_range
        try:
            return self.get_table(table, filters=params, fields=fields)
        except SharadarTruncatedError:
            if date_range is not None:
                start, end = date.fromisoformat(date_range[0]), date.fromisoformat(date_range[1])
                if start < end:
                    mid = start + (end - start) // 2
                    left = (start.isoformat(), mid.isoformat())
                    right = ((mid + timedelta(days=1)).isoformat(), end.isoformat())
                    return self._fetch_split(table, tickers, filters, fields, left, date_field) + self._fetch_split(
                        table, tickers, filters, fields, right, date_field
                    )
            if tickers is not None and len(tickers) > 1:
                half = len(tickers) // 2
                return self._fetch_split(table, tickers[:half], filters, fields, date_range, date_field) + (
                    self._fetch_split(table, tickers[half:], filters, fields, date_range, date_field)
                )
            raise

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
