#!/usr/bin/env python3
"""Rate-limit safe yfinance wrapper with exponential backoff.

Handles 429 (Too Many Requests) errors from Yahoo Finance API by implementing:
- Per-ticker delays (configurable, default 1.5s)
- Exponential backoff on rate-limit errors
- Automatic retry with jitter
- Logging and telemetry

Usage:
    from scripts.yfinance_safe import safe_download

    data = safe_download(
        ['AAPL', 'MSFT'],
        start='2026-05-26',
        end='2026-05-27',
        delay_sec=1.5,
        max_retries=3
    )
"""

from __future__ import annotations

import logging
import os
import random
import time

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# curl_cffi's Chrome impersonation uses BoringSSL which cannot trust a TLS-intercepting
# proxy's CA bundle. When running behind HTTPS_PROXY (cloud/CI environments), patch
# yfinance's TickerBase to use a plain curl_cffi session without impersonation so that
# the proxy CA bundle (REQUESTS_CA_BUNDLE / SSL_CERT_FILE) is respected.
if os.environ.get("HTTPS_PROXY"):
    try:
        import yfinance.base as _yfbase
        from curl_cffi import requests as _cffi_requests

        _orig_ticker_init = _yfbase.TickerBase.__init__

        import inspect as _inspect
        _orig_params = set(_inspect.signature(_orig_ticker_init).parameters)

        def _proxy_safe_ticker_init(self, ticker, session=None, proxy=None):
            if session is None:
                session = _cffi_requests.Session()
            kw = {"session": session}
            if "proxy" in _orig_params:
                kw["proxy"] = proxy
            _orig_ticker_init(self, ticker, **kw)

        _yfbase.TickerBase.__init__ = _proxy_safe_ticker_init
    except Exception:
        pass  # non-fatal — fall back to default behaviour


def safe_download(
    tickers: list[str],
    start: str,
    end: str,
    delay_sec: float = 1.5,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    jitter: bool = True,
) -> dict:
    """Download price data with rate-limit protection.

    Args:
        tickers: List of ticker symbols
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        delay_sec: Seconds to wait between ticker requests (default 1.5s)
        max_retries: Max retry attempts on rate-limit error (default 3)
        backoff_factor: Exponential backoff multiplier (default 2.0x)
        jitter: Add random jitter to delays (default True)

    Returns:
        Dict with keys:
            'data': Downloaded DataFrame (or empty if all failed)
            'failed_tickers': List of tickers that failed after all retries
            'rate_limit_hits': Count of 429 errors encountered
            'total_requests': Total API calls made
    """

    results = {
        "data": None,
        "failed_tickers": [],
        "rate_limit_hits": 0,
        "total_requests": 0,
    }

    if not tickers:
        return results

    logger.info(f"safe_download: {len(tickers)} tickers, " f"delay={delay_sec}s, max_retries={max_retries}")

    # Attempt batch download with retry logic
    retry_count = 0
    current_delay = delay_sec

    while retry_count < max_retries:
        try:
            logger.debug(f"Attempt {retry_count + 1}/{max_retries}")

            # Add delay before request (except first attempt)
            if retry_count > 0:
                actual_delay = current_delay
                if jitter:
                    actual_delay *= 0.5 + random.random()  # ±50% jitter
                logger.info(f"Rate-limit backoff: waiting {actual_delay:.1f}s " f"before retry {retry_count + 1}")
                time.sleep(actual_delay)
                current_delay *= backoff_factor

            # Attempt download
            results["total_requests"] += 1
            data = yf.download(
                tickers,
                start=start,
                end=end,
                progress=False,
                threads=False,  # Disable threads to avoid hammering API
            )

            if data is not None and not data.empty:
                results["data"] = data
                logger.info(f"✓ Download successful after {retry_count + 1} attempt(s)")
                return results
            else:
                logger.warning("Download returned empty data, will retry")
                retry_count += 1

        except Exception as e:
            error_str = str(e)

            # Check if it's a rate-limit error
            if "429" in error_str or "Too Many Requests" in error_str:
                results["rate_limit_hits"] += 1
                logger.warning(f"Rate-limit hit (429) on attempt {retry_count + 1}. " f"Will retry with backoff.")
                retry_count += 1

            elif "Expecting value" in error_str:
                # This is likely a malformed response from rate-limit block
                results["rate_limit_hits"] += 1
                logger.warning(
                    f"Malformed JSON response (likely rate-limit block) "
                    f"on attempt {retry_count + 1}. Will retry with backoff."
                )
                retry_count += 1

            else:
                # Other error - log and fail
                logger.error(f"Unrecoverable error: {error_str[:200]}")
                results["failed_tickers"] = tickers
                return results

    # All retries exhausted
    logger.error(f"All {max_retries} retry attempts failed. " f"Rate-limit hits: {results['rate_limit_hits']}")
    results["failed_tickers"] = tickers
    return results


def safe_download_per_ticker(
    tickers: list[str],
    start: str,
    end: str,
    delay_sec: float = 1.5,
    max_retries: int = 3,
) -> dict:
    """Download price data one ticker at a time with per-ticker delays.

    More conservative approach than batch download - fetches each ticker
    individually with enforced delays. Slower but more resilient to rate limits.

    Args:
        tickers: List of ticker symbols
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        delay_sec: Seconds to wait between ticker requests
        max_retries: Max retry attempts per ticker

    Returns:
        Dict with keys:
            'data': Combined DataFrame from all successful tickers
            'failed_tickers': List of tickers that failed
            'successful_tickers': Count of successfully fetched tickers
            'rate_limit_hits': Total 429 errors encountered
    """

    results = {
        "data": None,
        "failed_tickers": [],
        "successful_tickers": 0,
        "rate_limit_hits": 0,
    }

    all_data = []

    for i, ticker in enumerate(tickers):
        # Add delay before each request (except first)
        if i > 0:
            jittered_delay = delay_sec * (0.5 + random.random())
            logger.debug(f"Waiting {jittered_delay:.2f}s before {ticker}")
            time.sleep(jittered_delay)

        # Attempt to fetch this ticker
        retry_count = 0
        current_delay = delay_sec
        success = False

        while retry_count < max_retries and not success:
            try:
                logger.debug(f"Fetching {ticker} (attempt {retry_count + 1})")

                data = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    progress=False,
                )

                if data is not None and not data.empty:
                    # Modern yfinance returns MultiIndex columns even for a
                    # single ticker, e.g. ('Close', 'AARD'). Flatten to the
                    # field level so downstream row.get('Close')/row.get('ticker')
                    # yield scalars, not Series (whose repr would otherwise be
                    # written as the ticker value — see extend_price_csv_safe).
                    if isinstance(data.columns, pd.MultiIndex):
                        data = data.copy()
                        data.columns = data.columns.get_level_values(0)
                    # Add ticker column if missing
                    if "ticker" not in data.columns:
                        data["ticker"] = ticker
                    all_data.append(data)
                    results["successful_tickers"] += 1
                    success = True
                    logger.debug(f"✓ {ticker}: {len(data)} rows")
                else:
                    logger.warning(f"✗ {ticker}: empty response")
                    retry_count += 1

            except Exception as e:
                error_str = str(e)

                if "429" in error_str or "Too Many Requests" in error_str or "Expecting value" in error_str:
                    results["rate_limit_hits"] += 1
                    logger.warning(f"✗ {ticker}: rate-limit hit (attempt {retry_count + 1})")
                    retry_count += 1
                    if retry_count < max_retries:
                        backoff = current_delay * (1.5**retry_count)
                        jittered = backoff * (0.5 + random.random())
                        logger.info(f"  Backoff: {jittered:.1f}s before retry")
                        time.sleep(jittered)
                else:
                    logger.error(f"✗ {ticker}: {error_str[:100]}")
                    break

        if not success:
            results["failed_tickers"].append(ticker)

    # Combine all data
    if all_data:
        results["data"] = pd.concat(all_data, ignore_index=False)
        logger.info(f"Combined data: {len(all_data)} tickers, " f"{len(results['data'])} total rows")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Test with small universe
    print("Testing with 3 tickers...")
    result = safe_download_per_ticker(
        ["AAPL", "MSFT", "GOOG"],
        start="2026-05-20",
        end="2026-05-26",
        delay_sec=2.0,
        max_retries=3,
    )

    print("\nResults:")
    print(f"  Successful: {result['successful_tickers']}")
    print(f"  Failed: {result['failed_tickers']}")
    print(f"  Rate-limit hits: {result['rate_limit_hits']}")
    if result["data"] is not None:
        print(f"  Data shape: {result['data'].shape}")
