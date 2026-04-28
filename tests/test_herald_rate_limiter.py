"""Unit tests for Herald's per-domain rate limiter (no network, no real sleeps)."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Stub `requests` so the module can be imported in environments where it is
# not installed (e.g. the isolated pytest venv).
if "requests" not in sys.modules:
    sys.modules["requests"] = MagicMock()

import tools.fetch_company_press_releases as mod


@pytest.fixture(autouse=True)
def _reset_domain_state():
    """Wipe per-domain rate-limit tables between tests."""
    mod._domain_locks.clear()
    mod._domain_last_request.clear()
    yield
    mod._domain_locks.clear()
    mod._domain_last_request.clear()


# ---------------------------------------------------------------------------
# _domain_key
# ---------------------------------------------------------------------------


def test_domain_key_extracts_netloc():
    assert mod._domain_key("https://www.globenewswire.com/search?q=ACME") == "www.globenewswire.com"
    assert mod._domain_key("https://ir.example.com/news") == "ir.example.com"


def test_domain_key_different_paths_same_domain():
    assert mod._domain_key("https://ir.bio.com/q1") == mod._domain_key("https://ir.bio.com/q2")


def test_domain_key_different_subdomains_differ():
    assert mod._domain_key("https://ir.bio.com/") != mod._domain_key("https://www.bio.com/")


# ---------------------------------------------------------------------------
# _rate_limit_domain — same domain must sleep
# ---------------------------------------------------------------------------


def test_same_domain_triggers_sleep_on_second_call():
    """Second call to a recently-fetched domain calls time.sleep with ~RATE_LIMIT_SECONDS."""
    url = "https://ir.acme.com/news"
    key = mod._domain_key(url)

    # Simulate that domain was just fetched (last_request = now)
    mod._domain_locks[key] = threading.Lock()
    mod._domain_last_request[key] = time.monotonic()

    sleep_calls: list[float] = []
    with patch.object(mod.time, "sleep", side_effect=lambda s: sleep_calls.append(s)):
        mod._rate_limit_domain(url)

    assert len(sleep_calls) == 1, "Expected exactly one sleep for a hot domain"
    assert sleep_calls[0] == pytest.approx(mod.RATE_LIMIT_SECONDS, abs=0.05)


def test_cold_domain_does_not_sleep():
    """First-ever call to a domain (last_request=0.0) must not sleep."""
    url = "https://ir.newco.com/news"

    sleep_calls: list[float] = []
    with patch.object(mod.time, "sleep", side_effect=lambda s: sleep_calls.append(s)):
        mod._rate_limit_domain(url)

    assert len(sleep_calls) == 0, "Cold domain should not sleep"


def test_stale_domain_does_not_sleep():
    """Domain whose last_request is >RATE_LIMIT_SECONDS ago must not sleep."""
    url = "https://ir.old.com/news"
    key = mod._domain_key(url)

    mod._domain_locks[key] = threading.Lock()
    mod._domain_last_request[key] = time.monotonic() - (mod.RATE_LIMIT_SECONDS + 0.5)

    sleep_calls: list[float] = []
    with patch.object(mod.time, "sleep", side_effect=lambda s: sleep_calls.append(s)):
        mod._rate_limit_domain(url)

    assert len(sleep_calls) == 0, "Domain with elapsed > RATE_LIMIT_SECONDS should not sleep"


# ---------------------------------------------------------------------------
# _rate_limit_domain — different domains must NOT block each other
# ---------------------------------------------------------------------------


def test_different_domains_are_independent():
    """A hot domain A must not cause sleep when domain B is called."""
    url_a = "https://www.globenewswire.com/search?q=ACME"
    url_b = "https://www.businesswire.com/search?q=BETA"

    # Prime domain A as just-fetched
    key_a = mod._domain_key(url_a)
    mod._domain_locks[key_a] = threading.Lock()
    mod._domain_last_request[key_a] = time.monotonic()

    # Domain B is cold — calling it should not sleep due to A's window
    sleep_calls: list[float] = []
    with patch.object(mod.time, "sleep", side_effect=lambda s: sleep_calls.append(s)):
        mod._rate_limit_domain(url_b)

    assert len(sleep_calls) == 0, "Domain B must not be gated by domain A's rate limit"


def test_two_domains_initialized_independently():
    """After first calls, each domain has its own last_request entry."""
    url_a = "https://host-a.com/news"
    url_b = "https://host-b.com/news"

    with patch.object(mod.time, "sleep"):
        mod._rate_limit_domain(url_a)
        mod._rate_limit_domain(url_b)

    assert mod._domain_key(url_a) in mod._domain_last_request
    assert mod._domain_key(url_b) in mod._domain_last_request
    assert mod._domain_key(url_a) != mod._domain_key(url_b)


# ---------------------------------------------------------------------------
# Thread-safety: two threads on the same domain are serialised
# ---------------------------------------------------------------------------


def test_concurrent_same_domain_calls_are_serialised():
    """Two threads hitting the same domain must not overlap inside the domain lock."""
    url = "https://ir.concurrent.com/news"
    order: list[str] = []

    real_sleep = time.sleep

    def tracked_sleep(s: float) -> None:
        order.append("sleep_start")
        # Use a tiny real sleep so thread scheduling can interleave if lock is broken
        real_sleep(min(s, 0.05))
        order.append("sleep_end")

    with patch.object(mod.time, "sleep", side_effect=tracked_sleep):
        t1 = threading.Thread(target=mod._rate_limit_domain, args=(url,))
        t2 = threading.Thread(target=mod._rate_limit_domain, args=(url,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    # Exactly one sleep should have occurred (the second thread waits, first is cold)
    sleep_starts = order.count("sleep_start")
    assert sleep_starts == 1, f"Expected 1 sleep for serialised same-domain calls, got {sleep_starts}"
