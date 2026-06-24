#!/usr/bin/env python3
"""Grok classification layer for company press releases (Spec 044 Phase 2).

Takes raw PR records from fetch_company_press_releases.py and classifies
them into typed event records using xAI's structured outputs.

Usage:
    python tools/classify_press_releases.py --input data/press_releases/releases_2026-03-31.jsonl
    python tools/classify_press_releases.py --input data/press_releases/releases_2026-03-31.jsonl --dry-run
    python tools/classify_press_releases.py --input data/press_releases/releases_2026-03-31.jsonl --stdout-only
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env for API keys
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "dem_grok_news_feed.v1"
OUTPUT_DIR = PROJECT_ROOT / "data" / "press_releases" / "classified"

# Try to import OpenAI client for xAI
try:
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


SYSTEM_PROMPT = """You are a point-in-time biotech event normalizer for the Wake Robin DEM/CRT pipeline.

Your job: classify a company press release into a structured event record.

Rules:
1. Separate event_outcome_guess from price_direction_guess. A CRL is a MISS even if the stock goes up.
2. Mark informational_only=true for non-catalyst updates: earnings schedules, conference appearances, enrollment updates with no data, generic business updates.
3. Tag exogenous events (M&A, unrelated corporate actions) that are not the tracked catalyst.
4. Be conservative with confidence — 0.5 if uncertain, 0.9+ only for definitive events with official sources.

Event categories: mna, clinical, regulatory, financing, leadership, safety, legal, competitor, sector, other
Severity: critical (definitive M&A/approval/CRL/pivotal topline), high (thesis-changing), medium (meaningful), low (minor)
Outcome: hit, miss, mixed, unclear, not_applicable
Price direction: up, down, flat, unclear, not_applicable

Output JSON only. No commentary."""

USER_PROMPT_TEMPLATE = """Classify this press release:

Ticker: {ticker}
Company: {company}
Headline: {headline}
Source URL: {source_url}
Published: {published_at}

Return a JSON object with these fields:
- event_category (string)
- event_subtype (string, e.g. "phase3_topline", "earnings", "conference_presentation")
- severity (string)
- materiality (string: high/medium/low)
- new_or_stale (string: new/follow_on/stale)
- informational_only (boolean)
- informational_reason (string, empty if not informational)
- event_outcome_guess (string)
- event_outcome_reason (string)
- price_direction_guess (string)
- price_direction_reason (string)
- exogenous_to_primary_catalyst (boolean)
- exogenous_reason (string)
- safety_signal_flag (boolean)
- financing_signal_flag (boolean)
- mna_signal_flag (boolean)
- thesis_change_flag (boolean)
- why_it_matters (string, 1-2 sentences)
- confidence (float, 0-1)
- needs_review (boolean)
"""


def _classify_with_grok(
    ticker: str,
    company: str,
    headline: str,
    source_url: str,
    published_at: str,
    client: Any,
    model: str,
) -> Dict[str, Any]:
    """Classify a single PR using xAI Grok."""
    user_prompt = USER_PROMPT_TEMPLATE.format(
        ticker=ticker,
        company=company,
        headline=headline,
        source_url=source_url,
        published_at=published_at,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=800,
        response_format={"type": "json_object"},
    )

    text = response.choices[0].message.content
    return json.loads(text)


def _classify_locally(headline: str) -> Dict[str, Any]:
    """Fast local classification using keyword rules (no API call).

    Used as fallback when XAI_API_KEY is not set or for dry-run mode.
    """
    hl = headline.lower()

    # GUARD: Clinical event classification takes priority over informational suppression.
    # Phase 3 data presentations, ASCO readouts, etc. are substantive catalysts even if
    # they mention "presentation" or "conference" (Spec remediation 2026-06-02).
    clinical_guard_keywords = [
        "phase 3",
        "phase 2",
        "phase 1",
        "clinical data",
        "data readout",
        "asco",
        "clinical trial",
    ]
    is_clinical_guard_match = any(kw in hl for kw in clinical_guard_keywords)

    # Informational patterns (skip if clinical guard matched)
    informational_keywords = [
        "financial results",
        "quarterly report",
        "conference",
        "investor",
        "presentation",
        "schedules release",
        "participate in",
        "upcoming",
        "appoints",
        "board of directors",
    ]
    if any(kw in hl for kw in informational_keywords) and not is_clinical_guard_match:
        return {
            "event_category": "other",
            "event_subtype": "corporate_update",
            "severity": "low",
            "materiality": "low",
            "new_or_stale": "new",
            "informational_only": True,
            "informational_reason": "Corporate/financial update",
            "event_outcome_guess": "not_applicable",
            "event_outcome_reason": "",
            "price_direction_guess": "not_applicable",
            "price_direction_reason": "",
            "exogenous_to_primary_catalyst": False,
            "exogenous_reason": "",
            "safety_signal_flag": False,
            "financing_signal_flag": False,
            "mna_signal_flag": False,
            "thesis_change_flag": False,
            "why_it_matters": "Informational corporate update",
            "confidence": 0.7,
            "needs_review": False,
        }

    # Safety signal patterns (checked before clinical — "clinical hold on Phase 2"
    # should classify as safety, not clinical)
    safety_keywords = [
        "clinical hold",
        "partial hold",
        "adverse event",
        "serious adverse",
        "death",
        "fatality",
        "safety signal",
        "voluntary recall",
        "dose-limiting",
        "dose limiting",
        "hepatotoxicity",
        "black box warning",
    ]
    # Negation patterns that reverse safety interpretation (Spec 053)
    safety_negation = [
        "lifts clinical hold",
        "lift clinical hold",
        "lifted clinical hold",
        "clinical hold lifted",
        "clinical hold has been lifted",
        "removes clinical hold",
        "removal of clinical hold",
        "clinical hold removed",
        "resolves clinical hold",
        "hold has been resolved",
        "partial hold lifted",
        "partial hold removed",
        "removes partial clinical hold",
    ]
    has_safety = any(kw in hl for kw in safety_keywords)
    has_negation = any(neg in hl for neg in safety_negation)
    if has_safety and has_negation:
        # Safety resolution — positive regulatory event
        return {
            "event_category": "regulatory",
            "event_subtype": "safety_resolution",
            "severity": "high",
            "materiality": "high",
            "new_or_stale": "new",
            "informational_only": False,
            "informational_reason": "",
            "event_outcome_guess": "hit",
            "event_outcome_reason": headline[:100],
            "price_direction_guess": "up",
            "price_direction_reason": "Clinical hold lift is positive for stock",
            "exogenous_to_primary_catalyst": False,
            "exogenous_reason": "",
            "safety_signal_flag": False,
            "financing_signal_flag": False,
            "mna_signal_flag": False,
            "thesis_change_flag": True,
            "why_it_matters": headline[:150],
            "confidence": 0.7,
            "needs_review": False,
        }
    if has_safety:
        return {
            "event_category": "safety",
            "event_subtype": "safety_signal",
            "severity": "critical",
            "materiality": "high",
            "new_or_stale": "new",
            "informational_only": False,
            "informational_reason": "",
            "event_outcome_guess": "miss",
            "event_outcome_reason": headline[:100],
            "price_direction_guess": "down",
            "price_direction_reason": "Safety events typically negative for stock",
            "exogenous_to_primary_catalyst": False,
            "exogenous_reason": "",
            "safety_signal_flag": True,
            "financing_signal_flag": False,
            "mna_signal_flag": False,
            "thesis_change_flag": True,
            "why_it_matters": headline[:150],
            "confidence": 0.6,
            "needs_review": True,
        }

    # Clinical patterns
    clinical_keywords = [
        "phase 3",
        "phase 2",
        "phase 1",
        "topline",
        "primary endpoint",
        "readout",
        "clinical data",
        "trial results",
        "pivotal",
    ]
    if any(kw in hl for kw in clinical_keywords):
        is_positive = any(kw in hl for kw in ["positive", "met", "significant", "approved"])
        is_negative = any(kw in hl for kw in ["did not meet", "failed", "discontinued", "hold"])
        return {
            "event_category": "clinical",
            "event_subtype": "clinical_data",
            "severity": "high",
            "materiality": "high",
            "new_or_stale": "new",
            "informational_only": False,
            "informational_reason": "",
            "event_outcome_guess": "hit" if is_positive else "miss" if is_negative else "unclear",
            "event_outcome_reason": headline[:100],
            "price_direction_guess": "up" if is_positive else "down" if is_negative else "unclear",
            "price_direction_reason": "",
            "exogenous_to_primary_catalyst": False,
            "exogenous_reason": "",
            "safety_signal_flag": False,
            "financing_signal_flag": False,
            "mna_signal_flag": False,
            "thesis_change_flag": True,
            "why_it_matters": headline[:150],
            "confidence": 0.6,
            "needs_review": True,
        }

    # Regulatory patterns
    regulatory_keywords = [
        "fda",
        "approval",
        "nda",
        "bla",
        "pdufa",
        "breakthrough",
        "priority review",
        "complete response",
        "crl",
    ]
    if any(kw in hl for kw in regulatory_keywords):
        return {
            "event_category": "regulatory",
            "event_subtype": "regulatory_update",
            "severity": "high",
            "materiality": "high",
            "new_or_stale": "new",
            "informational_only": False,
            "informational_reason": "",
            "event_outcome_guess": "unclear",
            "event_outcome_reason": headline[:100],
            "price_direction_guess": "unclear",
            "price_direction_reason": "",
            "exogenous_to_primary_catalyst": False,
            "exogenous_reason": "",
            "safety_signal_flag": False,
            "financing_signal_flag": False,
            "mna_signal_flag": False,
            "thesis_change_flag": True,
            "why_it_matters": headline[:150],
            "confidence": 0.5,
            "needs_review": True,
        }

    # M&A patterns
    mna_keywords = [
        "acquire",
        "acquisition",
        "merger",
        "merge",
        "buyout",
        "takeover",
        "tender offer",
        "definitive agreement",
        "strategic combination",
        "to be acquired",
    ]
    if any(kw in hl for kw in mna_keywords):
        return {
            "event_category": "mna",
            "event_subtype": "mna_announcement",
            "severity": "critical",
            "materiality": "high",
            "new_or_stale": "new",
            "informational_only": False,
            "informational_reason": "",
            "event_outcome_guess": "unclear",
            "event_outcome_reason": headline[:100],
            "price_direction_guess": "unclear",
            "price_direction_reason": "",
            "exogenous_to_primary_catalyst": True,
            "exogenous_reason": "M&A event supersedes catalyst thesis",
            "safety_signal_flag": False,
            "financing_signal_flag": False,
            "mna_signal_flag": True,
            "thesis_change_flag": True,
            "why_it_matters": headline[:150],
            "confidence": 0.7,
            "needs_review": True,
        }

    # Financing patterns
    financing_keywords = [
        "public offering",
        "private placement",
        "registered direct",
        "at-the-market",
        "pricing of",
        "priced its",
        "raises $",
        "million offering",
        "billion offering",
        "common stock offering",
        "secondary offering",
        "shelf registration",
        "underwritten",
    ]
    if any(kw in hl for kw in financing_keywords):
        return {
            "event_category": "financing",
            "event_subtype": "capital_raise",
            "severity": "medium",
            "materiality": "medium",
            "new_or_stale": "new",
            "informational_only": False,
            "informational_reason": "",
            "event_outcome_guess": "not_applicable",
            "event_outcome_reason": "",
            "price_direction_guess": "down",
            "price_direction_reason": "Dilutive financing typically pressures stock",
            "exogenous_to_primary_catalyst": False,
            "exogenous_reason": "",
            "safety_signal_flag": False,
            "financing_signal_flag": True,
            "mna_signal_flag": False,
            "thesis_change_flag": False,
            "why_it_matters": headline[:150],
            "confidence": 0.6,
            "needs_review": False,
        }

    # Default
    return {
        "event_category": "other",
        "event_subtype": "unclassified",
        "severity": "low",
        "materiality": "low",
        "new_or_stale": "new",
        "informational_only": False,
        "informational_reason": "",
        "event_outcome_guess": "unclear",
        "event_outcome_reason": "",
        "price_direction_guess": "unclear",
        "price_direction_reason": "",
        "exogenous_to_primary_catalyst": False,
        "exogenous_reason": "",
        "safety_signal_flag": False,
        "financing_signal_flag": False,
        "mna_signal_flag": False,
        "thesis_change_flag": False,
        "why_it_matters": headline[:150],
        "confidence": 0.3,
        "needs_review": True,
    }


def _is_noise(headline: str) -> bool:
    """Detect non-company PR noise from GlobeNewswire (lawsuits, analyst notes, market reports).

    CH-1: decode HTML entities before matching so patterns like "levi & korsinsky"
    catch headlines that store the ampersand as `&amp;`.
    """
    hl = html.unescape(headline).lower()
    noise_patterns = [
        # Legal noise
        "investor alert",
        "pomerantz law",
        "johnson fistel",
        "class action",
        "securities fraud",
        "shareholders are encouraged",
        "investigation on behalf",
        "reminds investors",
        "reminds shareholders",
        "deadline reminder",
        "loss notice",
        "stock alert",
        "buy or sell",
        "hagens berman",
        "bernstein liebhard",
        "schall law",
        "rosen law",
        "glancy prongay",
        "kessler topaz",
        "levi & korsinsky",
        "faruqi & faruqi",
        "bronstein, gewirtz",
        # Market reports / industry analysis (ticker collision noise)
        "market size",
        "market volume",
        "market worth",
        "market forecast",
        "market report",
        "market analysis",
        "market growth",
        "industry report",
        "industry analysis",
        "market research",
        "research report",
        "global market",
        "regional market",
        "market trends",
        "market share",
        "competitive landscape",
        "swot analysis",
        "joint venture terms",
        "joint venture agreements",
        "partnership agreements analysis",
        # Job postings
        "employment inducement",
        "inducement grants",
        # Market research / analyst expanded (Spec 053)
        "market valuation",
        "market opportunity",
        "market outlook",
        "market segmentation",
        "market dynamics",
        "market overview",
        "cagr",
        "forecast period",
        "billion by 20",
        "million by 20",
        "price target",
        "initiates coverage",
        "maintains buy",
        "maintains sell",
        "maintains hold",
        "analyst report",
        "equity research",
        "masterbatch",
        "downgrades to",
        "upgrades to",
        "value chain analysis",
        "supply chain analysis",
        # CH-5 additions (2026-04-18) — evidence-driven from fingpt_pilot seed audit
        "halper sadeh",
        "investment opportunities in",
        "billion valuation by 20",
        "million valuation by 20",
        "clearance of ai-powered",
    ]
    return any(p in hl for p in noise_patterns)


def _load_company_names() -> Dict[str, List[str]]:
    """Load company name keywords from the IR source registry.

    Returns {TICKER: [significant_words]} where significant words are
    lowercase, length > 3, stripped of common suffixes (Inc, Corp, etc).
    """
    sources_path = PROJECT_ROOT / "production_data" / "company_ir_sources.json"
    if not sources_path.exists():
        return {}
    try:
        data = json.loads(sources_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}

    stop_words = frozenset(
        {
            "inc",
            "inc.",
            "corp",
            "corp.",
            "corporation",
            "ltd",
            "ltd.",
            "plc",
            "llc",
            "sa",
            "nv",
            "n.v.",
            "ag",
            "se",
            "therapeutics",
            "pharmaceuticals",
            "biosciences",
            "biotech",
            "biopharma",
            "holdings",
            "holding",
            "company",
            "group",
            "the",
        }
    )

    result: Dict[str, List[str]] = {}
    for entry in data.get("sources", []):
        ticker = entry.get("ticker", "").upper()
        company = entry.get("company", "")
        if not ticker or not company:
            continue
        tokens = company.replace(",", " ").split()
        # CH-3: always retain the first non-stop-word token regardless of length
        # (e.g., "SAB" from "SAB Biotherapeutics"). Other tokens still require > 3 chars.
        # This prevents short discriminative prefixes from being dropped when
        # longer tokens like "Biotherapeutics" survive the stop-word filter.
        words: List[str] = []
        first_kept = False
        for w in tokens:
            clean = w.lower().rstrip(".,")
            if not clean or clean in stop_words:
                continue
            if not first_kept:
                words.append(clean)
                first_kept = True
            elif len(clean) > 3:
                words.append(clean)
        if words:
            result[ticker] = words
    return result


def _is_ticker_collision(headline: str, ticker: str, company_names: Dict[str, List[str]]) -> bool:
    """Check if a headline likely belongs to a different company than the ticker.

    Returns True (collision) only if ALL of these hold:
      1. None of the company's significant name words appear in the headline
      2. The ticker symbol itself doesn't appear as a word in the headline
      3. The headline doesn't come from a company_ir source (those are pre-filtered by ticker)

    Conservative: prefers false negatives (letting a collision through) over
    false positives (suppressing a real company event).

    CH-1: HTML entities decoded before matching.
    CH-2: missing registry entry no longer short-circuits to "no collision"; the
    ticker-match and biotech-rescue checks still run so that headlines with neither
    the ticker symbol nor biotech context get flagged.
    """
    # CH-2: empty list (rather than None short-circuit) — continues through
    # the ticker-word + biotech-rescue checks below.
    words = company_names.get(ticker) or []

    # CH-1: decode HTML entities (`&amp;`, `&#174;`, `&#231;`) before lowercasing.
    hl_lower = html.unescape(headline).lower()

    # Check 1: any company name word in headline
    if words and any(w in hl_lower for w in words):
        return False

    # Check 2: ticker symbol appears as a standalone word (not substring of another word)
    # e.g., "BIIB Reports..." matches, but "Academy" containing "ACAD" does not
    import re as _re

    ticker_lower = ticker.lower()
    if _re.search(r"\b" + _re.escape(ticker_lower) + r"\b", hl_lower):
        return False

    # Check 3 (Spec 053, tightened by CH-4 2026-04-18): No company name, no ticker
    # — assume collision UNLESS headline contains ≥2 discriminative biotech terms.
    # Previously a single ≥1 match rescued the item, but generic tokens like
    # "approved", "drug", "patient", "regulatory" fire on non-biotech contexts
    # (device-SaaS clearances, banking "patient services", generic regulatory
    # filings), yielding false rescues. CH-4 removes those four generic tokens
    # from the match set and requires ≥2 distinct matches.
    matches = _count_biotech_indicator_matches(hl_lower)
    if matches >= _BIOTECH_RESCUE_MIN_MATCHES:
        return False  # likely a real biotech PR with unusual branding

    # No company name, no ticker, <2 biotech terms — likely a collision
    return True


# ---------------------------------------------------------------------------
# CH-4: biotech-rescue indicator set + match count
# ---------------------------------------------------------------------------

# Indicators that meaningfully signal a biotech PR. Generic tokens
# ("approved", "approval", "drug", "patient", "regulatory") were removed in CH-4
# because they match non-biotech contexts often enough to create false rescues.
# All entries must be lowercase; matching is substring against `hl_lower`.
_BIOTECH_RESCUE_INDICATORS = frozenset(
    [
        "phase",
        "trial",
        "fda",
        "clinical",
        "therapy",
        "therapeutic",
        "oncology",
        "dose",
        "efficacy",
        "endpoint",
        "enrollment",
        "pipeline",
        "nda",
        "bla",
        "biologic",
        "molecule",
        "antibody",
        "gene",
        "cell therapy",
        "mrna",
        "protein",
        "receptor",
        "inhibitor",
        "immuno",
        "orphan",
        "pdufa",
        "ema",
        "preclinical",
        "ind ",
        "snda",
        "sbla",
        "designation",
        "breakthrough",
    ]
)

# Minimum number of distinct discriminative indicators that must appear in
# a headline for the biotech-rescue to fire. Raised from 1 to 2 in CH-4.
_BIOTECH_RESCUE_MIN_MATCHES = 2

# CH-4 shadow: counterfactual rule kept for analysis-only comparison.
# Do NOT use in production — this is the legacy behavior (any single match
# rescues) applied to the discriminative indicator set.
_BIOTECH_RESCUE_COUNTERFACTUAL_MIN = 1


def _count_biotech_indicator_matches(hl_lower: str) -> int:
    """Count distinct discriminative biotech indicators present in the headline.

    Used by `_is_ticker_collision()` in production (threshold = 2) and by
    shadow-logging callers (counterfactual threshold = 1 for comparison).
    """
    return sum(1 for ind in _BIOTECH_RESCUE_INDICATORS if ind in hl_lower)


def _collision_counterfactual_would_rescue(hl_lower: str) -> bool:
    """CH-4 shadow-logging hook: would the counterfactual ≥1-match rule rescue
    this headline from being flagged as a collision? Analysis only."""
    return _count_biotech_indicator_matches(hl_lower) >= _BIOTECH_RESCUE_COUNTERFACTUAL_MIN


def classify_releases(
    raw_records: List[Dict[str, Any]],
    use_grok: bool = False,
    model: str = "grok-3-mini-fast",
) -> List[Dict[str, Any]]:
    """Classify a batch of raw PR records.

    Tiered strategy:
    1. Filter noise (lawsuits, stock alerts) — skip entirely
    1.5. Flag ticker collisions (headline doesn't match registered company)
    2. Local keyword classification for clear cases
    3. Grok only for ambiguous records (if enabled)
    """
    client = None
    if use_grok and HAS_OPENAI:
        api_key = os.getenv("XAI_API_KEY")
        if api_key:
            client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        else:
            logger.warning("XAI_API_KEY not set, falling back to local classification")

    company_names = _load_company_names()

    classified = []
    noise_skipped = 0
    collision_flagged = 0
    collision_soft_flagged = 0
    grok_calls = 0

    for rec in raw_records:
        ticker = rec.get("ticker", "")
        headline = rec.get("headline", "")
        if not ticker or not headline:
            continue

        # Tier 1: skip noise
        if _is_noise(headline):
            noise_skipped += 1
            continue

        # Tier 1.5: detect ticker collisions
        is_collision = _is_ticker_collision(headline, ticker.upper(), company_names)
        if is_collision:
            collision_flagged += 1

        # Tier 2: local classification
        result = _classify_locally(headline)

        # If collision detected, route by severity (P2, 2026-04-18):
        # - HARD collision (match_count == 0): suppress to informational (silent drop).
        # - SOFT collision (match_count >= 1, only flagged because CH-4 requires >=2):
        #   flag but do NOT suppress — stays visible in the escalation pool so Grok /
        #   human review can adjudicate. Prevents CH-4 from silencing real biotech
        #   edge cases like licensee-milestone approvals or single-term efficacy PRs.
        if is_collision and not result.get("informational_only"):
            result["ticker_collision_flag"] = True
            hl_lower = html.unescape(headline).lower()
            biotech_match_count = _count_biotech_indicator_matches(hl_lower)
            if biotech_match_count >= _BIOTECH_RESCUE_COUNTERFACTUAL_MIN:
                # Soft: partial biotech signal. Cap confidence but leave visible.
                result["collision_severity"] = "soft"
                result["confidence"] = min(result.get("confidence", 0.3), 0.4)
                collision_soft_flagged += 1
            else:
                # Hard: no biotech signal. Silent-drop as before.
                result["collision_severity"] = "hard"
                result["confidence"] = min(result.get("confidence", 0.3), 0.2)
                result["informational_only"] = True
                result["informational_reason"] = f"ticker_collision: headline does not match {ticker} company name"
        else:
            result["ticker_collision_flag"] = False
            result["collision_severity"] = "none"

        # Tier 3: escalate to Grok if ambiguous and Grok is available
        if client and result.get("needs_review") and not result.get("informational_only"):
            try:
                result = _classify_with_grok(
                    ticker=ticker,
                    company=rec.get("company", ""),
                    headline=headline,
                    source_url=rec.get("source_url", ""),
                    published_at=rec.get("published_at_utc", ""),
                    client=client,
                    model=model,
                )
                grok_calls += 1
            except Exception as e:
                logger.warning("Grok failed for %s: %s", ticker, e)

        # Build normalized event record
        event_id = str(uuid.uuid4())
        dedupe_raw = f"{ticker}|{result.get('event_category', '')}|{result.get('event_subtype', '')}|{rec.get('published_at_utc', '')}|{rec.get('source_url', '')}"
        dk = hashlib.sha256(dedupe_raw.encode()).hexdigest()[:16]

        normalized = {
            "event_id": event_id,
            "dedupe_key": dk,
            "ticker": ticker,
            "company": rec.get("company", ""),
            "headline": headline,
            "source_url": rec.get("source_url", ""),
            "source_type": rec.get("source_type", "company_ir"),
            "published_at_utc": rec.get("published_at_utc", ""),
            "classified_at_utc": datetime.now(timezone.utc).isoformat(),
            "classification_method": (
                "grok" if (client and grok_calls > 0 and result.get("needs_review") is False) else "local_keywords"
            ),
            **result,
        }
        classified.append(normalized)

    if noise_skipped:
        logger.info("Noise filtered: %d records (lawsuits, stock alerts)", noise_skipped)
    if collision_flagged:
        logger.info(
            "Ticker collisions flagged: %d records (%d soft → escalate, %d hard → silent drop)",
            collision_flagged,
            collision_soft_flagged,
            collision_flagged - collision_soft_flagged,
        )
    if grok_calls:
        logger.info("Grok API calls: %d (ambiguous records only)", grok_calls)

    return classified


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="PR classification (Spec 044 Phase 2)")
    parser.add_argument("--input", type=Path, required=True, help="Path to raw releases JSONL")
    parser.add_argument("--use-grok", action="store_true", help="Use xAI Grok API (requires XAI_API_KEY)")
    parser.add_argument("--model", default="grok-4-1-fast")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stdout-only", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}")
        return 1

    # Load raw records
    records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} raw PR records")

    # Classify
    classified = classify_releases(records, use_grok=args.use_grok, model=args.model)
    print(f"Classified {len(classified)} records")

    # Summary
    from collections import Counter

    cats = Counter(r.get("event_category", "?") for r in classified)
    info = sum(1 for r in classified if r.get("informational_only"))
    review = sum(1 for r in classified if r.get("needs_review"))
    print(f"\nCategories: {dict(cats)}")
    print(f"Informational: {info}/{len(classified)}")
    print(f"Needs review: {review}/{len(classified)}")

    if args.stdout_only or args.dry_run:
        for r in classified:
            print(json.dumps(r, indent=2, default=str))
        return 0

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem
    if stem.startswith("releases_"):
        out_stem = stem.replace("releases_", "classified_", 1)
    elif stem.startswith("deduped_"):
        out_stem = stem.replace("deduped_", "classified_", 1)
    else:
        out_stem = f"classified_{stem}"
    out_path = OUTPUT_DIR / f"{out_stem}.jsonl"
    with open(out_path, "w") as f:
        for r in classified:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"Wrote {len(classified)} classified records to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
