# Catalyst Source Design Note: Federal Register Deprecation + Replacement Plan

**Date:** 2026-02-14
**Context:** Catalyst coverage expansion (SEC multi-form + Federal Register)

## Finding: Federal Register is Low-Yield for Drug Decisions

The Federal Register API (`federalregister.gov/api/v1/`) was integrated to capture FDA regulatory notices (approvals, CRLs, RTFs, warning letters). After live testing against the full universe (353 tickers, 28 product-to-ticker mappings):

**Result: 0 drug-specific events returned.**

The Federal Register publishes:
- Agency information collection activities
- Fee rate announcements (PDUFA)
- Pilot program announcements
- Rulemaking/guidance documents

It does NOT publish individual drug approval decisions, CRLs, or RTFs. Those appear on:
- **Drugs@FDA** (fda.gov/drugs/drug-approvals-and-databases)
- **CDER approval reports** (structured, per-NDA/BLA)
- **FDA press releases** (semi-structured)

### Action Taken

FEDERAL_REGISTER demoted from priority=1 to priority=3 in the `catalyst_priority_map`.
- Known FDA event types (FDA_APPROVAL, FDA_CRL, etc.) still get pri=1 via event-type-specific rules
- Only unknown/novel types from FEDERAL_REGISTER are affected (now pri=3 instead of 1)
- The source remains wired and functional — just deprioritized

## Replacement Source Candidates (by ROI)

### 1. Drugs@FDA / CDER Approval Reports (HIGH ROI)
- **URL:** `api.fda.gov/drug/drugsfda.json` (openFDA)
- Structured JSON, free, no scraping needed
- Contains: approval dates, NDA/BLA numbers, sponsor names, active ingredients
- **Mapping:** sponsor_name → ticker (can be automated from universe.json company names)
- **Events:** FDA_APPROVAL (high confidence, exact date), FDA_CRL (inferred from withdrawn/refused status)

### 2. FDA Advisory Committee Calendar (ALREADY IMPLEMENTED)
- ADCOM events are collected via `collect_fda_adcom_events()`
- Currently returns 0 events for 2026-02-07 (no ADCOMs scheduled near that date)
- **No action needed** — this source is working, just sparse

### 3. Product-to-Ticker Map Expansion (HIGH ROI for all sources)
- Current map: **28 entries** for 353 tickers (7.9% coverage)
- **From CT.gov:** Parse `intervention_name` + `lead_sponsor` from trial_records.json
  - Already cached per archive — no new API calls
  - Drug code names (e.g., "AXS-05") often appear as intervention names
  - Lead sponsors can be mapped to tickers via company name matching
- **From universe.json:** Company names → sponsor names (fuzzy match)
- **Target:** 150+ mappings (42%+ coverage) from existing data alone

### 4. SEC Pattern Broadening (MEDIUM ROI)
Additional regex patterns for sec_8k_catalyst_collector.py:
- "PDUFA date was extended" / "extended by three months"
- "mid-cycle review" / "late-cycle meeting"
- "Type A/B/C meeting"
- "NDA/BLA resubmission accepted/filed/submitted"
- "complete response letter received" variants
- "labeling discussions"

## Recommended Next Steps (ordered)

1. **Run multi-form collection** with v2 optimizations (per-ticker cap + adsh cache)
2. **Measure coverage uplift** via audit script
3. **Build product→ticker map** from trial_records.json intervention names
4. **Add openFDA Drugs@FDA** collector (structured, high confidence)
5. **Broaden SEC text patterns** (low risk, medium yield)
