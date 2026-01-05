# Wake Robin Data Pipeline - Production Deployment Checklist

## Overview
This checklist ensures systematic deployment of the Wake Robin data collection pipeline for your 20-ticker pilot universe.

**Deployment Date**: _________________  
**Environment**: _________________  
**Deployed By**: _________________

---

## Phase 1: Pre-Deployment Validation ✓

### 1.1 Environment Setup
- [ ] Python 3.8+ installed (`python --version`)
- [ ] pip package manager available (`pip --version`)
- [ ] Network access verified (no firewall blocking financial sites)
- [ ] Working directory created and accessible

### 1.2 File Extraction
- [ ] Downloaded `wake_robin_data_pipeline.tar.gz`
- [ ] Extracted to target directory: `tar -xzf wake_robin_data_pipeline.tar.gz`
- [ ] Navigated to directory: `cd wake_robin_data_pipeline`
- [ ] Verified all files present (see file manifest below)

### 1.3 Dependencies Installation
- [ ] Installed required packages: `pip install -r requirements.txt`
- [ ] Verified yfinance installed: `python -c "import yfinance; print('OK')"`
- [ ] Verified requests installed: `python -c "import requests; print('OK')"`

### 1.4 Pipeline Validation
- [ ] Ran validation suite: `python validate_pipeline.py`
- [ ] All 7 tests passed (0 failures)
- [ ] Validation report generated in `outputs/`

---

## Phase 2: Pre-Flight Verification ✓

### 2.1 Network Connectivity Tests
- [ ] Ran pre-flight check: `python preflight_check.py`
- [ ] General network: ✓ PASS
- [ ] Yahoo Finance API: ✓ PASS
- [ ] SEC EDGAR API: ✓ PASS
- [ ] ClinicalTrials.gov API: ✓ PASS

**If any pre-flight checks fail:**
- Review error messages carefully
- Check firewall/proxy settings
- Verify no VPN blocking financial APIs
- Contact IT if corporate network blocks access

### 2.2 Universe Configuration
- [ ] Reviewed `universe/pilot_universe.json`
- [ ] Verified 20 tickers are correct for pilot
- [ ] Confirmed company names are accurate
- [ ] Stages (commercial/clinical) properly classified

---

## Phase 3: First Production Run ✓

### 3.1 Initial Data Collection
- [ ] **Timestamp**: _________________
- [ ] Ran pipeline: `python collect_universe_data.py`
- [ ] Collection completed without errors
- [ ] Runtime recorded: __________ minutes

### 3.2 Output Verification
- [ ] Snapshot file created: `outputs/universe_snapshot_YYYYMMDD_HHMMSS.json`
- [ ] Quality report created: `outputs/quality_report_YYYYMMDD_HHMMSS.json`
- [ ] Symlinks created: `universe_snapshot_latest.json`, `quality_report_latest.json`
- [ ] File sizes reasonable (snapshot ~40KB, report ~5KB)

### 3.3 Data Quality Assessment

**Coverage Targets:**
- [ ] Price data (Yahoo): ≥ 95% (expected: 100%)
- [ ] Financial data (SEC): ≥ 75% (expected: 80-85%)
- [ ] Clinical data (CT.gov): ≥ 80% (expected: 85-90%)

**Actual Coverage Achieved:**
- Price data: ____%
- Financial data: ____%
- Clinical data: ____%
- Overall average: ____%

**Quality Distribution:**
- Excellent (>80%): ___ tickers
- Good (60-80%): ___ tickers
- Fair (40-60%): ___ tickers
- Poor (<40%): ___ tickers

### 3.4 Data Validation Spot Checks

**Select 3 tickers at random and verify:**

**Ticker 1: _______**
- [ ] Price data looks reasonable (matches market)
- [ ] Market cap calculation correct (price × shares)
- [ ] Cash figure matches last 10-Q/K (if available)
- [ ] Clinical stage matches company investor relations

**Ticker 2: _______**
- [ ] Price data looks reasonable
- [ ] Financial data present and recent
- [ ] Trial count matches ClinicalTrials.gov search
- [ ] Lead indication correctly identified

**Ticker 3: _______**
- [ ] All data sources populated
- [ ] No obvious data quality issues
- [ ] Provenance timestamps recent (< 24 hours)
- [ ] Overall coverage score reasonable

---

## Phase 4: Cache Verification ✓

### 4.1 Cache System Test
- [ ] Ran pipeline second time immediately after first run
- [ ] Runtime significantly reduced (should be < 1 minute with full cache)
- [ ] All results marked `"from_cache": true` in raw responses
- [ ] No new API calls made (verify in terminal output)

### 4.2 Cache Invalidation Test
- [ ] Deleted one ticker's cache files: `rm cache/yahoo/VRTX.json cache/sec/VRTX*.json`
- [ ] Ran pipeline with force refresh: `python collect_universe_data.py`
- [ ] Only VRTX data re-fetched (others from cache)
- [ ] Cache files recreated for VRTX

---

## Phase 5: Error Handling Validation ✓

### 5.1 Graceful Failure Test
- [ ] Temporarily renamed universe file to simulate missing config
- [ ] Pipeline failed with clear error message (not crash)
- [ ] Restored universe file
- [ ] Pipeline runs successfully again

### 5.2 Network Interruption Simulation
- [ ] Noted behavior if network interrupted mid-collection
- [ ] Verified partial results not corrupted
- [ ] Confirmed atomic file operations prevent partial writes

---

## Phase 6: Integration Readiness ✓

### 6.1 Output Format Validation
- [ ] Opened `universe_snapshot_latest.json` in text editor
- [ ] Structure matches expected format (see schema below)
- [ ] All required fields present: `ticker`, `market_data`, `financials`, `clinical`, `data_quality`
- [ ] Provenance section includes timestamps and sources

### 6.2 Downstream Compatibility
- [ ] Verified snapshot can be loaded by scoring model
- [ ] Tested extraction of key metrics (cash, market_cap, lead_stage)
- [ ] Confirmed data types correct (numbers as numbers, not strings)
- [ ] No unexpected null/missing value patterns

---

## Phase 7: Operational Setup ✓

### 7.1 Scheduled Execution
- [ ] Determined collection schedule (recommended: Tuesday 7:00 AM)
- [ ] Set up cron job (Linux/Mac) or Task Scheduler (Windows)
- [ ] Tested scheduled execution runs successfully
- [ ] Confirmed outputs saved to correct location

**Cron example (Tuesday 7 AM):**
```bash
0 7 * * 2 cd /path/to/wake_robin_data_pipeline && python collect_universe_data.py >> logs/collection.log 2>&1
```

### 7.2 Monitoring & Alerts
- [ ] Identified key person to monitor collection runs
- [ ] Set up email/Slack notification for failures (if desired)
- [ ] Documented expected runtime (baseline: ___ minutes)
- [ ] Created log rotation policy (keep last 30 days)

### 7.3 Backup Strategy
- [ ] Configured automatic backup of `outputs/` directory
- [ ] Tested restore from backup
- [ ] Documented backup location: _________________
- [ ] Set retention policy (recommend: 90 days of snapshots)

---

## Phase 8: Documentation & Handoff ✓

### 8.1 Operational Documentation
- [ ] README.md reviewed and understood
- [ ] DEPLOYMENT_GUIDE.md accessible to team
- [ ] troubleshooting section bookmarked
- [ ] Contact info for support documented

### 8.2 Team Training
- [ ] Demonstrated pipeline to scoring team
- [ ] Walked through quality report interpretation
- [ ] Showed how to investigate low-coverage tickers
- [ ] Explained cache system and refresh strategies

### 8.3 Handoff Checklist
- [ ] Primary operator identified: _________________
- [ ] Backup operator identified: _________________
- [ ] Escalation path documented
- [ ] Access credentials confirmed (if needed)

---

## Phase 9: Production Certification ✓

### 9.1 Final Validation
- [ ] Completed 3 consecutive successful runs
- [ ] Data quality consistent across runs (±5%)
- [ ] No unexplained failures or warnings
- [ ] Spot checks validate accuracy

### 9.2 Deployment Sign-Off

**Certified By:**
- [ ] Technical Lead: _________________ Date: _________
- [ ] Data Quality: _________________ Date: _________
- [ ] Operations: _________________ Date: _________

### 9.3 Go-Live Declaration
- [ ] **Pipeline status: PRODUCTION**
- [ ] **Go-live date: _________________**
- [ ] Next scheduled run: _________________
- [ ] Integration with scoring model: [ ] Pending [ ] Complete

---

## Appendix: Expected File Manifest

```
wake_robin_data_pipeline/
├── README.md                       (6.9KB)
├── DEPLOYMENT_GUIDE.md             (7.4KB)
├── requirements.txt                (400 bytes)
├── collect_universe_data.py        (Main pipeline)
├── validate_pipeline.py            (Validation suite)
├── preflight_check.py              (Pre-flight tests)
├── demo_pipeline.py                (Demo mode)
├── test_collectors.py              (Quick tests)
├── collectors/
│   ├── __init__.py
│   ├── yahoo_collector.py
│   ├── sec_collector.py
│   └── trials_collector.py
├── universe/
│   └── pilot_universe.json         (20 tickers)
├── cache/                          (Created on first run)
│   ├── yahoo/
│   ├── sec/
│   └── trials/
└── outputs/                        (Created on first run)
    ├── universe_snapshot_*.json
    ├── quality_report_*.json
    └── validation_report_*.json
```

---

## Appendix: Data Quality Schema

**Expected snapshot structure:**
```json
{
  "ticker": "VRTX",
  "as_of_date": "2025-01-05T10:30:00",
  "market_data": {
    "price": 425.67,
    "market_cap": 110500000000,
    "shares_outstanding": 259500000,
    "volume_avg_30d": 1500000,
    "company_name": "Vertex Pharmaceuticals"
  },
  "financials": {
    "cash": 4300000000,
    "debt": 0,
    "net_debt": -4300000000,
    "revenue_ttm": 10200000000,
    "assets": 15800000000,
    "liabilities": 3400000000,
    "equity": 12400000000,
    "currency": "USD"
  },
  "clinical": {
    "total_trials": 45,
    "active_trials": 12,
    "completed_trials": 33,
    "lead_stage": "commercial",
    "by_phase": {
      "PHASE1": 5,
      "PHASE2": 8,
      "PHASE3": 15,
      "PHASE4": 17
    },
    "conditions": ["Cystic Fibrosis", "Sickle Cell Disease"],
    "top_trials": [...]
  },
  "data_quality": {
    "yahoo_success": true,
    "sec_success": true,
    "trials_success": true,
    "overall_coverage": 92.5,
    "has_price": true,
    "has_cash": true,
    "financial_coverage": 90.0,
    "has_clinical": true
  },
  "provenance": {
    "collection_timestamp": "2025-01-05T10:30:00",
    "sources": {
      "yahoo_finance": {...},
      "sec_edgar": {...},
      "clinicaltrials_gov": {...}
    }
  }
}
```

---

## Appendix: Troubleshooting Guide

### Issue: "Module not found: yfinance"
**Solution**: `pip install yfinance requests`

### Issue: "403 Forbidden" from Yahoo
**Solution**: Check firewall/VPN settings, may need to whitelist finance.yahoo.com

### Issue: "Could not resolve ticker to CIK"
**Solution**: Expected for some companies (foreign, recent IPOs). System handles gracefully.

### Issue: Low financial coverage (<60%)
**Solution**: 
1. Check if companies actually file with SEC (some don't)
2. Verify CIK resolution working
3. Inspect SEC cache files for errors

### Issue: Pipeline takes >10 minutes
**Solution**:
1. Check network latency
2. Verify cache is working
3. Consider reducing parallel requests if rate limiting

### Issue: Stale cached data
**Solution**: Delete cache directory or use `force_refresh=True` parameter

---

**End of Checklist**

*Pipeline Version: 1.0*  
*Last Updated: 2025-01-05*  
*Maintained By: Wake Robin Capital Management*
