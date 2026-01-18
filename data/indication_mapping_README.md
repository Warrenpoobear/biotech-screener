# Indication Mapping Registry

This document describes the indication-to-therapeutic-area mapping system used for PoS (Probability of Success) scoring in the biotech screener.

## Coverage Statistics

| Metric | Value |
|--------|-------|
| Total ticker overrides | 197 |
| Coverage achieved | 70.8% |
| Date created | 2026-01-18 |
| Version | 2.0.0 |

### Coverage by Source
- Manual overrides: 64 tickers (94%)
- Pattern matching: 4 tickers (6%)

### Coverage by Market Cap
| Bucket | Total | Covered | Coverage |
|--------|-------|---------|----------|
| Small ($300M-1B) | 2 | 2 | 100% |
| Mid ($1-5B) | 4 | 4 | 100% |
| Large ($5-20B) | 10 | 10 | 100% |
| Mega (>$20B) | 4 | 4 | 100% |

## Override Categories

| Category | Count | Example Tickers |
|----------|-------|-----------------|
| Oncology | 85 | AMGN, BMY, EXEL, INCY, JAZZ |
| Rare Disease | 32 | VRTX, ALNY, BMRN, CRSP, SRPT |
| CNS | 23 | BIIB, ALKS, NBIX, AXSM, ACAD |
| Autoimmune | 20 | ARGX, ABBV, AUPH, ARQT, KNSA |
| Infectious Disease | 15 | MRNA, BNTX, NVAX, GILD, BCRX |
| Respiratory | 8 | UTHR, INSM, TBPH, EOLS |
| Cardiovascular | 4 | CYTK, APLS |
| Metabolic | 6 | LLY, XERS, MNKD, MDGL |
| GI/Hepatology | 5 | TAK, IRWD, ARDX, ANAB, MIRM |
| Ophthalmology | 4 | REGN, HROW, OCGN, FDMT |
| Urology | 2 | URGN, OABI |

## Mapping Logic

The system uses a hierarchical approach:

1. **Ticker Override (Highest Priority)**
   - Direct ticker-to-indication mapping
   - Used for high-conviction, manually curated assignments
   - Confidence tier: 0.85

2. **Condition Pattern Matching**
   - Matches trial conditions against keyword patterns
   - Supports multiple patterns per therapeutic area
   - Confidence tier: 0.65-0.80

3. **Therapeutic Area Fallback**
   - Falls back to broad category matching
   - Confidence tier: 0.50

4. **Phase-Only Default**
   - Uses all_indications benchmark when no indication match
   - Confidence tier: 0.30

## File Structure

```
data/
├── indication_mapping.json          # Main mapping configuration
├── indication_mapping_README.md     # This file
└── pos_benchmarks_bio_2011_2020_v1.json  # BIO benchmark data
```

## Maintenance Protocol

### Weekly (Automated)
- Monitor coverage percentage
- Alert if coverage drops >5pp below baseline (70%)

### Monthly (Human Review)
- Review 10 newly-added tickers
- Verify mappings for top-20 ranked names
- Check pattern rule accuracy

### Quarterly (Full Audit)
- Re-run 20-ticker false positive audit
- Validate IC preservation vs baseline
- Update condition patterns for new disease categories

## Validation Commands

Run the validation script:
```bash
python scripts/validate_pos_coverage.py
```

Output is saved to:
```
production_data/pos_coverage_validation.json
```

## Adding New Overrides

To add a new ticker override, edit `data/indication_mapping.json`:

```json
{
  "ticker_overrides": {
    "NEWTICKER": "oncology"
  }
}
```

Valid therapeutic areas:
- oncology
- rare_disease
- infectious_disease
- cardiovascular
- cns
- autoimmune
- metabolic
- respiratory
- ophthalmology
- gi_hepatology
- urology
- other

## Change History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-18 | 2.0.0 | Expanded from 39.3% to 74.7% coverage |
| 2026-01-18 | 2.0.0 | Added 160+ ticker overrides |
| 2026-01-18 | 2.0.0 | Added urology therapeutic area |
| 2026-01-18 | 2.0.0 | Enhanced condition patterns |

## Owner

- **Team**: Wake Robin Capital Management
- **Review cadence**: Monthly
- **Last audit**: 2026-01-18
