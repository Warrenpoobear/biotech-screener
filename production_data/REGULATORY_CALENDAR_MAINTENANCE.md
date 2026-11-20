# Regulatory Calendar Maintenance

## File

`production_data/pdufa_dates.json` — forward-looking regulatory event calendar.

## Schema (v2)

Each record is a JSON object with these fields:

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `ticker` | yes | string | Stock ticker |
| `pdufa_date` | yes | YYYY-MM-DD | Expected regulatory action date |
| `event_type` | no | string | `PDUFA`, `DUFA`, `AdCom`, `CHMP_OPINION`, `FDA_DECISION` |
| `drug_name` | no | string | Drug/product name |
| `indication` | no | string | Target indication |
| `submission_type` | no | string | `NDA`, `BLA`, `sNDA`, `sBLA` |
| `confidence` | no | string | `HIGH` (confirmed), `MED` (estimated), `LOW` |
| `source` | no | string | `COMPANY_GUIDANCE`, `ANALYST_ESTIMATE`, `SEC_8K`, `PRESS_RELEASE`, `MANUAL` |
| `source_url` | no | string | URL to source document |
| `as_of_disclosed_at` | **yes** | YYYY-MM-DD | When this date was first publicly known |
| `notes` | no | string | Free-text context |
| `program` | no | string | Drug + indication combined |

## Example entry

```json
{
  "ticker": "ACME",
  "drug_name": "AcmeDrug",
  "indication": "Rare Disease",
  "pdufa_date": "2026-08-15",
  "event_type": "PDUFA",
  "submission_type": "NDA",
  "confidence": "HIGH",
  "source": "COMPANY_GUIDANCE",
  "source_url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=...",
  "as_of_disclosed_at": "2026-02-15",
  "notes": "NDA acceptance letter disclosed in 8-K on 2026-02-15"
}
```

## How to set `as_of_disclosed_at`

This is the **PIT (point-in-time) anchor**: the date the regulatory event date became publicly known.

- For **company guidance**: the date of the press release or 8-K filing
- For **SEC 8-K**: the filing date from EDGAR
- For **analyst estimates**: the date the estimate was published
- When in doubt, use the date you added the entry

**Why it matters:** PIT filtering prevents look-ahead bias in backtests. Records without this field are treated as LOW confidence.

## How to add new entries

1. Open `production_data/pdufa_dates.json`
2. Add a new object to the array
3. Fill in all required fields (`ticker`, `pdufa_date`, `as_of_disclosed_at`)
4. Run the audit script to validate:

```bash
python3 scripts/research/audit_regulatory_calendar_coverage.py \
    --as-of-date $(date +%Y-%m-%d) \
    --snapshot-root data/snapshots
```

5. Run tests:

```bash
python3 -m pytest tests/test_regulatory_calendar.py -v --override-ini="addopts="
```

## Rules

- **Don't delete old entries** — even past dates are useful for backtesting
- Mark stale entries via `notes` (e.g., "PDUFA date passed; decision: approved")
- To update a date, add a new entry with the corrected date and a later `as_of_disclosed_at`
- **Dedupe key**: `(ticker, pdufa_date, event_type)` — duplicates are flagged as warnings
- **Caveat (2026-11-15):** no reader honours `status` — every row is loaded as a scheduled event. A future date that has been superseded (review extended) or resolved early (approved before the target) is still read as upcoming. Until a loader supports supersession, remove the superseded row or move an early approval to its action date, and keep the old date in `notes` / `original_target_date`. See `docs/governance/2026-11-15-regulatory-calendar-refresh.md` §4.

## Sources for new entries

1. **Company press releases** — most reliable for PDUFA dates
2. **SEC 8-K filings** — search EDGAR for "PDUFA" or "action date"
3. **FDA PDUFA date tracker** — third-party aggregators
4. **Analyst reports** — use `confidence: "MED"` for estimates
5. **FDA AdCom calendar** — for upcoming advisory committee meetings
