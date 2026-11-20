# Curated regulatory calendar refresh (2026-11-15)

**Classification:** Tier 3 data input (`production_data/pdufa_dates.json`, read by `event_ledger.py` and `common/regulatory_calendar.py`). No code change.
**Authorization:** operator instruction 2026-11-15 ("update clinical and catalyst calendar data. check online sources").
**Hash rotation:** `governance/HASH_ROTATIONS.md` (2026-11-15 calendar entry).
**Clock:** intended to land before `oos_clock_start` for the selector-only order is recorded (expected 2026-11-16), so it is part of that window's starting configuration rather than a reset.

## 1. Method

All 47 entries dated on or after 2026-11-01 were checked against online sources, and a search was made for upcoming FDA action dates and advisory committee meetings for universe companies that the calendar lacked. Every change below was confirmed against a company release, an SEC filing, or (SMMT only) STAT News. Calendar aggregators were used only to find candidates.

Clinical-trial data needed no action: `production_data/trial_records.json` was refreshed on 2026-11-15 (23,275 trials, newest update posted 2026-11-13).

## 2. Changes (55 → 59 entries)

**Outcomes and status**

| Ticker | Event | Change | Source |
|---|---|---|---|
| SMMT | ivonescimab BLA, 2026-11-14 | `status: resolved`, `outcome: CRL` | Summit release and STAT, both 2026-11-15 |
| CYTK | aficamten MAPLE-HCM sNDA, 2026-11-14 | still `pending`: date passed (Saturday), no outcome announced | company 8-K list through 2026-11-04; news 2026-11-14 |
| TEVA | BXCL501 at-home sNDA, 2026-11-14 | still `pending`, no outcome announced; drug name corrected; Teva acquired the asset from BioXcel (closing 2026-10-27) | Teva release 2026-10-27; BioXcel 8-K |
| NUVB | IBTROZI sNDA, target 2027-01-04 | approved early, about 2026-09-17: date moved to 2026-09-17, `original_target_date` kept | company Q3 release; OncLive 2026-09-17 |
| IBRX | ANKTIVA papillary-only sBLA, target 2027-01-06 | approved early, announced 2026-10-14: date moved, `original_target_date` kept | company 8-K Ex. 99.1, 2026-10-14 |

**Removed (2)**

- **EXEL 2026-12-03:** the same NDA (zanzalintinib + atezolizumab, mCRC) as the existing 2027-03-03 entry. The FDA extended the review by three months in September 2026 after a major amendment (10-Q for the quarter ended 2026-10-02). The remaining entry's note, which called it a second application, is corrected.
- **ANAB 2026-12-12:** imsidolimab is Vanda's BLA (licensed from AnaptysBio) and is already carried under VNDA. Same partner rule as the 2026-11-03 refresh (ZLAB, LGND, RPRX excluded).

**Added (4)**

| Ticker | Drug | Date | Disclosed | Source |
|---|---|---|---|---|
| AQST | Libervant, final action for ages 12+ after Valtoco exclusivity expires | 2027-03-23 | 2026-11-03 | 8-K Ex. 99.1 |
| AQST | Libervant, Type 9 NDA for ages 6–11 | 2027-03-29 | 2026-11-03 | 8-K Ex. 99.1 |
| UTHR | ralinepag NDA, PAH | 2027-06-24 | 2026-08-24 | company release |
| CELC | REVTORPYK (gedatolisib) sNDA, PIK3CA-mutant breast cancer | 2027-06-25 | 2026-11-09 | company release |

**Indications filled in:** AGIO (sickle cell disease), PRAX relutrigine (SCN2A/SCN8A DEEs), VSTM (KRAS-mutated low-grade serous ovarian cancer; new packaging), ARGX 2027-02-18 (pediatric gMG, medium confidence), CYTK (obstructive HCM label update).

**Confirmed unchanged:** the other 38 upcoming dates. For GILD lenacapavir weekly oral, PHVS, INBX and ARGX the latest company release does not restate the date; nothing contradicts it.

## 3. Not added

- **SRPT** AMONDYS 45 / VYONDYS 53 traditional-approval sNDAs, reported as 2027-02-28: the company's own release could not be read (Business Wire and the IR site blocked access) and the 8-K exhibit of 2026-08-05 gives no date. Add once a primary source is read.
- **ETON** KHINDIVI label expansion, 2027-05-28 (8-K, 2026-11-05): genuine, but ETON is `pending_coverage` in the universe and is not ranked.
- **Month-only or unset dates:** BHVN troriluzole (Q1 2027; advisory committee planned, date not set), QURE AMT-130 advisory committee (not set), UTHR Tyvaso IPF (late April 2027), REGN odronextamab (February 2027), TEVA ecopipam (late Q1 2027).
- **No advisory committee date** on or after 2026-11-16 was found for any universe company.
- **Search gaps:** SEC full-text search was unavailable, so dates disclosed only inside 10-Q text or on earnings calls may be missing; about 250 smaller names were covered by broad searches only.

## 4. Departure from the maintenance guide

`production_data/REGULATORY_CALENDAR_MAINTENANCE.md` says not to delete entries and to record a date change as a new entry. No reader honours `status`: `event_ledger._load_pdufa_events` and `common/regulatory_calendar.py` treat every row as a scheduled PDUFA event. Kept as they were, the EXEL 2026-12-03 row and the NUVB and IBRX January 2027 rows would be read as upcoming catalysts that no longer exist (EXEL already carried both dates). They are therefore removed or moved, with the old date kept in `notes` or `original_target_date`.

Point-in-time cost: for as-of dates between each record's disclosure and the FDA action, a regenerated history would see the event at the action date (NUVB, IBRX) or would not see the superseded EXEL date. A loader that honours a supersession field would remove that cost; it is a code change in the scoring path and is not made here. The guide gains a note describing this.
