# Curated PDUFA calendar refresh (2026-11-03)

**Classification:** Tier 3 data input (catalyst / regulatory calendar). No code change.
**Authorization:** operator, in session 2026-11-03 ("Yes, open the PR"). **Merging is a separate operator decision** (see §4).
**Hash rotation:** `governance/HASH_ROTATIONS.md` (2026-11-03 entry)

## 1. Problem

`production_data/pdufa_dates.json` is the hand-curated calendar the screen reads (`event_ledger.py`, `run_screen.py` input `pdufa_dates`). It was last edited 2026-06-22. It held 11 events, of which only one was still upcoming (SMMT 2026-11-14), and Friday's regulatory-calendar gate warned that its newest entry was 187 days old.

`tools/build_pdufa_dates_extracted.py` keeps a shadow list (`production_data/pdufa_dates_extracted.json`) extracted from cached 8-K/6-K events. It never writes the curated file. Its 2026-11-01 run found 41 upcoming dates, all labelled HIGH confidence.

## 2. Verification

Each of the 41 dates was traced to its source filing through EDGAR full-text search (the exact matched phrase, the company CIK, and the filing date ± 4 days), and judged from the sentence it came from.

| Verdict | Count | Records |
|---|---|---|
| Confirmed: the company's own FDA action date | 33 | see the table in §3 |
| Partner's date, not this company's application | 4 | ZLAB (Vertex's povetacicept), LGND (Palvella's Qtorin), RPRX (Pharvaris's deucrictibant), AQST 2027-03-29 (Eton's product) |
| Not a PDUFA date | 4 | LRMR, OCUL, QURE (2026-12-31 matched from risk-factor text); MESO (Type B meeting text) |

ETON's date is genuine, but ETON is inactive in the universe and is excluded. That leaves **32 additions**. Each records the filing URL, the accession in `notes`, and `as_of_disclosed_at` (the public-disclosure date, 2026-06-15 to 2026-10-29), so point-in-time readers only see a date after it was disclosed.

**Caveats:**
- ARGX 2027-02-18 comes from a slide table that does not name the indication.
- CGEM's zipalertinib NDA was filed by partner Taiho.
- The existing ARGX (2026-06-07) and AXSM (2026-04-30) entries are past, separate applications. They are kept as history.

Also found: the extractor's HIGH label does not discriminate, and it mis-reads year-end placeholders and partner disclosures. Fixing the extractor is a separate change.

## 3. Additions (32)

CAPR 11-22 (extended), BBIO 11-27, COGT 11-30, VRTX 11-30, EXEL 12-03, VNDA 12-12, PRAX 12-27, VTRS 12-27, COGT 12-30, NUVB 01-04, IBRX 01-06, DYN 01-21, INSM 01-28, PRAX 01-29, GILD 02-02, BBIO 02-04, BLTE 02-12, ARGX 02-18, ARQT 02-23, CGEM 02-27, BMRN 02-28, PVLA 02-28, EXEL 03-03, NTLA 03-10, AQST 03-16, HRMY 04-01, INBX 04-14, PHVS 04-23, IMNM 04-28, AXSM 05-01, BBIO 05-08, ZBIO 05-27 (2026 dates first, then 2027).

## 4. Impact and clock

On 2026-10-30, with identical code and inputs except this file:

| Metric | Result |
|---|---|
| Top-30 overlap | 29/30 (TVTX out, PVLA in) |
| Headwind exposure | 59.9% → 56.6% |
| Catalyst ≤ 7d count | unchanged (6) |

**Forward-validation clock.** This is a feature-input change landing after `oos_clock_start = 2026-11-02`. Under `docs/FORWARD_VALIDATION_PROTOCOL.md` §1 an authorized change resets the clock for affected periods. At merge time the operator should do one of the following:
- (a) merge and move `oos_clock_start` to the first production run that uses this file (cost: the windows captured before it);
- (b) hold it until a planned reset point.

Not merging leaves the screen with one upcoming curated PDUFA date.
