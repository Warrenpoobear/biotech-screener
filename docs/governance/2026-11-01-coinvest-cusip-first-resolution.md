# Coinvest CUSIP-first ticker resolution (D5)

**Date:** 2026-11-01
**Classification:** Tier 3 (coinvest/selector input; production hash). This is a data-integrity fix: no new signal and no engine change.
**Authorization:** operator, in session 2026-11-01 ("Yes, fix before Monday").
**Relationship to the clock reset:** this is an addendum to `docs/governance/2026-11-01-coinvest-13f-integrity-clock-reset.md` (D1–D4). It lands before the first counted run, so the same reset covers it and `oos_clock_start` stays 2026-11-02.
**Hash rotation:** `governance/HASH_ROTATIONS.md` (2026-11-01, D5 entry)

## Defect

13F holdings in the PIT cache carry a ticker filled in by the cache warmer's live lookup. `scripts/build_coinvest_features_from_13f.py::_resolve_ticker` trusted that ticker and consulted `production_data/cusip_static_map.json` only when it was blank. The live lookup fails in two ways.

1. **Universe names left blank, so no coinvest signal.** Holder counts below are for roster managers as of 2026-10-30.

   | Ticker | Holders seen before | Holders after fix |
   |---|---|---|
   | DFTX | 0 | 9 |
   | DMRA | 0 | 9 |
   | ASND | 1 | 8 |
   | VOR | 0 | 7 |
   | PBLS | 0 | 7 |
   | ENGN | 0 | 7 |
   | KARD | 0 | 6 |
   | STRO | 0 | 5 |
   | KLRA | 1 | 5 |
   | GENB | 0 | 4 |

   History is affected as well, e.g. IONS (12 roster managers since 2020) and NUVB (9).

2. **Unrelated issuers tagged with universe tickers.** Examples:

   | CUSIP | Actual issuer | Tagged as |
   |---|---|---|
   | 68622V106 | Organon | ORIC (top 5) |
   | 46266C105 | IQVIA | IRWD |
   | 03073E105 | AmerisourceBergen | AMRX |
   | 45826H109 | Integer Holdings | INSM |
   | 35137L105 | Fox Corp | FOLD |

   Also Shockwave → SIGA, Aerie → ADPT, Progyny → PRTA, and others.

The lookup is also non-deterministic. Re-warming the same date flipped TRAX (lost 2 holders) and DFTX (gained 5). That violates CCFT "canonical" and the repo's CUSIP-first data-provenance rule.

## Change

- **Builder 1.1.0 → 1.2.0:**
  - CUSIP-first: a static-map entry wins over the cached ticker; an unmapped CUSIP falls back to the cached ticker as before.
  - CUSIPs are upper-cased (two cache entries were lowercase).
  - Debt CUSIPs (letters in the issue number) never resolve to equity. Example: an Immunocore convertible note was counted as IMCR.
- **`cusip_static_map.json`, 352 → 399 entries, all verified.** No entry rests on issuer-name matching alone (cf. VTRS/VKTX, 69e49c031).
  - **23 universe names via OpenFIGI.** Each must be a US-listed equity, have exactly one universe ticker, and have an agreeing issuer name. Examples: IONS, NUVB, DFTX, CNTX, VOR, STRO, ENGN, PBLS, TRAX, KARD, KLRA, RNAC, GENB, NBP (I-Mab, renamed NovaBridge).
  - **4 by price.** The 13F implied price (value ÷ shares) equals the ticker's 2026-06-30 close:
    - DMRA 36322Q206: $26.01
    - ASND K08588103: $266.72
    - AZN G0593M107: $189.62
    - ACIU H00263105: $2.38
  - **7 cache-confirmed.** The cached ticker is consistent and the issuer name is exact: IMTX, ENGN (old CUSIP), PRAX (old), SRZN, OCS, ELVN, PTGX.
  - **15 corrections.** Mis-tagged issuers are mapped to their true, non-universe tickers (OGN, IQV, COR, ITGR, FOXA, SWAV, AERI, PGNY, GM, IPG, TGT, AN, USB, PCG, VRT), so they drop out of coinvest.
  - **Deliberately not mapped:** N62509117. It is NewAmsterdam warrants ($21.37 implied price vs NAMS $33.89 close); the ordinary shares N62509109 were already mapped.
  - **Rejected:** CPRX (delisted in the universe), plus Humana, Anthem and Prologis (non-equity matches, not biotech).
- **`coinvest_signals.json`:** rebuilt as of 2026-10-30 (Q2 2026, 30 managers).

## Measured impact — 2026-10-30

Compared with the roster-only file (D1–D4, previously set to go live 2026-11-02), with identical code and inputs otherwise:

| | Roster-only | + CUSIP-first |
|---|---|---|
| Tickers with signal | 256 | 271 |
| Unresolved holdings | 611 | 539 |
| Top-30 overlap | — | 27/30 |
| Leaves / enters | | ABSI, PTGX, PVLA / ASND, BCRX, TNGX |
| coinvest_score_z SD | 0.684 | 0.676 (collapse guard > 0.10: PASS) |
| coinvest coverage | 76.8% | 80.9% |
| Phase-2 health | FAIL (headwind) | FAIL (headwind 59.9%) |

ORIC stays #5; its coinvest_score_z falls to 0.38 once Organon is removed. The newly visible names do not reach the top 30: DFTX #36, DMRA #48, VOR #57, PBLS #59.

## Notes and follow-ups

- Historical coinvest files and all prior production snapshots carry D5. Prior research and backtests that used coinvest inherit it.
- The cache warmer's live lookup still writes unreliable tickers into `data/caches/` (gitignored). The builder now ignores them wherever the map has an entry. Fixing the warmer itself is a separate change.
- `institutional_summary.py` and `run_screen.py::_convert_holdings_to_coinvest` have their own resolution paths and are not changed here. Neither is the production coinvest source; the selector reads the builder's `coinvest_signals.json`.
- Unmapped non-universe CUSIPs (about 2,100) are left alone; coinvest ignores non-universe tickers.
