# Sci-Cart R6 — Mechanism Coverage Design Memo (2026-06-25)

**Status:** PLANNING — no implementation in Phase 13  
**Governance:** READ_ONLY_DIAGNOSTIC only

## Problem

`mechanism_class` is null for ~99.9% of program records. CT.gov intervention names are drug identifiers, not mechanism descriptors. The built-in mechanism normalizer dictionary covers ~30 manual entries.

## Local data sources (cache-only)

| Source | Location | Coverage |
|--------|----------|----------|
| Manual mechanism dict | `scientific_cartography/normalize/mechanism_normalizer.py` | ~30 entries |
| ChEMBL enricher | `scientific_cartography/enrich/chembl_enricher.py` | Target-level bioactivity when cache present |
| OpenTargets | `scientific_cartography/enrich/open_targets_enricher.py` | Association scores, not MOA labels |
| Intervention strings | `production_data/trial_records.json` | Drug names / descriptions |

## Recommended approach

1. **Short term (Phase 13):** No mechanism resolver expansion — document unknown rate in diagnostics manifest.
2. **Medium term:** Drug-name → target mapping via ChEMBL cache + INN stem heuristics (READ_ONLY_DIAGNOSTIC).
3. **Long term:** Curated mechanism ontology file (`scientific_cartography/data/mechanism_aliases_v0_2.csv`) synced from operator review.

## Governance constraints

- No wiring of mechanism fields into ranker, selector, sizing, or `final_score`
- No external API calls during production screen
- Any new mapping file must be versioned and PIT-timestamped

## Estimated scope (future implementation)

- Dictionary expansion: ~200 lines + 40 tests
- Drug→target resolver: ~400 lines + integration tests with CT.gov sample fixtures
- Not scheduled until operator authorizes post-R4 normalization verdict
