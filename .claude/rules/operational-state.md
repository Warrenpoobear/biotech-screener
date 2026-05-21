# Operational State (Volatile — Updated Frequently)

*Last updated: 2026-05-20*

This file contains volatile operational status that changes weekly or daily.
It is NOT path-scoped — load on demand via `@` reference when needed.

## Architecture Freeze
- v1.14.0 freeze in effect until post-h20d checkpoint (~2026-05-26)
- `ranker_active_contract.py` on unmerged branch (`hygiene/ranker-active-contract-2026-04-30`), deferred post-freeze
- Spec 100 (ranker IC tooling correction) is highest-priority code change post-freeze

## Forward Shadow & IC Status
- Forward shadow accumulating since 2026-04-03. ~30+ trading days as of mid-May.
- coinvest_score_z IC (last measured 2026-05-13): Pooled mean IC = -0.031 (14 dates, 28.6% hit rate)
  - Pre-cohort (clean): -0.051 (11.1% hit)
  - Post-cohort (contaminated): -0.008 (60.0% hit)
  - Verdict: OBSERVE
- Ranker IC: UNMEASURED. Existing tools conflate composite_score with final_score (Spec 095). All prior ranker IC claims misattributed. Blocked until Spec 100.
- inst_delta_z: zeroed in selector since 2026-05-04. Active in ranker (NW-t = +3.32). Reinstatement requires IC recovery evidence.
- Refresh IC decomposition after Q1 2026 13F cache warm + cohort quarantine.

## 13F Cycle Status (Q1 2026 — COMPLETE)
- All three tracked managers filed Q1 2026 13F-HR on deadline day (May 15, 2026)
- Accession numbers: Fairmount 0001104659-26-062419, Deep Track 0001856083-26-000003, Logos Global 0001172661-26-002196

### Key Changes
- **Fairmount**: Added DAMORA ($225.7M, 16.3% — largest new, NOT signaled by 13D/13G). Massive APGE trim (-85.4%), COGT trim (-38.9%). Exits: KINIKSA, NUVALENT. Post-Q1: VRDN raised to 14.04% via $20M purchase May 11.
- **Deep Track**: AUM $6,124M (+9.2%). 63 positions. 16 new including ALMS ($149M), NUVL ($141M). Largest exit: DVAX ($242M). VRDN: accumulated to 5.4M shares post-Q1.
- **Logos Global**: AUM $2,003M (+21.0%). 66 positions. Massive CNTA add (+963%). New: UTHR ($47M), MDGL ($44.5M).
- **Top coinvest**: VRDN (FM 14.04% + DT 5.30%) entering Ph3 TED readout.
- **13D/13G pre-signal validation**: ~60-70% of major moves captured pre-filing. Largest surprises (DAMORA, CNTA) were invisible until 13F-HR.

### Post-Filing Action Sequence
1. Warm 13F cache
2. Run cohort quarantine
3. Check collapse guards (coinvest_score_z SD)
4. Refresh IC decomposition
5. 5-day observation window

### Next Cycle
Q2 2026 (period ending June 30, 2026). Filing deadline ~August 14, 2026. Monitor EDGAR starting ~August 11.

## CI Pipeline Status
- CI red since ~May 8. PR #285 open/unmerged.
- phase2-daily-production cron is dark.
- CI Fix Checklist produced May 14-16 but remediation not confirmed complete.

## Governance Artifacts
- PR #286 merged May 16: AGENT_ROUTING_POLICY.md, STATUS.md, HASH_ROTATIONS.md
- Enforcement status: policy is live. Pending: agent_registry.yml (PR 2), AGENT_DIRECTORY_MAP.md, CI registry validation, import-graph validation.
