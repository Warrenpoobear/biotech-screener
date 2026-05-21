---
paths:
  - governance/**
  - production_data/decision_rulesets/**
  - scripts/promote_ruleset.py
  - tools/ruleset_health_monitor.py
---

# Governance & Promotion

## Agent Routing Policy (governance/AGENT_ROUTING_POLICY.md)

### Tier Definitions
| Tier | Scope | Allowed Tools | Review |
|------|-------|---------------|--------|
| 0 — Deterministic Hot Path | Scripts, cron, tests | Static checks only. No LLM. | None |
| 1 — Low-Governance Utility | Documentation, CLI ergonomics | Codex CLI, low-risk agents | Minimal |
| 2 — Medium-Governance Engineering | Non-production analytics, ingestion | Codex draft + Claude Code review | Standard |
| 3 — High-Governance Production | CCFT, selector, ranker, scoring, catalyst, CRT, shadow, walk-forward, production hashes | Claude Code for implementation or mandatory review | Required |
| 4 — Governance/Research Judgment | Architecture changes, signal admission/retirement | Claude Chat/project chat + human approval | Approval required |

**Merge rule**: highest affected tier governs review requirements. Patch size is not evidence of safety.

### Walk-Forward Harness
Permanent Tier 3 surface. Evidence-breaking migrations are Tier 4 decisions requiring memo with cutover date, affected outputs, disposition of pre-migration evidence, and PM sign-off.

### Production Hash Rotation
Any diff changing a production hash requires an entry in `governance/HASH_ROTATIONS.md`:
old hash, new hash, effective date, affected surface, reason, downstream impact, reviewer.

## Promotion Governance
- **Manifest**: `production_data/decision_rulesets/manifest.json` — all rulesets with status (active/candidate/retired)
- **Battery**: `scripts/research/run_promotion_battery.py` -> bucketed verdicts + weekly live-sim -> PASS/FAIL
- **Promote**: `scripts/promote_ruleset.py` — blocks unless battery PASS
- **Health monitor**: `tools/ruleset_health_monitor.py` — post-promotion drift detection
- **Rollback**: `scripts/promote_ruleset.py --rollback --reason "..."` — first-class with auto-LKG discovery

## Promotion Story (v1.14.0)
1. Coinvest-only selector + pairwise_minimal ranker (ordinal-only) + EW Top-30 is production
2. True PIT evidence: +2.34pp/mo net, t=2.57, 69% hit rate, 67 periods
3. B6 passes Checklist v2: bootstrap +2.42pp/mo, CI [1.25%, 3.70%], LOSO ROBUST
4. Pairwise ordinal-only confirmed: ECE=0.129
5. Within-cohort roles: coinvest selects, financial penalizes, inst_delta active in ranker only
6. event_type_score: 5/5 Checklist v2 but overlay only
7. Forward shadow: 7 arms accumulating daily — evaluate after 30 trading days
8. K=30 validated by PIT sweep (stable K=25-35 plateau)
9. Regime caveat: bear/neutral alpha engine. Bounded underperformance in strong bull.
10. Governance hold (Spec 048) succeeded: prevented institutionalizing contaminated data

## Architecture Freeze Protocol
During freeze windows:
- No new enforcement logic or scoring changes
- Monitoring and documentation changes allowed
- CI fixes and test-only changes allowed
- Spec research continues but does not land in production
- Freeze lifts after explicit operator approval at checkpoint
