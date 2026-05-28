# Agent Routing Policy
## Purpose
This policy defines which agent/model tier may modify, review, or reason about each part of the biotech screener repository.
Routing is based on governance sensitivity, not task size.
Core rule:
> Use the cheapest execution path whose failure mode cannot contaminate point-in-time integrity, production evidence, selector/ranker behavior, catalyst interpretation, shadow evidence, or governance records.
A file's tier is determined by the highest-tier consumer of its output, not by the file's apparent simplicity.
Pricing changes do not change tier assignments. They only change which tools are economically feasible within a tier.
---
## Tier 0 - Deterministic Production Hot Path
Use deterministic scripts, cron, tests, static checks, and validated pipeline code.
No LLM may supervise, decide, mutate, or deploy production state in the hot path.
Scope:
- daily production pipeline
- scheduled artifact generation
- cache refreshes
- watchdog checks
- snapshot creation
- recurring validation commands
- production cron execution
- deterministic alerting
Allowed:
- scripts
- cron
- tests
- static checks
- CI
- deterministic validators
LLMs may inspect outputs after the fact, but must not directly mutate production state.
Boundary:
> Read, reason, propose. Never decide, write, deploy.
---
## Tier 1 - Low-Governance Utility Work
Allowed tools:
- Codex CLI
- OpenClaw low-risk agents
- deterministic scripts
Scope:
- documentation edits
- README/runbook improvements
- CLI ergonomics
- log formatting
- file organization
- non-production helper scripts
- non-scoring utilities
- test harness ergonomics that do not assert Tier 3 behavior
Restrictions:
Tier 1 work must not affect:
- scoring
- selector/ranker behavior
- point-in-time controls
- catalyst interpretation
- CRT evidence
- shadow outputs
- production hashes
- production snapshots
- downstream evidence used for governance decisions
---
## Tier 2 - Medium-Governance Engineering
Allowed tools:
- Codex CLI for first draft
- OpenClaw agents with Tier 2 authorization
- Claude Code review when required by the triggers below
Scope:
- non-production analytics tools
- validation scripts
- audit tools
- backfill tooling
- artifact comparison tools
- ingestion plumbing whose outputs do not feed Tier 3 systems
- cache readers/writers whose outputs do not feed Tier 3 systems
Important classification rule:
> A cache reader, ingestion script, parser, or utility becomes Tier 3 if its output is consumed directly or transitively by Tier 3 code.
Claude Code review is required before merge if:
- any modified file is imported directly or transitively by Tier 3 code
- the diff modifies a function signature called from Tier 3 code
- the diff changes an artifact consumed by selector, ranker, scoring, catalyst resolution, CRT, shadow monitoring, or walk-forward evidence
- the diff changes behavior of production-adjacent outputs
This trigger is mechanical, not discretionary.

**Verification tool:** Run `codegraph_impact("<changed_symbol>", depth=2)` on each changed function before opening the PR. If the impact graph reaches any Tier 3 surface (selector_engine, ranker_engine, decision_engine, final_score, rankings.csv, snapshot writer/schema), the Claude Code review trigger fires unconditionally. Report the full impact list in the PR description.
---
## Tier 3 - High-Governance Production and Evidence Surface
Allowed tools:
- Claude Code for implementation or mandatory review
- Claude Chat / project chat for architecture and spec decisions
- Codex only as a permitted exception for first drafts of Tier 3 code when all of the following are true:
  - the change has a written spec
  - the implementer flags the diff as Tier 3 in the PR description
  - Claude Code performs final review before merge
The default Tier 3 implementer is Claude Code.
Scope:
- CCFT enforcement
- point-in-time logic
- `data_available_timestamp`
- selector engine
- ranker engine
- module scoring
- final score construction
- module weights
- catalyst extraction/resolution
- SEC / 8-K catalyst feed
- CT.gov catalyst feed
- event timing logic
- Catalyst Resolution Tracker writes
- CRT label assignments
- CRT postmortem ingestion
- postmortem candidate ingestion
- automated postmortem generation pipelines
- automated CRT resolution writes from event-resolution detection
- any code path that creates evidence rows without human review
- shadow-signal output schemas
- shadow-monitoring artifacts
- isolation/ablation test harnesses
- walk-forward harness
- production cache invalidation
- ruleset hash generation
- portfolio construction
- any code that changes Top-30 membership
- any artifact that becomes evidence for future signal promotion, demotion, or weight-lock decisions
Requirements:
- No direct implementation without a written spec, candidate note, or explicit implementation instruction.
- Diffs must be small, reviewable, and test-backed.
- Claude Code review is required before merge.
- Tests asserting Tier 3 behavior are themselves Tier 3 and require Claude Code review, regardless of which tool drafted them.
- AI-generated tests for Tier 3 behavior must be reviewed for vacuity, leakage, and false-comfort failure modes.
---
## Walk-Forward Harness Special Status
The walk-forward harness is a permanent Tier 3 surface.
Before it exists:
- Claude Chat / project chat owns design and specification.
- Claude Code owns implementation or mandatory review.
After it ships:
- its outputs become the evidence base for future signal admission, promotion, demotion, and weighting decisions.
- any change to the harness remains Tier 3 forever.
- changes require an isolation test demonstrating that historical evidence remains reproducible bit-for-bit, unless the purpose of the change is an explicit, documented evidence-breaking migration.
No "small refactor" exception applies to the walk-forward harness.
Evidence-breaking migrations are Tier 4 decisions.
They require a memo documenting:
- reason for migration
- cutover date
- affected harness outputs
- disposition of pre-migration evidence: archived, deprecated, or invalidated
- expected impact on historical comparability
- explicit PM/human sign-off
Evidence-breaking migrations are not authorized by Claude Code review alone.
---
## Production Hash Rotation Rule
Production hashes are load-bearing identifiers.
Examples include:
- selector ruleset hash
- ranker ruleset hash
- scoring ruleset hash
- production model version hash
- any hash referenced by snapshots, CRT entries, postmortems, shadow comparisons, or governance memos
Any diff that changes a production hash requires a corresponding entry in `HASH_ROTATIONS.md` or equivalent.
Each hash-rotation entry must include:
- old hash
- new hash
- effective date
- affected surface
- reason for rotation
- expected downstream artifact impact
- reviewer
Claude Code review must verify that the hash-rotation entry exists before merge.
---
## Tier 4 - Governance / Research Judgment
Allowed tools:
- Claude Chat / project chat
- human approval before implementation
- Claude Code only after a Tier 4 decision has been converted into an implementation spec
Scope:
- model architecture changes
- new alpha signal admission
- signal retirement
- selector/ranker policy
- catalyst taxonomy changes
- ablation interpretation
- CRT anomaly diagnosis
- postmortems
- June/quarterly governance memos
- decision to unfreeze, restore, suppress, or retire a signal
- dynamic regime weighting
- Alt 3 / Alt 4 / Alt 6 decisions
- changes to this routing policy
Requirement:
Tier 4 output must be a memo, spec, or explicit implementation instruction.
Tier 4 does not directly modify code.
---
## OpenClaw Fleet Policy
OpenClaw remains the production agent runtime.
Backend routing must be configurable per agent.
Each agent must be declared in a central registry, preferably `agent_registry.yml`.
The registry must include, for each agent:
- agent name
- purpose
- maximum governance tier
- model backend
- allowed directories
- forbidden directories
- allowed commands
- required tests
- whether Claude Code review is required
- whether the agent may write files
- whether the agent may touch production artifacts
- whether the agent may touch CRT, shadow, or postmortem evidence
- owner or reviewer
Per-agent docstrings are not sufficient. The registry is the auditable source of truth.
Changing an agent backend is allowed as a config change only if the agent's maximum tier and review requirements are unchanged.
Changing an agent's maximum tier is a Tier 4 governance decision.
---
## Claude Chat Policy
Claude Chat / project chat is used for:
- architecture decisions
- governance memos
- CRT/anomaly triage
- selector-vs-ranker diagnosis
- model design decisions
- external methodology evaluation
- spec drafting
- signal admission or suppression reasoning
- determining whether an anomaly is a label problem, catalyst-taxonomy problem, data issue, or real signal degradation
Claude Chat may reason over production outputs after the fact.
Claude Chat may not directly mutate production state.
---
## Claude Code Policy
Claude Code is required for:
- Tier 3 implementation
- Tier 3 review
- governance-sensitive diff review
- tests asserting Tier 3 behavior
- production hash rotation verification
- selector/ranker/scoring/catalyst modifications
- any diff whose failure could contaminate future evidence
Claude Code is the senior reviewer for governance-sensitive code regardless of pricing.
---
## Codex CLI Policy
Codex CLI may be used for:
- Tier 1 work
- Tier 2 first drafts
- utility scripts
- documentation
- non-scoring ingestion plumbing
- tests that do not assert Tier 3 behavior
- mechanical fixes outside governance-sensitive paths
Codex CLI may not be the sole reviewer for Tier 3 changes.
Codex-generated Tier 3 tests require Claude Code review.
---
## Hermes Policy
Hermes is optional.
Do not make Hermes central unless a specific workflow is demonstrably slower or less reliable without it.
The repo, not Hermes session memory, is the source of truth.
Durable memory lives in:
- specs
- candidate notes
- audit memos
- governance docs
- tests
- hashes
- production artifacts
- routing policy
- agent registry
- system state documents
Adding Hermes must not create routing ambiguity.
---
## Local LLM Policy
Local LLMs are deferred.
Revisit only when both conditions hold:
1. hardware is sufficient for reliable local reasoning, and
2. a specific non-critical use case clearly justifies the added routing complexity.
Local LLMs are not approved for production-sensitive summarization, governance reasoning, Tier 3 code changes, CRT interpretation, or test-failure diagnosis.
---
## AI-Generated Test Policy
Tests are classified by the behavior they assert, not by the directory they live in.
Tests asserting Tier 3 behavior are Tier 3.
This includes tests for:
- CCFT enforcement
- PIT behavior
- selector/ranker logic
- score construction
- catalyst timing
- CRT resolution
- shadow outputs
- walk-forward evidence
- production hash behavior
- cache invalidation that affects Tier 3 consumers
AI-generated tests for Tier 3 behavior require Claude Code review before merge.
A vacuous test, leakage-tolerant test, or test that locks in incorrect behavior is treated as a governance failure.
---
## Merge Rule
Before merge, classify the diff by the highest affected tier.
Highest affected tier governs review requirements.
If any file, function, artifact, schema, test, or output touches Tier 3 or Tier 4 surfaces, Claude Code review is required regardless of patch size.
Patch size is not evidence of safety.
---
## Policy Maintenance
This policy is itself Tier 4.
Changes require a memo, not a direct edit.
Quarterly review is required to confirm tier classifications still match the actual codebase and production dataflow.
Drift is expected.
Pricing changes do not change tier assignments.
Provider economics may change which tools are affordable within a tier, but governance sensitivity remains invariant.
