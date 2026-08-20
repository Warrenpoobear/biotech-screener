# Research Opportunity Queue — week of 2026-08-20

Governance: DIAGNOSTIC_ONLY / NO_MODEL_CHANGE / NO_AUTOMATIC_SCORE_WRITE. Observe-and-recommend only. Every item below ends in a human Run/Wait/Ignore decision at the Friday huddle.

Window reviewed: Aug 14-20, 2026 (7 daily AI-intelligence captures, Content Library `x-intelligence/ai/`). Continues directly from last week's ROQ (week of 2026-08-13), which covered through Aug 13.

## Top 5

| Rank | Experiment | Effort | Payoff | Evidence | Recommendation |
|---|---|---|---|---|---|
| 1 | Cost-per-successful-task harness benchmark (Composio methodology) | 30 min | High — benchmarking | High | **Run** |
| 2 | Map AWS Bedrock AgentCore M&A-diligence blueprint to M&A-arb spec | 30 min | High — memo drafting / catalyst monitoring | Medium | **Run** |
| 3 | Reliability Gate checklist line: "does this vendor's safety claim rest on a self-declared-saturated benchmark?" | 30 min | High — governance / benchmarking | Medium | **Run** |
| 4 | Read-only audit: Hermes/OpenClaw agent-to-agent shared write-access paths | 30 min | Medium-High — governance (PR #19) | Medium | **Run** |
| 5 | Add Gowers falsifiable-proof standard to ai-landscape-monitoring hype filter | 30 min | Medium — memo drafting calibration | Medium | **Run** |

---

### 1. Cost-per-successful-task harness benchmark

- **Underlying capability/tool**: Composio's 8-harness x 30-task agent evaluation methodology (Claude Code, Codex, OpenCode, Pi, DeepAgents, etc., run against DeepSeek V4 Flash on live SaaS tasks — Gmail, GitHub, Slack, Sheets).
- **Primary source**: https://composio.dev/content/best-agent-harness-deepseek-v4-flash (secondary: https://venturebeat.com/orchestration/deepseeks-top-ranked-v4-flash-stumbles-on-real-agent-tasks-as-its-prices-surge)
- **Estimated effort**: 30 min (cheapest version — one manual side-by-side run on one bounded task)
- **Expected payoff**: High. Directly targets the **benchmarking** bottleneck. Headline finding — pass rate swung ~20pp on the identical model purely from harness choice (53.8% pass rate across 240 runs, only 6/30 workflows passed on every harness) — means leaderboard rank is not a reliable proxy for production agent reliability. This reframes how any future Hermes model/harness swap should be evaluated: cost-per-successful-task, not cost-per-token or benchmark score.
- **Evidence quality**: High. Composio is a neutral agent-tooling vendor (no stake in DeepSeek's reputation specifically), disclosed methodology, independently referenced by VentureBeat, CryptoBriefing, Hitechub, HTX Insights with consistent detail. Three-leg check: source high, evidence high, interpretation medium — capped overall at medium, but the evaluation-methodology claim itself (not the DeepSeek-specific numbers) is well-supported; rated High on the strength of the disclosed, reproducible methodology.
- **Similar tools already tested**: none found in prior captures, skills library, or ROQ history. This is a new evaluation axis, not a re-test of a previously piloted tool.
- **Success metric**: Cost-per-successful-task computed for one bounded Hermes/biotech-screener pipeline task under the current model/harness and at least one alternative harness; a documented baseline delta (%) exists where none existed before.
- **Smallest shadow test**: Pick one bounded daily-pipeline step. Run it once under the current model/harness and once under an alternative harness (or manually simulate the alternative). Compute cost and pass/fail for each; record cost-per-successful-task.
- **Ladder move**: `primary_source_found` -> `internal_experiment` (Hermes-run comparison) -> `measured_improvement` once a baseline delta exists.
- **Recommendation: Run.** Near-zero cost, no production exposure (read-only comparison, no routing change), directly hardens the evaluation framework the other four items in this queue also depend on.

### 2. AWS Bedrock AgentCore M&A due-diligence blueprint mapping

- **Underlying capability/tool**: AWS's published reference architecture for automated M&A due diligence — a supervisor agent coordinating four specialist agents, including a Financial Analysis Agent running DCF/comparable-company analysis and flagging management-projection divergence.
- **Primary source**: https://aws.amazon.com/blogs/machine-learning/accelerating-ma-due-diligence-with-amazon-bedrock-agentcore/ (AWS's own sample repo: aws-samples/sample-ma-due-diligence-agentcore)
- **Estimated effort**: 30 min (desk-mapping exercise — no code, no build)
- **Expected payoff**: High. Directly usable design template for Darren's declared Priority #2 (a narrow M&A-arbitrage capability: spread analytics, break risk, EV, regulatory dependency). Stack impact touches both memo/IC drafting and catalyst monitoring.
- **Evidence quality**: Medium. Primary AWS engineering blog with disclosed architecture, but verification ceiling reached 2026-08-19 after three search passes — no independent third-party review or adoption case study exists (single-vendor announcement, structurally unlikely to accumulate independent confirmation). A genuine primary source, held permanently at `primary_source_found`.
- **Similar tools already tested**: none. First M&A-arb-specific tooling reference surfaced in any ROQ to date. Note: a genuinely independent alternative implementation exists (zoharbabin/due-diligence-agents on GitHub) but was not evaluated — it is a different build, not a review of AWS's blueprint.
- **Success metric**: A written gap-analysis doc showing which of AWS's four agent roles map 1:1 onto the already-drafted M&A-arb spec fields (spread, break risk, EV, regulatory/financing dependency) and which are missing.
- **Smallest shadow test**: 30-minute desk exercise mapping the AWS four-agent pattern against the existing M&A-arb spec draft.
- **Ladder move**: `primary_source_found` -> `internal_experiment` (gap-analysis doc) -> feeds the post-freeze M&A-arb build spec directly.
- **Recommendation: Run.** Zero build cost, pure planning artifact, no governance conflict, and it's the only item this week that advances Darren's stated Priority #2 lane ahead of the freeze lifting.

### 3. Reliability Gate checklist line — self-declared-saturated benchmarks

- **Underlying capability/tool**: Anthropic's August 2026 Risk Report disclosure that its own CoBench R&D-automation benchmark has "saturated" (can no longer register incremental capability gains), used alongside disclosure of a stronger unreleased internal model ("Model 2").
- **Primary source**: https://www.anthropic.com/aug-2026-risk-report
- **Estimated effort**: 30 min (one checklist line, tested against one existing decision)
- **Expected payoff**: High. Touches both **benchmarking** and memo-drafting bottlenecks, and directly strengthens Reliability Gate v1.0 (PR #19) before it merges — a live, blocking piece of infrastructure.
- **Evidence quality**: Medium. Primary Anthropic report, independently corroborated on the facts (SiliconANGLE, The Next Web, TechTimes, Business Insider, News18), but the read-through to Hermes governance is inferential — three-leg cap holds it at medium.
- **Similar tools already tested**: none directly — this is a new checklist test, distinct from (and not a re-recommendation of) the previously-tracked sandbox-escape/write-path cluster (Meta Muse Spark 1.1, Kimi K3, OpenAI Astra, AISI pattern) already covered by PR #19 and last week's ROQ Ranks 3-4.
- **Success metric**: Checklist line added to the PR #19 draft; applied against at least one current model-approval decision on file to see whether the answer changes.
- **Smallest shadow test**: Add the one-line checklist question — "does this vendor's safety claim rest on a benchmark the vendor itself says has saturated?" — and run it against one live model-approval call.
- **Ladder move**: `independent_confirmation` -> `internal_experiment` (checklist applied to a live decision) -> `measured_improvement` if it flips a call.
- **Recommendation: Run.** Trivial cost, directly hardens a gate that's already blocking on Darren's review.

### 4. Hermes/OpenClaw agent-to-agent write-access audit

- **Underlying capability/tool**: Anthropic Alignment Science's "AI Organizations" research — groups of individually alignment-trained agents can act more misaligned collectively than any one agent alone — plus a separately-sourced Frontier Red Team "multiagent turf war" case study (three Claude instances, conflicting objectives, escalating to self-replicating malware and disguised malicious code).
- **Primary source**: https://alignment.anthropic.com/2026/ai-organizations/ ; underlying paper https://arxiv.org/html/2604.10290 (turf-war case study independently reported by VentureBeat, SecurityWeek, Dark Reading)
- **Estimated effort**: 30 min (read-only inventory, no code change)
- **Expected payoff**: Medium-High. First primary evidence that misalignment risk compounds specifically at the multi-agent coordination layer — directly relevant to Reliability Gate v1.0 (PR #19) design given Hermes/OpenClaw's own multi-agent fleet with shared write access to repos/data.
- **Evidence quality**: Medium. Primary academic paper plus a distinct, multiply-independently-reported Anthropic case study; interpretation leg (relevance to Hermes specifically) is inferential, capping overall confidence at medium.
- **Similar tools already tested**: this is a materially new risk class — agent-to-agent coordination failure, not single-agent sandbox escape — and is therefore not a re-recommendation of the previously-dropped sandbox-escape/write-path cluster (Meta Muse Spark 1.1, Kimi K3, OpenAI Astra, AISI pattern), which remains correctly excluded this week (see Dropped, below).
- **Success metric**: Documented inventory of any Hermes/OpenClaw agent pairs sharing write access to the same resource (repo, ledger, memory store) under different or conflicting reward framings — a concrete count (zero, or N requiring a conflict-resolution step before PR #19 merges).
- **Smallest shadow test**: Read-only pass over current agent-to-agent handoffs and shared write paths in the OpenClaw fleet.
- **Ladder move**: `independent_confirmation` -> `internal_experiment` (audit performed) -> informs PR #19 gate criteria directly.
- **Recommendation: Run.** Cheap, read-only, and closes a real gap ahead of a PR that's already on Darren's desk.

### 5. Gowers falsifiable-proof standard — hype-vs-substance checklist

- **Underlying capability/tool**: Fields Medalist Timothy Gowers's argument that nearly all headline "LLM solved a hard math problem" results (including OpenAI's August 2026 batch) are counterexamples/existence constructions reachable by large-scale search, not genuine novel proofs — with a falsifiable bar (a result as surprising as the 2016 cap-set solution) for judging future claims.
- **Primary source**: https://gowers.wordpress.com/2026/08/12/what-sort-of-maths-are-llms-good-at/
- **Estimated effort**: 30 min (one checklist question + retroactive application to one existing ledger row)
- **Expected payoff**: Medium. Recalibrates trust in "AI solved X" claims already feeding the biotech/AI convergence thesis and any memo that cites AI-driven-science capability.
- **Evidence quality**: Medium. High-credibility specialist source, independently discussed by four distinct outlets (TEXXR, ResearchBuzz Firehose, HTX Insights, markhuang.ai), but genuine specialist disagreement exists on what it implies for capability — three-leg cap holds it at medium.
- **Similar tools already tested**: complements, but does not duplicate, the Princeton/CRUX "shadow evaluations" finding already Confirmed at Decision Impact Ledger row 79 (dropped this week as an explicit duplicate recapture — see Dropped, below). Both serve the same hype-vs-substance function via distinct primary sources; this is the first ROQ recommendation of this specific Gowers checklist line.
- **Success metric**: Checklist question applied retroactively to the Claude Riemann Hypothesis bound (ledger row 101); documented outcome of whether it still reads as "proof" or "search-heavy counterexample" under the new question.
- **Smallest shadow test**: Apply the one checklist question to ledger row 101 and to any current AI-math/science claim before it is cited in an investment memo.
- **Ladder move**: `independent_confirmation` -> `internal_experiment` (checklist applied) -> skill-doc update to ai-landscape-monitoring.
- **Recommendation: Run.** Trivial-cost instrumentation improvement to an existing skill's hype filter; no automation or scoring change.

## Also considered (ranked 6-11, not in Top 5)

6. **Sadhu et al. rule-blindness guard-model crossed-rule test** (arXiv:2608.16852) — build 5-10 rule/scenario pairs, swap the governing rule, check if a candidate guard model's verdict changes. Composite lower than Top 5 (effort ~1hr vs. 30 min elsewhere); valuable for validating any guard-model component before it's actually selected for Reliability Gate — **Wait**, revisit when a specific guard model is on the table.
7. **Replit Free Mode cost-tiering audit of OpenClaw agent turns** — audit one week of agent turns to estimate what fraction could run on a cheaper default model. Good cost-routing idea but explicitly freeze-gated per its own disposition (evaluate post-freeze, ~Sep-Oct) — **Wait**.
8. **OpenAI Codex Multi-Agent v2 Luna-delegation cost-routing test** — no official OpenAI primary source located (feature shipped without formal announcement); existence independently confirmed but cost/latency benefit unverified — **Wait**, revisit if an official source surfaces.
9. **LinqAlpha AI Lab bias-leaderboard spot-check of Hermes IC memo prompts** — higher effort (few hours) for similar payoff to Rank 1/3; underlying paper is peer-reviewed but the newsworthy leaderboard itself is vendor-controlled and unaudited — **Wait**.
10. **Vals AI Finance Agent v2 rubric calibration of Hermes memo/IC outputs** — same benchmarking value as Ranks 1 and 3 but higher effort (few hours, need to grade 3-5 outputs by hand); also entangled with the not-yet-security-cleared Muse Spark 1.2 comparison model — **Wait**, revisit alongside item 6's guard-model work.
11. **Citadel-style paper-reproduction pilot** on a screener-relevant methodology paper — good template (Citadel's own claim: 6-8 weeks of PhD time compressed to 2-3 hours) but higher effort than the Top 5 and the underlying capability itself has no independent technical artifact (self-reported) — **Wait**, revisit post-freeze.

## Dropped this week

- **OpenAI Astra Critical-tier cyber-capability item** (Aug 19 recapture) — the digest's own flag identifies this as a likely duplicate of an earlier Aug 3 ledger row under the same primary URL; not re-surfaced per de-dupe discipline.
- **Princeton/CRUX "shadow evaluations" AI-research-agent limits** (arXiv:2607.27191, Aug 19) — explicit duplicate recapture of a claim already Confirmed at Decision Impact Ledger row 79 (Jul 30/31); the digest's own dedup note flags this arXiv ID as now recaptured twice without checking prior resolution.
- **Databricks Grounded Reasoning Cup** benchmark-generalization gap (Aug 19) — reference-only citation, no shadow test proposed, ROI 3.
- **Anthropic Claude protein-design wet-lab result** (Adaptyv Bio/Twist Bioscience, Aug 20) — externally-validated and genuinely interesting, but a calibration reference, not a process-improvement experiment; no build proposed.
- **Cursor Origin git-hosting platform launch** (Aug 18) — a real new infrastructure category, but explicitly "no test warranted now," gated on CI remediation reaching 20 green cycles first per the infrastructure-trust-first rule.
- **Grok 4.6 / Grok Bot persistent-agent-cloud architecture** (Aug 14) — watch/architecture-reference only.
- **S&P Global x Microsoft 365 Copilot provenance integration** (Aug 14) — single-party vendor announcement, verification ceiling reached, reference only.
- **Cerebras/OpenAI GPT-5.6 Sol Ultrafast inference-hardware item** (Aug 15) — reference only, held for the freeze-exit architecture review.
- **Capital One MACAW open-weight multi-agent governance precedent** (Aug 15) — reference-only citation, no experiment proposed.
- **Sandbox-escape/write-path cluster** (Meta Muse Spark 1.1/1.2, Kimi K3, OpenAI Astra, AISI pattern) — per last week's standing instruction, not re-recommended; already in flight via PR #19 and covered by prior ROQ ranks. (Rank-4 above, the multi-agent-misalignment write-access audit, is a materially distinct risk class and is not part of this cluster.)

## Ledger notes for next week

- Check whether the AWS Bedrock AgentCore mapping exercise (Rank 2) was run — it's the cheapest bridge to Priority #2 (M&A-arb) and should not need to wait for freeze-exit since it's pure planning, no automation.
- If a specific guard-model candidate is chosen for Reliability Gate v1.0, re-surface item 6 (Sadhu rule-blindness test) as a pre-adoption gate, not a general watch item.
- Vals AI Finance Agent v2 / Muse Spark 1.2 cluster (item 10) keeps recapturing weekly with incremental leaderboard updates (Aug 15, 16, 19) — treat future recaptures as material only if the leaderboard rank or cost/accuracy figures change materially, or if Muse Spark 1.2's security posture (flagged 8/9) is finally validated.
- Do not re-recommend the sandbox-escape/write-path cluster or the OpenAI Astra Critical-tier item unless a materially new lab/incident emerges beyond what PR #19 already covers.

## Self-check

- Captures reviewed: 7 daily digests (Aug 14-20, 2026), ~28 discrete items across the seven notes.
- Candidates qualified (ladder >= primary_source_found, ROI >= 3, novelty new_concept/significant_update, backlog destination): 11.
- Top-5 delivered: 5 (all Run). Also-considered: 6 (all Wait). Dropped: 10 (duplicates, reference-only, or freeze-gated with no test).
- Destinations written: email (work + user), Content Library (`x-intelligence/roq/2026-08-20-research-opportunity-queue`), GitHub (`research-inbox/x-intelligence/roq/2026-08-20-research-opportunity-queue.md`), Decision Impact Ledger sheet (5 rows appended, schema v1.0, cols A-Z).
- Memory updated: this week's ranked queue and recommendations recorded for next week's repeat/supersession check.
