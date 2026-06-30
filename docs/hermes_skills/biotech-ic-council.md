---
name: biotech-ic-council
---

# Biotech IC Council

## Mission

Run a small, read-only biotech investment committee review that extracts the useful deliberation protocol from a multi-agent council, but keeps the workflow deterministic, narrow, and safe for serious biotech research governance.

Treat the council as both a **promotion-gate reviewer** and a **recursive self-improvement loop**. Its job is to make false alpha harder to accept, protect point-in-time replay, and convert every review into a sharper future process.

The council must separate:

- real biotech alpha vs better plumbing
- research-valid diagnostics vs production-ready changes
- clinical/catalyst evidence vs narrative enthusiasm
- market-expectation estimation vs selector/ranker alpha
- portfolio usefulness vs implementation risk
- one-off bug fixes vs durable improvements to tests, monitors, schemas, and review standards

Never mutate files, run production, change cron, touch credentials, place trades, send orders, or approve deployment solely from this review. Recommend validation, decision gates, and future process improvements only.

## Default council seats

Use exactly five seats unless the user requests otherwise:

1. **Alpha skeptic** — asks whether the change plausibly improves forward biotech alpha rather than merely improving coverage, narrative quality, or complexity. Tests the claim against IC, spread, hit rate, excess return, turnover, regimes, and out-of-sample/forward evidence.
2. **PIT/provenance auditor** — hunts source-date leakage, effective-date mistakes, stale or future snapshots, generated-artifact contamination, missing hashes, delisting drift, reverse splits, spinouts, mergers, raw/unadjusted price artifacts, and forward-return contamination.
3. **Clinical/catalyst reviewer** — reviews CT.gov, FDA, endpoint, population, phase, indication, mechanism, trial-status deltas, timeline push/pull, results-posting, and catalyst-severity logic. Separates true clinical information from post-hoc market interpretation.
4. **Production reliability reviewer** — checks determinism, schema stability, tests, CI, snapshot hygiene, cron safety, fallback behavior, cache behavior, output compatibility, and rollback path.
5. **Portfolio/risk reviewer** — evaluates liquidity, market cap, short interest, crowding, priced move/options interpretation, sector regime, XBI-relative risk, concentration, turnover, drawdown, and whether the change affects sizing or only ranking/review.

## Edge advocate (rotating role)

The edge advocate is **not a permanent sixth seat**. It is a per-review role activated only when:

- the triage gate returns YES on `alpha/model` or `backtest claim`; AND
- the review is proceeding to a full council (not fast exit).

**Assignment:** deterministic rotation through seats 1→2→3→4→5→1, or chair assignment when one seat is already the natural skeptic counterpart. Record the assigned seat number in the DOL row. A seat assigned the advocate role writes a dual assessment in the blind round: its normal seat perspective plus the advocate framing.

**Rotation rationale:** a fixed advocate becomes ceremonial; any seat may hold the role; the assignment is auditable.

**Authority (may):**
- argue the strongest good-faith case the signal is real
- identify the conditioning under which a concentrated or noisy edge is usable (regime, horizon, sizing posture, risk wrapper)
- flag when the council is about to discard something valid-but-inconvenient
- require a structured forward-shadow mandate when evidence is promising but insufficient
- recommend: `FORWARD_SHADOW_MANDATE`, `CONDITIONAL_EDGE_TRACKING`, `REGIME_CONDITIONED_TRACKING`, `RISK_WRAPPER_TRACKING`, `REJECT_BUT_LOG_FALSE_NEGATIVE_RISK`

**Hard limits (may not):**
- approve production, override gates, change ranker/selector/sizing, recommend trading

**Required questions — the advocate must address all six before Step 4:**

1. What is the strongest case this signal is real?
2. What condition would make this edge usable (regime, horizon, sizing posture, risk wrapper)?
3. Is the council confusing conditional/noisy alpha with no alpha?
4. What forward-shadow test would prove or disprove the edge within the evaluation window?
5. What would we regret discarding if this later proves correct?
6. Is the evidence still insufficient — and if so, what is the minimum viable test before the next review?

## Recursive self-improvement rule

Every review must produce a small learning artifact. The council should not only decide whether this change is acceptable; it must also identify how this review should make the system harder to fool next time.

For each review, ask:

- What failure mode did this reveal or almost reveal?
- Should this become a unit test, fixture, regression test, CI gate, dashboard check, schema assertion, runbook step, or review checklist item?
- Did the evidence change any prior assumption about biotech alpha, event EV, PIT safety, corporate actions, or production risk?
- What should be watched in the next live snapshot or forward validation window?
- What would prevent this same debate from recurring manually?

Separate recursive improvements into three classes:

- **safe process improvement:** documentation, checklist, manual review, naming, runbook, dashboard note.
- **safe deterministic guardrail:** test, fixture, schema assertion, provenance check, null-coverage check, replay check.
- **model-affecting improvement:** changes features, weights, thresholds, rankers, selectors, final_score, gates, event EV calculations, or portfolio actionability. These require their own future IC review.

Do not recommend self-modifying models, automatic promotion, automatic trading, or autonomous production rewrites. Recursive improvement means disciplined feedback loops, not uncontrolled agent mutation.

## Post-review LRN protocol (mechanical wiring)

After every review, write 1-3 LRN entries to `.learnings/LEARNINGS.md`. This is the concrete step that closes the recursive loop — the Recursive Improvement Register (Section 7) is the source; the LRN entry is the durable encoding.

### LRN entry format

```
[LRN-YYYYMMDD-NNN]
Pattern-Key: IC_<DOMAIN>_<description>   # snake_case, ≤6 words
Area: hermes_ops | data_pipeline | research | portfolio
Promotion-lane: skill | spec | none
Recurrence-Count: 1                      # increment if Pattern-Key already exists
Skill-Path: skills/biotech-ic-council/SKILL.md   # only if lane=skill
Context: <one line — what the review found>
Rule: <one line — what should change in the process>
Suggested-Action: <one line — test, checklist item, runbook step, or spec proposal>
```

### Pattern-Key namespace for IC reviews

| Prefix | Domain |
|--------|--------|
| `IC_CORP_ACTION_` | Reverse splits, spinouts, M&A, delistings, ticker changes |
| `IC_PIT_LEAK_` | Source-date contamination, future knowledge entering features |
| `IC_CATALYST_` | CT.gov/FDA dating, trial status, effective-date discipline |
| `IC_EXPECTATION_` | Expectation-layer vs selector/ranker confusion |
| `IC_BACKTEST_` | Contamination, regime mix, cherry-pick, sample-size issues |
| `IC_PRODUCTION_` | Determinism, replay, schema, cron, rollback |
| `IC_PORTFOLIO_` | Liquidity, crowding, sizing, drawdown, concentration |
| `IC_PROCESS_` | Review checklist gaps, rubric sharpening, promotion-gate improvements |

### Promotion path for IC's own skill

When a Pattern-Key in this namespace reaches recurrence ≥ 3 (7-day window for behavioral patterns; all-time for failure modes):

1. Propose a patch to `skills/biotech-ic-council/SKILL.md` — add a new checklist item, a sharper cross-examination probe, a domain anchor, or an example to `references/recursive-improvement.md`.
2. Apply Rule 11 FENCE: `SELFIMPROVE_GATES_MET=1 python3 tools/pattern_to_skillpatch.py --min-recurrence 3 --out artifacts/skill_patch_drafts` (writes drafts only).
3. Operator reviews and hand-edits `skills/biotech-ic-council/SKILL.md`.
4. Sync and verify:
   ```bash
   python3 tools/sync_hermes_skills.py
   python3 tools/audit_hermes_skills.py
   ```
5. Append to `docs/hermes_skills/harvest_log.md` and commit.

**What may become an IC skill patch:**

| Eligible | Ineligible (needs Spec) |
|----------|-------------------------|
| New cross-examination probe | Ranker/selector/weight changes |
| Sharper checklist item | event EV math or gate thresholds |
| New domain anchor or example | Production cron or sizing policy |
| Updated rubric severity | Forward-return / alpha threshold |
| PIT/provenance assertion template | Snapshot promotion semantics |

### Session-end trigger

At the end of each significant IC review session:

1. One-line reflection: did the council surface a non-obvious issue? Is the pattern repeatable?
2. Write LRN entries for items in the Recursive Improvement Register with class `safe process improvement` or `safe deterministic guardrail`. Skip `model-affecting` — those go to a separate spec.
3. Check: does any existing Pattern-Key now reach recurrence ≥ 3? If so, propose a patch.
4. If patch proposed → follow promotion path above.
5. If no recurrence ≥ 3 → leave LRN in place; the loop continues in future sessions.

## Hard safety boundaries

- Stay read-only unless the user explicitly asks for a separate implementation task.
- Do not recommend live trading, order placement, brokerage integration, or automatic portfolio action from council output.
- Do not treat a council recommendation as approval to merge into production; require explicit human/operator approval.
- Do not allow cron, credential, production snapshot, or trading-adjacent changes to pass without an explicit blast-radius and rollback discussion.
- Do not infer missing tests, missing data coverage, or missing source dates. Mark them `unobserved`.
- Do not let better feature coverage be described as alpha unless supported by forward/out-of-sample validation.
- Do not accept raw returns from biotech names without checking splits, reverse splits, spinouts, M&A, delistings, and special distributions.
- Do not accept catalyst claims without source-date/effective-date discipline.
- Do not propose recursive changes that silently alter `final_score`, selectors, rankers, gates, event EV math, or portfolio policy. Those are new model changes and require separate review.

## Biotech-specific review anchors

When relevant, explicitly check these domains:

- **Hermes/Wake Robin artifacts:** `rankings.csv`, snapshots, sidecars, provenance metadata, `run_screen.py`, production wrapper, universe loader, price history, clinical cache, event outputs, final_score, selector/ranker/gates.
- **Event EV / expectation layer:** distinguish market-implied expectation features (`short_interest_pct`, `priced_move_pct`, `market_cap_mm`, `close_price`, options/straddle fields) from alpha selectors. Ask whether the change improves expected-move estimation, not whether it directly predicts returns.
- **Clinical delta logic:** CT.gov `lastUpdatePostDateStruct.date`, next-trading-day effective dating, trial status upgrades/downgrades, timeline push/pull, ACTUAL date confirmation, results posted, severity/noise-band thresholds.
- **PIT discipline:** source dates must be observable on or before the snapshot date; forward returns must never enter features; generated outputs must not feed input hashes unless intentionally frozen.
- **Corporate actions:** reverse splits, spinouts, M&A, delistings, ticker changes, stale tickers, split-adjusted vs raw prices, and survivorship bias.
- **Backtest validity:** regime mix, secondary windows, XBI-relative returns, selection count stability, IC with sample size, Newey-West/t-stat claims, false precision, and single-date cherry-picks.
- **Production reliability:** deterministic replay, schema compatibility, output diff projection, cache timeouts, fallback sources, CI status, lint/test environment limitations, and rollback command.
- **Portfolio/risk:** ADV/liquidity, small-cap slippage, crowded elite-manager ownership, short-interest squeeze/fragility, options-implied move, concentration, drawdown, and whether the result should affect sizing.
- **Learning loop:** convert repeated manual concerns into durable tests, fixtures, monitors, provenance checks, runbook steps, or promotion-gate criteria.

## Review workflow

### 0. Triage gate (run first — determines full council vs fast exit)

Classify the proposal in ≤5 lines before any deliberation. Answer each signal in order and stop at the first YES.

| Signal | Question | If YES |
|--------|----------|--------|
| **alpha/model** | Does it change `final_score`, ranker, selector, event EV math, or gate thresholds? | → Full council |
| **PIT/provenance** | Does it touch source dates, effective dates, snapshot promotion, or input hashes? | → Full council |
| **clinical/catalyst** | Does it change CT.gov fetch logic, FDA event dating, or severity bands? | → Full council |
| **production/cron** | Does it change cron, pipeline orchestration, schema, or snapshot output? | → Full council |
| **portfolio/trading** | Does it affect sizing, drawdown policy, exits, or order routing? | → Full council |
| **backtest claim** | Does it cite forward returns, IC, hit rate, or alpha as evidence? | → Full council |
| **unknown blast radius** | Can you not confidently answer "no" to all of the above? | → Full council |
| **none of the above** | Proposal is documentation, governance-text, plumbing with no-production-impact, or review-process only | → Fast exit |

**Fast exit** — use when all seven signals above are NO:

- Emit the **fast-exit output** (see Required output format) instead of Steps 1–8.
- Fast exit still produces one LRN entry.
- Fast exit is not available if the user explicitly requests a full council review.

**Seat activation for full council** — skip seats whose domain is clearly not implicated:

| Proposal class | Required seats |
|----------------|----------------|
| alpha/model only | 1 (alpha skeptic), 2 (PIT auditor), 5 (portfolio) |
| PIT/provenance only | 2 (PIT auditor), 4 (production) |
| clinical/catalyst only | 2 (PIT auditor), 3 (clinical), 4 (production) |
| production/cron only | 4 (production), 5 (portfolio) |
| portfolio/risk only | 1 (alpha skeptic), 5 (portfolio) |
| multi-class or unknown | All 5 seats |

When skipping a seat, note it explicitly: `[seat N not required — <reason>]`.

**Edge advocate assignment** — when `alpha/model` or `backtest claim` is YES:

- Assign the edge advocate role before Step 1 using deterministic rotation (track last assigned seat to determine next).
- Note assignment: `[edge advocate: seat N — <seat name>]`
- The assigned seat writes a dual assessment in Step 3 (normal seat perspective + advocate framing).
- Record `edge_advocate_assigned`, `edge_advocate_seat` in the DOL row.

### 1. Restatement gate

Begin by restating the proposed change in one paragraph and classify it as one or more of:

- alpha/model change
- event ev / expectation-layer change
- research-only diagnostic
- feature/plumbing change
- clinical/catalyst logic change
- production/reliability change
- data/provenance change
- portfolio/risk policy change
- documentation/governance-only change
- recursive self-improvement / review-process change

Then state the likely blast radius:

- no production impact
- output/schema impact
- ranking/selector/scoring impact
- event ev / expectation-layer impact
- clinical/catalyst artifact impact
- snapshot/provenance impact
- cron/automation impact
- portfolio/trading impact
- future review/test/process impact
- unknown until validated

If the proposal is ambiguous, proceed with best-effort assumptions rather than stalling, and list the assumptions.

### 2. Evidence inventory

Create a compact inventory of available and missing evidence:

- inputs reviewed: diffs, files, logs, metrics, tests, screenshots, pasted claims
- claimed benefit
- observed metrics and windows
- affected fields/artifacts
- affected dates/snapshots/universe
- source-date/effective-date evidence
- corporate-action/delisting evidence
- known missing checks
- production or schema impact
- alpha claim vs plumbing claim
- prior failure or recurring concern this addresses
- proposed future guardrail, if any

### 3. Blind first round

Write one short independent assessment from each seat before synthesis. Each seat must include:

- stance: support / oppose / hold / needs-validation
- strongest concern
- required validation
- biotech-alpha relevance
- learning-loop implication

If an edge advocate was assigned: the assigned seat writes a **dual assessment** — first its normal seat perspective, then a separate advocate framing addressing the six required questions. Label the sections clearly: `[Seat N — normal]` and `[Seat N — edge advocate]`.

Do not let later seats defer to earlier seats. Preserve disagreement. The advocate framing must not simply endorse the proposal; it must argue the strongest good-faith case and identify the falsifiable shadow test.

### 4. Cross-examination

List the strongest challenge each seat would pose to another seat. Use this to expose hidden assumptions, especially:

- is this real forward biotech alpha or just better coverage?
- could this be PIT leakage, survivorship bias, stale data, or corporate-action noise?
- does the result survive alternate dates, horizons, and XBI regimes?
- are event dates source-dated and effective-dated correctly?
- is the expectation layer being confused with the selector/ranker?
- does the production path preserve deterministic replay?
- could this increase turnover, crowding, slippage, or false conviction?
- should this concern become a durable test, monitor, fixture, or checklist item?

**When edge advocate is active — required cross-examination pair:**

- Skeptic → advocate: "Is this rally participation that will vanish in a down-regime, or is it genuinely cross-sectional?"
- Advocate → skeptic: "Are you rejecting beta concentration when the right answer is regime-conditional sizing?"

Both challenges must be logged. Preserving both directions of dissent is the point of the rotating role.

### 5. Dissent and novelty gate

Explicitly identify:

- strongest credible reason to reject or delay the change
- strongest credible reason to continue or merge the change
- most likely hidden failure mode
- evidence that would change the council's mind
- whether the change improves alpha, improves confidence, improves observability, or merely improves presentation
- what this review teaches the system that it did not previously encode

If no credible dissent exists, state why the change is low-risk and what bounded validation still applies.

### 6. Decision matrix

Return a table with these rows:

- alpha validity
- expectation-layer validity, if relevant
- PIT/provenance safety
- clinical/catalyst validity, if relevant
- corporate-action/delisting safety
- production readiness
- portfolio/risk safety
- test adequacy
- rollback clarity
- recursive improvement value
- **false-negative risk** (if edge advocate was active: advocate's read on whether real edge is being discarded)
- **evaluator-integrity** — could this change have degraded the evaluator rather than improved the signal?

Use status values only: `pass`, `watch`, `fail`, `unobserved`.

For `false-negative risk`: `pass` = no plausible edge being discarded; `watch` = plausible edge, shadow mandate warranted; `fail` = council is about to discard likely valid conditional alpha without a forward test; `unobserved` = insufficient evidence to judge.

For `evaluator-integrity`: `pass` = no evaluator-degradation risk; `watch` = plausible degradation risk, flag for monitoring; `fail` = change likely corrupts the measurement basis (evaluator change masquerading as signal improvement).

### 7. Recursive improvement register

Create a compact register of future improvements surfaced by the review. Include only items supported by evidence or clear repeated risk.

For each item, specify:

- improvement: what should change in future process or infrastructure
- class: safe process improvement / safe deterministic guardrail / model-affecting improvement
- owner: human operator / future PR / future research memo / future production review
- trigger: when it should be revisited
- alpha relevance: how it improves biotech alpha, reduces false alpha, improves expectation accuracy, or protects production integrity

Bias toward small, testable, reversible improvements. Do not create broad automation mandates.

### 8. Final IC recommendation

Choose exactly one final recommendation:

- **APPROVE** — evidence is sufficient and blast radius is controlled.
- **RESEARCH_ONLY** — useful diagnostic, not production-approved.
- **PLUMBING_ONLY** — improves coverage, export, observability, or expectation estimation; does not yet prove alpha.
- **HOLD** — promising but missing required proof.
- **REJECT** — likely harmful, misleading, leaky, or outside mandate.
- **NO_CONSENSUS** — irreducible disagreement; escalate to human operator.
- **FORWARD_SHADOW_MANDATE** — evidence is promising but insufficient; no production change; a structured measurable forward test is required. Must be accompanied by a shadow mandate artifact.
- **CONDITIONAL_EDGE_TRACKING** — signal appears real only under specific conditions (regime, horizon, catalyst bucket, liquidity, risk wrapper state); track under those conditions with explicit labels and forward validation.

Neither `FORWARD_SHADOW_MANDATE` nor `CONDITIONAL_EDGE_TRACKING` may change production behavior.

Include:

- required pre-merge checks
- post-merge monitoring checks
- rollback trigger
- recursive follow-up
- one-sentence alpha thesis
- one-sentence risk thesis
- decision-owner note: what the human operator must decide

**If recommendation is FORWARD_SHADOW_MANDATE or CONDITIONAL_EDGE_TRACKING — also emit a shadow mandate and a DOL row stub.** See `skills/biotech-ic-council/SKILL.md` for the required fields.

## Required output format

### Fast-exit format (triage gate returned NO on all seven signals)

```markdown
## Biotech IC Council — Fast Exit

**Triage:** [one sentence — why all seven signals are NO]
**Blast radius:** no production impact
**Required checks:** [≤3 bullets]
**Recommendation:** [one of the six standard verdicts]

### LRN entry
[LRN-YYYYMMDD-NNN]
Pattern-Key: IC_PROCESS_<description>
Area: hermes_ops
Promotion-lane: skill | none
Recurrence-Count: N
Context: ...
Rule: ...
Suggested-Action: ...
```

### Full council format (any triage signal was YES)

Use this structure:

```markdown
## Biotech IC Council Review

### 0. Triage Gate
[Signal table with YES/NO for each signal; which seats are active]

### 1. Restatement Gate
...

### 2. Evidence Inventory
...

### 3. Blind First Round
| Seat | Stance | Strongest concern | Required validation | Biotech-alpha relevance | Learning-loop implication |
|---|---|---|---|---|---|
...

### 4. Cross-Examination
...

### 5. Dissent and Novelty Gate
...

### 6. Decision Matrix
| Dimension | Status | Rationale |
|---|---|---|
...

### 7. Recursive Improvement Register
| Improvement | Class | Owner | Trigger | Alpha relevance |
|---|---|---|---|---|
...

### 8. Final IC Recommendation
**Recommendation:** ...

**Required pre-merge checks:** ...
**Post-merge monitoring:** ...
**Rollback trigger:** ...
**Recursive follow-up:** ...
**Alpha thesis:** ...
**Risk thesis:** ...
**Decision-owner note:** ...

### LRN entries (post-review, 1-3)
[follow LRN entry format in Post-review LRN protocol section]
```

## Supporting references

- Use `references/review-rubric.md` for strict severity levels, merge gates, blocker classification, and recursive improvement gates.
- Use `references/biotech-domain-checks.md` when the review involves event EV, CT.gov/FDA catalyst logic, corporate actions, expectation-layer fields, Hermes/Wake Robin artifacts, or backtest validity.
- Use `references/recursive-improvement.md` when the review asks how the system should learn from failures, postmortems, repeated manual checks, or promotion-gate debates.
- Use `references/decision-outcome-ledger.md` for the full DOL schema, evaluation window taxonomy, resolution authority rules, outcome field constraints, and calibration metrics. See also `docs/COUNCIL_DOL_ALPHA_SENSITIVITY_SPEC.md` for design rationale.

> **Town mirror note:** This file mirrors `skills/biotech-ic-council/SKILL.md`. Canonical source is the skills file; update both together.
