# Recursivity Charter — Biotech Screener

v0.1 DRAFT · 2026-06-29 · Owner: Darren Schulz · Status: proposed, not enforced

**Purpose.** Name which self-improvement loops in the biotech screener are live, which are gated, and which are forbidden — and state the evaluator-integrity checks that protect them. This is a governance artifact, not a model change. It exists so that "recursivity" never silently expands into self-modifying production alpha.

**Scope.** Applies to the biotech-screener pipeline, the OpenClaw/Hermes agent fleet, the IC Council review loop, and the self-improving learning loop. Does not authorize any code, cron, model, or production change by itself.

**Companion documents:** `docs/RECURSIVITY_CONCEPT_MEMO.md`, `docs/RECURSIVITY_DEEP_RESEARCH.md`

---

## 1. Definitions (one line each)

**Live loop** — runs autonomously today; self-completes; no per-instance operator approval required.

**Gated loop** — may run only behind an explicit clearance gate (spec + IC review + operator sign-off + forward-shadow validation).

**Forbidden loop** — must not be built or wired under any circumstance without an explicit charter amendment.

---

## 2. LIVE loops (autonomous, self-completing)

| Loop | Trigger | Bound | Authority |
|------|---------|-------|-----------|
| Inference-time self-refinement (single run) | Per agent run | No durable state change | observe / propose |
| Correction & self-reflection capture (self-improving) | Correction or post-task reflection | Logged, not promoted, until ≥3 recurrence | write to memory/.learnings/ |
| IC Council → LRN entry | Each significant review | 1–3 LRN entries, read-only review | observe-only |
| Process / deterministic-guardrail promotion | LRN reaches ≥3 recurrence | Checklist item, test, fixture, runbook, schema assertion only | future PR |
| Operational-state memory (routine-scoped) | Per-run | WARM, routine-scoped only — never global | write memory |

**Live-loop ceiling:** none of these may alter `final_score`, ranker, selector, gates, event-EV math, position sizing, or cron. If a "process" change would touch any of those, it is reclassified as model-affecting and moves to GATED.

---

## 3. GATED loops (clearance required)

| Loop | Gate sequence | Who approves |
|------|--------------|--------------|
| Model-affecting change (features, weights, thresholds, ranker, selector, final_score, gates, event-EV) | Spec → IC Council full review → operator approval → forward-shadow ≥30 trading days (true-PIT, no backfill) → promote | Operator (sole authority) |
| Archive-based variant search (DGM-style, research-only) | Charter amendment → offline only → shadow-arm evaluation → full Checklist v2 → operator clearance | Operator |
| Any new agent with authority > observe_and_propose | Spec + blast-radius + rollback + governance-tier review | Operator |
| Ranker-IC-dependent promotion | Blocked until Spec 100 Checklist-v2-against-final_score rerun executes | Operator |

**Gate integrity rule:** forward shadow is the only out-of-sample evidence. No promotion may rely on backtest-only evidence, self-grades, or model self-confidence.

---

## 4. FORBIDDEN loops (no build without charter amendment)

- Automatic promotion of any model-affecting change (no human in the loop).
- Self-grading reward — the model's own score/confidence feeding the promotion or reward signal (wireheading vector).
- Open-ended agent-variant search wired to the production path (selector / ranker / final_score).
- Production mutation from Town — Town is Hermes→delivery only; no repo, cron, or runtime write-back.
- Cron dependent on a gateway token — no autonomous job may require a live LLM gateway.
- Unbounded optimization depth against any finite evaluator (reward-hacking deepens with depth).

---

## 5. Evaluator-integrity checks (the anti-reward-hacking layer)

Because reward hacking is a structural equilibrium under any finite evaluation, the evaluator itself must be defended:

1. **Field-declaration rule** — every IC measurement must declare which score field and universe it measures; an IC number on the wrong field is misattribution, not evidence.

2. **Evaluator-degradation row** — IC Council Step 6 decision matrix includes a `evaluator-integrity` row: `pass/watch/fail` asking "Could this change have degraded the evaluator rather than improved the signal?"

3. **Decoupling** — promotion signal must be independent of the system's own output (forward shadow + operator, never self-score).

4. **Depth bound** — architecture-freeze / checkpoint gates cap iteration count against any single evaluator.

5. **PIT/provenance audit** — source dates ≤ snapshot date; forward returns never enter features; generated outputs never feed input hashes unless intentionally frozen.

---

## 6. Amendment procedure

This charter is Tier-4-equivalent (governance text). Changes require a dated memo and operator approval — never a silent edit. Each amendment records: date, what changed, why, and which loop class moved.

---

**Rationale sources:** `docs/RECURSIVITY_DEEP_RESEARCH.md`; "Reward Hacking as Equilibrium under Finite Evaluation" (2026); Anthropic "Natural Emergent Misalignment from Reward Hacking" (2025); "Does Self-Evaluation Enable Wireheading?" (2026).

**Internal cross-references:** `skills/self-improving/SKILL.md`, `skills/biotech-ic-council/SKILL.md`, `skills/ic_evaluation/SKILL.md`, `skills/screener_ops/SKILL.md`
