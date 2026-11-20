# CLAUDE.md — Wake Robin Capital Management Biotech Screener

## Project Identity
This is an institutional-grade biotech investment screening system.
Outputs must be reproducible, auditable, and deterministic.
Every decision must be traceable to a data source with a timestamp.

## North Star Rule
Backtest systems NEVER directly modify production screening behavior.
They produce evidence and proposals only. Governance review required before
any backtest finding changes a production signal weight.

## CCFT Principles (Non-Negotiable)
All data fixtures must be:
- Canonical: single authoritative source per data type
- Complete: no silent nulls or missing fields without explicit flags
- Frozen: historical snapshots are immutable once written
- Timestamped: data_available_timestamp <= as_of_date always enforced

## Active Ruleset

Current: `8887576e` (v1.14.0). Pinned in `run_screen.py` and `run_phase2_snapshot_delta.py` (must stay in sync).
**See `.claude/rules/operational-state.md` for full ruleset details, settings, and manifest.**

---

## Architecture Freeze Status

**Scoped production model freeze in effect (as of 2026-06-20).** Ranker, selector, sizing, final_score, portfolio, and snapshot files are frozen. Safe lanes: expectation verification, Event EV shadow, Sci-Cart diagnostics, observability, Hermes read-only. Lift requires explicit operator clearance.
See `.claude/rules/operational-state.md` for freeze scope and post-freeze priorities.

---

## Subagent Delegation

When a major biotech model change, validation finding, backtest correction, governance status change, or alpha/investability conclusion occurs, use the `model-doc-and-skill-sync` subagent to update `docs/MODEL_DOCUMENTATION.md` and synchronize any affected biotech skills. Documentation remains the source of truth; skill updates are downstream instruction hygiene only. No model, ranker, selector, sizing, production, or trading behavior may change through this agent.

When universe coverage, stale tickers, XBI/IBB constituents, missing biotech names, delisted names, ticker mapping, or ETF coverage drift are discussed, use the `universe-hygiene-auditor` subagent. The agent may write audit artifacts and proposals only; it must not directly mutate the production universe without separate operator approval.

---

## PIT Rules

1. **Never call the historical set "true PIT"** unless archived raw inputs, archived code, AND archived derived artifacts all exist as-of each date.
2. Historical benchmark outputs must carry `pseudo_pit_version` (1=contaminated, 2=cleaned).
3. Benchmark reruns must use the PIT-aware paths: `--pit-mode survivorship` or `--pit-mode full`.
4. Long-history conclusions are **provisional** until PIT-v2 financial rerun lands.
5. The forward monitor is the only true out-of-sample evidence. Accumulate it.

---

## Before Writing Any Code
1. State which module this change belongs to
2. Identify whether this is a new signal, validation change, or infrastructure change
3. Write the failing test FIRST — show me the red test before any implementation
4. Confirm no look-ahead bias: what is the data_available_timestamp?
5. Classify the diff by governance tier (Tier 0-4 per governance/AGENT_ROUTING_POLICY.md)

## Coding Standards
- All outputs: encoding='utf-8', lineterminator='\n', quoting=csv.QUOTE_MINIMAL
- SHA256 hash every scored output for audit trail
- Identical inputs must produce byte-identical outputs — no random seeds, no datetime.now()
- Use Point-in-Time fixtures — never fetch live data in tests

## What NOT To Do
- Do not refactor and add features in the same commit
- Do not change production agent weights without an ablation test showing Sharpe delta
- Do not use PubMed h-index API, options flow, or CapIQ — see approved data sources
- Do not introduce survivorship bias — graveyard list is at data/graveyard/

## Test Requirements
Every new signal must include:
1. Unit test with known fixture input -> expected output
2. Leakage test confirming data_available_timestamp compliance (see Trust Buckets in `.claude/rules/research-backtest.md` for signal safety assessment)
3. Ablation test stub showing Sharpe contribution >= 0.1

---

## Scoped Rules Reference

**See these files for detailed operational and governance context:**

- **`.claude/rules/operational-state.md`** — Active ruleset, 13F cycle, freeze dates, spec status, forward shadow IC. Updated weekly.
- **`.claude/rules/research-backtest.md`** — Evidence hierarchy, dead lanes, benchmark commands, promotion story. Load during research sessions.
- **`.claude/rules/production-pipeline.md`** — Decision engine architecture, pipeline flow, cron behavior, cache warming. Path-scoped to pipeline files.
- **`.claude/rules/governance.md`** — Tier definitions, promotion path, 13F onboarding, insider diagnostic, expectation layer, expression policy.
- **`.claude/rules/external-intel.md`** — OpenClaw status, Hermes competitive frame, industry AI adoption, developer profile.

---

## Git Remotes & PR Policy

This repository has two GitHub remotes. Both must receive every feature branch.

| Remote | GitHub repository | Role |
|--------|-------------------|------|
| `origin` | `Warrenpoobear/biotech-screener` | **Primary** |
| `WR-SW-Dev` | `WR-SW-Dev/biotech-screener` | Mirror — must stay in sync |

**Never push feature work directly to `main` on either remote.** A `pre-push`
hook (INC-2026-06-20-AUTOPUSH) blocks non-interactive pushes to `main`; that
guard is a backstop, not the policy. The policy is: branch, push, PR.

For every change, push the current branch to both remotes, then open one PR per
GitHub repository — commits and branches replicate across remotes, but **PRs are
repository-specific, so two separate PRs are always required**, with the same
title and description.

```bash
branch=$(git branch --show-current)

git push --set-upstream origin     "$branch"
git push --set-upstream WR-SW-Dev  "$branch"

# --fill is fine HERE: "$branch" is checked out locally, so gh can read its
# commits for the title and body. It does NOT work for the mirror sync below,
# where the branch exists only on the remote.
gh pr create --repo Warrenpoobear/biotech-screener --base main --head "$branch" --fill
gh pr create --repo WR-SW-Dev/biotech-screener     --base main --head "$branch" --fill
```

Verify both pushes and report both PR URLs.

**Stop conditions — report, never force-push:**

- A push or `gh pr create` fails.
- The two `main` branches have diverged. Check before opening the second PR:
  ```bash
  git fetch origin && git fetch WR-SW-Dev
  git rev-list --left-right --count origin/main...WR-SW-Dev/main
  ```
  Any result other than two zeros means a branch cut from one `main` will carry
  the other's missing commits into its PR diff. Opening that PR risks merging
  unrelated work across repositories. Reconcile the two `main` branches first,
  or cut a second branch from the other remote's `main` and cherry-pick the
  change onto it so each PR is a clean single-purpose diff.

Keeping both `main` branches synchronized is the standing expectation; treat
drift between them as a problem to fix, not a condition to work around.

**`main` sync lands by pull request, and is currently MANUAL.**
`.github/workflows/mirror-main-to-wr-sw-dev.yml` runs after every merge to
`origin/main`. `WR-SW-Dev/main` is a **protected branch**, so a direct push is
declined (`protected branch hook declined`) — the workflow therefore pushes the
`mirror/main-sync` branch and opens a PR against `main` there. Merging that PR
is a human step, by design; the protection is respected, not bypassed.

**The workflow is dormant until `WRSWDEV_MIRROR_TOKEN` exists.** Without the
secret it logs a notice and exits 0 — deliberately a skip, not a failure, so it
does not put a red X on every merge. Check the workflow's run summary: `Action:
skipped` means dormant, not broken. To enable it, add a fine-grained PAT on
`WR-SW-Dev/biotech-screener` with **Contents: read and write** and **Pull
requests: read and write** — no bypass of the branch protection is needed.

Until then, sync by hand (no token or admin required — a branch push to that
repo is allowed, only `main` is protected):

```bash
# 1. Safety check. Three-dot diff = changes made on the MIRROR side since the
#    branches parted. Must be EMPTY: anything here is content the mirror has
#    and origin does not, and syncing would revert it. Two-dot (`..`) does NOT
#    answer this — it is just the forward diff inverted, and always non-empty.
git fetch origin && git fetch WR-SW-Dev
git diff --stat origin/main...WR-SW-Dev/main     # empty => safe to proceed

# 2. Push origin/main onto the mirror's sync branch.
git push WR-SW-Dev main:refs/heads/mirror/main-sync

# 3. Open the PR. Do NOT use --fill: it derives the title and body by
#    inspecting the head branch LOCALLY, but mirror/main-sync only ever exists
#    on the remote (step 2 pushes main directly to that ref, creating no local
#    branch). gh then fails with "Use '--' to separate paths from revisions".
gh pr create --repo WR-SW-Dev/biotech-screener \
  --base main --head mirror/main-sync \
  --title "mirror: sync main from Warrenpoobear/biotech-screener" \
  --body "Mirror sync from Warrenpoobear/biotech-screener, carried verbatim. No commits originate here. Verified before pushing: three-dot diff origin/main...WR-SW-Dev/main is empty."
```

Then merge that PR. Confirm parity by **tree**, not by SHA — the merge commit
lives only on the mirror, so the histories differ permanently even when the
content is identical:

```bash
git rev-parse origin/main^{tree} WR-SW-Dev/main^{tree}   # equal => in sync
```

It never force-pushes and never pushes to `main`. Before proposing anything it
checks whether the mirror carries content `origin` lacks and fails loudly if so,
rather than opening a PR that would revert someone's work.

So drift does **not** self-heal any more — expect an open mirror PR on
WR-SW-Dev after each merge here, and merge it to close the gap. If the
divergence check above is non-zero, look for that PR first.

Note the two branches will differ in *history* even when fully in sync, because
merging the mirror PR creates a merge commit that `origin` never sees. Compare
**trees**, not commit SHAs, when asking whether the mirror is current:

```bash
git rev-parse origin/main^{tree} WR-SW-Dev/main^{tree}   # equal => in sync
```

This does **not** replace the dual push and two PRs — the mirror only moves
`main`. Feature branches still go to both remotes, and each GitHub repository
still needs its own PR.

---

## Key File Locations

| Area | File |
|------|------|
| Main orchestrator | `run_screen.py` |
| Decision Engine | `decision_engine.py` |
| Selector Engine | `selector_engine.py` |
| Ranker Engine | `ranker_engine.py` |
| Daily Production | `tools/run_daily_production.py` |
| Shadow Portfolio | `tools/live_shadow_portfolio.py` |
| Promotion Battery | `scripts/research/run_promotion_battery.py` |
| Ruleset Manifest | `production_data/decision_rulesets/manifest.json` |
| Governance Policy | `governance/AGENT_ROUTING_POLICY.md` |
| Agent Registry | `agents/AGENT_REGISTRY.json` |
| **Live Portfolio Rules** | `production_data/AGENTIC_ACCOUNT_RULES.md` |
| **Phase 2 Entry Prices** | `production_data/phase2_entry_prices.json` |

## Live Portfolio (Agentic Account 802349084)

A live Robinhood account (802349084) tracks the model top-30 in real money. Operational rules at `production_data/AGENTIC_ACCOUNT_RULES.md`. Key rules: weekly Monday equal-weight rebalance, review at drawdown vs XBI ≤ −10pp and hard exit at ≤ −15pp (changed 2026-11-15 from −2pp), IRAs managed independently. Claude Code skills for execution: `biotech-rebalance`, `biotech-portfolio-status`, `biotech-governance-check`, etc. (see `~/.claude/skills/biotech-*/`). Entry prices for P&L tracking in `production_data/phase2_entry_prices.json` — do NOT use screener snapshot reference prices as entry prices.
