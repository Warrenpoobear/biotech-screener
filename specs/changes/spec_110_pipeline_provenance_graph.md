# Spec 110 — Pipeline Provenance Graph

**Phase:** Phase 2 Step 4.5 (Post-Governance KG)  
**Status:** DESIGN  
**Owner:** Data Governance + PIT Safety  
**T0:** 2026-05-19  
**Expected Completion:** 2026-05-28  

---

## 1. Purpose

Provide deterministic, queryable lineage from raw data source → feature engineering → scoring module → snapshot artifact → validation artifact.

**Goals:**
1. Answer "Why did this company's score change?" with data provenance
2. Detect stale, missing, backfilled, or quarantined features
3. Verify PIT safety: which snapshot used which code commit and raw data as-of-date
4. Understand impact radius: "If SEC cache breaks, which outputs are affected?"
5. Audit data freshness: which raw sources fed today's rankings.csv?

**Non-goals:**
- Not a production scoring tool
- Not a graph database or search engine
- Not LLM-extracted graph facts
- Not enforced gating (operator review required)

---

## 2. Design

### 2.1 Node Types (13 types)

| Node Type | Purpose | Examples |
|-----------|---------|----------|
| **RawSource** | External data sources | SEC Edgar cache, 13F filings, Morningstar API, ClinicalTrials, Alpaca |
| **CacheFile** | Cached intermediate files | `/data/sec_cache/*.json`, `/data/13f_pit_index/*.jsonl`, `/data/morningstar/*.json` |
| **SnapshotArtifact** | Daily snapshot output | `data/snapshots/2026-05-19/`, rankings.csv, universe.json |
| **FeatureColumn** | Derived feature | `inst_delta_z`, `catalyst_score`, `clinical_score_v2_z`, `financial_score` |
| **FeatureModule** | Code that produces feature | `selector_engine.py`, `run_score_catalyst.py`, `Module 3 / clinical`, `Module 5 / financial` |
| **ScoringModule** | Code path in selector/ranker | `selector.py`, `ranker_v2_pairwise.py`, `run_true_ranker_ic.py` |
| **ModelArtifact** | Trained model or weights | `ranker_v2_model.json`, `coinvest_context_frame.json` |
| **CodeCommit** | Git commit hash + message | `8bee00e4`, `fb129fcf`, `3185d752` |
| **ValidationReport** | QA/audit artifact | `artifacts/audit/*_2026-05-19.md`, `drift_report_*.json` |
| **QuarantineMarker** | Feature/data flagged as invalid | `composite_score (INVALID)`, `inst_delta_z (DISTORTION)`, `13f_quarantine_active` |
| **BackfillRecord** | Historical data reconstructed | Spec 102 backfill, historical Morningstar, re-run catalyst event detector |
| **GateLock** | Governance gate | `13f_clearance`, `phase_2_step_3_complete`, `h20d_decision` |
| **ConfigVersion** | Schema or parameter version | `ranker_v2_model.v1`, `selector_schema.v2`, `institutional_summary.v1` |

### 2.2 Edge Types (8 types)

| Edge Type | Semantics | Examples |
|-----------|-----------|----------|
| **PRODUCES** | A→B: A generates B | RawSource → CacheFile, FeatureModule → FeatureColumn |
| **CONSUMES** | A→B: A reads/uses B | ScoringModule → FeatureColumn, SnapshotArtifact → ModelArtifact |
| **DERIVES** | A→B: B is logical derivation of A | FeatureColumn → FeatureColumn (inst_delta_z from 13f_pit_index) |
| **IMPLEMENTS** | A→B: Code version A implements feature/module B | CodeCommit → FeatureModule, CodeCommit → ScoringModule |
| **VALIDATES** | A→B: A certifies/audits B | ValidationReport → SnapshotArtifact, ValidationReport → FeatureColumn |
| **QUARANTINES** | A→B: A marks B as invalid/stale | QuarantineMarker → FeatureColumn, GateLock → FeatureModule |
| **BACKFILLS** | A→B: A reconstructs historical B | BackfillRecord → CacheFile, BackfillRecord → FeatureColumn |
| **GATED_BY** | A→B: A cannot proceed until B | SnapshotArtifact → GateLock, FeatureModule → GateLock |

### 2.3 Schema

```json
{
  "schema": "pipeline_provenance_graph.v1",
  "as_of_date": "2026-05-19",
  "generated_at": "2026-05-19T14:30:00Z",
  "nodes": [
    {
      "id": "source_sec_edgar",
      "type": "RawSource",
      "title": "SEC Edgar 8-K/6-K filings",
      "status": "active",
      "freshness_days": 1,
      "last_ingest": "2026-05-19T12:00:00Z",
      "related_artifacts": ["artifacts/audit/sec_8k_coverage_*.md"]
    },
    {
      "id": "cache_13f_pit_index",
      "type": "CacheFile",
      "title": "13F PIT-corrected index (daily)",
      "path": "data/13f_pit_index/",
      "status": "active",
      "last_updated": "2026-05-19T00:00:00Z",
      "freshness_days": 0,
      "as_of_date": "2026-05-19",
      "quarantine_marker": "13f_quarantine_active"
    },
    {
      "id": "feature_inst_delta_z",
      "type": "FeatureColumn",
      "title": "Institutional investor delta (z-scored)",
      "status": "active",
      "evidence_status": "forward_shadow_only",
      "last_backfill": "2026-05-01",
      "distortion_marker": "post_cohort_change_2026_04_25",
      "related_quarantine": "inst_delta_distortion_active"
    },
    {
      "id": "snapshot_2026_05_19",
      "type": "SnapshotArtifact",
      "title": "Daily snapshot 2026-05-19",
      "path": "data/snapshots/2026-05-19/",
      "created_at": "2026-05-19T09:47:00Z",
      "code_commit": "b236ed19",
      "validation_status": "PASS",
      "related_artifacts": ["data/snapshots/2026-05-19/rankings.csv", "data/snapshots/2026-05-19/universe.json"]
    }
  ],
  "edges": [
    {
      "source": "source_sec_edgar",
      "target": "cache_sec_cache",
      "type": "PRODUCES",
      "as_of_date": "2026-05-19"
    },
    {
      "source": "cache_13f_pit_index",
      "target": "feature_inst_delta_z",
      "type": "PRODUCES",
      "module": "selector_engine.py:inst_delta_z_from_pit",
      "as_of_date": "2026-05-19"
    },
    {
      "source": "feature_inst_delta_z",
      "target": "snapshot_2026_05_19",
      "type": "CONSUMES",
      "as_of_date": "2026-05-19"
    },
    {
      "source": "snapshot_2026_05_19",
      "target": "gate_13f_clearance",
      "type": "GATED_BY",
      "reason": "outputs invalid until 13F quarantine clears"
    }
  ]
}
```

---

## 3. Implementation

### 3.1 Four-Layer Pipeline

**Layer 1: Extract**
- Scan `data/snapshots/*/` for snapshot metadata (commit, as-of-date)
- Scan `data/*cache*/` for cache files and timestamps
- Scan `production_data/` for model/config versions
- Scan `artifacts/audit/` for validation reports
- Parse git log for code commits and feature implementation
- Parse memory files for quarantine markers and gate status

**Layer 2: Build Graph**
- Create node for each source, cache, feature, module, snapshot
- Link PRODUCES edges (source → cache → feature)
- Link CONSUMES edges (feature → snapshot, model → scoring)
- Link IMPLEMENTS edges (commit → module)
- Link VALIDATES edges (validation report → snapshot)
- Link QUARANTINES edges (marker → feature)
- Link GATED_BY edges (snapshot → gate)

**Layer 3: Validate**
- Check for missing sources (cache file references non-existent source)
- Check for broken lineage (feature references cache that doesn't exist)
- Check for stale edges (snapshot references old code commit)
- Check for dangling nodes (features in snapshot but not in graph)
- Verify PIT safety: snapshot commit ≤ feature last-update date

**Layer 4: Emit**
- Write `artifacts/ops/pipeline_provenance/nodes.jsonl`
- Write `artifacts/ops/pipeline_provenance/edges.jsonl`
- Write `artifacts/ops/pipeline_provenance/lineage_report.md` (human-readable lineage summary)
- Write `artifacts/ops/pipeline_provenance/freshness_audit.json` (data age report)

### 3.2 Query Patterns (5 Core Queries)

#### Query 1: `lineage <feature>`
Show all sources that feed a feature.

```
$ python3 query_pipeline_provenance.py lineage inst_delta_z

inst_delta_z
  Sources:
    source_sec_edgar → cache_sec_13f_form4 → cache_13f_pit_index → feature_inst_delta_z
    Freshness: 0d (as of 2026-05-19)
    Status: ACTIVE but QUARANTINED (post_cohort_change_2026_04_25)
```

#### Query 2: `snapshot-inputs <snapshot_id>`
List all raw sources and features that fed a snapshot.

```
$ python3 query_pipeline_provenance.py snapshot-inputs 2026-05-19

Snapshot: 2026-05-19
Code commit: b236ed19 (2026-05-17)
Validation: PASS

Inputs (10 features):
  ✓ coinvest_score_z ← source_sec_edgar (0d)
  ✓ financial_score ← source_morningstar (1d)
  ✓ inst_delta_z ← source_sec_13f (0d, QUARANTINED)
  ✓ catalyst_score ← source_clinicaltrials (2d)
  ✗ clinical_score_v2_z ← source_unspecified (MISSING)
```

#### Query 3: `breakage-impact <source>`
If a source breaks, which snapshots/features are affected?

```
$ python3 query_pipeline_provenance.py breakage-impact source_sec_edgar

If source_sec_edgar breaks:

Immediate impact (0d):
  - cache_sec_13f_form4
  - feature_inst_delta_z
  - snapshot_2026_05_19 (current)

Downstream impact (3d):
  - All snapshots using inst_delta_z
  - selector ranking (inst_delta_z is 3rd selector input)
```

#### Query 4: `stale-features <snapshot_id>`
List features older than N days.

```
$ python3 query_pipeline_provenance.py stale-features 2026-05-19 --threshold-days 7

Stale sources (age > 7d):
  ⚠ source_morningstar: 8d (last ingest 2026-05-11)
  ⚠ source_clinicaltrials: 14d (last refresh 2026-05-05)

Features affected:
  - financial_score_morningstar
  - catalyst_score_clinicaltrials
```

#### Query 5: `validate-snapshot <snapshot_id>`
Check PIT safety for a snapshot.

```
$ python3 query_pipeline_provenance.py validate-snapshot 2026-05-19

Snapshot: 2026-05-19
Code commit: b236ed19 (2026-05-17)

PIT Safety Check:
  ✓ Code commit before snapshot date (2026-05-17 < 2026-05-19)
  ✓ All features updated before snapshot (inst_delta 2026-05-19, financial 2026-05-18)
  ✓ No forward-looking data in source caches
  ✗ WARNING: 13F quarantine active; outputs flagged INVALID until clearance

Status: VALID (with governance annotation)
```

---

## 4. Key Features

### 4.1 PIT Safety

- Verify snapshot date ≥ all source/feature as-of-dates
- Flag forward-looking data (source dated after snapshot)
- Track code commit at time of snapshot (reproducibility)
- Link to validation reports (who approved this snapshot?)

### 4.2 Stale Data Detection

Track `last_updated` and `freshness_days` for each node.

Rules:
- RawSource stale if `freshness_days > expected_cadence` (e.g., SEC should update daily, but Morningstar only weekly)
- CacheFile stale if `freshness_days > 3`
- FeatureColumn stale if underlying source is stale
- SnapshotArtifact flagged if any feature input is stale

### 4.3 Quarantine Integration

Link QuarantineMarker nodes to affected features/snapshots:
- `13f_quarantine_active` → marks all 13F-derived features as GATED_BY 13F clearance
- `inst_delta_distortion_active` → marks inst_delta_z as forward_shadow_only
- `composite_score_INVALID` → marks IC evidence chains as broken

### 4.4 Backfill Tracking

BACKFILLS edges show which data was reconstructed:
- Spec 102 (historical backfill): 19 snapshots 2026-04-20 through 2026-05-13
- Marks which features were back-filled vs forward-only

---

## 5. Test Plan

### 5.1 Unit Tests

**Test Class: NodeCreation**
- Verify all 13 node types can be created
- Verify required fields (id, type, status)
- Verify optional metadata (freshness_days, quarantine_marker)

**Test Class: EdgeCreation**
- Verify all 8 edge types can be created
- Verify source/target nodes exist
- Verify edge metadata (as_of_date, module, reason)

**Test Class: LineageChains**
- Build a complete chain: RawSource → Cache → Feature → Snapshot
- Verify lineage can be traced end-to-end
- Verify cycle detection (no circular dependencies)

**Test Class: PITSafety**
- Snapshot date ≥ all input as-of-dates
- Code commit date < snapshot date
- No forward-looking data

**Test Class: QueryPatterns**
- `lineage inst_delta_z` returns correct source chain
- `snapshot-inputs 2026-05-19` lists all 10+ features
- `breakage-impact source_sec_edgar` identifies 3+ downstream features
- `stale-features 2026-05-19` correctly detects age > threshold
- `validate-snapshot 2026-05-19` flags PIT violations and quarantine gates

**Test Class: QuarantineIntegration**
- QuarantineMarker nodes correctly mark features as INVALID
- GATED_BY edges prevent snapshot inference until gate clears
- 13F quarantine marker affects all 13F-derived features

### 5.2 Integration Test

Build a real graph from current state:
- Extract: ~50 nodes (10 sources, 10 caches, 15 features, 5 modules, 3 snapshots, 7+ gate/quarantine markers)
- Edges: ~80 (PRODUCES, CONSUMES, DERIVES, IMPLEMENTS, VALIDATES, QUARANTINES, GATED_BY)
- Query each pattern and verify outputs match expected lineage

### 5.3 Regression Test

Run quarterly to verify:
- No broken source references
- All active features have producing modules
- All snapshots link to code commits
- All validation reports reference existing snapshots

---

## 6. Implementation Notes

### 6.1 Tools

**`build_pipeline_provenance.py`** (500 lines)
- Layer 1-4 extraction, normalization, validation, output
- Read from: `data/snapshots/`, `production_data/`, `artifacts/audit/`, git log, memory files
- Write to: `artifacts/ops/pipeline_provenance/{nodes,edges}.jsonl`

**`query_pipeline_provenance.py`** (400 lines)
- Implement 5 query patterns
- CLI with BFS traversal for lineage and impact analysis
- Output as text, JSON, or markdown

### 6.2 Scope Boundaries

**In scope:**
- Data lineage (source → snapshot)
- Feature engineering lineage
- Code commit tracking
- Quarantine/gate status
- PIT safety verification
- Stale data detection

**Out of scope:**
- Real-time stream lineage
- Graph database (use JSONL only)
- LLM reasoning over graph
- Scoring logic inference
- Automatic remediation

### 6.3 Integration Points

- Feed data to `governance-spec-enforcement` skill (lineage for QA)
- Link from `build_hermes_knowledge_layer.py` (artifact dependencies)
- Reference in `phase-2-step-4-readiness` (validation infrastructure)
- Export PIT safety report to governance memo

---

## 7. Success Criteria

### 7.1 Phase A (Build)

- [ ] Schema locked (13 node types, 8 edge types)
- [ ] Build tool complete and tested (unit + integration)
- [ ] Query tool complete (5 patterns)
- [ ] 50+ nodes, 80+ edges extracted from real state
- [ ] All 5 queries return correct results
- [ ] Tests: 30+ unit tests, 100% pass

### 7.2 Phase B (Validation)

- [ ] PIT safety check accurate (no false positives on real snapshots)
- [ ] Stale data detection correct (compares to expected cadence)
- [ ] Quarantine marker integration working (13F gate blocks snapshots)
- [ ] Breakage impact analysis correct (source loss → feature → snapshot)
- [ ] Operator can answer "why did score change?" with lineage output

### 7.3 Phase C (Operationalization)

- [ ] Cron job: daily 08:00 ET (post-morning snapshot)
- [ ] Artifacts published to `artifacts/ops/pipeline_provenance/`
- [ ] Lineage report added to governance memo
- [ ] Linked from Hermes skills for operator query
- [ ] Integration test passes monthly (no broken lineage)

---

## 8. Timeline

| Phase | Week | Deliverable |
|-------|------|-------------|
| A1 | 2026-05-19 to 2026-05-21 | Schema locked, build tool 80% |
| A2 | 2026-05-22 to 2026-05-24 | Build tool 100%, query tool 80% |
| B1 | 2026-05-25 to 2026-05-28 | All tests pass, real graph verified |
| B2 | 2026-05-29 to 2026-06-02 | PIT safety validation, operator walkthrough |
| C1 | 2026-06-03+ | Cron deployment, integration with governance skills |

---

## 9. Relationship to Spec 089

| Aspect | Spec 089 (Governance KG) | Spec 110 (Pipeline Provenance) |
|--------|---|---|
| **Query Focus** | Blockers, contradictions, spec status | Lineage, freshness, impact radius |
| **Node Types** | 12 (specs, policies, gates, artifacts) | 13 (sources, caches, features, modules) |
| **Edge Types** | 15 (IMPLEMENTS, BLOCKS, DEPENDS_ON, etc.) | 8 (PRODUCES, CONSUMES, DERIVES, etc.) |
| **Primary User** | Governance operator reviewing architecture | Debug operator answering "why?" questions |
| **Enforcement** | Read-only; gates check via memory | Read-only; gates annotate snapshots |
| **Freshness** | Updates on spec/policy change | Updates daily post-snapshot |
| **Scope** | What's blocked and why | What fed this output and is it fresh |

---

## 10. Example Lineage Report

```
# Pipeline Provenance Report — 2026-05-19

## Data Sources (Age Summary)

| Source | Type | Last Update | Freshness | Status |
|--------|------|---|---|---|
| SEC Edgar 8-K/6-K | External | 2026-05-19 12:00 ET | 0d | ✓ Active |
| SEC EDGAR 13F | External | 2026-05-19 00:00 ET | 0d | ⚠ QUARANTINED |
| Morningstar API | External | 2026-05-18 18:00 ET | 1d | ✓ Active |
| ClinicalTrials.gov | External | 2026-05-05 14:22 ET | 14d | ⚠ Stale |
| Alpaca Markets | External | 2026-05-19 16:00 ET | 0d | ✓ Active |

## Feature Lineage (Top 10)

```
inst_delta_z
  ← source_sec_edgar (13F filings)
  ← cache_13f_pit_index (as_of_date: 2026-05-19)
  ← selector_engine.py:inst_delta_z_from_pit (commit: b236ed19)
  → snapshot_2026_05_19 (selector input #3)
  ⚠ Status: QUARANTINED (post_cohort_change_2026_04_25; forward_shadow_only)
  ⚠ Gate: 13f_quarantine_active (snapshot invalid until clearance)

financial_score
  ← source_morningstar (ESG/financial metrics)
  ← cache_morningstar_esg (as_of_date: 2026-05-18)
  ← run_score_financial.py (Module 5, commit: b236ed19)
  → snapshot_2026_05_19 (ranker input)
  ✓ Status: ACTIVE, 1d fresh
  ✓ Gate: None (no quarantine)

catalyst_score
  ← source_clinicaltrials (trial data)
  ← cache_trials_phase_3 (as_of_date: 2026-05-05)
  ← run_score_catalyst.py (Module 3, commit: b236ed19)
  → snapshot_2026_05_19 (ranker input)
  ⚠ Status: ACTIVE, 14d STALE
  ⚠ Action: Re-ingest ClinicalTrials source
```

## PIT Safety Audit

✓ Snapshot date: 2026-05-19
✓ Code commit: b236ed19 (2026-05-17)
✓ All feature as-of-dates ≤ snapshot date
✓ No forward-looking data detected
⚠ Quarantine gate active; outputs flagged INVALID for ranking decisions

Verdict: **VALID FOR ANALYSIS, INVALID FOR PRODUCTION** (pending 13F clearance)
```

---

## 11. Blocking Dependencies

- **13F Quarantine Clearance** — inst_delta_z remains gated until 2026-05-26
- **Spec 089 KG** — uses node/edge schema patterns from governance KG
- **Phase 2 Step 3** — requires stable snapshot infrastructure

**Not blocking:** Can run in parallel with 13F refresh (Phase 2 Step 4).

---

## 12. Rollback

If pipeline provenance reveals systemic lineage failures:
1. Disable daily cron (preserve last good state)
2. Fix source extraction in build tool
3. Re-run on last N snapshots
4. Publish updated artifacts
5. Operator review before re-enabling automation

No production scoring changes; KG is read-only.

