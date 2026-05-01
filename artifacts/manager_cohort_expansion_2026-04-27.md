# Manager-cohort expansion artifact analysis

Generated: 2026-05-01
Saturday rebuild: 2026-04-25 (38 → 42 managers, manual snapshot)
Monday organic:   2026-04-27 (first cron-built snapshot post-expansion)

## Quarantine state (from cohort_state.json)

**Saturday 2026-04-25:** quarantined — institutional_cohort_changed=True
  - new managers: ['Fairmount Funds Management', 'Vestal Point Capital', 'Kynam Capital Management', 'Soleus Capital Management']
  - coinvest_score_z_valid: True
  - inst_delta_z_valid: **False**
  - rank_delta_valid: **False**
  - valid_from_snapshot: 2026-04-27

**Monday 2026-04-27:** clean — no cohort_state.json, deltas vs Saturday are decision-grade.

## sec_13f_cache gate (Monday)
  status: **PASS**
  detail: coverage=100.0%, (42/42 managers)

## Top-30 cohort changes

Saturday top-30 entrants (4): ['ABVX', 'BCAX', 'MIRM', 'NBIX']
  - persisted into Monday top-30: ['MIRM']
  - reverted out of Monday top-30: ['ABVX', 'BCAX', 'NBIX']

Monday top-30 vs Saturday top-30:
  - new on Monday (not in Saturday top-30): ['DYN', 'INSM', 'KYMR']
  - dropped on Monday (was in Saturday top-30): ['ABVX', 'BCAX', 'NBIX']
  - cohort overlap: 27/30

## inst_delta_z collapse on the 5 phantom-delta names

Saturday rebuild artificially inflated inst_delta_z because new managers' holdings looked like 'new institutional buys'. Monday's run uses Saturday as prior, so the artifact should collapse toward zero.

| Ticker | Sat inst_delta_z | Mon inst_delta_z | Δ (collapse if negative) |
|---|---|---|---|
| ELVN | -0.039 | -0.039 | +0.000 |
| GERN | +0.257 | +0.257 | +0.000 |
| NRIX | +0.553 | +0.553 | +0.000 |
| TYRA | +0.553 | +0.553 | +0.000 |
| COGT | +4.404 | +4.404 | +0.000 |

## Coinvest_score_z impact for the 4 cohort entrants

| Ticker | Sat coinvest_z | Mon coinvest_z | Δ |
|---|---|---|---|
| ABVX | +0.705 | +0.688 | -0.017 |
| BCAX | +0.483 | +0.475 | -0.008 |
| MIRM | +0.534 | +0.522 | -0.012 |
| NBIX | +0.652 | +0.639 | -0.013 |

## Verdict

⚠ **Cohort-expansion artifact dominated.** 3/4 Saturday entrants reverted on Monday — the rank movement was largely phantom inst_delta, not persistent signal.
