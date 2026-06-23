# PIT Gap Forward Return Method Decision — 2026-06-22

**Status:** OPERATOR_DECISION_RECORDED — 2026-06-22  
**Governance:** Production model freeze ACTIVE; implementation requires separate explicit authorization  
**Context:** Follows accepted feasibility memo (PR #381) and archive-ceiling finding (PR #383, merged)

---

## Operator Decision (2026-06-22)

**1. Method A — PRIMARY**
Same-archive basis. Use for 5d and 20d where coverage exists. No 60d conclusion from Method A.
Lowest adjustment-basis risk. Accepted as the official primary methodology.

**2. Method B — SENSITIVITY ONLY**
Single May 7 archive basis. Authorized only to estimate 60d coverage and compare directional
consistency with Method A. All Method B outputs must be labeled `SENSITIVITY_ONLY_NOT_PRIMARY_EVIDENCE`.
Must not override Method A results.

**3. Method C — DEFERRED**
No external provider fetch. No IEX, Tiingo, Alpaca, or yfinance calls.

**4. PR #382 — REMAINS QUARANTINED**
Do not reuse its code directly. Do not merge. Do not treat its outputs as accepted evidence.
May be consulted as a cautionary reference only.

**5. Implementation constraints**
- Fresh script; do not copy from PR #382.
- Separate new branch, explicitly authorized before writing.
- Research-only: `scripts/research/` only.
- No production files: no ranker, selector, sizing, final_score, gates, snapshots, portfolio.
- Output panels remain quarantined/research-only until separately reviewed.
- No model decision or freeze-lift conclusion from output alone.

**Next step:** Write `PIT_GAP_FORWARD_RETURN_ASSEMBLY_SPEC_2026_06_22.md` — a short spec
defining exact inputs, outputs, validation checks, and acceptance thresholds for Method A
primary + Method B sensitivity. No implementation code until spec is approved.

---

---

## Background

The accepted feasibility memo (PR #381) established that the gap period
(2026-01-16 to 2026-05-07, 88 canonical snapshots) has 99.5% **anchor price** coverage
from `data/pit_archives/`. The merged archive-ceiling note (PR #383) established that
**forward-return horizon coverage** is separately constrained by each archive's `last_date`.

Key constraint (from PR #383): early archives were mass-rebuilt on 2026-04-10 and have
`last_date = 2026-04-10`. April/May archives were built daily and have `last_date` equal to
their creation date. This limits forward-return availability under the same-archive method.

Three methods are available to compute gap-period forward returns. This memo compares them
so the operator can choose one and explicitly authorize the next implementation step.

---

## Method A: Same-Archive Basis (Conservative)

### Description

For each gap snapshot date, use that date's own pit_archive (or nearest prior archive for
the 3 missing dates) as the source for BOTH the anchor price AND all forward prices. This
guarantees split-adjustment consistency within each return computation.

### Coverage (from PR #383 archive-ceiling finding — unreviewed, not accepted evidence)

| Horizon | Snapshots with any coverage | Date range |
|---------|----------------------------|------------|
| 5d | ~57 / 88 | 2026-01-16 to ~2026-03-31 |
| 20d | ~41 / 88 | 2026-01-16 to ~2026-03-12 |
| 60d | ~0 / 88 | — (archive ceiling at April 10; 60d from Jan 16 ≈ April 9-14, borderline) |

*Note: These coverage numbers come from the quarantined PR #382 run. They are indicative
only and must be independently confirmed once a method is approved.*

### Adjustment-basis risk

Lowest. Both anchor and forward prices are from the same file, same download pass,
same split-adjustment state. Returns cancel any split factor applied between anchor
and forward dates.

### Key limitation

No 60d coverage. Partial 5d/20d. April/May gap snapshots yield no returns at all
under this method, because their archives only contain prices through their creation date.

### Authorization required

Implement `assemble_gap_forward_returns.py` (currently quarantined in PR #382).
Commit to `scripts/research/`. Run manually. Review output before accepting.

---

## Method B: Single Latest Archive Basis

### Description

Use the single most recent pit_archive (`data/pit_archives/2026-05-07/`,
`last_date = 2026-05-07`) for ALL gap snapshot return computations — both anchor and
forward prices. Every return is computed from one consistent price series.

### Coverage (projected, not yet verified)

| Horizon | Snapshots with projected coverage | Date range |
|---------|----------------------------------|------------|
| 5d | ~82 / 88 | 2026-01-16 to ~2026-04-30 |
| 20d | ~57 / 88 | 2026-01-16 to ~2026-04-09 |
| 60d | ~35-40 / 88 | 2026-01-16 to ~2026-02-28 |

*Note: Projections based on May 7 archive having `last_date = 2026-05-07`.
Must be confirmed against actual archive contents before use.*

### Adjustment-basis justification

Split-adjustment factors cancel in return calculations when both anchor and forward
prices use the same adjustment convention. Using a single archive for all computations
ensures consistency — arguably more consistent than Method A (which uses different
archives for different snapshots, each with its own retroactive rebuild date).

The key risk: if a stock was delisted between Jan 16 and May 7, the May 7 archive
may have no price rows for that stock on the anchor date, or may have distorted prices
near the delisting. Known case: ATXS (last trading date 2026-01-23) — would still
need post-acquisition exclusion.

### Key advantage over Method A

~35-40 snapshots of 60d coverage becomes available. This is meaningful: 35 snapshots
of daily 60d returns would roughly correspond to the Jan–Feb portion of the gap period,
filling in the most data-sparse part of the daily backtest panel.

### Adjustment-basis risk

Low-to-moderate. The May 7 archive contains prices that may reflect splits applied
between the gap period and May 7. These splits cancel in return calculations
(adjusted_fwd/adjusted_anchor = true_return) but any corporate actions other than
clean splits (tender offers, spin-offs) could produce distortions. ATXS is handled
by exclusion; other corporate actions in Jan-May 2026 must be checked.

### Authorization required

Explicit operator approval of Method B. Then implement `--use-latest-archive` flag
in a new script (NOT the quarantined PR #382 code). Commit to `scripts/research/`.
Dry-run first. Review output before accepting.

---

## Method C: External Historical Price Provider

### Description

Fetch gap-period historical prices from an external provider (IEX Cloud, Tiingo, Alpaca,
or Polygon) for all gap snapshot dates and required forward horizons. This bypasses the
pit_archives entirely for forward price computation, though pit_archives can still serve
as anchor price source.

### Coverage (projected)

| Horizon | Snapshots with projected coverage |
|---------|----------------------------------|
| 5d | 88 / 88 |
| 20d | 88 / 88 |
| 60d | 88 / 88 (all gap snapshots; forward dates are now in the past) |

### Key advantage

Potentially complete 60d coverage for all 88 gap snapshots. Provider data also serves
as an independent price source that can be used to cross-validate the pit_archive prices.

### Key risks and operational burden

1. **Provider selection and contract:** Must pick a provider with documented data sourcing,
   split-adjustment methodology, and historical data availability. `iex_cloud_price_download.py`
   exists in `scripts/` (noted in feasibility memo §5). Starter tier ~$9/mo.

2. **Adjustment-basis consistency:** Provider prices may use different split-adjustment
   conventions than the pit_archives. This introduces a potential basis difference between
   anchor prices (from pit_archives) and forward prices (from provider), or requires using
   provider prices for both anchor and forward.

3. **Look-ahead for splits:** Providers backfill split adjustments retroactively, which is
   standard practice for backtesting. This is no worse than the pit_archive retroactive rebuild.

4. **Data integrity gates:** Would require the same validation suite as Method A/B
   (continuity check, coverage completeness, ATXS exclusion, era reconciliation).

5. **Operational cost:** New dependency, new API key, new data pipeline path.
   Adds complexity vs. using existing cached sources.

### Authorization required

Operator approval of Method C, provider selection, and cost acceptance.
Then implement a fetch script. Validate against pit_archive prices on overlapping dates.
Store fetched data in a new directory (e.g., `data/external_prices/`) with its own
manifest and governance label (`RESEARCH_ONLY`).

---

## Recommendation

**Method B** is the recommended path if 60d coverage is a priority.

Rationale:
- The May 7 archive already exists and is fully cached — no external dependency, no cost.
- Single-archive consistency is arguably better than per-snapshot archive switching (Method A).
- ~35-40 snapshots of 60d returns covers the Jan–Feb gap period, which is the most
  data-sparse portion of the daily backtest panel.
- Split-adjustment risk is low and well-understood for this period (Jan–May 2026 is
  recent; major corporate actions are known: ATXS handled, others checkable).

**Method A** is appropriate if the operator prefers minimal risk and does not need 60d gap
coverage (e.g., if the forward monitor accumulation since May 8 provides sufficient IC data).

**Method C** adds operational burden not justified by the marginal coverage gain over Method B,
given that pit_archives already cover the gap period with known-consistent prices.

---

## Decision Gate

Operator must select one method and explicitly state:

1. Which method is authorized (A, B, or C).
2. Whether PR #382 quarantined code may be used as a starting point (Option B from the
   prior governance decision), or whether a fresh script must be written from scratch.
3. For Method B or C: whether to implement the `--use-latest-archive` flag in a new script,
   or whether to design a separate standalone script.

**No executable code until this decision is made and recorded.**

---

**Prepared:** 2026-06-22 (design memo — markdown only)  
**Next action:** Operator selects method → explicit authorization → fresh implementation branch
