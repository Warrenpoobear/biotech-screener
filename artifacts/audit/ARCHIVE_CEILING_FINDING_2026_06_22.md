# Archive Ceiling Finding — 2026-06-22

**Status:** DATA_INTEGRITY_FINDING — NOT ACCEPTED EVIDENCE  
**Source:** Quarantined autonomous run (PR #382 — under manual review)  
**Governance:** Production model freeze ACTIVE; no model change implied

---

## Finding: Early pit_archives Have an April 10 Forward Horizon Ceiling

The PIT feasibility memo (PR #381, accepted) assessed the gap period as 99.5% coverable
from `data/pit_archives/`. A subsequent autonomous run (PR #382, quarantined) discovered
a practical limitation: early archives contain prices only through April 10, not through
dates needed for 60-day forward returns.

**This finding is noted here as an unreviewed discovery. It is not accepted evidence
and must not be cited in model decisions or IC calculations until manually reviewed.**

---

## What Was Found (Pending Review)

### Archive Content Range

Early gap archives (2026-01-16 through approximately 2026-03-31) were retroactively
rebuilt on 2026-04-10. All share the same `price_history.csv` content (identical SHA256),
with `last_date = 2026-04-10`.

April 2026 and May 2026 archives were built forward-looking (daily), containing prices
only through their respective archive creation dates.

### Implied Forward Coverage Limits (Same-Archive Method)

| Horizon | Last snapshot with coverage | Basis |
|---------|----------------------------|-------|
| 5d | ~2026-03-31 | March 31 + 5td = April 7 < April 10 ceiling |
| 20d | ~2026-03-12 | March 12 + 20td = April 9 ≈ ceiling |
| 60d | ~0 snapshots | Jan 16 + 60td ≈ April 9-14; most tickers lack that exact date |

### Manifest SHA256 Mismatches

All early archives report SHA256 mismatches against their manifests. This is consistent
with a retroactive rebuild (manifests reflect the pre-rebuild file; current files are the
rebuild). Assessment: manifests are stale, not the price data. Requires operator confirmation.

### ATXS Handling

ATXS (acquired 2026-01-23) reportedly excluded correctly for 14 post-acquisition rows.
Not yet confirmed via manual spot-check.

---

## How This Affects the Feasibility Memo (PR #381)

The accepted feasibility memo (PR #381) found 99.5% ticker-date pair coverage.
That finding referred to **anchor price** coverage, not forward return availability.

The archive ceiling does not invalidate the feasibility memo's core verdict
(`PASS_PIT_PRICE_GAP_CLOSURE_PATH_IDENTIFIED_NO_MODEL_CHANGE`), but it adds a
constraint: the "same-archive" adjustment basis method yields:

- Partial 5d/20d coverage (first ~57/41 of 88 gap snapshots)
- No 60d coverage via this method

The feasibility memo did not assess forward horizon coverage explicitly. This finding
should be treated as a follow-on constraint requiring manual evaluation.

---

## Option A Path (Pending Operator Decision)

Using a single latest archive (2026-05-07, `last_date = 2026-05-07`) for all
computations would provide consistent split-adjustment basis and yield:

- 5d: ~82 snapshots
- 20d: ~57 snapshots  
- 60d: ~35-40 snapshots (Jan 16 – Feb 28)

This approach requires a separate explicit authorization and a new reviewed script.
Do not use PR #382's code for this — that code is under quarantine review.

---

## What Remains Authorized

- The feasibility memo (PR #381) remains accepted evidence of the gap closure path.
- This document records the archive ceiling as an **unreviewed finding** that updates
  the practical scope of that path.
- No assembly code, no panel outputs, and no IC calculations are authorized from
  PR #382 until that PR passes manual review under Option B.

---

**Prepared:** 2026-06-22 (operator-directed markdown-only extraction)  
**Governance:** Operator must decide between Option A (strict containment) and
Option B (line-by-line PR #382 review) before any code or data from the quarantined
run is used.
