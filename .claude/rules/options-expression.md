---
paths:
  - event_ev/**
  - tools/biotech_hedge_report.py
  - tools/build_trade_plan.py
---

# Options Expression Layer & Long-Call Contracts

## Expression Layer (Spec 062)
- **Status**: Shadow-only, merged to main. Zero alpha impact.
- **Module**: `event_ev/expression_layer.py` — classification -> mapping -> gates -> sizing
- **Attribution**: `event_ev/expression_attribution.py` — JSONL logging, CRT resolution, kill switches
- **Wiring**: `run_screen.py` emits `expression_overlay_summary.json` + `expression_recommendations.json`
- **Tests**: 123 (83 expression + 40 attribution)
- **Policy**: overlay-only. Does NOT enter selector/ranker/construction. Expression layer must NEVER be imported by `selector_engine.py`, `ranker_engine.py`, or `decision_engine.py`.
- **Review horizon**: 30 days from first emission. No threshold tuning before then.

## Long-Call Contract Recommendations (Post-Screen)

For every name passing the long-call filter, recommend the best executable contract.

### Step 1 — Expiry Selection
- Choose first liquid expiry AFTER catalyst date with 14-35 calendar days cushion
- If catalyst_days 21-45: allow tighter 7-21 day post-event cushion
- Avoid expiries before the catalyst. Avoid very long expiries unless nearer are illiquid.
- Prefer standard monthly expiries over odd weeklies when liquidity is similar.

### Step 2 — Strike Selection
- Target call delta 0.30-0.50
- Higher conviction: 0.40-0.50 delta. Lower conviction / high IV: 0.30-0.40 delta.
- Avoid ultra-OTM lottery strikes. Avoid deep ITM unless spread/liquidity clearly superior.

### Step 3 — Liquidity Filter
Reject if: open_interest too low, volume too low, bid/ask spread too wide, pricing stale.
If exact spread fields unavailable, use best proxies and state the limitation.

### Step 4 — Entry Economics
Compute/estimate: mid premium, breakeven move, event-date implied move, crush-adjusted move, delta, DTE.
Prefer contracts where: directional thesis confirmed by RR/skew, implied move not extreme, room to profit after IV compression, premium reasonable relative to conviction.

### Step 5 — Rank & Output
Primary contract by: (1) expiry covering catalyst, (2) strongest liquidity, (3) delta in band, (4) best breakeven, (5) cleaner spread.
Backup: one strike lower/higher with similar expiry, or next best expiry.

```
ticker:
  catalyst: <event_type> in <N> days
  thesis: <1-2 lines>
  primary_contract:
    expiry / DTE / strike / delta / premium_or_mid / OI / volume / spread / breakeven_move_pct / why
  backup_contract: <same fields>
  no_trade_reason: <if applicable>
```

### Constraints
- If chain data unavailable, look for nearest chain artifact in repo for that date
- State missing field limitations explicitly
- Do NOT change DEM scoring/ranking logic — post-screen recommendation only
