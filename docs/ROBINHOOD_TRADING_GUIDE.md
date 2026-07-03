# Robinhood Trading System — Review-First, Approval-Required

Two-stage execution system for biotech screener top-K portfolio construction.

## MCP Connection & Live Position Fetch

**The Robinhood MCP is a hosted remote server, not a local install.** Do NOT run
`pip install robinhood-mcp` / `mcp-server-robinhood` — those are unofficial third-party
packages that take your username/password. The official server is authorized via OAuth.

### Connect (one-time, desktop only)

Claude Code (Hermes environment):
```bash
claude mcp add robinhood-trading --transport http https://agent.robinhood.com/mcp/trading
# then inside Claude Code:
#   /mcp  ->  select robinhood-trading  ->  complete OAuth in a desktop browser
```

Notes:
- Auth requires a desktop browser. On a headless VM, copy the printed OAuth URL into a
  local browser, approve, and the token lands back on the box.
- Agentic Trading is gated by rollout — Robinhood emails you when you have access.
- Read access spans ALL your Robinhood accounts (IRAs, Individual, Agentic).
  Trade placement is restricted to the Agentic account (802349084) only.

### Live position fetch (read-only) — for mark-to-market / analysis

Read-only. Does NOT place, review, or modify any orders.
```
Using the robinhood-trading MCP (read tools only):
1. get_accounts  -> list all accounts
2. get_equity_positions (all accounts) -> ticker, quantity, average_buy_price, market_value
3. get_portfolio (each account) -> equity, market_value, cash, buying_power
4. get_equity_quotes (Top-30 names) -> last_trade_price, timestamp
Write pretty JSON to artifacts/trading/live_positions_YYYY-MM-DD.json with a top-level
"fetched_at" timestamp and "data_source": "robinhood-mcp-live". Commit the artifact.
```

### Verify

`get_portfolio` returns real balances (not the carry-forward baseline), and the next
daily monitoring run shows real equity instead of `robinhood_mcp_status: UNAVAILABLE`.

---

## Quick Start

### Stage 1: Generate Trade Plan (Review Only)

**Option A: Top 15 (Conservative, $5 minimum)**
```bash
python3 tools/robinhood_top30_trade_plan_v2_mcp.py \
  --top-k 15 \
  --target-gross-dollars 100 \
  --account agentic \
  --min-order-dollars 5
```

**Option B: Top 30 (Full diversification with fractional shares)**
```bash
python3 tools/robinhood_top30_trade_plan_v2_mcp.py \
  --top-k 30 \
  --target-gross-dollars 100 \
  --account agentic \
  --min-order-dollars 1
```

Output: Human-readable blotter + JSON artifact in `artifacts/trading/`

**Example output:**
```
✓ Loaded snapshot from 2026-06-10 (0 days old)
✓ Selected top 15 eligible tickers
✓ 13 positions pass guardrails

PROPOSED ORDERS
  1. BUY 0.100 COGT @ $50.00 = $5.00  (Rank 1, Tier A)
  2. BUY 0.100 DNTH @ $50.00 = $5.00  (Rank 2, Tier A)
  ... (11 more)

SUMMARY
  Orders: 13
  Gross Notional: $65.00
  Status: PENDING_HUMAN_REVIEW

✓ Blotter saved: artifacts/trading/robinhood_top15_plan_2026-06-10.json
```

### Stage 2: Review & Execute (Approval Required)

**Only after reviewing the blotter above:**

```bash
python3 tools/robinhood_execute_trades_v2_mcp.py \
  --blotter artifacts/trading/robinhood_top30_plan_2026-06-10.json \
  --account agentic
```

Output: Each order reviewed, then **prompts for approval:**

```
ORDER REVIEW
✓  1. BUY 0.100 COGT @ $50.00 = $5.00
      Review: OK | Slippage: 5 bps
✓  2. BUY 0.100 DNTH @ $50.00 = $5.00
      Review: OK | Slippage: 5 bps
... (11 more)

⚠️  EXECUTION APPROVAL REQUIRED
Review the orders above carefully.
To proceed with LIVE execution, type exactly:
  EXECUTE APPROVED ORDERS

>> [TYPE HERE]
```

**Type exactly:** `EXECUTE APPROVED ORDERS`

Orders are placed, execution logged to `artifacts/trading/execution_*.json`.

## Guardrails Enforced

### Hard Execution Limits

- **No options, margin, shorts, crypto, after-hours**
- **Tier A/B only** — Tier C positions dropped
- **Catalyst ≥8 days** — avoid imminent catalysts in new positions
- **Max 5% per name** (configurable)
- **Min order size $1** (default, supports fractional shares; configurable)
- **Fractional shares supported** (0.0001 to 6 decimal places)

### Fail-Closed Behavior

Script refuses to trade if:
- Snapshot is >7 days old
- Decision portfolio missing
- Account/holdings/cash data missing or stale

## Command-Line Reference

### Trade Plan Generation (v2 with MCP hooks)

```bash
python3 tools/robinhood_top30_trade_plan_v2_mcp.py \
  [--snapshot latest|YYYY-MM-DD]    # Default: latest snapshot
  [--top-k N]                        # Default: 30
  [--account NAME]                   # agentic|roth|traditional|default
  [--target-gross-dollars AMOUNT]    # Default: 100.0
  [--max-single-name-pct PCT]        # Default: 5.0 (per-name cap in $)
  [--min-order-dollars AMT]          # Default: 1.0 (supports fractional)
  [--fetch-real-quotes]              # Attempt real quotes via MCP (optional)
  [--dry-run]                        # Default: true (generate only)
```

**Sizing guide for fractional shares:**
- Top 15 @ $100: ~$6.67/name ($5 min) → 13-15 orders
- Top 30 @ $100: ~$3.33/name ($1 min) → 25-30 orders
- Top 30 @ $500: ~$16.67/name ($5 min) → 28-30 orders (safer margin)

### Order Execution (v2 with MCP hooks)

```bash
python3 tools/robinhood_execute_trades_v2_mcp.py \
  --blotter PATH                     # Path to trade plan JSON
  [--account NAME]                   # Default: agentic
  [--account-number NNNNNNNNNN]      # Default: 802349084 (Agentic)
  [--skip-approval]                  # Skip approval (testing only)
```

**Note:** Stage 2 calls `review_equity_order` on each trade and skips any Robinhood rejects (minimum notional, tradability, etc.).

## Workflow Examples

### Example 1: Top 15, $100, Conservative ($5 minimum)

```bash
# Stage 1: Generate
python3 tools/robinhood_top30_trade_plan_v2_mcp.py \
  --top-k 15 \
  --target-gross-dollars 100 \
  --account agentic \
  --min-order-dollars 5

# Review blotter, then

# Stage 2: Execute
python3 tools/robinhood_execute_trades_v2_mcp.py \
  --blotter artifacts/trading/robinhood_top15_plan_2026-06-10.json \
  --account agentic
```

### Example 2: Top 30, $100, Fractional Shares ($1 minimum)

```bash
# Stage 1: Generate
python3 tools/robinhood_top30_trade_plan_v2_mcp.py \
  --top-k 30 \
  --target-gross-dollars 100 \
  --account agentic \
  --min-order-dollars 1

# Review blotter, then

# Stage 2: Execute
python3 tools/robinhood_execute_trades_v2_mcp.py \
  --blotter artifacts/trading/robinhood_top30_plan_2026-06-10.json \
  --account agentic
```

### Example 3: Top 30, $500, Better Safety Margin

```bash
# Stage 1: Generate
python3 tools/robinhood_top30_trade_plan_v2_mcp.py \
  --top-k 30 \
  --target-gross-dollars 500 \
  --account agentic \
  --min-order-dollars 5

# Stage 2: Execute
python3 tools/robinhood_execute_trades_v2_mcp.py \
  --blotter artifacts/trading/robinhood_top30_plan_2026-06-10.json \
  --account agentic
```

### Example 3: Single Name, Manual Approval

```bash
python3 tools/robinhood_top30_trade_plan.py \
  --top-k 1 \
  --target-gross-dollars 50 \
  --account agentic

python3 tools/robinhood_execute_trades.py \
  --blotter artifacts/trading/robinhood_top1_plan_2026-06-10.json
```

## Audit Trail

Every execution generates two JSON artifacts:

1. **Trade Plan** (Stage 1): `artifacts/trading/robinhood_topK_plan_YYYY-MM-DD.json`
   - Snapshot date, eligible tickers, guardrails, order list
   - Metadata: account, limits, mode (DRY_RUN/EXECUTION)

2. **Execution Log** (Stage 2): `artifacts/trading/execution_YYYYMMDD_HHMMSS.json`
   - Review results per order
   - Order IDs placed
   - Timestamp, approval status

## Design Principles

### 1. Review-First

No orders are placed by default. Generation (`--dry-run`) is the default.

### 2. Approval-Required

Execution requires explicit human input: `EXECUTE APPROVED ORDERS`.

### 3. Fail-Closed

If snapshot is stale, data is missing, or guardrails fail, the script exits with an error. It does not silently degrade or place orders.

### 4. Deterministic

Same snapshot + same parameters = same order list (no randomness).

### 5. Auditable

Every decision recorded in JSON artifacts with timestamps, guardrail decisions, and review results.

## Testing (Development)

```bash
# Generate without snapshot validation (for unit testing)
python3 tools/robinhood_top30_trade_plan.py --dry-run

# Execute with skipped approval (for CI/testing)
python3 tools/robinhood_execute_trades.py \
  --blotter artifacts/trading/... \
  --skip-approval
```

## Integration: Cloud Agent

(TODO) Set up a cloud routine that:

1. Runs `robinhood_top30_trade_plan.py` daily
2. Generates blotter
3. Posts blotter to Slack or email
4. Waits for human approval message
5. Calls `robinhood_execute_trades.py` on approval
6. Logs execution results

Example Slack approval flow:

```
Bot: Review the attached blotter.
     To approve, reply with: EXECUTE APPROVED ORDERS

User: EXECUTE APPROVED ORDERS

Bot: ✓ Orders placed. Execution logged.
```
