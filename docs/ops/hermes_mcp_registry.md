# Hermes MCP Registry

Authoritative list of registered MCP servers and their tool permissions.  
**Policy:** `docs/ops/hermes_mcp_security_policy.md`  
**Last updated:** 2026-07-02

---

## Registry

### `robinhood-trading`

| Field | Value |
|---|---|
| Admitted | 2026-06-10 |
| Approver | Operator (agentic account 802349084 validation) |
| Transport | Hosted remote MCP (HTTP) — `https://agent.robinhood.com/mcp/trading` |
| Connection | OAuth (desktop browser). NOT a local install — do not `pip install`/run a server. |
| Minimum tier | 0 (read), 3 (write-trade) |
| Autonomous use | Read tools only |

**Read scope (per Robinhood Agentic Trading docs, verified 2026-07-02):**
Read access spans **ALL** Robinhood accounts on the login — Traditional IRA (4010),
Roth IRA (0727), Individual (1219), and the Agentic account (802349084) — including
account numbers, all positions/balances, and all transactions/order history.
**Trade placement is restricted to the Agentic account (802349084) only.**
(Prior registry text implying read access was scoped to 802349084 was inaccurate.)

**Allowed tools (autonomous):**

| Tool | Category | Description |
|---|---|---|
| `get_portfolio` | read-portfolio | Account balances and buying power |
| `get_equity_positions` | read-portfolio | Current holdings |
| `get_equity_orders` | read-portfolio | Order history |
| `get_equity_quotes` | read-only-market | Real-time quotes |
| `get_equity_fundamentals` | read-only-market | Company fundamentals |
| `get_equity_historicals` | read-only-market | OHLCV history |
| `get_equity_tradability` | read-only-market | Tradability status |
| `get_accounts` | read-portfolio | Account metadata |
| `get_watchlists` | read-portfolio | Watchlist contents |
| `search` | read-only-market | Symbol search |

**Human-in-loop required (interactive sessions only):**

| Tool | Category | Notes |
|---|---|---|
| `place_equity_order` | write-trade | Requires operator to be present; no cron use; Agentic account only |
| `review_equity_order` | write-trade | Confirmation step before execution |
| `cancel_equity_order` | write-trade | Emergency cancellation; operator-initiated |
| `place_option_order` | write-trade | Options; operator-only |
| `cancel_option_order` | write-trade | Options; operator-only |

**Not used / deferred:**

| Tool | Reason |
|---|---|
| `create_watchlist`, `update_watchlist` | Not required by current workflow |
| `create_scan`, `run_scan` | Not in use |
| `get_realized_pnl` | Available when needed |

**Connection runbook:** see `docs/ROBINHOOD_TRADING_GUIDE.md` → "MCP Connection & Live Position Fetch".

---

### `codegraph`

| Field | Value |
|---|---|
| Admitted | 2026-06-17 |
| Approver | Operator (Phase 7B integration) |
| Minimum tier | 0 (read-only) |
| Autonomous use | Allowed |

**All tools read-only:**

| Tool | Category |
|---|---|
| `codegraph_context` | read-only-market (code intelligence) |
| `codegraph_search` | read-only-market |
| `codegraph_explore` | read-only-market |
| `codegraph_trace` | read-only-market |
| `codegraph_callers` | read-only-market |
| `codegraph_callees` | read-only-market |
| `codegraph_impact` | read-only-market |
| `codegraph_node` | read-only-market |
| `codegraph_files` | read-only-market |
| `codegraph_status` | read-only-market |

---

### `claude_ai_Microsoft_365`

| Field | Value |
|---|---|
| Admitted | 2026-06-26 |
| Approver | Operator |
| Minimum tier | 0 (read), restricted (email) |
| Autonomous use | Read only; email via Town bridge pattern only |

**Tools:**

| Tool | Category | Autonomous use |
|---|---|---|
| `outlook_email_search` | external-search | Allowed |
| `outlook_calendar_search` | external-search | Allowed |
| `chat_message_search` | external-search | Allowed |
| `sharepoint_search` | external-search | Allowed |
| `sharepoint_folder_search` | external-search | Allowed |
| `read_resource` | external-search | Allowed |
| `get_me` | read-portfolio | Allowed |
| `find_meeting_availability` | external-search | Allowed |
| `outlook_find_available_time` | external-search | Allowed |

**Note:** This server does not expose email send — that is handled by
`common/operator_delivery.py` + `common/alert_email.py` via the Town bridge.

---

## Admission procedure

To add a new MCP server:

1. List all tools it exposes with category assignments
2. Determine minimum tier for each tool
3. Identify which tools require human-in-loop
4. Get operator approval
5. Add entry to this registry with admission date
6. Update `docs/ops/hermes_mcp_security_policy.md` Active Servers summary

---

## Decommission procedure

To remove an MCP server:

1. Mark entry as `RETIRED` with decommission date and reason
2. Confirm no active agents call any of its tools
3. Remove from Hermes/Cursor config
4. Retain registry entry for audit trail (do not delete)
