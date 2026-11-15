# Spec 118 — Generalized Alpha Model for Public Equity Screening

**Status:** RESEARCH / DESIGN  
**Created:** 2026-11-15  
**Governance Tier:** 4 (Research Judgment)  
**Companion Specs:** Spec 116 (EES biotech overlay), Spec 117 (EES broad equity methodology)  
**Scope:** Methodology only — no implementation, no production change

---

## 1. Purpose

This document specifies a multi-factor alpha model for ranking public equities by
expected excess return. It is designed to operate upstream of the EES gate system
(Spec 117): the alpha model ranks; the EES gates clean the ranked output.

The model draws on empirical lessons from the DEM biotech screener, where a single
institutional signal (coinvest_score_z) outperformed every multi-factor composite tested.
The core lesson: **fewer orthogonal factors with clear economic mechanisms outperform
larger factor sets optimized in-sample.** This model applies that principle to broad
equity.

---

## 2. Design Philosophy

### 2.1 Mechanism-First Factor Selection

Every factor in this model must answer: **what economic mechanism generates the return,
and why does the market fail to price it fully?**

Factors without a clear mechanism are excluded, regardless of in-sample performance.
History is full of signals that backtest well and fail forward — the only defense is
requiring a causal story that explains *why* the edge persists.

The three permissible mechanism classes:

| Mechanism | Why It Persists | Examples |
|---|---|---|
| **Information asymmetry** | Some participants have earlier or better information; the signal decays as information diffuses | Institutional flows, insider activity, channel checks |
| **Behavioral mispricing** | Cognitive biases cause systematic over/underreaction that sophisticated capital cannot fully arbitrage | Momentum (underreaction), earnings anchoring, loss aversion |
| **Structural friction** | Market structure creates predictable supply/demand imbalances that persist for non-informational reasons | Index rebalancing flows, tax-loss selling, forced liquidation |

A factor must map to exactly one mechanism class. Factors that require multiple
overlapping stories ("it's partly momentum, partly value, partly quality") are over-
fitted descriptions of noise.

### 2.2 Independence by Design

The DEM experience: coinvest_score_z (institutional momentum) and financial_score
(valuation) were the two selector inputs. They were independently validated. When
financial_score collapsed under PIT-safe data (IC went from -0.067 to -0.001), the
selector survived because coinvest stood on its own.

**Rule:** Each factor must pass a standalone Checklist v2 battery before entering the
composite. A factor that only works "in combination" is not a factor — it is a parameter
that overfits to the other factors' residual.

**Correlation budget:** No two factors in the composite may have a cross-sectional
correlation above 0.35. If two candidate factors correlate above this threshold, keep the
one with stronger standalone evidence and drop the other. Do not residualize or
orthogonalize — that creates fragile synthetic signals that are hard to monitor and hard
to kill when they fail.

### 2.3 Ordinal Construction

The DEM pairwise ranker has ECE = 0.129 (poor calibration). The Sharpe comparison
between rank-weighted and equal-weighted portfolios was -0.09pp (t = -0.95) — rank
weighting destroyed value.

**Rule:** The alpha model produces an ordinal ranking only. It does not produce confidence
scores, position sizes, or portfolio weights. Construction is equal-weight across the top
N names. This is not a limitation — it is a design choice validated by the DEM evidence.

The composite score is used exclusively for sorting. Its magnitude has no economic
meaning. Only the rank matters.

---

## 3. Factor Specifications

The model uses five factors from three mechanism classes. Each factor is scored cross-
sectionally and normalized to a common scale before compositing.

### 3.1 Factor 1: Institutional Accumulation (IA)

**Mechanism class:** Information asymmetry  
**DEM analog:** coinvest_score_z (the dominant production signal)

#### 3.1.1 Thesis

Concentrated, high-conviction institutional investors — hedge funds, activist funds,
specialist sector funds — conduct deep fundamental research and accumulate positions
before the market fully prices their information. Their buying is observable via
regulatory filings with a known lag, creating a window where the information has been
acted on but not fully disseminated.

This is the generalization of DEM's coinvest signal, which tracks Fairmount Funds, Deep
Track Capital, and Logos Global Management in biotech. The broad equity version tracks a
curated roster of high-conviction managers across all sectors.

#### 3.1.2 Inputs

| Input | Source | Frequency | Lag |
|---|---|---|---|
| 13F-HR filings (institutional positions) | SEC EDGAR | Quarterly | T+45 days from quarter end |
| 13D filings (activist >5% stakes) | SEC EDGAR | Event-driven | T+10 days from crossing threshold |
| 13G filings (passive >5% stakes) | SEC EDGAR | Event-driven | T+10 days (or T+45 for year-end) |
| Form 4 (insider transactions) | SEC EDGAR | Event-driven | T+2 business days |
| Schedule 14A (proxy filings) | SEC EDGAR | Event-driven | Varies |

#### 3.1.3 Manager Registry

The signal depends on a curated registry of high-conviction managers. Not all
institutional investors are informative — index funds, quant funds with high turnover,
and diversified long-only managers carry little signal.

**Registry criteria:**

| Criterion | Threshold | Rationale |
|---|---|---|
| Concentration | Top-10 positions > 60% of 13F portfolio | Conviction, not diversification |
| Active share | > 80% vs benchmark | Not closet indexing |
| Historical hit rate | > 55% on 12-month forward return for new positions | Actually good at picking stocks |
| Filing consistency | 8+ consecutive quarterly filings | Enough history to measure |
| AUM | $500M — $15B | Large enough to be serious; small enough to be nimble |
| Style persistence | < 30% turnover per quarter | Conviction holders, not traders |

**Registry governance:**
- Initial roster: 30-50 managers meeting all criteria
- Annual re-evaluation: managers dropping below criteria are moved to `retired`
- Additions require 8 quarters of observable history meeting criteria
- The roster is a **governed input** — changes to the roster reset the out-of-sample
  clock, exactly as in DEM (see 2026-11-01 coinvest 13F integrity clock reset)

#### 3.1.4 Scoring

The score measures the breadth and recency of informed accumulation:

```
For each name:

    manager_signals = []
    for manager in registry where manager holds the name:
        signal = {
            "direction": +1 if position increased, -1 if decreased, 0 if unchanged
            "magnitude": abs(position_change_pct) relative to manager's portfolio
            "recency": filing_age_days (lower = more recent)
            "conviction": position_size_pct of manager's portfolio
            "manager_tier": weight based on historical hit rate
        }
        manager_signals.append(signal)

    # Breadth: how many informed managers agree
    n_accumulators = count(s for s in manager_signals where s.direction == +1)
    n_reducers = count(s for s in manager_signals where s.direction == -1)
    breadth = (n_accumulators - n_reducers) / (n_accumulators + n_reducers + 1)

    # Recency-weighted conviction
    for s in manager_signals:
        s.weight = s.conviction × s.manager_tier × decay(s.recency, halflife=90)

    conviction_score = sum(s.direction × s.weight for s in manager_signals)

    # Composite
    IA_raw = 0.40 × breadth + 0.60 × conviction_score
    IA = cross_sectional_z_score(IA_raw)
```

**Decay function:** Exponential with 90-day half-life. A 13F filing is freshest on
disclosure day and loses half its information content every 90 days as the market
incorporates it. Filings older than 360 days contribute negligibly.

**Cross-sectional z-score:** Computed per scoring run across the full universe. This
ensures the signal is relative (how much institutional interest does this name have
vs the universe median) rather than absolute.

#### 3.1.5 Key Design Decisions

1. **CUSIP-first resolution:** Entity matching goes through CUSIP, never issuer name
   strings. DEM learned this the hard way — different filings use different names for the
   same entity. (See DEM `cusip_static_map.json` and the 2026-11-01 CUSIP-first
   resolution fix.)

2. **Filing age is computed at run time**, not stored at build time. DEM's coinvest
   builder 1.1.0 was specifically designed to recompute filing age dynamically. A frozen
   filing age is stale by definition.

3. **Staleness gate:** If the most recent filing for a manager is >230 days old, that
   manager's signals are suppressed. The manager may have changed CIK, stopped filing, or
   ceased operations. (Learned from DEM: Broadfin moved CIK and its old book persisted
   as a ghost signal.)

4. **No double-counting:** If a manager files both a 13F and a 13D/13G for the same
   name in the same quarter, use the 13D/13G (higher conviction signal) and suppress the
   13F entry. The 13D/13G is a voluntary disclosure of a large stake; it supersedes the
   routine quarterly report.

---

### 3.2 Factor 2: Earnings Quality (EQ)

**Mechanism class:** Behavioral mispricing  
**DEM analog:** None (DEM operates in pre-revenue biotech where earnings are irrelevant)

#### 3.2.1 Thesis

The market anchors on reported earnings (net income, EPS) and systematically underweights
the quality of those earnings. Companies can temporarily inflate reported earnings through
accruals — recognizing revenue early, deferring expenses, capitalizing costs. High
accruals predict future earnings disappointments because the non-cash component of
earnings is less persistent than the cash component.

This is the Sloan (1996) accrual anomaly. Thirty years of academic evidence confirms it
persists, particularly in small and mid-cap equities where analyst coverage is thinner
and the information processing is slower. The mechanism is behavioral: investors fixate
on the headline number and underweight the footnotes.

#### 3.2.2 Inputs

| Input | Source | Frequency | PIT Anchor |
|---|---|---|---|
| Cash flow from operations (CFO) | 10-K / 10-Q | Quarterly | SEC filing accepted date |
| Net income (NI) | 10-K / 10-Q | Quarterly | SEC filing accepted date |
| Total assets (current and prior period) | Balance sheet | Quarterly | SEC filing accepted date |
| Change in net operating assets (ΔNOA) | Balance sheet | Quarterly | SEC filing accepted date |
| Revenue | Income statement | Quarterly | SEC filing accepted date |
| Accounts receivable | Balance sheet | Quarterly | SEC filing accepted date |

**PIT discipline:** Every financial input uses the SEC filing accepted date (the date
EDGAR received the document), not the fiscal period end date. A company's Q3 ending
September 30 is not available until the 10-Q is filed — typically 30-45 days later.

#### 3.2.3 Scoring

Three sub-components, each capturing a different aspect of earnings quality:

**A. Accrual ratio (Sloan measure)**
```
accruals = (net_income - cfo) / avg_total_assets
EQ_accrual = -accrual_ratio  # negative: lower accruals = higher quality
```

**B. Cash conversion**
```
cash_conversion = cfo / (net_income + ε)
EQ_cash = cash_conversion  # positive: higher conversion = higher quality
```

For companies with negative net income, cash conversion is undefined. Use the absolute
cash flow ratio instead:
```
if net_income < 0:
    EQ_cash = cfo / (|net_income| + ε)  # positive CFO despite losses = better quality
```

**C. Revenue quality (DSO change)**
```
dso_current = (accounts_receivable / revenue) × 90  # days sales outstanding
dso_prior = (accounts_receivable_prior / revenue_prior) × 90
delta_dso = dso_current - dso_prior
EQ_dso = -delta_dso / (dso_prior + ε)  # negative: rising DSO = deteriorating quality
```

Rising DSO means the company is booking revenue but not collecting cash — a classic
quality warning.

**Composite:**
```
EQ_raw = 0.45 × z(EQ_accrual) + 0.35 × z(EQ_cash) + 0.20 × z(EQ_dso)
EQ = cross_sectional_z_score(EQ_raw)
```

Accrual ratio gets the most weight because it has the deepest academic evidence and the
cleanest economic interpretation. Cash conversion is second. DSO change is a supporting
signal — useful but noisier.

#### 3.2.4 Applicability Filter

Earnings quality is meaningless for pre-revenue or pre-profit companies. These names are
excluded from EQ scoring:

- Revenue < $10M (trailing 4 quarters): **EQ = neutral (z = 0)**
- Net income AND CFO both negative for 4+ consecutive quarters: **EQ = neutral (z = 0)**

Neutral means the factor does not contribute to the composite for these names — it does
not penalize them, but it does not help them either. The other factors carry the ranking
burden for pre-profit names.

---

### 3.3 Factor 3: Price Momentum (PM)

**Mechanism class:** Behavioral mispricing  
**DEM analog:** None directly (DEM is cross-sectional, not time-series)

#### 3.3.1 Thesis

Prices underreact to gradual information arrival. When a company's fundamentals improve
(or deteriorate) over multiple quarters, the market adjusts incrementally rather than
jumping to the new fair value. This creates a trend that is partially predictable from
past returns.

The academic evidence (Jegadeesh & Titman 1993, updated through 2020s) shows 6-12 month
momentum with 1-month reversal exclusion is persistent across markets and time periods.
The key refinement: **momentum crashes** (sharp reversals in high-momentum names) are
the primary risk. The drawdown guard mitigates this.

#### 3.3.2 Inputs

| Input | Source | Frequency | PIT Anchor |
|---|---|---|---|
| Adjusted close prices (12 months) | Market data vendor | Daily | Same-day |
| 52-week high | Derived from price series | Daily | Same-day |
| Sector classification | Industry vendor or exchange | Quarterly | — |

#### 3.3.3 Scoring

**A. Core momentum (12-1 month return)**
```
ret_12m = (price_today / price_252_trading_days_ago) - 1
ret_1m = (price_today / price_21_trading_days_ago) - 1

core_momentum = ret_12m - ret_1m  # skip the most recent month
```

The 1-month exclusion removes short-term reversal (liquidity provision, mean reversion)
that contaminates the momentum signal.

**B. Sector-relative momentum**
```
sector_ret_12_1 = median(core_momentum for names in same sector)
relative_momentum = core_momentum - sector_ret_12_1
```

Sector-relative momentum isolates idiosyncratic strength from sector rotation. A biotech
company up 30% in a sector that is up 25% has +5% relative momentum — much less
impressive than the raw number.

**C. Drawdown guard**
```
drawdown_from_high = (price_today / 52_week_high) - 1  # negative value

if drawdown_from_high < -0.40:
    PM = -3.0  # hard penalty — force to bottom of ranking
elif drawdown_from_high < -0.25:
    PM_adjustment = drawdown_from_high × 2.0  # scale penalty
else:
    PM_adjustment = 0.0
```

The drawdown guard is the single most important risk control in a momentum strategy.
Names down >40% from their high are either (a) in secular decline, (b) experiencing
a fundamental break, or (c) value traps. In all three cases, they are not momentum
candidates.

Names down 25-40% receive a graded penalty rather than a hard exclusion. Some of these
are genuine dip-buying opportunities; others are early-stage declines. The penalty
reduces their ranking without eliminating them.

**Composite:**
```
PM_raw = 0.60 × z(relative_momentum) + 0.40 × z(core_momentum) + PM_adjustment
PM = cross_sectional_z_score(PM_raw)
```

Relative momentum gets more weight because it has better risk-adjusted performance
(removing sector beta reduces volatility without reducing return) and is more
interpretable ("this name is outperforming its sector" is a stronger statement than
"this name is going up").

---

### 3.4 Factor 4: Catalyst Density (CD)

**Mechanism class:** Structural friction  
**DEM analog:** event_type_score (5/5 Checklist v2 pass, overlay-only)

#### 3.4.1 Thesis

Names with imminent, identifiable catalysts have a structural advantage: the thesis gets
tested soon. This matters for portfolio efficiency — capital is not tied up waiting for
an indefinite "eventually." Names without near-term catalysts may be correctly valued in
the long run, but they impose an opportunity cost.

This is not a directional signal — it does not predict whether the catalyst will be
positive or negative. The other factors (institutional accumulation, earnings quality,
momentum) handle direction. Catalyst density ensures the portfolio is concentrated in
names where the clock is running.

In the DEM system, event_type_score earned a 5/5 Checklist v2 rating as an overlay. It
did not improve the B6 selector as a ranking input, but it added value as a filtering
dimension. In the broad equity model, it is a ranking input because the universe is
larger and more heterogeneous — many names have no catalyst at all, and excluding them
improves portfolio efficiency.

#### 3.4.2 Inputs

| Input | Source | Frequency | PIT Anchor |
|---|---|---|---|
| Earnings dates (confirmed) | Corporate calendar | Rolling | Announcement date |
| Earnings dates (estimated) | Sell-side consensus | Rolling | Consensus snapshot date |
| Regulatory decisions (FDA, FTC, etc.) | Agency calendars | Event-driven | Publication date |
| Corporate events (investor days, guidance updates) | 8-K filings | Event-driven | Filing date |
| Index rebalance dates | Index provider | Quarterly | Announcement date |
| M&A milestones (shareholder votes, regulatory deadlines) | Proxy filings | Event-driven | Filing date |
| Activist campaign milestones (13D deadlines, board elections) | SEC filings | Event-driven | Filing date |
| Product launch dates | Company guidance, trade publications | Varies | Disclosure date |

#### 3.4.3 Scoring

```
For each name, within a 90-day forward window:

    catalysts = identify_upcoming_catalysts(name, as_of_date, horizon=90)

    weighted_density = 0
    for c in catalysts:
        # Materiality weight: how much this event typically moves the stock
        mat_w = MATERIALITY_WEIGHTS[c.event_type]

        # Timing weight: imminent catalysts count more
        days_away = (c.expected_date - as_of_date).days
        time_w = max(0, 1.0 - days_away / 90.0)

        # Precision weight: confirmed dates count more than estimates
        prec_w = PRECISION_WEIGHTS[c.date_precision]

        weighted_density += mat_w × time_w × prec_w

    CD = cross_sectional_z_score(weighted_density)
```

**Materiality weights:**

| Event Type | Weight | Rationale |
|---|---|---|
| Earnings release | 1.0 | Universal; always material |
| FDA binary decision (PDUFA, AdCom) | 2.5 | Highest single-event vol for pharma/biotech |
| M&A shareholder vote | 2.0 | Deal-breaker event |
| Regulatory decision (non-FDA) | 1.5 | Material but less binary |
| Activist proxy contest / board election | 1.8 | Governance catalyst |
| Index rebalance (inclusion) | 1.2 | Structural demand |
| Investor day / guidance update | 0.8 | Informative but lower vol |
| Product launch | 1.3 | Revenue catalyst |
| Conference presentation (medical/industry) | 0.5 | Low materiality individually |

**Precision weights:**

| Date Precision | Weight | |
|---|---|---|
| Confirmed (exact date announced) | 1.0 | |
| Estimated (consensus, typical pattern) | 0.7 | |
| Approximate (within a month) | 0.4 | |
| Vague (within a quarter) | 0.2 | |

#### 3.4.4 Zero-Catalyst Treatment

Names with no identifiable catalyst in the 90-day window receive `CD = -1.5` (a fixed
penalty, approximately the 10th percentile). They are not excluded — the other factors
may still rank them highly — but they start at a disadvantage.

This is a design choice. In DEM, event_type_score works as a binary (has catalyst / does
not). In broad equity, the graduated weighting is more appropriate because the catalyst
landscape is richer and more varied.

---

### 3.5 Factor 5: Financial Robustness (FR)

**Mechanism class:** Information asymmetry (specifically: the market misprices tail risk
in financially fragile companies)  
**DEM analog:** financial_score (negative penalty in the ranker — but contaminated by
market-cap look-ahead in historical data)

#### 3.5.1 Thesis

Companies with weak balance sheets and deteriorating financial health are overpriced
relative to their fundamental risk. The market systematically underestimates the
probability and magnitude of financial distress, particularly for companies that are not
yet in obvious trouble (the "gray zone" between healthy and distressed).

This is **not** a value signal. It does not say cheap companies are good (that is the
trap that EES Spec 117 filters out). It says financially fragile companies are bad — a
negative screen applied as a ranking penalty.

**Critical lesson from DEM:** The biotech screener's `financial_score` appeared to be a
strong negative penalty (NW-t = -3.41). This was entirely driven by market-cap look-
ahead: today's market cap correlated with past returns, and `financial_score` divided
cash by market cap. With point-in-time market data, `financial_score` IC = -0.001 — dead.

The broad equity version avoids this by (a) not using market cap in any formula, and
(b) using only balance-sheet and cash-flow ratios where numerator and denominator are
from the same filing.

#### 3.5.2 Inputs

| Input | Source | Frequency | PIT Anchor |
|---|---|---|---|
| Current ratio (current assets / current liabilities) | Balance sheet | Quarterly | Filing accepted date |
| Interest coverage (EBIT / interest expense) | Income statement | Quarterly | Filing accepted date |
| Debt-to-equity (total debt / total equity) | Balance sheet | Quarterly | Filing accepted date |
| Free cash flow (CFO - capex) | Cash flow statement | Quarterly | Filing accepted date |
| Cash and equivalents | Balance sheet | Quarterly | Filing accepted date |
| Altman Z-score components | Financial statements | Quarterly | Filing accepted date |
| Revenue trend (YoY growth, trailing 4Q) | Income statement | Quarterly | Filing accepted date |

**Explicitly excluded:**
- Market capitalization (look-ahead risk; see DEM contamination)
- Enterprise value (contains market cap)
- Any ratio with stock price in the denominator (P/E, P/B, etc. — these are valuation
  signals, not robustness signals, and belong in the EES trap overlay, not here)

#### 3.5.3 Scoring

**A. Liquidity adequacy**
```
current_ratio_score = clamp((current_ratio - 1.0) / 1.5, -1.0, 1.0)
# Below 1.0 = cannot cover current liabilities; penalty scales
```

**B. Leverage discipline**
```
if total_equity <= 0:
    leverage_score = -1.0  # negative equity = maximum penalty
else:
    de_ratio = total_debt / total_equity
    leverage_score = clamp(1.0 - de_ratio / 3.0, -1.0, 1.0)
    # D/E > 3.0 = maximum penalty; D/E = 0 = maximum score
```

**C. Debt service capacity**
```
if interest_expense <= 0:
    coverage_score = 1.0  # no debt service = no risk
else:
    coverage_score = clamp((interest_coverage - 2.0) / 8.0, -1.0, 1.0)
    # Below 2.0 = danger zone; above 10.0 = comfortably covered
```

**D. Cash flow sustainability**
```
if revenue <= 0:
    fcf_score = 0.0  # pre-revenue; neutral
else:
    fcf_margin = free_cash_flow / revenue
    fcf_score = clamp(fcf_margin / 0.15, -1.0, 1.0)
    # Positive FCF margin scores well; negative FCF margin penalized
```

**E. Revenue trajectory**
```
if revenue_prior_year <= 0:
    rev_score = 0.0  # no basis for comparison
else:
    yoy_growth = (revenue_ttm - revenue_prior_year) / revenue_prior_year
    rev_score = clamp(yoy_growth / 0.30, -1.0, 1.0)
    # Growth up to 30%+ scores maximally; decline penalized proportionally
```

**Composite:**
```
FR_raw = (0.20 × current_ratio_score
        + 0.20 × leverage_score
        + 0.20 × coverage_score
        + 0.25 × fcf_score
        + 0.15 × rev_score)
FR = cross_sectional_z_score(FR_raw)
```

FCF gets the highest weight because it is the most forward-looking and hardest to
manipulate. Leverage and liquidity are static snapshots that can be window-dressed at
quarter-end. Cash flow sustainability is a better measure of genuine financial health.

#### 3.5.4 Sector-Specific Adjustments

Some sectors structurally carry more leverage (utilities, REITs, financials) or lower
current ratios (subscription businesses with deferred revenue). The sub-scores are
adjusted relative to sector medians:

```
adjusted_score = raw_score - sector_median_score
```

This prevents utilities from being universally penalized for high leverage when high
leverage is the norm for their business model. The question is not "is this company
levered?" but "is this company more levered than its sector peers?"

#### 3.5.5 Financial Company Exclusion

Banks, insurance companies, broker-dealers, and other financial institutions have
fundamentally different balance sheet structures. Current ratio, D/E, and interest
coverage are meaningless for these entities. Financial companies (GICS sector 40) receive:

- Liquidity adequacy: **replaced by** Tier 1 capital ratio (if available) or neutral
- Leverage discipline: **replaced by** tangible equity / assets or neutral
- Debt service capacity: **neutral** (interest is their business, not a cost to cover)
- FCF sustainability and revenue trajectory: **unchanged**

---

## 4. Composite Construction

### 4.1 Composite Formula

```
alpha_score = w_IA × IA + w_EQ × EQ + w_PM × PM + w_CD × CD + w_FR × FR
```

### 4.2 Default Weights

| Factor | Weight | Rationale |
|---|---|---|
| Institutional Accumulation (IA) | 0.30 | Strongest mechanism; DEM's dominant signal |
| Earnings Quality (EQ) | 0.20 | Deep academic evidence; mechanistically clean |
| Price Momentum (PM) | 0.20 | Persistent behavioral anomaly; diversifies IA |
| Catalyst Density (CD) | 0.15 | Portfolio efficiency; ensures catalysts are imminent |
| Financial Robustness (FR) | 0.15 | Negative screen; removes fragile names |

**Why these weights:**

Institutional accumulation gets the most weight because the DEM experience demonstrated
that informed capital flow is the most durable and hardest-to-replicate signal. The
biotech coinvest_score_z was the only signal that survived every robustness test. In broad
equity, the same mechanism applies: when skilled managers with concentrated portfolios
accumulate a position, they have information the market has not fully priced.

Earnings quality and momentum each get 0.20. They are independently strong (decades of
academic evidence) and mechanistically distinct (accounting manipulation vs price trend).
Their cross-sectional correlation is typically low (< 0.15) because high-quality
earnings and price momentum are driven by different aspects of a company's situation.

Catalyst density and financial robustness each get 0.15. They are supporting factors:
catalyst density improves timing efficiency (capital is not idle); financial robustness
removes tail risk (fragile companies may rally but their downside is catastrophic).
Neither is strong enough as a standalone signal to warrant higher weight.

### 4.3 Weight Governance

Weights are **fixed at deployment and changed only through governed re-evaluation.**

The DEM learned this through the v1.14.0 promotion process: every weight change is a
model change that resets the out-of-sample clock. The temptation to tune weights to
recent performance is strong and must be resisted — it is the fastest path to overfitting.

Weight changes require:
1. A written rationale citing forward (not backtested) evidence
2. Standalone Checklist v2 confirmation that the factor being upweighted still passes
3. A 20-day forward shadow comparison: new weights vs old weights
4. Operator sign-off

### 4.4 Missing Data Handling

When a factor's inputs are unavailable for a name:

| Scenario | Treatment |
|---|---|
| All inputs for one factor missing | Factor weight redistributed pro-rata to other factors |
| Partial inputs for one factor missing | Factor scored on available sub-components; confidence flagged |
| More than two factors missing | Name receives `alpha_score = NaN`; excluded from ranking |

```python
# Weight redistribution example: EQ unavailable (pre-revenue company)
available_weights = {IA: 0.30, PM: 0.20, CD: 0.15, FR: 0.15}
total_available = sum(available_weights.values())  # 0.80
normalized = {k: v / total_available for k, v in available_weights.items()}
# Result: IA: 0.375, PM: 0.25, CD: 0.1875, FR: 0.1875
```

This preserves the relative ordering of available factors while ensuring the composite
uses the full [0, 1] scoring range.

---

## 5. Normalization

### 5.1 Cross-Sectional Z-Score

Every factor is converted to a z-score relative to the scoring-day universe:

```
z_i = (raw_score_i - median(raw_scores)) / MAD(raw_scores)
```

MAD (median absolute deviation) is used instead of standard deviation because it is
robust to outliers. A single extreme value does not distort the entire distribution.

Z-scores are **winsorized** at ±3.0 to prevent extreme outliers from dominating the
composite:

```
z_i = clamp(z_i, -3.0, +3.0)
```

### 5.2 Why Not Percentile Ranks?

Percentile ranks are an alternative normalization that many factor models use. They have
the advantage of being uniformly distributed and outlier-proof.

The disadvantage: they destroy information about the magnitude of differences. A name
at the 90th percentile and one at the 99th percentile are treated as having a similar
score, when in reality the 99th percentile name may be dramatically stronger on that
factor.

Z-scores preserve this magnitude information while still being comparable across factors.
The winsorization handles the outlier problem. This is a judgment call — either approach
is defensible. The z-score approach is more informative for operator review ("this name
is 2.5 standard deviations above median on institutional accumulation" is more useful
than "this name is at the 95th percentile").

---

## 6. Integration with EES Gates

### 6.1 Pipeline Architecture

```
Full Universe (all public equity, e.g., Russell 3000 or broader)
    │
    ▼
Alpha Model (this spec): rank by composite score
    │
    ├── IA (0.30) + EQ (0.20) + PM (0.20) + CD (0.15) + FR (0.15)
    │
    ▼
EES Broad Equity Gates (Spec 117): exclude
    │
    ├── Bottom 20% on trap overlay (remove value traps)
    ├── Bottom 10% on structure overlay (remove bad structure)
    │
    ▼
Top-N Selection: equal-weight the top N surviving names
    │
    ├── Default N = 30 (DEM-validated; stable K=25-35 plateau)
    ├── Larger universes may warrant N = 50-100
    │
    ▼
Portfolio: equal-weight, rebalance monthly
```

### 6.2 Independence Requirement

The alpha model and the EES gates must be validated independently:

1. **Alpha model alone:** Does the top-N portfolio (without EES gates) outperform the
   equal-weight universe? This tests whether the ranking has signal.
2. **EES gates alone:** Does the gated universe (without alpha ranking) outperform the
   ungated universe? This tests whether the gates remove drag.
3. **Combined:** Does the gated-then-ranked portfolio outperform both the ungated-ranked
   and the gated-unranked portfolios? This tests whether the combination is additive.

If the combined system is not better than either component alone, the integration has no
value and the components should be used separately.

### 6.3 Factor Overlap with EES

Two potential overlaps must be monitored:

| Alpha Factor | EES Dimension | Overlap Risk | Mitigation |
|---|---|---|---|
| Financial Robustness (FR) | Structure overlay (liquidity friction) | Both penalize financially weak names | FR focuses on balance sheet; structure overlay on execution cost. Monitor cross-correlation; if > 0.35, reduce FR weight or merge into EES. |
| Institutional Accumulation (IA) | Trap overlay (consensus divergence) | Both reflect informed opinion about fair value | IA measures what informed capital *does*; trap overlay measures what consensus *implies*. Low conceptual overlap but monitor empirically. |

---

## 7. Validation Framework

### 7.1 Per-Factor Standalone Tests

Each factor must pass before entering the composite:

| Test | Requirement |
|---|---|
| Cross-sectional Spearman IC | Mean IC > 0 at 20d horizon, t > 2.0 |
| Fama-MacBeth alpha | Significant after controlling for Fama-French 5 + momentum |
| Bootstrap 95% CI | Lower bound > 0 on monthly excess return |
| LOSO (Leave-One-Year-Out) | IC positive in >= 75% of years |
| FDR (Benjamini-Hochberg) | Survives at 5% across all tested sub-components |

Factors failing any test are excluded. Marginal factors (e.g., passing IC but failing
LOSO) are placed in shadow mode for forward monitoring.

### 7.2 Composite Tests

| Test | Requirement |
|---|---|
| Top-30 vs EW universe excess return | Positive, t > 2.0 over evaluation period |
| Top-30 Sharpe ratio | > 0.30 (after transaction costs at 50 bps round-trip) |
| Maximum drawdown relative to benchmark | < 25pp |
| Hit rate (monthly) | > 55% |
| Turnover | < 30% monthly (consistent with low-frequency rebalancing) |

### 7.3 Forward Shadow

Before live deployment, accumulate 60+ trading days of forward shadow data:

| Shadow Arm | Description |
|---|---|
| Arm 1: Alpha model only (no EES gates) | Tests alpha model standalone |
| Arm 2: EES gates only (equal-weight survivors) | Tests EES standalone |
| Arm 3: Alpha + EES gates (production configuration) | Tests the combined system |
| Arm 4: EES gates + random ranking (control) | Tests whether alpha adds value beyond EES |
| Arm 5: Universe equal-weight (baseline) | Benchmark |

All arms are equal-weight, Top-30 (or Top-N), rebalanced monthly, recorded daily. The
evaluation gate: Arm 3 must beat Arm 5 by > 0.5pp/month with > 55% hit rate over the
shadow period.

---

## 8. Data Infrastructure

### 8.1 Point-in-Time Requirements

| Data Type | PIT Anchor | Staleness Limit | Action if Stale |
|---|---|---|---|
| 13F filings | SEC filing accepted date | 230 days | Suppress manager |
| Financial statements | SEC filing accepted date | 120 days | Use last available; flag confidence |
| Prices / volume | Market close date | 1 business day | Hard fail — do not score |
| Analyst estimates | Consensus snapshot date | 5 business days | Use last snapshot; flag confidence |
| Event calendar | Disclosure / announcement date | 1 business day | Hard fail on confirmed events |
| Short interest | FINRA reporting date | 15 business days | Use last available |

### 8.2 Archival

Every scoring run archives:
1. All input data as consumed (not just the final scores)
2. The scoring parameters (weights, gate thresholds, registry version)
3. The output (all factor scores, composite, gates, eligibility)
4. A SHA256 hash of the output for audit trail

Archival enables true point-in-time reconstruction: re-running the model on any
historical date should produce byte-identical output when given the same archived inputs.
This is the CCFT (Canonical, Complete, Frozen, Timestamped) principle from DEM.

---

## 9. Rebalancing and Construction

### 9.1 Rebalancing Frequency

Monthly, on a fixed calendar day (e.g., first trading day of each month). Not quarterly
(too slow for momentum), not weekly (too much turnover and transaction cost).

The DEM biotech screener rebalances weekly (Monday) because the biotech universe has
higher event frequency and shorter catalyst horizons. Broad equity has longer cycles
and higher transaction costs per name, justifying a monthly cadence.

### 9.2 Construction Rules

| Parameter | Value | Rationale |
|---|---|---|
| Weighting | Equal-weight | Ordinal model; no confidence sizing (ECE lesson from DEM) |
| Portfolio size (N) | 30 (default), 50-100 for larger AUM | DEM validated K=30; K=25-35 stable plateau |
| Turnover buffer | 5 ranks | A name ranked 31st is held if it was in the prior Top-30; replaced only at rank 36+ |
| Maximum sector weight | 25% | Prevents inadvertent sector concentration |
| Minimum market cap | $500M (or $1B for institutional) | Below this, liquidity friction dominates |
| Maximum single-name weight at rebalance | 1/N | Equal-weight enforced |

### 9.3 Turnover Buffer

The 5-rank buffer reduces unnecessary trading. Without it, a name oscillating between
rank 29 and 31 generates two round-trip trades per month for zero information content.
The buffer requires a clear deterioration in ranking (falling to 36+) before selling.

```
if name in prior_portfolio:
    sell if current_rank > N + buffer  # e.g., rank 36+ for N=30, buffer=5
else:
    buy if current_rank <= N           # only new entrants at top-30
```

### 9.4 Transaction Cost Model

All performance reporting must be net of estimated transaction costs:

| Cost Component | Estimate | Source |
|---|---|---|
| Commission | 0 (assumed) | Most brokers are zero-commission for equity |
| Half-spread (market impact) | 5-25 bps depending on market cap | Empirical from execution data |
| Base assumption for reporting | 50 bps round-trip (25 bps each way) | Conservative default |
| Sensitivity analysis | Also report at 25 and 100 bps | Bracketing |

The DEM biotech screener uses 50 bps as its default cost assumption. This is
appropriate for the biotech universe (smaller names, wider spreads). For large-cap-heavy
broad equity, 25-30 bps may be more realistic, but reporting at 50 bps as the primary
figure provides a conservative baseline.

---

## 10. Monitoring and Kill Switches

### 10.1 Ongoing Monitoring

| Metric | Frequency | Alert Threshold | Action |
|---|---|---|---|
| Per-factor IC (cross-sectional Spearman) | Weekly | IC < 0 for 8 consecutive weeks | Shadow the factor |
| Composite IC | Weekly | IC < 0 for 4 consecutive weeks | Full model review |
| Factor cross-correlations | Monthly | Any pair > 0.50 | Merge or drop one factor |
| Portfolio vs benchmark spread | Monthly | Negative for 3 consecutive months | Governance review |
| Turnover | Monthly | > 50% | Investigate (likely data issue or regime break) |
| Coverage (% of universe scorable) | Daily | < 80% | Data pipeline investigation |
| Alpha vs EES gate interaction | Monthly | Gated alpha underperforms ungated alpha | Review gate thresholds |

### 10.2 Factor Kill Switches

Each factor has an automatic kill switch:

```
if factor_rolling_ic(window=52_weeks) < -0.02:
    factor_weight = 0  # killed
    redistribute weight to surviving factors
    log event with full diagnostic
    alert operator
```

A killed factor remains in shadow mode — it continues to be scored and monitored, but its
weight is zero. Reinstatement requires:
1. IC recovery to positive for 26 consecutive weeks
2. Checklist v2 re-run on the recovery period
3. Operator approval

This is modeled on DEM's treatment of inst_delta_z, which was zeroed in the selector on
2026-05-04 when its rolling IC went negative. Reinstatement conditions were documented
in the governance log but have not been met.

### 10.3 Full Model Kill Switch

If the composite IC is negative for 13 consecutive weeks (one quarter):

1. The model enters **diagnostic mode** — rankings are still produced but flagged as
   `LOW_CONFIDENCE`
2. The operator is alerted
3. A full review is triggered: Is this a regime break? A data contamination? A structural
   change in the market?
4. No automatic recovery — operator must explicitly re-enable after review

---

## 11. Known Limitations and Open Questions

### 11.1 Limitations

1. **Manager registry is the model's biggest risk.** The institutional accumulation
   factor's performance depends entirely on selecting the right managers. A registry
   populated with managers who are skilled in one regime (e.g., 2020-2024 growth markets)
   may underperform in a different regime. The registry is also the most subjective
   input — the selection criteria are objective, but the initial candidate pool is
   discretionary.

2. **Earnings quality is irrelevant for ~20% of listed equity.** Pre-revenue biotech,
   early-stage tech, SPACs, and recent IPOs have no meaningful earnings to evaluate. These
   names are scored with a reduced factor set, which changes the composite's effective
   weights.

3. **Momentum crashes are mitigated but not eliminated.** The drawdown guard reduces
   crash exposure but does not prevent it. In a market-wide deleveraging (2008, 2020
   March), all momentum names sell off simultaneously and the guard triggers too late.

4. **Catalyst density depends on event calendar completeness.** If the event source
   misses a catalyst (e.g., an unannounced FDA decision, an activist campaign not yet
   filed), the factor underweights that name. The factor is only as good as the data
   pipeline feeding it.

5. **Financial robustness may penalize capital-intensive growth.** Companies in heavy
   investment cycles (building factories, acquiring competitors) have temporarily weak
   balance sheets. The sector adjustment (§3.5.4) partially addresses this, but within-
   sector variation remains.

6. **The 50 bps transaction cost assumption may be too high for large caps and too low
   for micro caps.** A tiered cost model by market cap would be more accurate but adds
   complexity.

### 11.2 Open Questions

1. **Should the manager registry use trailing returns or trailing IC to select managers?**
   Returns are simpler and more robust to measurement error. IC is more directly related
   to the signal's value. The current spec uses hit rate (returns), but IC-based selection
   should be tested.

2. **Is monthly rebalancing optimal?** The DEM biotech screener rebalances weekly.
   Quarterly might be better for large-cap broad equity (lower turnover, lower costs).
   This is an empirical question.

3. **Should the drawdown guard in momentum be absolute (-40%) or relative to the
   market?** In a bear market where the index is down 30%, a name down 40% may be
   underperforming by only 10%. The current spec uses absolute levels, which is more
   conservative.

4. **How should the model handle dual-listed securities, ADRs, and holding companies?**
   These create data quality issues (which price to use, which financial statements) and
   potential double-counting in the institutional accumulation factor.

5. **Should the composite weights adapt to regime?** In bear markets, financial robustness
   may deserve more weight. In bull markets, momentum may dominate. The current spec uses
   fixed weights (§4.3) to avoid overfitting, but regime-conditional weights are a
   research avenue.

---

## 12. Differences from DEM Biotech Model

| Dimension | DEM Biotech | Broad Equity Alpha (this spec) |
|---|---|---|
| Primary signal | coinvest_score_z (100% of selector) | Institutional accumulation (30% of composite) |
| Factor count | 1 (coinvest) to 2 (+ financial) | 5 |
| Universe | ~300 biotech names | Russell 3000+ |
| Earnings relevance | None (pre-revenue biotech) | Central (earnings quality factor) |
| Momentum | Not used | 20% of composite (12-1 month) |
| Catalyst treatment | event_type_score (overlay only) | Catalyst density (15% of composite) |
| Financial health | financial_score (contaminated by market-cap look-ahead) | Rebuilt with no market-cap dependency |
| Construction | EW Top-30, weekly rebalance | EW Top-30-100, monthly rebalance |
| Ranker role | pairwise_minimal (shadow since 2026-11-13) | This IS the ranker; no separate ranker step |
| EES integration | Diagnostic overlay (not in decision path) | Post-ranking gate (in decision path) |

---

## Appendix A: Factor Summary Card

```
┌─────────────────────────────────────────────────────────────┐
│  GENERALIZED EQUITY ALPHA MODEL — FACTOR SUMMARY            │
├──────────────────────┬──────────────┬───────────────────────┤
│  Factor              │  Weight      │  Mechanism            │
├──────────────────────┼──────────────┼───────────────────────┤
│  Institutional       │  0.30        │  Information           │
│  Accumulation (IA)   │              │  asymmetry            │
├──────────────────────┼──────────────┼───────────────────────┤
│  Earnings            │  0.20        │  Behavioral           │
│  Quality (EQ)        │              │  mispricing           │
├──────────────────────┼──────────────┼───────────────────────┤
│  Price               │  0.20        │  Behavioral           │
│  Momentum (PM)       │              │  mispricing           │
├──────────────────────┼──────────────┼───────────────────────┤
│  Catalyst            │  0.15        │  Structural           │
│  Density (CD)        │              │  friction             │
├──────────────────────┼──────────────┼───────────────────────┤
│  Financial           │  0.15        │  Information           │
│  Robustness (FR)     │              │  asymmetry            │
├──────────────────────┴──────────────┴───────────────────────┤
│  Construction: ordinal ranking → EES gates → EW Top-N       │
│  Rebalance: monthly │ Turnover buffer: 5 ranks              │
│  Validation: Checklist v2 per factor + 60-day forward shadow│
└─────────────────────────────────────────────────────────────┘
```

## Appendix B: Full Pipeline Diagram

```
    ┌──────────────────────────────────────────────────────────┐
    │                   DATA LAYER                             │
    │                                                          │
    │  13F/13D/13G  ──┐                                       │
    │  Form 4        ──┤── Manager Registry ── IA score        │
    │                  │                                       │
    │  10-K/10-Q    ──┤── Financial Extraction                │
    │                  ├── EQ score (accruals, cash, DSO)      │
    │                  └── FR score (liquidity, leverage, FCF) │
    │                                                          │
    │  Price data    ──── PM score (12-1 momentum + drawdown)  │
    │                                                          │
    │  Event calendar ── CD score (catalyst density)           │
    │                                                          │
    │  Options data  ──┐                                       │
    │  Consensus     ──┤── EES Spec 117 (trap + structure)    │
    │  SI data       ──┘                                       │
    └───────────────────────────┬──────────────────────────────┘
                                │
                                ▼
    ┌──────────────────────────────────────────────────────────┐
    │                  SCORING LAYER                           │
    │                                                          │
    │  Per name:                                               │
    │    IA_z ←── institutional accumulation (z-scored)        │
    │    EQ_z ←── earnings quality (z-scored)                  │
    │    PM_z ←── price momentum (z-scored)                    │
    │    CD_z ←── catalyst density (z-scored)                  │
    │    FR_z ←── financial robustness (z-scored)              │
    │                                                          │
    │  alpha_score = 0.30×IA + 0.20×EQ + 0.20×PM             │
    │              + 0.15×CD + 0.15×FR                         │
    │                                                          │
    │  Sort by alpha_score descending                          │
    └───────────────────────────┬──────────────────────────────┘
                                │
                                ▼
    ┌──────────────────────────────────────────────────────────┐
    │                  GATE LAYER (EES Spec 117)               │
    │                                                          │
    │  Trap gate: exclude bottom 20% trap_overlay              │
    │  Structure gate: exclude bottom 10% structure_overlay    │
    │  Quality-tier adjustment: tighter for LOW quality        │
    │                                                          │
    │  ees_eligible = trap_gate AND structure_gate              │
    └───────────────────────────┬──────────────────────────────┘
                                │
                                ▼
    ┌──────────────────────────────────────────────────────────┐
    │                  CONSTRUCTION LAYER                      │
    │                                                          │
    │  Top-N eligible names by alpha_score                     │
    │  Equal-weight (1/N per name)                             │
    │  Sector cap: 25% max                                     │
    │  Turnover buffer: hold if rank <= N+5; sell if > N+5     │
    │  Rebalance: first trading day of month                   │
    └───────────────────────────┬──────────────────────────────┘
                                │
                                ▼
    ┌──────────────────────────────────────────────────────────┐
    │                  MONITORING LAYER                        │
    │                                                          │
    │  Weekly: per-factor IC, composite IC                     │
    │  Monthly: factor correlations, gate interaction, spread  │
    │  Kill switches: factor (52-week IC < -0.02)              │
    │                 model (13-week composite IC < 0)         │
    │  Forward shadow: 5 arms, 60+ trading days                │
    └──────────────────────────────────────────────────────────┘
```

## Appendix C: Lessons from DEM Applied to This Design

| DEM Lesson | How It Shaped This Spec |
|---|---|
| Coinvest was the only durable signal | IA gets the highest weight (0.30) and is the only information-asymmetry factor in the ranking |
| financial_score was contaminated by market-cap look-ahead | FR explicitly excludes market cap and any price-denominated ratio |
| inst_delta_z went negative and was zeroed | Every factor has an automatic kill switch with documented reinstatement conditions |
| Pairwise ranker had ECE = 0.129 (poor calibration) | Construction is equal-weight; composite is ordinal only |
| event_type_score was 5/5 but did not improve the selector | CD is a ranking input (not overlay) because the broader universe has more variation |
| Backtested claims were deprecated system-wide (2026-11-13) | Validation requires forward shadow (60+ days) before any performance claims |
| Registry changes reset the out-of-sample clock | Manager registry is a governed input; changes trigger clock reset |
| The trap inversion (cheap underperforms) | EES gates sit downstream of ranking; the alpha model does not screen for value |
| CUSIP-first, not issuer-first | Institutional data matched by CUSIP throughout |
| PIT contamination is the single biggest threat | Every input has a PIT anchor and staleness limit; no exceptions |
