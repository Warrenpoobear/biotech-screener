# Spec 119 — Generalized Alpha Model for a Grokbot Screener

**Status:** RESEARCH / DESIGN  
**Created:** 2026-11-15  
**Governance Tier:** 4 (Research Judgment)  
**Companion Specs:** Spec 116 (EES biotech), Spec 117 (EES broad equity), Spec 118 (quant alpha model)  
**Scope:** Methodology only — no implementation, no production change  
**Inference Engine:** xAI Grok (1M context via SuperGrok OAuth, OpenAI-compatible API)

---

## 1. Purpose

This document specifies a hybrid alpha model for a Grok-powered equity screening bot
("grokbot"). Unlike the pure quantitative model in Spec 118, this system uses an LLM as
a first-class analytical component — processing unstructured information (earnings
transcripts, news, filings, management commentary) that quantitative factors cannot
capture.

The grokbot does not replace the quantitative stack. It augments it with a structured
information-processing layer that extracts, scores, and governs LLM-derived signals
alongside traditional factor scores. The critical design challenge: **how to make an
LLM's outputs reproducible, auditable, and governable enough to enter a systematic
screening pipeline.**

### 1.1 Why Grok

Grok is the inference engine for this design for specific architectural reasons:

- **1M token context** (via SuperGrok OAuth): sufficient to process an entire 10-K, a
  full earnings transcript, or multiple analyst reports in a single call
- **Real-time information access**: Grok's native web access (via X/Twitter integration)
  provides a live information feed that traditional data vendors lag by hours or days
- **OpenAI-compatible API**: slots into existing infrastructure without custom client
  libraries. The Hermes fleet already has xAI Grok integration (v0.14.0)
- **Cost profile**: competitive inference pricing for the context window size, making
  per-name scoring economically viable at universe scale

The design is LLM-agnostic in principle — the scoring framework, governance controls, and
validation requirements apply regardless of which model runs the inference. But the
specific prompt engineering, context management, and output parsing are calibrated to
Grok's behavior.

### 1.2 Lessons from the DEM Grok Experience

The DEM biotech screener operated a `grok_biotech_watch` agent that used Grok for news
monitoring. The April 2026 audit (grok_news_feed_audit_2026_04_26.md) found:

1. **Broken-and-dormant**: XAI_API_KEY missing since 2026-04-20, last run 2026-03-31
2. **Hallucination**: Grok fabricated a 2023-dated clinical trial event for a name that
   had no such trial
3. **Source hierarchy violation**: Grok output carried `official_confirmation: True` and
   `source_type: "official_company"` — the model asserted authority it did not have
4. **Path drift**: Producer and consumer used different directory and filename conventions
5. **No schema validation**: Producer wrote ad-hoc dicts, not validated data contracts

Every one of these failures informs this spec's design. The grokbot screener treats LLM
output as **untrusted data by default** — structured, scored, and validated before it
touches any decision.

---

## 2. Architecture

### 2.1 Three-Layer Design

```
Layer 1: QUANTITATIVE BACKBONE (Spec 118 factors)
    │     Institutional Accumulation, Earnings Quality,
    │     Price Momentum, Catalyst Density, Financial Robustness
    │     ─── deterministic, reproducible, auditable ───
    │
    ▼
Layer 2: LLM INTELLIGENCE LAYER (this spec)
    │     Earnings Signal Extraction, Narrative Quality Assessment,
    │     Catalyst Intelligence, Management Credibility Score
    │     ─── structured prompts, schema-validated output,
    │         confidence-gated, reproducibility-tracked ───
    │
    ▼
Layer 3: EES GATE LAYER (Spec 117)
          Trap overlay, Structure overlay
          ─── percentile-based exclusion gates ───
```

Layer 1 is the foundation — the quantitative backbone from Spec 118, unchanged. It runs
with or without Layer 2. If the LLM layer fails, degrades, or is killed, the system
falls back to pure quantitative scoring with no disruption.

Layer 2 is the LLM intelligence layer — four structured signals extracted by Grok from
unstructured sources. Each signal is scored on a fixed schema, validated against ground
truth where available, and gated by confidence thresholds before entering the composite.

Layer 3 is the EES gate system from Spec 117, applied after both layers have scored.

**Fallback guarantee:** The system must produce valid rankings if Layer 2 is entirely
unavailable (API down, budget exhausted, model degraded). Layer 2 is additive; it is
never load-bearing.

### 2.2 Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                 UNSTRUCTURED SOURCES                        │
│                                                             │
│  Earnings call transcripts  ─┐                              │
│  10-K / 10-Q full text       ├── Grok inference calls       │
│  News articles / press releases│  (structured prompts,      │
│  Analyst reports             ─┤   schema-validated output)  │
│  Management presentations    ─┤                             │
│  SEC comment letters         ─┤                             │
│  Industry conference notes   ─┘                             │
│                                                             │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              SIGNAL EXTRACTION (Layer 2)                     │
│                                                             │
│  ES: Earnings Signal Extraction ────┐                       │
│  NQ: Narrative Quality Assessment ──┤── LLM scores          │
│  CI: Catalyst Intelligence ─────────┤   (schema-validated,  │
│  MC: Management Credibility ────────┘   confidence-gated)   │
│                                                             │
│  Each signal:                                               │
│    prompt_hash ── output_hash ── confidence ── score        │
│                                                             │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    COMPOSITE                                │
│                                                             │
│  alpha_score = quant_composite × (1 - llm_weight)           │
│              + llm_composite × llm_weight                   │
│                                                             │
│  llm_weight = base_llm_weight × avg_confidence              │
│  (scales to 0 when confidence is low or data is stale)      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. LLM Signal Specifications

### 3.1 Signal 1: Earnings Signal Extraction (ES)

**Purpose:** Extract forward-looking information from earnings call transcripts that
consensus estimates have not yet incorporated.

**Why LLM:** Earnings calls contain 60-90 minutes of management commentary, analyst Q&A,
and tone/emphasis signals that are compressed into a single EPS number by consensus.
An LLM can read the full transcript and extract specific forward-looking statements,
hedged language, evasive non-answers, and emphasis patterns that a quantitative model
cannot process.

#### 3.1.1 Input

The most recent earnings call transcript, sourced from:
- Primary: SEC 8-K filing (exhibit 99.1 or 99.2, when the transcript is filed)
- Fallback: Third-party transcript provider (Seeking Alpha, FactSet, Bloomberg)
- PIT anchor: Transcript available date (filing date for SEC; publication date otherwise)
- Staleness limit: 100 calendar days (one full quarter plus buffer)

#### 3.1.2 Prompt Structure

```
SYSTEM:
You are a senior equity analyst extracting forward-looking signals from
earnings call transcripts. You produce structured JSON output only.
Do not editorialize. Do not hallucinate facts not in the transcript.
If a field cannot be determined from the transcript, output null.

USER:
Analyze the following earnings call transcript for {ticker} ({company_name}),
held on {call_date}.

Extract these signals:

1. GUIDANCE_DIRECTION: Did management raise, maintain, lower, or withdraw
   forward guidance? One of: RAISED, MAINTAINED, LOWERED, WITHDRAWN, NONE.
   Include the specific metric(s) guided and the direction of change.

2. TONE_SHIFT: Compare management's tone on this call vs the typical tone
   for this company. Score from -2 (significantly more negative/defensive)
   to +2 (significantly more confident/expansive). 0 = neutral/typical.
   Cite the specific language that drove your assessment.

3. ANALYST_FRICTION: How contentious was the Q&A? Score 0-3.
   0 = routine/friendly, 1 = pointed questions with direct answers,
   2 = repeated pressing on specific topics, 3 = hostile/adversarial.
   Note which topics generated friction.

4. FORWARD_STATEMENTS: List up to 5 concrete forward-looking statements
   (revenue targets, margin expectations, capex plans, product timelines,
   hiring/headcount signals). For each, note whether it is NEW (not in
   prior guidance), REITERATED, or MODIFIED.

5. EVASION_FLAGS: List any questions where management gave a non-answer,
   deflected, or explicitly declined to comment. These are potential risk
   signals.

6. CATALYST_MENTIONS: Any mention of upcoming events (product launches,
   regulatory decisions, restructuring milestones, M&A activity, debt
   maturity) with approximate timing.

Output valid JSON matching the schema below. Every field is required.
Null is acceptable where the transcript provides no basis for a judgment.

[TRANSCRIPT FOLLOWS]
{transcript_text}
```

#### 3.1.3 Output Schema

```json
{
  "ticker": "string",
  "call_date": "YYYY-MM-DD",
  "guidance_direction": "RAISED|MAINTAINED|LOWERED|WITHDRAWN|NONE|null",
  "guidance_details": "string|null",
  "tone_shift": "integer [-2, +2]|null",
  "tone_evidence": "string|null",
  "analyst_friction": "integer [0, 3]|null",
  "friction_topics": ["string"],
  "forward_statements": [
    {
      "statement": "string",
      "category": "revenue|margin|capex|product|hiring|other",
      "status": "NEW|REITERATED|MODIFIED",
      "specificity": "quantified|directional|vague"
    }
  ],
  "evasion_flags": [
    {
      "question_topic": "string",
      "management_response": "deflected|declined|partial|redirected"
    }
  ],
  "catalyst_mentions": [
    {
      "event": "string",
      "approximate_timing": "string",
      "management_confidence": "high|medium|low"
    }
  ],
  "extraction_confidence": "float [0, 1]"
}
```

#### 3.1.4 Scoring

```
ES_raw = (
    0.35 × guidance_score(guidance_direction)
  + 0.25 × tone_shift / 2.0
  + 0.15 × -(analyst_friction / 3.0)
  + 0.15 × forward_statement_score(forward_statements)
  + 0.10 × -(evasion_count / 3.0)
)

ES = cross_sectional_z_score(ES_raw)
```

Where:
```
guidance_score:
    RAISED = +1.0
    MAINTAINED = +0.2  (maintaining in a deteriorating environment = mildly positive)
    LOWERED = -0.8
    WITHDRAWN = -1.0
    NONE / null = 0.0

forward_statement_score:
    Sum of: NEW & quantified = +0.3, NEW & directional = +0.2,
            MODIFIED upward = +0.1, REITERATED = 0.0,
            MODIFIED downward = -0.2
    Clamped to [-1.0, +1.0]
```

---

### 3.2 Signal 2: Narrative Quality Assessment (NQ)

**Purpose:** Assess the quality, consistency, and credibility of management's narrative
across multiple documents (10-K, earnings calls, investor presentations). Detect
narrative deterioration, internal contradictions, and language shifts that precede
fundamental problems.

**Why LLM:** This requires comparing documents across time and extracting semantic
changes — "last quarter they said X, this quarter they said Y, and the shift is
significant." No quantitative factor can do this.

#### 3.2.1 Input

Two to four documents per name, in chronological order:
1. Most recent 10-K or 10-Q (MD&A section + Risk Factors)
2. Most recent earnings call transcript
3. Prior-quarter earnings call transcript (for comparison)
4. (Optional) Most recent investor presentation or 8-K

Context budget: ~200K tokens per name (well within Grok's 1M window for 2-4 documents).

PIT anchor: Each document's filing/publication date.

#### 3.2.2 Prompt Structure

```
SYSTEM:
You are a forensic accounting analyst assessing the quality and
consistency of management narrative across multiple filings. You detect
changes in risk language, hedging, forward commitment, and internal
contradictions. Output structured JSON only.

USER:
Assess narrative quality for {ticker} ({company_name}).

Documents provided (in chronological order):
1. {doc_type_1} filed {date_1}
2. {doc_type_2} filed {date_2}
[...]

Evaluate:

1. RISK_LANGUAGE_TREND: Compare risk factor language between the most
   recent filing and the prior comparable filing. Score: IMPROVING
   (fewer/lighter risks), STABLE (materially unchanged), DETERIORATING
   (new risks, stronger language, more specificity on threats),
   MATERIALLY_CHANGED (wholesale rewrite of risk section).

2. COMMITMENT_CONSISTENCY: Do forward-looking statements in the earnings
   call align with the 10-K/10-Q forward statements? Score: CONSISTENT,
   MINOR_DRIFT (slight changes in emphasis or timeline), CONTRADICTORY
   (statements that conflict), WALKBACK (explicit retreat from prior
   commitments).

3. LANGUAGE_COMPLEXITY_SHIFT: Has the readability of management's
   communication changed? Increased complexity (longer sentences, more
   jargon, more passive voice) often precedes bad news. Score:
   SIMPLER, UNCHANGED, MORE_COMPLEX, SIGNIFICANTLY_MORE_COMPLEX.

4. HEDGING_INTENSITY: Rate the density of hedging language ("may",
   "could", "subject to", "no assurance") in forward-looking statements.
   Score: LOW, MODERATE, HIGH, EXTREME.

5. SPECIFIC_CONTRADICTIONS: List any statements that directly contradict
   prior filings or prior earnings call statements. Quote both the
   original and the contradicting statement.

Output valid JSON matching the schema.

[DOCUMENTS FOLLOW]
```

#### 3.2.3 Scoring

```
NQ_raw = (
    0.30 × risk_language_score
  + 0.25 × commitment_score
  + 0.20 × -(complexity_score)
  + 0.15 × -(hedging_score)
  + 0.10 × -(contradiction_penalty)
)

NQ = cross_sectional_z_score(NQ_raw)
```

Where:
```
risk_language_score:
    IMPROVING = +0.5, STABLE = 0.0,
    DETERIORATING = -0.5, MATERIALLY_CHANGED = -1.0

commitment_score:
    CONSISTENT = +0.5, MINOR_DRIFT = 0.0,
    CONTRADICTORY = -0.7, WALKBACK = -1.0

complexity_score:
    SIMPLER = -0.3 (often positive), UNCHANGED = 0.0,
    MORE_COMPLEX = +0.5, SIGNIFICANTLY_MORE_COMPLEX = +1.0

hedging_score:
    LOW = 0.0, MODERATE = 0.3, HIGH = 0.7, EXTREME = 1.0

contradiction_penalty:
    0 contradictions = 0.0
    1 contradiction = 0.3
    2+ contradictions = 0.6 + 0.1 per additional (capped at 1.0)
```

---

### 3.3 Signal 3: Catalyst Intelligence (CI)

**Purpose:** Synthesize multi-source intelligence about upcoming catalysts — combining
news, filings, conference presentations, and industry context to assess catalyst
probability, timing precision, and potential magnitude.

**Why LLM:** Catalyst assessment requires reading across heterogeneous sources and making
inferential judgments: "the company mentioned Phase 3 enrollment completed in the earnings
call, the ClinicalTrials.gov record shows a primary completion date next quarter, and a
competitor just failed in the same indication — this changes the probability distribution."
No structured data pipeline captures this synthesis.

#### 3.3.1 Input

Per-name catalyst package assembled from:
- Most recent earnings call catalyst mentions (from ES extraction)
- ClinicalTrials.gov / event calendar entries
- Recent news (past 30 days) mentioning the company or its catalysts
- Sector/competitor context (peer company outcomes in same space)

Context budget: ~100K tokens per name.

#### 3.3.2 Prompt Structure

```
SYSTEM:
You are a catalyst research analyst. For each upcoming catalyst, assess
probability, timing confidence, and potential magnitude based on all
available evidence. You must cite specific sources for every assessment.
Do not invent information. If evidence is thin, say so explicitly and
lower your confidence.

USER:
Assess upcoming catalysts for {ticker} ({company_name}).

Known catalysts from structured data:
{catalyst_list_from_event_calendar}

Additional context:
{recent_news_summaries}
{earnings_call_catalyst_mentions}
{competitor_outcomes}

For each catalyst, provide:

1. EVENT: What is the catalyst?
2. TIMING_ASSESSMENT: Best estimate of when, and how confident (EXACT,
   APPROXIMATE, UNCERTAIN, UNKNOWN).
3. PROBABILITY_SHIFT: Based on all evidence, has the probability of a
   favorable outcome shifted vs. base rates? HIGHER_THAN_BASE,
   AT_BASE, LOWER_THAN_BASE. Cite the evidence driving the shift.
4. MAGNITUDE_ESTIMATE: If favorable, approximate magnitude of stock move
   (LARGE >15%, MODERATE 5-15%, SMALL <5%). If unfavorable, same scale.
5. ASYMMETRY: Is the risk/reward asymmetric? FAVORABLE (more upside
   than downside), SYMMETRIC, UNFAVORABLE (more downside). Explain.
6. EVIDENCE_QUALITY: How much direct evidence supports this assessment?
   STRONG (multiple independent sources), MODERATE (1-2 sources),
   WEAK (inference only), SPECULATIVE.

Output valid JSON matching the schema.
```

#### 3.3.3 Scoring

```
For each catalyst c:
    c_score = (
        probability_shift_score × 0.30
      + asymmetry_score × 0.30
      + magnitude_score × 0.20
      + timing_precision_score × 0.10
      + evidence_quality_weight × 0.10
    )

CI_raw = weighted_average(c_score for c in catalysts,
                          weights = evidence_quality_weight per catalyst)

CI = cross_sectional_z_score(CI_raw)
```

Where:
```
probability_shift_score:
    HIGHER_THAN_BASE = +0.5, AT_BASE = 0.0, LOWER_THAN_BASE = -0.5

asymmetry_score:
    FAVORABLE = +0.5, SYMMETRIC = 0.0, UNFAVORABLE = -0.5

magnitude_score:
    LARGE = 0.8, MODERATE = 0.4, SMALL = 0.1
    (signed by probability_shift direction)

timing_precision_score:
    EXACT = 1.0, APPROXIMATE = 0.6, UNCERTAIN = 0.2, UNKNOWN = 0.0

evidence_quality_weight:
    STRONG = 1.0, MODERATE = 0.7, WEAK = 0.3, SPECULATIVE = 0.0
    (SPECULATIVE evidence is logged but scores zero — it does not enter
    the composite. This is the anti-hallucination gate.)
```

**SPECULATIVE = 0 is non-negotiable.** The DEM grok_biotech_watch audit found Grok
fabricating clinical trial events. Any assessment the LLM cannot anchor to a cited source
must contribute nothing to the score. This is the single most important governance
control in the entire LLM layer.

---

### 3.4 Signal 4: Management Credibility Score (MC)

**Purpose:** Track management's historical credibility — whether their forward-looking
statements have been accurate, and whether they have a pattern of overpromising.

**Why LLM:** Credibility assessment requires comparing past promises against realized
outcomes across multiple quarters and documents. An LLM can read "in Q1 2025 management
said they expected 20% revenue growth" and compare it to "actual Q1 2026 revenue grew
8%" to build a credibility score.

#### 3.4.1 Input

Longitudinal document set per name:
- Last 4 earnings call transcripts (covering ~1 year of promises and outcomes)
- Annual guidance history (from 10-K / 8-K filings)
- Actual outcomes (from financial data feeds, not LLM-derived)

**Critical:** Actual outcomes (revenue, margins, EPS) are fed from the quantitative data
pipeline, not from LLM extraction. The LLM extracts the *promises*; the scoring function
compares them to *ground truth from structured data*. This prevents circular reasoning
(the LLM both inventing and evaluating its own facts).

#### 3.4.2 Prompt Structure

```
SYSTEM:
You are evaluating management credibility by extracting promises from
earnings call transcripts. You extract only. You do not evaluate whether
promises were kept — that comparison is done by a separate system using
actual financial data.

USER:
Extract forward-looking commitments from the following earnings call
transcripts for {ticker}. For each commitment:

1. QUARTER: Which earnings call contained the commitment
2. PROMISE: The specific forward-looking statement
3. METRIC: What was promised (revenue, margins, timeline, etc.)
4. SPECIFICITY: QUANTIFIED (a number), DIRECTIONAL (up/down/flat),
   or VAGUE (no measurable benchmark)
5. TIMEFRAME: When the promise was supposed to be realized

[TRANSCRIPTS FOLLOW]
```

#### 3.4.3 Scoring

Promise extraction is done by Grok. Promise evaluation is done deterministically:

```
For each quantified promise:
    actual = lookup_actual_outcome(metric, timeframe)  # from financial data
    if actual is None:
        skip (timeframe not yet reached)

    if metric is quantified:
        deviation = (actual - promised) / |promised|
        promise_score = clamp(-deviation × 5.0, -1.0, +1.0)
        # Beat by 5% → +0.25; missed by 10% → -0.50
    elif metric is directional:
        promise_score = +0.3 if direction matched, -0.3 if not

MC_raw = mean(promise_scores) × credibility_consistency_factor

# Consistency: penalize volatile credibility (sometimes right, sometimes very wrong)
credibility_consistency_factor = 1.0 - stdev(promise_scores) × 0.5

MC = cross_sectional_z_score(MC_raw)
```

**Why the LLM only extracts, not evaluates:** The DEM grok_biotech_watch audit found
Grok asserting `official_confirmation: True` on its own output. If the LLM both
extracts promises and judges whether they were kept, it has no incentive to be honest
about extraction accuracy — it can "correctly evaluate" hallucinated promises. Separating
extraction (LLM) from evaluation (deterministic comparison to ground truth) breaks this
failure mode.

---

## 4. LLM Governance Controls

### 4.1 Output Validation

Every LLM response must pass schema validation before being scored:

```python
def validate_llm_output(raw_output: str, schema: dict) -> dict | None:
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        log_failure("json_parse_error", raw_output)
        return None

    errors = jsonschema.validate(parsed, schema)
    if errors:
        log_failure("schema_validation", errors)
        return None

    # Anti-hallucination checks
    if parsed.get("call_date") and not is_valid_date(parsed["call_date"]):
        log_failure("invalid_date", parsed["call_date"])
        return None

    if parsed.get("ticker") != expected_ticker:
        log_failure("ticker_mismatch", parsed["ticker"])
        return None

    return parsed
```

Failed validation = that signal scores null for that name on that run. The quantitative
backbone carries the ranking. Failed validations are logged and reported in the daily
diagnostic.

### 4.2 Confidence Gating

Each LLM signal carries a confidence score. The LLM self-reports confidence, but the
system independently validates it:

```python
def effective_confidence(llm_confidence: float, validation_checks: dict) -> float:
    conf = llm_confidence

    # Penalize stale input
    if input_age_days > 60:
        conf *= 0.7
    if input_age_days > 90:
        conf *= 0.5

    # Penalize sparse evidence
    if evidence_quality == "WEAK":
        conf *= 0.5
    if evidence_quality == "SPECULATIVE":
        conf = 0.0  # hard zero

    # Penalize if LLM claims high confidence but evidence is thin
    if llm_confidence > 0.8 and evidence_quality in ("WEAK", "SPECULATIVE"):
        conf = 0.3  # override: model is overconfident

    return clamp(conf, 0.0, 1.0)
```

Signals with `effective_confidence < 0.3` are suppressed (score = 0, weight
redistributed to quantitative factors).

### 4.3 Reproducibility Tracking

LLM outputs are non-deterministic. The same prompt may produce different scores on
different runs. This is a fundamental governance challenge.

**Mitigation strategy — hash-and-log:**

```python
@dataclass
class LLMScoringRecord:
    ticker: str
    signal: str                    # ES, NQ, CI, MC
    as_of_date: str
    prompt_hash: str               # SHA256 of the full prompt (with documents)
    input_document_hashes: list    # SHA256 of each input document
    model_id: str                  # exact Grok model version
    temperature: float             # inference parameter (must be 0.0 for scoring)
    raw_output_hash: str           # SHA256 of raw LLM response
    parsed_score: float
    effective_confidence: float
    validation_passed: bool
    timestamp: str                 # ISO 8601
```

Every scoring call is logged with this record. The audit trail allows:
1. Detecting when the same inputs produce different scores (non-reproducibility)
2. Attributing score changes to input changes vs model changes
3. Replaying scoring calls when a new model version ships

**Temperature = 0.0 is mandatory for scoring calls.** The DEM Hermes fleet uses
temperature 0.2 for monitoring agents (where some creativity is acceptable). For scoring,
any non-determinism is a governance violation. Temperature 0.0 does not guarantee
identical outputs across model versions, but it eliminates sampling randomness within a
version.

### 4.4 Source Hierarchy

All LLM-derived information is classified by source authority:

| Source Level | Definition | Permitted Use |
|---|---|---|
| **VERIFIED** | Confirmed by structured data (SEC filing, exchange data, official calendar) | Full weight in scoring |
| **CORROBORATED** | LLM extraction matches 2+ independent unstructured sources | 70% weight |
| **SINGLE_SOURCE** | LLM extraction from one unstructured source | 50% weight |
| **LLM_INFERENCE** | LLM's own analytical judgment, not anchored to a specific source | 30% weight |
| **SPECULATIVE** | LLM cannot cite any source | **0% weight — hard zero** |

**Rule from DEM audit:** The LLM must NEVER mark its own output as "official" or
"confirmed." The source hierarchy is applied by the scoring pipeline, not self-reported
by the model. This is the exact failure the grok_biotech_watch audit identified
(P1 finding: "Strip/override `official_confirmation` from Grok responses").

### 4.5 Hallucination Detection

Three layers of hallucination defense:

**Layer A — Input anchoring:** Every factual claim in the LLM output must cite a
passage from the input documents. The prompt instructs the model to cite; the
validation checks for citations.

**Layer B — Cross-reference:** Key facts (dates, numbers, names) are cross-referenced
against structured data. If the LLM says "revenue grew 15% to $2.1B" and the financial
data shows revenue of $1.9B, the discrepancy is flagged and the signal confidence drops
to 0.

**Layer C — Temporal validation:** All dates mentioned in LLM output must be plausible.
The DEM audit found Grok fabricating a 2023-dated event in a 2026 context. The validator
rejects any date outside `[as_of_year - 2, as_of_year + 1]` and any future date more
than 18 months from the scoring date.

```python
def check_temporal_validity(llm_output: dict, as_of_date: date) -> list:
    violations = []
    for date_field in extract_all_dates(llm_output):
        d = parse_date(date_field)
        if d.year < as_of_date.year - 2 or d.year > as_of_date.year + 1:
            violations.append(f"implausible date: {date_field}")
        if d > as_of_date + timedelta(days=548):
            violations.append(f"future date too far: {date_field}")
    return violations
```

---

## 5. Composite Construction

### 5.1 LLM Layer Composite

```
llm_composite = w_ES × ES + w_NQ × NQ + w_CI × CI + w_MC × MC
```

| Signal | Weight | Rationale |
|---|---|---|
| Earnings Signal (ES) | 0.35 | Highest information density; most time-sensitive |
| Narrative Quality (NQ) | 0.25 | Leading indicator of fundamental problems; deep evidence |
| Catalyst Intelligence (CI) | 0.20 | Synthesis across sources; highest hallucination risk (lower weight) |
| Management Credibility (MC) | 0.20 | Longitudinal; empirically grounded (promise vs actual) |

### 5.2 Blending Quant and LLM Layers

```
final_alpha = (1 - effective_llm_weight) × quant_composite
            + effective_llm_weight × llm_composite
```

Where:
```
base_llm_weight = 0.25  # LLM layer is 25% of the composite at full confidence

effective_llm_weight = base_llm_weight × avg_effective_confidence
```

The LLM layer's contribution scales with its confidence. At full confidence (1.0), the
LLM is 25% of the composite and the quantitative backbone is 75%. At zero confidence
(all signals failed validation or are stale), the LLM weight is 0 and the system runs on
pure quant.

**Why 25% cap:** The LLM layer is newer, less tested, and non-deterministic. The
quantitative backbone (Spec 118) has clear mechanisms, deterministic outputs, and a
validation framework. Until the LLM layer accumulates its own forward shadow evidence
(60+ trading days with positive IC), it should not exceed 25% of the composite.

**Promotion path to higher weight:** If the LLM layer demonstrates positive IC for 26
consecutive weeks and passes a standalone Checklist v2 battery, the base_llm_weight can
be raised to 0.35. Maximum cap: 0.40 (the quant backbone always dominates).

### 5.3 Per-Name LLM Availability

Not every name will have LLM scores. Some names have no earnings call transcript.
Some names are too small to generate news coverage. Some names' documents exceed the
context window even at 1M tokens.

When a name has partial LLM coverage:

| Available Signals | Treatment |
|---|---|
| All 4 LLM signals | Full llm_composite as described |
| 2-3 signals | Redistribute weights pro-rata among available signals |
| 1 signal only | Suppress LLM layer (effective_llm_weight = 0) |
| 0 signals | Pure quant scoring |

A single LLM signal is too noisy to contribute meaningfully. The minimum threshold is 2
of 4 signals.

---

## 6. Inference Infrastructure

### 6.1 Scoring Cadence

| Signal | Scoring Frequency | Trigger |
|---|---|---|
| ES (Earnings Signal) | Event-driven | New transcript available (typically quarterly) |
| NQ (Narrative Quality) | Quarterly | New 10-K/10-Q filed |
| CI (Catalyst Intelligence) | Bi-weekly | Scheduled sweep of catalyst pipeline |
| MC (Management Credibility) | Quarterly | After earnings actuals are available |

Total inference calls per scoring cycle (assuming 500-name universe):
- ES: ~125/quarter (names that reported this quarter) × 1 call = ~125 calls
- NQ: ~500/quarter × 1 call = ~500 calls
- CI: ~500 every 2 weeks × 1 call = ~500 calls × 6/quarter = ~3,000 calls
- MC: ~500/quarter × 1 call = ~500 calls

**Total: ~4,125 Grok API calls per quarter**, or ~65 per business day (manageable within
rate limits and budget).

### 6.2 Context Management

Grok's 1M token context is large but not infinite. Context budget per signal:

| Signal | Typical Context Size | Max Context Size |
|---|---|---|
| ES | ~30K tokens (one transcript) | ~60K tokens (long transcript + appendix) |
| NQ | ~150K tokens (3-4 documents) | ~300K tokens (verbose 10-K + transcripts) |
| CI | ~50K tokens (catalyst package + news) | ~100K tokens (many news articles) |
| MC | ~120K tokens (4 transcripts) | ~250K tokens (detailed historical set) |

If a name's documents exceed the max context size, truncate from the oldest documents
first (keep the most recent, which is most relevant). Log the truncation in the scoring
record.

### 6.3 Cost Model

Assuming Grok pricing at approximately $5/M input tokens and $15/M output tokens (as of
late 2026; actual pricing varies by tier):

| Signal | Avg Input Tokens | Avg Output Tokens | Cost/Call | Calls/Quarter | Cost/Quarter |
|---|---|---|---|---|---|
| ES | 30,000 | 2,000 | ~$0.18 | 125 | ~$23 |
| NQ | 150,000 | 3,000 | ~$0.80 | 500 | ~$400 |
| CI | 50,000 | 3,000 | ~$0.30 | 3,000 | ~$900 |
| MC | 120,000 | 2,000 | ~$0.63 | 500 | ~$315 |
| **Total** | | | | **4,125** | **~$1,638** |

~$1,600/quarter is economically viable for an institutional screener. For a retail or
smaller operation, CI frequency could be reduced to monthly (~$450 savings), bringing the
total under $1,200/quarter.

### 6.4 Failure Handling

| Failure Mode | Detection | Response |
|---|---|---|
| API timeout (>30s) | HTTP timeout | Retry once with backoff; on second failure, score = null |
| Rate limit (429) | HTTP status | Queue and retry with exponential backoff (max 5 min) |
| Invalid JSON response | Parse failure | Score = null; log for prompt debugging |
| Schema validation failure | Validation check | Score = null; log with raw output for analysis |
| Model version change | `model_id` differs from prior run | Log version change; compare outputs to prior version |
| API key expired / missing | 401/403 | **Hard stop on LLM layer**; fall back to pure quant |
| Budget exhausted | Cost tracking | Reduce CI frequency to monthly; if still over, suspend LLM layer |

---

## 7. Validation Framework

### 7.1 LLM Signal Validation (Unique to This Architecture)

In addition to the standard Checklist v2 tests (Spec 118 §7), LLM signals require:

| Test | Method | Requirement |
|---|---|---|
| **Extraction accuracy** | Compare ES guidance_direction against actual guidance from consensus data | >85% agreement |
| **Hallucination rate** | Count factual claims that fail cross-reference | <5% of claims per signal |
| **Reproducibility** | Score same inputs twice (temperature=0); compare | >95% identical scores |
| **Prompt sensitivity** | Paraphrase prompts; compare scores | Score correlation >0.85 across paraphrases |
| **Model version stability** | Score same inputs across Grok versions | Score correlation >0.80 |
| **Incremental IC** | IC of composite with LLM vs without LLM | Positive incremental IC, t > 2.0 |
| **Degradation test** | Artificially degrade one signal; measure composite impact | No single signal's failure drops Sharpe >15% |

### 7.2 Forward Shadow Arms

Extend the Spec 118 shadow with LLM-specific arms:

| Arm | Description |
|---|---|
| Arm 1 | Pure quant (Spec 118, no LLM) |
| Arm 2 | Quant + LLM (full grokbot, production config) |
| Arm 3 | Quant + ES only (just earnings signal) |
| Arm 4 | Quant + NQ only (just narrative quality) |
| Arm 5 | LLM only (no quant backbone) — control for whether LLM signals work standalone |
| Arm 6 | EES gates only (no alpha model) — baseline |

The key comparison: Arm 2 vs Arm 1. If the grokbot does not beat the pure quant model
over 60+ trading days, the LLM layer has no value and should be disabled.

---

## 8. Operational Considerations

### 8.1 Model Version Management

Grok model versions change without notice. The scoring pipeline must:

1. Log the exact model ID on every call (`grok-2`, `grok-2.5`, etc.)
2. Detect when the model ID changes between scoring runs
3. Trigger a re-scoring of the most recent 10 names with the new model
4. If score correlation between old and new model is <0.80, alert the operator
5. If correlation is <0.60, suspend the LLM layer pending review

### 8.2 Prompt Versioning

Prompts are code. They are versioned, hashed, and governed:

```python
PROMPT_REGISTRY = {
    "ES_v1.0": {
        "hash": "sha256:abc123...",
        "effective_date": "2026-11-15",
        "status": "active",
    },
    "ES_v0.9": {
        "hash": "sha256:def456...",
        "effective_date": "2026-10-01",
        "status": "retired",
        "retired_reason": "v1.0 improved extraction accuracy from 82% to 89%",
    },
}
```

Prompt changes are governed at Tier 3 (same as production scoring changes). A prompt
change triggers:
1. Extraction accuracy test on a holdout set of 50 transcripts
2. Score comparison: new prompt vs old prompt on the same inputs
3. If scores diverge by >10% on average, it is a model change and resets the OOS clock

### 8.3 Data Privacy and Compliance

The grokbot sends company filings and transcripts to an external API (xAI). Compliance
considerations:

1. **Only public information** is sent — SEC filings, published transcripts, public news.
   No proprietary research, no internal models, no portfolio positions.
2. **No PII** is included in prompts. The system sends ticker symbols and document text,
   not client data, account information, or trading records.
3. **API logs are retained** on the grokbot's side for audit purposes. xAI's data
   retention policy must be reviewed and documented before deployment.
4. **The system must function without the API.** If compliance requires cutting off the
   LLM layer (e.g., during a regulatory examination), the quant backbone continues.

---

## 9. Risks and Mitigations

### 9.1 LLM Risks (Unique to This Architecture)

| Risk | Severity | Mitigation |
|---|---|---|
| **Hallucination enters scoring** | HIGH | Three-layer defense (§4.5); SPECULATIVE = hard zero; cross-reference against structured data |
| **Model version regression** | HIGH | Version tracking; re-scoring gate; suspension at correlation <0.60 |
| **Prompt injection via documents** | MEDIUM | Input documents are from regulated sources (SEC, exchanges); no user-supplied text enters prompts |
| **Overconfidence in LLM signal** | HIGH | 25% weight cap; confidence gating; effective_llm_weight scales with confidence |
| **Non-reproducibility** | MEDIUM | Temperature = 0.0; hash-and-log; reproducibility testing |
| **API dependency** | MEDIUM | Fallback to pure quant; no scoring call is blocking; async with timeout |
| **Cost overrun** | LOW | Budget tracking; CI frequency reduction; hard suspension at budget cap |
| **Source hierarchy violation** | HIGH | Pipeline applies source levels, not the LLM; `official_confirmation` stripped from all LLM output (DEM audit P1) |

### 9.2 Architectural Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **LLM layer becomes load-bearing** | HIGH | Fallback guarantee: system must rank without LLM. Tested monthly. |
| **Overfitting to Grok's idiosyncrasies** | MEDIUM | Schema-based scoring (not raw text); model-agnostic design; periodic testing on alternative models |
| **Compound complexity** | MEDIUM | Each layer (quant, LLM, EES) validated independently before combining |
| **Governance gap for non-deterministic outputs** | HIGH | Every LLM output is a logged, hashed, schema-validated record; the system is auditable even if individual outputs are not perfectly reproducible |

---

## 10. Comparison: Pure Quant (Spec 118) vs Grokbot (This Spec)

| Dimension | Pure Quant (Spec 118) | Grokbot (This Spec) |
|---|---|---|
| Signal sources | Structured data only | Structured + unstructured (filings, transcripts, news) |
| Determinism | Byte-identical on same inputs | Near-deterministic (temperature=0, same model version) |
| Reproducibility | Perfect | ~95% (logged and tracked) |
| Auditability | Full (every input→output traceable) | Full (hash-and-log on every LLM call) |
| Failure mode | Data pipeline failure | Data pipeline failure OR API failure OR hallucination |
| Fallback | None needed (self-contained) | Pure quant backbone (tested monthly) |
| Information advantage | None vs well-resourced managers | Processes unstructured information at scale |
| Governance burden | Standard (Checklist v2) | Higher (Checklist v2 + LLM-specific tests) |
| Cost (infrastructure) | ~$0 (data feeds only) | ~$1,600/quarter (API calls) |
| Edge persistence | Depends on factor persistence | Depends on LLM capability vs market adoption of same |
| Coverage | 100% of scored universe | ~80% (20% of names lack sufficient documents) |

### 10.1 When to Choose Which

**Choose Pure Quant (Spec 118) when:**
- Governance requirements demand perfect reproducibility
- Infrastructure budget is minimal
- The universe is large-cap / heavily covered (less information asymmetry to exploit)
- Operational simplicity is valued over marginal alpha

**Choose Grokbot (this spec) when:**
- The universe includes mid/small-cap with thin coverage (more unstructured information to exploit)
- The operator can tolerate ~95% reproducibility with full audit trails
- The infrastructure supports API integration and cost management
- The marginal alpha from processing earnings calls, filings, and news justifies the
  governance overhead

**Do not choose Grokbot as a replacement for the quant backbone.** The grokbot adds a
layer; it does not substitute one. The quant backbone is the structural integrity of the
system. The LLM layer is the sensing capability. A building needs both a frame and
windows.

---

## Appendix A: Full Pipeline Diagram

```
    ┌──────────────────────────────────────────────────────────┐
    │                 LAYER 1: QUANTITATIVE BACKBONE            │
    │                 (Spec 118 — deterministic)                │
    │                                                          │
    │  IA (0.30) + EQ (0.20) + PM (0.20) + CD (0.15)          │
    │  + FR (0.15) = quant_composite                           │
    └───────────────────────────┬──────────────────────────────┘
                                │
    ┌──────────────────────────────────────────────────────────┐
    │                 LAYER 2: LLM INTELLIGENCE                │
    │                 (This spec — Grok-powered)                │
    │                                                          │
    │  ES (0.35) + NQ (0.25) + CI (0.20) + MC (0.20)          │
    │  = llm_composite                                         │
    │                                                          │
    │  Schema validation → Confidence gating →                 │
    │  Source hierarchy → Hallucination detection               │
    └───────────────────────────┬──────────────────────────────┘
                                │
    ┌──────────────────────────────────────────────────────────┐
    │                 BLENDING                                 │
    │                                                          │
    │  effective_llm_weight = 0.25 × avg_confidence            │
    │  alpha = (1 - eff_w) × quant + eff_w × llm              │
    │                                                          │
    │  Sort by alpha descending                                │
    └───────────────────────────┬──────────────────────────────┘
                                │
    ┌──────────────────────────────────────────────────────────┐
    │                 LAYER 3: EES GATES (Spec 117)            │
    │                                                          │
    │  Trap gate (20%) + Structure gate (10%)                  │
    │  = ees_eligible                                          │
    └───────────────────────────┬──────────────────────────────┘
                                │
    ┌──────────────────────────────────────────────────────────┐
    │                 CONSTRUCTION                             │
    │                                                          │
    │  Top-N eligible by alpha → Equal-weight                  │
    │  Sector cap 25% → Turnover buffer 5 ranks               │
    │  Monthly rebalance                                       │
    └──────────────────────────────────────────────────────────┘
```

## Appendix B: Governance Checklist for LLM Layer Deployment

Before the LLM layer enters the composite (even at 25% weight):

- [ ] All four prompts pass extraction accuracy test (>85% on holdout set)
- [ ] Hallucination rate <5% across all signals on holdout set
- [ ] Reproducibility >95% on temperature=0 same-model retest
- [ ] Prompt sensitivity test: score correlation >0.85 across 3 paraphrases
- [ ] Source hierarchy enforced: no LLM output carries `official` or `confirmed` tags
- [ ] Temporal validation active: dates outside ±2 years rejected
- [ ] Schema validation active: all outputs parsed and validated before scoring
- [ ] Confidence gating active: SPECULATIVE = hard zero
- [ ] Fallback tested: system produces valid rankings with LLM layer disabled
- [ ] Forward shadow running (all 6 arms) for 60+ trading days
- [ ] Incremental IC (Arm 2 vs Arm 1) positive, t > 2.0
- [ ] Cost tracking active; budget cap set
- [ ] Operator sign-off on governance controls

## Appendix C: DEM Audit Lessons Applied to This Design

| DEM grok_biotech_watch Finding | This Spec's Response |
|---|---|
| Grok fabricated a 2023-dated clinical trial event | Temporal validation (§4.5, Layer C): reject dates outside ±2 years |
| Grok output carried `official_confirmation: True` | Source hierarchy (§4.4): pipeline assigns source level, never the LLM |
| Producer and consumer used different paths | Schema-first design: every output validated against a contract before scoring |
| No schema validation on producer output | Mandatory schema validation (§4.1): parse failure = null score |
| XAI_API_KEY missing with no alerting | Failure handling (§6.4): 401/403 = hard stop on LLM layer + operator alert |
| Feed was broken-and-dormant for 26 days | Daily health check: if no LLM scores produced in 3 business days, alert |
| financial_score contaminated by market-cap look-ahead | MC signal (§3.4): LLM extracts promises, deterministic code evaluates against ground truth — no self-evaluation |
