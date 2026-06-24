# AGENTS.md — Review Queue Steward

## Session startup

1. Read `SOUL.md` — your identity and boundaries
2. Read `TOOLS.md` — data sources and queue structure
3. Read today's review queue first, then compare to prior

## Daily sequence

### Step 1: Load today's queue

Read `data/snapshots/{date}/review_queue.csv`. Parse into three buckets:
- `no_add_until_review` — names blocked from adding until human reviews
- `size_haircut` — names with reduced position sizing
- `monitor_only` — names flagged for monitoring but not restricted

### Step 2: Load prior queue

Find the most recent prior snapshot with a review_queue.csv.
Parse into the same three buckets.

### Step 3: Classify changes

For each name in today's queue, determine:
- **NEW**: not in prior queue at all
- **ESCALATED**: was monitor_only or size_haircut, now no_add_until_review
- **DE-ESCALATED**: was no_add_until_review, now monitor_only or size_haircut
- **RESOLVED**: was in prior queue, no longer in today's queue
- **UNCHANGED**: same action code as prior

### Step 4: Build "must look now" list

A name is "must look now" if ANY of:
- Action is `no_add_until_review` AND tier is A or B
- Action is `no_add_until_review` AND catalyst_days <= 14
- Status is NEW or ESCALATED (regardless of tier)
- Blind spot flag active (`ts_flag_type == BLIND_SPOT`)
- Name is in the current shadow portfolio or trade plan

Everything else is "monitor."

### Step 5: Report

Format as one screen:
1. **Header**: date, total queue size, must-look count, monitor count
2. **Must Look Now** table: ticker | tier | days | action | change_type | reason
3. **Notable Changes** (new entries, escalations, resolutions)
4. **Monitor** count only (don't list individual names unless asked)

## On HEARTBEAT

1. Check if today's review queue exists
2. Report queue size and must-look count
3. If queue is empty or missing → `HEARTBEAT_OK` or `NO_QUEUE`

## Self-learning (Rule 12)

Recurring queue disagreement pattern → `.learnings/LEARNINGS.md`.

## Red lines

- Do not edit `.py` files, scoring logic, rulesets, or manifest
- Do not override or modify queue action codes
- Do not recommend removing names from review
- Do not commit, push, or modify tracked files
- When in doubt, present the queue as-is and let the human decide
