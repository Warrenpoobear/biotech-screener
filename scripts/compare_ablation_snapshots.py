"""Compare two preserved snapshot folders (OFF vs cache_only ablation).

Reports:
  - Source-mix deltas (from catalyst_source_mix.json, if present)
  - Top-60 / top-100 overlap + entries/exits
  - Biggest rank movers
  - Catalyst UX conversions: no_upcoming/missing -> specific_days/blended_window (and reverse)
  - Provenance shifts: catalyst_source changes (and event_type changes)
  - Nearest catalyst improvements: catalyst_days missing->present (and reverse), strength shifts

Usage:
  python scripts/compare_ablation_snapshots.py \
    data/snapshots/2026-02-11__sec_off \
    data/snapshots/2026-02-11__sec_cache_only

  # Machine-readable summary:
  python scripts/compare_ablation_snapshots.py \
    data/snapshots/2026-02-11__sec_off \
    data/snapshots/2026-02-11__sec_cache_only \
    --json-out ablation_summary.json
"""
import argparse, csv, json, sys
from pathlib import Path
from collections import Counter, defaultdict

parser = argparse.ArgumentParser(description="Compare two ablation snapshots")
parser.add_argument("dir_a", type=Path, help="Baseline snapshot (e.g. __sec_off)")
parser.add_argument("dir_b", type=Path, help="Treatment snapshot (e.g. __sec_cache_only)")
parser.add_argument("--json-out", type=Path, default=None,
                    help="Write machine-readable summary JSON to this path")
args = parser.parse_args()

A_dir, B_dir = args.dir_a, args.dir_b

GOOD_MODES = {"specific_days", "blended_window"}
BAD_MODES  = {"no_upcoming", "missing", ""}

def load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None

def norm(s):
    return (s or "").strip()

def norm_lower(s):
    return norm(s).lower()

def parse_int(s):
    s = norm(s).lower()
    if not s or s in ("", "nan", "missing", "none", "n/a"):
        return None
    try:
        return int(float(s))
    except (ValueError, OverflowError):
        return None

def read_rankings(p: Path):
    with p.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise RuntimeError(f"{p} has no header")

        # case-insensitive column mapping
        cols = {c.lower(): c for c in r.fieldnames}

        ticker_col = cols.get("ticker") or cols.get("symbol") or r.fieldnames[0]
        want = {
            "catalyst_mode": cols.get("catalyst_mode"),
            "de_catalyst_mode": cols.get("de_catalyst_mode"),
            "catalyst_source": cols.get("catalyst_source"),
            "catalyst_event_type": cols.get("catalyst_event_type"),
            "catalyst_days": cols.get("catalyst_days"),
            "catalyst_strength": cols.get("catalyst_strength"),
        }

        rows = {}
        order = []
        for row in r:
            t = norm(row.get(ticker_col))
            if not t or t in rows:
                continue
            rows[t] = row
            order.append(t)

        rank = {t: i+1 for i, t in enumerate(order)}
        return order, rank, rows, ticker_col, want

def top_overlap(listA, listB, N):
    setA, setB = set(listA[:N]), set(listB[:N])
    inter = setA & setB
    return len(inter), sorted(setB - setA), sorted(setA - setB)

def diff_counts(name, a, b, top=25):
    a = a or {}
    b = b or {}
    keys = set(a) | set(b)
    deltas = [(k, int(b.get(k,0)) - int(a.get(k,0)), int(a.get(k,0)), int(b.get(k,0))) for k in keys]
    deltas.sort(key=lambda x: (abs(x[1]), x[0]), reverse=True)
    print(f"\n{name} deltas (B - A):")
    any_change = False
    for k, d, av, bv in deltas[:top]:
        if d != 0:
            any_change = True
            print(f"  {k}: {d:+d}  (A={av}, B={bv})")
    if not any_change:
        print("  (no changes)")

# ---- Load source mix ----
mixA = load_json(A_dir / "catalyst_source_mix.json")
mixB = load_json(B_dir / "catalyst_source_mix.json")

print(f"A: {A_dir}\nB: {B_dir}")

if mixA and mixB:
    print("\n== Catalyst source mix ==")
    print(f"total_events: {mixA.get('total_events')} -> {mixB.get('total_events')}")
    print(f"unique_tickers_with_events: {mixA.get('unique_tickers_with_events')} -> {mixB.get('unique_tickers_with_events')}")
    diff_counts("by_source", mixA.get("by_source"), mixB.get("by_source"))
    diff_counts("by_confidence", mixA.get("by_confidence"), mixB.get("by_confidence"))
    diff_counts("by_date_precision", mixA.get("by_date_precision"), mixB.get("by_date_precision"))
else:
    print("\n== Catalyst source mix ==")
    print("NOTE: catalyst_source_mix.json missing in one or both snapshots (skipping mix deltas).")

# ---- Load rankings ----
listA, rankA, rowsA, tick_colA, wantA = read_rankings(A_dir / "rankings.csv")
listB, rankB, rowsB, tick_colB, wantB = read_rankings(B_dir / "rankings.csv")

print("\n== Rankings columns detected ==")
print(f"ticker column: A={tick_colA}, B={tick_colB}")
print("A cols:", {k:v for k,v in wantA.items() if v})
print("B cols:", {k:v for k,v in wantB.items() if v})

# ---- Overlap & churn ----
print("\n== Top overlap ==")
for N in (60, 100):
    ov, entered, exited = top_overlap(listA, listB, N)
    print(f"Top-{N} overlap: {ov}/{N}")
    if entered:
        print(f"  Entered B (first 20): {', '.join(entered[:20])}")
    if exited:
        print(f"  Exited  B (first 20): {', '.join(exited[:20])}")

# Biggest movers within top-100 intersection
top100 = set(listA[:100]) & set(listB[:100])
movers = []
for t in top100:
    movers.append((abs(rankB[t] - rankA[t]), rankA[t], rankB[t], t))
movers.sort(reverse=True)

print("\n== Biggest rank movers within top-100 intersection ==")
shown = 0
for d, ra, rb, t in movers:
    if d == 0:
        break
    print(f"  {t}: {ra} -> {rb}  ({rb-ra:+d})")
    shown += 1
    if shown >= 20:
        break
if shown == 0:
    print("  (no movers)")

# ---- Catalyst UX delta ----
print("\n== Catalyst UX delta (from rankings.csv) ==")

col_modeA = wantA.get("catalyst_mode")
col_modeB = wantB.get("catalyst_mode")
col_srcA  = wantA.get("catalyst_source")
col_srcB  = wantB.get("catalyst_source")
col_evtA  = wantA.get("catalyst_event_type")
col_evtB  = wantB.get("catalyst_event_type")
col_daysA = wantA.get("catalyst_days")
col_daysB = wantB.get("catalyst_days")
col_strA  = wantA.get("catalyst_strength")
col_strB  = wantB.get("catalyst_strength")

missing_cols = [k for k, v in {**wantA, **wantB}.items()
                if not wantA.get(k) or not wantB.get(k)]
if missing_cols:
    print(f"\nNOTE: columns missing in one/both snapshots (sections skipped): {missing_cols}")

common = sorted(set(rowsA) & set(rowsB))

mode_trans = Counter()
good_from_bad = []
bad_from_good = []
src_changed = []
evt_changed = []
days_gained = []
days_lost = []
strength_changed = []

for t in common:
    rA, rB = rowsA[t], rowsB[t]

    if col_modeA and col_modeB:
        mA = norm_lower(rA.get(col_modeA))
        mB = norm_lower(rB.get(col_modeB))
        mode_trans[(mA, mB)] += 1
        if (mA in BAD_MODES) and (mB in GOOD_MODES):
            good_from_bad.append(t)
        if (mA in GOOD_MODES) and (mB in BAD_MODES):
            bad_from_good.append(t)

    if col_srcA and col_srcB:
        sA = norm(rA.get(col_srcA))
        sB = norm(rB.get(col_srcB))
        if sA and sB and sA != sB:
            src_changed.append((t, sA, sB))

    if col_evtA and col_evtB:
        eA = norm(rA.get(col_evtA))
        eB = norm(rB.get(col_evtB))
        if eA and eB and eA != eB:
            evt_changed.append((t, eA, eB))

    if col_daysA and col_daysB:
        dA = parse_int(rA.get(col_daysA))
        dB = parse_int(rB.get(col_daysB))
        if dA is None and dB is not None:
            days_gained.append((t, dB))
        if dA is not None and dB is None:
            days_lost.append((t, dA))

    if col_strA and col_strB:
        stA = norm(rA.get(col_strA))
        stB = norm(rB.get(col_strB))
        if stA and stB and stA != stB:
            strength_changed.append((t, stA, stB))

print(f"Tickers compared (intersection): {len(common)}")

if col_modeA and col_modeB:
    print("\nConversions (bad -> good): "
          f"{len(good_from_bad)}  (no_upcoming/missing -> specific_days/blended_window)")
    if good_from_bad:
        print("  First 25:", ", ".join(good_from_bad[:25]))

    print("\nRegressions (good -> bad): "
          f"{len(bad_from_good)}  (specific_days/blended_window -> no_upcoming/missing)")
    if bad_from_good:
        print("  First 25:", ", ".join(bad_from_good[:25]))

    print("\nMost common catalyst_mode transitions (top 12):")
    for (a,b), c in mode_trans.most_common(12):
        print(f"  {a or '(blank)'} -> {b or '(blank)'} : {c}")
else:
    print("\n(catalyst_mode column missing — skipping conversions)")

if col_daysA and col_daysB:
    print(f"\nNearest catalyst days gained (missing -> present): {len(days_gained)}")
    if days_gained:
        days_gained.sort(key=lambda x: x[1])
        print("  Closest 15:", ", ".join([f"{t}({d}d)" for t,d in days_gained[:15]]))

    print(f"\nNearest catalyst days lost (present -> missing): {len(days_lost)}")
    if days_lost:
        days_lost.sort(key=lambda x: x[1])
        print("  First 15:", ", ".join([f"{t}({d}d)" for t,d in days_lost[:15]]))
else:
    print("\n(catalyst_days column missing — skipping days gained/lost)")

if col_srcA and col_srcB:
    print(f"\nCatalyst source changed (catalyst_source): {len(src_changed)}")
    if src_changed:
        print("  First 20:", ", ".join([f"{t}:{a}->{b}" for t,a,b in src_changed[:20]]))
else:
    print("\n(catalyst_source column missing — skipping provenance shifts)")

if col_evtA and col_evtB:
    print(f"\nCatalyst event type changed (catalyst_event_type): {len(evt_changed)}")
    if evt_changed:
        print("  First 20:", ", ".join([f"{t}:{a}->{b}" for t,a,b in evt_changed[:20]]))
else:
    print("\n(catalyst_event_type column missing — skipping event type shifts)")

if col_strA and col_strB:
    print(f"\nCatalyst strength changed (catalyst_strength): {len(strength_changed)}")
    if strength_changed:
        print("  First 20:", ", ".join([f"{t}:{a}->{b}" for t,a,b in strength_changed[:20]]))
else:
    print("\n(catalyst_strength column missing — skipping strength shifts)")

# ---- Machine-readable JSON summary ----
if args.json_out:
    ov60, _, _ = top_overlap(listA, listB, 60)
    ov100, _, _ = top_overlap(listA, listB, 100)
    summary = {
        "dir_a": str(A_dir),
        "dir_b": str(B_dir),
        "tickers_compared": len(common),
        "top60_overlap": ov60,
        "top100_overlap": ov100,
        "bad_to_good_count": len(good_from_bad),
        "good_to_bad_count": len(bad_from_good),
        "source_changed_count": len(src_changed),
        "event_type_changed_count": len(evt_changed),
        "days_gained_count": len(days_gained),
        "days_lost_count": len(days_lost),
        "strength_changed_count": len(strength_changed),
        "bad_to_good_tickers": good_from_bad,
        "good_to_bad_tickers": bad_from_good,
    }
    args.json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nJSON summary written to {args.json_out}")
