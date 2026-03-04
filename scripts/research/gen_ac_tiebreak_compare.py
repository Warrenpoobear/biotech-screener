"""Generate compare.md for alpha_cohort tiebreak sweep."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

runs = [
    ("w=0.000 (baseline)", ROOT / "output/ac_tiebreak_sweep/w0p00/ac_tb_w0p00/eval/summary.json"),
    ("w=0.005",            ROOT / "output/ac_tiebreak_sweep/w0p005/ac_tb_w0p005/eval/summary.json"),
    ("w=0.010",            ROOT / "output/ac_tiebreak_sweep/w0p01/ac_tb_w0p01/eval/summary.json"),
    ("w=0.020",            ROOT / "output/ac_tiebreak_sweep/w0p02/ac_tb_w0p02/eval/summary.json"),
]

results = []
for label, path in runs:
    with open(path) as f:
        s = json.load(f)
    bh = s["by_horizon"]
    r84, r126 = bh["84"], bh["126"]
    results.append({
        "label": label, "n": s["n_evaluated"],
        "84_ic": r84["mean_ic"], "84_net": r84["mean_net_return"],
        "84_gross": r84["mean_gross_return"], "84_excess": r84.get("mean_excess_return"),
        "84_hedged": r84.get("mean_hedged_return"), "84_turn": r84["mean_turnover"],
        "126_ic": r126["mean_ic"], "126_net": r126["mean_net_return"],
        "126_gross": r126["mean_gross_return"], "126_excess": r126.get("mean_excess_return"),
        "126_hedged": r126.get("mean_hedged_return"), "126_turn": r126["mean_turnover"],
    })

base = results[0]

def pp(v):
    return f"{v*100:.3f}%" if v is not None else "—"

def delta(a, b):
    return f"{(a-b)*100:+.3f}pp" if a is not None and b is not None else "—"

lines = [
    "# Alpha Cohort Tiebreak Sweep — OOS 2020–2024",
    "",
    f"**Window**: 2020-03-31 – 2024-12-31 | **n_dates**: {base['n']} | **Snapshot**: snapshots_reranked_baseline_oos",
    "**Setup**: --rerank (re-sorts via ruleset) | buffer=30 | top-k=20 | cost=30bps",
    "",
    "## Results",
    "",
    "### 126d (primary)",
    "",
    "| Weight | Gross | Net | Excess | Hedged | Turnover | IC | Δ Net vs baseline |",
    "|--------|-------|-----|--------|--------|----------|----|-------------------|",
]
for r in results:
    d = delta(r["126_net"], base["126_net"]) if r is not base else "—"
    lines.append(f"| {r['label']} | {pp(r['126_gross'])} | {pp(r['126_net'])} | {pp(r['126_excess'])} | {pp(r['126_hedged'])} | {pp(r['126_turn'])} | {r['126_ic']:.4f} | {d} |")

lines += ["", "### 84d (guardrail)", "",
    "| Weight | Gross | Net | Excess | Hedged | Turnover | IC | Δ Net vs baseline |",
    "|--------|-------|-----|--------|--------|----------|----|-------------------|",
]
for r in results:
    d = delta(r["84_net"], base["84_net"]) if r is not base else "—"
    lines.append(f"| {r['label']} | {pp(r['84_gross'])} | {pp(r['84_net'])} | {pp(r['84_excess'])} | {pp(r['84_hedged'])} | {pp(r['84_turn'])} | {r['84_ic']:.4f} | {d} |")

lines += ["", "## Verdict", "",
    "**Promotion threshold**: 126d net ≥ +0.20pp AND 84d net ≥ −0.05pp.",
    "",
    "| Weight | 126d Δ Net | 84d Δ Net | Clears 126d bar? | Clears 84d bar? | Decision |",
    "|--------|-----------|-----------|-----------------|-----------------|----------|",
]
for r in results[1:]:
    d126 = (r["126_net"] - base["126_net"]) * 100
    d84  = (r["84_net"]  - base["84_net"])  * 100
    ok126 = d126 >= 0.20
    ok84  = d84  >= -0.05
    dec = "PASS" if ok126 and ok84 else "FAIL"
    lines.append(f"| {r['label']} | {d126:+.3f}pp | {d84:+.3f}pp | {'YES' if ok126 else 'NO'} | {'YES' if ok84 else 'NO'} | **{dec}** |")

lines += [
    "",
    "**Outcome**: No weight clears the promotion bar. Signal is flat (w=0.005) to mildly negative "
    "(w=0.01–0.02) at all tested weights. Monotonically worsening with weight. "
    "**ARCHIVE — do not promote alpha_cohort_tiebreak_weight.**",
]

out = "\n".join(lines)
out_path = ROOT / "output/ac_tiebreak_sweep/compare.md"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(out)
print(out)
print(f"\n--- written to {out_path} ---")
