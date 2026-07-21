#!/usr/bin/env python3
"""
Regression test for --score-field support in measure_final_score_ic_spec100.py.

Guarantees:
  1. The function default for score_field is "final_score".
  2. Calling with score_field omitted == calling with score_field="final_score"
     (byte-identical result dict) — i.e. the original DEM-gate behavior is unchanged.
  3. score_field actually switches which column is correlated:
     - default path's IC matches a manual Spearman on the `final_score` column
     - a non-default path (catalyst_score) matches a manual Spearman on that column
     - the two generally differ (parameterization is real, not a no-op)

Read-only: builds synthetic in-memory snapshots; touches no production data,
no real snapshots, no ranker/selector/model code. Run:

    python3 tools/test_measure_signal_ic_score_field.py
"""

import importlib.util
import math
from pathlib import Path

# Load the tool module by path (filename is not a clean import name)
_TOOL = Path(__file__).resolve().parent / "measure_final_score_ic_spec100.py"
_spec = importlib.util.spec_from_file_location("spec100_ic_tool", _TOOL)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _make_snapshots():
    """
    Two snapshots 20 days apart, 12 cohort tickers (actionable_rank 1..12).
    final_score and catalyst_score are deliberately given DIFFERENT orderings so
    that switching score_field must change the IC. close_price moves so forward
    returns are well-defined.
    """
    base, fwd = "2026-03-01", "2026-03-21"  # 20 calendar days
    n = 12
    base_rows, fwd_rows = [], []
    for i in range(n):
        t = f"TST{i:02d}"
        # final_score decreasing with rank; catalyst_score in a different order
        final_score = 0.90 - i * 0.05
        catalyst_score = 50.0 + ((i * 7) % n)  # scrambled vs rank
        base_price = 100.0 + i
        # forward return loosely tracks final_score (so final_score IC is positive)
        fwd_price = base_price * (1.0 + (0.10 - i * 0.012))
        base_rows.append(
            {
                "ticker": t,
                "actionable_rank": str(i + 1),
                "final_score": f"{final_score:.6f}",
                "catalyst_score": f"{catalyst_score:.6f}",
                "composite_score": f"{0.05 - i*0.001:.6f}",
                "close_price": f"{base_price:.4f}",
            }
        )
        fwd_rows.append(
            {
                "ticker": t,
                "actionable_rank": str(i + 1),
                "final_score": f"{final_score:.6f}",
                "catalyst_score": f"{catalyst_score:.6f}",
                "composite_score": f"{0.05 - i*0.001:.6f}",
                "close_price": f"{fwd_price:.4f}",
            }
        )
    snaps = {
        base: {"date": base, "rows": base_rows},
        fwd: {"date": fwd, "rows": fwd_rows},
    }
    return base, fwd, snaps


def _manual_spearman_ic(snaps, base, fwd, score_field):
    """Independent reference: Spearman IC of score_field vs forward return on cohort."""
    base_rows = snaps[base]["rows"]
    scores, rets = [], []
    for r in base_rows:
        rank = float(r["actionable_rank"])
        if rank > 60:
            continue
        s = float(r[score_field])
        bp = float(r["close_price"])
        # find forward price
        fp = None
        for fr in snaps[fwd]["rows"]:
            if fr["ticker"] == r["ticker"]:
                fp = float(fr["close_price"])
                break
        if fp is None or bp <= 0:
            continue
        scores.append(s)
        rets.append((fp - bp) / bp)
    return mod._spearman_ic(scores, rets)[0]


def run():
    base, fwd, snaps = _make_snapshots()
    failures = []

    # --- Check 1: function default is "final_score" ---
    import inspect

    sig = inspect.signature(mod.measure_final_score_ic)
    default = sig.parameters["score_field"].default
    if default != "final_score":
        failures.append(f"default score_field is {default!r}, expected 'final_score'")
    else:
        print("PASS  default score_field == 'final_score'")

    # --- Check 2: omitted == explicit final_score (byte-identical result) ---
    r_omit = mod.measure_final_score_ic(snaps[base], snaps, 20)
    r_expl = mod.measure_final_score_ic(snaps[base], snaps, 20, score_field="final_score")
    if r_omit != r_expl:
        failures.append("omitted score_field != explicit 'final_score' result")
    else:
        print("PASS  omitted == explicit final_score (identical result dict)")

    # --- Check 3a: default path matches manual Spearman on final_score ---
    manual_final = _manual_spearman_ic(snaps, base, fwd, "final_score")
    tool_final = r_omit["final_score_ic"]
    if not math.isclose(manual_final, tool_final, abs_tol=1e-9):
        failures.append(f"final_score IC mismatch: tool={tool_final} manual={manual_final}")
    else:
        print(f"PASS  final_score IC matches manual ({tool_final:+.6f})")

    # --- Check 3b: catalyst path matches manual Spearman on catalyst_score ---
    r_cat = mod.measure_final_score_ic(snaps[base], snaps, 20, score_field="catalyst_score")
    manual_cat = _manual_spearman_ic(snaps, base, fwd, "catalyst_score")
    tool_cat = r_cat["final_score_ic"]
    if not math.isclose(manual_cat, tool_cat, abs_tol=1e-9):
        failures.append(f"catalyst_score IC mismatch: tool={tool_cat} manual={manual_cat}")
    else:
        print(f"PASS  catalyst_score IC matches manual ({tool_cat:+.6f})")

    # --- Check 3c: parameterization is real (fields differ) ---
    if math.isclose(tool_final, tool_cat, abs_tol=1e-9):
        failures.append("final_score and catalyst_score IC identical — score_field is a no-op")
    else:
        print(f"PASS  score_field switches column (final={tool_final:+.4f} vs catalyst={tool_cat:+.4f})")

    # --- Check 4: result records which field was used ---
    if r_cat.get("score_field") != "catalyst_score":
        failures.append(f"result score_field tag wrong: {r_cat.get('score_field')!r}")
    else:
        print("PASS  result dict records score_field used")

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL REGRESSION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
