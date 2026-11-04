"""Real-dataset benchmark runner for LeakLens.

Builds real-data cases (Wine, Breast Cancer, Bank Marketing) with documented
leak injections and evaluates:
  1. Rule-based baseline (signals only, zero experiments)
  2. LeakLens adaptive investigation system
  3. Run-all-experiments upper bound (budget reference)

This is the strongest evidence that the system works on real, messy data and
not only on synthetic rows.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

warnings.filterwarnings("ignore")  # sklearn convergence chatter (not errors)

from benchmarks.real_cases import build_real_cases
from benchmarks.evaluation import (
    all_experiments_count,
    baseline_predict,
    GROUND_TRUTH,
    run_leaklens_case,
    score_approach,
)

OUT = Path(__file__).resolve().parent / "real_benchmark_results.json"


def main() -> int:
    cases = build_real_cases()

    print(f"Real-dataset benchmark: {len(cases)} cases across "
          f"{len({c.name.split('_',1)[0] for c in cases})} real datasets\n")
    rows = []
    for c in cases:
        rows.append(
            (c.name, c.df.shape[0], c.df.shape[1],
             c.category, ",".join(c.leaked_features) or "—")
        )
    for row in rows:
        print("  {:<42s} {:<6d} {:<4d} {:<28s} {}".format(*row))
    print()

    # Per-case adaptive verdict (transparency: what LeakLens reported vs truth).
    print("Per-case LeakLens adaptive results:")
    per_case = []
    def adaptive_predict(c):
        started = time.perf_counter()
        inv = run_leaklens_case(c, seed=42)
        seconds = time.perf_counter() - started
        pred_cats = sorted({f.category for f in inv.findings})
        truth = sorted(GROUND_TRUTH.get(c.category, set()))
        hit = "OK" if set(pred_cats) == set(truth) else "PART" if set(pred_cats) & set(truth) else "MISS"
        print(f"  {c.name:<42s} truth={truth!s:<52s} predicted={pred_cats!s:<52s} {hit}")
        per_case.append({"case": c.name, "truth": truth, "predicted": pred_cats, "verdict": hit})
        return [(f.category, f.affected_features) for f in inv.findings], len(inv.results), seconds

    adaptive = score_approach(cases, adaptive_predict)
    print()

    baseline = score_approach(cases, baseline_predict)
    upper_bound_mean = sum(all_experiments_count(c) for c in cases) / len(cases)

    print("=" * 70)
    print(f"{'Metric':28s} {'Rule-based':>12s} {'LeakLens':>12s}")
    print("=" * 70)
    for key in ("precision", "recall", "f1", "false_positive_rate_clean", "root_cause_accuracy"):
        print(f"{key:28s} {baseline[key]:>12.3f} {adaptive[key]:>12.3f}")
    print(f"{'mean experiments':28s} {baseline['mean_experiments']:>12.2f} {adaptive['mean_experiments']:>12.2f}")
    print(f"{'mean seconds/case':28s} {baseline['mean_seconds']:>12.2f} {adaptive['mean_seconds']:>12.2f}")
    print(f"{'run-all upper bound':28s} {upper_bound_mean:>12.2f}")
    print("=" * 70)

    results = {
        "datasets": sorted({c.name.split("_", 1)[0] for c in cases}),
        "n_cases": len(cases),
        "rule_based_baseline": baseline,
        "leaklens_adaptive": adaptive,
        "run_all_experiments_mean": round(upper_bound_mean, 2),
        "per_case": per_case,
        "note": (
            "Real data from sklearn (Wine, Breast Cancer) + UCI Bank Marketing; "
            "each non-clean case has one documented leakage injection. Metrics count "
            "all reported findings, including review flags. Experiment counts exclude "
            "profiling and signal calculations."
        ),
    }
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved results to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
