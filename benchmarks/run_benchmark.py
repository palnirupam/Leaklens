"""Synthetic benchmark runner for LeakLens.

Compares:
  1. Rule-based baseline detector (signals only, zero experiments)
  2. LeakLens adaptive investigation system
  3. "Run-all experiments" upper bound (budget reference)

Gemini-only analysis is wired but requires a valid GEMINI_API_KEY; without one
it is reported as skipped (we never fabricate its results).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.data.synthetic import generate_benchmark_cases
from benchmarks.evaluation import (
    all_experiments_count,
    baseline_predict,
    leaklens_predict,
    score_approach,
    score_gemini,
)

PER_CATEGORY = 6
OUT = Path(__file__).resolve().parent / "benchmark_results.json"


def main() -> int:
    cases = generate_benchmark_cases(per_category=PER_CATEGORY, seed_offset=100, n_rows=500)

    print(f"Generating {len(cases)} synthetic benchmark cases "
          f"({PER_CATEGORY} per category across 5 categories)\n")

    baseline = score_approach(cases, baseline_predict)
    adaptive = score_approach(cases, leaklens_predict(seed=42))
    upper_bound_mean = sum(all_experiments_count(c) for c in cases) / len(cases)

    print("=" * 70)
    print(f"{'Metric':28s} {'Rule-based':>12s} {'LeakLens':>12s}")
    print("=" * 70)
    for key in ("precision", "recall", "f1", "false_positive_rate_clean", "root_cause_accuracy"):
        print(f"{key:28s} {baseline[key]:>12.3f} {adaptive[key]:>12.3f}")
    print(f"{'mean experiments':28s} {baseline['mean_experiments']:>12.2f} {adaptive['mean_experiments']:>12.2f}")
    print(f"{'mean seconds/case':28s} {baseline['mean_seconds']:>12.2f} {adaptive['mean_seconds']:>12.2f}")
    print(f"{'runtime-experiments upper bound':28s} {upper_bound_mean:>12.2f}")
    print("=" * 70)

    results = {
        "per_category": PER_CATEGORY,
        "n_cases": len(cases),
        "rule_based_baseline": baseline,
        "leaklens_adaptive": adaptive,
        "run_all_experiments_mean": upper_bound_mean,
        "gemini_only": score_gemini(cases),
        "note": (
            "LeakLens adaptive uses the same signals as the rule-based baseline but "
            "investigates them with budget-aware experiments. Metrics count all reported "
            "findings, including review flags; they do not measure confirmed leakage alone. "
            "Experiment counts exclude profiling and signal calculations."
        ),
    }
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved results to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
