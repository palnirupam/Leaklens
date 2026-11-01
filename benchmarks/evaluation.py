"""Benchmark evaluation utilities.

Ground-truth categories per synthetic case:
- ``clean``                     -> {} (no leakage)
- ``target_derived_feature``    -> {target_derived_feature}
- ``prediction_time_unavailable`` -> {prediction_time_unavailable}
- ``temporal_leakage``          -> {temporal_leakage, target_derived_feature}
                                 (the injected feature is future-derived AND
                                 target-correlated)
- ``duplicate_contamination``   -> {duplicate_contamination}
"""
from __future__ import annotations

import tempfile
import os
import time
from typing import Callable, Optional

from app.data.synthetic import SyntheticCase, generate_benchmark_cases
from app.core.models import Investigation, InvestigationConfig
from app.services.orchestrator import InvestigationOrchestrator

GROUND_TRUTH: dict[str, set[str]] = {
    "clean": set(),
    "target_derived_feature": {"target_derived_feature"},
    "prediction_time_unavailable": {"prediction_time_unavailable"},
    "temporal_leakage": {"temporal_leakage", "target_derived_feature"},
    "duplicate_contamination": {"duplicate_contamination"},
}


def run_leaklens_case(case: SyntheticCase, seed: int = 42) -> Investigation:
    orch = InvestigationOrchestrator()
    cfg = InvestigationConfig(
        dataset_path="",
        target_column=case.target,
        timestamp_column=case.timestamp_column,
        entity_column=case.entity_column,
        unavailable_at_prediction=case.unavailable_at_prediction,
        random_seed=seed,
        source_filename=case.name,
    )
    inv = orch.create(cfg)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "data.csv")
        case.df.to_csv(p, index=False)
        cfg.dataset_path = p
        orch.run(inv)
    return inv


def baseline_findings(case: SyntheticCase) -> list[tuple[str, list[str]]]:
    """Pure signal-level baseline: every detected signal becomes a finding."""
    from app.detectors.dataset_investigator import investigate
    from app.detectors.leakage_detectors import detect_signals

    prof = investigate(case.df, case.target)
    sig = detect_signals(
        case.df, prof, case.target,
        timestamp_column=case.timestamp_column,
        entity_column=case.entity_column,
        unavailable_at_prediction=case.unavailable_at_prediction,
    )
    out: list[tuple[str, list[str]]] = []
    for feat, _ in sig.high_association_features:
        out.append(("target_derived_feature", [feat]))
    if sig.conflicting_duplicates > 0:
        out.append(("duplicate_contamination", []))
    if sig.temporal is not None:
        out.append(("temporal_leakage", []))
    for feat in sig.prediction_time_unavailable:
        out.append(("prediction_time_unavailable", [feat]))
    return out


def score_approach(
    cases: list[SyntheticCase],
    predict: Callable[[SyntheticCase], tuple[list[tuple[str, list[str]]], int, float]],
) -> dict:
    """predict(case) -> (findings[(category, affected)], n_experiments, seconds)."""
    true_pos = 0
    false_pos = 0
    false_neg = 0
    clean_fp_cases = 0
    root_cause_hits = 0
    root_cause_total = 0
    total_experiments = 0
    total_seconds = 0.0
    n_clean = 0

    for case in cases:
        gt = GROUND_TRUTH[case.category]
        findings, n_exp, seconds = predict(case)
        cats = {c for c, _ in findings}

        if case.category == "clean":
            n_clean += 1
            if cats:
                clean_fp_cases += 1

        tp = len(cats & gt)
        fp = len(cats - gt)
        fn = len(gt - cats)
        true_pos += tp
        false_pos += fp
        false_neg += fn

        # root-cause accuracy: affected features should include injected leak features
        cat_map = {c: a for c, a in findings}
        for gcat in gt:
            if gcat in cat_map:
                root_cause_total += 1
                affected = set(cat_map[gcat])
                leaked = set(case.leaked_features)
                if (case.category in ("duplicate_contamination",)) or (not leaked) or (affected & leaked):
                    root_cause_hits += 1
                elif gcat == "target_derived_feature" and affected & leaked:
                    root_cause_hits += 1

        total_experiments += n_exp
        total_seconds += seconds

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) else 0.0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n_cases": len(cases),
        "true_pos": true_pos,
        "false_pos": false_pos,
        "false_neg": false_neg,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "false_positive_rate_clean": round(clean_fp_cases / n_clean, 3) if n_clean else 0.0,
        "root_cause_accuracy": round(root_cause_hits / root_cause_total, 3) if root_cause_total else 0.0,
        "mean_experiments": round(total_experiments / len(cases), 2),
        "mean_seconds": round(total_seconds / len(cases), 2),
        "total_seconds": round(total_seconds, 1),
    }


def baseline_predict(case: SyntheticCase) -> tuple[list[tuple[str, list[str]]], int, float]:
    t0 = time.perf_counter()
    findings = baseline_findings(case)
    return findings, 0, time.perf_counter() - t0


def leakaklens_predict(seed: int = 42):
    def _f(case: SyntheticCase) -> tuple[list[tuple[str, list[str]]], int, float]:
        t0 = time.perf_counter()
        inv = run_leaklens_case(case, seed=seed)
        findings = [(f.category, f.affected_features) for f in inv.findings]
        return findings, len(inv.results), time.perf_counter() - t0
    return _f


def all_experiments_count(case: SyntheticCase) -> int:
    """Upper bound: number of candidate experiments if we ran every possible check."""
    n = 0
    from app.detectors.dataset_investigator import investigate
    from app.detectors.leakage_detectors import detect_signals
    prof = investigate(case.df, case.target)
    sig = detect_signals(case.df, prof, case.target, timestamp_column=case.timestamp_column,
                         entity_column=case.entity_column,
                         unavailable_at_prediction=case.unavailable_at_prediction)
    n += len(sig.high_association_features) * 2  # ablation + permutation per feature
    if sig.conflicting_duplicates > 0:
        n += 1
    if sig.temporal is not None:
        n += 1 + len(sig.high_association_features)
    n += len(sig.prediction_time_unavailable)
    return max(1, n)
