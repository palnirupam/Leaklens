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
from typing import Callable

from app.data.synthetic import SyntheticCase
from app.agents.experiment_executor import _candidate_experiments
from app.agents.hypothesis_planner import HypothesisPlanner
from app.core.models import Investigation, InvestigationConfig
from app.detectors.dataset_investigator import investigate
from app.detectors.leakage_detectors import detect_signals
from app.services.gemini_client import GeminiClient
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
    if sig.duplicate_rows > 0 or sig.repeated_entities > 0:
        out.append(("duplicate_contamination", [case.entity_column] if case.entity_column else []))
    if sig.temporal and not sig.temporal.get("error") and sig.temporal.get("gap_auc", 0.0) > 0.10:
        out.append(("temporal_leakage", []))
    for feat in sig.prediction_time_unavailable:
        out.append(("prediction_time_unavailable", [feat]))
    for first, second, _, _ in sig.interaction_features:
        out.append(("feature_interaction_leakage", [first, second]))
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

        # Root-cause accuracy includes missed categories in the denominator.
        # Multiple findings in one category contribute the union of features.
        cat_map: dict[str, set[str]] = {}
        for category, affected in findings:
            cat_map.setdefault(category, set()).update(affected)
        for gcat in gt:
            root_cause_total += 1
            if gcat in cat_map:
                affected = cat_map[gcat]
                leaked = set(case.leaked_features)
                if (case.category in ("duplicate_contamination",)) or (not leaked) or (affected & leaked):
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


def leaklens_predict(seed: int = 42):
    def _f(case: SyntheticCase) -> tuple[list[tuple[str, list[str]]], int, float]:
        t0 = time.perf_counter()
        inv = run_leaklens_case(case, seed=seed)
        findings = [(f.category, f.affected_features) for f in inv.findings]
        return findings, len(inv.results), time.perf_counter() - t0
    return _f


# Compatibility for older imports; remove after downstream users migrate.
leakaklens_predict = leaklens_predict


def all_experiments_count(case: SyntheticCase) -> int:
    """Upper bound: number of candidate experiments if we ran every possible check."""
    prof = investigate(case.df, case.target)
    sig = detect_signals(case.df, prof, case.target, timestamp_column=case.timestamp_column,
                         entity_column=case.entity_column,
                         unavailable_at_prediction=case.unavailable_at_prediction)
    hypotheses, _ = HypothesisPlanner().plan("benchmark", prof, sig, [])
    return sum(len(_candidate_experiments(h)) for h in hypotheses)


def score_gemini(cases: list[SyntheticCase]) -> dict:
    """Score Gemini hypotheses from the same metadata/signals, without adaptive experiments."""
    client = GeminiClient()
    if not client.is_configured:
        return {"status": "skipped (no GEMINI_API_KEY configured)"}
    planner = HypothesisPlanner(client)

    def predict(case):
        started = time.perf_counter()
        profile = investigate(case.df, case.target)
        signals = detect_signals(
            case.df, profile, case.target,
            timestamp_column=case.timestamp_column,
            entity_column=case.entity_column,
            unavailable_at_prediction=case.unavailable_at_prediction,
        )
        suggestions = planner._gemini_suggest("benchmark", profile, signals)
        if suggestions is None:
            raise RuntimeError(f"Gemini returned no valid response for {case.name}.")
        return [(h.category, h.affected_features) for h in suggestions], 0, time.perf_counter() - started

    try:
        return {"status": "completed", **score_approach(cases, predict)}
    except RuntimeError as exc:
        return {"status": "failed", "error": str(exc)}
