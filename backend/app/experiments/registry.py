"""Experiment definitions and runner.

Experiments are the only place numerical evidence is produced. Each experiment
is a pure function over (df, target, params) returning an :class:`ExperimentResult`
with metrics, evidence, interpretation and a deterministic seed.

Cost model: experiments expose a ``cost_ms`` estimate so the adaptive executor
can prioritize cheap, high-value checks.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Optional

import pandas as pd

from app.core.models import ExperimentResult
from app.ml.pipeline import evaluate_split, cross_val_auc

# --- experiment metadata ----------------------------------------------------
EXPERIMENT_META: dict[str, dict[str, Any]] = {
    "feature_ablation": {
        "label": "Feature ablation (drop suspicious feature)",
        "cost_ms": 400.0,
        "diagnostic_value": 1.0,
    },
    "permutation_test": {
        "label": "Shuffled-target sanity check",
        "cost_ms": 500.0,
        "diagnostic_value": 0.9,
    },
    "duplicate_conflict_check": {
        "label": "Duplicate / entity contamination check",
        "cost_ms": 20.0,
        "diagnostic_value": 0.8,
    },
    "temporal_split_compare": {
        "label": "Random vs chronological split comparison",
        "cost_ms": 500.0,
        "diagnostic_value": 0.85,
    },
    "availability_check": {
        "label": "Prediction-time availability impact",
        "cost_ms": 400.0,
        "diagnostic_value": 0.7,
    },
    "preprocessing_ast_review": {
        "label": "Static training-script review",
        "cost_ms": 10.0,
        "diagnostic_value": 0.6,
    },
}


def _run_feature_ablation(
    df: pd.DataFrame, target: str, params: dict[str, Any], seed: int
) -> ExperimentResult:
    feat = params["feature"]
    other = [c for c in df.columns if c not in (target, feat, params.get("timestamp_column"), params.get("entity_column"))]
    baseline = cross_val_auc(df, other + [feat], target, seed=seed)
    ablated = cross_val_auc(df, other, target, seed=seed)
    drop = baseline["cv_auc_mean"] - ablated["cv_auc_mean"]
    return ExperimentResult(
        experiment_id=params.get("experiment_id", ""),
        hypothesis_id=params.get("hypothesis_id", ""),
        name="feature_ablation",
        category="target_derived_feature",
        status="completed",
        config={"feature": feat, "seed": seed},
        metrics={
            "all_features_auc": baseline["cv_auc_mean"],
            "ablated_auc": ablated["cv_auc_mean"],
            "auc_drop": drop,
        },
        evidence={
            "auc_drop": drop,
            "unsignificant_cutoff": 0.05,
            "explanation": (
                f"Dropping `{feat}` changed CV-AUC by {drop:+.3f}. "
                "A large drop indicates the feature carries information strongly "
                "associated with the target (suspected target-derived leakage)."
            ),
        },
        interpretation=(
            "strong_contribution" if drop > 0.1
            else "moderate_contribution" if drop > 0.03
            else "weak_or_no_contribution"
        ),
    )


def _run_permutation(
    df: pd.DataFrame, target: str, params: dict[str, Any], seed: int
) -> ExperimentResult:
    feat = params["feature"]
    other = [c for c in df.columns if c not in (target, feat, params.get("timestamp_column"), params.get("entity_column"))]
    shuffled = df.copy()
    rng = __import__("numpy").random.default_rng(seed)
    perm = rng.permutation(len(shuffled))
    shuffled.loc[:, target] = shuffled[target].iloc[perm].values
    shuffled_auc = cross_val_auc(shuffled, other + [feat], target, seed=seed)["cv_auc_mean"]
    real_auc = cross_val_auc(df, other + [feat], target, seed=seed)["cv_auc_mean"]
    return ExperimentResult(
        experiment_id=params.get("experiment_id", ""),
        hypothesis_id=params.get("hypothesis_id", ""),
        name="permutation_test",
        category="target_derived_feature",
        status="completed",
        config={"feature": feat, "seed": seed},
        metrics={"real_auc": real_auc, "shuffled_auc": shuffled_auc, "gap": real_auc - shuffled_auc},
        evidence={
            "real_auc": real_auc,
            "shuffled_auc": shuffled_auc,
            "explanation": (
                f"Real CV-AUC={real_auc:.3f} vs shuffled-target CV-AUC={shuffled_auc:.3f}. "
                "If the model still performs well on shuffled labels, the predictive signal "
                "does not depend on the target (less indicative of target leakage)."
            ),
        },
        interpretation="target-linked" if real_auc - shuffled_auc > 0.1 else "not_target_linked",
    )


def _run_duplicate_conflict(
    df: pd.DataFrame, target: str, params: dict[str, Any], seed: int
) -> ExperimentResult:
    from app.detectors.leakage_detectors import detect_signals
    from app.detectors.dataset_investigator import investigate
    prof = investigate(df, target)
    sig = detect_signals(df, prof, target, entity_column=params.get("entity_column"))
    return ExperimentResult(
        experiment_id=params.get("experiment_id", ""),
        hypothesis_id=params.get("hypothesis_id", ""),
        name="duplicate_conflict_check",
        category="duplicate_contamination",
        status="completed",
        config={},
        metrics={
            "duplicate_rows": sig.duplicate_rows,
            "conflicting_duplicates": sig.conflicting_duplicates,
            "entity_conflicts": sig.entity_conflicts,
        },
        evidence={
            "duplicate_rows": sig.duplicate_rows,
            "conflicting_duplicates": sig.conflicting_duplicates,
            "explanation": (
                f"Found {sig.duplicate_rows} feature-duplicate rows, "
                f"of which {sig.conflicting_duplicates} carry conflicting labels. "
                "Conflicting duplicates are a strong indicator of train/test contamination."
            ),
        },
        interpretation=(
            "contamination_present" if sig.conflicting_duplicates > 0 else "no_contamination"
        ),
    )


def _run_temporal_split(
    df: pd.DataFrame, target: str, params: dict[str, Any], seed: int
) -> ExperimentResult:
    ts_col = params["timestamp_column"]
    features = [c for c in df.columns if c not in (target, ts_col, params.get("entity_column"))]
    rand = evaluate_split(df, features, target, seed=seed)
    chrono = evaluate_split(df, features, target, seed=seed, chronological=True, time_column=ts_col)
    gap = rand["roc_auc"] - chrono["roc_auc"]
    return ExperimentResult(
        experiment_id=params.get("experiment_id", ""),
        hypothesis_id=params.get("hypothesis_id", ""),
        name="temporal_split_compare",
        category="temporal_leakage",
        status="completed",
        config={"timestamp_column": ts_col, "seed": seed},
        metrics={
            "random_auc": rand["roc_auc"],
            "chronological_auc": chrono["roc_auc"],
            "gap_auc": gap,
        },
        evidence={
            "random_auc": rand["roc_auc"],
            "chronological_auc": chrono["roc_auc"],
            "explanation": (
                f"Random-split AUC={rand['roc_auc']:.3f} vs chronological-split "
                f"AUC={chrono['roc_auc']:.3f}. If random splits outperform chronological "
                "ones markedly, the model may be using future information."
            ),
        },
        interpretation=(
            "future_leak_suspected" if gap > 0.1 else "no_clear_temporal_gap"
        ),
    )


def _run_availability(
    df: pd.DataFrame, target: str, params: dict[str, Any], seed: int
) -> ExperimentResult:
    feat = params["feature"]
    other = [c for c in df.columns if c not in (target, params.get("timestamp_column"), params.get("entity_column"))]
    with_feat = cross_val_auc(df, other + [feat], target, seed=seed)["cv_auc_mean"]
    without_feat = cross_val_auc(df, other, target, seed=seed)["cv_auc_mean"]
    return ExperimentResult(
        experiment_id=params.get("experiment_id", ""),
        hypothesis_id=params.get("hypothesis_id", ""),
        name="availability_check",
        category="prediction_time_unavailable",
        status="completed",
        config={"feature": feat, "seed": seed},
        metrics={"with_feature_auc": with_feat, "without_feature_auc": without_feat},
        evidence={
            "with_feature_auc": with_feat,
            "without_feature_auc": without_feat,
            "explanation": (
                f"Feature `{feat}` is marked unavailable at prediction time. "
                f"Including it yields CV-AUC={with_feat:.3f}; excluding it yields "
                f"{without_feat:.3f}. This is a deployment hazard, not proof of "
                "train/test contamination."
            ),
        },
        interpretation="unavailable_at_prediction",
    )


def _run_ast_review(
    df: pd.DataFrame, target: str, params: dict[str, Any], seed: int
) -> ExperimentResult:
    findings = params.get("code_findings", [])
    return ExperimentResult(
        experiment_id=params.get("experiment_id", ""),
        hypothesis_id=params.get("hypothesis_id", ""),
        name="preprocessing_ast_review",
        category="preprocessing_leakage",
        status="completed",
        config={},
        metrics={"code_findings": len(findings)},
        evidence={
            "code_findings": findings,
            "explanation": (
                "Static AST review flagged the above preprocessing patterns as suspicious. "
                "These are suspected code patterns and are not treated as proven runtime leakage."
            ),
        },
        interpretation="static_suspicion",
    )


_RUNNERS: dict[str, Callable[..., ExperimentResult]] = {
    "feature_ablation": _run_feature_ablation,
    "permutation_test": _run_permutation,
    "duplicate_conflict_check": _run_duplicate_conflict,
    "temporal_split_compare": _run_temporal_split,
    "availability_check": _run_availability,
    "preprocessing_ast_review": _run_ast_review,
}


def run_experiment(
    kind: str,
    df: pd.DataFrame,
    target: str,
    params: dict[str, Any],
    seed: int = 42,
) -> ExperimentResult:
    """Execute one experiment and time it."""
    if kind not in _RUNNERS:
        return ExperimentResult(
            experiment_id=params.get("experiment_id", ""),
            hypothesis_id=params.get("hypothesis_id", ""),
            name=kind,
            category=params.get("category", ""),
            status="failed",
            error=f"Unknown experiment kind: {kind}",
        )
    t0 = time.perf_counter()
    try:
        result = _RUNNERS[kind](df, target, params, seed)
    except Exception as exc:  # noqa: BLE001
        return ExperimentResult(
            experiment_id=params.get("experiment_id", ""),
            hypothesis_id=params.get("hypothesis_id", ""),
            name=kind,
            category=params.get("category", ""),
            status="failed",
            error=str(exc),
        )
    result.cost_ms = (time.perf_counter() - t0) * 1000.0
    return result
