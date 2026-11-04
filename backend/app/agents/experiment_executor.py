"""Agent 3 — Adaptive Experiment Executor.

Implements **budget-aware adaptive leakage investigation**: it maintains a set
of candidate hypotheses, scores every available diagnostic experiment with a
deterministic priority formula, runs the highest-value/lowest-cost experiment,
updates hypothesis confidence from the observed result, and repeats.

The priority formula (explainable and deterministic):

    score(a) = evidence_strength(h) * diagnostic_value(a) / (1 + norm_cost(a))

where:
- ``evidence_strength(h)`` grows as experiments for hypothesis ``h`` return
  supporting evidence,
- ``diagnostic_value(a)`` is the fixed expected information gain of experiment ``a``,
- ``norm_cost(a)`` is its normalized estimated execution cost.

Ties are broken by hypothesis and action order so identical inputs always yield
the same experiment ordering (determinism).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from app.core.models import Investigation, CandidateHypothesis, ExperimentResult
from app.experiments.registry import EXPERIMENT_META, run_experiment

CONFIRM_AUC_DROP = 0.10
CONFIRM_GAP = 0.10
MAX_EXPERIMENTS = 12
CONFIDENCE_CONFIRMED = 0.8
CONFIDENCE_REJECTED = 0.2


def _candidate_experiments(h: CandidateHypothesis) -> list[dict]:
    """Return candidate actions (kind + params) for a hypothesis."""
    cat = h.category
    actions: list[dict] = []
    if cat == "target_derived_feature":
        for feat in h.affected_features or [""]:
            actions.append({"kind": "feature_ablation", "params": {"feature": feat}})
        # one permutation test on the first affected feature
        if h.affected_features:
            actions.append({"kind": "permutation_test", "params": {"feature": h.affected_features[0]}})
    elif cat == "duplicate_contamination":
        actions.append({"kind": "duplicate_conflict_check", "params": {}})
    elif cat == "temporal_leakage":
        actions.append({"kind": "temporal_split_compare", "params": {}})
    elif cat == "prediction_time_unavailable":
        for feat in h.affected_features or []:
            actions.append({"kind": "availability_check", "params": {"feature": feat}})
    elif cat == "preprocessing_leakage":
        actions.append({"kind": "preprocessing_ast_review", "params": {}})
    elif cat == "evaluation_protocol_risk":
        actions.append({"kind": "preprocessing_ast_review", "params": {}})
    return actions


def _interpret_outcome(h: CandidateHypothesis, result: ExperimentResult) -> bool:
    """Return True if the experiment supports the hypothesis."""
    interp = result.interpretation or ""
    ev = result.evidence or {}
    if h.category == "target_derived_feature":
        if result.name == "feature_ablation":
            return ev.get("auc_drop", 0.0) > CONFIRM_AUC_DROP
        if result.name == "permutation_test":
            return interp == "feature_dependent"
    if h.category == "duplicate_contamination":
        return interp in {
            "conflicting_duplicate_records",
            "duplicate_overlap_risk",
            "entity_overlap_risk",
        }
    if h.category == "temporal_leakage":
        if result.name == "temporal_split_compare":
            return ev.get("gap_auc", 0.0) > CONFIRM_GAP
    if h.category == "prediction_time_unavailable":
        return True  # availability is a deployment fact from config
    if h.category in ("preprocessing_leakage", "evaluation_protocol_risk"):
        return bool(ev.get("code_findings"))
    return False


def _update_confidence(h: CandidateHypothesis, supporting: bool) -> None:
    if supporting:
        h.confidence = min(1.0, h.confidence + 0.25)
        h.status = "investigating"
    else:
        # A single negative diagnostic must not prevent the remaining,
        # independent experiment(s) from running.
        h.confidence = max(0.0, h.confidence - 0.15)
        h.status = "investigating"


class AdaptiveExperimentExecutor:
    def __init__(
        self,
        max_experiments: int = MAX_EXPERIMENTS,
        seed: int = 42,
        timestamp_column: Optional[str] = None,
        entity_column: Optional[str] = None,
    ) -> None:
        self.max_experiments = max_experiments
        self.seed = seed
        self.timestamp_column = timestamp_column
        self.entity_column = entity_column

    # -- priority formula ------------------------------------------------------
    def _evidence_strength(self, h: CandidateHypothesis) -> float:
        if not h.experiments:
            return max(0.1, h.confidence)
        return min(1.0, h.confidence)

    def _score(self, h: CandidateHypothesis, action: dict, done: set[str]) -> float:
        meta = EXPERIMENT_META.get(action["kind"], {})
        diag = float(meta.get("diagnostic_value", 0.5))
        cost = float(meta.get("cost_ms", 200.0))
        norm_cost = cost / 1000.0
        evidence = self._evidence_strength(h)
        return evidence * diag / (1.0 + norm_cost)

    # -- main loop ---------------------------------------------------------------
    def execute(
        self,
        investigation: Investigation,
        df: pd.DataFrame,
        target: str,
        code_findings: list,
    ) -> None:
        done: set[str] = set()

        for _ in range(self.max_experiments):
            # build candidate actions across open hypotheses
            candidates: list[tuple[float, int, int, str, CandidateHypothesis, dict]] = []
            for hypothesis_index, h in enumerate(investigation.hypotheses):
                if h.status in ("confirmed", "rejected"):
                    continue
                for action_index, action in enumerate(_candidate_experiments(h)):
                    action_id = f"{h.id}::{action['kind']}::{action['params'].get('feature', '')}"
                    if action_id in done:
                        continue
                    score = self._score(h, action, done)
                    candidates.append((-score, hypothesis_index, action_index, action_id, h, action))

            if not candidates:
                break

            # Random hypothesis IDs must not decide which experiment gets the budget.
            candidates.sort(key=lambda t: t[:3])
            _, _, _, action_id, h, action = candidates[0]
            done.add(action_id)

            # execute
            params = dict(action["params"])
            params.update(
                {
                    "experiment_id": f"exp_{h.id}_{len(investigation.results) + 1}",
                    "hypothesis_id": h.id,
                    "category": h.category,
                    "timestamp_column": self.timestamp_column,
                    "entity_column": self.entity_column,
                    "code_findings": code_findings,
                }
            )
            result = run_experiment(
                action["kind"], df, target, params, seed=self.seed
            )
            result.hypothesis_id = h.id
            investigation.results.append(result)
            h.experiments.append(result.experiment_id)

            if result.status == "failed":
                # inconclusive; leave confidence unchanged
                continue

            supporting = _interpret_outcome(h, result)
            _update_confidence(h, supporting)

            if supporting and h.confidence >= CONFIDENCE_CONFIRMED:
                h.status = "confirmed"
                # optionally run one verification re-check later
