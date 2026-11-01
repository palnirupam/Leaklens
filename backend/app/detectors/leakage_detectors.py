"""Leakage detection signals.

These are *signals* — statistics the hypothesis planner turns into candidate
hypotheses. No single signal is treated as proof; findings require experimental
evidence from the adaptive executor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from app.ml.pipeline import evaluate_split

# association threshold above which a feature is flagged "suspiciously predictive"
HIGH_ASSOCIATION = 0.5


@dataclass
class DetectionSignals:
    #: features with association >= HIGH_ASSOCIATION (target-derived suspicion)
    high_association_features: list[tuple[str, float]] = field(default_factory=list)
    #: exact duplicate row count
    duplicate_rows: int = 0
    #: duplicate pairs that share a row but disagree on target (conflicting)
    conflicting_duplicates: int = 0
    #: entity-level overlap signal (rows sharing an entity value with mixed labels)
    entity_conflicts: int = 0
    entity_column: Optional[str] = None
    #: temporal random-vs-chronological comparison (dict or None)
    temporal: Optional[dict] = None
    timestamp_column: Optional[str] = None
    #: features excluded because they are unavailable at prediction time
    prediction_time_unavailable: list[str] = field(default_factory=list)


def detect_signals(
    df: pd.DataFrame,
    profile,
    target_column: str,
    timestamp_column: Optional[str] = None,
    entity_column: Optional[str] = None,
    unavailable_at_prediction: Optional[list[str]] = None,
    seed: int = 42,
) -> DetectionSignals:
    signals = DetectionSignals()
    signals.prediction_time_unavailable = list(unavailable_at_prediction or [])

    # 1) suspiciously predictive features from profile association
    for col in profile.columns:
        if col.target_association >= HIGH_ASSOCIATION and col.name not in signals.prediction_time_unavailable:
            signals.high_association_features.append((col.name, col.target_association))

    # 2) contamination checks
    # NOTE: compare by FEATURE columns only. `df.duplicated()` compares all
    # columns including the target, which hides duplicate-feature rows that
    # carry conflicting labels — precisely the contamination signal.
    feature_cols = [c for c in df.columns if c != target_column]
    if feature_cols:
        feat_dup = df.duplicated(subset=feature_cols, keep=False)
        signals.duplicate_rows = int(feat_dup.sum())
        if signals.duplicate_rows > 0:
            dup = df[feat_dup]
            g = dup.groupby(feature_cols)[target_column].nunique()
            signals.conflicting_duplicates = int((g > 1).sum())

    # entity overlap
    if entity_column and entity_column in df.columns:
        signals.entity_column = entity_column
        if df[entity_column].nunique() < len(df):
            ent = df[[entity_column, target_column]].copy()
            g = ent.groupby(entity_column)[target_column].nunique()
            signals.entity_conflicts = int((g > 1).sum())

    # 3) temporal leak: random vs chronological split
    if timestamp_column and timestamp_column in df.columns:
        try:
            features = [c for c in df.columns if c not in (target_column, timestamp_column, entity_column)]
            rand = evaluate_split(df, features, target_column, seed=seed)
            chrono = evaluate_split(
                df, features, target_column, seed=seed, chronological=True, time_column=timestamp_column
            )
            signals.temporal = {
                "random_auc": rand["roc_auc"],
                "chronological_auc": chrono["roc_auc"],
                "random_f1": rand["f1"],
                "chronological_f1": chrono["f1"],
                "gap_auc": rand["roc_auc"] - chrono["roc_auc"],
            }
            signals.timestamp_column = timestamp_column
        except Exception as exc:  # noqa: BLE001
            signals.temporal = {"error": str(exc)}

    return signals
