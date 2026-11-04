"""Leakage detection signals.

These are *signals* — statistics the hypothesis planner turns into candidate
hypotheses. No single signal is treated as proof; findings require experimental
evidence from the adaptive executor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from app.ml.pipeline import encode_target, evaluate_split

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
    repeated_entities: int = 0
    repeated_entity_rows: int = 0
    entity_column: Optional[str] = None
    #: temporal random-vs-chronological comparison (dict or None)
    temporal: Optional[dict] = None
    timestamp_column: Optional[str] = None
    #: features excluded because they are unavailable at prediction time
    prediction_time_unavailable: list[str] = field(default_factory=list)
    #: feature pairs whose *joint* nonlinear signal is strong while each feature
    #: alone is uninformative (e.g. XOR-style interactions): (f1, f2, best_single_auc, joint_auc)
    interaction_features: list[tuple[str, str, float, float]] = field(default_factory=list)


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
        counts = df[entity_column].value_counts(dropna=False)
        repeated = counts[counts > 1]
        signals.repeated_entities = int(len(repeated))
        signals.repeated_entity_rows = int(repeated.sum()) if len(repeated) else 0
        if signals.repeated_entities:
            ent = df[[entity_column, target_column]].copy()
            g = ent.groupby(entity_column, dropna=False)[target_column].nunique()
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

    # 4) interaction check: pairs with a strong *joint* signal but weak marginals
    signals.interaction_features = _detect_interactions(
        df, target_column, seed=seed
    )

    return signals


def _pair_auc(
    sub: pd.DataFrame, feats: list[str], target: str, seed: int
) -> float:
    """Cross-validated ROC-AUC of a shallow decision tree on the given features."""
    X = sub[feats].to_numpy(dtype=float)
    y = encode_target(sub, target)
    classes, counts = np.unique(y, return_counts=True)
    n_splits = min(3, int(counts.min()))
    if len(classes) < 2 or n_splits < 2:
        return 0.5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs: list[float] = []
    for tr, te in skf.split(X, y):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = DecisionTreeClassifier(max_depth=4, random_state=seed)
        clf.fit(X[tr], y[tr])
        probabilities = clf.predict_proba(X[te])
        if probabilities.shape[1] < 2:
            continue
        if len(clf.classes_) == 2:
            aucs.append(float(roc_auc_score(y[te], probabilities[:, 1])))
        else:
            aucs.append(
                float(
                    roc_auc_score(
                        y[te],
                        probabilities,
                        labels=clf.classes_,
                        multi_class="ovr",
                        average="weighted",
                    )
                )
            )
    return float(np.mean(aucs)) if aucs else 0.5


def _detect_interactions(
    df: pd.DataFrame,
    target_column: str,
    seed: int = 42,
    max_pairs_features: int = 7,
    sample: int = 3000,
) -> list[tuple[str, str, float, float]]:
    """Find numeric feature pairs that reveal a strong *joint* signal while each
    feature is individually weak (e.g. XOR-style). Uses a shallow tree so that
    nonlinear interactions are visible — a linear model cannot see XOR.

    This is an intentionally conservative *flag* (review), not proof of leak.
    """
    numeric = [
        c for c in df.columns
        if c != target_column and df[c].dtype.kind in "iufc"
    ]
    if len(numeric) < 2:
        return []
    sub = df[numeric + [target_column]]
    if len(sub) > sample:
        sub = sub.sample(sample, random_state=seed)
    cols = numeric[:max_pairs_features]
    out: list[tuple[str, str, float, float]] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            fi, fj = cols[i], cols[j]
            try:
                mi = _pair_auc(sub, [fi], target_column, seed)
                mj = _pair_auc(sub, [fj], target_column, seed)
                joint = _pair_auc(sub, [fi, fj], target_column, seed)
            except Exception:  # noqa: BLE001
                continue
            if max(mi, mj) < 0.6 and joint >= 0.85:
                out.append((fi, fj, round(max(mi, mj), 3), round(joint, 3)))
    return out
