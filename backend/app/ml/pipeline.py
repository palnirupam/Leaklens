"""Scikit-learn helpers used by experiments.

All numerical evaluation here is deterministic given the same inputs and seed.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import FunctionTransformer


def _to_float(X) -> np.ndarray:
    """Coerce boolean/integer columns to float so SimpleImputer accepts them."""
    return np.asarray(X, dtype=float)


def build_preprocessor(features: Sequence[str], df: pd.DataFrame) -> ColumnTransformer:
    # Treat booleans as numeric too (their dtype kind is "b", not in "iufc").
    # Cast to float inside the numeric pipeline to avoid SimpleImputer's
    # "does not support dtype bool" error.
    numeric = [c for c in features if df[c].dtype.kind in "iufcb"]
    categorical = [c for c in features if c not in numeric]
    return ColumnTransformer(
        transformers=[
            (
                "num",
                make_pipeline(
                    FunctionTransformer(_to_float, validate=False),
                    SimpleImputer(strategy="median"),
                ),
                numeric,
            ),
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore"),
                ),
                categorical,
            ),
        ]
    )


def encode_target(
    df: pd.DataFrame,
    target: str,
    pos_class: Optional[str] = None,
) -> np.ndarray:
    """Encode any non-null classification target as deterministic integer labels.

    String labels (for example ``yes``/``no``) and multiclass targets are both
    supported.  The previous implementation returned raw numeric multiclass
    labels and treated every string target as binary, which made ROC-AUC either
    fail or silently describe the wrong problem.
    """
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} is missing.")
    y = df[target]
    if y.isna().any():
        raise ValueError("Target column contains missing values.")

    values = y.astype(str)
    classes = sorted(values.unique().tolist())
    if len(classes) < 2:
        raise ValueError("Target column must contain at least two classes.")

    if pos_class is not None:
        if len(classes) != 2:
            raise ValueError("pos_class can only be used with a binary target.")
        positive = str(pos_class)
        if positive not in classes:
            raise ValueError(f"Positive class {positive!r} is not present in the target.")
        return (values == positive).astype(int).to_numpy()

    codes, _ = pd.factorize(values, sort=True)
    return codes.astype(int, copy=False)


# Backwards-compatible private name used by older callers/tests.
_encode_target = encode_target


def _validated_cv_splits(y: np.ndarray, requested: int) -> int:
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Target column must contain at least two classes.")
    n_splits = min(int(requested), int(counts.min()))
    if n_splits < 2:
        raise ValueError("Each target class needs at least two rows for validation.")
    return n_splits


def _roc_auc(y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> float:
    """Return binary or weighted one-vs-rest multiclass ROC-AUC."""
    if len(classes) == 2:
        positive_index = int(np.where(classes == 1)[0][0]) if 1 in classes else 1
        return float(roc_auc_score(y_true, probabilities[:, positive_index]))
    return float(
        roc_auc_score(
            y_true,
            probabilities,
            labels=classes,
            multi_class="ovr",
            average="weighted",
        )
    )


def _ordered_row_positions(df: pd.DataFrame, time_column: str) -> np.ndarray:
    values = df[time_column]
    if values.isna().any():
        raise ValueError(f"Timestamp column {time_column!r} contains missing values.")
    if values.dtype.kind in "iufcM":
        return np.argsort(values.to_numpy(), kind="stable")

    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    if parsed.isna().any():
        raise ValueError(
            f"Timestamp column {time_column!r} contains values that cannot be parsed."
        )
    return np.argsort(parsed.to_numpy(), kind="stable")


def evaluate_split(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    test_size: float = 0.25,
    seed: int = 42,
    pos_class: Optional[str] = None,
    chronological: bool = False,
    time_column: Optional[str] = None,
) -> dict:
    """Train logistic-regression baseline and return metrics for one split."""
    feature_list = list(dict.fromkeys(features))
    if not feature_list:
        raise ValueError("At least one feature is required for model evaluation.")
    y_full = encode_target(df, target, pos_class)
    X_raw = df[feature_list]

    if chronological:
        if not time_column:
            raise ValueError("chronological split requires time_column")
        order = _ordered_row_positions(df, time_column)
        X_raw = X_raw.iloc[order]
        y_full = y_full[order]
        n = len(X_raw)
        split = int((1 - test_size) * n)
        X_train, X_test = X_raw.iloc[:split], X_raw.iloc[split:]
        y_train, y_test = y_full[:split], y_full[split:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_raw, y_full, test_size=test_size, stratify=y_full, random_state=seed
        )
        X_train, y_train, X_test, y_test = X_train, y_train, X_test, y_test

    pre = build_preprocessor(feature_list, df)
    model = LogisticRegression(max_iter=3000, random_state=seed)
    Xtr = pre.fit_transform(X_train)
    Xte = pre.transform(X_test)
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)
    classes = model.classes_
    f1_average = "binary" if len(classes) == 2 else "weighted"

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, average=f1_average, zero_division=0)),
        "roc_auc": _roc_auc(y_test, y_prob, classes),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_classes": int(len(classes)),
    }


def cross_val_auc(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    seed: int = 42,
    pos_class: Optional[str] = None,
    cv: int = 3,
) -> dict:
    feature_list = list(dict.fromkeys(features))
    if not feature_list:
        raise ValueError("At least one feature is required for model evaluation.")
    y_full = encode_target(df, target, pos_class)
    X_raw = df[feature_list]
    n_splits = _validated_cv_splits(y_full, cv)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X_raw, y_full):
        # Preprocessing is deliberately fit inside each fold.  Fitting it once
        # on the full dataset leaks validation-fold medians/categories into the
        # training folds and invalidates the leakage investigation itself.
        pre = build_preprocessor(feature_list, df)
        Xtr = pre.fit_transform(X_raw.iloc[tr])
        Xte = pre.transform(X_raw.iloc[te])
        model = LogisticRegression(max_iter=3000, random_state=seed)
        model.fit(Xtr, y_full[tr])
        y_prob = model.predict_proba(Xte)
        aucs.append(_roc_auc(y_full[te], y_prob, model.classes_))
    return {
        "cv_auc_mean": float(np.mean(aucs)),
        "cv_auc_std": float(np.std(aucs)),
        "cv": n_splits,
        "n_classes": int(len(np.unique(y_full))),
    }
