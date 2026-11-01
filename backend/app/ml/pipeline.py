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


def _encode_target(df: pd.DataFrame, target: str, pos_class: Optional[str] = None) -> np.ndarray:
    y = df[target]
    if y.dtype == object or y.dtype.kind in "OU":
        vals = y.astype(str)
        if pos_class is not None:
            return (vals == pos_class).astype(int).values
        # choose the most frequent class as negative by ordering alphabetically
        classes = sorted(vals.unique())
        return (vals == classes[-1]).astype(int).values
    return y.values


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
    y_full = _encode_target(df, target, pos_class)
    X_raw = df[list(features)]

    if chronological:
        if not time_column:
            raise ValueError("chronological split requires time_column")
        order = df[time_column].argsort().values
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

    pre = build_preprocessor(features, df)
    model = LogisticRegression(max_iter=3000, random_state=seed)
    Xtr = pre.fit_transform(X_train)
    Xte = pre.transform(X_test)
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }


def cross_val_auc(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    seed: int = 42,
    pos_class: Optional[str] = None,
    cv: int = 3,
) -> dict:
    y_full = _encode_target(df, target, pos_class)
    pre = build_preprocessor(features, df)
    Xt = pre.fit_transform(df[list(features)])
    model = LogisticRegression(max_iter=3000, random_state=seed)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(Xt, y_full):
        model.fit(Xt[tr], y_full[tr])
        y_prob = model.predict_proba(Xt[te])[:, 1]
        aucs.append(roc_auc_score(y_full[te], y_prob))
    return {"cv_auc_mean": float(np.mean(aucs)), "cv_auc_std": float(np.std(aucs)), "cv": cv}
