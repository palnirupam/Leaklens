"""Agent 1 — Dataset Investigator.

Reads a CSV, builds a structured :class:`DatasetProfile`: column types, nulls,
duplicates, target distribution and feature-target association strengths.
Pure pandas/numpy analytics — no statistical claim is treated as proof.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from app.core.models import DatasetProfile, ColumnProfile

MAX_FEATURE_CLASSES = 50  # cap Cramer's V evaluation on high-cardinality columns


def _infer_dtype(col: pd.Series) -> str:
    kind = col.dtype.kind
    if kind in "iufc":
        return "numeric"
    if col.dtype.kind == "M":
        return "datetime"
    return "categorical"


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramer's V between two categorical (or discretized) vectors."""
    xc = x.astype("category").cat.codes
    yc = y.astype("category").cat.codes
    # handle NaN codes (-1)
    mask = (xc >= 0) & (yc >= 0)
    xc, yc = xc[mask], yc[mask]
    if len(xc) == 0 or xc.nunique() < 2 or yc.nunique() < 2:
        return 0.0
    tab = pd.crosstab(xc, yc).to_numpy()
    n = tab.sum()
    if n == 0:
        return 0.0
    row = tab.sum(axis=1, keepdims=True)
    col = tab.sum(axis=0, keepdims=True)
    expected = row @ col / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((tab - expected) ** 2 / np.where(expected == 0, 1, expected))
    r, k = tab.shape
    v = np.sqrt(chi2 / (n * min(r - 1, k - 1))) if min(r - 1, k - 1) > 0 else 0.0
    return float(min(1.0, v))


def _target_association(col: pd.Series, target: pd.Series) -> float:
    if col.dtype.kind in "iufc":
        # Binary target -> point-biserial correlation.  Multiclass target ->
        # correlation ratio (eta), which is invariant to arbitrary class labels.
        c = pd.to_numeric(col, errors="coerce")
        mask = target.notna() & c.notna()
        if mask.sum() < 10 or c[mask].nunique() < 2:
            return 0.0
        values = c[mask].astype(float)
        labels, classes = pd.factorize(target[mask].astype(str), sort=True)
        if len(classes) < 2:
            return 0.0
        if len(classes) == 2:
            r = np.corrcoef(values.to_numpy(), labels)[0, 1]
            return float(abs(r)) if not np.isnan(r) else 0.0

        grand_mean = float(values.mean())
        total = float(((values - grand_mean) ** 2).sum())
        if total <= 0:
            return 0.0
        between = 0.0
        for code in range(len(classes)):
            group = values[labels == code]
            if len(group):
                between += len(group) * (float(group.mean()) - grand_mean) ** 2
        return float(min(1.0, np.sqrt(max(0.0, between / total))))
    # categorical vs target -> Cramer's V
    return _cramers_v(col, target)


def investigate(
    df: pd.DataFrame,
    target_column: str,
    max_rows: int = 500_000,
) -> DatasetProfile:
    if len(df) > max_rows:
        df = df.head(max_rows)

    profile = DatasetProfile(
        row_count=int(len(df)),
        column_count=int(df.shape[1]),
        duplicate_rows=int(df.duplicated().sum()),
        duplicate_rows_pct=float(df.duplicated().mean() * 100),
        target_column=target_column,
    )

    target = df[target_column]
    target_classes = int(target.nunique(dropna=True))
    profile.target_is_binary = target_classes == 2
    profile.target_class_balance = {
        str(k): float(v) for k, v in target.value_counts(normalize=True).head(20).items()
    }
    if target.isna().any():
        profile.warnings.append("Target column contains missing values.")
    if target_classes < 2:
        profile.warnings.append("Target column has fewer than two classes.")
    elif target_classes > 20:
        profile.warnings.append(
            f"Target has {target_classes} classes; LeakLens is intended for classification."
        )

    for colname in df.columns:
        col = df[colname]
        if colname == target_column:
            continue
        dtype = _infer_dtype(col)
        null_count = int(col.isna().sum())
        null_pct = float(null_count / len(df) * 100) if len(df) else 0.0
        n_unique = int(col.nunique())

        cp = ColumnProfile(
            name=colname,
            dtype=dtype,
            null_count=null_count,
            null_pct=null_pct,
            n_unique=n_unique,
            is_numeric=dtype == "numeric",
            is_categorical=dtype == "categorical",
            is_datetime=dtype == "datetime",
            top_values=[str(v) for v in col.value_counts().head(3).index.tolist()],
        )

        if cp.is_numeric:
            num = pd.to_numeric(col, errors="coerce")
            cp.mean = None if num.isna().all() else float(num.mean())
            cp.std = None if num.isna().all() else float(num.std())
            cp.min = None if num.isna().all() else float(num.min())
            cp.max = None if num.isna().all() else float(num.max())

        # association with target (numeric always; categorical only for reasonable cardinality)
        if cp.is_numeric:
            cp.target_association = _target_association(col, target)
        elif cp.n_unique <= MAX_FEATURE_CLASSES and cp.n_unique > 1:
            cp.target_association = _target_association(col, target)

        if cp.null_pct > 50:
            cp.suspicious = True
            cp.suspicious_reason = f"{cp.null_pct:.1f}% missing values"

        profile.columns.append(cp)

    profile.notes.append(f"Profiled {profile.row_count} rows × {profile.column_count} columns")
    return profile
