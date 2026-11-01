"""Synthetic dataset generator with injectable, ground-truth leakage cases.

This is used for the demo fallback and for the research benchmark. Every
generated case knows the leak category that was injected and which feature(s)
are genuinely contaminated, so investigation results can be scored.

Categories:
- "clean"      : no injected leakage
- "target_derived_feature": one feature is a noisy copy of the target
- "prediction_time_unavailable": a feature is defined but not available at
                 prediction time (set via unavailable_at_prediction)
- "temporal_leakage": timestamp column present; a feature encodes future info
- "duplicate_contamination": duplicate rows / repeated entities with
                 conflicting labels (train/test contamination)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


LEAK_CATEGORIES = (
    "clean",
    "target_derived_feature",
    "prediction_time_unavailable",
    "temporal_leakage",
    "duplicate_contamination",
)


@dataclass
class SyntheticCase:
    """A generated dataset plus its ground truth."""

    name: str
    df: pd.DataFrame
    target: str
    category: str
    seed: int
    leaked_features: list[str] = field(default_factory=list)
    entity_column: Optional[str] = None
    timestamp_column: Optional[str] = None
    #: extra columns that are "unavailable at prediction time" (set for the
    #: investigation config when this case is used)
    unavailable_at_prediction: list[str] = field(default_factory=list)
    description: str = ""


def _sig(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_case(
    category: str,
    n_rows: int = 600,
    seed: int = 0,
    n_features: int = 6,
) -> SyntheticCase:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_features))
    # latent true score based on first 4 real features (kept meaningful)
    score = 0.6 * X[:, 0] - 0.4 * X[:, 1] + 0.3 * X[:, 2] - 0.2 * X[:, 3]
    p = _sig(score)
    y = (rng.uniform(size=n_rows) < p).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["target"] = y

    base = SyntheticCase(
        name=f"{category}_seed{seed}",
        df=df,
        target="target",
        category=category,
        seed=seed,
        description=f"synthetic {category} case (n={n_rows})",
    )

    if category == "clean":
        return base

    if category == "target_derived_feature":
        # f_derived is a noisy linear copy of the target -> leaks target info
        derived = rng.normal(size=n_rows) + 2.0 * y.astype(float)
        df["f_derived"] = derived
        base.leaked_features = ["f_derived"]

    elif category == "prediction_time_unavailable":
        # a valid feature that is simply not known at prediction time;
        # not a leakage of target info but a deployment hazard.
        df["call_duration"] = rng.uniform(1, 300, size=n_rows)
        base.unavailable_at_prediction = ["call_duration"]
        base.leaked_features = ["call_duration"]

    elif category == "temporal_leakage":
        # Build an autocorrelated target so that *future* outcomes are
        # predictive of the current outcome. This makes random splitting leak
        # future information, while chronological splitting does not.
        rng2 = np.random.default_rng(seed + 999)
        phi = 0.95
        z = np.zeros(n_rows)
        noise = rng2.normal(0, 1, n_rows)
        for t in range(1, n_rows):
            z[t] = phi * z[t - 1] + noise[t]
        # target depends on the strongly autocorrelated latent z
        gain = 3.0
        p2 = _sig(gain * z)
        y2 = (rng2.uniform(size=n_rows) < p2).astype(int)
        df["target"] = y2
        # future hint uses the target of the *next* time step (a peek ahead)
        future_y = np.roll(y2, -1)
        future_y[-1] = y2[-1]  # no future row beyond the end
        df["future_outcome_hint"] = future_y
        ts = pd.Timestamp("2023-01-01") + pd.to_timedelta(np.arange(n_rows) * 86400, unit="s")
        df["timestamp"] = ts
        base.timestamp_column = "timestamp"
        base.leaked_features = ["future_outcome_hint"]

    elif category == "duplicate_contamination":
        # introduce repeated rows with conflicting labels (classic contamination).
        half = n_rows // 2
        dup_idx = rng.choice(np.arange(half), size=half // 3, replace=False)
        dup_rows = df.iloc[dup_idx].copy()
        # flip ~40% of the duplicate labels to create conflicting-label pairs
        flip = rng.random(len(dup_rows)) < 0.4
        dup_rows.loc[flip, "target"] = 1 - dup_rows.loc[flip, "target"]
        df = pd.concat([df, dup_rows], ignore_index=True)
        base.df = df
        base.leaked_features = []

    else:
        raise ValueError(f"unknown category {category!r}")

    base.df = df
    return base


def generate_benchmark_cases(
    per_category: int = 6,
    seed_offset: int = 100,
    n_rows: int = 500,
) -> list[SyntheticCase]:
    cases: list[SyntheticCase] = []
    for category in LEAK_CATEGORIES:
        for i in range(per_category):
            seed = seed_offset + i
            cases.append(generate_case(category, n_rows=n_rows, seed=seed))
    return cases
