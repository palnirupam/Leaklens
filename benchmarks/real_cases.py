"""Real-dataset leakage benchmark cases for LeakLens.

Unlike the fully synthetic benchmark, these cases are built on REAL data:
  - Wine (UCI, via scikit-learn)          - small, numeric, 3-class
  - Breast Cancer Wisconsin (UCI, via SK)  - numeric, binary
  - Bank Marketing (UCI demo CSV)          - mixed categorical/numeric, binary

Each case is a *controlled experiment*: we start from a real dataset and inject
exactly one documented leakage mechanism (target-derived feature, duplicates,
temporal/future leakage, or a prediction-time-unavailable feature). The clean
case has no injection and should produce no findings.

The dataset objects expose the same fields as the synthetic ``SyntheticCase``
(``df``, ``target``, ``timestamp_column``, ``entity_column``,
``unavailable_at_prediction``, ``category``, ``leaked_features``) so the shared
``benchmarks.evaluation`` scoring works unchanged.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RealCase:
    name: str
    df: pd.DataFrame
    target: str = "target"
    timestamp_column: str | None = None
    entity_column: str | None = None
    unavailable_at_prediction: list[str] = field(default_factory=list)
    category: str = "clean"
    leaked_features: list[str] = field(default_factory=list)


def _sig(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_real_datasets() -> dict[str, tuple[pd.DataFrame, str]]:
    """Load real tabular datasets. Returns {name: (df, target_column)}.

    sklearn's bundled datasets are used so the benchmark works fully offline.
    Bank Marketing is added if the demo CSV is present.
    """
    out: dict[str, tuple[pd.DataFrame, str]] = {}

    from sklearn.datasets import load_wine, load_breast_cancer

    wine = load_wine()
    out["wine"] = (
        pd.DataFrame(wine.data, columns=[f"f{i}" for i in range(wine.data.shape[1])]),
        "target",
    )
    out["wine"][0]["target"] = wine.target

    bc = load_breast_cancer()
    out["breast"] = (
        pd.DataFrame(bc.data, columns=[f"f{i}" for i in range(bc.data.shape[1])]),
        "target",
    )
    out["breast"][0]["target"] = bc.target

    bank_csv = Path(__file__).resolve().parent.parent / "data" / "demo" / "bank_marketing.csv"
    if bank_csv.exists():
        bank = pd.read_csv(bank_csv)
        # `duration` is a documented post-call feature; drop it so the clean
        # case is genuinely leak-free, then re-encode target as 0/1.
        if "duration" in bank.columns:
            bank = bank.drop(columns=["duration"])
        y = (bank["y"].astype(str).str.strip().str.lower().eq("yes")).astype(int)
        bank = bank.drop(columns=["y"])
        bank = pd.get_dummies(bank, drop_first=True)
        bank["target"] = y
        # The raw UCI dataset contains many exact duplicate rows *and* feature
        # profiles that map to different targets. Remove both so the "clean"
        # control case is genuinely contamination-free (the injected duplicate
        # case re-adds contamination deliberately).
        bank = bank.drop_duplicates()
        feat_cols = [c for c in bank.columns if c != "target"]
        multi = bank.set_index(feat_cols)
        uniq_targets = multi.groupby(level=list(range(len(feat_cols))))["target"].nunique()
        safe_idx = uniq_targets[uniq_targets == 1].index
        bank = multi.loc[safe_idx].reset_index()
        out["bank"] = (bank, "target")

    return out


def _base(df: pd.DataFrame, category: str, seed: int) -> RealCase:
    rng = np.random.default_rng(seed)
    return RealCase(
        name=f"{category}",
        df=df.copy(),
        target="target",
        category=category,
        leaked_features=[],
    ), rng


def _inject_target_derived(case: RealCase, rng: np.random.Generator) -> None:
    y = case.df["target"].to_numpy(dtype=float)
    n = len(case.df)
    case.df["f_derived"] = rng.normal(size=n) + 2.0 * y
    case.leaked_features = ["f_derived"]


def _inject_duplicates(case: RealCase, rng: np.random.Generator) -> None:
    df = case.df
    n = len(df)
    half = n // 2
    dup_idx = rng.choice(np.arange(half), size=max(1, half // 3), replace=False)
    dup_rows = df.iloc[dup_idx].copy()
    flip = rng.random(len(dup_rows)) < 0.4
    dup_rows.loc[flip, "target"] = 1 - dup_rows.loc[flip, "target"]
    case.df = pd.concat([df, dup_rows], ignore_index=True)
    case.leaked_features = []


def _inject_temporal(case: RealCase, rng: np.random.Generator) -> None:
    df = case.df
    n = len(df)
    phi = 0.95
    z = np.zeros(n)
    noise = rng.normal(0, 1, n)
    for t in range(1, n):
        z[t] = phi * z[t - 1] + noise[t]
    y = (rng.uniform(size=n) < _sig(3.0 * z)).astype(int)
    df["target"] = y
    future_y = np.roll(y, -1)
    future_y[-1] = y[-1]
    df["future_outcome_hint"] = future_y
    case.timestamp_column = "timestamp"
    case.df["timestamp"] = pd.Timestamp("2023-01-01") + pd.to_timedelta(np.arange(n) * 86400, unit="s")
    case.leaked_features = ["future_outcome_hint"]


def _inject_unavailable(case: RealCase, rng: np.random.Generator) -> None:
    n = len(case.df)
    case.df["call_duration"] = rng.uniform(1, 300, size=n)
    case.unavailable_at_prediction = ["call_duration"]
    case.leaked_features = ["call_duration"]


INJECTIONS = {
    "target_derived_feature": _inject_target_derived,
    "duplicate_contamination": _inject_duplicates,
    "temporal_leakage": _inject_temporal,
    "prediction_time_unavailable": _inject_unavailable,
}


def build_real_cases() -> list[RealCase]:
    """Build a deterministic suite of real-data cases.

    Domain (per dataset): a clean control, plus 2 injected-leak scenarios.
    """
    datasets = load_real_datasets()
    plan: dict[str, list[str]] = {
        "wine": ["clean", "target_derived_feature", "duplicate_contamination"],
        "breast": ["clean", "target_derived_feature", "prediction_time_unavailable"],
        "bank": ["clean", "temporal_leakage", "duplicate_contamination"],
    }

    cases: list[RealCase] = []
    seed = 0
    for ds_name, (df, _target) in datasets.items():
        for i, category in enumerate(plan.get(ds_name, ["clean"])):
            seed += 1
            case, rng = _base(df, category, seed)
            case.name = f"{ds_name}_{category}"
            if category != "clean":
                INJECTIONS[category](case, rng)
            cases.append(case)

    return cases
