from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml import pipeline


def test_cross_val_fits_preprocessing_inside_each_fold(monkeypatch):
    fit_sizes: list[int] = []

    class RecordingPreprocessor:
        def fit_transform(self, frame):
            fit_sizes.append(len(frame))
            return frame.to_numpy(dtype=float)

        def transform(self, frame):
            return frame.to_numpy(dtype=float)

    monkeypatch.setattr(
        pipeline,
        "build_preprocessor",
        lambda features, df: RecordingPreprocessor(),
    )
    df = pd.DataFrame(
        {
            "x": np.arange(12, dtype=float),
            "target": [0, 1] * 6,
        }
    )

    result = pipeline.cross_val_auc(df, ["x"], "target", cv=3)

    assert result["cv"] == 3
    assert fit_sizes == [8, 8, 8]


def test_string_binary_target_is_supported():
    rng = np.random.default_rng(4)
    x = rng.normal(size=90)
    df = pd.DataFrame(
        {
            "x": x,
            "target": np.where(x > 0, "yes", "no"),
        }
    )
    result = pipeline.cross_val_auc(df, ["x"], "target")
    assert result["n_classes"] == 2
    assert result["cv_auc_mean"] > 0.9


def test_multiclass_target_is_supported():
    rng = np.random.default_rng(8)
    target = np.repeat(["low", "mid", "high"], 30)
    centers = {"low": -3.0, "mid": 0.0, "high": 3.0}
    x = np.array([centers[label] for label in target]) + rng.normal(0, 0.2, len(target))
    df = pd.DataFrame({"x": x, "target": target})

    result = pipeline.cross_val_auc(df, ["x"], "target")

    assert result["n_classes"] == 3
    assert result["cv_auc_mean"] > 0.9


def test_duplicate_feature_names_are_deduplicated():
    rng = np.random.default_rng(11)
    x = rng.normal(size=60)
    df = pd.DataFrame({"x": x, "target": (x > 0).astype(int)})
    result = pipeline.cross_val_auc(df, ["x", "x"], "target")
    assert result["cv_auc_mean"] > 0.9
