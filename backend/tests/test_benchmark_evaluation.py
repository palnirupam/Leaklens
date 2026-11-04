from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.data.synthetic import SyntheticCase, generate_case
from benchmarks import evaluation
from benchmarks.evaluation import score_approach


def test_root_cause_accuracy_counts_missed_categories():
    case = SimpleNamespace(
        category="target_derived_feature",
        leaked_features=["leak"],
    )

    result = score_approach([case], lambda _: ([], 0, 0.0))

    assert result["recall"] == 0.0
    assert result["root_cause_accuracy"] == 0.0


def test_root_cause_accuracy_unions_same_category_features():
    case = SimpleNamespace(
        category="target_derived_feature",
        leaked_features=["actual_leak"],
    )

    def predict(_):
        return (
            [
                ("target_derived_feature", ["legitimate"]),
                ("target_derived_feature", ["actual_leak"]),
            ],
            2,
            0.1,
        )

    result = score_approach([case], predict)
    assert result["root_cause_accuracy"] == 1.0


def test_clean_has_no_candidate_experiments():
    assert evaluation.all_experiments_count(generate_case("clean", seed=5)) == 0


def test_baseline_includes_repeated_entities():
    case = SyntheticCase(
        name="entities", category="duplicate_contamination", seed=0, target="target",
        entity_column="entity",
        df=pd.DataFrame({"x": range(20), "entity": ["same"] * 20, "target": [0, 1] * 10}),
    )
    assert evaluation.baseline_findings(case) == [("duplicate_contamination", ["entity"])]
    assert evaluation.all_experiments_count(case) == 1


def test_duplicate_injection_preserves_multiclass_labels():
    from benchmarks.real_cases import RealCase, _inject_duplicates

    frame = pd.DataFrame({"x": range(180), "target": [0, 1, 2] * 60})
    case = RealCase("multiclass", frame)
    _inject_duplicates(case, np.random.default_rng(2))
    assert set(case.df["target"]) == {0, 1, 2}
    assert (case.df.groupby("x")["target"].nunique() > 1).any()


def test_gemini_benchmark_skips_without_credentials(monkeypatch):
    monkeypatch.setattr(evaluation, "GeminiClient", lambda: SimpleNamespace(is_configured=False))
    result = evaluation.score_gemini([generate_case("clean")])
    assert result == {"status": "skipped (no GEMINI_API_KEY configured)"}


@pytest.mark.parametrize("valid", [True, False])
def test_gemini_benchmark_calls_provider_and_does_not_score_failures(monkeypatch, valid):
    calls = []

    def generate(prompt):
        calls.append(prompt)
        return {"hypotheses": [{
            "category": "target_derived_feature", "title": "Target-derived feature",
            "affected_features": ["f_derived"],
        }]} if valid else None

    client = SimpleNamespace(is_configured=True, generate_structured=generate,
                             sanitized_frame_summary=lambda *_: {})
    monkeypatch.setattr(evaluation, "GeminiClient", lambda: client)
    result = evaluation.score_gemini([generate_case("target_derived_feature", n_rows=100)])
    assert len(calls) == 1
    assert result["status"] == ("completed" if valid else "failed")
    if valid:
        assert result["recall"] == 1.0
        assert result["mean_experiments"] == 0
    else:
        assert "recall" not in result
