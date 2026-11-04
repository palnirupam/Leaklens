from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.experiments import registry


def test_availability_check_actually_removes_feature(monkeypatch):
    calls: list[tuple[str, ...]] = []

    def fake_auc(df, features, target, seed=42):
        calls.append(tuple(features))
        return {"cv_auc_mean": 0.8 if "late_feature" in features else 0.6}

    monkeypatch.setattr(registry, "cross_val_auc", fake_auc)
    df = pd.DataFrame(
        {
            "base": [0, 1, 0, 1],
            "late_feature": [1, 1, 0, 0],
            "target": [0, 1, 0, 1],
        }
    )

    result = registry._run_availability(
        df,
        "target",
        {"feature": "late_feature"},
        seed=42,
    )

    assert calls == [("base", "late_feature"), ("base",)]
    assert result.metrics["with_feature_auc"] == 0.8
    assert result.metrics["without_feature_auc"] == 0.6


def test_permutation_test_shuffles_feature_not_target(monkeypatch):
    original = pd.DataFrame(
        {
            "base": np.arange(20),
            "suspicious": np.arange(20),
            "target": [0, 1] * 10,
        }
    )
    seen: list[pd.DataFrame] = []

    def fake_auc(df, features, target, seed=42):
        seen.append(df.copy())
        unchanged = np.array_equal(df["suspicious"].to_numpy(), original["suspicious"].to_numpy())
        return {"cv_auc_mean": 0.9 if unchanged else 0.6}

    monkeypatch.setattr(registry, "cross_val_auc", fake_auc)
    result = registry._run_permutation(
        original,
        "target",
        {"feature": "suspicious"},
        seed=7,
    )

    assert len(seen) == 2
    assert all(frame["target"].equals(original["target"]) for frame in seen)
    assert not seen[0]["suspicious"].equals(original["suspicious"])
    assert result.interpretation == "feature_dependent"
    assert result.evidence["auc_drop"] > 0.1


@pytest.mark.parametrize("random_auc,has_finding", [(0.9, True), (0.65, False)])
def test_temporal_gap_reaches_executor_and_verifier(monkeypatch, random_auc, has_finding):
    from app.agents.evidence_verifier import EvidenceVerifier
    from app.agents.experiment_executor import AdaptiveExperimentExecutor
    from app.core.models import CandidateHypothesis, Investigation, InvestigationConfig

    def evaluate(df, features, target, **kwargs):
        return {"roc_auc": 0.6 if kwargs.get("chronological") else random_auc}

    monkeypatch.setattr(registry, "evaluate_split", evaluate)
    inv = Investigation("inv_test", InvestigationConfig("", "target", timestamp_column="time"))
    inv.hypotheses = [CandidateHypothesis(
        id="hyp_time", investigation_id=inv.id, category="temporal_leakage",
        title="Temporal risk", description="", rationale="", confidence=0.3,
    )]
    frame = pd.DataFrame({"x": [0, 1], "target": [0, 1], "time": [1, 2]})
    AdaptiveExperimentExecutor(timestamp_column="time").execute(inv, frame, "target", [])

    result = inv.results[0]
    assert result.evidence["gap_auc"] == pytest.approx(random_auc - 0.6)
    assert inv.hypotheses[0].confidence == pytest.approx(0.55 if has_finding else 0.15)
    assert bool(EvidenceVerifier().verify(inv)) is has_finding


@pytest.mark.parametrize("code_findings", [[], [{"line": 1, "pattern": "preprocessing_before_split"}]])
def test_static_finding_requires_code_evidence_and_has_valid_severity(code_findings):
    from app.agents.evidence_verifier import EvidenceVerifier
    from app.core.models import CandidateHypothesis, Investigation, InvestigationConfig, SEVERITY_LEVELS

    inv = Investigation("inv_test", InvestigationConfig("", "target"))
    inv.hypotheses = [CandidateHypothesis(
        id="hyp_static", investigation_id=inv.id, category="preprocessing_leakage",
        title="Preprocessing risk", description="", rationale="", source="gemini",
    )]
    inv.results = [registry.run_experiment(
        "preprocessing_ast_review", pd.DataFrame(), "target",
        {"hypothesis_id": "hyp_static", "code_findings": code_findings},
    )]
    findings = EvidenceVerifier().verify(inv)
    assert bool(findings) is bool(code_findings)
    assert all(f.severity in SEVERITY_LEVELS for f in findings)


@pytest.mark.parametrize("configured", [True, False])
def test_availability_uses_configuration_even_when_budget_is_exhausted(configured):
    from app.agents.evidence_verifier import EvidenceVerifier
    from app.agents.experiment_executor import AdaptiveExperimentExecutor
    from app.core.models import CandidateHypothesis, Investigation, InvestigationConfig

    inv = Investigation("inv_test", InvestigationConfig(
        "", "target", unavailable_at_prediction=["late"] if configured else [],
    ))
    inv.hypotheses = [CandidateHypothesis(
        id="hyp_late", investigation_id=inv.id, category="prediction_time_unavailable",
        title="Late feature", description="", rationale="", affected_features=["late"], source="gemini",
    )]
    AdaptiveExperimentExecutor(max_experiments=0).execute(inv, pd.DataFrame(), "target", [])
    findings = EvidenceVerifier().verify(inv)
    assert inv.results == []
    assert bool(findings) is configured
    assert all(f.verification_status == "supported_by_configuration" for f in findings)
