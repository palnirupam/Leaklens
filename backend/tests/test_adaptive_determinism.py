from app.data.synthetic import generate_case
from app.core.models import InvestigationConfig
from app.services.orchestrator import InvestigationOrchestrator


def _run_on(category: str, seed: int = 7):
    case = generate_case(category, seed=seed)
    orch = InvestigationOrchestrator()
    cfg = InvestigationConfig(
        dataset_path=case.name,  # path unused (we pass df through run? no)
        target_column=case.target,
        timestamp_column=case.timestamp_column,
        entity_column=case.entity_column,
        unavailable_at_prediction=case.unavailable_at_prediction,
        random_seed=seed,
    )
    inv = orch.create(cfg)
    # Save case df to temp and run with a real path
    import tempfile, os, pandas as pd
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "data.csv")
        case.df.to_csv(p, index=False)
        cfg.dataset_path = p
        orch.run(inv)
    return inv


def _decision_path(inv) -> list[str]:
    # (experiment kind, feature, interpretation) reflects the actual decision order
    return [f"{r.name}:{r.config.get('feature', '')}:{r.interpretation}" for r in inv.results]


def test_executor_deterministic_on_target_derived():
    a = _run_on("target_derived_feature")
    b = _run_on("target_derived_feature")
    assert _decision_path(a) == _decision_path(b)
    cat_a = sorted(f.category for f in a.findings)
    cat_b = sorted(f.category for f in b.findings)
    assert cat_a == cat_b
    assert a.status == "completed"


def test_executor_deterministic_on_temporal():
    a = _run_on("temporal_leakage", seed=9)
    b = _run_on("temporal_leakage", seed=9)
    assert _decision_path(a) == _decision_path(b)
    assert sorted(f.category for f in a.findings) == sorted(f.category for f in b.findings)


def test_equal_priority_ignores_random_ids_and_experiment_ids_are_unique(monkeypatch):
    import pandas as pd
    from app.agents import experiment_executor
    from app.core.models import CandidateHypothesis, ExperimentResult, Investigation

    def experiment(kind, df, target, params, seed):
        return ExperimentResult(
            experiment_id=params["experiment_id"], hypothesis_id=params["hypothesis_id"],
            name=kind, category=params["category"], status="completed", config=params,
        )

    monkeypatch.setattr(experiment_executor, "run_experiment", experiment)

    def run(ids):
        inv = Investigation("inv_test", InvestigationConfig("", "target"))
        inv.hypotheses = [CandidateHypothesis(
            id=hyp_id, investigation_id=inv.id, category="target_derived_feature",
            title=feature, description="", rationale="", affected_features=[feature], confidence=0.3,
        ) for hyp_id, feature in zip(ids, ("first", "second"))]
        experiment_executor.AdaptiveExperimentExecutor().execute(inv, pd.DataFrame(), "target", [])
        return inv

    a = run(("hyp_zzzzzzzz", "hyp_aaaaaaaa"))
    b = run(("hyp_aaaaaaaa", "hyp_zzzzzzzz"))
    assert _decision_path(a) == _decision_path(b)
    assert len(a.results) == 4
    assert len({r.experiment_id for r in a.results}) == 4
