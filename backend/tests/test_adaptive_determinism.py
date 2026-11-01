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
