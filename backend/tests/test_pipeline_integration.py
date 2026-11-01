import tempfile
import os

from app.data.synthetic import generate_case
from app.core.models import InvestigationConfig
from app.services.orchestrator import InvestigationOrchestrator


def _run(category: str, seed: int = 3):
    case = generate_case(category, seed=seed)
    orch = InvestigationOrchestrator()
    cfg = InvestigationConfig(
        dataset_path="",  # replaced below
        target_column=case.target,
        timestamp_column=case.timestamp_column,
        entity_column=case.entity_column,
        unavailable_at_prediction=case.unavailable_at_prediction,
        random_seed=seed,
    )
    inv = orch.create(cfg)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "data.csv")
        case.df.to_csv(p, index=False)
        cfg.dataset_path = p
        orch.run(inv)
    return inv


def test_clean_no_findings():
    inv = _run("clean")
    assert inv.status == "completed"
    assert inv.findings == []


def test_target_derived_finding():
    inv = _run("target_derived_feature")
    cats = {f.category for f in inv.findings}
    assert "target_derived_feature" in cats


def test_duplicate_contamination_finding():
    inv = _run("duplicate_contamination")
    cats = {f.category for f in inv.findings}
    assert "duplicate_contamination" in cats


def test_prediction_time_unavailable_finding():
    inv = _run("prediction_time_unavailable")
    cats = {f.category for f in inv.findings}
    assert "prediction_time_unavailable" in cats
