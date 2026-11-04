from __future__ import annotations

import pandas as pd
import pytest
from fastapi import HTTPException

from app.api.investigations import _validate_config_columns, _validate_target
from app.core.models import InvestigationConfig
from app.services.orchestrator import InvestigationOrchestrator


def test_target_validation_accepts_string_classes():
    df = pd.DataFrame({"x": [1, 2], "target": ["no", "yes"]})
    _validate_target(df, "target")


def test_target_validation_rejects_missing_values():
    df = pd.DataFrame({"x": [1, 2, 3], "target": [0, None, 1]})
    with pytest.raises(HTTPException, match="missing values"):
        _validate_target(df, "target")


def test_optional_columns_must_exist():
    df = pd.DataFrame({"x": [1, 2], "target": [0, 1]})
    with pytest.raises(HTTPException, match="timestamp"):
        _validate_config_columns(df, "target", "when", None, [])
    with pytest.raises(HTTPException, match="not found"):
        _validate_config_columns(df, "target", None, None, ["late_feature"])


def test_orchestrator_preserves_caller_investigation_id():
    cfg = InvestigationConfig(dataset_path="unused.csv", target_column="target")
    inv = InvestigationOrchestrator().create(cfg, investigation_id="inv_fixed")
    assert inv.id == "inv_fixed"
