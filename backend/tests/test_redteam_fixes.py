"""Regression tests for the two red-team findings:

1. XOR-style interaction leak is now *surfaced* (as a review flag) instead of
   being completely invisible.
2. A strong but *legitimate* feature is no longer auto-labelled as a
   "confirmed critical leak".
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd

from app.core.models import Investigation, InvestigationConfig
from app.services.orchestrator import InvestigationOrchestrator


def _run_on_df(df: pd.DataFrame, target: str = "target") -> Investigation:
    orch = InvestigationOrchestrator()
    cfg = InvestigationConfig(
        dataset_path="",
        target_column=target,
        random_seed=42,
        source_filename="redteam",
    )
    inv = orch.create(cfg)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "data.csv")
        df.to_csv(p, index=False)
        cfg.dataset_path = p
        orch.run(inv)
    return inv


def test_xor_interaction_is_surfaced_as_review_flag():
    rng = np.random.default_rng(7)
    n = 1200
    f1 = rng.integers(0, 2, n)
    f2 = rng.integers(0, 2, n)
    y = f1 ^ f2  # 100% informative joint signal, 0% marginal
    df = pd.DataFrame({"f1": f1, "f2": f2, "target": y})

    inv = _run_on_df(df)
    inter = [f for f in inv.findings if f.category == "feature_interaction_leakage"]
    # The interaction is now visible as a review flag...
    assert inter, "expected a feature_interaction finding for XOR data"
    # ...but never auto-confirmed as leakage.
    for f in inter:
        assert f.evidence_status == "suspicious", f.evidence_status
        assert f.severity != "critical", f.severity


def test_strong_legitimate_feature_not_confirmed_critical():
    rng = np.random.default_rng(3)
    n = 800
    noise = rng.normal(size=(n, 4))
    y = (rng.normal(size=n) > 0).astype(int)
    df = pd.DataFrame(noise, columns=[f"noise{i}" for i in range(4)])
    df["tumor_size"] = rng.normal(size=n) + 3.0 * y  # strong, legitimate signal
    df["target"] = y

    inv = _run_on_df(df)
    td = [f for f in inv.findings
          if f.category == "target_derived_feature" and f.affected_features == ["tumor_size"]]
    assert td, "expected a finding for the strong feature"
    f = td[0]
    # Downgraded from confirmed/critical when there is no mechanism-level proof:
    assert f.evidence_status in ("potential", "strong", "suspicious"), f.evidence_status
    assert f.severity != "critical", f.severity
    # Advice is conditional, not an unconditional deletion command.
    assert "NOT proof" in f.correction
