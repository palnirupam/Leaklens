"""Investigation orchestrator.

Runs the full agent pipeline for one investigation:
  1. Dataset Investigator   -> profile
  2. Static code analysis    -> AST findings
  3. Leakage signals         -> DetectionSignals
  4. Hypothesis Planner      -> candidate hypotheses (Gemini optional)
  5. Adaptive Experiment Executor -> results, updated hypotheses
  6. Evidence Verifier       -> findings
  7. Summary
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from app.agents.evidence_verifier import EvidenceVerifier
from app.agents.experiment_executor import AdaptiveExperimentExecutor
from app.agents.hypothesis_planner import HypothesisPlanner
from app.core.models import Investigation, InvestigationConfig
from app.detectors.ast_analyzer import analyze_script
from app.detectors.dataset_investigator import investigate
from app.detectors.leakage_detectors import detect_signals
from app.services.gemini_client import GeminiClient


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class InvestigationOrchestrator:
    def __init__(self, gemini: Optional[GeminiClient] = None) -> None:
        self.gemini = gemini or GeminiClient()
        self.planner = HypothesisPlanner(self.gemini)
        self.verifier = EvidenceVerifier()
        #: optional callback(persisted active investigation) called at stage changes
        self.on_progress = None

    def _persist(self, inv: Investigation) -> None:
        if self.on_progress is not None:
            try:
                self.on_progress(inv)
            except Exception:  # noqa: BLE001
                pass

    def create(
        self,
        config: InvestigationConfig,
        investigation_id: Optional[str] = None,
    ) -> Investigation:
        return Investigation(
            id=investigation_id or f"inv_{uuid.uuid4().hex[:12]}",
            config=config,
            status="queued",
            created_at=_utcnow(),
        )

    def run(self, investigation: Investigation) -> Investigation:
        cfg = investigation.config
        investigation.status = "running"
        investigation.stage = "dataset_profiling"
        investigation.current_agent = "DatasetInvestigator"
        investigation.progress = 0.05

        df = pd.read_csv(cfg.dataset_path)

        # 1) dataset profile
        profile = investigate(df, cfg.target_column)
        investigation.profile = profile
        investigation.progress = 0.2
        self._persist(investigation)

        # 2) static code analysis
        code_findings = []
        if cfg.script_path:
            investigation.stage = "static_code_analysis"
            investigation.current_agent = "LeakageHypothesisPlanner"
            try:
                source = open(cfg.script_path, encoding="utf-8").read()
                analysis = analyze_script(source, cfg.script_path)
                code_findings = analysis.findings
            except OSError:
                code_findings = []
        investigation.progress = 0.3

        # 3) detection signals
        investigation.stage = "signal_detection"
        signals = detect_signals(
            df,
            profile,
            cfg.target_column,
            timestamp_column=cfg.timestamp_column,
            entity_column=cfg.entity_column,
            unavailable_at_prediction=cfg.unavailable_at_prediction,
            seed=cfg.random_seed,
        )
        investigation.progress = 0.4

        # 4) hypothesis planning
        investigation.stage = "hypothesis_planning"
        investigation.current_agent = "LeakageHypothesisPlanner"
        hypotheses, gemini_used = self.planner.plan(
            investigation.id, profile, signals, code_findings,
            gemini_enabled=cfg.gemini_enabled,
        )
        investigation.hypotheses = hypotheses
        investigation.gemini_used = gemini_used
        investigation.progress = 0.55
        self._persist(investigation)

        # 5) adaptive experiments
        investigation.stage = "adaptive_experiments"
        investigation.current_agent = "AdaptiveExperimentExecutor"
        executor = AdaptiveExperimentExecutor(
            seed=cfg.random_seed,
            timestamp_column=cfg.timestamp_column,
            entity_column=cfg.entity_column,
        )
        executor.execute(investigation, df, cfg.target_column, code_findings)
        investigation.progress = 0.8
        self._persist(investigation)

        # 6) evidence verification
        investigation.stage = "evidence_verification"
        investigation.current_agent = "EvidenceVerificationAgent"
        self.verifier.verify(investigation)
        investigation.progress = 0.95

        # 7) summary
        investigation.stage = "completed"
        investigation.current_agent = ""
        investigation.status = "completed"
        investigation.progress = 1.0
        investigation.finished_at = _utcnow()
        investigation.summary = _build_summary(investigation)
        return investigation


def _build_summary(inv: Investigation) -> str:
    n_findings = len(inv.findings)
    confirmed = sum(1 for f in inv.findings if f.evidence_status in ("confirmed", "strong"))
    suspicious = sum(1 for f in inv.findings if f.evidence_status in ("potential", "suspicious"))
    return (
        f"Investigation complete: {n_findings} finding(s), "
        f"{confirmed} supported by experiments, {suspicious} flagged for manual review."
    )
