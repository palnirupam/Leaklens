"""Domain entities for LeakLens.

These are lightweight, framework-free data holders. Persistence happens in
:mod:`app.core.store`; API serialization happens in :mod:`app.schemas`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Static enums / consts
# ---------------------------------------------------------------------------

INVESTIGATION_STATUS = ("queued", "running", "completed", "failed", "cancelled")

LEAKAGE_CATEGORIES = (
    "target_derived_feature",   # A feature is computed from/contains target info
    "prediction_time_availability",  # feature unavailable at prediction time (not leak)
    "train_test_contamination",  # duplicates / entity overlap across splits
    "preprocessing_leakage",    # fit-before-split transforms
    "temporal_leakage",         # future information via random split
    "feature_interaction_leakage",  # strong *joint* signal, weak marginals (review)
    "correlation_suspicious",   # high predictive power, needs manual review
    "none_detected",            # no leak found by supported checks
)

SEVERITY_LEVELS = ("info", "low", "medium", "high", "critical")

EVIDENCE_STATUS = ("confirmed", "strong", "potential", "suspicious", "inconclusive", "none")

AGENT_NAMES = (
    "DatasetInvestigator",
    "LeakageHypothesisPlanner",
    "AdaptiveExperimentExecutor",
    "EvidenceVerificationAgent",
)


@dataclass
class InvestigationConfig:
    """User-supplied configuration for an investigation."""

    dataset_path: str
    target_column: str
    timestamp_column: Optional[str] = None
    entity_column: Optional[str] = None
    script_path: Optional[str] = None
    #: features unavailable at prediction time (e.g. call duration)
    unavailable_at_prediction: list[str] = field(default_factory=list)
    gemini_enabled: bool = False
    random_seed: int = 42
    #: metadata about the original upload (for display)
    source_filename: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    null_count: int
    null_pct: float
    n_unique: int
    is_numeric: bool
    is_categorical: bool
    is_datetime: bool
    suspicious: bool = False
    suspicious_reason: str = ""
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    top_values: list[Any] = field(default_factory=list)
    #: feature-target association strength (Cramer's V for categorical, |r| for numeric)
    target_association: float = 0.0


@dataclass
class DatasetProfile:
    row_count: int
    column_count: int
    duplicate_rows: int
    duplicate_rows_pct: float
    columns: list[ColumnProfile] = field(default_factory=list)
    target_column: str = ""
    target_class_balance: dict[str, float] = field(default_factory=dict)
    target_is_binary: bool = True
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def column(self, name: str) -> Optional[ColumnProfile]:
        for c in self.columns:
            if c.name == name:
                return c
        return None


@dataclass
class CandidateHypothesis:
    id: str
    investigation_id: str
    category: str
    title: str
    description: str
    rationale: str
    affected_features: list[str] = field(default_factory=list)
    code_paths: list[str] = field(default_factory=list)
    source: str = "rule"  # rule | gemini
    confidence: float = 0.0  # prior belief 0..1
    status: str = "open"  # open | investigating | confirmed | rejected
    experiments: list[str] = field(default_factory=list)


@dataclass
class ExperimentResult:
    experiment_id: str
    hypothesis_id: str
    name: str
    category: str
    status: str = "pending"  # pending | running | completed | failed | inconclusive
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    interpretation: str = ""
    cost_ms: float = 0.0
    error: Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class Finding:
    id: str
    investigation_id: str
    category: str
    severity: str
    evidence_status: str
    title: str
    description: str
    affected_features: list[str] = field(default_factory=list)
    code_paths: list[str] = field(default_factory=list)
    supporting_experiments: list[str] = field(default_factory=list)
    correction: str = ""
    verification_status: str = "unverified"
    verified_by: str = ""


@dataclass
class Investigation:
    id: str
    config: InvestigationConfig
    status: str = "queued"
    stage: str = "queued"
    current_agent: str = ""
    progress: float = 0.0  # 0..1
    profile: Optional[DatasetProfile] = None
    hypotheses: list[CandidateHypothesis] = field(default_factory=list)
    results: list[ExperimentResult] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    error: Optional[str] = None
    created_at: str = ""
    finished_at: Optional[str] = None
    gemini_used: bool = False
    summary: str = ""
