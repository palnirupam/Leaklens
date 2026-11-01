"""Pydantic schemas for the HTTP API."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    gemini_configured: bool


class InvestigationCreateRequest(BaseModel):
    target_column: str
    timestamp_column: Optional[str] = None
    entity_column: Optional[str] = None
    unavailable_at_prediction: list[str] = Field(default_factory=list)
    gemini_enabled: bool = False
    random_seed: int = 42


class InvestigationResponse(BaseModel):
    id: str
    status: str
    stage: str
    progress: float
    current_agent: str
    created_at: str
    finished_at: Optional[str] = None
    error: Optional[str] = None
    gemini_used: bool
    source_filename: str
    target_column: str
    finding_count: int = 0
    experiment_count: int = 0
    summary: str = ""


class StageUpdate(BaseModel):
    state: dict[str, Any] = Field(default_factory=dict)


class ReportRequest(BaseModel):
    format: str = "json"  # json | html
