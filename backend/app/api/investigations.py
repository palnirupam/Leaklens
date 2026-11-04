"""Investigation API endpoints."""
from __future__ import annotations

import io
import json
import threading
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from app.core.config import settings
from app.core.models import InvestigationConfig
from app.core.security import require_api_key
from app.core.store import get_store
from app.schemas import InvestigationResponse
from app.services.gemini_client import GeminiClient
from app.services.orchestrator import InvestigationOrchestrator

# Every investigation endpoint requires the shared token when one is configured.
router = APIRouter(tags=["investigations"], dependencies=[Depends(require_api_key)])

_tasks: dict[str, threading.Thread] = {}
_orchestrator = InvestigationOrchestrator(GeminiClient())
_store = get_store()
_orchestrator.on_progress = _store.save


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_df(content: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc!r}") from exc
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV is empty.")
    if len(df) > settings.max_csv_rows:
        raise HTTPException(
            status_code=413,
            detail=f"CSV has too many rows (maximum {settings.max_csv_rows:,}).",
        )
    return df


def _validate_target(df: pd.DataFrame, target: str) -> None:
    if target not in df.columns:
        raise HTTPException(
            status_code=400, detail=f"Target column '{target}' not found in CSV."
        )
    if df[target].isna().any():
        raise HTTPException(status_code=400, detail="Target column contains missing values.")
    if df[target].nunique(dropna=True) < 2:
        raise HTTPException(
            status_code=400, detail="Target column must contain at least two classes."
        )


def _validate_config_columns(
    df: pd.DataFrame,
    target: str,
    timestamp_column: Optional[str],
    entity_column: Optional[str],
    unavailable: list[str],
) -> None:
    configured = {
        "timestamp": timestamp_column,
        "entity": entity_column,
    }
    for label, column in configured.items():
        if column and column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Configured {label} column '{column}' not found in CSV.",
            )
        if column and column == target:
            raise HTTPException(
                status_code=400,
                detail=f"The target column cannot also be the {label} column.",
            )
    missing = [column for column in unavailable if column not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Unavailable-at-prediction columns not found: {', '.join(missing)}.",
        )
    if target in unavailable:
        raise HTTPException(
            status_code=400,
            detail="The target column cannot be listed as unavailable at prediction time.",
        )


def _to_response(inv) -> InvestigationResponse:
    return InvestigationResponse(
        id=inv.id,
        status=inv.status,
        stage=inv.stage,
        progress=round(inv.progress, 2),
        current_agent=inv.current_agent,
        created_at=inv.created_at,
        finished_at=inv.finished_at,
        error=inv.error,
        gemini_used=inv.gemini_used,
        source_filename=inv.config.source_filename,
        target_column=inv.config.target_column,
        finding_count=len(inv.findings),
        experiment_count=len(inv.results),
        summary=inv.summary,
    )


def _parse_list(v: Optional[str]) -> list[str]:
    if not v:
        return []
    v = v.strip()
    if not v:
        return []
    try:
        parsed = json.loads(v)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:  # noqa: BLE001
        pass
    return [x.strip() for x in v.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------

@router.post("/investigations", response_model=InvestigationResponse, status_code=202)
async def create_investigation(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    script: Optional[UploadFile] = File(None),
    timestamp_column: Optional[str] = Form(None),
    entity_column: Optional[str] = Form(None),
    unavailable_at_prediction: Optional[str] = Form(None),
    gemini_enabled: bool = Form(False),
    random_seed: int = Form(42),
) -> InvestigationResponse:
    # size limits
    max_bytes = settings.max_upload_mb * 1024 * 1024
    content = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail="File too large.")

    df = _load_df(content)
    _validate_target(df, target_column)

    unavailable = list(dict.fromkeys(_parse_list(unavailable_at_prediction)))
    timestamp_column = timestamp_column or None
    entity_column = entity_column or None
    _validate_config_columns(
        df,
        target_column,
        timestamp_column,
        entity_column,
        unavailable,
    )

    script_content: Optional[bytes] = None
    if script is not None and script.filename:
        script_content = await script.read()
        if len(script_content) > 512 * 1024:
            raise HTTPException(status_code=413, detail="Script file too large.")

    inv_id = f"inv_{uuid.uuid4().hex[:12]}"
    inv_dir = settings.uploads_dir / inv_id
    inv_dir.mkdir(parents=True, exist_ok=True)

    src_name = file.filename or "dataset.csv"
    ds_path = inv_dir / "dataset.csv"
    ds_path.write_bytes(content)

    script_path: Optional[Path] = None
    if script_content is not None:
        script_path = inv_dir / "training_script.py"
        script_path.write_bytes(script_content)

    cfg = InvestigationConfig(
        dataset_path=str(ds_path),
        target_column=target_column,
        timestamp_column=timestamp_column,
        entity_column=entity_column,
        script_path=str(script_path) if script_path else None,
        unavailable_at_prediction=unavailable,
        gemini_enabled=gemini_enabled,
        random_seed=random_seed,
        source_filename=src_name,
    )

    inv = _orchestrator.create(cfg, investigation_id=inv_id)
    _store.save(inv)

    def _run() -> None:
        try:
            _orchestrator.run(inv)
        except Exception as exc:  # noqa: BLE001
            inv.status = "failed"
            inv.error = str(exc)
            inv.stage = "failed"
        finally:
            _store.save(inv)
            _tasks.pop(inv.id, None)

    t = threading.Thread(target=_run, daemon=True)
    _tasks[inv.id] = t
    t.start()

    return _to_response(inv)


@router.get("/investigations", response_model=list[InvestigationResponse])
def list_investigations(limit: int = 50) -> list[InvestigationResponse]:
    return [_to_response(inv) for inv in _store.list(limit=min(limit, 200))]


@router.get("/investigations/{investigation_id}", response_model=InvestigationResponse)
def get_investigation(investigation_id: str) -> InvestigationResponse:
    inv = _store.get(investigation_id)
    if inv is None:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    return _to_response(inv)


@router.get("/investigations/{investigation_id}/detail")
def get_investigation_detail(investigation_id: str) -> dict:
    inv = _store.get(investigation_id)
    if inv is None:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    from app.reporting.reports import investigation_to_dict
    return investigation_to_dict(inv)


@router.get("/investigations/{investigation_id}/findings")
def get_findings(investigation_id: str) -> list[dict]:
    inv = _store.get(investigation_id)
    if inv is None:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    from app.reporting.reports import investigation_to_dict
    return investigation_to_dict(inv)["findings"]


@router.get("/investigations/{investigation_id}/report")
def get_report(investigation_id: str, format: str = "json") -> Response:
    inv = _store.get(investigation_id)
    if inv is None:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    from app.reporting.reports import render_html, render_json

    if format == "html":
        return Response(
            content=render_html(inv),
            media_type="text/html",
            headers={"Content-Disposition": f'attachment; filename="investigation_{investigation_id}.html"'},
        )
    return Response(
        content=render_json(inv),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="investigation_{investigation_id}.json"'},
    )


@router.get("/stats")
def stats() -> dict:
    invs = _store.list(limit=1000)
    total = len(invs)
    confirmed = sum(1 for i in invs for f in i.findings if f.evidence_status in ("confirmed", "strong"))
    potential = sum(1 for i in invs for f in i.findings if f.evidence_status in ("potential", "suspicious"))
    durations = []
    for i in invs:
        if i.finished_at and i.created_at:
            from datetime import datetime
            try:
                t0 = datetime.fromisoformat(i.created_at)
                t1 = datetime.fromisoformat(i.finished_at)
                durations.append((t1 - t0).total_seconds())
            except Exception:  # noqa: BLE001
                pass
    avg_duration = (sum(durations) / len(durations)) if durations else 0.0
    return {
        "total_investigations": total,
        "confirmed_findings": confirmed,
        "potential_findings": potential,
        "avg_duration_sec": round(avg_duration, 1),
        "recent": [_to_response(i).model_dump() for i in invs[:8]],
    }
