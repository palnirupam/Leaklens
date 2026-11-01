"""SQLite-backed persistence for investigations.

Each investigation is stored as a JSON row plus lightweight indexed columns
for listing. Keeps the whole object graph (config, profile, hypotheses,
experiments, findings) together.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .models import DatasetProfile, Finding, Investigation, CandidateHypothesis, ExperimentResult
from .models import InvestigationConfig


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(o: Any) -> Any:
    import dataclasses
    if dataclasses.is_dataclass(o):
        return o.__dict__
    return str(o)


class Store:
    """Thin persistence layer; one table holding serialized investigations."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create()

    def _create(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS investigations (
                id TEXT PRIMARY KEY,
                status TEXT,
                created_at TEXT,
                finished_at TEXT,
                updated_at TEXT,
                display_title TEXT,
                payload TEXT
            )
            """
        )
        self._conn.commit()

    # -- helpers ------------------------------------------------------------
    def _row_to_investigation(self, row: sqlite3.Row) -> Investigation:
        payload = json.loads(row["payload"])
        return from_payload(payload)

    # -- write ---------------------------------------------------------------
    def save(self, inv: Investigation) -> None:
        now = _utcnow()
        self._conn.execute(
            """
            INSERT INTO investigations (id, status, created_at, finished_at, updated_at, display_title, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status=excluded.status,
                finished_at=excluded.finished_at,
                updated_at=excluded.updated_at,
                display_title=excluded.display_title,
                payload=excluded.payload
            """,
            (
                inv.id,
                inv.status,
                inv.created_at,
                inv.finished_at,
                now,
                _display_title(inv),
                json.dumps(inv.__dict__, default=_json_default),
            ),
        )
        self._conn.commit()

    # -- read / list ----------------------------------------------------------
    def get(self, investigation_id: str) -> Optional[Investigation]:
        cur = self._conn.execute("SELECT * FROM investigations WHERE id=?", (investigation_id,))
        row = cur.fetchone()
        return self._row_to_investigation(row) if row else None

    def list(self, limit: int = 50) -> list[Investigation]:
        cur = self._conn.execute(
            "SELECT * FROM investigations ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        return [self._row_to_investigation(r) for r in cur.fetchall()]

    def count(self, status: Optional[str] = None) -> int:
        if status:
            cur = self._conn.execute("SELECT COUNT(*) FROM investigations WHERE status=?", (status,))
        else:
            cur = self._conn.execute("SELECT COUNT(*) FROM investigations")
        return int(cur.fetchone()[0])


def _display_title(inv: Investigation) -> str:
    src = inv.config.source_filename or Path(inv.config.dataset_path).name or "investigation"
    return f"{src} → {inv.config.target_column}"


# ---------------------------------------------------------------------------
# (De)serialization
# ---------------------------------------------------------------------------

def from_payload(payload: dict[str, Any]) -> Investigation:
    cfg = InvestigationConfig(**payload["config"])
    inv = Investigation(id=payload["id"], config=cfg)
    inv.status = payload.get("status", "queued")
    inv.stage = payload.get("stage", "queued")
    inv.current_agent = payload.get("current_agent", "")
    inv.progress = float(payload.get("progress", 0.0))
    inv.created_at = payload.get("created_at", "")
    inv.finished_at = payload.get("finished_at")
    inv.error = payload.get("error")
    inv.gemini_used = bool(payload.get("gemini_used", False))
    inv.summary = payload.get("summary", "")
    if payload.get("profile"):
        inv.profile = _profile_from_dict(payload["profile"])
    inv.hypotheses = [_hypothesis_from_dict(h) for h in payload.get("hypotheses", [])]
    inv.results = [_experiment_from_dict(x) for x in payload.get("results", [])]
    inv.findings = [_finding_from_dict(f) for f in payload.get("findings", [])]
    return inv


def _profile_from_dict(d: dict[str, Any]) -> DatasetProfile:
    from .models import ColumnProfile
    cols = []
    for c in d.get("columns", []):
        if isinstance(c, dict):
            # tolerate rows without an explicit list
            cols.append(ColumnProfile(**{k: v for k, v in c.items() if k in ColumnProfile.__dataclass_fields__}))
        elif hasattr(c, "name"):
            cols.append(ColumnProfile(**c.__dict__))
    p = DatasetProfile(
        row_count=d.get("row_count", 0),
        column_count=d.get("column_count", 0),
        duplicate_rows=d.get("duplicate_rows", 0),
        duplicate_rows_pct=d.get("duplicate_rows_pct", 0.0),
        columns=cols,
        target_column=d.get("target_column", ""),
        target_class_balance=d.get("target_class_balance", {}),
        target_is_binary=d.get("target_is_binary", True),
        notes=list(d.get("notes", [])),
        warnings=list(d.get("warnings", [])),
    )
    return p


def _hypothesis_from_dict(d: dict[str, Any]) -> CandidateHypothesis:
    return CandidateHypothesis(**{k: v for k, v in d.items() if k in CandidateHypothesis.__dataclass_fields__})


def _experiment_from_dict(d: dict[str, Any]) -> ExperimentResult:
    return ExperimentResult(**{k: v for k, v in d.items() if k in ExperimentResult.__dataclass_fields__})


def _finding_from_dict(d: dict[str, Any]) -> Finding:
    return Finding(**{k: v for k, v in d.items() if k in Finding.__dataclass_fields__})


# singleton convenience (used by API layer)
def get_store(db_path: Path | str | None = None) -> Store:
    from .config import settings
    return Store(db_path or settings.db_path)
