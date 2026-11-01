"""Gemini API client.

Safety rules:
- The API key is only ever read from backend/.env (never hardcoded).
- This service lives only in the backend; it is never exposed to React.
- By default only sanitized metadata is sent; the caller decides what data to
  include. `ALLOW_DATA_TO_GEMINI` gates richer context.
- Responses are validated and parsed as JSON; on any failure the caller gets
  ``None`` and the app proceeds in offline (rule-based) mode.
- Result caching avoids repeated calls.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

from app.core.config import settings


class GeminiClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        self._api_key = api_key if api_key is not None else settings.gemini_api_key
        self.model = model if model is not None else settings.gemini_model
        self._client = None
        self._error: Optional[str] = None
        self.cache: dict[str, Any] = {}
        self._cache_file = settings.data_dir / "gemini_cache.json"
        self._load_cache()

    # -- setup ---------------------------------------------------------------
    def _ensure_client(self):
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self._api_key)
            except Exception as exc:  # noqa: BLE001
                self._error = f"failed to init Gemini: {exc!r}"
        return self._client

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    @property
    def last_error(self) -> Optional[str]:
        return self._error

    # -- caching ---------------------------------------------------------------
    def _load_cache(self) -> None:
        try:
            if self._cache_file.exists():
                self.cache = json.loads(self._cache_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            self.cache = {}

    def _persist_cache(self) -> None:
        try:
            self._cache_file.write_text(json.dumps(self.cache), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    # -- main call -------------------------------------------------------------
    def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        max_retries: int | None = None,
    ) -> Optional[dict[str, Any]]:
        """Return a parsed JSON dict or None on any failure."""
        if not self.is_configured:
            self._error = "GEMINI_API_KEY not set"
            return None

        key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if settings.gemini_cache and key in self.cache:
            cached = self.cache[key]
            return json.loads(cached) if isinstance(cached, str) else cached

        client = self._ensure_client()
        if client is None:
            return None

        retries = max_retries if max_retries is not None else settings.gemini_max_retries
        config: dict[str, Any] = {"response_mime_type": "application/json"}
        if schema:
            config["response_schema"] = schema

        for attempt in range(retries):
            try:
                resp = client.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
                text = resp.text
                parsed = json.loads(text)
                if settings.gemini_cache:
                    self.cache[key] = text
                    self._persist_cache()
                return parsed
            except Exception as exc:  # noqa: BLE001
                self._error = str(exc)
                # rate limit / quota handling -> back off
                msg = str(exc).lower()
                if "429" in msg or "rate" in msg or "quota" in msg or "resource exhausted" in msg:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                if attempt < retries - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                return None
        return None

    # -- convenience ------------------------------------------------------------
    def sanitized_frame_summary(self, profile, signals) -> dict[str, Any]:
        """Build a small, non-privacy-invasive summary to send to Gemini."""
        cols = [
            {
                "name": c.name,
                "dtype": c.dtype,
                "null_pct": round(c.null_pct, 2),
                "n_unique": c.n_unique,
                "association_with_target": round(c.target_association, 3),
            }
            for c in profile.columns
        ]
        return {
            "target": profile.target_column,
            "target_is_binary": profile.target_is_binary,
            "target_balance": {k: round(v, 3) for k, v in profile.target_class_balance.items()},
            "row_count": profile.row_count,
            "duplicate_rows": signals.duplicate_rows,
            "conflicting_duplicates": signals.conflicting_duplicates,
            "entity_conflicts": signals.entity_conflicts,
            "high_association_features": [
                {"feature": f, "association": round(a, 3)} for f, a in signals.high_association_features
            ],
            "temporal": signals.temporal,
            "prediction_time_unavailable": signals.prediction_time_unavailable,
            "columns": cols,
        }
