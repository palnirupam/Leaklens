from types import SimpleNamespace

import pandas as pd
import pytest

from app.agents.hypothesis_planner import HypothesisPlanner
from app.core.config import settings
from app.detectors.dataset_investigator import investigate
from app.detectors.leakage_detectors import DetectionSignals
from app.services.gemini_client import GeminiClient


@pytest.mark.parametrize("response", [
    [], {"hypotheses": None}, {"hypotheses": [None]},
    {"hypotheses": [{"category": []}]},
    {"hypotheses": [{"category": "invented"}]},
    {"hypotheses": [{"category": "target_derived_feature", "affected_features": None}]},
    {"hypotheses": [{"category": "target_derived_feature", "affected_features": "x"}]},
    {"hypotheses": [{"category": "target_derived_feature", "affected_features": ["missing"]}]},
    {"hypotheses": [{"category": "target_derived_feature", "affected_features": ["x"], "title": 12}]},
    {"hypotheses": [{"category": "prediction_time_unavailable", "affected_features": ["x"]}]},
])
def test_invalid_gemini_response_falls_back_without_crashing(response):
    client = SimpleNamespace(is_configured=True, generate_structured=lambda _: response,
                             sanitized_frame_summary=lambda *_: {})
    profile = investigate(pd.DataFrame({"x": range(20), "target": [0, 1] * 10}), "target")
    hypotheses, used = HypothesisPlanner(client).plan("inv_test", profile, DetectionSignals(), [], True)
    assert hypotheses == []
    assert used is False


def test_valid_empty_gemini_response_is_successful():
    client = SimpleNamespace(is_configured=True, generate_structured=lambda _: {"hypotheses": []},
                             sanitized_frame_summary=lambda *_: {})
    profile = investigate(pd.DataFrame({"x": range(20), "target": [0, 1] * 10}), "target")
    hypotheses, used = HypothesisPlanner(client).plan("inv_test", profile, DetectionSignals(), [], True)
    assert hypotheses == []
    assert used is True


def test_cache_separates_models_and_recovers_from_corruption(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "gemini_cache", True)
    client = GeminiClient(api_key="test", model="model-a")
    client.cache = {}
    client._cache_file = tmp_path / "cache.json"
    calls = []

    def generate(**kwargs):
        calls.append(kwargs["model"])
        return SimpleNamespace(text='{"hypotheses": []}')

    client._client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    assert client.generate_structured("prompt") == {"hypotheses": []}
    client._error = "previous error"
    assert client.generate_structured("prompt") == {"hypotheses": []}
    assert client.last_error is None
    assert calls == ["model-a"]
    client.cache[next(iter(client.cache))] = "invalid JSON"
    assert client.generate_structured("prompt") == {"hypotheses": []}
    client.model = "model-b"
    assert client.generate_structured("prompt") == {"hypotheses": []}
    assert calls == ["model-a", "model-a", "model-b"]


def test_provider_non_object_json_returns_failure(monkeypatch):
    monkeypatch.setattr(settings, "gemini_cache", False)
    client = GeminiClient(api_key="test")
    client._client = SimpleNamespace(models=SimpleNamespace(
        generate_content=lambda **_: SimpleNamespace(text='[]'),
    ))
    assert client.generate_structured("prompt", max_retries=1) is None
    assert "JSON object" in client.last_error
