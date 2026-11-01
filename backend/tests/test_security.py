from fastapi.testclient import TestClient

from app.core.security import assert_safe_binding


def test_loopback_binding_allowed_without_key():
    # loopback hosts are fine even with no API key
    assert_safe_binding("127.0.0.1", api_key="")
    assert_safe_binding("localhost", api_key="")


def test_non_loopback_refused_without_key():
    import pytest
    with pytest.raises(RuntimeError, match="Refusing to bind"):
        assert_safe_binding("0.0.0.0", api_key="")


def test_non_loopback_allowed_with_key():
    assert_safe_binding("0.0.0.0", api_key="s3cret")


def test_api_requires_token_when_configured(monkeypatch):
    from app.main import app
    from app.core.config import settings
    monkeypatch.setattr(settings, "api_key", "s3cret")

    client = TestClient(app)
    # unauthenticated -> 401
    r = client.get("/api/stats")
    assert r.status_code == 401
    # wrong token -> 401
    r = client.get("/api/stats", headers={"X-API-Key": "wrong"})
    assert r.status_code == 401
    # correct token -> 200
    r = client.get("/api/stats", headers={"X-API-Key": "s3cret"})
    assert r.status_code == 200


def test_health_stays_open(monkeypatch):
    from app.main import app
    from app.core.config import settings
    monkeypatch.setattr(settings, "api_key", "s3cret")
    client = TestClient(app)
    r = client.get("/api/health")
    assert r.status_code == 200
