"""API security helpers.

LeakLens is a local research tool. Two layers of protection:

1. **Loopback binding guard** — unless an API key is configured, the server
   refuses to start on a non-loopback interface. This makes it impossible to
   accidentally expose the unauthenticated API to the network.
2. **Shared-token auth** — when ``settings.api_key`` is set, every ``/api``
   route (except ``/api/health``) requires the ``X-API-Key`` header to match it.
"""
from __future__ import annotations

from typing import Optional

from fastapi import Depends, Header, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import settings

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def assert_safe_binding(host: str, api_key: str | None = None) -> None:
    """Raise unless the deployment is safe (loopback OR authentication enabled)."""
    key = api_key if api_key is not None else settings.api_key
    host_norm = (host or "").strip().lower()
    loopback = host_norm in LOOPBACK_HOSTS or host_norm.startswith("127.")
    if not loopback and not key:
        raise RuntimeError(
            "Refusing to bind to a non-loopback interface without authentication. "
            "Set API_KEY in backend/.env (or run on 127.0.0.1) before exposing "
            "LeakLens to the network."
        )


_bearer = HTTPBearer(auto_error=False)


def require_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    x_api_key: Optional[str] = Header(default=None),
) -> None:
    """FastAPI dependency enforcing the shared token when configured."""
    key = settings.api_key
    if not key:
        return  # auth disabled; loopback guard prevents non-local exposure
    supplied = credentials.credentials if credentials is not None else x_api_key
    if supplied is None or supplied != key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
