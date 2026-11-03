"""HTTP contract: error envelope, request ids, auth schemes.

Security behaviour (401/403, isolation) is covered in tests/security.
"""

import pytest
from fastapi.testclient import TestClient

from fastrag import Principal, RAGService, Role, Settings
from fastrag.api import create_app
from fastrag.security.auth import ApiKeyStore, Authenticator


@pytest.fixture
def client_and_key():
    service = RAGService.from_settings(Settings.for_testing())
    keys = ApiKeyStore()
    key = keys.create(Principal("acme", "ops", roles=frozenset({Role.ADMIN})))
    with TestClient(create_app(service, Authenticator(api_keys=keys))) as client:
        yield client, key


def test_validation_error_does_not_echo_input(client_and_key):
    client, key = client_and_key
    secret_text = "TOP-SECRET-BODY " * 10
    resp = client.post(
        "/v1/documents",
        headers={"X-API-Key": key},
        json={"source_id": "a", "text": secret_text, "acl": []},
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["type"] == "ValidationError"
    assert body["request_id"] == resp.headers["X-Request-ID"]
    assert "TOP-SECRET" not in resp.text


def test_unknown_route_uses_envelope(client_and_key):
    client, _ = client_and_key
    resp = client.get("/nope", headers={"X-Request-ID": "req-123"})
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "HTTPError"
    assert resp.json()["request_id"] == "req-123"
    assert resp.headers["X-Request-ID"] == "req-123"


def test_unhandled_error_is_generic_500(client_and_key, monkeypatch):
    client, key = client_and_key
    service = client.app.state.service

    def boom(*args, **kwargs):
        raise RuntimeError("database password is hunter2")

    monkeypatch.setattr(service.retriever, "retrieve", boom)
    resp = client.post("/v1/search", headers={"X-API-Key": key}, json={"question": "hi"})
    assert resp.status_code == 500
    assert resp.json()["error"]["message"] == "internal server error"
    assert "hunter2" not in resp.text
    assert resp.headers["X-Request-ID"]


def test_openapi_declares_security_schemes(client_and_key):
    client, _ = client_and_key
    schemes = client.get("/openapi.json").json()["components"]["securitySchemes"]
    assert {"APIKeyHeader", "HTTPBearer"} <= set(schemes)


def test_metrics_disabled_by_default(client_and_key):
    client, _ = client_and_key
    assert client.get("/metrics").status_code == 404


def test_metadata_is_bounded(client_and_key):
    client, key = client_and_key
    resp = client.post(
        "/v1/documents",
        headers={"X-API-Key": key},
        json={"source_id": "a", "text": "x", "acl": ["tenant:*"], "metadata": {"k": "v" * 20_000}},
    )
    assert resp.status_code == 422
