"""Asynchronous ingestion jobs: background ingest, progress, visibility, limits."""

import threading

import fakeredis
import pytest
from fastapi.testclient import TestClient

from fastrag import DocumentIn, Principal, RAGService, Role, Settings
from fastrag.api import create_app
from fastrag.domain import EVERYONE
from fastrag.errors import NotFound, PermissionDenied, RateLimited
from fastrag.observability.logging import request_id_var
from fastrag.security.auth import ApiKeyStore, Authenticator

INGESTER = Principal("acme", "ingrid", roles=frozenset({Role.INGESTER}))
ADMIN = Principal("acme", "root", roles=frozenset({Role.ADMIN}))
READER = Principal("acme", "rita")


def make(**overrides) -> RAGService:
    defaults = {"rate_limit_per_minute": 0, "user_rate_limit_per_minute": 0}
    return RAGService.from_settings(Settings.for_testing(**{**defaults, **overrides}))


def docs(n: int, acl=frozenset({EVERYONE})) -> list[DocumentIn]:
    return [DocumentIn(f"d{i}.md", f"Policy number {i}: rule {i} applies.", acl) for i in range(n)]


def test_job_ingests_in_the_background_and_reports_progress():
    service = make()
    token = request_id_var.set("req-job-1")
    try:
        job = service.submit_ingest_job(INGESTER, docs(5))
    finally:
        request_id_var.reset(token)
    assert job.status == "queued" and job.counts == {"pending": 5}
    service.jobs.wait(job.id)
    done = service.ingest_job(INGESTER, job.id)
    assert done.status == "succeeded" and done.counts == {"created": 5}
    assert all(item.doc_id and item.version == 1 for item in done.items)
    assert len(service.store.list_documents("acme")) == 5
    # Every document is audited as the submitter, tied to the job and its request.
    entries = [e for e in service.audit.sink.read() if e["action"] == "ingest"]
    assert len(entries) == 5
    assert {e["user_id"] for e in entries} == {"ingrid"}
    assert {e["details"]["job_id"] for e in entries} == {job.id}
    assert {e["request_id"] for e in entries} == {"req-job-1"}


def test_one_bad_document_makes_the_job_partial_not_failed():
    service = make()
    batch = docs(2) + [DocumentIn("bad.md", "text", frozenset({"not-a-token"}))]
    job = service.submit_ingest_job(INGESTER, batch)
    service.jobs.wait(job.id)
    done = service.ingest_job(INGESTER, job.id)
    assert done.status == "partial" and done.counts == {"created": 2, "failed": 1}
    assert done.items[2].error.startswith("InvalidInput")


def test_permissions_still_apply_per_document():
    service = make()
    service.ingest(ADMIN, DocumentIn("locked.md", "Admin only.", frozenset({EVERYONE})))
    service.store._conn.execute("UPDATE documents SET editors = '[]'")
    job = service.submit_ingest_job(
        INGESTER, [DocumentIn("locked.md", "Hijack.", frozenset({EVERYONE}))]
    )
    service.jobs.wait(job.id)
    item = service.ingest_job(INGESTER, job.id).items[0]
    assert item.status == "failed" and item.error.startswith("PermissionDenied")


def test_jobs_are_visible_to_their_submitter_and_admins_only():
    service = make()
    job = service.submit_ingest_job(INGESTER, docs(1))
    service.jobs.wait(job.id)
    assert service.ingest_job(ADMIN, job.id).id == job.id
    colleague = Principal("acme", "olaf", roles=frozenset({Role.INGESTER}))
    stranger = Principal("globex", "ingrid", roles=frozenset({Role.ADMIN}))
    for other in (colleague, stranger):
        with pytest.raises(NotFound):
            service.ingest_job(other, job.id)
    with pytest.raises(NotFound):
        service.ingest_job(INGESTER, "no-such-job")


def test_readers_cannot_submit_and_tenants_have_a_job_cap():
    service = make(max_active_jobs_per_tenant=1, ingest_workers=1)
    with pytest.raises(PermissionDenied):
        service.submit_ingest_job(READER, docs(1))
    gate = threading.Event()
    real = service._ingest_for_job

    def slow(*args):
        gate.wait(5)
        return real(*args)

    service.jobs._ingest = slow
    first = service.submit_ingest_job(INGESTER, docs(1))
    with pytest.raises(RateLimited):
        service.submit_ingest_job(INGESTER, docs(1))
    other_tenant = Principal("globex", "g", roles=frozenset({Role.INGESTER}))
    service.submit_ingest_job(other_tenant, docs(1))  # caps are per tenant
    gate.set()
    service.jobs.wait(first.id)
    service.submit_ingest_job(INGESTER, docs(1))  # a slot is free again


def test_shutdown_cancels_jobs_that_have_not_started():
    service = make(ingest_workers=1)
    gate = threading.Event()
    real = service._ingest_for_job
    service.jobs._ingest = lambda *a: (gate.wait(5), real(*a))[1]
    running = service.submit_ingest_job(INGESTER, docs(1))
    queued = service.submit_ingest_job(INGESTER, docs(1))
    service.close()
    gate.set()
    service.jobs.wait(running.id)
    assert service.jobs.store.get(queued.id).status == "cancelled"


def test_http_submit_returns_202_and_a_pollable_location():
    service = make()
    keys = ApiKeyStore()
    key = keys.create(INGESTER)
    client = TestClient(create_app(service, Authenticator(api_keys=keys)))
    body = {
        "documents": [
            {"source_id": f"h{i}.md", "text": f"Handbook part {i}.", "acl": ["tenant:*"]}
            for i in range(3)
        ]
    }
    resp = client.post("/v1/ingest-jobs", headers={"X-API-Key": key}, json=body)
    assert resp.status_code == 202
    location = resp.headers["Location"]
    service.jobs.wait(resp.json()["id"])
    polled = client.get(location, headers={"X-API-Key": key}).json()
    assert polled["status"] == "succeeded" and polled["counts"] == {"created": 3}
    huge = {
        "documents": [
            {"source_id": f"x{i}", "text": "x" * 1_900_000, "acl": ["tenant:*"]} for i in range(11)
        ]
    }
    assert client.post("/v1/ingest-jobs", headers={"X-API-Key": key}, json=huge).status_code == 422


def test_job_status_is_shared_across_replicas_through_redis():
    redis = fakeredis.FakeRedis()
    a = RAGService.from_settings(
        Settings.for_testing(rate_limit_per_minute=0, user_rate_limit_per_minute=0), redis=redis
    )
    b = RAGService.from_settings(
        Settings.for_testing(rate_limit_per_minute=0, user_rate_limit_per_minute=0), redis=redis
    )
    job = a.submit_ingest_job(INGESTER, docs(2))
    a.jobs.wait(job.id)
    assert b.ingest_job(INGESTER, job.id).status == "succeeded"
    assert b.jobs.store.active_count("acme") == 0


def test_jobs_abandoned_by_a_dead_worker_stop_counting_and_report_failure():
    from fastrag.jobs import IngestJob, JobItem, MemoryJobStore, RedisJobStore

    now = [1_000_000.0]
    redis = fakeredis.FakeRedis()
    for store in (
        MemoryJobStore(clock=lambda: now[0]),
        RedisJobStore(redis, clock=lambda: now[0]),
    ):
        service = make(max_active_jobs_per_tenant=1)
        service.jobs.store = store
        service.jobs._clock = lambda: now[0]
        # A job left "running" by a process that crashed an hour ago.
        orphan = IngestJob(
            id="orphan", tenant_id="acme", user_id="ingrid", status="running",
            items=[JobItem("a.md")], created_at=now[0] - 3600, updated_at=now[0] - 3600,
        )  # fmt: skip
        store.save(orphan)
        assert store.active_count("acme") == 0  # it no longer blocks the tenant
        reported = service.ingest_job(INGESTER, "orphan")
        assert reported.status == "failed" and reported.items[0].error.startswith("worker stopped")
        job = service.submit_ingest_job(INGESTER, docs(1))
        service.jobs.wait(job.id)


def test_shutdown_stops_a_running_job_between_documents():
    service = make(ingest_workers=1)
    started, gate = threading.Event(), threading.Event()
    real = service._ingest_for_job

    def slow(*args):
        started.set()
        gate.wait(5)
        return real(*args)

    service.jobs._ingest = slow
    job = service.submit_ingest_job(INGESTER, docs(3))
    started.wait(5)
    closer = threading.Thread(target=service.jobs.shutdown)
    closer.start()
    gate.set()
    closer.join(10)
    final = service.jobs.store.get(job.id)
    assert final.status == "cancelled"
    assert final.counts == {"created": 1, "cancelled": 2}  # finished work is kept


def test_global_queue_caps_bound_memory():
    service = make(max_queued_jobs=1, ingest_workers=1)
    gate = threading.Event()
    real = service._ingest_for_job
    service.jobs._ingest = lambda *a: (gate.wait(5), real(*a))[1]
    first = service.submit_ingest_job(INGESTER, docs(1))
    other_tenant = Principal("globex", "g", roles=frozenset({Role.INGESTER}))
    with pytest.raises(RateLimited):  # this replica's queue is full, whatever the tenant
        service.submit_ingest_job(other_tenant, docs(1))
    gate.set()
    service.jobs.wait(first.id)
    big = make(max_queued_chars=50)
    with pytest.raises(RateLimited):
        big.submit_ingest_job(INGESTER, [DocumentIn("x.md", "x" * 100, frozenset({EVERYONE}))])


def test_background_documents_are_paced_per_tenant():
    service = make()
    calls = []
    service._pace_job = lambda tenant, stop: calls.append(tenant)
    service.jobs._pace = service._pace_job
    job = service.submit_ingest_job(INGESTER, docs(3))
    service.jobs.wait(job.id)
    assert calls == ["acme"] * 3
