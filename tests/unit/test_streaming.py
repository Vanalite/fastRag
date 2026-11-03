"""Streaming answers: service events, the SSE endpoint, and Claude's streaming call."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from fastrag import DocumentIn, Principal, RAGService, Role, Settings
from fastrag.api import create_app
from fastrag.domain import EVERYONE
from fastrag.errors import GenerationError
from fastrag.generation import ClaudeGenerator, Generation
from fastrag.security.auth import ApiKeyStore, Authenticator

ADMIN = Principal("acme", "root", roles=frozenset({Role.ADMIN}))
ALICE = Principal("acme", "alice")
QUESTION = "How long is visitor parking free?"


def make(generator=None, **overrides) -> RAGService:
    settings = Settings.for_testing(
        **{"rate_limit_per_minute": 0, "user_rate_limit_per_minute": 0, **overrides}
    )
    service = RAGService.from_settings(settings, generator=generator)
    service.ingest(
        ADMIN,
        DocumentIn("p.md", "Visitor parking is free for three hours.", frozenset({EVERYONE})),
    )
    return service


def sse_events(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


class ScriptedStream:
    """A generator that streams fixed pieces, optionally failing partway."""

    name = "scripted"

    def __init__(self, pieces, fail_after=None):
        self.pieces, self.fail_after = pieces, fail_after

    def generate(self, question, sources):
        return Generation(text="".join(self.pieces))

    def stream(self, question, sources):
        for i, piece in enumerate(self.pieces):
            if self.fail_after is not None and i == self.fail_after:
                raise GenerationError("upstream model failed: secret detail")
            yield piece
        yield Generation(text="".join(self.pieces))


def test_service_streams_deltas_then_a_validated_answer():
    service = make(ScriptedStream(["Three hours ", "[1]", " and also [9]."]))
    events = list(service.ask_stream(ALICE, QUESTION))
    assert [e.type for e in events] == ["delta", "delta", "delta", "answer"]
    answer = events[-1].answer
    assert "[9]" not in answer.text and [c.index for c in answer.citations] == [1]
    entry = service.audit.sink.read()[-1]
    assert entry["outcome"] == "success" and entry["details"]["streamed"] is True
    assert entry["details"]["invalid_citations"] == [9]
    # The validated answer was cached; a repeat is served whole.
    again = list(service.ask_stream(ALICE, QUESTION))
    assert [e.type for e in again] == ["answer"] and again[0].answer.cached


def test_abandoned_stream_is_audited_as_cancelled():
    service = make(ScriptedStream(["a ", "b ", "c [1]"]))
    events = service.ask_stream(ALICE, QUESTION)
    next(events)
    events.close()  # the client went away
    assert service.audit.sink.read()[-1]["outcome"] == "cancelled"


def _client(service):
    keys = ApiKeyStore()
    key = keys.create(ALICE)
    return TestClient(create_app(service, Authenticator(api_keys=keys))), key


def test_sse_endpoint_streams_and_ends_with_the_answer():
    client, key = _client(make(ScriptedStream(["Parking is free ", "for three hours [1]."])))
    resp = client.post("/v1/ask/stream", headers={"X-API-Key": key}, json={"question": QUESTION})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = sse_events(resp.text)
    assert [name for name, _ in events] == ["delta", "delta", "answer"]
    assert events[-1][1]["grounded"] is True
    assert events[-1][1]["citations"][0]["source_id"] == "p.md"


def test_errors_before_streaming_are_ordinary_http_errors():
    client, key = _client(make(ScriptedStream(["x"]), rate_limit_per_minute=1, rate_limit_burst=1))
    assert client.post("/v1/ask/stream", json={"question": QUESTION}).status_code == 401
    assert (
        client.post("/v1/ask/stream", headers={"X-API-Key": key}, json={"question": ""}).status_code
        == 422
    )
    client.post("/v1/ask/stream", headers={"X-API-Key": key}, json={"question": QUESTION})
    limited = client.post("/v1/ask/stream", headers={"X-API-Key": key}, json={"question": "x y"})
    assert limited.status_code == 429


def test_failure_mid_stream_is_reported_in_band_without_details():
    service = make(ScriptedStream(["Parking ", "is free"], fail_after=1))
    client, key = _client(service)
    resp = client.post("/v1/ask/stream", headers={"X-API-Key": key}, json={"question": QUESTION})
    events = sse_events(resp.text)
    assert [name for name, _ in events] == ["delta", "error"]
    assert events[-1][1]["error"]["message"] == "the answer generator failed; try again later"
    assert "secret detail" not in resp.text
    assert events[-1][1]["request_id"] == resp.headers["X-Request-ID"]
    entry = service.audit.sink.read()[-1]
    assert entry["outcome"] == "generation_error"
    assert entry["request_id"] == resp.headers["X-Request-ID"]


class FakeStreamingClient:
    def __init__(self, deltas, stop_reason="end_turn"):
        self.calls = []
        final = SimpleNamespace(
            stop_reason=stop_reason,
            content=[SimpleNamespace(type="text", text="".join(deltas))],
            usage=SimpleNamespace(input_tokens=10, output_tokens=3),
        )

        class _Stream:
            text_stream = iter(deltas)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get_final_message(self):
                return final

        def stream(**kwargs):
            self.calls.append(kwargs)
            return _Stream()

        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=stream))


def test_claude_stream_uses_the_same_request_and_checks_the_final_message():
    client = FakeStreamingClient(["Three ", "hours [1]."])
    items = list(ClaudeGenerator(client=client).stream("q", []))
    assert items[:2] == ["Three ", "hours [1]."]
    assert isinstance(items[-1], Generation) and items[-1].usage.output_tokens == 3
    (call,) = client.calls
    assert call["fallbacks"] == "default" and call["model"] == "claude-opus-5-5"
    refused = FakeStreamingClient(["partial"], stop_reason="refusal")
    with pytest.raises(GenerationError):
        list(ClaudeGenerator(client=refused).stream("q", []))


def test_operations_emit_spans_without_sensitive_attributes():
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    service = make(ScriptedStream(["Three hours [1]."]))
    list(service.ask_stream(ALICE, QUESTION))
    spans = {s.name: s for s in exporter.get_finished_spans()}
    ask = spans["fastrag.ask"]
    assert ask.attributes["fastrag.outcome"] == "success"
    assert ask.attributes["fastrag.tenant"] == "acme"
    assert ask.attributes["fastrag.shown_chunks"] == 1
    assert ask.attributes["fastrag.retrieval_ms"] >= 0
    assert "fastrag.generation_ms" in ask.attributes
    flat = json.dumps({k: str(v) for k, v in ask.attributes.items()})
    assert "alice" not in flat and "parking" not in flat.lower()
    assert "_timings" not in json.dumps(service.audit.sink.read()[-1])


def _drive(service, stop_after: str | None):
    """Run the SSE body like Starlette would, optionally dropping the client early."""
    import asyncio

    from fastrag.api.app import _event_stream

    events = service.ask_stream(ALICE, QUESTION)
    first = next(events)

    async def run():
        stream = _event_stream(first, events, "req-42")
        async for chunk in stream:
            if stop_after and chunk.startswith(f"event: {stop_after}"):
                await stream.aclose()  # the client went away here
                return

    asyncio.run(run())
    return service.audit.sink.read()[-1]


def test_client_disconnect_mid_answer_is_audited_as_cancelled_immediately():
    entry = _drive(make(ScriptedStream(["a ", "b ", "c [1]"])), stop_after="delta")
    assert entry["outcome"] == "cancelled"
    assert entry["request_id"] == "req-42"


def test_disconnect_right_after_the_answer_still_audits_success():
    entry = _drive(make(ScriptedStream(["Three hours [1]."])), stop_after="answer")
    assert entry["outcome"] == "success"
    assert entry["request_id"] == "req-42"
