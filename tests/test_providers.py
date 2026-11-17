"""Provider clients against the documented request/response shapes (mocked HTTP)."""

import json

import httpx
import pytest
import respx
from pydantic import SecretStr

from fastrag.config import Settings
from fastrag.ingest.ocr import VlmOcr
from fastrag.providers.embeddings import OpenAIEmbedder, TeiEmbedder
from fastrag.providers.http import ProviderError
from fastrag.providers.llm import build_llm, build_llms
from fastrag.providers.reranker import CohereReranker, JinaReranker, TeiReranker


def _s(**kw):
    return Settings(embedding_dim=3, **kw)


@respx.mock
async def test_tei_embedder_batches_and_normalizes():
    route = respx.post("http://tei/embed").mock(
        side_effect=lambda req: httpx.Response(200, json=[[3.0, 4.0, 0.0]] * len(json.loads(req.content)["inputs"]))
    )
    emb = TeiEmbedder(_s(embedding_url="http://tei", embedding_batch_size=2))
    vecs = await emb.embed_documents(["a", "b", "c"])
    assert route.call_count == 2 and vecs[0] == pytest.approx([0.6, 0.8, 0.0])
    assert json.loads(route.calls[0].request.content) == {"inputs": ["a", "b"], "normalize": True, "truncate": True}


@respx.mock
async def test_embedder_rejects_wrong_dimension():
    respx.post("http://tei/embed").mock(return_value=httpx.Response(200, json=[[1.0, 0.0]]))
    with pytest.raises(ProviderError, match="dimensions"):
        await TeiEmbedder(_s(embedding_url="http://tei")).embed_query("x")


@respx.mock
async def test_openai_embedder_orders_by_index_and_sends_key():
    route = respx.post("https://api.example/v1/embeddings").mock(
        return_value=httpx.Response(
            200, json={"data": [{"index": 1, "embedding": [0, 1, 0]}, {"index": 0, "embedding": [1, 0, 0]}]}
        )
    )
    emb = OpenAIEmbedder(
        _s(embedding_url="https://api.example", embedding_api_key=SecretStr("k"), embedding_model="BAAI/bge-m3")
    )
    vecs = await emb.embed_documents(["a", "b"])
    assert vecs == [[1, 0, 0], [0, 1, 0]]
    req = route.calls[0].request
    assert req.headers["authorization"] == "Bearer k"
    assert json.loads(req.content) == {"model": "BAAI/bge-m3", "input": ["a", "b"]}


@respx.mock
async def test_http_retries_on_503_then_succeeds():
    route = respx.post("http://tei/embed").mock(
        side_effect=[httpx.Response(503), httpx.Response(200, json=[[1.0, 0, 0]])]
    )
    assert await TeiEmbedder(_s(embedding_url="http://tei")).embed_query("x") == [1.0, 0, 0]
    assert route.call_count == 2


@respx.mock
async def test_tei_reranker_applies_sigmoid_to_raw_scores():
    route = respx.post("http://rr/rerank").mock(
        return_value=httpx.Response(200, json=[{"index": 1, "score": -11.2}, {"index": 0, "score": 8.1}])
    )
    scores = await TeiReranker(_s(reranker_url="http://rr")).rerank("q", ["a", "b"])
    assert scores[0] > 0.99 and scores[1] < 0.001
    assert json.loads(route.calls[0].request.content)["raw_scores"] is True


@respx.mock
async def test_cohere_reranker_v2_shape():
    route = respx.post("https://api.cohere.com/v2/rerank").mock(
        return_value=httpx.Response(
            200, json={"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}]}
        )
    )
    rr = CohereReranker(_s(reranker_provider="cohere", reranker_model="rerank-v3.5", reranker_api_key=SecretStr("c")))
    assert await rr.rerank("q", ["a", "b"]) == [0.1, 0.9]
    body = json.loads(route.calls[0].request.content)
    assert body == {"model": "rerank-v3.5", "query": "q", "documents": ["a", "b"], "top_n": 2}
    assert route.calls[0].request.headers["authorization"] == "Bearer c"


@respx.mock
async def test_jina_reranker_shape():
    respx.post("https://api.jina.ai/v1/rerank").mock(
        return_value=httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.7}]})
    )
    rr = JinaReranker(_s(reranker_provider="jina", reranker_model="jina-reranker-v2-base-multilingual"))
    assert await rr.rerank("q", ["a"]) == [0.7]


@respx.mock
async def test_reranker_does_not_retry():
    route = respx.post("http://rr/rerank").mock(return_value=httpx.Response(503))
    with pytest.raises(ProviderError):
        await TeiReranker(_s(reranker_url="http://rr")).rerank("q", ["a"])
    assert route.call_count == 1


@respx.mock
async def test_chat_llm_and_vlm_ocr_shapes():
    route = respx.post("http://llm/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": " Mô tả. "}}]})
    )
    llm = build_llm(_s(llm_url="http://llm", llm_model="m"))
    assert llm is not None and llm.name == "default"
    assert await llm.complete("sys", "user") == "Mô tả."
    body = json.loads(route.calls[0].request.content)
    assert body["messages"][0] == {"role": "system", "content": "sys"} and body["temperature"] == 0

    ocr = VlmOcr(_s(ocr_provider="vlm", ocr_url="http://llm", ocr_model="vlm"))
    text = await ocr.page_text([b"\x89PNG fake"])
    content = json.loads(route.calls[1].request.content)["messages"][0]["content"]
    assert text == " Mô tả. " and content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_named_llm_providers(monkeypatch):
    monkeypatch.setenv("CHEAP_LLM_KEY", "sk-cheap")
    s = _s(
        llm_url="http://llm",
        llm_model="big",
        llm_providers={
            "cheap": {"url": "http://cheap/v1/chat/completions", "model": "small", "api_key_env": "CHEAP_LLM_KEY"}
        },
    )
    llms = build_llms(s)
    assert sorted(llms) == ["cheap", "default"]
    assert llms["cheap"].url == "http://cheap/v1/chat/completions" and llms["cheap"].model == "small"
    assert llms["cheap"].headers == {"Authorization": "Bearer sk-cheap"}
    assert build_llm(s) is not None and build_llm(s).name == "default"
    # Only a named provider: it becomes the default.
    only = _s(llm_providers={"cheap": {"url": "http://cheap", "model": "small"}})
    assert build_llm(only) is not None and build_llm(only).name == "cheap"


def test_named_llm_provider_errors(monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    with pytest.raises(ValueError, match="MISSING_KEY"):
        build_llms(_s(llm_providers={"x": {"url": "http://x", "model": "m", "api_key_env": "MISSING_KEY"}}))
    with pytest.raises(ValueError, match="reserved"):
        build_llms(_s(llm_providers={"default": {"url": "http://x", "model": "m"}}))
