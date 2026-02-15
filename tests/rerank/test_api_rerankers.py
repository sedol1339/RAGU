from unittest.mock import AsyncMock

import pytest
from tenacity import RetryError

from ragu.rerank.api_rerankers import VLLMReranker
import ragu.rerank.api_rerankers as api_module


class _FakeResponse:
    def __init__(self, payload, status_error: Exception | None = None):
        self._payload = payload
        self._status_error = status_error

    def raise_for_status(self):
        if self._status_error:
            raise self._status_error

    def json(self):
        return self._payload


@pytest.fixture
def make_reranker(monkeypatch):
    def _make(**kwargs):
        client = AsyncMock()
        monkeypatch.setattr(api_module.httpx, "AsyncClient", lambda timeout: client)
        reranker = VLLMReranker(model_name="bge-reranker", **kwargs)
        return reranker, client

    return _make


@pytest.mark.asyncio
async def test_vllm_rerank_builds_request_and_sorts_results(make_reranker):
    reranker, client = make_reranker(base_url="http://localhost:8000/v1", api_token="token")
    client.post.return_value = _FakeResponse(
        {
            "data": [
                {"index": 0, "score": 0.3},
                {"index": 1, "score": 0.9},
                {"index": 2, "score": 0.5},
            ]
        }
    )

    results = await reranker.rerank("query", ["d0", "d1", "d2"], top_k=2)

    assert results == [(1, 0.9), (2, 0.5)]
    assert client.post.await_count == 1

    _, kwargs = client.post.await_args
    assert kwargs["headers"]["Authorization"] == "Bearer token"
    assert kwargs["headers"]["Content-Type"] == "application/json"
    assert kwargs["json"] == {
        "model": "bge-reranker",
        "text_1": "query",
        "text_2": ["d0", "d1", "d2"],
    }
    assert client.post.await_args.args[0] == "http://localhost:8000/v1/score"


@pytest.mark.asyncio
async def test_vllm_rerank_without_token_omits_authorization_header(make_reranker):
    reranker, client = make_reranker(base_url="http://localhost:8000/v1", api_token=None)
    client.post.return_value = _FakeResponse({"data": [{"index": 0, "score": 1.0}]})

    await reranker.rerank("q", ["doc"])

    _, kwargs = client.post.await_args
    assert "Authorization" not in kwargs["headers"]


@pytest.mark.asyncio
async def test_vllm_rerank_empty_documents_short_circuit(make_reranker):
    reranker, client = make_reranker()

    result = await reranker.rerank("query", [])

    assert result == []
    client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_vllm_rerank_retries_and_succeeds(make_reranker):
    reranker, client = make_reranker()
    client.post.side_effect = [
        RuntimeError("temporary 1"),
        RuntimeError("temporary 2"),
        _FakeResponse({"data": [{"index": 0, "score": 0.7}]}),
    ]

    result = await reranker.rerank("query", ["doc"])

    assert result == [(0, 0.7)]
    assert client.post.await_count == 3


@pytest.mark.asyncio
async def test_vllm_rerank_retries_and_raises(make_reranker):
    reranker, client = make_reranker()
    client.post.side_effect = RuntimeError("always fails")

    with pytest.raises(RetryError):
        await reranker.rerank("query", ["doc"])
    assert client.post.await_count == 3


@pytest.mark.asyncio
async def test_vllm_aclose_swallows_close_errors(make_reranker):
    reranker, client = make_reranker()
    client.aclose.side_effect = RuntimeError("close failed")

    await reranker.aclose()

    client.aclose.assert_awaited_once()
