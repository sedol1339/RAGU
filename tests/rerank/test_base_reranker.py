import asyncio

import pytest

from ragu.rerank.base_reranker import BaseReranker
import ragu.rerank.base_reranker as base_module


class DummyReranker(BaseReranker):
    async def rerank(self, x: str, others: list[str], **kwargs) -> list[tuple[int, float]]:
        top_k = kwargs.get("top_k")
        results = [(i, float(len(doc))) for i, doc in enumerate(others)]
        results.sort(key=lambda item: item[1], reverse=True)
        if top_k is not None:
            results = results[:top_k]
        return results


@pytest.mark.asyncio
async def test_batch_rerank_raises_for_length_mismatch():
    reranker = DummyReranker()
    with pytest.raises(ValueError, match="Length of queries and documents must match"):
        await reranker.batch_rerank(["q1"], [["d1"], ["d2"]])


@pytest.mark.asyncio
async def test_batch_rerank_returns_empty_for_empty_queries():
    reranker = DummyReranker()
    assert await reranker.batch_rerank([], []) == []


@pytest.mark.asyncio
async def test_batch_rerank_sequential_mode():
    reranker = DummyReranker()
    results = await reranker.batch_rerank(
        queries=["q1", "q2"],
        documents=[["aaa", "b"], ["x", "yyyy"]],
        top_k=1,
    )
    assert results == [[(0, 3.0)], [(1, 4.0)]]


@pytest.mark.asyncio
async def test_batch_rerank_uses_async_runner_when_limiters_present(monkeypatch):
    reranker = DummyReranker()
    reranker._sem = asyncio.Semaphore(1)
    reranker._rps = object()
    reranker._rpm = object()

    calls: list[tuple[str, str, list[str]]] = []

    class FakeRunner:
        def __init__(self, semaphore, rps_limiter, rpm_limiter, progress_bar):
            assert semaphore is reranker._sem
            assert rps_limiter is reranker._rps
            assert rpm_limiter is reranker._rpm
            assert progress_bar is not None

        async def make_request(self, func, **kwargs):
            calls.append(("make_request", kwargs["x"], kwargs["others"]))
            return await func(**kwargs)

    monkeypatch.setattr(base_module, "AsyncRunner", FakeRunner)

    results = await reranker.batch_rerank(
        queries=["q1", "q2"],
        documents=[["a", "bbb"], ["zz"]],
        progress_bar_desc="batch rerank test",
    )

    assert calls == [
        ("make_request", "q1", ["a", "bbb"]),
        ("make_request", "q2", ["zz"]),
    ]
    assert results == [[(1, 3.0), (0, 1.0)], [(0, 2.0)]]


@pytest.mark.asyncio
async def test_call_delegates_to_rerank():
    reranker = DummyReranker()
    result = await reranker("query", ["doc1", "doc2"], top_k=1)
    assert result == [(0, 4.0)]

