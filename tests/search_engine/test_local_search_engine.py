from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ragu.graph.types import Entity, Relation
from ragu.search_engine.local_search import LocalSearchEngine
from ragu.search_engine.search_functional import _find_most_related_edges_from_entities
from ragu.search_engine.types import LocalSearchResult


@pytest.mark.asyncio
async def test_local_search_collects_entities_relations_chunks_and_summaries(real_kg, kg_fixture_ids):
    entity_ids = kg_fixture_ids["entity_ids"]
    real_kg.index.entity_vector_db.query = AsyncMock(
        return_value=[
            {"__id__": entity_ids[0]},
            {"__id__": entity_ids[1]},
            {"__id__": "ent-missing"},
        ]
    )
    engine = LocalSearchEngine(
        client=SimpleNamespace(generate=AsyncMock()),
        knowledge_graph=real_kg,
        embedder=object(),
    )

    result = await engine.a_search("query", top_k=3)

    assert isinstance(result, LocalSearchResult)
    assert [e.id for e in result.entities] == entity_ids[:2]
    assert isinstance(result.relations, list)
    assert isinstance(result.chunks, list)
    assert isinstance(result.summaries, list)
    assert isinstance(result.documents_id, list)


@pytest.mark.asyncio
async def test_local_query_returns_raw_result_when_no_response_attr(monkeypatch, real_kg):
    client = SimpleNamespace(generate=AsyncMock(return_value=["raw-result"]))
    engine = LocalSearchEngine(client=client, knowledge_graph=real_kg, embedder=object())
    engine.truncation = lambda s: s
    engine.a_search = AsyncMock(return_value=LocalSearchResult())

    from ragu.search_engine import local_search as local_module
    monkeypatch.setattr(
        local_module,
        "render",
        lambda messages, **kwargs: [[{"role": "user", "content": "prompt"}]],
    )
    monkeypatch.setattr(
        engine,
        "get_prompt",
        lambda _: SimpleNamespace(messages=[{"role": "user", "content": "{{query}}"}], pydantic_model=None),
    )

    result = await engine.a_query("question")
    assert result == "raw-result"
