import json

import pytest
from pydantic import BaseModel

from ragu.common.cache import (
    EmbeddingCache,
    PendingRequest,
    TextCache,
    make_embedding_cache_key,
    make_llm_cache_key,
)
from ragu.common.prompts import ChatMessages, UserMessage


class _AnswerModel(BaseModel):
    answer: str
    score: int


@pytest.fixture(autouse=True)
def reset_text_cache_singleton():
    TextCache._TextCache__instance = None  # type: ignore[attr-defined]
    yield
    TextCache._TextCache__instance = None  # type: ignore[attr-defined]


def test_make_llm_cache_key_is_deterministic_and_kwargs_order_independent():
    key1 = make_llm_cache_key(
        content="question",
        model_name="gpt-4o-mini",
        schema=_AnswerModel,
        kwargs={"temperature": 0.2, "top_p": 1.0},
    )
    key2 = make_llm_cache_key(
        content="question",
        model_name="gpt-4o-mini",
        schema=_AnswerModel,
        kwargs={"top_p": 1.0, "temperature": 0.2},
    )

    assert key1 == key2
    assert key1.startswith("llm-cache-")


def test_make_llm_cache_key_changes_when_request_shape_changes():
    base = make_llm_cache_key(content="question", model_name="gpt-4o-mini")
    with_schema = make_llm_cache_key(content="question", model_name="gpt-4o-mini", schema=_AnswerModel)
    other_model = make_llm_cache_key(content="question", model_name="gpt-4.1")

    assert base != with_schema
    assert base != other_model


def test_make_embedding_cache_key_is_deterministic_and_prefixed():
    key1 = make_embedding_cache_key("hello", "text-embedding-3-large")
    key2 = make_embedding_cache_key("hello", "text-embedding-3-large")
    key3 = make_embedding_cache_key("hello!", "text-embedding-3-large")

    assert key1 == key2
    assert key1 != key3
    assert key1.startswith("emb-cache-")


def test_pending_request_dataclass_holds_values():
    messages = ChatMessages.from_messages([UserMessage(content="hello")])
    item = PendingRequest(index=3, messages=messages, cache_key="llm-cache-123")

    assert item.index == 3
    assert item.cache_key == "llm-cache-123"
    assert item.messages.to_str() == "[user]: hello"


@pytest.mark.asyncio
async def test_text_cache_set_get_string_and_persists_payload(tmp_path):
    cache_path = tmp_path / "llm_cache.json"
    cache = TextCache(cache_path=cache_path, flush_every_n_writes=1)

    await cache.set("k1", "value", model_name="gpt-4o-mini", input_instruction="prompt")
    got = await cache.get("k1")

    assert got == "value"
    assert cache_path.exists()

    payload = json.loads(cache_path.read_text(encoding="utf-8"))["k1"]
    assert payload["response"] == "value"
    assert payload["additional_payload"]["model_name"] == "gpt-4o-mini"
    assert payload["additional_payload"]["input_instruction"] == "prompt"
    assert isinstance(payload["time"], str)


@pytest.mark.asyncio
async def test_text_cache_get_supports_legacy_raw_values(tmp_path):
    cache = TextCache(cache_path=tmp_path / "llm_cache.json", flush_every_n_writes=100)
    cache._mem_cache["legacy"] = "old-format"

    assert await cache.get("legacy") == "old-format"


@pytest.mark.asyncio
async def test_text_cache_round_trip_pydantic_schema(tmp_path):
    cache = TextCache(cache_path=tmp_path / "llm_cache.json", flush_every_n_writes=100)
    value = _AnswerModel(answer="ok", score=7)

    await cache.set("k2", value)
    got = await cache.get("k2", schema=_AnswerModel)

    assert isinstance(got, _AnswerModel)
    assert got.model_dump() == value.model_dump()


@pytest.mark.asyncio
async def test_text_cache_schema_decode_failure_returns_none(tmp_path):
    cache = TextCache(cache_path=tmp_path / "llm_cache.json", flush_every_n_writes=100)
    cache._mem_cache["broken"] = {
        "response": {"answer": "ok"},  # missing required "score"
        "additional_payload": {},
        "time": "2026-01-01T00:00:00+00:00",
    }

    assert await cache.get("broken", schema=_AnswerModel) is None


@pytest.mark.asyncio
async def test_text_cache_flush_empty_cache_resets_pending_counter(tmp_path):
    cache = TextCache(cache_path=tmp_path / "llm_cache.json", flush_every_n_writes=100)
    cache._pending_disk_writes = 5

    await cache.flush_cache()

    assert cache._pending_disk_writes == 0
    assert not (tmp_path / "llm_cache.json").exists()


@pytest.mark.asyncio
async def test_text_cache_close_flushes_pending_writes(tmp_path):
    cache_path = tmp_path / "llm_cache.json"
    cache = TextCache(cache_path=cache_path, flush_every_n_writes=100)

    await cache.set("k3", "value")
    assert not cache_path.exists()

    await cache.close()
    assert cache_path.exists()
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    assert data["k3"]["response"] == "value"


@pytest.mark.asyncio
async def test_text_cache_loads_existing_cache_from_disk(tmp_path):
    cache_path = tmp_path / "llm_cache.json"
    cache_path.write_text(
        json.dumps({"k4": {"response": "persisted", "additional_payload": {}, "time": "t"}}),
        encoding="utf-8",
    )

    cache = TextCache(cache_path=cache_path, flush_every_n_writes=100)
    assert await cache.get("k4") == "persisted"


@pytest.mark.asyncio
async def test_embedding_cache_set_get_and_auto_flush(tmp_path):
    cache_path = tmp_path / "emb_cache.pkl"
    cache = EmbeddingCache(cache_path=cache_path, flush_every_n_writes=1)

    await cache.set("e1", [0.1, 0.2, 0.3])
    assert await cache.get("e1") == [0.1, 0.2, 0.3]
    assert cache_path.exists()

    reloaded = EmbeddingCache(cache_path=cache_path, flush_every_n_writes=1)
    assert await reloaded.get("e1") == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_embedding_cache_close_flushes_pending_writes(tmp_path):
    cache_path = tmp_path / "emb_cache.pkl"
    cache = EmbeddingCache(cache_path=cache_path, flush_every_n_writes=100)

    await cache.set("e2", [1.0, 2.0])
    assert not cache_path.exists()

    await cache.close()
    assert cache_path.exists()


@pytest.mark.asyncio
async def test_embedding_cache_handles_corrupted_file_gracefully(tmp_path):
    cache_path = tmp_path / "emb_cache.pkl"
    cache_path.write_text("not-a-pickle", encoding="utf-8")

    cache = EmbeddingCache(cache_path=cache_path, flush_every_n_writes=100)
    assert await cache.get("missing") is None

