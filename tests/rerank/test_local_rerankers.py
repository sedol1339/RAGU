import builtins
import sys
import types

import pytest

from ragu.rerank.local_rerankers import CrossEncoderReranker


class _ArrayLike:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


class _FakeCrossEncoder:
    def __init__(self, model_name_or_path, **kwargs):
        self.model_name_or_path = model_name_or_path
        self.kwargs = kwargs
        self.predict_calls = []
        self.outputs = []

    def predict(self, batch, show_progress_bar=False):
        self.predict_calls.append((list(batch), show_progress_bar))
        next_output = self.outputs.pop(0)
        return next_output


def test_cross_encoder_reranker_raises_import_error(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("missing package")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="SentenceTransformer"):
        CrossEncoderReranker("fake-model")


def test_cross_encoder_reranker_init_forwards_kwargs(monkeypatch):
    fake_module = types.SimpleNamespace(CrossEncoder=_FakeCrossEncoder)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    reranker = CrossEncoderReranker("fake-model", device="cpu")

    assert reranker.model.model_name_or_path == "fake-model"
    assert reranker.model.kwargs["device"] == "cpu"


@pytest.mark.asyncio
async def test_cross_encoder_rerank_empty_input(monkeypatch):
    fake_module = types.SimpleNamespace(CrossEncoder=_FakeCrossEncoder)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    reranker = CrossEncoderReranker("fake-model")

    assert await reranker.rerank("query", []) == []
    assert reranker.model.predict_calls == []


@pytest.mark.asyncio
async def test_cross_encoder_rerank_sorts_scores_and_applies_top_k(monkeypatch):
    fake_module = types.SimpleNamespace(CrossEncoder=_FakeCrossEncoder)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    reranker = CrossEncoderReranker("fake-model")

    reranker.model.outputs = [[0.1, 0.9, 0.5]]
    result = await reranker.rerank("q", ["d0", "d1", "d2"], top_k=2)

    assert result == [(1, 0.9), (2, 0.5)]
    assert len(reranker.model.predict_calls) == 1
    assert reranker.model.predict_calls[0][1] is False


@pytest.mark.asyncio
async def test_cross_encoder_rerank_batches_and_handles_tolist_outputs(monkeypatch):
    fake_module = types.SimpleNamespace(CrossEncoder=_FakeCrossEncoder)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    reranker = CrossEncoderReranker("fake-model")

    reranker.model.outputs = [
        _ArrayLike([0.2, 0.6]),
        _ArrayLike([0.4]),
    ]

    result = await reranker.rerank("q", ["a", "bbb", "cc"], batch_size=2)

    assert len(reranker.model.predict_calls) == 2
    first_batch, second_batch = reranker.model.predict_calls[0][0], reranker.model.predict_calls[1][0]
    assert first_batch == [("q", "a"), ("q", "bbb")]
    assert second_batch == [("q", "cc")]
    assert result == [(1, 0.6), (2, 0.4), (0, 0.2)]

