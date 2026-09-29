"""CPU-only regressions for embedding token preparation and request scheduling."""

# pylint: disable=missing-function-docstring,protected-access
import asyncio
import concurrent.futures
import threading
from collections import deque

import pytest

from mlc_llm.serve.embedding_engine import AsyncEmbeddingEngine


class _FakeTokenizer:
    def __init__(self):
        self.calls = []

    def encode(self, text):
        self.calls.append(text)
        if text == "bad":
            raise ValueError("invalid tokenizer input")
        return [len(word) for word in text.split()]


def _make_engine(embed_impl=None, *, dynamic=True, max_batch_tokens=100):
    engine = AsyncEmbeddingEngine.__new__(AsyncEmbeddingEngine)
    engine.tokenizer = _FakeTokenizer()
    engine.model_type = "decoder"
    engine._metadata = {"prefill_chunk_size": 100, "context_window_size": 100}
    engine._decoder_tokenizer_appends_eos = False
    engine._decoder_eos_token_id = None
    engine._dynamic_batching = dynamic
    engine._max_batch_tokens = max_batch_tokens
    engine._pending = deque()
    engine._outstanding = set()
    engine._batch_lock = threading.Lock()
    engine._drain_scheduled = False
    engine._terminated = False
    engine._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    if embed_impl is not None:
        engine._embed_token_lists = embed_impl
    return engine


@pytest.fixture
def make_engine():
    engines = []

    def make(*args, **kwargs):
        engine = _make_engine(*args, **kwargs)
        engines.append(engine)
        return engine

    yield make
    for engine in engines:
        engine.terminate()
        engine._executor.shutdown(wait=True)
        assert not engine._pending
        assert not engine._outstanding


def _positional_embed(calls):
    def embed(token_lists):
        calls.append(token_lists)
        return [[float(i)] for i in range(len(token_lists))], sum(map(len, token_lists))

    return embed


async def _enqueue_together(engine, inputs):
    """Hold the worker until all requests have reached the queue."""
    release = threading.Event()
    engine._executor.submit(release.wait, 5)
    try:
        tasks = [asyncio.create_task(engine.async_embed(texts)) for texts in inputs]
        await asyncio.sleep(0)
    finally:
        release.set()
    return await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 5)


@pytest.mark.asyncio
async def test_concurrent_requests_fuse_and_scatter_in_order(make_engine):
    calls = []
    engine = make_engine(_positional_embed(calls))
    results = await _enqueue_together(engine, [["a", "bb cc"], ["ddd"], ["e f", "ggg"]])
    assert len(calls) == 1
    assert results == [([[0.0], [1.0]], 3), ([[2.0]], 1), ([[3.0], [4.0]], 3)]


@pytest.mark.asyncio
async def test_budget_and_usage_include_eos_without_retokenizing(make_engine):
    calls = []
    engine = make_engine(_positional_embed(calls), max_batch_tokens=5)
    engine._decoder_eos_token_id = 99
    results = await _enqueue_together(engine, [["w w"] for _ in range(6)])
    # Raw lengths would incorrectly fit two requests; prepared lengths are three.
    assert [len(call) for call in calls] == [1] * 6
    assert results == [([[0.0]], 3)] * 6
    assert engine.tokenizer.calls == ["w w"] * 6


@pytest.mark.asyncio
async def test_oversized_request_runs_alone_and_keeps_fifo_order(make_engine):
    calls = []
    engine = make_engine(_positional_embed(calls), max_batch_tokens=3)
    results = await _enqueue_together(engine, [["a"], ["bb " * 5], ["ccc"], ["dddd"]])
    assert calls == [[[1]], [[2] * 5], [[3], [4]]]
    assert [usage for _, usage in results] == [1, 5, 1, 1]
    assert len(engine.tokenizer.calls) == 4


@pytest.mark.asyncio
async def test_usage_and_budget_use_truncated_tokens(make_engine):
    calls = []
    engine = make_engine(_positional_embed(calls), max_batch_tokens=6)
    engine._metadata["context_window_size"] = 3
    engine._decoder_eos_token_id = 99
    results = await _enqueue_together(engine, [["a " * 20], ["bb " * 20]])
    assert calls == [[[1, 1, 99], [2, 2, 99]]]
    assert results == [([[0.0]], 3), ([[1.0]], 3)]


@pytest.mark.asyncio
async def test_tokenizer_error_isolated_and_worker_recovers(make_engine):
    calls = []
    engine = make_engine(_positional_embed(calls))
    results = await _enqueue_together(engine, [["a"], ["bad"], ["bb"]])
    assert isinstance(results[1], ValueError)
    assert results[0] == ([[0.0]], 1)
    assert results[2] == ([[1.0]], 1)
    with pytest.raises(ValueError, match="invalid tokenizer input"):
        await asyncio.wait_for(engine.async_embed(["bad"]), 2)
    assert await asyncio.wait_for(engine.async_embed(["good"]), 2) == ([[0.0]], 1)


@pytest.mark.asyncio
async def test_forward_failure_settles_batch_and_worker_recovers(make_engine):
    calls = []

    def fail_once(tokens):
        calls.append(tokens)
        if len(calls) == 1:
            raise RuntimeError("inference failed")
        return [[0.0] for _ in tokens], sum(map(len, tokens))

    engine = make_engine(fail_once)
    results = await _enqueue_together(engine, [["a"]] * 3)
    assert all(isinstance(result, RuntimeError) for result in results)
    assert await asyncio.wait_for(engine.async_embed(["good"]), 2) == ([[0.0]], 1)


@pytest.mark.asyncio
async def test_opt_out_uses_per_request_forwards(make_engine):
    calls = []
    engine = make_engine(_positional_embed(calls), dynamic=False)
    results = await _enqueue_together(engine, [["t t"]] * 4)
    assert len(calls) == 4
    assert results == [([[0.0]], 2)] * 4


@pytest.mark.asyncio
async def test_lone_request_is_processed_immediately(make_engine):
    calls = []
    engine = make_engine(_positional_embed(calls))
    assert await asyncio.wait_for(engine.async_embed(["one two three"]), 2) == ([[0.0]], 3)
    assert calls == [[[3, 3, 5]]]


@pytest.mark.asyncio
@pytest.mark.parametrize("dynamic", [True, False])
async def test_terminate_during_tokenization_settles_all_requests(make_engine, dynamic):
    calls = []
    engine = make_engine(_positional_embed(calls), dynamic=dynamic, max_batch_tokens=1)
    started, release = threading.Event(), threading.Event()

    class GatedTokenizer:
        def encode(self, _):
            started.set()
            release.wait(5)
            return [1]

    engine.tokenizer = GatedTokenizer()
    tasks = [asyncio.create_task(engine.async_embed([text])) for text in ["first", "second"]]
    try:
        assert await asyncio.to_thread(started.wait, 5)
        engine.terminate()
        results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 2)
        assert all(isinstance(result, RuntimeError) for result in results)
        with pytest.raises(RuntimeError, match="terminated"):
            await engine.async_embed(["new"])
    finally:
        release.set()
    assert not calls


@pytest.mark.asyncio
async def test_cancelled_queued_request_is_not_tokenized(make_engine):
    calls = []
    engine = make_engine(_positional_embed(calls))
    release = threading.Event()
    engine._executor.submit(release.wait, 5)
    try:
        cancelled = asyncio.create_task(engine.async_embed(["cancelled"]))
        good = asyncio.create_task(engine.async_embed(["good"]))
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        await asyncio.sleep(0)
    finally:
        release.set()
    assert await asyncio.wait_for(good, 2) == ([[0.0]], 1)
    assert engine.tokenizer.calls == ["good"]


@pytest.mark.asyncio
@pytest.mark.parametrize("terminate", [True, False])
async def test_cancellation_during_forward_does_not_poison_worker(make_engine, terminate):
    started, release = threading.Event(), threading.Event()

    def embed(tokens):
        started.set()
        release.wait(5)
        return [[0.0] for _ in tokens], sum(map(len, tokens))

    engine = make_engine(embed)
    task = asyncio.create_task(engine.async_embed(["first"]))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        if terminate:
            engine.terminate()
    finally:
        release.set()
    if not terminate:
        assert await asyncio.wait_for(engine.async_embed(["second"]), 2) == ([[0.0]], 1)


@pytest.mark.asyncio
async def test_empty_token_input_is_rejected_without_shifting_results(make_engine):
    calls = []
    engine = make_engine(_positional_embed(calls))
    results = await _enqueue_together(engine, [["a"], [""], ["bb"]])
    assert isinstance(results[1], ValueError)
    assert results[2] == ([[1.0]], 1)
    assert engine.embed([]) == ([], 0)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_token_budget_rejected_before_loading_model(value):
    with pytest.raises(ValueError, match="positive integer"):
        AsyncEmbeddingEngine("unused", "unused", max_batch_tokens=value)


def test_encoder_preparation_preserves_special_tokens_at_truncation(make_engine):
    engine = make_engine()
    engine.model_type = "encoder"
    engine._metadata["prefill_chunk_size"] = 4
    engine._cls_token_id, engine._sep_token_id = 98, 99
    assert engine._tokenize_inputs(["a bb ccc dddd", ""]) == [[98, 1, 2, 99], [98, 99]]


def test_decoder_does_not_duplicate_postprocessor_or_existing_eos(make_engine):
    engine = make_engine()
    engine._decoder_eos_token_id = 3
    assert engine._tokenize_inputs(["one two"]) == [[3, 3]]
    engine._decoder_tokenizer_appends_eos = True
    assert engine._tokenize_inputs(["four"]) == [[4]]


@pytest.mark.parametrize("postprocessor", [True, False])
def test_decoder_truncation_preserves_terminal_pooling_token(make_engine, postprocessor):
    engine = make_engine()
    engine._metadata["context_window_size"] = 3
    engine._decoder_tokenizer_appends_eos = postprocessor
    engine._decoder_eos_token_id = 99
    engine.tokenizer.encode = lambda _: [1, 2, 3, 99] if postprocessor else [1, 2, 3]
    assert engine._tokenize_inputs(["long"]) == [[1, 2, 99]]


@pytest.mark.parametrize("lengths", [[146] * 7, [1] * 100, [15, 16, 17], [1024]])
def test_batch_cache_accounts_for_page_fragmentation(make_engine, lengths):
    engine = make_engine()
    capacities = []

    def capture_cache(*args):
        capacities.append([list(shape) for shape in args])
        raise RuntimeError("cache captured")

    engine._create_kv_cache_func = capture_cache
    with pytest.raises(RuntimeError, match="cache captured"):
        engine._batch_embed_decoder([[1] * n for n in lengths], sum(lengths), 1024, 1024, 0)
    pages = sum((n + 15) // 16 for n in lengths)
    assert capacities == [[[len(lengths)], [max(1024, pages * 16)], [1024], [16], [0]]]


def test_decoder_sub_batches_preserve_oversized_input_order(make_engine):
    engine = make_engine()
    engine._metadata["prefill_chunk_size"] = 3
    calls = []

    def forward(kind):
        def embed(tokens, total, *_):
            calls.append((kind, list(map(len, tokens))))
            return [[float(len(t))] for t in tokens], total

        return embed

    engine._batch_embed_decoder = forward("batch")
    engine._sequential_embed_decoder = forward("sequential")
    assert engine.embed(["a", "b " * 5, "cc", "ddd"]) == ([[1.0], [5.0], [1.0], [1.0]], 8)
    assert calls == [("batch", [1]), ("sequential", [5]), ("batch", [1, 1])]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
