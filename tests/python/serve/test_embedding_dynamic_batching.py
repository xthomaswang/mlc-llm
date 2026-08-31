"""Unit tests for cross-request dynamic batching in AsyncEmbeddingEngine."""

# pylint: disable=missing-function-docstring,protected-access
import asyncio
import concurrent.futures
import threading
from collections import deque

import pytest

from mlc_llm.serve.embedding_engine import AsyncEmbeddingEngine


class _FakeTokenizer:
    def encode(self, text):
        return [0] * len(text.split())


def _make_engine(embed_impl, *, dynamic=True, max_batch_tokens=100):
    engine = AsyncEmbeddingEngine.__new__(AsyncEmbeddingEngine)
    engine.tokenizer = _FakeTokenizer()
    engine._dynamic_batching = dynamic
    engine._max_batch_tokens = max_batch_tokens
    engine._pending = deque()
    engine._batch_lock = threading.Lock()
    engine._drain_scheduled = False
    engine._terminated = False
    engine._executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="embedding-test"
    )
    engine.embed = embed_impl
    return engine


def _positional_embed(calls, first_started=None, first_release=None):
    """Fake embed: vector = [position in the fused batch]; records call inputs."""

    def embed(inputs):
        calls.append(list(inputs))
        if len(calls) == 1:
            if first_started is not None:
                first_started.set()
            if first_release is not None:
                first_release.wait(timeout=5)
        total = sum(len(text.split()) for text in inputs)
        return [[float(i)] for i in range(len(inputs))], total

    return embed


@pytest.mark.asyncio
async def test_concurrent_requests_fuse_into_one_forward():
    calls = []
    started, release = threading.Event(), threading.Event()
    engine = _make_engine(_positional_embed(calls, started, release))

    first = asyncio.ensure_future(engine.async_embed(["a a a"]))
    await asyncio.get_running_loop().run_in_executor(None, lambda: started.wait(5))
    others = [asyncio.ensure_future(engine.async_embed([f"text {i}"])) for i in range(8)]
    await asyncio.sleep(0.05)  # let them enqueue while the first forward is blocked
    release.set()

    first_result = await first
    other_results = await asyncio.gather(*others)

    assert len(calls) == 2, "8 concurrent requests should fuse into one forward"
    assert calls[1] == [f"text {i}" for i in range(8)]
    assert first_result == ([[0.0]], 3)
    # Scattering: request i got the vector at its fused-batch position, and
    # per-request usage equals its own token count (2 words each).
    for i, (vectors, usage) in enumerate(other_results):
        assert vectors == [[float(i)]]
        assert usage == 2
    engine.terminate()


@pytest.mark.asyncio
async def test_token_budget_splits_fused_batches():
    calls = []
    started, release = threading.Event(), threading.Event()
    engine = _make_engine(_positional_embed(calls, started, release), max_batch_tokens=5)

    first = asyncio.ensure_future(engine.async_embed(["a"]))
    await asyncio.get_running_loop().run_in_executor(None, lambda: started.wait(5))
    others = [asyncio.ensure_future(engine.async_embed(["w w"])) for _ in range(6)]
    await asyncio.sleep(0.05)
    release.set()

    await first
    await asyncio.gather(*others)

    # 6 x 2-token requests under a 5-token budget: 2 requests per fused round.
    assert len(calls) == 4
    assert [len(c) for c in calls[1:]] == [2, 2, 2]
    engine.terminate()


@pytest.mark.asyncio
async def test_error_propagates_to_all_fused_requests():
    calls = []
    started, release = threading.Event(), threading.Event()

    def failing_embed(inputs):
        calls.append(list(inputs))
        if len(calls) == 1:
            started.set()
            release.wait(timeout=5)
            return [[0.0] for _ in inputs], 1
        raise RuntimeError("boom")

    engine = _make_engine(failing_embed)
    first = asyncio.ensure_future(engine.async_embed(["a"]))
    await asyncio.get_running_loop().run_in_executor(None, lambda: started.wait(5))
    others = [asyncio.ensure_future(engine.async_embed(["b"])) for _ in range(3)]
    await asyncio.sleep(0.05)
    release.set()

    await first
    for task in others:
        with pytest.raises(RuntimeError, match="boom"):
            await task
    engine.terminate()


@pytest.mark.asyncio
async def test_opt_out_uses_per_request_forwards():
    calls = []
    engine = _make_engine(_positional_embed(calls), dynamic=False)

    results = await asyncio.gather(*[engine.async_embed([f"t {i}"]) for i in range(4)])

    assert len(calls) == 4, "with batching disabled every request runs its own forward"
    for vectors, _ in results:
        assert vectors == [[0.0]]
    engine.terminate()


@pytest.mark.asyncio
async def test_lone_request_is_processed_immediately():
    calls = []
    engine = _make_engine(_positional_embed(calls))

    vectors, usage = await engine.async_embed(["one two three"])

    assert calls == [["one two three"]]
    assert vectors == [[0.0]]
    assert usage == 3, "a lone request reports the engine-computed token count"
    engine.terminate()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
