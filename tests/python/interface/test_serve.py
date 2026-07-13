"""Unit tests for task-specific serve routing."""

import importlib
from unittest import mock

import pytest

pytestmark = [pytest.mark.unittest]


def _serve_kwargs(model: str) -> dict:
    return {
        "model": model,
        "device": "cpu",
        "model_lib": "model.so",
        "mode": "local",
        "enable_debug": False,
        "additional_models": [],
        "tensor_parallel_shards": None,
        "pipeline_parallel_stages": None,
        "opt": None,
        "max_num_sequence": None,
        "max_total_sequence_length": None,
        "max_single_sequence_length": None,
        "prefill_chunk_size": None,
        "sliding_window_size": None,
        "attention_sink_size": None,
        "max_history_size": None,
        "gpu_memory_utilization": None,
        "speculative_mode": "disable",
        "spec_draft_length": None,
        "spec_tree_width": None,
        "prefix_cache_mode": "disable",
        "prefix_cache_max_num_recycling_seqs": None,
        "prefill_mode": "hybrid",
        "enable_tracing": False,
        "host": "127.0.0.1",
        "port": 8000,
        "allow_credentials": False,
        "allow_origins": ["*"],
        "allow_methods": ["*"],
        "allow_headers": ["*"],
        "api_key": None,
    }


def test_serve_routes_resolved_remote_model_path(monkeypatch, tmp_path):
    """Task detection resolves a remote model once for the embedding runtime."""
    serve = importlib.import_module("mlc_llm.interface.serve")
    config_path = tmp_path / "mlc-chat-config.json"
    route = mock.MagicMock()
    monkeypatch.setattr(
        serve,
        "_detect_model_task_and_config",
        lambda model: ("embedding", config_path),
    )
    monkeypatch.setattr(serve, "_serve_embedding", route)

    serve.serve(**_serve_kwargs("HF://org/embedding-model"))

    assert route.call_args.kwargs["model"] == "HF://org/embedding-model"
    assert route.call_args.kwargs["model_path"] == str(tmp_path)


def test_embedding_serve_uses_resolved_model_path(monkeypatch, tmp_path):
    """Embedding runtime receives the resolved local model path."""
    serve = importlib.import_module("mlc_llm.interface.serve")

    runtime = mock.MagicMock()
    runtime_ctor = mock.MagicMock(return_value=runtime)
    monkeypatch.setattr(serve, "AsyncEmbeddingEngine", runtime_ctor)
    monkeypatch.setattr(serve.uvicorn, "run", lambda *args, **kwargs: None)

    serve._serve_embedding(
        model="HF://org/embedding-model",
        model_path=str(tmp_path),
        device="cpu",
        model_lib="model.so",
        host="127.0.0.1",
        port=8000,
        allow_credentials=False,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        api_key=None,
    )

    runtime_ctor.assert_called_once_with(
        model=str(tmp_path),
        model_lib="model.so",
        device="cpu",
    )
    runtime.terminate.assert_called_once_with()
