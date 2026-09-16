import json
import queue
import threading
from dataclasses import replace

import httpx
import pytest

from k_openvino import serve


class FakeTokenizer:
    def apply_chat_template(
        self, messages, *, tools=None, tokenize, add_generation_prompt, enable_thinking
    ):
        assert messages == [{"role": "user", "content": "Hi"}]
        assert tools is None
        assert not tokenize
        assert add_generation_prompt
        assert enable_thinking is True
        return "prompt"

    def encode(self, text):
        return list(range(len(text)))


class FakePipeline:
    def generate(self, prompt, config, streamer=None):
        assert prompt == "prompt"
        assert streamer is not None
        streamer("hello")
        streamer(" world")


class FakeAliveProcess:
    def __init__(self, *, alive=True, exitcode=None):
        self._alive = alive
        self.exitcode = exitcode

    def is_alive(self):
        return self._alive


class FakeWorkerCommandQueue:
    def __init__(self, event_q, messages):
        self.event_q = event_q
        self.messages = messages

    def put(self, payload):
        if payload.get("kind") != "generate":
            return
        for msg in self.messages:
            self.event_q.put(msg)


def make_fake_stream_worker(messages, *, alive=True):
    event_q = queue.Queue()
    # Fakes duck-type the real multiprocessing Process/Queue/Event interface —
    # same typeshed-vs-runtime gap as serve.py's own `ctx.Process` comment.
    return serve._StreamWorkerState(
        model="qwen3-4b",
        process=FakeAliveProcess(alive=alive, exitcode=None if alive else -9),  # type: ignore[arg-type]
        cmd_q=FakeWorkerCommandQueue(event_q, messages),  # type: ignore[arg-type]
        event_q=event_q,  # type: ignore[arg-type]
        cancel_event=threading.Event(),  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_stream_includes_final_usage(monkeypatch):
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(
        serve, "_discover_models", lambda: {"qwen3-4b": {"ir": "unused"}}
    )
    monkeypatch.setattr(serve, "_get_tokenizer", lambda model: tokenizer)
    monkeypatch.setattr(
        serve,
        "_ensure_stream_worker",
        lambda model: make_fake_stream_worker(
            [
                {"kind": "event", "event": {"type": "content", "text": "hello"}},
                {"kind": "event", "event": {"type": "content", "text": " world"}},
                {
                    "kind": "done",
                    "completion": "hello world",
                    "finish_reason": "stop",
                    "cancelled": False,
                },
            ]
        ),
    )

    transport = httpx.ASGITransport(app=serve.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-4b",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert response.status_code == 200
    assert events[-1]["choices"] == []
    assert events[-1]["usage"] == {
        "prompt_tokens": 6,
        "completion_tokens": 11,
        "total_tokens": 17,
    }


def test_parse_full_plain_content():
    parsed = serve._parse_full("Hello, world!")
    assert parsed == {
        "reasoning_content": "",
        "content": "Hello, world!",
        "tool_calls": [],
    }


def test_parse_full_reasoning_then_content():
    parsed = serve._parse_full("<think>I should just say hi.</think>\n\nHi there!")
    assert parsed["reasoning_content"] == "I should just say hi."
    assert parsed["content"] == "Hi there!"
    assert parsed["tool_calls"] == []


def test_parse_full_tool_call():
    text = '<think>Need to list files.</think>\n<tool_call>\n{"name": "bash", "arguments": {"command": "ls /tmp"}}\n</tool_call>'
    parsed = serve._parse_full(text)
    assert parsed["reasoning_content"] == "Need to list files."
    assert parsed["content"] == ""
    assert len(parsed["tool_calls"]) == 1
    call = parsed["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "bash"
    assert json.loads(call["function"]["arguments"]) == {"command": "ls /tmp"}
    assert call["id"].startswith("call_")


def test_parse_full_multiple_tool_calls():
    text = (
        '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
        '<tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call>'
    )
    parsed = serve._parse_full(text)
    assert [c["function"]["name"] for c in parsed["tool_calls"]] == ["a", "b"]


def test_parse_full_malformed_tool_call_dropped():
    text = "<tool_call>{not valid json}</tool_call>trailing text"
    parsed = serve._parse_full(text)
    assert parsed["tool_calls"] == []
    assert parsed["content"] == "trailing text"


def test_stream_parser_handles_tag_split_across_feeds():
    """A tag split mid-token across streamer callbacks must still be detected —
    this is the exact scenario the buffering/lookback logic exists to handle."""
    parser = serve._StreamParser()
    events = []
    for chunk in ["<thi", "nk>Reasoning", " here</th", "ink>Answer"]:
        events.extend(parser.feed(chunk))
    events.extend(parser.finalize())
    reasoning = "".join(e["text"] for e in events if e["type"] == "reasoning")
    content = "".join(e["text"] for e in events if e["type"] == "content")
    assert reasoning == "Reasoning here"
    assert content == "Answer"


def test_stream_parser_tool_call_split_across_feeds():
    parser = serve._StreamParser()
    events = []
    for chunk in [
        "<tool_call>",
        '{"name": "bash",',
        ' "arguments": {}}',
        "</tool_call>",
    ]:
        events.extend(parser.feed(chunk))
    events.extend(parser.finalize())
    tool_events = [e for e in events if e["type"] == "tool_call"]
    assert len(tool_events) == 1
    assert tool_events[0]["name"] == "bash"


class _ReasoningOnlyResult:
    text = "<think>I thought about it but never concluded anything concrete.</think>"


class ReasoningOnlyPipeline:
    def generate(self, prompt, config, streamer=None):
        return _ReasoningOnlyResult()


@pytest.mark.asyncio
async def test_reasoning_only_response_falls_back_to_content(monkeypatch):
    """Regression test for opencode issue #37073 pattern: a turn that puts
    everything in reasoning_content and leaves content empty must never
    vanish from the client — content falls back to the reasoning text."""
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(
        serve, "_discover_models", lambda: {"qwen3-4b": {"ir": "unused"}}
    )
    monkeypatch.setattr(serve, "_get_tokenizer", lambda model: tokenizer)
    monkeypatch.setattr(
        serve, "_load", lambda model: (ReasoningOnlyPipeline(), tokenizer)
    )

    transport = httpx.ASGITransport(app=serve.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-4b",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )

    body = response.json()
    message = body["choices"][0]["message"]
    assert message["reasoning_content"] == (
        "I thought about it but never concluded anything concrete."
    )
    assert message["content"]
    assert message["content"] == message["reasoning_content"]


@pytest.mark.asyncio
async def test_stream_tool_call_stops_generation_early(monkeypatch):
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(
        serve, "_discover_models", lambda: {"qwen3-4b": {"ir": "unused"}}
    )
    monkeypatch.setattr(serve, "_get_tokenizer", lambda model: tokenizer)
    monkeypatch.setattr(
        serve,
        "_ensure_stream_worker",
        lambda model: make_fake_stream_worker(
            [
                {
                    "kind": "event",
                    "event": {
                        "type": "tool_call",
                        "name": "read",
                        "arguments": {"filePath": "/tmp/x"},
                    },
                },
                {
                    "kind": "done",
                    "completion": "",
                    "finish_reason": "tool_calls",
                    "cancelled": False,
                },
            ]
        ),
    )

    transport = httpx.ASGITransport(app=serve.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-4b",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    tool_call_events = [
        e
        for e in events
        if e.get("choices") and e["choices"][0]["delta"].get("tool_calls")
    ]
    content_events = [
        e
        for e in events
        if e.get("choices") and e["choices"][0]["delta"].get("content")
    ]
    assert len(tool_call_events) == 1
    assert content_events == []
    assert (
        tool_call_events[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
        == "read"
    )
    assert events[-2]["choices"][0]["finish_reason"] == "tool_calls"


class FakeDisconnectRequest:
    async def is_disconnected(self):
        return True


@pytest.mark.asyncio
async def test_watch_disconnect_cancels_then_terminates_worker(monkeypatch):
    state = make_fake_stream_worker([])
    calls = []
    monkeypatch.setattr(
        serve,
        "CONFIG",
        replace(serve.CONFIG, stream_cancel_grace_seconds=0.0),
    )
    monkeypatch.setattr(
        serve,
        "_force_terminate_stream_worker_if_same",
        lambda current: calls.append(current),
    )
    await serve._watch_disconnect(
        FakeDisconnectRequest(),  # type: ignore[arg-type]
        state,
        threading.Event(),
    )
    assert state.cancel_event.is_set()
    assert calls == [state]


# ---------------------------------------------------------------------------
# Per-model overrides (`server.overrides.json` living inside the model dir)
# ---------------------------------------------------------------------------


def test_read_model_overrides_reads_json_object(tmp_path):
    (tmp_path / serve.CONFIG.override_filename).write_text(
        '{"enable_thinking": false, "max_context": 4096}'
    )
    assert serve._read_model_overrides(tmp_path) == {
        "enable_thinking": False,
        "max_context": 4096,
    }


def test_read_model_overrides_tolerates_missing_and_malformed(tmp_path):
    """A bad override file must never break model discovery — just be ignored."""
    # Absent file -> defaults
    assert serve._read_model_overrides(tmp_path) == {}
    # Malformed JSON -> defaults
    override = tmp_path / serve.CONFIG.override_filename
    override.write_text("{not: valid json")
    assert serve._read_model_overrides(tmp_path) == {}
    # Valid JSON but not an object -> defaults
    override.write_text('["not", "an", "object"]')
    assert serve._read_model_overrides(tmp_path) == {}


def _write_fake_model(root, name, *, declared_ctx=131072, override=None):
    """Create a minimal LLM-layout model dir that _discover_models will accept."""
    model_dir = root / name
    model_dir.mkdir(parents=True)
    (model_dir / "openvino_model.xml").write_text("<net/>")
    (model_dir / "openvino_model.bin").write_text("weights")
    (model_dir / "config.json").write_text(
        json.dumps({"max_position_embeddings": declared_ctx})
    )
    if override is not None:
        (model_dir / serve.CONFIG.override_filename).write_text(json.dumps(override))
    return model_dir


def test_discover_models_applies_per_model_override(monkeypatch, tmp_path):
    """The override file wins over the global defaults, per model only."""
    _write_fake_model(
        tmp_path,
        "overridden",
        declared_ctx=131072,
        override={
            "enable_thinking": False,
            "max_context": 4096,
            "default_output_tokens": 512,
        },
    )
    _write_fake_model(tmp_path, "untouched", declared_ctx=131072)
    monkeypatch.setattr(serve, "MODELS_DIR", tmp_path)

    models = serve._discover_models()

    # Overridden model: the file's values win over every global default.
    assert models["overridden"]["context_length"] == 4096
    assert models["overridden"]["max_output_tokens"] == 512
    assert models["overridden"]["overrides"]["enable_thinking"] is False

    # Untouched model: global defaults still apply (min of declared and cap).
    assert models["untouched"]["context_length"] == serve.CONFIG.max_context
    assert models["untouched"]["max_output_tokens"] == serve.CONFIG.default_output_tokens
    assert models["untouched"]["overrides"] == {}


def test_override_cannot_exceed_what_the_model_declares(monkeypatch, tmp_path):
    """An override may lower the context, never inflate it past the model's own."""
    _write_fake_model(tmp_path, "small", declared_ctx=2048, override={"max_context": 999999})
    monkeypatch.setattr(serve, "MODELS_DIR", tmp_path)
    assert serve._discover_models()["small"]["context_length"] == 2048


def test_effective_device_prefers_recorded_runtime_value(monkeypatch):
    """A silent GPU -> CPU fallback must be reportable to clients."""
    monkeypatch.setitem(serve._pipeline_devices, "qwen3-4b", "CPU")
    assert serve._effective_device("qwen3-4b") == "CPU"


# ---------------------------------------------------------------------------
# Context-length guard (413) and dead-worker detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_oversized_prompt_is_rejected_with_413(monkeypatch):
    """Advertised max_input_tokens is now enforced, not just published."""
    tokenizer = FakeTokenizer()  # encode() -> one token per character ("prompt") = 6
    monkeypatch.setattr(
        serve,
        "_discover_models",
        lambda: {"qwen3-4b": {"ir": "unused", "context_length": 3}},
    )
    monkeypatch.setattr(serve, "_get_tokenizer", lambda model: tokenizer)

    transport = httpx.ASGITransport(app=serve.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-4b",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

    assert response.status_code == 413
    error = response.json()["error"]
    assert error["type"] == "context_length_exceeded"
    assert error["prompt_tokens"] == 6
    assert error["context_limit"] == 3
    assert "compact" in error["action"]


@pytest.mark.asyncio
async def test_dead_worker_surfaces_error_instead_of_hanging(monkeypatch):
    """Regression guard for the silent infinite wait: a worker that dies without
    sending `done`/`error` must produce a visible error, not an eternal stream."""
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(
        serve, "_discover_models", lambda: {"qwen3-4b": {"ir": "unused"}}
    )
    monkeypatch.setattr(serve, "_get_tokenizer", lambda model: tokenizer)
    monkeypatch.setattr(
        serve,
        "_ensure_stream_worker",
        lambda model: make_fake_stream_worker([], alive=False),
    )

    transport = httpx.ASGITransport(app=serve.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-4b",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert "WorkerDied" in response.text
    # The stream must terminate cleanly rather than being cut mid-flight.
    assert response.text.rstrip().endswith("data: [DONE]")


@pytest.mark.asyncio
async def test_stream_emits_metrics_block(monkeypatch):
    """Device and timing facts travel with the stream so a client is never blind."""
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(
        serve, "_discover_models", lambda: {"qwen3-4b": {"ir": "unused"}}
    )
    monkeypatch.setattr(serve, "_get_tokenizer", lambda model: tokenizer)
    monkeypatch.setattr(
        serve,
        "_ensure_stream_worker",
        lambda model: make_fake_stream_worker(
            [
                {"kind": "event", "event": {"type": "content", "text": "hi"}},
                {
                    "kind": "done",
                    "completion": "hi",
                    "finish_reason": "stop",
                    "cancelled": False,
                    "prefill_ms": 123,
                    "total_ms": 456,
                    "completion_tokens": 1,
                    "tokens_per_s": 12.5,
                },
            ]
        ),
    )

    transport = httpx.ASGITransport(app=serve.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-4b",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    metrics = events[-1]["metrics"]
    assert metrics["device"] == "GPU"
    assert metrics["prefill_ms"] == 123
    assert metrics["tokens_per_s"] == 12.5
