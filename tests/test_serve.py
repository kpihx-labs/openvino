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
    def __init__(self, *, alive=True):
        self._alive = alive

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


def make_fake_stream_worker(messages):
    event_q = queue.Queue()
    # Fakes duck-type the real multiprocessing Process/Queue/Event interface —
    # same typeshed-vs-runtime gap as serve.py's own `ctx.Process` comment.
    return serve._StreamWorkerState(
        model="qwen3-4b",
        process=FakeAliveProcess(),  # type: ignore[arg-type]
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
