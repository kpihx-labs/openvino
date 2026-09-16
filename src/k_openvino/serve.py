"""OpenVINO GenAI — OpenAI-compatible chat server (local sovereign LLM backend)."""

from __future__ import annotations

import asyncio
import base64
import gc
import json
import logging
import multiprocessing as mp
import os
import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue as MPQueue
from multiprocessing.synchronize import Event as MPEvent
from pathlib import Path

import numpy as np
import openvino as ov
import openvino_genai
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from transformers import AutoTokenizer

from k_openvino.config import CONFIG

MODELS_DIR = CONFIG.models_dir

logging.basicConfig(level=CONFIG.log_level)
logger = logging.getLogger("k_openvino")

app = FastAPI(title="openvino-serve")
_current: dict[str, object] = {"name": None, "pipe": None, "tok": None}
_tokenizers: dict[str, object] = {}
_gen_lock = threading.Lock()
_stream_worker_lock = threading.Lock()


@dataclass
class _StreamWorkerState:
    model: str
    process: BaseProcess
    cmd_q: MPQueue[dict]
    event_q: MPQueue[dict]
    cancel_event: MPEvent


_stream_worker: _StreamWorkerState | None = None


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    """100% API transparency — never let FastAPI's default plain-text 500 through.

    Without this, an uncaught exception returns a non-JSON body that OpenCode's
    AI SDK (APICallError) cannot render — the client stays silently stuck instead
    of showing the error in red like every other provider. Always return the
    OpenAI-compatible {"error": {...}} shape so the SDK parses and displays it.
    """
    logger.error(
        "Unhandled exception on %s %s: %s",
        request.method,
        request.url.path,
        exc,
        exc_info=exc,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(exc),
                "type": "server_error",
                "code": type(exc).__name__,
            }
        },
    )


def _token_count(tokenizer: object, text: str) -> int:
    """Count tokens with the model tokenizer, falling back conservatively."""
    try:
        return len(tokenizer.encode(text))  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return len(text) // 4


def _normalize_content(content: object) -> str:
    """Extract plain text from an OpenAI message `content` field.

    `content` is either a plain string or a multimodal parts array
    (`{"type": "text", ...}` / `image_url` / `input_audio` / `file`, see the
    OpenAI chat completions spec). Text parts are joined; non-text parts are
    ignored here (images are collected separately via `_extract_images`).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        return "\n".join(parts)
    return "" if content is None else str(content)


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """Normalize every message's `content` to a plain string for the chat template."""
    out = []
    for m in messages:
        m2 = dict(m)
        if "content" in m2:
            m2["content"] = _normalize_content(m2["content"])
        out.append(m2)
    return out


def _template_messages(messages: list[dict]) -> list[dict]:
    """Build messages for ``apply_chat_template``, preserving multimodal image parts.

    ``_normalize_messages`` strips everything but text (good for token counting,
    fatal for prompt building: the Gemma4 jinja only emits ``<|image|>`` when it
    sees a ``{"type": "image_url"}`` / ``{"type": "image"}`` part in ``content``).
    This keeps ``text`` + ``image``/``image_url`` parts in order and drops only the
    modalities we have no tensors for (audio/video/file), so the template puts
    the tag inside the user turn instead of us prepending it before ``<bos>``.

    Args:
        messages: OpenAI-format message list.  Each ``content`` may be a plain
            string or a list of multimodal parts (``text``, ``image``,
            ``image_url``, ``audio``, ``video``, ``file``).

    Returns:
        New list with the same structure, but ``audio``/``video``/``file`` parts
        stripped (no tensor backing) while ``text``+``image`` parts preserved.

    Examples:
        >>> _template_messages([{"role": "user", "content": "hi"}])
        [{"role": "user", "content": "hi"}]

        >>> _template_messages([{"role": "user", "content": [
        ...     {"type": "text", "text": "Describe"},
        ...     {"type": "image_url", "image_url": {"url": "file:///a.png"}}
        ... ]}])[0]["content"]
        [{"type": "text", "text": "Describe"}, {"type": "image_url", ...}]
    """
    out: list[dict] = []
    for m in messages:
        m2 = dict(m)
        content = m2.get("content")
        if isinstance(content, list):
            parts: list[dict] = []
            for p in content:
                if not isinstance(p, dict):
                    continue
                t = p.get("type")
                if t == "text":
                    parts.append({"type": "text", "text": p.get("text", "")})
                elif t in ("image", "image_url"):
                    parts.append(p)
            m2["content"] = parts if parts else _normalize_content(content)
        out.append(m2)
    return out


def _image_to_array(pic: Image.Image) -> np.ndarray:
    """Convert a PIL image to NHWC u8 ndarray layout expected by VLMPipeline."""
    rgb = pic.convert("RGB")
    # Official GenAI samples force uint8 NHWC [1,H,W,3] — dtype drift breaks vision.
    return np.asarray(rgb, dtype=np.uint8)[None]


def _load_image_url(url: str) -> np.ndarray | None:
    """Load an OpenAI `image_url` value (data URI or local path) into an NHWC array.

    Remote http(s) URLs are intentionally unsupported here (sovereign local-only);
    pass a data URI or an absolute filesystem path instead.
    """
    if not url or not isinstance(url, str):
        return None
    raw = url.strip()
    if raw.startswith("data:"):
        # data:[<mediatype>][;base64],<data>
        m = re.match(r"^data:[^;]*;base64,(.+)$", raw, flags=re.DOTALL)
        if not m:
            return None
        try:
            blob = base64.b64decode(m.group(1))
        except Exception:  # noqa: BLE001
            return None
        from io import BytesIO

        try:
            return _image_to_array(Image.open(BytesIO(blob)))
        except Exception:  # noqa: BLE001
            return None
    # file:///abs/path or plain filesystem path
    raw = raw.removeprefix("file://")
    path = Path(raw)
    if path.is_file():
        try:
            return _image_to_array(Image.open(path))
        except Exception:  # noqa: BLE001
            return None
    return None


def _ensure_image_tags(prompt: str, n_images: int) -> str:
    """Ensure the prompt references each image (GenAI / gemma4 need an image tag).

    Without a tag, GenAI prepends the vision block before bos and vision binding
    is scrambled.  Prefer native ``<|image|>`` when missing; also accept GenAI's
    universal ``<ov_genai_image_i>`` if already present.  Fallback insertion
    places the tag *inside* the user turn (after ``<|turn>user\\n``), never
    before ``<bos>``.

    Args:
        prompt: Rendered chat prompt string (from ``apply_chat_template``).
        n_images: Number of image tensors that will be passed to ``generate()``.

    Returns:
        Prompt with image tags injected if needed; unchanged if tags already
        present or ``n_images <= 0``.

    Examples:
        >>> _ensure_image_tags("<bos>hello", 0)
        '<bos>hello'

        >>> _ensure_image_tags("<bos><|turn>user\\nHi<turn|>", 1)
        '<bos><|turn>user\\n<|image|>Hi<turn|>'
    """
    if n_images <= 0:
        return prompt
    if (
        "<|image|>" in prompt
        or "<ov_genai_image_" in prompt
        or "<start_of_image>" in prompt
    ):
        return prompt
    tags = "".join("<|image|>" for _ in range(n_images))
    # Fallback placement MUST stay inside the user turn — prepending before
    # `<bos>` scrambles vision binding (same as having no tag at all). The
    # native template path (`_template_messages`) already puts the tag
    # correctly; this only rescues hand-built prompts that missed it.
    marker = "<|turn>user\n"
    idx = prompt.find(marker)
    if idx != -1:
        insert_at = idx + len(marker)
        return prompt[:insert_at] + tags + prompt[insert_at:]
    return f"{tags}{prompt}"


def _extract_images(messages: list[dict]) -> list[np.ndarray]:
    """Collect images from OpenAI multimodal message parts (architecture-agnostic).

    Iterates over all messages, finds ``image_url`` parts, and loads each URL
    (``file://`` path or plain filesystem path) into an NHWC ``uint8`` ndarray
    suitable for ``openvino.Tensor``.

    Args:
        messages: OpenAI-format message list with optional ``content`` arrays
            containing ``{"type": "image_url", "image_url": {"url": ...}}`` parts.

    Returns:
        List of NHWC ``uint8`` ndarrays, one per loadable image.  Empty list
        if no images found or all URLs fail to load.

    Examples:
        >>> _extract_images([{"role": "user", "content": "hi"}])
        []

        >>> _extract_images([{"role": "user", "content": [
        ...     {"type": "image_url", "image_url": {"url": "file:///a.png"}}
        ... ]}])
        [array([[[...]]], dtype=uint8)]
    """
    images: list[np.ndarray] = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else image_url
            arr = _load_image_url(url) if isinstance(url, str) else None
            if arr is not None:
                images.append(arr)
    return images


def _result_text(result: object) -> str:
    """Normalize OpenVINO GenAI generate() output across LLM and VLM result types.

    ``LLMPipeline.generate()`` returns a ``DecodedResults`` with a ``texts``
    attribute; ``VLMPipeline.generate()`` returns a ``VLMDecodedResults`` with
    either ``texts`` or ``text``.  This function extracts the first text string
    from whichever format is present.

    Args:
        result: Raw return value from ``LLMPipeline.generate()`` or
            ``VLMPipeline.generate()``.

    Returns:
        The generated text string, or ``str(result)`` as last resort.

    Examples:
        >>> _result_text(type("R", (), {"texts": ["hello"]})())
        'hello'

        >>> _result_text(type("R", (), {"text": "world"})())
        'world'
    """
    if hasattr(result, "texts"):
        texts = getattr(result, "texts", [])  # type: ignore[arg-type]
        if texts:
            return str(texts[0])
    if hasattr(result, "text"):
        return str(getattr(result, "text", ""))  # type: ignore[arg-type]
    return str(result)


def _ir_kind(model_dir: Path) -> str:
    """Detect pipeline kind from IR layout on disk (no model-name hardcoding).

    VLM exports write `openvino_language_model.xml` (+ vision embeddings);
    plain CausalLM exports write `openvino_model.xml`.
    """
    if (model_dir / "openvino_language_model.xml").exists():
        return "vlm"
    if (model_dir / "openvino_model.xml").exists():
        return "llm"
    return "unknown"


class _StreamParser:
    """Incremental parser for Qwen-style `<think>`/`<tool_call>` tagged output.

    Feed raw subwords as they stream from the model; get back structured
    events (`reasoning` / `content` / `tool_call`) used to build both the
    non-streaming response and the streaming SSE deltas from ONE shared
    grammar — never duplicate the tag-parsing logic between the two paths.

    Tool calls are buffered whole (not char-by-char JSON deltas) and emitted
    as a single complete `tool_call` event once `</tool_call>` closes — this
    matches how many OpenAI-compatible providers behave when the underlying
    model doesn't natively support incremental JSON argument streaming, and
    avoids the well-documented fragility of reassembling partial JSON across
    chunks (see openai-python issues #3201/#3203).
    """

    _OPEN_TAGS = ("<think>", "<tool_call>")

    def __init__(self) -> None:
        self.buffer = ""
        self.state = "scan"  # scan | thinking | toolcall

    @staticmethod
    def _longest_tag_prefix(s: str, tags: tuple[str, ...]) -> int:
        """Length of the longest suffix of `s` that is a strict prefix of one of
        `tags` (not a full match) — that suffix must stay buffered in case the
        tag is split across chunks. Scoped to only the tags relevant to the
        CURRENT state, so a fully-formed tag from a DIFFERENT phase (e.g. a
        closing `</tool_call>` while scanning for an opening tag) is never
        mistaken for "might still be growing into something."""
        max_len = 0
        for tag in tags:
            for k in range(min(len(s), len(tag) - 1), 0, -1):
                if s.endswith(tag[:k]):
                    max_len = max(max_len, k)
                    break
        return max_len

    def feed(self, subword: str) -> list[dict]:
        self.buffer += subword
        events: list[dict] = []
        progress = True
        while progress:
            progress = False
            if self.state == "scan":
                think_idx = self.buffer.find("<think>")
                tool_idx = self.buffer.find("<tool_call>")
                candidates = [i for i in (think_idx, tool_idx) if i != -1]
                if candidates:
                    idx = min(candidates)
                    if idx > 0:
                        events.append({"type": "content", "text": self.buffer[:idx]})
                    if idx == think_idx:
                        self.buffer = self.buffer[idx + len("<think>") :]
                        self.state = "thinking"
                    else:
                        self.buffer = self.buffer[idx + len("<tool_call>") :]
                        self.state = "toolcall"
                    progress = True
                else:
                    safe_len = len(self.buffer) - self._longest_tag_prefix(
                        self.buffer, self._OPEN_TAGS
                    )
                    if safe_len > 0:
                        events.append(
                            {"type": "content", "text": self.buffer[:safe_len]}
                        )
                        self.buffer = self.buffer[safe_len:]
                        progress = True
            elif self.state == "thinking":
                idx = self.buffer.find("</think>")
                if idx != -1:
                    if idx > 0:
                        events.append({"type": "reasoning", "text": self.buffer[:idx]})
                    self.buffer = self.buffer[idx + len("</think>") :]
                    self.state = "scan"
                    progress = True
                else:
                    safe_len = len(self.buffer) - self._longest_tag_prefix(
                        self.buffer, ("</think>",)
                    )
                    if safe_len > 0:
                        events.append(
                            {"type": "reasoning", "text": self.buffer[:safe_len]}
                        )
                        self.buffer = self.buffer[safe_len:]
                        progress = True
            elif self.state == "toolcall":
                idx = self.buffer.find("</tool_call>")
                if idx != -1:
                    raw = self.buffer[:idx].strip()
                    self.buffer = self.buffer[idx + len("</tool_call>") :]
                    self.state = "scan"
                    try:
                        obj = json.loads(raw)
                        events.append(
                            {
                                "type": "tool_call",
                                "name": obj.get("name", ""),
                                "arguments": obj.get("arguments", {}),
                            }
                        )
                    except json.JSONDecodeError:
                        logger.warning("Unparseable tool_call block: %r", raw)
                    progress = True
                # else: keep buffering — no partial emission for tool calls.
        return events

    def finalize(self) -> list[dict]:
        events: list[dict] = []
        if self.buffer:
            if self.state == "thinking":
                events.append({"type": "reasoning", "text": self.buffer})
            elif self.state == "scan":
                events.append({"type": "content", "text": self.buffer})
            # Unclosed toolcall at end-of-stream: malformed, dropped (logged).
            elif self.state == "toolcall":
                logger.warning("Unclosed <tool_call> at end of stream: %r", self.buffer)
        self.buffer = ""
        return events


def _parse_full(text: str) -> dict:
    """Parse a complete (non-streaming) generation through `_StreamParser`."""
    parser = _StreamParser()
    events = parser.feed(text) + parser.finalize()
    reasoning = "".join(e["text"] for e in events if e["type"] == "reasoning")
    content = "".join(e["text"] for e in events if e["type"] == "content")
    tool_calls = [
        {
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": e["name"], "arguments": json.dumps(e["arguments"])},
        }
        for e in events
        if e["type"] == "tool_call"
    ]
    return {
        "reasoning_content": reasoning.strip(),
        "content": content.strip(),
        "tool_calls": tool_calls,
    }


def _model_ir(name: str):
    discovered = _discover_models()
    if name not in discovered:
        raise RuntimeError(
            f"Model not found: {name} (available: {list(discovered.keys())})"
        )
    return discovered[name]["ir"]


def _generate(
    pipe,
    prompt: str,
    config,
    *,
    images: list | None = None,
    streamer=None,
):
    """Call generate() on LLMPipeline or VLMPipeline with the right signature.

    VLMPipeline rejects a positional GenerationConfig (text-only overload is
    ``generate(prompt, **kwargs)``). LLMPipeline keeps the historical
    positional ``generate(prompt, config, streamer=...)`` form.
    """
    is_vlm = isinstance(pipe, openvino_genai.VLMPipeline)
    if is_vlm:
        n_img = len(images) if images else 0
        prompt = _ensure_image_tags(prompt, n_img)
        kwargs: dict = {"generation_config": config}
        if streamer is not None:
            kwargs["streamer"] = streamer
        if images:
            kwargs["images"] = [
                ov.Tensor(arr) if not isinstance(arr, ov.Tensor) else arr
                for arr in images
            ]
        return pipe.generate(prompt, **kwargs)
    if streamer is not None:
        return pipe.generate(prompt, config, streamer=streamer)
    return pipe.generate(prompt, config)


def _build_pipeline(name: str, ir):
    device = os.environ.get("OPENVINO_DEVICE", "GPU")
    # GPU Arc OOM even for 0.6B (5.4G peak + 3G swap) -> LATENCY + 1 stream to cut VRAM
    cfg: dict = {}
    if device == "GPU":
        cfg = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1"}
    kind = _ir_kind(Path(ir))
    # VLM INT4 on Arc: disable dynamic quantization (group size 0) -> community-
    # validated for vision path; override via OPENVINO_DQ_GROUP_SIZE.
    # NOTE (2026-09-16): DYNAMIC_QUANTIZATION_GROUP_SIZE=0 breaks Gemma4 VLM image
    # path (0 completion tokens, empty output). Validated by A/B on GPU:
    #   no props -> image works, DQ=0 -> empty. Only apply for LLM, not VLM.
    if kind == "llm":
        dq = os.environ.get("OPENVINO_DQ_GROUP_SIZE", "0")
        try:
            cfg["DYNAMIC_QUANTIZATION_GROUP_SIZE"] = int(dq)
        except ValueError:
            cfg["DYNAMIC_QUANTIZATION_GROUP_SIZE"] = 0
    pipe_cls = (
        openvino_genai.VLMPipeline if kind == "vlm" else openvino_genai.LLMPipeline
    )

    def _try(dev: str):
        # GenAI constructors take properties as **kwargs, never a positional dict.
        if cfg:
            try:
                return pipe_cls(str(ir), dev, **cfg)
            except TypeError:
                return pipe_cls(str(ir), dev)
        return pipe_cls(str(ir), dev)

    try:
        return _try(device)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "GPU load failed for %s (device=%s, kind=%s): %s — falling back to CPU",
            name,
            device,
            kind,
            e,
        )
        return _try("CPU")


def _get_tokenizer(name: str):
    tok = _tokenizers.get(name)
    if tok is not None:
        return tok
    ir = _model_ir(name)
    tok = AutoTokenizer.from_pretrained(str(ir))
    _tokenizers[name] = tok
    return tok


def _stream_worker_main(model_name: str, model_dir: str, cmd_q, event_q, cancel_event):
    """Dedicated per-model generation worker.

    Why a subprocess at all?
    - Level 1: the parent can request graceful cancellation on client disconnect
      by setting `cancel_event`, which the stream callback converts to
      `StreamingStatus.CANCEL` on the next chunk.
    - Level 2: if disconnect happens during prefill (before any callback/chunk),
      Python's high-level LLMPipeline.generate() offers no out-of-band cancel API.
      The parent can then kill THIS subprocess without taking down the whole
      FastAPI server, solving the last "model busy after interrupted" residue.
    """
    pipe = _build_pipeline(model_name, model_dir)
    while True:
        cmd = cmd_q.get()
        if cmd is None or cmd.get("kind") == "shutdown":
            return
        if cmd.get("kind") != "generate":
            continue
        cancel_event.clear()
        parser = _StreamParser()
        completion_parts: list[str] = []
        tool_call_seen = False
        config = openvino_genai.GenerationConfig()
        config.max_new_tokens = int(cmd["max_tokens"])
        temperature = float(cmd["temperature"])
        config.do_sample = temperature > 0
        if config.do_sample:
            config.temperature = temperature

        def streamer(
            subword: str,
            *,
            _completion_parts=completion_parts,
            _parser=parser,
        ):
            nonlocal tool_call_seen
            if cancel_event.is_set():
                return openvino_genai.StreamingStatus.CANCEL
            _completion_parts.append(subword)
            status = openvino_genai.StreamingStatus.RUNNING
            for event in _parser.feed(subword):
                event_q.put({"kind": "event", "event": event})
                if event["type"] == "tool_call":
                    tool_call_seen = True
                    status = openvino_genai.StreamingStatus.TOOL_CALL_STOP
                    break
            if cancel_event.is_set():
                return openvino_genai.StreamingStatus.CANCEL
            return status

        try:
            # Images travel as picklable numpy arrays across the subprocess boundary
            _generate(
                pipe,
                cmd["prompt"],
                config,
                images=cmd.get("images") or None,
                streamer=streamer,
            )
        except Exception as e:  # noqa: BLE001
            event_q.put(
                {
                    "kind": "error",
                    "message": str(e),
                    "code": type(e).__name__,
                }
            )
        finally:
            for event in parser.finalize():
                event_q.put({"kind": "event", "event": event})
            event_q.put(
                {
                    "kind": "done",
                    "completion": "".join(completion_parts),
                    "finish_reason": "tool_calls" if tool_call_seen else "stop",
                    "cancelled": cancel_event.is_set(),
                }
            )
            cancel_event.clear()


def _shutdown_stream_worker_locked() -> None:
    global _stream_worker
    state = _stream_worker
    if state is None:
        return
    try:
        state.cancel_event.set()
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to set worker cancel_event during shutdown: %s", e)
    try:
        state.cmd_q.put({"kind": "shutdown"})
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to send worker shutdown command: %s", e)
    proc = state.process
    try:
        proc.join(timeout=CONFIG.stream_worker_shutdown_timeout_seconds)
    except Exception as e:  # noqa: BLE001
        logger.warning("Worker join raised during shutdown: %s", e)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=CONFIG.stream_worker_shutdown_timeout_seconds)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=CONFIG.stream_worker_shutdown_timeout_seconds)
    _stream_worker = None


def _force_terminate_stream_worker_if_same(state: _StreamWorkerState) -> None:
    global _stream_worker
    with _stream_worker_lock:
        current = _stream_worker
        if current is None or current.process is not state.process:
            return
        current.cancel_event.set()
        proc = current.process
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=CONFIG.stream_worker_shutdown_timeout_seconds)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=CONFIG.stream_worker_shutdown_timeout_seconds)
        _stream_worker = None


def _ensure_stream_worker(name: str) -> _StreamWorkerState:
    global _stream_worker
    with _stream_worker_lock:
        state = _stream_worker
        if state is not None and state.model == name and state.process.is_alive():
            return state
        if state is not None:
            _shutdown_stream_worker_locked()
        ctx = mp.get_context(CONFIG.stream_worker_spawn_method)
        cmd_q = ctx.Queue()
        event_q = ctx.Queue()
        cancel_event = ctx.Event()
        # typeshed types `get_context(method: str)` generically as `BaseContext`,
        # which doesn't declare `Process` (only the spawn/fork/forkserver context
        # subclasses do) — runtime always resolves it correctly for any of the
        # three supported methods.
        proc = ctx.Process(  # type: ignore[attr-defined]
            target=_stream_worker_main,
            args=(name, str(_model_ir(name)), cmd_q, event_q, cancel_event),
            daemon=True,
        )
        proc.start()
        _stream_worker = _StreamWorkerState(
            model=name,
            process=proc,
            cmd_q=cmd_q,
            event_q=event_q,
            cancel_event=cancel_event,
        )
        return _stream_worker


def _loaded_model_name() -> str | None:
    state = _stream_worker
    if state is not None and state.process.is_alive():
        return state.model
    return _current["name"] if isinstance(_current["name"], str) else None


async def _watch_disconnect(
    req: Request, state: _StreamWorkerState, worker_done: threading.Event
) -> None:
    while not worker_done.is_set():
        if await req.is_disconnected():
            state.cancel_event.set()
            deadline = time.monotonic() + CONFIG.stream_cancel_grace_seconds
            while not worker_done.is_set() and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
            if not worker_done.is_set():
                await asyncio.to_thread(_force_terminate_stream_worker_if_same, state)
            return
        await asyncio.sleep(0.05)


def _discover_models() -> dict[str, dict]:
    models: dict[str, dict] = {}
    if not MODELS_DIR.exists():
        return models
    for p in MODELS_DIR.iterdir():
        if not p.is_dir():
            continue
        kind = _ir_kind(p)
        has_bin = (p / "openvino_model.bin").exists() or (
            p / "openvino_language_model.bin"
        ).exists()
        if kind in ("llm", "vlm") and has_bin:
            # Read context_length and max_output_tokens from config.json
            context_length = CONFIG.max_context
            max_output_tokens = CONFIG.default_output_tokens
            cfg = p / "config.json"
            if cfg.exists():
                try:
                    j = json.loads(cfg.read_text())
                    for key in (
                        "max_position_embeddings",
                        "max_position",
                        "model_max_length",
                        "seq_length",
                    ):
                        if key in j and isinstance(j[key], int):
                            context_length = int(j[key])
                            break
                    text_cfg = j.get("text_config")
                    if isinstance(text_cfg, dict):
                        for key in (
                            "max_position_embeddings",
                            "max_position",
                            "model_max_length",
                            "seq_length",
                        ):
                            if key in text_cfg and isinstance(text_cfg[key], int):
                                context_length = int(text_cfg[key])
                                break
                    for out_key in ("max_output_tokens", "max_new_tokens"):
                        if out_key in j and isinstance(j[out_key], int):
                            max_output_tokens = int(j[out_key])
                            break
                except Exception:  # noqa: BLE001, S110
                    pass
            context_length = min(context_length, CONFIG.max_context)
            # Never a ratio of context_length — a flat absolute default, capped only as a
            # safety net for a hypothetical future model with a smaller context than the default.
            max_output_tokens = min(
                max_output_tokens, CONFIG.default_output_tokens, context_length
            )
            models[p.name] = {
                "ir": p,
                "kind": kind,
                "context_length": context_length,
                "max_output_tokens": max_output_tokens,
            }
    return models


def _load(name: str):
    if _current["name"] == name and _current["pipe"] is not None:
        return _current["pipe"], _current["tok"]  # type: ignore[return-value]
    ir = _model_ir(name)
    pipe = _build_pipeline(name, ir)
    tok = _get_tokenizer(name)
    _current["name"] = name
    _current["pipe"] = pipe
    _current["tok"] = tok
    return pipe, tok


@app.get("/health")
def health():
    models = _discover_models()
    return {
        "status": "ok",
        "loaded": _loaded_model_name(),
        "models": sorted(models.keys()),
    }


@app.get("/v1/models")
def list_models():
    models = _discover_models()
    data = [
        {
            "id": m,
            "object": "model",
            "owned_by": "openvino",
            "context_length": info["context_length"],
            # modelsDiscovery needs BOTH context + output limits to show (Y%)
            "max_output_tokens": info.get(
                "max_output_tokens", CONFIG.default_output_tokens
            ),
        }
        for m, info in sorted(models.items())
    ]
    return {"object": "list", "data": data}


@app.get("/v1/model/info")
def model_info():
    """LiteLLM-shaped model info endpoint — consumed by the `opencode-models-
    discovery` plugin (`modelInfoFormat: "litellm"`) to auto-populate OpenCode's
    `reasoning` capability flag for every model, dynamically, from this server's
    own state. NEVER hand-list models in opencode.jsonc: this endpoint is the
    single source of truth — every Qwen3 IR served here supports reasoning, so
    every discovered model reports `supports_reasoning: true` unconditionally,
    with zero per-model config anywhere outside this function.
    """
    models = _discover_models()
    data = [
        {
            "model_name": m,
            "model_info": {
                "key": m,
                "mode": "chat",
                "max_input_tokens": info["context_length"],
                "max_output_tokens": info.get(
                    "max_output_tokens", CONFIG.default_output_tokens
                ),
                "supports_reasoning": True,
                "supports_function_calling": True,
            },
        }
        for m, info in sorted(models.items())
    ]
    return {"data": data}


@app.get("/v1/models/{model_id}")
def get_model(model_id: str):
    models = _discover_models()
    # Handle both with and without colon normalization
    candidates = [model_id, model_id.replace(":", "-"), model_id.replace("-", ":")]
    for cand in candidates:
        if cand in models:
            info = models[cand]
            return {
                "id": cand,
                "object": "model",
                "owned_by": "openvino",
                "context_length": info["context_length"],
                "max_output_tokens": info.get(
                    "max_output_tokens", CONFIG.default_output_tokens
                ),
            }
    norm = model_id.replace(":", "-").lower()
    for k, info in models.items():
        if k.lower() == norm or k.replace(":", "-").lower() == norm:
            return {
                "id": k,
                "object": "model",
                "owned_by": "openvino",
                "context_length": info["context_length"],
                "max_output_tokens": info.get(
                    "max_output_tokens", CONFIG.default_output_tokens
                ),
            }
    return JSONResponse(
        status_code=404, content={"error": f"model {model_id} not found"}
    )


@app.post("/v1/unload")
async def unload(req: Request):
    """Unload the currently loaded model (free VRAM)."""
    try:
        body = await req.json()
    except Exception:  # noqa: BLE001
        body = {}
    if not isinstance(body, dict):
        body = {}
    requested = body.get("model")
    loaded = _loaded_model_name()
    if loaded is None:
        return {"status": "ok", "unloaded": None, "message": "no model loaded"}
    # If a specific model was requested and it doesn't match the loaded one, 404
    if (
        requested
        and requested != loaded
        and requested.replace(":", "-") != loaded  # type: ignore[attr-defined]
        and loaded.replace(":", "-") != requested  # type: ignore[attr-defined]
    ):
        return JSONResponse(
            status_code=404,
            content={"error": f"model {requested} not loaded, currently {loaded}"},
        )
    unloaded = loaded
    await asyncio.to_thread(_shutdown_stream_worker_locked)
    _current["name"] = None
    _current["pipe"] = None
    _current["tok"] = None
    await asyncio.to_thread(gc.collect)
    return {"status": "ok", "unloaded": unloaded}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    # Every call below can block for a while (disk I/O, GPU pipeline construction,
    # tokenizing a large context) — none of it may run directly on the event loop
    # inside this `async def`, or it freezes ALL routes (including /health) for its
    # entire duration. Offload everything via asyncio.to_thread.
    discovered = await asyncio.to_thread(_discover_models)
    default_model = min(discovered.keys()) if discovered else ""
    model = body.get("model", default_model)
    if model not in discovered:
        return JSONResponse(
            status_code=404,
            content={
                "error": f"unknown model {model}",
                "available": sorted(discovered.keys()),
            },
        )
    messages = body.get("messages", [])
    tools = body.get("tools") or None
    stream = body.get("stream", False)
    max_tokens = int(body.get("max_tokens", 1024))
    temperature = float(body.get("temperature", 0.7))
    enable_thinking = bool(body.get("enable_thinking", CONFIG.default_enable_thinking))
    images = await asyncio.to_thread(_extract_images, messages)
    tok = await asyncio.to_thread(_get_tokenizer, model)
    # `tools=` is what actually teaches the model tool-calling exists and how to
    # format it (Qwen's template injects the <tools>/<tool_call> instructions
    # ONLY when this is passed) — omitting it silently produces a model that
    # narrates actions in plain text instead of ever emitting a real tool call.
    # `enable_thinking=False` skips Qwen3's verbose chain-of-thought — a fast
    # agent (Lite) should act, not ruminate for hundreds of tokens first.
    prompt = await asyncio.to_thread(
        tok.apply_chat_template,  # type: ignore[attr-defined]
        _template_messages(messages),
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = max_tokens
    config.do_sample = temperature > 0
    if config.do_sample:
        config.temperature = temperature
    if not stream:
        pipe, tok = await asyncio.to_thread(_load, model)  # type: ignore[assignment]
        # Non-blocking lock acquire + generate() offloaded to a thread — a blocking
        # call directly inside this `async def` would freeze the ENTIRE single-worker
        # event loop (all routes, including /health) for the whole generation duration.
        acquired = _gen_lock.acquire(blocking=False)
        if not acquired:
            return JSONResponse(
                status_code=429,
                content={"error": "model busy — try again momentarily", "model": model},
            )
        try:
            result = await asyncio.to_thread(
                _generate,
                pipe,
                prompt,
                config,
                images=images or None,
            )
        finally:
            _gen_lock.release()
        text = _result_text(result)
        parsed = await asyncio.to_thread(_parse_full, text)
        # Real usage for opencode's 490.9K (47%) · $59.11 display — was 0 (vide)
        prompt_tokens = await asyncio.to_thread(_token_count, tok, prompt)
        completion_tokens = await asyncio.to_thread(_token_count, tok, text)
        message: dict = {"role": "assistant"}
        if parsed["reasoning_content"]:
            # Standard `@ai-sdk/openai-compatible` interleaved-reasoning field —
            # same convention already handled for DeepSeek's thinking mode.
            message["reasoning_content"] = parsed["reasoning_content"]
        if parsed["tool_calls"]:
            message["content"] = parsed["content"] or None
            message["tool_calls"] = parsed["tool_calls"]
            finish_reason = "tool_calls"
        else:
            # OpenCode only creates a visible text part on a non-empty content
            # delta (`text-start` fires on non-null content) — a turn that put
            # everything in reasoning_content and left content empty vanishes
            # entirely in the TUI, and OpenCode retries the same turn forever
            # (known upstream pattern, e.g. opencode issue #37073). Never let
            # a turn end with truly nothing to show: fall back to the
            # reasoning itself rather than silently losing the turn.
            message["content"] = parsed["content"] or parsed["reasoning_content"]
            finish_reason = "stop"
        return {
            "id": "chatcmpl-openvino",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    # Streaming uses a dedicated worker subprocess per active model. Level 1:
    # client disconnect sets a shared cancel event so the worker returns
    # `StreamingStatus.CANCEL` on the next chunk and releases the lock quickly.
    # Level 2: if disconnect happens during prefill (before the first chunk,
    # when no streamer callback can run yet), the parent kills the worker
    # subprocess after a short grace period and lazily respawns it on the next
    # request — no whole-server restart, no permanently stuck `model busy`.
    acquired = _gen_lock.acquire(blocking=False)
    if not acquired:
        return JSONResponse(
            status_code=429,
            content={"error": "model busy — try again momentarily", "model": model},
        )
    state = await asyncio.to_thread(_ensure_stream_worker, model)
    state.cancel_event.clear()
    q: queue.Queue[dict | None] = queue.Queue()
    response_id = "chatcmpl-openvino"
    created = int(time.time())
    worker_done = threading.Event()
    worker_result: dict[str, object] = {
        "completion": "",
        "finish_reason": "stop",
        "cancelled": False,
        "error": None,
    }
    state.cmd_q.put(
        {
            "kind": "generate",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            # Picklable numpy arrays (reconstructed as ov.Tensor in the worker)
            "images": images,
        }
    )

    def bridge():
        try:
            while True:
                try:
                    msg = state.event_q.get(timeout=0.1)
                except queue.Empty:
                    if worker_done.is_set():
                        break
                    if state.cancel_event.is_set() and not state.process.is_alive():
                        worker_result["cancelled"] = True
                        worker_done.set()
                        break
                    continue
                kind = msg.get("kind")
                if kind == "event":
                    q.put(msg["event"])
                    continue
                if kind == "error":
                    worker_result["error"] = msg
                    worker_done.set()
                    break
                if kind == "done":
                    worker_result.update(msg)
                    worker_done.set()
                    break
        finally:
            q.put(None)
            try:
                _gen_lock.release()
            except RuntimeError:
                pass

    threading.Thread(target=bridge, daemon=True).start()
    asyncio.create_task(_watch_disconnect(req, state, worker_done))

    def gen():
        tool_call_seen = False
        content_seen = False
        reasoning_parts: list[str] = []
        tool_call_index = 0
        while True:
            event = q.get()
            if event is None:
                break
            if event["type"] == "content":
                content_seen = True
                delta = {"content": event["text"]}
            elif event["type"] == "reasoning":
                # Standard `@ai-sdk/openai-compatible` interleaved-reasoning
                # field — same convention already handled for DeepSeek.
                reasoning_parts.append(event["text"])
                delta = {"reasoning_content": event["text"]}
            else:  # tool_call — buffered whole, emitted as one complete delta
                tool_call_seen = True
                delta = {
                    "tool_calls": [
                        {
                            "index": tool_call_index,
                            "id": f"call_{uuid.uuid4().hex[:24]}",
                            "type": "function",
                            "function": {
                                "name": event["name"],
                                "arguments": json.dumps(event["arguments"]),
                            },
                        }
                    ]
                }
                tool_call_index += 1
            yield (
                "data: "
                + json.dumps(
                    {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {"index": 0, "delta": delta, "finish_reason": None}
                        ],
                    }
                )
                + "\n\n"
            )
        if not content_seen and not tool_call_seen and reasoning_parts:
            # OpenCode only creates a visible text part on a non-empty content
            # delta (`text-start` fires on non-null content) — a turn that put
            # everything in reasoning_content and streamed no real content
            # vanishes entirely in the TUI, and OpenCode retries the same turn
            # forever (known upstream pattern, e.g. opencode issue #37073).
            # Never let a turn end with truly nothing to show: fall back to
            # the reasoning itself rather than silently losing the turn.
            yield (
                "data: "
                + json.dumps(
                    {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "".join(reasoning_parts)},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                + "\n\n"
            )
        if worker_result["error"] and not content_seen and not tool_call_seen:
            err = worker_result["error"]  # type: ignore[assignment]
            yield (
                "data: "
                + json.dumps(
                    {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": (
                                        f"⚠️ {err['code']}: {err['message']}"
                                        if isinstance(err, dict)
                                        else "⚠️ generation error"
                                    )
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                + "\n\n"
            )
            content_seen = True
        completion = str(worker_result["completion"])
        prompt_tokens = _token_count(tok, prompt)
        completion_tokens = _token_count(tok, completion)
        final_finish_reason = str(
            worker_result["finish_reason"]
            or ("tool_calls" if tool_call_seen else "stop")
        )
        yield (
            "data: "
            + json.dumps(
                {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": final_finish_reason,
                        }
                    ],
                }
            )
            + "\n\n"
        )
        # OpenAI-compatible streaming usage: a final empty-choices chunk.
        # OpenCode consumes this event to render the live context indicator (xK (y%) · $…).
        yield (
            "data: "
            + json.dumps(
                {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
            )
            + "\n\n"
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


def main() -> None:
    import uvicorn

    uvicorn.run(
        app, host=CONFIG.host, port=CONFIG.port, log_level=CONFIG.log_level.lower()
    )


if __name__ == "__main__":
    main()
