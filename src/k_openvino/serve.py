"""OpenVINO GenAI — OpenAI-compatible chat server (local sovereign LLM backend)."""

from __future__ import annotations

import json
import os
import queue
import threading

import openvino_genai
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoTokenizer

from k_openvino.config import CONFIG

MODELS_DIR = CONFIG.models_dir

app = FastAPI(title="openvino-serve")
_current: dict[str, object] = {"name": None, "pipe": None, "tok": None}


def _discover_models() -> dict[str, dict]:
    models: dict[str, dict] = {}
    if not MODELS_DIR.exists():
        return models
    for p in MODELS_DIR.iterdir():
        if not p.is_dir():
            continue
        has_ir = (p / "openvino_model.xml").exists() and (
            p / "openvino_model.bin"
        ).exists()
        if has_ir:
            # Read context_length from config.json (max_position_embeddings → 32768/131072)
            context_length = 32768  # default Qwen3
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
                except Exception:  # noqa: BLE001, S110
                    pass
            models[p.name] = {"ir": p, "context_length": context_length}
    return models


def _load(name: str):
    if _current["name"] == name and _current["pipe"] is not None:
        return _current["pipe"], _current["tok"]  # type: ignore[return-value]
    discovered = _discover_models()
    if name not in discovered:
        raise RuntimeError(
            f"Model not found: {name} (available: {list(discovered.keys())})"
        )
    ir = discovered[name]["ir"]
    device = os.environ.get("OPENVINO_DEVICE", "GPU")
    # GPU Arc OOM even for 0.6B (5.4G peak + 3G swap) → LATENCY + 1 stream to cut VRAM
    cfg: dict[str, str] = {}
    if device == "GPU":
        cfg = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1"}
    try:
        pipe = (
            openvino_genai.LLMPipeline(str(ir), device, cfg)
            if cfg
            else openvino_genai.LLMPipeline(str(ir), device)
        )
    except Exception:  # noqa: BLE001
        pipe = openvino_genai.LLMPipeline(str(ir), "CPU")
    tok = AutoTokenizer.from_pretrained(str(ir))
    _current["name"] = name
    _current["pipe"] = pipe
    _current["tok"] = tok
    return pipe, tok


@app.get("/health")
def health():
    models = _discover_models()
    return {"status": "ok", "loaded": _current["name"], "models": sorted(models.keys())}


@app.get("/v1/models")
def list_models():
    models = _discover_models()
    data = [
        {
            "id": m,
            "object": "model",
            "owned_by": "openvino",
            "context_length": info["context_length"],
        }
        for m, info in sorted(models.items())
    ]
    return {"object": "list", "data": data}


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
            }
    norm = model_id.replace(":", "-").lower()
    for k, info in models.items():
        if k.lower() == norm or k.replace(":", "-").lower() == norm:
            return {
                "id": k,
                "object": "model",
                "owned_by": "openvino",
                "context_length": info["context_length"],
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
    loaded: str | None = _current["name"]  # type: ignore[assignment]
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
    _current["name"] = None
    _current["pipe"] = None
    _current["tok"] = None
    import gc

    gc.collect()
    return {"status": "ok", "unloaded": unloaded}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    discovered = _discover_models()
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
    stream = body.get("stream", False)
    max_tokens = int(body.get("max_tokens", 1024))
    temperature = float(body.get("temperature", 0.7))
    pipe, tok = _load(model)  # type: ignore[assignment]
    prompt = tok.apply_chat_template(  # type: ignore[attr-defined]
        messages, tokenize=False, add_generation_prompt=True
    )
    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = max_tokens
    config.do_sample = temperature > 0
    if config.do_sample:
        config.temperature = temperature
    if not stream:
        result = pipe.generate(prompt, config)  # type: ignore[union-attr]
        text = result.text if hasattr(result, "text") else str(result)
        return {
            "id": "chatcmpl-openvino",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    q: queue.Queue[str | None] = queue.Queue()

    def streamer(subword: str) -> bool:
        q.put(subword)
        return False

    def run():
        pipe.generate(prompt, config, streamer=streamer)  # type: ignore[union-attr]
        q.put(None)

    threading.Thread(target=run, daemon=True).start()

    def gen():
        while True:
            tok_chunk = q.get()
            if tok_chunk is None:
                break
            yield (
                "data: "
                + json.dumps({"choices": [{"delta": {"content": tok_chunk}}]})
                + "\n\n"
            )
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=CONFIG.host, port=CONFIG.port)


if __name__ == "__main__":
    main()
