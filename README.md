# openvino — Sovereign Local LLM (OpenVINO GenAI)

OpenAI-compatible server + Ollama-like CLI for Qwen3 and other text models exported to OpenVINO IR. Intel Arc iGPU (Vulkan) ready, 100% `uv`, `typer` + `rich`, no `sh` wrappers.

## Stack

- Python 3.12, `uv`, `openvino-genai==2026.3.1`, `optimum-intel==2.1.0`, `transformers`, `fastapi` + `uvicorn`, `huggingface_hub`, `httpx`, `typer`, `rich`, `python-dotenv`
- `src/openvino/config.py` — plain frozen dataclass (no YAML)
- `CONTRACT.md` — source of truth

## Install

```bash
# From project root
uv sync --python 3.12
make uv-link   # or: uv tool install --editable . --force
openvino --help
```

Or directly:

```bash
uv tool install --editable . --force
```

## Usage

```bash
openvino ls                          # list local IR models
openvino ps                          # show loaded model
openvino pull qwen3:1.7b            # HF Qwen/Qwen3-1.7B → IR
openvino pull Qwen/Qwen3-4B         # HF id direct
openvino rm qwen3:1.7b
openvino search qwen3
openvino show qwen3:1.7b
openvino run qwen3:1.7b "Bonjour"

# Server (systemd)
systemctl --user enable --now openvino.service
curl -s http://127.0.0.1:11437/health | jq
curl -s http://127.0.0.1:11437/v1/models | jq

# Env
OPENVINO_HOME=~/.local/share/openvino  # models root
OPENVINO_URL=http://127.0.0.1:11437
OPENVINO_HOST=127.0.0.1:11437   # or just :11437 (Ollama-style, host optional)
OPENVINO_DEVICE=GPU             # default device for every model
```

## Per-model overrides

Every model directory may carry a `server.overrides.json` file. It travels with
the model, and any key it defines wins over the global `config.py` defaults, so
one model can be tuned without touching the others:

```
~/.local/share/openvino/models/gemma-4-12b-it/server.overrides.json
```

```jsonc
{
  // Generation behaviour
  "enable_thinking": false,        // drop the verbose reasoning block (example: gemma)
  "max_context": 16384,            // effective prompt ceiling for THIS model
  "default_output_tokens": 4096,   // advertised response ceiling

  // Execution
  "device": "GPU",                 // GPU | CPU | AUTO
  "performance_hint": "LATENCY",
  "num_streams": 1,
  "kv_cache_precision": "u8",      // shrinks the KV cache when supported

  // Anything else passed straight to compile_model
  "extra_properties": { "DYNAMIC_QUANTIZATION_GROUP_SIZE": 0 }
}
```

Resolution order, strongest first: **request body** (`enable_thinking`,
`max_tokens`) > **model override** > **`config.py`**. An override may lower the
context but never exceed what the model itself declares. A missing, malformed, or
non-object file is ignored with a warning instead of breaking discovery.

The effective limits are published per model, so a client always sees the truth:

```bash
curl -s http://127.0.0.1:11437/v1/model/info | jq
# max_input_tokens / max_output_tokens / device per model
```

A request whose prompt reaches the effective `max_context` is rejected with HTTP
413 `context_length_exceeded` (with `prompt_tokens`, `context_limit`, and an
action telling the caller to compact), and `max_tokens` is clamped so the prompt
plus the response always fits.

## Response metrics

Both streaming and non-streaming responses carry an execution `metrics` block, so
a silent GPU to CPU fallback or a slow prefill is visible instead of guessed:

```jsonc
"metrics": {
  "device": "GPU",        // device that really served the request
  "prefill_ms": 1443,     // generate() start -> first token (excludes model load)
  "total_ms": 1758,
  "completion_tokens": 4,
  "tokens_per_s": 12.73   // decode rate
}
```

While a streaming request waits for its first token (model load and prompt
prefill), the server emits a `: keepalive` SSE comment every
`heartbeat_seconds` (default 10), so a slow start is never mistaken for a hang.
If the worker process dies mid-request, the client receives an explicit
`WorkerDied` error and a `[DONE]` terminator instead of an endless wait.

## Development

```bash
make check   # ruff + pyright + pytest
make help
```
