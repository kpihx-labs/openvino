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
```

## Development

```bash
make check   # ruff + pyright + pytest
make help
```
