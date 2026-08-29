# CHANGELOG

## 0.2.0 — 2026-08-29

- serve: async-safe tool-calling + reasoning protocol (`tools=`/`enable_thinking` wiring into `apply_chat_template`, `_StreamParser` shared `<think>`/`<tool_call>` grammar for streaming+non-streaming, `reasoning_content` field, anti-loop fallback when a turn would otherwise end empty)
- serve: dedicated per-model subprocess for streaming generation — real client-disconnect cancellation (graceful `cancel_event` then forced terminate/kill), global OpenAI-style JSON exception handler
- serve: dynamic capability discovery via `GET /v1/models` + `GET /v1/model/info` (litellm format, `context_length` from `config.json`) for OpenCode auto-discovery
- serve: real `usage.prompt_tokens`/`completion_tokens` (was always `0`)
- serve: GPU default port `11437` (was `11436`, conflicted with the `ollama6` DxO tunnel) + flexible `LATENCY`/`NUM_STREAMS` GPU config with CPU fallback on any machine
- serve: single binary `openvino` (removed `KNOWN`), `POST /v1/unload`, non-blocking server startup
- cli: `_hf_accurate_size()` replaces unreliable `safetensors.total` (real sibling-file sizes, `transformers` checkpoint convention) in both `pull` and `search`
- cli: `HF_HOME` override no longer hides `HF_TOKEN` (now propagates + copies token to the custom cache)
- cli: every `--option` gained a short form (`-H`/`-m`/`-s`/`-P`)
- config: `max_context`/`default_output_tokens`/`default_enable_thinking` caps, stream worker spawn/cancel-grace/shutdown timeouts (`stream_worker_spawn_method` now `Literal["spawn","fork","forkserver"]`)
- tests: `tests/test_serve.py` (13 tests — streaming, tool-calls, reasoning fallback, disconnect handling)
- fix: `_StreamWorkerState` fields properly typed (`BaseProcess`/`MPQueue`/`MPEvent` instead of bare `object`) — 0 pyright errors
- docs: fixed stale port `11436` → `11437` in `README.md`

## 0.1.0 — 2026-08-27

- Initial release — `uv init --package openvino` + k-project template (Makefile, CONTRACT, AGENTS.md, config.py)
- OpenVINO GenAI server (`serve.py`, FastAPI, dynamic model discovery, `OPENVINO_HOME`, GPU→CPU fallback)
- Ollama-like CLI (`cli.py`, Typer + Rich: `ls/ps/pull/rm/search/show/run`, `OPENVINO_URL`/`HOST`)
- `pyproject.toml` 100% `uv` (`uv_build`, no hatch, `typer` + `rich` + `openvino-genai` etc.)
- `uv tool install --editable .` ready, no `sh` wrapper
