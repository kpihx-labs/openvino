# CHANGELOG

## 0.4.0 - 2026-09-16

- serve: per-model overrides via `<model_dir>/server.overrides.json`: any key wins over the global default, so a model carries its own tuning wherever it is copied. Resolution order: request body > model override > `config.py`. Supported keys: `enable_thinking`, `max_context`, `default_output_tokens`, `device`, `performance_hint`, `num_streams`, `kv_cache_precision`, `extra_properties`. An override may lower the context but never exceed what the model itself declares
- serve: FIX context limit is now ENFORCED, not merely advertised. A prompt at or above the effective `max_context` is rejected with HTTP 413 `context_length_exceeded` (prompt_tokens, context_limit, and a compact/shorten action), and `max_tokens` is clamped so prompt plus response always fits
- serve: FIX stream worker death no longer hangs the client forever. The bridge now detects `not process.is_alive()`, reports a `WorkerDied` error through the stream, releases the generation lock, and finishes with `[DONE]`. Previously a worker killed without sending a result left the bridge spinning in `except queue.Empty`, holding the lock and starving every later request with a permanent 429
- serve: SSE keepalive (`: keepalive`) every `heartbeat_seconds` (default 10) while a request waits for its first token, so a slow model load is no longer indistinguishable from a hang
- serve: `metrics` block on both streaming and non-streaming responses: `device` (real execution device, so a silent GPU to CPU fallback is visible), `prefill_ms`, `total_ms`, `completion_tokens`, `tokens_per_s` (decode rate, computed from first-token time rather than end-to-end)
- serve: `device` exposed on `/v1/models` and `/v1/model/info`; pipeline build logs the resolved device and properties at INFO, and the CPU fallback logs at ERROR
- serve: shutdown-time worker death logged as WARNING (SIGTERM is orderly) instead of ERROR; SIGKILL still logs at ERROR
- config: add `heartbeat_seconds` and `override_filename`
- systemd: `MemoryHigh=18G`, `MemoryMax=20G`, `OOMPolicy=stop`: contains the service so an overrun stops THIS unit instead of letting the kernel OOM killer choose across the whole user session (it previously killed Edge, browser-proxy, Bitwarden, tmux panes and user@1000)
- tests: 21 (was 13): per-model override resolution, override cannot inflate declared context, malformed override tolerance, 413 guard, dead-worker error path, metrics block

## 0.3.0 — 2026-09-15

- cli: purge baked-in model-family whitelists — `OPENVINO_COMPATIBLE_TYPES` is optional restrict-only (unset = any architecture); multimodal export task chosen from HF config STRUCTURE (`vision_config` / architecture markers / processor siblings), never from a name list
- cli: `pull --task` / `--weight-format` overrides; multimodal defaults to `image-text-to-text` + `OPENVINO_WEIGHT_FORMAT` (default `int4`); IR detection covers both `openvino_model.xml` and `openvino_language_model.xml`
- cli: architecture-agnostic `ACTIVATIONS_SCALE_FACTOR` 8.0→`OPENVINO_ACTIVATIONS_SCALE` (default 64.0) post-export patch when present
- serve: FIX — VLMPipeline text-only/image generate uses `generation_config=` kwarg (positional GenerationConfig raises TypeError on openvino_genai 2026.3.1)
- cli: FIX — stop wiping `~/.cache/hf-export` after pull (kept a 23.9G re-download after failed export); set disk-backed `TMPDIR=~/.cache/openvino-export-tmp` so INT4 FP16 intermediates do not ENOSPC on `/tmp` tmpfs (`basic_ios::clear: iostream error` / optimum-intel#1707)
- serve: general VLM path — IR layout selects `VLMPipeline` vs `LLMPipeline`; OpenAI `image_url` parts (data URI / local path) → tensors; LLM models unchanged
- deps: `transformers>=5.10,<5.11` via uv override (optimum-intel 2.1.0 still declares `<5.6`); add `pillow` + `numpy`; Makefile install/link pass `--overrides`
- serve: FIX VLM image path: `_template_messages` preserves multimodal parts for chat template (tag inside user turn, not before bos); `DYNAMIC_QUANTIZATION_GROUP_SIZE=0` moved from VLM to LLM only (breaks Gemma4 VLM image: 0 tokens, empty output)

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
