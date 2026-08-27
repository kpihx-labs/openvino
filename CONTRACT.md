# CONTRACT

This document is the non-negotiable source of truth for the project contract.

If implementation, wrappers, or docs conflict with this contract, this contract wins.

---

## 1) Metadata

- `schema_version`: `openvino.contract.v1`
- `contract_version`: `1.0.0`

Keep the same metadata in `pyproject.toml` under `[tool.openvino.contract]`.

---

## 2) Conventions

- Envelope shape: JSON over HTTP (OpenAI-compatible `/v1/chat/completions`, `/v1/models`, `/health`)
- Naming: model id = directory name in `OPENVINO_HOME/models/<name>` (e.g. `qwen3-1.7b` ↔ `qwen3:1.7b`)
- Time units: `max_tokens` int, IDs: `model` string

---

## 3) Shared types

- `GET /health` → `{status, loaded, models}`
- `GET /v1/models` → `{object:"list", data:[{id, object:"model", owned_by:"openvino"}]}`
- `POST /v1/chat/completions` → OpenAI chat completions (messages, model, max_tokens, temperature, stream)

---

## 4) Error model

- `404` unknown model `{error, available}`
- `500` generation failure `{error}`

---

## 5) Internal API

For each operation:

- **ls** → list local IR models (server or filesystem fallback)
- **ps** → show loaded model
- **pull <name>** → `optimum-cli export openvino --model <HF> --task text-generation-with-past models/<local>`
- **rm <name>** → `rm -rf models/<name>`
- **search <query>** → Hugging Face Hub search
- **show <name>** → list model files
- **run <name> "prompt"` → `POST /v1/chat/completions`

---

## 6) Transport mappings

Map each CLI command to exactly one internal API operation; HTTP transport is OpenAI-compatible.

---

## 7) Compliance rules

- Wrappers must not alter business semantics.
- Wrappers must not remap canonical error codes.
- Any breaking change requires `schema_version` major bump.
