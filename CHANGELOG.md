# CHANGELOG

## 0.1.0 — 2026-08-27

- Initial release — `uv init --package openvino` + k-project template (Makefile, CONTRACT, AGENTS.md, config.py)
- OpenVINO GenAI server (`serve.py`, FastAPI, dynamic model discovery, `OPENVINO_HOME`, GPU→CPU fallback)
- Ollama-like CLI (`cli.py`, Typer + Rich: `ls/ps/pull/rm/search/show/run`, `OPENVINO_URL`/`HOST`)
- `pyproject.toml` 100% `uv` (`uv_build`, no hatch, `typer` + `rich` + `openvino-genai` etc.)
- `uv tool install --editable .` ready, no `sh` wrapper
