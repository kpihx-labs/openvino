"""OpenVINO CLI — Ollama-like front-end for the local OpenVINO GenAI server."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import httpx
import typer
from rich.console import Console
from rich.table import Table

from k_openvino.config import CONFIG

app = typer.Typer(help="OpenVINO — local LLM (Ollama-like CLI)", add_completion=False)
server_app = typer.Typer(help="Manage the OpenVINO server (systemd)")
console = Console()


def _url() -> str:
    raw = (
        os.environ.get("OPENVINO_URL") or os.environ.get("OPENVINO_HOST") or CONFIG.url
    )
    raw = raw.strip()
    if raw.startswith(("http://", "https://")):
        return raw.rstrip("/")
    if raw.startswith(":"):
        raw = "127.0.0.1" + raw
    if ":" in raw and not raw.startswith("http"):
        return f"http://{raw}"
    return raw


def _models_dir() -> Path:
    return CONFIG.models_dir


def _local_name(arg: str) -> str:
    if "/" in arg:
        return arg.split("/")[-1].lower().replace("_", "-")
    return arg.lower().replace(":", "-")


def _hf_id(arg: str, hf_opt: str | None) -> str:
    if hf_opt:
        return hf_opt
    return arg


def _hf_accurate_size(api: object, hf_id: str) -> int | None:
    """Real download size in bytes for the checkpoint `optimum-cli` will actually load.

    `info.safetensors.total` is unreliable and can undercount by 2x+ (e.g.
    mistralai/Mistral-7B-Instruct-v0.3: reported 6.8G, real download 14.5G —
    the pre-flight disk/RAM safety check passed on the wrong number and the
    export later crashed with ENOSPC). Naively summing every sibling weight
    file is also wrong: some repos ship duplicate full-weight distributions
    in different formats (that same repo ALSO carries a single 13.5G
    `consolidated.safetensors` for `mistral-inference`, on top of the 13.5G
    `model-0000X-of-00003.safetensors` shards `transformers`/`optimum-cli`
    actually loads — a naive sum double-counts to ~27G). Target the exact
    `transformers` checkpoint convention `optimum-cli` follows, in priority
    order, so both `pull` and `search` get the one real number.
    """
    try:
        info = api.model_info(hf_id, files_metadata=True)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return None
    siblings = {s.rfilename: s.size for s in (info.siblings or []) if s.size}
    shard_sizes = [
        size
        for name, size in siblings.items()
        if name.startswith("model-") and name.endswith(".safetensors")
    ]
    if shard_sizes:
        return sum(shard_sizes)
    if "model.safetensors" in siblings:
        return siblings["model.safetensors"]
    shard_sizes = [
        size
        for name, size in siblings.items()
        if name.startswith("pytorch_model-") and name.endswith(".bin")
    ]
    if shard_sizes:
        return sum(shard_sizes)
    if "pytorch_model.bin" in siblings:
        return siblings["pytorch_model.bin"]
    # Fallback — no recognized transformers convention found, best-effort sum
    # (may overcount if the repo carries duplicate formats).
    weight_exts = (".safetensors", ".bin", ".pt", ".pth")
    sizes = [size for name, size in siblings.items() if name.endswith(weight_exts)]
    return sum(sizes) if sizes else None


@app.command("ls")
def ls() -> None:
    """List local IR models."""
    url = _url()
    try:
        r = httpx.get(f"{url}/v1/models", timeout=5.0)
        if r.status_code == 200:
            data = r.json().get("data", [])
            if data:
                table = Table(title=f"OpenVINO models @ {url}")
                table.add_column("NAME", style="cyan")
                table.add_column("SIZE", justify="right")
                table.add_column("MODIFIED")
                for m in data:
                    name = m["id"]
                    p = _models_dir() / name
                    if not p.exists():
                        p = _models_dir() / name.replace(":", "-")
                    size = ""
                    modified = ""
                    if p.exists():
                        total = sum(
                            f.stat().st_size for f in p.rglob("*") if f.is_file()
                        )
                        size = (
                            f"{total / 1e9:.2f} GB"
                            if total > 1e9
                            else f"{total / 1e6:.0f} MB"
                        )
                        modified = str(p.stat().st_mtime)[:19]
                    table.add_row(name, size, modified)
                console.print(table)
                return
    except Exception:  # noqa: BLE001, S110
        pass
    md = _models_dir()
    if not md.exists() or not any(md.iterdir()):
        console.print(
            f"[yellow]No models in {md}[/yellow] (server {url} not reachable or empty)"
        )
        console.print("Try: openvino pull <HF_ID>  (e.g. Qwen/Qwen3-1.7B)")
        return
    table = Table(title=f"OpenVINO models (filesystem) @ {md}")
    table.add_column("NAME", style="cyan")
    table.add_column("SIZE", justify="right")
    for p in sorted(md.iterdir()):
        if p.is_dir() and (p / "openvino_model.xml").exists():
            total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            size = f"{total / 1e9:.2f} GB" if total > 1e9 else f"{total / 1e6:.0f} MB"
            table.add_row(p.name, size)
    console.print(table)


@app.command("ps")
def ps() -> None:
    """Show running model (like ollama ps)."""
    url = _url()
    try:
        r = httpx.get(f"{url}/health", timeout=5.0)
        r.raise_for_status()
        j = r.json()
        console.print(f"[bold]Server:[/bold] {url}  — status {j.get('status')}")
        loaded = j.get("loaded")
        models = j.get("models", [])
        if loaded:
            console.print(f"[bold]Loaded:[/bold] [green]{loaded}[/green]")
        else:
            console.print("[dim]No model loaded[/dim]")
        if models:
            console.print(f"[bold]Available:[/bold] {', '.join(models)}")
        else:
            console.print("[yellow]No models installed[/yellow]")
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Cannot reach server at {url}: {e}[/red]")
        console.print(
            f"Check: systemctl --user status openvino.service  or  OPENVINO_URL={url}"
        )


@app.command("pull")
def pull(
    name: str = typer.Argument(
        ..., help="Model name (e.g. qwen3:1.7b) or HF id (Qwen/Qwen3-1.7B)"
    ),
    hf: str = typer.Option(None, "--hf", "-H", help="Override HF repo id"),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force pull even if not safe (à vos risques et périls)",
    ),
) -> None:
    """Export a Hugging Face model to OpenVINO IR (like ollama pull) — strict by default, safe only."""
    hf_id = _hf_id(name, hf)
    local = _local_name(name if "/" not in name else hf_id)
    dest = _models_dir() / local
    if dest.exists() and (dest / "openvino_model.xml").exists():
        console.print(
            f"[yellow]{local} already exists at {dest} — skip (rm first to re-pull)[/yellow]"
        )
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    # ── Strict, flexible, dynamic safety checks (disk/RAM/GPU) — block by default if not safe ──
    if not force:
        # PC caps (dynamic, no hardcoding)
        import shutil as _shutil2

        # Disk avail for models + HF cache (conservative: need 2× model size + 2GB margin)
        models_dir = _models_dir()
        hf_cache_dir = Path.home() / ".cache/hf-export"
        try:
            models_avail = (
                _shutil2.disk_usage(models_dir).free
                if models_dir.exists()
                else _shutil2.disk_usage(Path.home()).free
            )
            cache_avail = (
                _shutil2.disk_usage(hf_cache_dir).free
                if hf_cache_dir.exists()
                else _shutil2.disk_usage(Path.home()).free
            )
            disk_avail = min(models_avail, cache_avail)
        except Exception:  # noqa: BLE001
            disk_avail = 0
        # RAM avail
        ram_avail = 0
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        ram_avail = int(line.split()[1]) * 1024  # kB → bytes
                        break
        except Exception:  # noqa: BLE001, S110
            pass
        # GPU — dynamic, not hardcoded for one machine (shared memory = RAM on Arc, else VRAM)
        gpu_name = "unknown"
        try:
            lspci = subprocess.run(
                ["lspci"], capture_output=True, text=True, timeout=3, check=False
            )
            if "Intel" in lspci.stdout and "Arc" in lspci.stdout:
                gpu_name = "Intel Arc"
            elif "Intel" in lspci.stdout:
                gpu_name = "Intel iGPU"
        except Exception:  # noqa: BLE001, S110
            pass
        # Model caps via HF (size + type) — strict but flexible, dynamic (not hardcoded for one machine)
        model_size = None
        model_type = None
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            model_size = _hf_accurate_size(api, hf_id)
            info = api.model_info(hf_id)
            # Type from config
            if info.config and isinstance(info.config, dict):
                model_type = info.config.get("model_type")
            elif hasattr(info, "cardData") and info.cardData:
                # fallback: try to get from cardData
                pass
        except Exception:  # noqa: BLE001, S110
            pass
        # Whitelist — dynamic, not hardcoded for one machine (override via OPENVINO_COMPATIBLE_TYPES)
        compatible_types = set(
            os.environ.get(
                "OPENVINO_COMPATIBLE_TYPES",
                "qwen3,qwen2,qwen,llama,mistral,gemma,phi,gpt_neox,chatglm",
            ).split(",")
        )
        if model_type and model_type not in compatible_types:
            console.print(
                f"[red]✗ pull bloqué — model_type '{model_type}' non supporté sur Arc (whitelist: {', '.join(sorted(compatible_types))})[/red]"
            )
            console.print(
                "[dim]Utilise --force pour forcer (à vos risques et périls)[/dim]"
            )
            sys.exit(1)
        # Size checks (conservative: need disk 2× size + RAM 1.5× size)
        if model_size:
            # Dynamic, not hardcoded for one machine — override via OPENVINO_* env
            disk_factor = float(os.environ.get("OPENVINO_DISK_FACTOR", "2.0"))
            ram_factor = float(os.environ.get("OPENVINO_RAM_FACTOR", "1.5"))
            disk_margin = int(os.environ.get("OPENVINO_DISK_MARGIN_GB", "2")) * 1024**3
            ram_margin = int(os.environ.get("OPENVINO_RAM_MARGIN_GB", "2")) * 1024**3
            need_disk = int(model_size * disk_factor) + disk_margin
            need_ram = int(model_size * ram_factor) + ram_margin
            size_gb = model_size / 1024**3
            disk_gb = disk_avail / 1024**3
            ram_gb = ram_avail / 1024**3
            # Disk check
            if disk_avail < need_disk:
                console.print(
                    f"[red]✗ pull bloqué — disque insuffisant: {size_gb:.1f}G model → besoin ~{need_disk / 1024**3:.1f}G, dispo {disk_gb:.1f}G[/red]"
                )
                console.print(
                    f"[dim]Modèle: {hf_id} ({size_gb:.1f}G), disque dispo: {disk_gb:.1f}G (besoin 2× + 2G). Nettoie ou utilise --force[/dim]"
                )
                console.print(
                    "[dim]Hier Qwen/Qwen3-8B (16G) a rempli le disque (5G dispo) et a OOM le PC — bloqué par défaut pour sûreté.[/dim]"
                )
                sys.exit(1)
            # RAM check (Arc shares RAM, so RAM ≈ VRAM)
            if ram_avail and ram_avail < need_ram:
                console.print(
                    f"[red]✗ pull bloqué — RAM insuffisante: {size_gb:.1f}G model → besoin ~{need_ram / 1024**3:.1f}G, dispo {ram_gb:.1f}G ({gpu_name})[/red]"
                )
                console.print(
                    "[dim]Utilise --force pour forcer, mais risque OOM et plantage PC (comme hier avec 8B)[/dim]"
                )
                sys.exit(1)
            console.print(
                f"[dim]✓ Check strict: {size_gb:.1f}G model, disque {disk_gb:.1f}G, RAM {ram_gb:.1f}G, GPU {gpu_name} — sûr[/dim]"
            )
    # Hint if unauthenticated (rate-limit causes 0% + incomplete total)
    if (
        not os.environ.get("HF_TOKEN")
        and not (Path.home() / ".cache/huggingface/token").exists()
    ):
        console.print(
            "[yellow]Note: HF_TOKEN not set — unauthenticated pull will be rate-limited (1-2 MB/s) and show `incomplete total` + `Fetching 0/5` until first shard completes.[/yellow]"
        )
        console.print(
            "[dim]Fix: `hf auth login` (nouveau) ou `HF_TOKEN` dans ~/.agents/.env[/dim]"
        )
    # Locate optimum-cli robustly (project venv, uv tool venv, or PATH)
    import shutil as _shutil

    which = _shutil.which("optimum-cli")
    candidates: list[Path] = [
        Path.home() / "KpihX-Labs/AI/openvino/.venv/bin/optimum-cli",
        Path.home() / ".local/share/uv/tools/k-openvino/bin/optimum-cli",
        Path.home() / ".local/share/openvino/venv/bin/optimum-cli",
    ]
    if which:
        candidates.append(Path(which))
    optimum = next((p for p in candidates if p.exists() and p.is_file()), None)
    use_uv_run = optimum is None
    if use_uv_run:
        cmd = [
            "uv",
            "run",
            "--project",
            str(Path.home() / "KpihX-Labs/AI/openvino"),
            "optimum-cli",
            "export",
            "openvino",
            "--model",
            hf_id,
            "--task",
            "text-generation-with-past",
            str(dest),
        ]
    else:
        cmd = [
            str(optimum),
            "export",
            "openvino",
            "--model",
            hf_id,
            "--task",
            "text-generation-with-past",
            str(dest),
        ]
    console.print(f"[bold]Pulling[/bold] {hf_id} → [cyan]{dest}[/cyan]")
    console.print(f"[dim]{' '.join(cmd)}[/dim]")
    env = os.environ.copy()
    hf_cache = Path.home() / ".cache/hf-export"
    env["HF_HOME"] = str(hf_cache)
    # Fix: HF_HOME override hides token at ~/.cache/huggingface/token → ensure HF_TOKEN is set
    if not env.get("HF_TOKEN"):
        token_path = Path.home() / ".cache/huggingface/token"
        if token_path.exists():
            try:
                env["HF_TOKEN"] = token_path.read_text().strip()
            except Exception:  # noqa: BLE001, S110
                pass
        # Also copy token files to custom HF_HOME for huggingface_hub's file lookup
        try:
            hf_cache.mkdir(parents=True, exist_ok=True)
            for src_name in ("token", "stored_tokens"):
                src = Path.home() / ".cache/huggingface" / src_name
                dst = hf_cache / src_name
                if src.exists() and not dst.exists():
                    shutil.copy(src, dst)
        except Exception:  # noqa: BLE001, S110
            pass
    try:
        subprocess.run(cmd, check=True, env=env)
        console.print(
            f"[green]✓ {local} ready[/green] — try: openvino ls  &&  curl {_url()}/v1/models"
        )
    except subprocess.CalledProcessError as e:
        console.print(f"[red]pull failed: {e}[/red]")
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        sys.exit(1)
    finally:
        if hf_cache.exists():
            shutil.rmtree(hf_cache, ignore_errors=True)


@app.command("rm")
def rm_cmd(name: str = typer.Argument(..., help="Model name to remove")) -> None:
    """Remove a local IR model (like ollama rm)."""
    candidates = [name, name.replace(":", "-"), name.replace("-", ":")]
    for cand in candidates:
        p = _models_dir() / cand
        if p.exists():
            console.print(f"Removing [cyan]{cand}[/cyan] at {p}")
            shutil.rmtree(p)
            console.print(f"[green]✓ removed {cand}[/green]")
            return
    p = _models_dir() / _local_name(name)
    if p.exists():
        shutil.rmtree(p)
        console.print(f"[green]✓ removed {_local_name(name)}[/green]")
        return
    console.print(f"[red]Model not found: {name} in {_models_dir()}[/red]")
    sys.exit(1)


@app.command("search")
def search(
    query: str = typer.Argument(..., help="Search Hugging Face for models"),
    all: bool = typer.Option(
        False, "--all", "-a", help="Show all, even incompatible (yellow with reason)"
    ),
    limit: int = typer.Option(
        20, "--limit", "-n", help="Number of results (default 20, e.g. 100)"
    ),
    max_params: str = typer.Option(
        None,
        "--max-params",
        "-m",
        help="Max params (e.g. 1.7B, 8B, 5B) — filter by model size",
    ),
) -> None:
    """Search Hugging Face Hub for models (online) — strict by default, safe only."""
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        # PC caps for dynamic checks (same as pull) — flexible, not hardcoded for one machine
        import shutil as _shutil2

        models_dir = _models_dir()
        hf_cache_dir = Path.home() / ".cache/hf-export"
        try:
            models_avail = (
                _shutil2.disk_usage(models_dir).free
                if models_dir.exists()
                else _shutil2.disk_usage(Path.home()).free
            )
            cache_avail = (
                _shutil2.disk_usage(hf_cache_dir).free
                if hf_cache_dir.exists()
                else _shutil2.disk_usage(Path.home()).free
            )
            disk_avail = min(models_avail, cache_avail)
        except Exception:  # noqa: BLE001
            disk_avail = 0
        ram_avail = 0
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        ram_avail = int(line.split()[1]) * 1024
                        break
        except Exception:  # noqa: BLE001, S110
            pass
        compatible_types = set(
            os.environ.get(
                "OPENVINO_COMPATIBLE_TYPES",
                "qwen3,qwen2,qwen,llama,mistral,gemma,phi,gpt_neox,chatglm",
            ).split(",")
        )

        # Parse max_params (e.g. 8B → 8e9) — flexible, dynamic
        max_size: float | None = None
        if max_params:
            try:
                s = max_params.strip().upper()
                mult = 1.0
                if s.endswith("B"):
                    mult = 1e9
                    s = s[:-1]
                elif s.endswith("M"):
                    mult = 1e6
                    s = s[:-1]
                elif s.endswith("K"):
                    mult = 1e3
                    s = s[:-1]
                max_size = float(s) * mult
            except Exception:  # noqa: BLE001
                console.print(
                    f"[red]Invalid --max-params '{max_params}' (use e.g. 1.7B, 8B, 5B)[/red]"
                )
                sys.exit(1)
        # Fetch a bit more than limit to account for filtering (strict mode may skip many)
        fetch_limit = max(limit * 2, 30) if not all else limit * 2
        raw_models = api.list_models(search=query, limit=fetch_limit, sort="downloads")
        table = Table(
            title=f"Hugging Face search: {query} ({'all' if all else 'compatible only'}{f', max {max_params}' if max_params else ''})"
        )
        table.add_column("ID", style="cyan")
        table.add_column("Size", justify="right")
        table.add_column("Type", style="dim")
        table.add_column("Downloads", justify="right")
        table.add_column("Likes", justify="right")
        if all:
            table.add_column("Status", style="yellow")
        shown = 0
        skipped = 0
        for m in raw_models:
            # Try to get size and type — dynamic, not hardcoded for one machine
            m_id = m.id
            m_type = None
            m_size = None
            # From list result, try to get config
            if hasattr(m, "config") and isinstance(m.config, dict):
                m_type = m.config.get("model_type")
            # Always refetch for accurate size via real sibling file sizes —
            # `safetensors.total` (list or model_info) is unreliable, see _hf_accurate_size.
            m_size = _hf_accurate_size(api, m_id) or m_size
            try:
                info2 = api.model_info(m_id)
                if info2.config and isinstance(info2.config, dict):
                    m_type = info2.config.get("model_type") or m_type
            except Exception:  # noqa: BLE001, S110
                pass
            # Fallback: try model_info for more accurate size/type (but may be slow, so only if needed for strict check)
            # For strict mode, we need to know if it's compatible; if we can't get info, assume unknown and show as compatible
            reason = None
            if m_type and m_type not in compatible_types:
                reason = f"type {m_type} not in Arc whitelist"
            elif m_size:
                # Dynamic, not hardcoded for one machine — same env as pull
                _disk_factor = float(os.environ.get("OPENVINO_DISK_FACTOR", "2.0"))
                _ram_factor = float(os.environ.get("OPENVINO_RAM_FACTOR", "1.5"))
                _disk_margin = (
                    int(os.environ.get("OPENVINO_DISK_MARGIN_GB", "2")) * 1024**3
                )
                _ram_margin = (
                    int(os.environ.get("OPENVINO_RAM_MARGIN_GB", "2")) * 1024**3
                )
                need_disk = int(m_size * _disk_factor) + _disk_margin
                need_ram = int(m_size * _ram_factor) + _ram_margin
                if disk_avail < need_disk:
                    reason = f"disk {m_size / 1024**3:.1f}G → need {need_disk / 1024**3:.1f}G, avail {disk_avail / 1024**3:.1f}G"
                elif ram_avail and ram_avail < need_ram:
                    reason = f"RAM {m_size / 1024**3:.1f}G → need {need_ram / 1024**3:.1f}G, avail {ram_avail / 1024**3:.1f}G"
                elif max_size and m_size > max_size:
                    reason = f"size {m_size / 1024**3:.1f}G > max {max_params}"
            # Strict: skip incompatible unless --all
            if reason and not all:
                skipped += 1
                continue
            size_str = f"{m_size / 1024**3:.1f}G" if m_size else "?"
            type_str = m_type or "?"
            if reason and all:
                # Yellow with reason
                table.add_row(
                    f"[yellow]{m_id}[/yellow]",
                    f"[yellow]{size_str}[/yellow]",
                    f"[yellow]{type_str}[/yellow]",
                    f"[yellow]{m.downloads}[/yellow]",
                    f"[yellow]{m.likes}[/yellow]",
                    f"[yellow]{reason}[/yellow]",
                )
            else:
                table.add_row(m_id, size_str, type_str, str(m.downloads), str(m.likes))
            shown += 1
            if shown >= limit:
                break
        console.print(table)
        if skipped and not all:
            console.print(
                f"[dim]{skipped} incompatible hidden — use --all to see all (yellow with reason)[/dim]"
            )
        console.print(
            "[dim]Pull with: openvino pull <model-id>  (e.g. Qwen/Qwen3-1.7B)[/dim]"
        )
        if not all:
            console.print(
                "[dim]Strict by default: only Arc-compatible + fits disk/RAM shown. Use --all for all.[/dim]"
            )
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]search failed (offline?): {e}[/red]")
        console.print("[dim]Try: openvino pull <HF_ID>  (e.g. Qwen/Qwen3-1.7B)[/dim]")


@app.command("show")
def show(name: str = typer.Argument(..., help="Model name")) -> None:
    """Show model files and info."""
    p = _models_dir() / name
    if not p.exists():
        p = _models_dir() / name.replace(":", "-")
    if not p.exists():
        p = _models_dir() / _local_name(name)
    if not p.exists():
        console.print(f"[red]Model not found: {name}[/red]")
        sys.exit(1)
    console.print(f"[bold]{p.name}[/bold] @ {p}")
    for f in sorted(p.iterdir()):
        size = f.stat().st_size
        console.print(f"  {f.name:30} {size / 1e6:.1f} MB")
    cfg = p / "config.json"
    if cfg.exists():
        j = json.loads(cfg.read_text())
        console.print(
            f"[dim]model_type={j.get('model_type')} hidden_size={j.get('hidden_size')}[/dim]"
        )


@app.command("run")
def run(
    name: str = typer.Argument(..., help="Model name"),
    prompt: str = typer.Argument(..., help="Prompt"),
) -> None:
    """Quick chat test via the server (like ollama run)."""
    url = _url()
    try:
        r = httpx.post(
            f"{url}/v1/chat/completions",
            json={
                "model": name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 256,
            },
            timeout=120.0,
        )
        r.raise_for_status()
        j = r.json()
        text = j["choices"][0]["message"]["content"]
        console.print(f"[bold cyan]{name}:[/bold cyan] {text}")
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]run failed: {e}[/red]")
        sys.exit(1)


@app.command("stop")
def stop(model: str = typer.Argument(..., help="Model name to stop/unload")) -> None:
    """Stop/unload a model (like `ollama stop <model>`) — calls POST /v1/unload."""
    url = _url()
    try:
        console.print(f"[dim]Stopping model [cyan]{model}[/cyan]...[/dim]")
        r = httpx.post(f"{url}/v1/unload", json={"model": model}, timeout=30.0)
        if r.status_code == 200:
            j = r.json()
            unloaded = j.get("unloaded")
            if unloaded:
                console.print(
                    f"[green]✓ unloaded {unloaded} (VRAM freed, IR remains)[/green]"
                )
            else:
                console.print("[dim]No model was loaded[/dim]")
                console.print(
                    f"[green]✓ stop {model} (IR remains, use `rm` to delete)[/green]"
                )
        else:
            console.print(f"[yellow]Server returned {r.status_code}: {r.text}[/yellow]")
            console.print(
                f"[green]✓ stop {model} (IR remains, use `rm` to delete)[/green]"
            )
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Could not reach server: {e}[/red]")
        console.print("[dim]Local stop done — restart server to clear memory.[/dim]")


@server_app.command("install")
def server_install() -> None:
    """Install the systemd service."""
    service_src = Path(__file__).parent.parent.parent / "openvino.service"
    if not service_src.exists():
        service_src = Path.home() / "KpihX-Labs/AI/openvino/openvino.service"
    user_service = Path.home() / ".config/systemd/user/openvino.service"
    if not service_src.exists():
        console.print(f"[red]Service file not found: {service_src}[/red]")
        sys.exit(1)
    user_service.parent.mkdir(parents=True, exist_ok=True)
    try:
        if user_service.exists() and user_service.resolve() == service_src.resolve():
            console.print(f"[dim]Service already installed @ {user_service}[/dim]")
        else:
            shutil.copy(service_src, user_service)
    except shutil.SameFileError:
        console.print(f"[dim]Service already installed @ {user_service}[/dim]")
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    subprocess.run(["systemctl", "--user", "enable", "openvino.service"], check=False)
    console.print(f"[green]✓ Service installed @ {user_service}[/green]")


@server_app.command("uninstall")
def server_uninstall() -> None:
    """Uninstall the systemd service."""
    user_service = Path.home() / ".config/systemd/user/openvino.service"
    wants = Path.home() / ".config/systemd/user/default.target.wants/openvino.service"
    try:
        subprocess.run(
            ["systemctl", "--user", "disable", "--now", "openvino.service"],
            check=False,
        )
    except Exception:  # noqa: BLE001, S110
        pass
    for p in (user_service, wants):
        if p.exists() or p.is_symlink():
            p.unlink(missing_ok=True)
            console.print(f"[dim]Removed {p}[/dim]")
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    console.print("[green]✓ Service uninstalled[/green]")


@server_app.command("start")
def server_start(
    foreground: bool = typer.Option(
        False, "--foreground", "-f", help="Run in foreground"
    ),
) -> None:
    """Start the server (non-blocking, like `ollama serve`)."""
    if foreground:
        from k_openvino.serve import main as serve_main

        console.print(f"[bold]Starting server in foreground @ {CONFIG.url} ...[/bold]")
        serve_main()
        return
    # Ensure installed
    user_service = Path.home() / ".config/systemd/user/openvino.service"
    if not user_service.exists():
        server_install()
    try:
        subprocess.run(["systemctl", "--user", "start", "openvino.service"], check=True)
        console.print(
            f"[green]✓ Server started @ {_url()}[/green] (systemd, non-blocking)"
        )
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Failed to start: {e}[/red]")
        console.print("[dim]Try: openvino server start --foreground[/dim]")
        sys.exit(1)


@server_app.command("stop")
def server_stop() -> None:
    """Stop the server."""
    try:
        subprocess.run(["systemctl", "--user", "stop", "openvino.service"], check=True)
        console.print("[green]✓ Server stopped[/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Failed to stop: {e}[/red]")
        sys.exit(1)


@server_app.command("restart")
def server_restart() -> None:
    """Restart the server."""
    try:
        subprocess.run(
            ["systemctl", "--user", "restart", "openvino.service"], check=True
        )
        console.print("[green]✓ Server restarted[/green]")
        subprocess.run(
            ["systemctl", "--user", "status", "openvino.service", "--no-pager", "-l"],
            check=False,
        )
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Failed to restart: {e}[/red]")
        sys.exit(1)


@server_app.command("status")
def server_status() -> None:
    """Show server status (systemd)."""
    subprocess.run(
        ["systemctl", "--user", "status", "openvino.service", "--no-pager", "-l"],
        check=False,
    )


@server_app.command("logs")
def server_logs(
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow logs"),
    lines: int = typer.Option(100, "--lines", "-n", help="Number of lines"),
    since: str = typer.Option(None, "--since", "-s", help="Since (e.g. 5m, 1h, today)"),
    no_pager: bool = typer.Option(False, "--no-pager", "-P", help="No pager"),
) -> None:
    """Show server logs (journalctl)."""
    cmd = ["journalctl", "--user", "-u", "openvino.service", "-n", str(lines)]
    if follow:
        cmd.append("-f")
    if since:
        cmd.extend(["--since", since])
    if no_pager:
        cmd.append("--no-pager")
    # Flexibility: also accept single-dash forms via Typer's help, but journalctl needs long form
    subprocess.run(cmd, check=False)


app.add_typer(server_app, name="server")


@app.command("help")
def help_cmd() -> None:
    """Show help."""
    console.print("[bold]openvino — OpenVINO GenAI CLI[/bold]")
    console.print("Commands: ls, ps, pull, rm, search, show, run, stop, server, help")
    console.print("Server subcommands: server install/start/restart/status/stop/logs")
    console.print("Use: openvino <command> --help  or  openvino server <sub> --help")
    console.print(f"Server: {_url()}  Home: {_models_dir()}")


if __name__ == "__main__":
    app()
