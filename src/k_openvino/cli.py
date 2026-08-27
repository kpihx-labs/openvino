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
    hf: str = typer.Option(None, "--hf", help="Override HF repo id"),
) -> None:
    """Export a Hugging Face model to OpenVINO IR (like ollama pull)."""
    hf_id = _hf_id(name, hf)
    local = _local_name(name if "/" not in name else hf_id)
    dest = _models_dir() / local
    if dest.exists() and (dest / "openvino_model.xml").exists():
        console.print(
            f"[yellow]{local} already exists at {dest} — skip (rm first to re-pull)[/yellow]"
        )
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Use the venv's optimum-cli (installed via project deps)
    venv_optimum = Path.home() / ".local/share/openvino/venv/bin/optimum-cli"
    cmd = [
        str(venv_optimum),
        "export",
        "openvino",
        "--model",
        hf_id,
        "--task",
        "text-generation-with-past",
        str(dest),
    ]
    if not venv_optimum.exists():
        cmd = [
            "optimum-cli",
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
) -> None:
    """Search Hugging Face Hub for models (online)."""
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        models = api.list_models(search=query, limit=20, sort="downloads")
        table = Table(title=f"Hugging Face search: {query}")
        table.add_column("ID", style="cyan")
        table.add_column("Downloads", justify="right")
        table.add_column("Likes", justify="right")
        for m in models:
            table.add_row(m.id, str(m.downloads), str(m.likes))
        console.print(table)
        console.print(
            "[dim]Pull with: openvino pull <model-id>  (e.g. Qwen/Qwen3-1.7B)[/dim]"
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
    since: str = typer.Option(None, "--since", help="Since (e.g. 5m, 1h, today)"),
    no_pager: bool = typer.Option(False, "--no-pager", help="No pager"),
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
