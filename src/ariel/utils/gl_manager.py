"""
Find out which MuJoCo OpenGL backends work and save the answer.

Run ``python -m ariel.utils.gl_manager`` to (re)create
``mujoco_user_config.json`` in the current folder. Each backend is probed in
its own subprocess, so a backend that crashes is recorded as failed and the
manager carries on.
"""

import contextlib
import json
import os
import platform as platform_info
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from ariel.utils.gl_backend import (
    BACKEND_ORDER,
    CONFIG_NAME,
    CONFIG_VERSION,
    config_path,
    mujoco_version,
    platform_backends,
)

DEFAULT_TIMEOUT = 30.0

CommandFactory = Callable[[str], list[str]]


@dataclass(frozen=True)
class BackendResult:
    """Outcome of probing one backend."""

    ok: bool
    render_ms: float | None = None
    error: str | None = None


def default_command(backend: str) -> list[str]:
    """
    Return the command that probes one backend.

    Parameters
    ----------
    backend
        Value for ``MUJOCO_GL``.

    Returns
    -------
    list[str]
        The worker command line.
    """
    return [sys.executable, "-m", "ariel.utils._gl_probe_worker", backend]


def _parse_worker_output(stdout: str) -> BackendResult:
    lines = [line for line in stdout.splitlines() if line.strip()]
    try:
        payload = json.loads(lines[-1])
        ok = bool(payload["ok"])
    except (IndexError, ValueError, KeyError, TypeError):
        return BackendResult(ok=False, error="worker output unreadable")
    if not ok:
        return BackendResult(ok=False, error=str(payload.get("error")))
    return BackendResult(ok=True, render_ms=float(payload["render_ms"]))


def probe_backend(
    backend: str,
    command: CommandFactory = default_command,
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> BackendResult:
    """
    Probe one backend in a subprocess; never raises.

    Parameters
    ----------
    backend
        Value for ``MUJOCO_GL``.
    command
        Builds the worker command line for a backend.
    timeout
        Seconds before the worker is killed and the backend counts as failed.

    Returns
    -------
    BackendResult
        Success with the render time, or failure with the reason (crash,
        timeout, error message).
    """
    try:
        done = subprocess.run(  # noqa: S603 (command comes from our code)
            command(backend),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return BackendResult(ok=False, error=f"timed out after {timeout:g} s")
    except OSError as exc:
        return BackendResult(ok=False, error=f"could not start worker: {exc}")
    if done.returncode != 0:
        tail = done.stderr.strip().splitlines()[-1:] or [""]
        return BackendResult(
            ok=False,
            error=f"worker failed (exit code {done.returncode}) {tail[0]}"[
                :200
            ].rstrip(),
        )
    return _parse_worker_output(done.stdout)


def select_backend(results: Mapping[str, BackendResult]) -> str | None:
    """
    Pick the working backend that renders fastest.

    Parameters
    ----------
    results
        Probe results by backend name.

    Returns
    -------
    str | None
        The backend name, or None if none works. Ties follow the order
        glfw, egl, osmesa.
    """
    working = [
        (r.render_ms if r.render_ms is not None else float("inf"), i, name)
        for i, name in enumerate(BACKEND_ORDER)
        if (r := results.get(name)) is not None and r.ok
    ]
    return min(working)[2] if working else None


def _write_atomically(path: Path, config: dict[str, Any]) -> None:
    handle, name = tempfile.mkstemp(
        dir=path.parent, prefix=".gl-config-", suffix=".tmp"
    )
    tmp = Path(name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2)
            stream.write("\n")
        tmp.replace(path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def run(
    path: Path,
    *,
    command: CommandFactory = default_command,
    timeout: float = DEFAULT_TIMEOUT,
    console: Console | None = None,
    platform: str | None = None,
) -> dict[str, Any]:
    """
    Probe every backend of the platform and write the config file.

    Parameters
    ----------
    path
        Where to write ``mujoco_user_config.json``.
    command
        Builds the worker command line for a backend.
    timeout
        Seconds allowed per backend.
    console
        Where to show progress; a new console by default.
    platform
        A ``sys.platform`` value; the running platform by default.

    Returns
    -------
    dict[str, Any]
        The configuration that was written.
    """
    console = console or Console()
    platform = platform or sys.platform
    backends = platform_backends(platform)
    console.print(
        "[bold]One-off setup:[/bold] testing which OpenGL backends work on "
        f"this machine. The result is saved to [cyan]{path.name}[/cyan]; "
        "this will not be repeated."
    )
    results: dict[str, BackendResult] = {}
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("OpenGL backends", total=len(backends))
        for name in backends:
            progress.update(task, description=f"Testing {name}")
            results[name] = probe_backend(name, command, timeout=timeout)
            progress.advance(task)
        progress.update(task, description="Done")
    glfw = results.get("glfw")
    config: dict[str, Any] = {
        "version": CONFIG_VERSION,
        "platform": platform,
        "python": platform_info.python_version(),
        "mujoco": mujoco_version(),
        "backends": {name: asdict(r) for name, r in results.items()},
        "selected": select_backend(results),
        "viewer_available": bool(glfw and glfw.ok),
    }
    _write_atomically(path, config)
    return config


def main() -> int:
    """
    Probe the backends and write the config into the current folder.

    Returns
    -------
    int
        0 whatever the probe found: a machine with no working backend is a
        valid result, not an error.
    """
    console = Console()
    config = run(config_path(), console=console)
    selected = config["selected"]
    console.print(
        f"Selected backend: [bold]{selected or 'none (using MuJoCo default)'}"
        f"[/bold]; viewer available: {config['viewer_available']}. "
        f"Saved to {CONFIG_NAME}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
