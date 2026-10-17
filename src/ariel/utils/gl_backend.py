"""Names and paths shared by the OpenGL backend probe and its readers.

This module imports neither ``mujoco`` nor any other ARIEL package: MuJoCo
reads ``MUJOCO_GL`` when it is imported, so code that decides the backend
must run before that import.
"""

import contextlib
import importlib.metadata
import json
import os
import sys
import time
import warnings
from collections.abc import Callable, Mapping, MutableMapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rich.console import Console

CONFIG_NAME = "mujoco_user_config.json"
CONFIG_VERSION = 1

# Order of preference when two backends render equally fast.
BACKEND_ORDER = ("glfw", "egl", "osmesa")

LOCK_SUFFIX = ".lock"
# A probe takes at most three backends x 30 s; a lock older than this is stale.
LOCK_TIMEOUT = 120.0
_POLL_SECONDS = 0.1


def platform_backends(platform: str) -> tuple[str, ...]:
    """
    Return the MuJoCo OpenGL backends worth probing on a platform.

    Parameters
    ----------
    platform
        A ``sys.platform`` value.

    Returns
    -------
    tuple[str, ...]
        ``glfw``, ``egl`` and ``osmesa`` on Linux, only ``glfw`` elsewhere.
    """
    if platform.startswith("linux"):
        return BACKEND_ORDER
    return ("glfw",)


def config_path() -> Path:
    """
    Return where the probe result lives: the current working directory.

    Returns
    -------
    Path
        ``<cwd>/mujoco_user_config.json``.
    """
    return Path.cwd() / CONFIG_NAME


def _warn(path: Path, problem: str) -> None:
    warnings.warn(
        f"Ignoring {CONFIG_NAME} ({path.parent}): {problem}. "
        "Run `python -m ariel.utils.gl_manager` to recreate it.",
        UserWarning,
        stacklevel=3,
    )


def read_config(path: Path) -> dict[str, Any] | None:
    """
    Read the probe result; never raises.

    Parameters
    ----------
    path
        The ``mujoco_user_config.json`` file.

    Returns
    -------
    dict[str, Any] | None
        The configuration, or None if the file is missing, unreadable or
        malformed (the last two also emit a ``UserWarning``).
    """
    if not path.is_file():
        return None
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _warn(path, f"cannot be read ({exc})")
        return None
    selected = config.get("selected") if isinstance(config, dict) else 0
    if not isinstance(config, dict) or not (
        selected is None or selected in BACKEND_ORDER
    ):
        _warn(path, "unexpected content")
        return None
    return config


def apply_config(
    config: Mapping[str, Any],
    *,
    environ: MutableMapping[str, str] | None = None,
    modules: Mapping[str, object] | None = None,
) -> str | None:
    """
    Set ``MUJOCO_GL`` from the probe result when that can still work.

    Parameters
    ----------
    config
        A configuration returned by `read_config`.
    environ
        The environment to change; ``os.environ`` by default.
    modules
        The imported modules; ``sys.modules`` by default.

    Returns
    -------
    str | None
        The backend that was set, or None if nothing was changed: a preset
        ``MUJOCO_GL`` always wins, nothing was selected, or ``mujoco`` is
        already imported (it reads the variable at import time).
    """
    environ = os.environ if environ is None else environ
    modules = sys.modules if modules is None else modules
    selected = config.get("selected")
    if selected is None or "MUJOCO_GL" in environ or "mujoco" in modules:
        return None
    environ["MUJOCO_GL"] = selected
    return str(selected)


def configure_from_file(
    path: Path | None = None,
    *,
    environ: MutableMapping[str, str] | None = None,
    modules: Mapping[str, object] | None = None,
) -> str | None:
    """
    Read the config file and apply it; never raises.

    Parameters
    ----------
    path
        The config file; ``<cwd>/mujoco_user_config.json`` by default.
    environ
        The environment to change; ``os.environ`` by default.
    modules
        The imported modules; ``sys.modules`` by default.

    Returns
    -------
    str | None
        The backend that was set, or None.
    """
    try:
        path = path or config_path()
        config = read_config(path)
    except OSError:  # for example the working folder no longer exists
        return None
    if config is None:
        return None
    return apply_config(config, environ=environ, modules=modules)


def mujoco_version() -> str:
    """
    Return the installed mujoco version without importing mujoco.

    Returns
    -------
    str
        The version, or ``"unknown"`` if the package is not installed.
    """
    try:
        return importlib.metadata.version("mujoco")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _current_config(path: Path) -> dict[str, Any] | None:
    """Read the config; None if missing, invalid or from another setup."""
    config = read_config(path)
    if config is None:
        return None
    recorded_version = config.get("mujoco")
    recorded_platform = config.get("platform")
    if recorded_version is not None and recorded_version != mujoco_version():
        return None
    if recorded_platform is not None and recorded_platform != sys.platform:
        return None
    return config


def _try_lock(lock: Path) -> bool:
    """Create the lock file; False if another process holds it."""
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(handle, str(os.getpid()).encode())
    os.close(handle)
    return True


def _wait_for_lock(lock: Path, timeout: float) -> None:
    """Wait until the lock is gone; remove it if it is stale."""
    deadline = time.monotonic() + timeout
    while lock.exists():
        try:
            age = time.time() - lock.stat().st_mtime
        except OSError:  # released between exists() and stat()
            return
        if age > timeout or time.monotonic() > deadline:
            with contextlib.suppress(OSError):
                lock.unlink()
            return
        time.sleep(_POLL_SECONDS)


def ensure_gl_backend(
    path: Path | None = None,
    *,
    run: Callable[[Path], dict[str, Any]] | None = None,
    environ: MutableMapping[str, str] | None = None,
    modules: Mapping[str, object] | None = None,
    console: "Console | None" = None,
    lock_timeout: float = LOCK_TIMEOUT,
) -> str | None:
    """
    Make sure the OpenGL probe result exists and apply it; never raises.

    If ``MUJOCO_GL`` is set, nothing is done: the variable wins, so a probe
    would be pointless. If the config file is missing (or corrupt), the
    backends are probed once; processes that start at the same time wait for
    that probe instead of repeating it. Call this before ``import mujoco``.

    Parameters
    ----------
    path
        The config file; ``<cwd>/mujoco_user_config.json`` by default.
    run
        Runs the probe and writes the file; the OpenGL manager by default.
    environ
        The environment to change; ``os.environ`` by default.
    modules
        The imported modules; ``sys.modules`` by default.
    console
        Where to tell the user about the probe; a new console by default.
    lock_timeout
        Seconds after which a lock file counts as stale.

    Returns
    -------
    str | None
        The backend that was set, or None.
    """
    environ = os.environ if environ is None else environ
    modules = sys.modules if modules is None else modules
    if "MUJOCO_GL" in environ:
        return None
    try:
        path = path or config_path()
        config = _current_config(path)
        if config is None:
            config = _probe_once(path, run, console, lock_timeout, modules)
    except OSError as exc:
        warnings.warn(
            f"cannot save {CONFIG_NAME}: {exc}. The default MuJoCo OpenGL "
            "backend is used.",
            UserWarning,
            stacklevel=2,
        )
        return None
    if config is None:
        return None
    return apply_config(config, environ=environ, modules=modules)


def _probe_once(
    path: Path,
    run: Callable[[Path], dict[str, Any]] | None,
    console: "Console | None",
    lock_timeout: float,
    modules: Mapping[str, object],
) -> dict[str, Any] | None:
    """Probe under a lock; return the config or None if it is unavailable."""
    lock = path.with_name(path.name + LOCK_SUFFIX)
    while not _try_lock(lock):
        _wait_for_lock(lock, lock_timeout)
        config = _current_config(path)
        if config is not None:
            return config
    try:
        config = _current_config(path)  # someone may have just finished
        if config is not None:
            return config
        if run is None:
            from ariel.utils.gl_manager import (  # noqa: PLC0415
                run as manager_run,
            )

            def run(target: Path) -> dict[str, Any]:
                return manager_run(target, console=console)

        config = run(path)
        if "mujoco" in modules:
            if console is None:
                from rich.console import Console  # noqa: PLC0415

                console = Console()
            console.print(
                "The OpenGL choice was saved but mujoco is already imported "
                "in this process, so it applies the next time you start "
                "Python."
            )
    finally:
        with contextlib.suppress(OSError):
            lock.unlink()
    return config
