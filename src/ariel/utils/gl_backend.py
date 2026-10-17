"""Names and paths shared by the OpenGL backend probe and its readers.

This module imports neither ``mujoco`` nor any other ARIEL package: MuJoCo
reads ``MUJOCO_GL`` when it is imported, so code that decides the backend
must run before that import.
"""

import json
import os
import sys
import warnings
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

CONFIG_NAME = "mujoco_user_config.json"
CONFIG_VERSION = 1

# Order of preference when two backends render equally fast.
BACKEND_ORDER = ("glfw", "egl", "osmesa")


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
