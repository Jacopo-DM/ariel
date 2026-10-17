"""Names and paths shared by the OpenGL backend probe and its readers.

This module imports neither ``mujoco`` nor any other ARIEL package: MuJoCo
reads ``MUJOCO_GL`` when it is imported, so code that decides the backend
must run before that import.
"""

from pathlib import Path

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
