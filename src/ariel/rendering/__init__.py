# ruff: file-ignore[non-empty-init-module, module-import-not-at-top-of-file]  # package entry point: the OpenGL choice must run before mujoco is imported
"""Render MuJoCo scenes to frames and videos, and where the files go.

Importing this package first runs ``ensure_gl_backend()``: MuJoCo reads
``MUJOCO_GL`` when it is imported, and the renderer modules import it.
"""

from ariel.opengl import ensure_gl_backend

ensure_gl_backend()

# isort: split

# Local libraries
from ariel.rendering.file_ops import generate_save_path
from ariel.rendering.renderers import (
    single_frame_renderer,
    tracking_video_renderer,
    video_renderer,
)
from ariel.rendering.video_recorder import VideoRecorder

__all__ = [
    "VideoRecorder",
    "generate_save_path",
    "single_frame_renderer",
    "tracking_video_renderer",
    "video_renderer",
]
