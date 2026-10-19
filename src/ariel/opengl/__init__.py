"""Choice of the OpenGL backend that MuJoCo uses.

``ariel/__init__`` imports this package first, so nothing here may import
MuJoCo: MuJoCo reads ``MUJOCO_GL`` when it loads, after the choice is made.
"""

# Local libraries
from ariel.opengl.gl_backend import configure_from_file, ensure_gl_backend

__all__ = ["configure_from_file", "ensure_gl_backend"]
