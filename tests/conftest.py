"""Test configuration shared by every test, and the Hypothesis profile."""

import os

from hypothesis import settings

# Importing ariel.rendering probes the OpenGL backends when
# MUJOCO_GL is unset and no mujoco_user_config.json exists. A preset variable
# means "nothing to decide", so the tests never probe or write that file in
# the repository. "glfw" is also MuJoCo's own default.
os.environ.setdefault("MUJOCO_GL", "glfw")

settings.register_profile("default", deadline=None)
settings.register_profile("ci", deadline=None, derandomize=True)
settings.load_profile("ci" if os.environ.get("CI") else "default")
