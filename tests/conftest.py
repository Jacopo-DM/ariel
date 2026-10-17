"""Test configuration shared by every test."""

import os

# Importing ariel.utils.renderers probes the OpenGL backends when
# MUJOCO_GL is unset and no mujoco_user_config.json exists. A preset variable
# means "nothing to decide", so the tests never probe or write that file in
# the repository. "glfw" is also MuJoCo's own default.
os.environ.setdefault("MUJOCO_GL", "glfw")
