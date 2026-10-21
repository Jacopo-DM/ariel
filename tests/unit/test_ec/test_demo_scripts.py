"""Test: the demo entry point of RevDE."""

from __future__ import annotations

import importlib
import inspect
import runpy

from ariel.ec.optimizers import revde

_gl_probe_worker = importlib.import_module("ariel.opengl._gl_probe_worker")


# --- demo entry points ------------------------------------------------------ #


def test_revde_demo_logs_parents_and_children() -> None:
    """The RevDE demo runs without error."""
    revde.main()


def test_revde_script() -> None:
    """Running revde.py as a script calls main()."""
    runpy.run_path(inspect.getfile(revde), run_name="__main__")
