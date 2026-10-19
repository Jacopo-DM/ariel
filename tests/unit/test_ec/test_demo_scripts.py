"""Test: the demo entry points of RevDE and NDE."""

from __future__ import annotations

import importlib.metadata
import inspect
import runpy

from ariel.ec.genotypes.nde import nde
from ariel.ec.optimizers import revde

_gl_probe_worker = importlib.import_module("ariel.opengl._gl_probe_worker")


# --- demo entry points ------------------------------------------------------ #


def test_revde_demo_logs_parents_and_children() -> None:
    """The RevDE demo runs without error."""
    revde.main()


def test_revde_script() -> None:
    """Running revde.py as a script calls main()."""
    runpy.run_path(inspect.getfile(revde), run_name="__main__")


def test_nde_script() -> None:
    """Running nde.py as a script prints the output shapes."""
    runpy.run_path(inspect.getfile(nde), run_name="__main__")
