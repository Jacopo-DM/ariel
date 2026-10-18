"""
Probe one MuJoCo OpenGL backend in a throw-away process.

Run as ``python -m ariel.utils._gl_probe_worker <backend>``. A broken backend
can crash the whole process (for example a failing ``glfwCreateWindow``), so
the manager starts one worker per backend. The worker imports ``mujoco`` only
and prints a single JSON line: ``{"ok": true, "render_ms": 3.2}`` or
``{"ok": false, "error": "..."}``.
"""

import json
import os
import sys
import time

EXPECTED_ARGV_LENGTH = 2

_XML = "<mujoco><worldbody><geom size='.1'/></worldbody></mujoco>"
_SIZE = 64
_FRAMES = 5


def probe(backend: str) -> dict[str, object]:
    """
    Render a few small frames with one backend.

    Parameters
    ----------
    backend
        Value for ``MUJOCO_GL``.

    Returns
    -------
    dict[str, object]
        ``ok`` and the mean ``render_ms``, or ``ok`` false with an ``error``.
    """
    os.environ["MUJOCO_GL"] = backend
    try:  # ruff: ignore[too-many-statements-in-try-clause]  # public signature or class shape kept as is; changing it changes the API
        import mujoco  # ruff: ignore[import-outside-top-level] (must follow setting MUJOCO_GL)

        model = mujoco.MjModel.from_xml_string(_XML)
        data = mujoco.MjData(model)
        with mujoco.Renderer(model, _SIZE, _SIZE) as renderer:
            mujoco.mj_forward(model, data)
            renderer.update_scene(data)
            renderer.render()  # warm-up, not timed
            start = time.perf_counter()
            for _ in range(_FRAMES):
                renderer.update_scene(data)
                renderer.render()
            elapsed = time.perf_counter() - start
    except Exception as exc:  # ruff: ignore[blind-except] (any failure means "unusable")
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "render_ms": 1000.0 * elapsed / _FRAMES}


def main(argv: list[str]) -> int:
    """
    Probe the backend named in ``argv[1]`` and print the JSON result.

    Parameters
    ----------
    argv
        Command-line arguments, program name first.

    Returns
    -------
    int
        Exit code: 0 when a result was printed, 2 on a usage error.
    """
    if len(argv) != EXPECTED_ARGV_LENGTH:
        sys.stderr.write("usage: _gl_probe_worker <backend>\n")
        return 2
    sys.stdout.write(json.dumps(probe(argv[1])) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
