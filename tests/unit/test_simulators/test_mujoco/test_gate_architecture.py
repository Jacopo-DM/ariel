"""Architecture test: every MuJoCo compile in the repo goes through the gate.

A spec is compiled with ``compile_model(spec)`` or a world's ``compile()``,
both of which run ARIEL's physical checks. Two files are exempt: the gate
itself, and the OpenGL probe, which renders one box to test the driver and
must not import other ARIEL packages (physical-checks D2).
"""

# Standard library
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
FOLDERS = ("src", "tests", "tools", "examples", "docs")
SKIPPED_PARTS = {"_build", "autoapi", "apidocs", ".ipynb_checkpoints"}
EXEMPT = {
    Path("src/ariel/simulators/mujoco/gate.py"),
    Path("src/ariel/opengl/_gl_probe_worker.py"),
    Path(__file__).resolve().relative_to(REPO_ROOT),
}

# A world's compile() runs the gate; these are the names worlds go by
WORLD_RECEIVERS = {"loaded", "full", "arena", "make()"}
NAMED_CALL = re.compile(r"([A-Za-z_]\w*(?:\(\))?)\.compile\(\)")
CALL_WITH_ARGUMENTS = re.compile(r"\([^()]+\)\.compile\(\)")
FROM_XML = re.compile(r"MjModel\.from_xml_\w+\(")

# The first four lines go around the gate; the rest do not
SAMPLE = (
    "model = spec.compile()",
    "model = world.spec.compile()",
    "model = mj.MjSpec.from_string(xml).compile()",
    "model = mujoco.MjModel.from_xml_string(xml)",
    "model = world.compile()",
    "model = world_tmp.compile()",
    "model = compile_model(spec)",
    "model = make().compile()",
    "pattern = re.compile('x')",
)


def _violations(text: str) -> list[int]:
    """Return the line numbers that compile a model outside the gate.

    Parameters
    ----------
    text : str
        The contents of a ``.py`` or ``.ipynb`` file.

    Returns
    -------
    list[int]
        One-based line numbers, in order.
    """
    found: list[int] = []
    for number, line in enumerate(text.splitlines(), start=1):
        receivers = [match.group(1) for match in NAMED_CALL.finditer(line)]
        outside = [
            receiver
            for receiver in receivers
            if not receiver.startswith("world")
            and receiver not in WORLD_RECEIVERS
        ]
        if outside or CALL_WITH_ARGUMENTS.search(line) or FROM_XML.search(line):
            found.append(number)
    return found


def _files() -> list[Path]:
    """List the Python files and notebooks the rule covers.

    Returns
    -------
    list[Path]
        Paths relative to the repository root, sorted.
    """
    found = [
        path.relative_to(REPO_ROOT)
        for folder in FOLDERS
        for pattern in ("*.py", "*.ipynb")
        for path in (REPO_ROOT / folder).rglob(pattern)
    ]
    return sorted(
        path
        for path in found
        if path not in EXEMPT and not SKIPPED_PARTS & set(path.parts)
    )


def test_the_rule_finds_each_way_around_the_gate() -> None:
    """Each pattern is caught; a world's compile and the gate are not."""
    assert _violations("\n".join(SAMPLE)) == [1, 2, 3, 4]


def test_every_compile_goes_through_the_gate() -> None:
    """No tracked source compiles a MuJoCo model past ARIEL's checks."""
    files = _files()
    assert files, "found no files: check REPO_ROOT"
    found = [
        f"{path}:{line}"
        for path in files
        for line in _violations((REPO_ROOT / path).read_text("utf-8"))
    ]
    assert found == []
