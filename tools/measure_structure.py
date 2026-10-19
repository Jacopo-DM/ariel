"""Print structure numbers for ``src/ariel`` and ``tests``.

Run from the repo root: ``uv run --no-sync python tools/measure_structure.py``.
Read-only. It reports file sizes, modules that import many others and
imports of another package's module file (``ariel.ec.ea``) instead of a
package ``__init__`` (``ariel.ec``, or a subpackage such as
``ariel.simulation.environments``). Thresholds are 400 lines and 5 imports.
Duplication is measured apart, with ``npx jscpd@5.4.1 --min-lines 8 src tests``.

Notes
-----
    * Only absolute ``from ariel... import`` and ``import ariel...`` count.
"""

import ast
import sys
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "ariel"
MAX_LINES = 400
MAX_IMPORTS = 5


class Structure(BaseModel):
    """The numbers measured for one package folder."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    files: int
    lines: int
    long_files: tuple[tuple[int, str], ...]
    wide_modules: tuple[tuple[int, str], ...]
    deep_imports: tuple[tuple[int, str], ...]


def module_name(path: Path, src: Path) -> str:
    """Return the dotted module name of a file under ``src``.

    Parameters
    ----------
    path : Path
        A Python file under ``src``.
    src : Path
        The folder that holds the top-level package.

    Returns
    -------
    str
        For example ``ariel.ec.ea``; a package ``__init__`` gives the package.
    """
    parts = list(path.relative_to(src).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def imports_of(path: Path) -> set[str]:
    """Return the ``ariel`` modules a file imports, as dotted names.

    Parameters
    ----------
    path : Path
        A Python file.

    Returns
    -------
    set[str]
        Dotted module names that start with ``ariel``.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.level == 0
            and node.module.split(".")[0] == PACKAGE
        ):
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(
                alias.name
                for alias in node.names
                if alias.name.split(".")[0] == PACKAGE
            )
    return found


def subpackage(module: str) -> str:
    """Return the subpackage of ``ariel`` a module lives in.

    Parameters
    ----------
    module : str
        A dotted module name.

    Returns
    -------
    str
        The second part of the name, or an empty string for ``ariel`` itself.
    """
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else ""


def measure(folder: Path, src: Path, root: Path) -> Structure:
    """Measure every Python file under a folder.

    Parameters
    ----------
    folder : Path
        The folder to walk.
    src : Path
        The folder that holds the top-level package, used for module names.
    root : Path
        The folder that file paths in the result are relative to.

    Returns
    -------
    Structure
        Sizes, wide modules and imports that skip a package entry point.
    """
    files = sorted(folder.rglob("*.py"))
    lines = {f: len(f.read_text(encoding="utf-8").splitlines()) for f in files}
    imports = {f: imports_of(f) for f in files}
    long_files = tuple(
        sorted(
            (
                (lines[f], str(f.relative_to(root)))
                for f in files
                if lines[f] > MAX_LINES
            ),
            reverse=True,
        ),
    )
    wide = tuple(
        sorted(
            (
                (len(imports[f]), str(f.relative_to(root)))
                for f in files
                if len(imports[f]) > MAX_IMPORTS
            ),
            reverse=True,
        ),
    )
    deep: Counter[str] = Counter()
    for f in files:
        if not f.is_relative_to(src):
            continue  # tests may import any module
        own = subpackage(module_name(f, src))
        for target in imports[f]:
            parts = target.split(".")
            is_module = src.joinpath(*parts).with_suffix(".py").is_file()
            if is_module and parts[1] != own:
                deep[f"{own or PACKAGE} -> {parts[1]}"] += 1
    return Structure(
        files=len(files),
        lines=sum(lines.values()),
        long_files=long_files,
        wide_modules=wide,
        deep_imports=tuple(
            sorted(((n, edge) for edge, n in deep.items()), reverse=True),
        ),
    )


def render(name: str, result: Structure) -> str:
    """Format a measurement as text.

    Parameters
    ----------
    name : str
        The label of the folder.
    result : Structure
        The measurement.

    Returns
    -------
    str
        The report, one fact per line.
    """
    wide = len(result.wide_modules)
    deep = sum(n for n, _ in result.deep_imports)
    out = [
        f"{name}: {result.files} files, {result.lines} lines",
        f"files over {MAX_LINES} lines: {len(result.long_files)}",
        *(f"  {n:5d}  {path}" for n, path in result.long_files),
        f"modules importing more than {MAX_IMPORTS} others: {wide}",
        *(f"  {n:3d}  {path}" for n, path in result.wide_modules),
        f"imports that skip a package entry point: {deep}",
        *(f"  {n:3d}  {edge}" for n, edge in result.deep_imports),
    ]
    return "\n".join(out) + "\n"


def main() -> int:
    """Print the numbers for ``src/ariel`` and ``tests``.

    Returns
    -------
    int
        Always 0; the script only reports.
    """
    src = ROOT / "src"
    sys.stdout.write(render("src/ariel", measure(src / PACKAGE, src, ROOT)))
    sys.stdout.write("\n")
    sys.stdout.write(render("tests", measure(ROOT / "tests", src, ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
