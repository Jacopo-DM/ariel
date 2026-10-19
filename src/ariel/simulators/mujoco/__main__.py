"""Print the derivation of the MuJoCo settings and what differs from MuJoCo.

Run ``python -m ariel.simulators.mujoco``; add ``--out FILE`` to write the
MJCF file.
"""

# Standard library
import argparse
from pathlib import Path

# Local libraries
from ariel import console
from ariel.simulators.mujoco.bands import evaluate_bands, warn_outside
from ariel.simulators.mujoco.bands import summary as band_summary
from ariel.simulators.mujoco.mjcf import differences, generate_xml
from ariel.simulators.mujoco.mjcf import summary as difference_summary
from ariel.simulators.mujoco.seeds import Seeds
from ariel.simulators.mujoco.settings import Value, engine_defaults
from ariel.simulators.mujoco.table import derivation_table


def _say(line: str) -> None:
    """Print one line without wrapping or markup.

    Parameters
    ----------
    line : str
        The text.
    """
    console.print(line, soft_wrap=True, markup=False, highlight=False)


def _show(value: Value) -> str:
    """Format a setting value for one line.

    Parameters
    ----------
    value : Value
        The value.

    Returns
    -------
    str
        Numbers in general format, tuples separated by spaces.
    """
    if isinstance(value, tuple):
        return " ".join(f"{item:g}" for item in value)
    if isinstance(value, bool):
        return str(value)
    return f"{value:g}" if isinstance(value, int | float) else str(value)


def main(argv: list[str] | None = None) -> int:
    """Print the report, and write the MJCF file when asked.

    Parameters
    ----------
    argv : list[str] | None
        The arguments; ``sys.argv`` when None.

    Returns
    -------
    int
        The exit status.
    """
    parser = argparse.ArgumentParser(prog="python -m ariel.simulators.mujoco")
    parser.add_argument("--out", type=Path, help="write the MJCF file here")
    arguments = parser.parse_args(argv)
    seeds = Seeds()
    _say("Derivation (seeds and formulas)")
    for row in derivation_table(seeds):
        unit = f" {row.unit}" if row.unit else ""
        _say(f"  {row.name} = {row.value:g}{unit}  [{row.kind}] {row.source}")
    defaults = engine_defaults()
    _say("Differences from MuJoCo's defaults")
    for setting in differences(seeds):
        _say(
            f"  {setting.name}: {_show(defaults[setting.name])} -> "
            f"{_show(setting.value)}  [{setting.kind}] {setting.reason}"
        )
    _say(difference_summary(seeds))
    results = evaluate_bands(seeds)
    warn_outside(results)
    _say(band_summary(results))
    if arguments.out is not None:
        arguments.out.write_text(generate_xml(seeds), encoding="utf-8")
        _say(f"Wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
