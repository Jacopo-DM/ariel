"""A world with nothing added, for tests of BaseWorld itself."""

from ariel.simulation.environments import BaseWorld


class PlainWorld(BaseWorld):
    """BaseWorld with an empty ``_expand_spec``; BaseWorld never calls it."""

    def _expand_spec(self) -> None:  # pragma: no cover  # ruff: ignore[no-self-use]  # an override; never called
        return
