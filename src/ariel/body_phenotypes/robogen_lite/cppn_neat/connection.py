"""Connection gene of a CPPN genome."""

from __future__ import annotations


class Connection:
    """A connection gene in a NEAT genome."""

    def __init__(
        self,
        in_id: int,
        out_id: int,
        weight: float,
        enabled: bool,  # ruff: ignore[boolean-type-hint-positional-argument] (positional in Genome and copy())
        innov_id: int,
    ) -> None:
        """Create a connection from `in_id` to `out_id` with a weight."""
        self.in_id = in_id
        self.out_id = out_id
        self.weight = weight
        self.enabled = enabled
        self.innov_id = innov_id

    def copy(self) -> Connection:
        """Return a new Connection object with identical values."""
        return Connection(
            self.in_id, self.out_id, self.weight, self.enabled, self.innov_id
        )
