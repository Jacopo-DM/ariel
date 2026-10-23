"""Crossover and grafting of Sims genomes, after Sims (1994)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ariel.body_phenotypes import IDX_OF_CORE

from .genome import SimsConnection, SimsGenome
from .mutations import drop_unreached

if TYPE_CHECKING:
    import numpy as np


def _redirect(
    connection: SimsConnection, size: int, rng: np.random.Generator
) -> SimsConnection:
    """Point a connection past the last of `size` nodes at a random template.

    Returns
    -------
    SimsConnection
        The connection, unchanged if its child exists.
    """
    if connection.child < size:
        return connection
    return SimsConnection.model_validate(
        connection.model_dump() | {"child": int(rng.integers(1, size))}
    )


def crossover(
    first: SimsGenome, second: SimsGenome, rng: np.random.Generator
) -> SimsGenome:
    """Cross two genomes at one point of their template rows.

    The templates of both parents are lined up by index. The child takes
    those before a random cut from `first` and those from the cut on from
    `second`, so it is as long as `second`, or as `first` when the cut
    falls at the end of the shorter one. Each connection comes along with
    the template it starts from, `first`'s connections first; one whose
    child does not exist in the child genome points at a random template
    instead, and is dropped if the child genome is the core alone.

    Parameters
    ----------
    first
        The parent that gives the core and the templates before the cut.
    second
        The parent that gives the templates from the cut on.
    rng
        The source of randomness.

    Returns
    -------
    SimsGenome
        The child.
    """
    cut = int(rng.integers(1, min(len(first.nodes), len(second.nodes)) + 1))
    nodes = (*first.nodes[:cut], *second.nodes[cut:])
    inherited = [c for c in first.connections if c.parent < cut]
    inherited += [c for c in second.connections if c.parent >= cut]
    if len(nodes) == 1:
        return SimsGenome(nodes=nodes)
    connections = tuple(_redirect(c, len(nodes), rng) for c in inherited)
    return SimsGenome(nodes=nodes, connections=connections)


def graft(
    first: SimsGenome, second: SimsGenome, rng: np.random.Generator
) -> SimsGenome:
    """Graft part of `second` onto a copy of `first`.

    `second`'s templates are appended to `first`'s with their connections
    among themselves (the ones from `second`'s core are left behind). One
    of `first`'s connections, picked at random, then points at one of the
    appended templates instead, and every template nothing reaches any
    more is dropped. Without a connection in `first` or a template in
    `second`, there is nothing to graft, and the child is `first`.

    Parameters
    ----------
    first
        The parent that is copied.
    second
        The parent a piece is taken from.
    rng
        The source of randomness.

    Returns
    -------
    SimsGenome
        The child.
    """
    if not first.connections or len(second.nodes) == 1:
        return first
    offset = len(first.nodes) - 1  # second's template k becomes k + offset
    moved = [
        SimsConnection.model_validate(
            c.model_dump()
            | {"parent": c.parent + offset, "child": c.child + offset}
        )
        for c in second.connections
        if c.parent != IDX_OF_CORE
    ]
    index = int(rng.integers(len(first.connections)))
    target = int(rng.integers(1, len(second.nodes))) + offset
    redirected = SimsConnection.model_validate(
        first.connections[index].model_dump() | {"child": target}
    )
    connections = (
        *first.connections[:index],
        redirected,
        *first.connections[index + 1 :],
        *moved,
    )
    child = SimsGenome(
        nodes=(*first.nodes, *second.nodes[1:]), connections=connections
    )
    return drop_unreached(child)
