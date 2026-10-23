"""Draw random Sims genomes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ariel.body_phenotypes import (
    ALLOWED_FACES,
    ALLOWED_ROTATIONS,
    IDX_OF_CORE,
    ModuleType,
)

from .genome import SimsConnection, SimsGenome, SimsNode
from .unfold import DEFAULT_MAX_MODULES, max_copy_limit, unfold

if TYPE_CHECKING:
    import numpy as np

MAX_TEMPLATES = 4  # module templates besides the core
EXTRA_PER_TEMPLATE = 2  # extra connections, at most this many per template
MIN_MODULES = 10  # a smaller body is drawn again (the user, 2026-10-23)
REFLECT_CHANCE = 0.5
TERMINAL_CHANCE = 0.1
TYPES = (ModuleType.BRICK, ModuleType.HINGE)


def random_node(rng: np.random.Generator, cap: int) -> SimsNode:
    """Draw a brick or hinge template with copy limit 1 to `cap`.

    Returns
    -------
    SimsNode
        The template, with a rotation its type allows.
    """
    kind = TYPES[rng.integers(len(TYPES))]
    rotations = ALLOWED_ROTATIONS[kind]
    return SimsNode(
        type=kind,
        rotation=rotations[rng.integers(len(rotations))],
        copy_limit=int(rng.integers(1, cap + 1)),
    )


def random_connection(
    rng: np.random.Generator,
    nodes: tuple[SimsNode, ...],
    parent: int,
    child: int,
    reflect_chance: float,
) -> SimsConnection:
    """Draw a connection from `parent` to `child` on a face it allows.

    Returns
    -------
    SimsConnection
        Mirrored with `reflect_chance`, terminal-only a tenth of the time.
    """
    faces = ALLOWED_FACES[nodes[parent].type]
    return SimsConnection(
        parent=parent,
        child=child,
        face=faces[rng.integers(len(faces))],
        reflect=bool(rng.random() < reflect_chance),
        terminal_only=bool(rng.random() < TERMINAL_CHANCE),
    )


def _draw(rng: np.random.Generator, cap: int, *, mirror: bool) -> SimsGenome:
    templates = int(rng.integers(1, MAX_TEMPLATES + 1))
    nodes = (
        SimsNode(type=ModuleType.CORE),
        *(random_node(rng, cap) for _ in range(templates)),
    )
    chance = REFLECT_CHANCE if mirror else 0.0
    # Template k is reached from the core or an earlier template
    connections = [
        random_connection(
            rng,
            nodes,
            int(rng.integers(child)) if child > 1 else IDX_OF_CORE,
            child,
            chance,
        )
        for child in range(1, len(nodes))
    ]
    extra = int(rng.integers(EXTRA_PER_TEMPLATE * templates + 1))
    connections += [
        random_connection(
            rng,
            nodes,
            int(rng.integers(len(nodes))),
            int(rng.integers(1, len(nodes))),
            chance,
        )
        for _ in range(extra)
    ]
    return SimsGenome(nodes=nodes, connections=tuple(connections))


def random_genome(
    rng: np.random.Generator,
    max_modules: int = DEFAULT_MAX_MODULES,
    *,
    mirror: bool = True,
) -> SimsGenome:
    """Draw a random genome for bodies of at most `max_modules` modules.

    The genome has the core and 1 to 4 templates, each a brick or a hinge
    with any rotation it allows and a copy limit from 1 to the highest
    `max_modules` allows. Each template is reached by one connection from
    the core or an earlier template, and up to 2 more connections per
    template join any node to any template. Every connection uses a face
    its parent allows, is mirrored half the time when `mirror` is on, and is
    terminal-only a tenth of the time. A genome whose body would have fewer
    than 10 modules (or `max_modules`, if smaller) is drawn again.

    Parameters
    ----------
    rng
        The source of randomness.
    max_modules
        Most modules the body may have; it sets the highest copy limit.
    mirror
        Whether connections may carry the mirror flag. Without it the
        core's four faces are used about equally often; with it LEFT and
        RIGHT are used more, since mirrored twins grow only on the sides.

    Returns
    -------
    SimsGenome
        The genome.
    """
    cap = max_copy_limit(max_modules)
    least = min(MIN_MODULES, max_modules)
    while True:
        genome = _draw(rng, cap, mirror=mirror)
        if len(unfold(genome, max_modules)) >= least:
            return genome
