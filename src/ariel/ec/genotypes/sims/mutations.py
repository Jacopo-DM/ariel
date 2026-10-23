"""Mutation operators for Sims genomes, after Sims (1994).

Each operator returns a new genome that differs from the one it was given;
the genome itself is frozen and never changes. `mutate` applies one
operator, chosen at random among those that apply to the genome.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ariel.body_phenotypes import (
    ALLOWED_FACES,
    ALLOWED_ROTATIONS,
    IDX_OF_CORE,
)
from ariel.parameters import FrozenModel

from .generate import (
    REFLECT_CHANCE,
    TYPES,
    random_connection,
    random_node,
)
from .genome import SimsGenome
from .unfold import DEFAULT_MAX_MODULES, max_copy_limit

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import numpy as np

    type Operator = Callable[[SimsGenome, np.random.Generator], SimsGenome]


MIN_FOR_NEW_CHILD = 3  # the core and two templates: another child exists


def _pick[T](rng: np.random.Generator, options: Sequence[T]) -> T:
    return options[int(rng.integers(len(options)))]


def _replace[M: FrozenModel](model: M, update: Mapping[str, object]) -> M:
    """Return a copy of `model` with `update` applied and checked.

    Returns
    -------
    M
        The new model; pydantic's ``model_copy`` would skip the checks.
    """
    return type(model).model_validate(model.model_dump() | dict(update))


def reached(genome: SimsGenome) -> set[int]:
    """Return the nodes a path of connections leads to from the core.

    Returns
    -------
    set[int]
        Node indices, the core included.
    """
    found = {IDX_OF_CORE}
    stack = [IDX_OF_CORE]
    while stack:
        node = stack.pop()
        for connection in genome.connections:
            if connection.parent == node and connection.child not in found:
                found.add(connection.child)
                stack.append(connection.child)
    return found


def _without(genome: SimsGenome, dropped: set[int]) -> SimsGenome:
    """Return `genome` without the `dropped` nodes and their connections.

    Returns
    -------
    SimsGenome
        The genome with the remaining nodes numbered again from 0.
    """
    kept = [i for i in range(len(genome.nodes)) if i not in dropped]
    new_index = {old: new for new, old in enumerate(kept)}
    return SimsGenome(
        nodes=tuple(genome.nodes[i] for i in kept),
        connections=tuple(
            _replace(
                c, {"parent": new_index[c.parent], "child": new_index[c.child]}
            )
            for c in genome.connections
            if c.parent in new_index and c.child in new_index
        ),
    )


def change_node(
    genome: SimsGenome, rng: np.random.Generator, cap: int
) -> SimsGenome:
    """Give one template a new type, rotation or copy limit (up to `cap`).

    A template that turns into a hinge keeps its outgoing connections; any
    on a face a hinge lacks moves to a face it has.

    Returns
    -------
    SimsGenome
        The changed genome.
    """
    index = int(rng.integers(1, len(genome.nodes)))
    node = genome.nodes[index]
    changes: list[str] = ["type", "rotation"]
    if cap > 1:
        changes.append("copy_limit")
    change = _pick(rng, changes)
    if change == "type":
        # Brick and hinge allow the same rotations, so the rotation stays
        kind = _pick(rng, [t for t in TYPES if t is not node.type])
        node = _replace(node, {"type": kind})
    elif change == "rotation":
        rotations = ALLOWED_ROTATIONS[node.type]
        rotation = _pick(rng, [r for r in rotations if r is not node.rotation])
        node = _replace(node, {"rotation": rotation})
    else:
        limits = [n for n in range(1, cap + 1) if n != node.copy_limit]
        node = _replace(node, {"copy_limit": _pick(rng, limits)})
    nodes = (*genome.nodes[:index], node, *genome.nodes[index + 1 :])
    faces = ALLOWED_FACES[node.type]
    connections = tuple(
        _replace(c, {"face": _pick(rng, faces)})
        if c.parent == index and c.face not in faces
        else c
        for c in genome.connections
    )
    return SimsGenome(nodes=nodes, connections=connections)


def change_connection(
    genome: SimsGenome, rng: np.random.Generator, *, mirror: bool
) -> SimsGenome:
    """Change one connection's parent, child, face or flags.

    The mirror flag is switched on only when `mirror` is on; it can always
    be switched off.

    Returns
    -------
    SimsGenome
        The changed genome.
    """
    index = int(rng.integers(len(genome.connections)))
    connection = genome.connections[index]
    count = len(genome.nodes)
    faces = ALLOWED_FACES[genome.nodes[connection.parent].type]
    changes = ["terminal_only", "parent"]
    if len(faces) > 1:
        changes.append("face")
    if count >= MIN_FOR_NEW_CHILD:
        changes.append("child")
    if mirror or connection.reflect:
        changes.append("reflect")
    change = _pick(rng, changes)
    if change == "parent":
        parent = _pick(rng, [p for p in range(count) if p != connection.parent])
        new_faces = ALLOWED_FACES[genome.nodes[parent].type]
        update: dict[str, object] = {
            "parent": parent,
            "face": _pick(rng, new_faces),
        }
    elif change == "face":
        update = {
            "face": _pick(rng, [f for f in faces if f is not connection.face])
        }
    elif change == "child":
        children = [k for k in range(1, count) if k != connection.child]
        update = {"child": _pick(rng, children)}
    else:
        update = {change: not getattr(connection, change)}
    changed = _replace(connection, update)
    connections = (
        *genome.connections[:index],
        changed,
        *genome.connections[index + 1 :],
    )
    return SimsGenome(nodes=genome.nodes, connections=connections)


def add_node(
    genome: SimsGenome, rng: np.random.Generator, cap: int, *, mirror: bool
) -> SimsGenome:
    """Add a random template and a connection to it from any node.

    Returns
    -------
    SimsGenome
        The genome with one more template, reached by the new connection.
    """
    nodes = (*genome.nodes, random_node(rng, cap))
    parent = int(rng.integers(len(genome.nodes)))
    chance = REFLECT_CHANCE if mirror else 0.0
    added = random_connection(rng, nodes, parent, len(genome.nodes), chance)
    return SimsGenome(nodes=nodes, connections=(*genome.connections, added))


def remove_node(genome: SimsGenome, rng: np.random.Generator) -> SimsGenome:
    """Remove one template and every connection to or from it.

    Returns
    -------
    SimsGenome
        The genome with one template fewer.
    """
    return _without(genome, {int(rng.integers(1, len(genome.nodes)))})


def add_connection(
    genome: SimsGenome, rng: np.random.Generator, *, mirror: bool
) -> SimsGenome:
    """Add a random connection from any node to any template.

    Returns
    -------
    SimsGenome
        The genome with the connection appended (it comes last, so it loses
        a face it shares with an older connection).
    """
    count = len(genome.nodes)
    chance = REFLECT_CHANCE if mirror else 0.0
    added = random_connection(
        rng,
        genome.nodes,
        int(rng.integers(count)),
        int(rng.integers(1, count)),
        chance,
    )
    return SimsGenome(
        nodes=genome.nodes, connections=(*genome.connections, added)
    )


def remove_connection(
    genome: SimsGenome, rng: np.random.Generator
) -> SimsGenome:
    """Remove one connection.

    Returns
    -------
    SimsGenome
        The genome with one connection fewer.
    """
    index = int(rng.integers(len(genome.connections)))
    connections = (
        *genome.connections[:index],
        *genome.connections[index + 1 :],
    )
    return SimsGenome(nodes=genome.nodes, connections=connections)


def drop_unreached(genome: SimsGenome) -> SimsGenome:
    """Remove every template no path of connections reaches from the core.

    Returns
    -------
    SimsGenome
        The genome without those templates and their connections.
    """
    return _without(genome, set(range(len(genome.nodes))) - reached(genome))


def _operators(genome: SimsGenome, cap: int, *, mirror: bool) -> list[Operator]:
    """Return the operators that change `genome`.

    Returns
    -------
    list[Operator]
        Each takes the genome and a generator.
    """
    templates = len(genome.nodes) > 1
    operators: list[Operator] = [
        lambda g, r: add_node(g, r, cap, mirror=mirror)
    ]
    if templates:
        operators += [
            lambda g, r: change_node(g, r, cap),
            remove_node,
            lambda g, r: add_connection(g, r, mirror=mirror),
        ]
    if genome.connections:
        operators += [
            lambda g, r: change_connection(g, r, mirror=mirror),
            remove_connection,
        ]
    if len(reached(genome)) < len(genome.nodes):
        operators.append(lambda g, _: drop_unreached(g))
    return operators


def mutate(
    genome: SimsGenome,
    rng: np.random.Generator,
    max_modules: int = DEFAULT_MAX_MODULES,
    *,
    mirror: bool = True,
) -> SimsGenome:
    """Apply one mutation, chosen at random among those that apply.

    Parameters
    ----------
    genome
        The genome to start from; it is not changed.
    rng
        The source of randomness.
    max_modules
        Most modules a body may have; it caps new copy limits at a fifth.
    mirror
        Whether a mutation may switch a mirror flag on, as in
        `random_genome`.

    Returns
    -------
    SimsGenome
        A new genome that differs from `genome`.
    """
    operators = _operators(genome, max_copy_limit(max_modules), mirror=mirror)
    return _pick(rng, operators)(genome, rng)
