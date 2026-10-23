"""Hypothesis strategies and small genomes for the Sims genotype tests."""

from hypothesis import strategies as st

from ariel.body_phenotypes import (
    ALLOWED_FACES,
    ALLOWED_ROTATIONS,
    ModuleFaces,
    ModuleType,
)
from ariel.ec.genotypes.sims import SimsConnection, SimsGenome, SimsNode

MAX_TEMPLATES = 6
MAX_CONNECTIONS = 10
MAX_COPY_LIMIT = 6  # 20% of a 30-module body


@st.composite
def _node(draw: st.DrawFn) -> SimsNode:
    kind = draw(st.sampled_from([ModuleType.BRICK, ModuleType.HINGE]))
    return SimsNode(
        type=kind,
        rotation=draw(st.sampled_from(ALLOWED_ROTATIONS[kind])),
        copy_limit=draw(st.integers(1, MAX_COPY_LIMIT)),
    )


@st.composite
def genomes(draw: st.DrawFn) -> SimsGenome:
    """Draw a valid genome: the core, up to six templates, any connections.

    Returns
    -------
    SimsGenome
        A genome that passes every check.
    """
    nodes = (
        SimsNode(type=ModuleType.CORE),
        *draw(st.lists(_node(), max_size=MAX_TEMPLATES)),
    )
    connections: list[SimsConnection] = []
    if len(nodes) > 1:
        for _ in range(draw(st.integers(0, MAX_CONNECTIONS))):
            parent = draw(st.integers(0, len(nodes) - 1))
            faces: list[ModuleFaces] = ALLOWED_FACES[nodes[parent].type]
            connections.append(
                SimsConnection(
                    parent=parent,
                    child=draw(st.integers(1, len(nodes) - 1)),
                    face=draw(st.sampled_from(faces)),
                    reflect=draw(st.booleans()),
                    terminal_only=draw(st.booleans()),
                )
            )
    return SimsGenome(nodes=nodes, connections=tuple(connections))


def arm_genome() -> SimsGenome:
    """Return the example of PLAN.md: a mirrored pair of 3-brick arms.

    Returns
    -------
    SimsGenome
        Core plus brick template A (copy limit 3) that grows on its own
        FRONT face; the core grows A on its LEFT face, mirrored.
    """
    return SimsGenome(
        nodes=(
            SimsNode(type=ModuleType.CORE),
            SimsNode(type=ModuleType.BRICK, copy_limit=3),
        ),
        connections=(
            SimsConnection(
                parent=0, child=1, face=ModuleFaces.LEFT, reflect=True
            ),
            SimsConnection(parent=1, child=1, face=ModuleFaces.FRONT),
        ),
    )
