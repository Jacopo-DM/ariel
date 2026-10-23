"""Test: unfolding a Sims genome into a body graph."""

from __future__ import annotations

import itertools

import mujoco
import networkx as nx
import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import example, given

from ariel.body_phenotypes import (
    ALLOWED_FACES,
    ModuleFaces,
    ModuleRotationsIdx,
    ModuleRotationsTheta,
    ModuleType,
    construct_mjspec_from_graph,
)
from ariel.ec.genotypes.sims import (
    SimsConnection,
    SimsGenome,
    SimsNode,
    max_copy_limit,
    unfold,
)
from ariel.ec.genotypes.sims.unfold import (
    DEFAULT_MAX_MODULES,
    core_mirror,
    mirror_child,
)
from ariel.simulators import compile_model
from tests.unit.test_ec.test_genotypes.test_sims.genomes import (
    arm_genome,
    genomes,
)

CORE = SimsNode(type=ModuleType.CORE)
FRONT, BACK = ModuleFaces.FRONT, ModuleFaces.BACK
LEFT, RIGHT = ModuleFaces.LEFT, ModuleFaces.RIGHT
TOP, BOTTOM = ModuleFaces.TOP, ModuleFaces.BOTTOM
MIRROR = np.array([-1.0, 1.0, 1.0])  # through the core's centre plane
SYMMETRY_TOLERANCE = 1e-3  # metres; the modules round their quaternions
# Without that rounding a mirror is exact up to floating point: the rounding
# alone leaves 0.15 mm in a 5-module body and 1.10 mm in a 23-module chain
# of 45-degree bricks and hinges, and 0 mm at every length without it
EXACT_TOLERANCE = 1e-6  # metres
CORNERS = np.array([
    [x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)
])
ARM_MODULES = 7
SMALL_BODY = 3
PAIRED_BODY = 16  # core, two 6-module arms, a 3-brick spine
LEG_BODY = 29  # core and four legs of 7 after four rounds
LEG_LENGTH = 7
BIG_BODY = 100
BIG_CAP = 20
DEFAULT_CAP = 6
CHAIN = 3
SPINE = 4
BEND = 0.6  # radians


def _genome(
    nodes: list[SimsNode], connections: list[SimsConnection]
) -> SimsGenome:
    return SimsGenome(nodes=(CORE, *nodes), connections=tuple(connections))


def _connect(
    parent: int,
    child: int,
    face: ModuleFaces,
    *,
    reflect: bool = False,
    terminal_only: bool = False,
) -> SimsConnection:
    return SimsConnection(
        parent=parent,
        child=child,
        face=face,
        reflect=reflect,
        terminal_only=terminal_only,
    )


def _faces(graph: nx.DiGraph[int], node: int) -> set[str]:
    return {str(graph.edges[edge]["face"]) for edge in graph.out_edges(node)}


def _chain_lengths(graph: nx.DiGraph[int]) -> list[int]:
    return sorted(
        len(nx.descendants(graph, child)) + 1 for child in graph.successors(0)
    )


def _corners(
    graph: nx.DiGraph[int], bends: dict[str, float] | None = None
) -> npt.NDArray[np.float64]:
    """Return every geom corner of the built body, in the core's frame.

    Returns
    -------
    npt.NDArray[np.float64]
        One row per corner.
    """
    model = compile_model(construct_mjspec_from_graph(graph).spec)
    data = mujoco.MjData(model)
    for prefix, angle in (bends or {}).items():
        for joint in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            if name.startswith(prefix):
                data.qpos[model.jnt_qposadr[joint]] = angle
    mujoco.mj_forward(model, data)
    corners = np.concatenate([
        data.geom_xpos[geom]
        + (CORNERS * model.geom_size[geom])
        @ data.geom_xmat[geom].reshape(3, 3).T
        for geom in range(model.ngeom)
    ])
    core = model.geom_bodyid[0]
    local = (corners - data.xpos[core]) @ data.xmat[core].reshape(3, 3)
    return np.asarray(local, dtype=np.float64)


def _asymmetry(
    graph: nx.DiGraph[int], bends: dict[str, float] | None = None
) -> float:
    """Return how far the body is from its own mirror image, in metres.

    Returns
    -------
    float
        The largest distance from a mirrored corner to the nearest corner.
    """
    corners = _corners(graph, bends)
    mirrored = corners * MIRROR
    return max(
        float(np.min(np.linalg.norm(corners - point, axis=1)))
        for point in mirrored
    )


@pytest.mark.parametrize(
    ("max_modules", "cap"),
    [(30, DEFAULT_CAP), (BIG_BODY, BIG_CAP), (SMALL_BODY, 1)],
)
def test_max_copy_limit(max_modules: int, cap: int) -> None:
    """The copy limit is at most a fifth of the module maximum, at least 1."""
    assert max_copy_limit(max_modules) == cap


def test_core_alone() -> None:
    """A genome with no connections unfolds into the core alone."""
    graph = unfold(SimsGenome())
    assert len(graph) == 1
    assert graph.nodes[0] == {
        "type": "CORE",
        "rotation": "DEG_0",
        "template": 0,
    }


def test_arm_example() -> None:
    """The PLAN.md example: a mirrored pair of 3-brick arms, 7 modules."""
    graph = unfold(arm_genome())
    assert len(graph) == ARM_MODULES
    assert _faces(graph, 0) == {"LEFT", "RIGHT"}
    assert _chain_lengths(graph) == [CHAIN, CHAIN]
    assert {graph.nodes[n]["template"] for n in graph if n} == {1}
    assert _asymmetry(graph) < SYMMETRY_TOLERANCE


def test_copy_limit_ends_a_chain() -> None:
    """A template connected to itself stops at its copy limit."""
    genome = _genome(
        [SimsNode(type=ModuleType.BRICK, copy_limit=CHAIN)],
        [_connect(0, 1, FRONT), _connect(1, 1, FRONT)],
    )
    assert _chain_lengths(unfold(genome)) == [CHAIN]


def test_copy_limit_is_capped_by_the_maximum() -> None:
    """A copy limit above a fifth of the maximum is cut to that fifth."""
    genome = _genome(
        [SimsNode(type=ModuleType.BRICK, copy_limit=50)],
        [_connect(0, 1, FRONT), _connect(1, 1, FRONT)],
    )
    assert _chain_lengths(unfold(genome)) == [DEFAULT_CAP]
    assert _chain_lengths(unfold(genome, max_modules=BIG_BODY)) == [BIG_CAP]


def _branching_legs() -> SimsGenome:
    """Return four legs of 6 bricks, each brick with a side brick: 49 modules.

    Returns
    -------
    SimsGenome
        A genome that asks for more than 30 modules.
    """
    return _genome(
        [
            SimsNode(type=ModuleType.BRICK, copy_limit=DEFAULT_CAP),
            SimsNode(type=ModuleType.BRICK),
        ],
        [
            *(_connect(0, 1, face) for face in (FRONT, BACK, LEFT, RIGHT)),
            _connect(1, 1, FRONT),
            _connect(1, 2, TOP),
        ],
    )


def test_chains_grow_evenly_when_the_maximum_cuts_them() -> None:
    """Every leg grows one module per round, so the four stay equal."""
    graph = unfold(_branching_legs(), max_modules=LEG_BODY)
    assert len(graph) == LEG_BODY
    assert _chain_lengths(graph) == [LEG_LENGTH] * 4


def test_twins_are_placed_together_or_not_at_all() -> None:
    """With room for one more module, a mirrored pair is left out whole."""
    genome = _genome(
        [
            SimsNode(type=ModuleType.BRICK, copy_limit=CHAIN),
            SimsNode(type=ModuleType.BRICK),
            SimsNode(type=ModuleType.BRICK, copy_limit=CHAIN),
        ],
        [
            _connect(0, 1, LEFT, reflect=True),
            _connect(0, 3, FRONT),
            _connect(1, 1, FRONT),
            _connect(1, 2, TOP),
            _connect(3, 3, FRONT),
        ],
    )
    full = unfold(genome, max_modules=PAIRED_BODY)
    assert len(full) == PAIRED_BODY
    cut = unfold(genome, max_modules=PAIRED_BODY - 1)
    assert len(cut) == PAIRED_BODY - 2  # the last pair did not fit
    assert _asymmetry(cut) < SYMMETRY_TOLERANCE


def test_terminal_connection_grows_from_the_last_copy_only() -> None:
    """A terminal-only hinge appears once, on the third brick of three."""
    genome = _genome(
        [
            SimsNode(type=ModuleType.BRICK, copy_limit=CHAIN),
            SimsNode(type=ModuleType.HINGE),
        ],
        [
            _connect(0, 1, FRONT),
            _connect(1, 1, FRONT),
            _connect(1, 2, TOP, terminal_only=True),
        ],
    )
    graph = unfold(genome)
    hinges = [n for n in graph if graph.nodes[n]["type"] == "HINGE"]
    assert len(hinges) == 1
    (parent,) = graph.predecessors(hinges[0])
    assert nx.shortest_path_length(graph, 0, parent) == CHAIN


def test_first_connection_on_a_face_wins() -> None:
    """Of two connections on the core's FRONT face, the first is built."""
    genome = _genome(
        [SimsNode(type=ModuleType.BRICK), SimsNode(type=ModuleType.HINGE)],
        [_connect(0, 1, FRONT), _connect(0, 2, FRONT)],
    )
    graph = unfold(genome)
    assert len(graph) == SMALL_BODY - 1
    assert graph.nodes[1]["type"] == "BRICK"


def test_unreached_templates_add_nothing() -> None:
    """A template no connection reaches never appears."""
    genome = _genome(
        [SimsNode(type=ModuleType.BRICK), SimsNode(type=ModuleType.HINGE)],
        [_connect(0, 1, FRONT)],
    )
    assert [d["type"] for _, d in unfold(genome).nodes(data=True)] == [
        "CORE",
        "BRICK",
    ]


@pytest.mark.parametrize(
    "connections",
    [
        pytest.param([_connect(0, 1, FRONT, reflect=True)], id="on-plane-face"),
        pytest.param(
            [_connect(0, 1, LEFT), _connect(1, 1, RIGHT, reflect=True)],
            id="off-plane-module",
        ),
    ],
)
def test_reflect_adds_nothing_where_it_cannot(
    connections: list[SimsConnection],
) -> None:
    """A mirror flag on the centre plane, or off it, grows no twin."""
    genome = _genome(
        [SimsNode(type=ModuleType.BRICK, copy_limit=2)], connections
    )
    assert len(unfold(genome)) == len(connections) + 1


def test_reflect_inside_a_twin_adds_nothing() -> None:
    """A flagged connection inside a mirrored pair grows no extra twins."""
    genome = _genome(
        [SimsNode(type=ModuleType.BRICK, copy_limit=CHAIN)],
        [
            _connect(0, 1, LEFT, reflect=True),
            _connect(1, 1, FRONT, reflect=True),
        ],
    )
    assert len(unfold(genome)) == ARM_MODULES


def test_mirrored_45_degree_hinge_is_tilted_minus_45() -> None:
    """The twin of a 45-degree hinge is at -45, and bends as a mirror."""
    genome = _genome(
        [SimsNode(type=ModuleType.HINGE, rotation=ModuleRotationsIdx.DEG_45)],
        [_connect(0, 1, LEFT, reflect=True)],
    )
    graph = unfold(genome)
    assert [graph.nodes[n]["rotation"] for n in (1, 2)] == [
        "DEG_45",
        "DEG_MINUS_45",
    ]
    bends = {"0-1-": BEND, "0-2-": -BEND}  # a mirror flips the angle's sign
    assert _asymmetry(graph, bends) < SYMMETRY_TOLERANCE


def test_spine_with_mirrored_legs() -> None:
    """A spine along the centre plane gets a leg on each side of each brick."""
    genome = _genome(
        [
            SimsNode(type=ModuleType.BRICK, copy_limit=SPINE),
            SimsNode(type=ModuleType.HINGE, rotation=ModuleRotationsIdx.DEG_45),
            SimsNode(type=ModuleType.BRICK, rotation=ModuleRotationsIdx.DEG_45),
        ],
        [
            _connect(0, 1, FRONT),
            _connect(1, 1, FRONT),
            _connect(1, 2, RIGHT, reflect=True),
            _connect(2, 3, FRONT),
        ],
    )
    graph = unfold(genome)
    spine = [n for n in graph if graph.nodes[n]["template"] == 1]
    assert len(spine) == SPINE
    assert all(_faces(graph, n) >= {"LEFT", "RIGHT"} for n in spine)
    assert _asymmetry(graph) < SYMMETRY_TOLERANCE


@pytest.mark.parametrize(
    ("brick_tilt", "hinge_tilt", "face"),
    list(
        itertools.product(
            ModuleRotationsIdx, ModuleRotationsIdx, (FRONT, TOP, BOTTOM, LEFT)
        )
    ),
)
def test_mirrored_limbs_are_exact_mirrors(
    brick_tilt: ModuleRotationsIdx,
    hinge_tilt: ModuleRotationsIdx,
    face: ModuleFaces,
) -> None:
    """Every tilt of brick and hinge, on any face, mirrors exactly."""
    genome = _genome(
        [
            SimsNode(type=ModuleType.BRICK, rotation=brick_tilt, copy_limit=2),
            SimsNode(type=ModuleType.HINGE, rotation=hinge_tilt),
            SimsNode(type=ModuleType.BRICK),
        ],
        [
            _connect(0, 1, RIGHT, reflect=True),
            _connect(1, 1, face),
            _connect(1, 2, RIGHT if face is LEFT else LEFT),
            _connect(2, 3, FRONT),
        ],
    )
    assert _asymmetry(unfold(genome)) < SYMMETRY_TOLERANCE


def test_mirror_child_needs_a_reflection() -> None:
    """A matrix that maps no face onto a mirror image raises ValueError."""
    identity = tuple(float(v) for v in np.eye(3).ravel())
    with pytest.raises(ValueError, match="No mirror image"):
        mirror_child(ModuleType.BRICK, identity, TOP, "DEG_0", ModuleType.BRICK)


def test_core_mirror_swaps_left_and_right() -> None:
    """The core's reflection is through the plane between LEFT and RIGHT."""
    for face, image in ((LEFT, RIGHT), (FRONT, FRONT), (BACK, BACK)):
        placed = mirror_child(
            ModuleType.CORE, core_mirror(), face, "DEG_0", ModuleType.BRICK
        )
        assert placed.face is image


@given(genomes())
def test_every_genome_unfolds_into_a_buildable_body(genome: SimsGenome) -> None:
    """Any valid genome gives a tree the builder accepts, within the maximum."""
    graph = unfold(genome)
    assert len(graph) <= DEFAULT_MAX_MODULES
    assert nx.is_arborescence(graph)
    for parent, child in graph.edges:
        kind = ModuleType[graph.nodes[parent]["type"]]
        assert (
            ModuleFaces[graph.edges[parent, child]["face"]]
            in (ALLOWED_FACES[kind])
        )
    assert all(
        rotation in ModuleRotationsTheta.__members__
        for _, rotation in graph.nodes(data="rotation")
    )
    compile_model(construct_mjspec_from_graph(graph).spec)


def _unrounded(
    values: npt.ArrayLike, decimals: int = 0, out: None = None
) -> npt.NDArray[np.float64]:
    """Stand in for ``np.round`` and leave the values as they are.

    Returns
    -------
    npt.NDArray[np.float64]
        The values, unrounded.
    """
    del decimals, out
    return np.asarray(values, dtype=np.float64)


# A 23-module chain of 45-degree bricks and hinges, drawn by Hypothesis on
# 2026-10-23: 1.10 mm asymmetric with the modules' rounding, 0 without
CHAIN_OF_45 = SimsGenome(
    nodes=(
        CORE,
        SimsNode(
            type=ModuleType.BRICK,
            rotation=ModuleRotationsIdx.DEG_45,
            copy_limit=DEFAULT_CAP,
        ),
        SimsNode(
            type=ModuleType.HINGE,
            rotation=ModuleRotationsIdx.DEG_45,
            copy_limit=DEFAULT_CAP - 1,
        ),
    ),
    connections=(_connect(2, 1, FRONT), _connect(1, 2, RIGHT)),
)


@given(genomes())
@example(CHAIN_OF_45)
def test_mirrored_limbs_make_symmetric_bodies(genome: SimsGenome) -> None:
    """A core that grows only one mirrored limb gives an exact mirror image.

    The modules round their quaternions to 3 decimals, and in a long chain
    that rounding alone adds up past a millimetre; this test builds without
    it, so it checks the mirroring itself, exactly, at any chain length.
    """
    limb = (
        (_connect(0, 1, LEFT, reflect=True),) if len(genome.nodes) > 1 else ()
    )
    rest = tuple(c for c in genome.connections if c.parent != 0)
    symmetric = genome.model_copy(update={"connections": limb + rest})
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(np, "round", _unrounded)
        asymmetry = _asymmetry(unfold(symmetric))
    assert asymmetry < EXACT_TOLERANCE
