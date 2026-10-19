"""Test: CPPN genome construction, mutation, crossover, activation, IO."""

import json
import math
import random
from pathlib import Path

import pytest

from ariel.body_phenotypes.robogen_lite.cppn_neat.activations import sigmoid
from ariel.body_phenotypes.robogen_lite.cppn_neat.connection import Connection
from ariel.body_phenotypes.robogen_lite.cppn_neat.genome import Genome
from ariel.body_phenotypes.robogen_lite.cppn_neat.id_manager import IdManager
from ariel.body_phenotypes.robogen_lite.cppn_neat.node import Node

NUM_INPUTS = 2
NUM_OUTPUTS = 1
BIAS = 0.5
WEIGHT_A = 2.0
WEIGHT_B = -1.0
FITTER = 10.0
LESS_FIT = 1.0
NODE_PAIR = 2
FIRST_INNOV = 100
FIRST_ID = 1
UNKNOWN_NODE = 99
EXTRA_INNOV = 7
LAST_NODE_START = 3
LAST_INNOV_START = 7


def _node(node_id: int, typ: str, *, bias: float = 0.0) -> Node:
    # Node genes store the activation by name; the annotation says callable.
    return Node(node_id, typ, "sigmoid" if typ != "input" else None, bias)  # type: ignore[arg-type]  # known bug


def _chain_genome() -> Genome:
    # Two inputs -> one output, with known weights and bias.
    nodes = {
        0: _node(0, "input"),
        1: _node(1, "input"),
        2: _node(2, "output", bias=BIAS),
    }
    connections = {
        0: Connection(0, 2, WEIGHT_A, enabled=True, innov_id=0),
        1: Connection(1, 2, WEIGHT_B, enabled=True, innov_id=1),
    }
    return Genome(nodes, connections, fitness=0.0)


def _ids() -> IdManager:
    return IdManager(node_start=10, innov_start=FIRST_INNOV)


# --- construction ----------------------------------------------------------- #


def test_random_genome_is_fully_connected() -> None:
    """Inputs and outputs exist and every input feeds every output."""
    genome = Genome.random(
        NUM_INPUTS, NUM_OUTPUTS, NUM_INPUTS + NUM_OUTPUTS, FIRST_INNOV
    )
    types = sorted(n.typ for n in genome.nodes.values())
    assert types == ["input"] * NUM_INPUTS + ["output"] * NUM_OUTPUTS
    assert len(genome.connections) == NUM_INPUTS * NUM_OUTPUTS
    assert min(genome.connections) == FIRST_INNOV
    assert genome.fitness == pytest.approx(0.0)


def test_copy_is_deep() -> None:
    """Changing the copy leaves the original alone."""
    original = _chain_genome()
    original.fitness = FITTER
    clone = original.copy()
    clone.nodes[2].bias = 99.0
    clone.connections[0].weight = 99.0
    assert original.nodes[2].bias == BIAS
    assert original.connections[0].weight == WEIGHT_A
    assert clone.fitness == FITTER


def test_node_and_connection_copy() -> None:
    """Copies hold the same values and are new objects."""
    node = _node(5, "hidden", bias=BIAS)
    node_copy = node.copy()
    assert node_copy is not node
    assert vars(node_copy) == vars(node)
    conn = Connection(1, 2, WEIGHT_A, enabled=False, innov_id=7)
    conn_copy = conn.copy()
    assert conn_copy is not conn
    assert vars(conn_copy) == vars(conn)


# --- adding genes ----------------------------------------------------------- #


def test_add_connection_and_node_reject_duplicates() -> None:
    """Adding the same gene twice raises ValueError."""
    genome = _chain_genome()
    node = genome.nodes[2]
    conn = genome.connections[0]
    with pytest.raises(ValueError, match="Connection already exists"):
        genome.add_connection(conn)
    with pytest.raises(ValueError, match="Node already exists"):
        genome.add_node(node)


def test_add_node_registers_by_id() -> None:
    """A new node is stored under its id."""
    genome = _chain_genome()
    hidden = _node(7, "hidden")
    genome.add_node(hidden)
    assert genome.nodes[7] is hidden


# --- mutation --------------------------------------------------------------- #


def test_mutate_add_node_splits_a_connection() -> None:
    """Add-node disables one connection and adds a node plus two links."""
    random.seed(0)
    genome = _chain_genome()
    ids = _ids()
    genome.mutate(
        node_add_rate=1.0,
        conn_add_rate=0.0,
        next_innov_id_getter=ids.get_next_innov_id,
        next_node_id_getter=ids.get_next_node_id,
    )
    assert len(genome.nodes) == NUM_INPUTS + NUM_OUTPUTS + 1
    assert len(genome.connections) == NUM_INPUTS * NUM_OUTPUTS + NODE_PAIR
    assert sum(not c.enabled for c in genome.connections.values()) == 1
    hidden = [n for n in genome.nodes.values() if n.typ == "hidden"]
    assert len(hidden) == 1


def test_mutate_add_node_without_connections_does_nothing() -> None:
    """With no connection to split, the genome is unchanged."""
    genome = Genome({0: _node(0, "input")}, {}, fitness=0.0)
    ids = _ids()
    genome.mutate(1.0, 0.0, ids.get_next_innov_id, ids.get_next_node_id)
    assert len(genome.nodes) == 1


def test_mutate_add_connection_links_two_nodes() -> None:
    """A hidden node without links gets connected."""
    random.seed(1)
    genome = _chain_genome()
    genome.add_node(_node(7, "hidden"))
    ids = _ids()
    before = len(genome.connections)
    for _ in range(50):
        genome.mutate(0.0, 1.0, ids.get_next_innov_id, ids.get_next_node_id)
    assert len(genome.connections) > before
    for conn in genome.connections.values():
        assert genome.nodes[conn.in_id].typ != "output"


def test_mutate_add_connection_needs_two_nodes() -> None:
    """A genome with one node cannot get a connection."""
    genome = Genome({0: _node(0, "input")}, {}, fitness=0.0)
    ids = _ids()
    genome.mutate(0.0, 1.0, ids.get_next_innov_id, ids.get_next_node_id)
    assert genome.connections == {}


def test_mutate_add_connection_never_makes_a_duplicate() -> None:
    """Two nodes that are already connected stay connected once."""
    genome = Genome(
        {0: _node(0, "input"), 1: _node(1, "output")},
        {0: Connection(0, 1, WEIGHT_A, enabled=True, innov_id=0)},
        fitness=0.0,
    )
    ids = _ids()
    for _ in range(20):
        genome.mutate(0.0, 1.0, ids.get_next_innov_id, ids.get_next_node_id)
    assert len(genome.connections) == 1


def test_mutate_add_connection_rejects_output_as_source() -> None:
    """Two output nodes cannot be linked (an output may not feed a node)."""
    genome = Genome(
        {0: _node(0, "output"), 1: _node(1, "output")}, {}, fitness=0.0
    )
    ids = _ids()
    for _ in range(10):
        genome.mutate(0.0, 1.0, ids.get_next_innov_id, ids.get_next_node_id)
    assert genome.connections == {}


def test_mutate_add_connection_swaps_when_destination_is_input() -> None:
    """If the random pair is (output, input) the direction is flipped."""
    genome = Genome(
        {0: _node(0, "output"), 1: _node(1, "input")}, {}, fitness=0.0
    )
    ids = _ids()
    for _ in range(10):
        genome.mutate(0.0, 1.0, ids.get_next_innov_id, ids.get_next_node_id)
    assert [(c.in_id, c.out_id) for c in genome.connections.values()] == [
        (1, 0)
    ]


# --- crossover -------------------------------------------------------------- #


def _parent_with_extra_gene(fitness: float) -> Genome:
    genome = _chain_genome()
    genome.fitness = fitness
    genome.add_node(_node(7, "hidden"))
    genome.add_connection(Connection(0, 7, 1.0, enabled=True, innov_id=5))
    return genome


def test_crossover_keeps_matching_and_fitter_genes() -> None:
    """Genes only in the less fit parent are dropped when fitness differs."""
    random.seed(2)
    fitter = _parent_with_extra_gene(FITTER)
    less = _chain_genome()
    less.fitness = LESS_FIT
    child = fitter.crossover(less)
    assert set(child.connections) == {0, 1, 5}
    assert child.fitness == pytest.approx(0.0)


def test_crossover_equal_fitness_inherits_every_gene() -> None:
    """Equal fitness: the union of genes is inherited."""
    random.seed(3)
    a = _parent_with_extra_gene(FITTER)
    b = _chain_genome()
    b.fitness = FITTER
    b.add_connection(Connection(1, 2, 3.0, enabled=True, innov_id=9))
    child = a.crossover(b)
    assert set(child.connections) == {0, 1, 5, 9}


def test_crossover_equal_fitness_shorter_genome_is_parent_b() -> None:
    """With equal fitness the genome with fewer genes plays parent B."""
    random.seed(4)
    short = _chain_genome()
    long = _parent_with_extra_gene(0.0)
    child = short.crossover(long)
    assert set(child.connections) == {0, 1, 5}


def test_crossover_minimisation_prefers_lower_fitness() -> None:
    """In minimisation the lower fitness is the fitter parent."""
    random.seed(5)
    better = _parent_with_extra_gene(LESS_FIT)
    worse = _chain_genome()
    worse.fitness = FITTER
    child = worse.crossover(better, is_maximisation=False)
    assert {0, 1} <= set(child.connections)


def test_crossover_minimisation_self_better_drops_other_genes() -> None:
    """Minimisation, self fitter: the other parent's own genes are dropped."""
    random.seed(6)
    better = _chain_genome()
    better.fitness = LESS_FIT
    worse = _parent_with_extra_gene(FITTER)
    child = better.crossover(worse, is_maximisation=False)
    assert set(child.connections) == {0, 1}


def test_crossover_maximisation_self_worse_inherits_from_other() -> None:
    """Maximisation, self is worse: the child gets the genes of the other."""
    worse = _chain_genome()
    worse.fitness = LESS_FIT
    better = _parent_with_extra_gene(FITTER)
    child = worse.crossover(better)
    assert set(child.connections) == {0, 1, 5}


@pytest.mark.parametrize("is_maximisation", [True, False])
def test_crossover_1000_pairs_inherit_every_gene_of_the_fitter_parent(
    *, is_maximisation: bool
) -> None:
    """1000 random pairs never raise and keep all genes of the fitter one."""
    rng = random.Random(42)  # ruff: ignore[suspicious-non-cryptographic-random-usage]  # test data
    random.seed(42)
    for _ in range(1000):
        a = _chain_genome()
        b = _chain_genome()
        for genome in (a, b):
            genome.fitness = float(rng.randint(0, 3))
            for innov in rng.sample(range(5, 12), rng.randint(0, 4)):
                genome.add_connection(
                    Connection(1, 2, 1.0, enabled=True, innov_id=innov)
                )
        child = a.crossover(b, is_maximisation=is_maximisation)
        if a.fitness == b.fitness:
            continue
        a_fitter = (a.fitness > b.fitness) == is_maximisation
        fitter = a if a_fitter else b
        assert set(fitter.connections) <= set(child.connections)


def test_crossover_skips_genes_only_the_less_fit_parent_has() -> None:
    """With different fitness the less fit parent's own genes are dropped."""
    random.seed(7)
    fitter = _chain_genome()
    fitter.fitness = FITTER
    less = _parent_with_extra_gene(LESS_FIT)
    child = fitter.crossover(less)
    assert set(child.connections) == {0, 1}


def test_crossover_ignores_connections_to_unknown_nodes() -> None:
    """A connection whose node is in neither parent adds no node gene."""
    random.seed(8)
    a = _chain_genome()
    a.fitness = FITTER
    a.connections[EXTRA_INNOV] = Connection(
        0, UNKNOWN_NODE, 1.0, enabled=True, innov_id=EXTRA_INNOV
    )
    child = a.crossover(_chain_genome())
    assert UNKNOWN_NODE not in child.nodes
    assert EXTRA_INNOV in child.connections


# --- ordering and activation ------------------------------------------------ #


def test_node_ordering_puts_inputs_before_outputs() -> None:
    """The feed-forward order starts with the inputs."""
    order = _chain_genome().get_node_ordering()
    assert order[-1] == NUM_INPUTS
    assert set(order[:NUM_INPUTS]) == {0, 1}


def test_node_ordering_rejects_cycles() -> None:
    """A recurrent genome has no feed-forward order."""
    genome = _chain_genome()
    genome.add_node(_node(7, "hidden"))
    genome.add_connection(Connection(2, 7, 1.0, enabled=True, innov_id=5))
    genome.add_connection(Connection(7, 2, 1.0, enabled=True, innov_id=6))
    with pytest.raises(Exception, match="cycle"):
        genome.get_node_ordering()


def test_activate_feed_forward_matches_hand_calculation() -> None:
    """The output is sigmoid(w1 x1 + w2 x2 + bias)."""
    x0, x1 = 0.3, 0.8
    out = _chain_genome().activate([x0, x1])
    expected = sigmoid(WEIGHT_A * x0 + WEIGHT_B * x1 + BIAS)
    assert out == [pytest.approx(expected)]


def test_activate_ignores_disabled_connections() -> None:
    """A disabled connection does not contribute."""
    genome = _chain_genome()
    genome.connections[1].enabled = False
    x0, x1 = 0.3, 0.8
    assert genome.activate([x0, x1]) == [
        pytest.approx(sigmoid(WEIGHT_A * x0 + BIAS))
    ]


def test_activate_recurrent_genome_skips_disabled_connections() -> None:
    """A disabled connection of a cyclic genome does not feed its target."""
    genome = _chain_genome()
    genome.add_node(_node(7, "hidden"))
    genome.add_connection(Connection(2, 7, 1.0, enabled=True, innov_id=5))
    genome.add_connection(Connection(7, 2, 1.0, enabled=True, innov_id=6))
    genome.add_connection(Connection(0, 7, 1.0, enabled=False, innov_id=8))
    out = genome.activate([0.1, 0.2])
    assert math.isfinite(out[0])


def test_activate_wrong_input_count_raises() -> None:
    """The input vector must match the number of input nodes."""
    with pytest.raises(ValueError, match="Expected 2 inputs"):
        _chain_genome().activate([1.0])


def test_activate_recurrent_genome_uses_relaxation() -> None:
    """A genome with a cycle still produces finite outputs."""
    genome = _chain_genome()
    genome.add_node(_node(7, "hidden"))
    genome.add_connection(Connection(2, 7, 1.0, enabled=True, innov_id=5))
    genome.add_connection(Connection(7, 2, 1.0, enabled=True, innov_id=6))
    out = genome.activate([0.1, 0.2])
    assert len(out) == NUM_OUTPUTS
    assert math.isfinite(out[0])


# --- serialisation ---------------------------------------------------------- #


def test_to_dict_from_dict_roundtrip() -> None:
    """A genome survives a dict (and JSON) roundtrip."""
    genome = _chain_genome()
    data = json.loads(json.dumps(genome.to_dict()))
    rebuilt = Genome.from_dict(data)
    assert rebuilt.to_dict() == genome.to_dict()
    assert rebuilt.activate([0.1, 0.2]) == genome.activate([0.1, 0.2])


def test_from_dict_rejects_bad_data() -> None:
    """Missing keys raise a ValueError (pydantic ValidationError)."""
    with pytest.raises(ValueError, match="nodes"):
        Genome.from_dict({"connections": []})


# --- id manager ------------------------------------------------------------- #


def test_id_manager_counts_up_in_memory() -> None:
    """In-memory mode starts after the given ids."""
    ids = IdManager(node_start=LAST_NODE_START, innov_start=LAST_INNOV_START)
    assert ids.get_next_node_id() == LAST_NODE_START + 1
    assert ids.get_next_innov_id() == LAST_INNOV_START + 1


def test_id_manager_set_initial_ids(capsys: pytest.CaptureFixture[str]) -> None:
    """The counters follow the topology and the change is printed."""
    ids = IdManager(node_start=0, innov_start=0)
    ids.set_initial_ids(NUM_INPUTS, NUM_OUTPUTS)
    assert ids.get_next_node_id() == NUM_INPUTS + NUM_OUTPUTS
    assert ids.get_next_innov_id() == NUM_INPUTS * NUM_OUTPUTS
    assert "IdManager initialized" in capsys.readouterr().out


def test_id_manager_persists(tmp_path: Path) -> None:
    """save() then a new manager on the same file continues the count."""
    path = tmp_path / "ids.json"
    first = IdManager(save_path=path)
    first.get_next_node_id()
    first.get_next_innov_id()
    first.save()
    second = IdManager(save_path=path)
    assert second.get_next_node_id() == NODE_PAIR
    assert second.get_next_innov_id() == NODE_PAIR


def test_id_manager_corrupt_file_starts_fresh(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unreadable state file is reported and ignored."""
    path = tmp_path / "ids.json"
    path.write_text("{not json", encoding="utf-8")
    ids = IdManager(save_path=path)
    assert ids.get_next_node_id() == FIRST_ID
    assert "Error loading ID state" in capsys.readouterr().out


def test_id_manager_save_failure_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A save into a missing folder prints an error instead of raising."""
    ids = IdManager(save_path=tmp_path / "missing" / "ids.json")
    ids.save()
    assert "Error saving ID state" in capsys.readouterr().out
