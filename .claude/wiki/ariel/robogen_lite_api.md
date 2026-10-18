---
title: "robogen_lite — ARIEL Body Phenotype API"
authors: "CI-Group (VU Amsterdam)"
year: 2024
source: "https://ci-group.github.io/ariel/autoapi/ariel/body_phenotypes/robogen_lite/index.html"
tags: [ariel, body-phenotype, modular-robot, robogen, cppn, neat, mujoco, evolutionary-robotics]
type: ariel_reference
date_ingested: 2026-04-11
---

# robogen_lite_api

Module path: `ariel.body_phenotypes.robogen_lite`

Provides modular robot morphology generation and evolution ("ARIEL-robots") using CPPN-NEAT encoding and MuJoCo simulation. The full pipeline is: **Genome → Decoder → NetworkX DiGraph → Constructor → CoreModule (MuJoCo spec)**.

---

## Module Structure

```
robogen_lite/
├── config.py            # Enums and constants (ModuleType, faces, rotations)
├── constructor.py       # Graph → MuJoCo spec conversion
├── cppn_neat/           # NEAT-based CPPN implementation
│   ├── activations      # Activation functions (ActivationFunction enum)
│   ├── connection       # Connection genes
│   ├── genome           # Genome class (main entry point)
│   ├── id_manager       # Innovation / node ID counters
│   ├── node             # Node gene
│   └── tests
├── decoders/            # CPPN genome → morphology graph
│   ├── cppn_best_first  # Greedy best-first decoder
│   └── hi_prob_decoding # High-probability decoder
├── modules/             # Physical module classes
│   ├── module.py        # Abstract base Module
│   ├── core.py          # CoreModule
│   ├── brick.py         # BrickModule
│   └── hinge.py         # HingeModule
└── prebuilt_robots/     # Ready-to-use morphologies
    ├── gecko
    ├── spider
    └── spider_with_blocks
```

---

## config — Enums & Constants

### ModuleType (enum)
| Name   | Value |
|--------|-------|
| CORE   | 0     |
| BRICK  | 1     |
| HINGE  | 2     |
| NONE   | 3     |

### ModuleFaces (enum) — 6 attachment faces
`FRONT=0, BACK=1, RIGHT=2, LEFT=3, TOP=4, BOTTOM=5`

### ModuleRotationsIdx / ModuleRotationsTheta (enums)
8 rotation steps in 45° increments: `DEG_0` (0°) through `DEG_315` (315°).

### ModuleInstance (Pydantic BaseModel)
Represents one module in a genome/graph node:
- `type_: ModuleType`
- `rotation_: ModuleRotationsIdx`
- `links_: dict[ModuleFaces, int]` — maps face → child module index
- `ALLOWED_FACES_`: type-specific valid attachment faces
- `ALLOWED_ROTATIONS_`: type-specific valid rotation indices

### Numeric constants
| Constant              | Value |
|-----------------------|-------|
| `IDX_OF_CORE`         | 0     |
| `NUM_OF_TYPES_OF_MODULES` | (total module types) |
| `NUM_OF_FACES`        | 6     |
| `NUM_OF_ROTATIONS`    | 8     |

---

## Modules — Physical Components

All module classes inherit from `Module` (abstract base).

### CoreModule
```python
# ariel.body_phenotypes.robogen_lite.modules.core
```
Central body; other modules attach to it.

| Attribute / Constant  | Type / Value               |
|-----------------------|----------------------------|
| `index`               | `int \| None = None`       |
| `module_type`         | `ModuleType`               |
| `CORE_MASS`           | `float = 1`                |
| `CORE_DIMENSIONS`     | `tuple = (0.1, 0.1, 0.1)` |

**`rotate(angle: float) → None`** — raises `AttributeError` (core does not support rotation).

### BrickModule
```python
# ariel.body_phenotypes.robogen_lite.modules.brick
```
Passive structural block. Inherits `rotate(angle: float)`.

### HingeModule
```python
# ariel.body_phenotypes.robogen_lite.modules.hinge
```
Articulated joint. Constructor: `HingeModule(index: int)`.

| Constant            | Value                    |
|---------------------|--------------------------|
| `SHRINK`            | 0.99                     |
| `STATOR_MASS`       | 0.02                     |
| `ROTOR_MASS`        | 0.04                     |
| `STATOR_DIMENSIONS` | (0.025, 0.03, 0.025)     |
| `ROTOR_DIMENSIONS`  | (0.025, 0.02, 0.025)     |

**`rotate(angle: float) → None`** — rotates hinge by `angle` radians.

Common properties on all modules: `sites`, `spec`, `body`.

---

## constructor — Graph → MuJoCo

```python
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
```

### `construct_mjspec_from_graph(graph: networkx.DiGraph) → CoreModule`
Converts a NetworkX directed graph (robot structure) into a MuJoCo-simulatable `CoreModule` tree.

- **Input:** `nx.DiGraph` where nodes encode `ModuleInstance` data and edges encode parent→child attachment.
- **Output:** `CoreModule` root object containing all sub-modules with MuJoCo specs populated.
- **Raises:** `ValueError` if the graph contains an unrecognized module type.

This is the bridge between evolutionary genotype decoding and physical simulation.

---

## cppn_neat — NEAT Genome

### Genome
```python
from ariel.body_phenotypes.robogen_lite.cppn_neat.genome import Genome
```

#### Attributes
- `nodes: dict[int, Node]`
- `connections: dict[int, Connection]`
- `fitness: float`
- `serialized: dict | None` — optional cached dict

#### Construction
```python
genome = Genome.random(
    num_inputs=...,
    num_outputs=...,
    next_node_id=...,
    next_innov_id=...
)
```
Creates a fully-connected input→output topology.

#### Mutation & Crossover
```python
genome.mutate(
    node_add_rate=...,
    conn_add_rate=...,
    next_innov_id_getter=...,
    next_node_id_getter=...
)
child = genome.crossover(other_genome)   # → Genome
```

#### Network Evaluation
```python
outputs = genome.activate(inputs)        # → list[float]
ordering = genome.get_node_ordering()    # topological sort (Kahn's algorithm)
```
`activate` falls back to iterative relaxation if cycles exist.

#### Serialization
```python
d = genome.to_dict()
genome2 = Genome.from_dict(d, fitness=0.0)
copy = genome.copy()                     # deep copy
```

### Node
```python
Node(_id: int, _typ: str, _activation: ActivationFunction, _bias: float)
```
- `typ` — node type (input / hidden / output)
- `activation` — `ActivationFunction` enum value
- `bias` — float bias
- `copy()` → new Node with identical values

---

## decoders — CPPN → Morphology Graph

### MorphologyDecoderBestFirst
```python
from ariel.body_phenotypes.robogen_lite.decoders.cppn_best_first import MorphologyDecoderBestFirst
```

```python
decoder = MorphologyDecoderBestFirst(
    cppn_genome: Genome,
    max_modules: int = 20      # hard cap on robot size
)
graph: nx.DiGraph = decoder.decode()
```

Greedy best-first search: iteratively selects the highest-scoring attachment position (via softmax over raw CPPN outputs) until `max_modules` is reached.

**`softmax(raw_scores: NDArray[float32]) → NDArray[float32]`** — helper in this module.

### hi_prob_decoding
Alternative decoder using a high-probability selection strategy (details in submodule page).

---

## prebuilt_robots

Ready-to-use `CoreModule` instances for benchmarking:

| Submodule           | Robot       |
|---------------------|-------------|
| `gecko`             | Gecko       |
| `spider`            | Spider      |
| `spider_with_blocks`| Spider + blocks |

```python
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.spider import ...
```

---

## Typical Full Pipeline

```python
from ariel.body_phenotypes.robogen_lite.cppn_neat.genome import Genome
from ariel.body_phenotypes.robogen_lite.decoders.cppn_best_first import MorphologyDecoderBestFirst
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph

# 1. Create / evolve genome
genome = Genome.random(num_inputs=8, num_outputs=6,
                       next_node_id=id_mgr, next_innov_id=innov_mgr)

# 2. Decode genome → morphology graph
decoder = MorphologyDecoderBestFirst(genome, max_modules=20)
graph = decoder.decode()          # nx.DiGraph

# 3. Build MuJoCo spec
robot = construct_mjspec_from_graph(graph)   # CoreModule

# 4. Simulate with MuJoCo (robot.spec etc.)
```

---

## Key Design Notes

- **Module attachment:** each module exposes up to 6 faces (`ModuleFaces`); only type-specific `ALLOWED_FACES_` are valid attachment points.
- **Rotation:** 8 discrete orientations (45° steps); `CoreModule` does not support rotation.
- **Graph encoding:** a `nx.DiGraph` is the canonical intermediate representation between decoder and constructor.
- **CPPN outputs → module selection:** softmax over raw scores governs which face/module type is placed next.
- **max_modules=20** is the default; smaller values speed up evaluation, larger values allow more complex morphologies.

See also: [[cppn_neat_genome]], [[MjSpec]], body_brain_learning (page not written yet)
