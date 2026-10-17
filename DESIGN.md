# DESIGN

## Specs

None recorded.

## Decisions

Stack and architecture as found in the repo. Reasons are not recorded unless stated.

- Python ≥ 3.12 with uv for environments and dependencies (`uv.lock`); setuptools build backend; `nox` for task automation (docs only today). Reason: the README names uv; no other reason recorded.
- Simulation on MuJoCo (`mujoco`, `mujoco-mjx[warp]`, `dm-control`); bodies are built as MuJoCo `MjSpec` objects. Reason not recorded.
- Evolution engine in `ariel.ec`: `EA` with operations over a `Population` of `Individual`s stored with SQLModel/SQLAlchemy (database-backed, with restart from the database and an `Archive`). Reason not recorded.
- Body encodings: RoboGen-lite modular robots (`Core`, `Brick`, `Hinge`) decoded from genotypes (tree, CPPN-NEAT, NDE in PyTorch, vectors). The Lynx arm is a separate body family.
- Controllers are CPG-based (`SimpleCPG`, `NaCPG`), and tasks are free-function fitness functions.
- Optimisation libraries available: DEAP, EvoTorch, Nevergrad, plus an in-repo `RevDE`.
- Dashboards use Plotly/Panel/NiceGUI.
- Typing and docs: strict ruff (`ALL`), strict mypy and pyright, NumPy docstrings, 100% coverage target. Free functions and `@dataclass` are preferred over classes (`.claude/dev/functional_vs_oo.md`).
- Global state at import: `ariel/__init__.py` sets `DATA`, a seeded `RNG` (seed 42) and rich logging. Reason not recorded.

## Rejected approaches

None recorded. The `ompl` dependency was removed in commit 08f5b10 ("problematic"); no other reason is recorded.

## Results

None recorded.

## Gotchas

See "Things that bite" in CLAUDE.md for repo-wide ones. None specific to a design.
