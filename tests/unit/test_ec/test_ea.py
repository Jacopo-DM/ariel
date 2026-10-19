"""Test: the EA class."""

from pathlib import Path

import pytest

from ariel.ec import EA, EAOperation
from ariel.ec.individual import Individual
from ariel.ec.population import Population
from tests.unit.test_ec.ea_helpers import (
    ALIVE_COUNT,
    FIRST_GENERATION_ID,
    INITIAL_POP_SIZE,
    NUM_STEPS_RUN,
    STEPS_TAKEN,
    make_ea,
    make_pop,
    noop,
)


def test_ea_initializes(tmp_path: Path) -> None:
    """EA initializes without error given population and ops."""
    ea = make_ea(tmp_path)
    assert ea.current_generation == 0


def test_ea_no_operations_raises(tmp_path: Path) -> None:
    """EA raises ValueError when operations is None."""
    with pytest.raises(ValueError, match="operations"):
        EA(
            population=make_pop(),
            operations=None,
            db_file_path=tmp_path / "db.db",
            quiet=True,
        )


def test_ea_no_population_no_restart_raises(tmp_path: Path) -> None:
    """EA raises ValueError when neither population nor restart is given."""
    with pytest.raises(ValueError, match="population"):
        EA(
            population=None,
            operations=[noop],
            db_file_path=tmp_path / "db.db",
            quiet=True,
        )


def test_ea_creates_database_file(tmp_path: Path) -> None:
    """EA creates the SQLite database file on disk."""
    db = tmp_path / "ea.db"
    make_ea(tmp_path, ops=[noop])
    assert db.exists() or (tmp_path / "test.db").exists()


def test_ea_db_handling_halt_raises(tmp_path: Path) -> None:
    """db_handling='halt' raises FileExistsError when db already exists."""
    db = tmp_path / "existing.db"
    db.touch()
    with pytest.raises(FileExistsError):
        EA(
            population=make_pop(),
            operations=[noop],
            db_file_path=db,
            quiet=True,
            db_handling="halt",
        )


def test_ea_db_handling_delete_overwrites(tmp_path: Path) -> None:
    """db_handling='delete' recreates the db without raising."""
    db = tmp_path / "run.db"
    ea1 = EA(
        population=make_pop(),
        operations=[noop],
        db_file_path=db,
        quiet=True,
        db_handling="delete",
    )
    # Explicitly dispose the engine so Windows releases the file lock
    ea1.engine.dispose()
    del ea1
    # Second run on same path — should succeed
    EA(
        population=make_pop(),
        operations=[noop],
        db_file_path=db,
        quiet=True,
        db_handling="delete",
    )


def test_ea_population_stored(tmp_path: Path) -> None:
    """EA stores the initial population."""
    pop = make_pop(INITIAL_POP_SIZE)
    ea = make_ea(tmp_path, pop=pop)
    assert len(ea.population) == INITIAL_POP_SIZE


def test_ea_custom_first_generation(tmp_path: Path) -> None:
    """first_generation_id is stored on the EA."""
    ea = make_ea(tmp_path, first_generation_id=FIRST_GENERATION_ID)
    assert ea.current_generation == FIRST_GENERATION_ID


def test_ea_commit_sets_time_of_birth(tmp_path: Path) -> None:
    """_commit assigns time_of_birth=-1 individuals the current generation."""
    ea = make_ea(tmp_path)
    # Fetch fresh from DB so SQLAlchemy session context is active
    pop = ea._fetch()  # ruff: ignore[private-member-access]
    for ind in pop:
        assert ind.time_of_birth == 0


def test_ea_fetch_returns_population(tmp_path: Path) -> None:
    """_fetch returns a Population instance."""
    ea = make_ea(tmp_path)
    pop = ea._fetch()  # ruff: ignore[private-member-access]
    assert isinstance(pop, Population)


def test_ea_fetch_only_alive(tmp_path: Path) -> None:
    """_fetch(only_alive=True) returns only alive individuals."""
    ea = make_ea(tmp_path)
    pop = ea._fetch(only_alive=True)  # ruff: ignore[private-member-access]
    assert all(ind.alive for ind in pop)


def test_ea_fetch_sort_desc(tmp_path: Path) -> None:
    """_fetch(sort='desc') returns individuals sorted by fitness descending."""
    ea = make_ea(tmp_path)
    pop = ea._fetch(sort="desc")  # ruff: ignore[private-member-access]
    fitnesses = [ind.fitness_ for ind in pop if ind.fitness_ is not None]
    assert fitnesses == sorted(fitnesses, reverse=True)


def test_ea_fetch_sort_asc(tmp_path: Path) -> None:
    """_fetch(sort='asc') returns individuals sorted by fitness ascending."""
    ea = make_ea(tmp_path)
    pop = ea._fetch(sort="asc")  # ruff: ignore[private-member-access]
    fitnesses = [ind.fitness_ for ind in pop if ind.fitness_ is not None]
    assert fitnesses == sorted(fitnesses)


def test_ea_fetch_requires_eval_false(tmp_path: Path) -> None:
    """_fetch(requires_eval=False) returns only evaluated individuals."""
    ea = make_ea(tmp_path)
    pop = ea._fetch(requires_eval=False)  # ruff: ignore[private-member-access]
    assert all(not ind.requires_eval for ind in pop)


def test_ea_fetch_population_updates_self(tmp_path: Path) -> None:
    """fetch_population() updates self.population from the database."""
    ea = make_ea(tmp_path)
    original_len = len(ea.population)
    ea.fetch_population()
    assert len(ea.population) == original_len


def test_ea_size_property(tmp_path: Path) -> None:
    """Size returns the number of alive individuals."""
    ea = make_ea(tmp_path, pop=make_pop(ALIVE_COUNT))
    assert ea.size == ALIVE_COUNT


def test_ea_get_solution_best(tmp_path: Path) -> None:
    """get_solution('best') returns an Individual."""
    ea = make_ea(tmp_path)
    sol = ea.get_solution(mode="best")
    assert isinstance(sol, Individual)


def test_ea_get_solution_median(tmp_path: Path) -> None:
    """get_solution('median') returns an Individual."""
    ea = make_ea(tmp_path)
    sol = ea.get_solution(mode="median")
    assert isinstance(sol, Individual)


def test_ea_get_solution_worst(tmp_path: Path) -> None:
    """get_solution('worst') returns an Individual."""
    ea = make_ea(tmp_path)
    sol = ea.get_solution(mode="worst")
    assert isinstance(sol, Individual)


def test_ea_step_increments_generation(tmp_path: Path) -> None:
    """step() increments current_generation by 1."""
    ea = make_ea(tmp_path)
    ea.step()
    assert ea.current_generation == 1


def test_ea_step_applies_operations(tmp_path: Path) -> None:
    """step() applies all operations and the result is committed to the DB."""
    called = []

    @EAOperation
    def recording_op(population: Population) -> Population:
        called.append(True)
        return population

    ea = make_ea(tmp_path, pop=make_pop(3), ops=[recording_op])
    ea.step()
    assert len(called) == 1


def test_ea_step_twice(tmp_path: Path) -> None:
    """Two steps advance generation to 2."""
    ea = make_ea(tmp_path)
    ea.step()
    ea.step()
    assert ea.current_generation == STEPS_TAKEN


def test_ea_run_executes_num_steps(tmp_path: Path) -> None:
    """run() executes exactly num_steps generations."""
    ea = make_ea(tmp_path, num_steps=NUM_STEPS_RUN)
    ea.run()
    assert ea.current_generation == NUM_STEPS_RUN


def test_ea_run_zero_steps(tmp_path: Path) -> None:
    """run() with num_steps=0 leaves generation unchanged."""
    ea = make_ea(tmp_path, num_steps=0)
    ea.run()
    assert ea.current_generation == 0
