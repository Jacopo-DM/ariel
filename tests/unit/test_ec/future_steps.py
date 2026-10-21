"""Step functions from a module with postponed annotations.

With ``from __future__ import annotations`` every annotation is stored as
text, so ``EAOperation`` sees ``"Population"`` instead of the class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ariel.ec.population import Population

if TYPE_CHECKING:
    from collections.abc import Sequence


def keep_all(population: Population) -> Population:
    """Copy the population (a valid step).

    Returns
    -------
    Population
        A new population with the same individuals.
    """
    return Population(list(population))


def from_sequence(population: Sequence[int]) -> Population:
    """Take a type that exists only for the type checker (not resolvable).

    Returns
    -------
    Population
        An empty population.
    """
    del population
    return Population([])
