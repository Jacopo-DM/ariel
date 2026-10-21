"""Vector-decoding for ARIEL-robots."""

# Evaluate type annotations in a deferred manner (ruff: UP037)
from __future__ import annotations

# Standard library
from typing import TYPE_CHECKING, TypeVar, cast, overload

# Third-party libraries
import networkx as nx
import numpy as np
import numpy.typing as npt
from networkx import DiGraph

# Local libraries
from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_FACES,
    ALLOWED_ROTATIONS,
    IDX_OF_CORE,
    ModuleFaces,
    ModuleRotationsIdx,
    ModuleType,
)

# Third-party libraries
from ariel.body_phenotypes.robogen_lite.decoders._blueprint import Blueprint

if TYPE_CHECKING:
    from collections.abc import Sequence

T = TypeVar("T")


class VectorDecoder(Blueprint):
    """Implements the vector-decoding algorithm."""

    @overload
    def assign_symbols_from_range(  # pragma: no cover  # typing stub only
        self,
        vector: npt.NDArray[np.float64] | Sequence[float],
        symbols: Sequence[Sequence[T]],
        weight: Sequence[Sequence[float]] | None = None,
    ) -> list[T]: ...

    @overload
    def assign_symbols_from_range(  # pragma: no cover  # typing stub only
        self,
        vector: npt.NDArray[np.float64] | Sequence[float],
        symbols: Sequence[T],
        weight: Sequence[float] | None = None,
    ) -> list[T]: ...

    def assign_symbols_from_range(  # ruff: ignore[no-self-use]  # keeps the interface shared with the sibling classes
        self,
        vector: npt.NDArray[np.float64] | Sequence[float],
        symbols: Sequence[T] | Sequence[Sequence[T]],
        weight: Sequence[float] | Sequence[Sequence[float]] | None = None,
    ) -> list[T]:
        """Map each value in [0, 1] to a symbol; ValueError if out of range.

        Returns
        -------
        list[T]
            One symbol per value.

        Raises
        ------
        ValueError
            If a value is outside [0, 1].
        """
        vector = np.asarray(vector, dtype=np.float64)

        if np.any(vector < 0) or np.any(vector > 1):
            msg = "All values in the vector must be within [0, 1]."
            raise ValueError(msg)

        result: list[T] = []

        # Detect per-element symbols
        per_element = isinstance(symbols[0], (list, tuple, np.ndarray))

        for i, value in enumerate(vector):
            # per_element tells which of the two shapes symbols has
            current_symbols = cast(
                "Sequence[T]", symbols[i] if per_element else symbols
            )

            num_symbols = len(current_symbols)

            # Handle weights
            if weight is not None:
                current_weight = (
                    np.asarray(weight[i], dtype=np.float64)
                    if per_element
                    else np.asarray(weight, dtype=np.float64)
                )
                current_weight /= current_weight.sum()
            else:
                current_weight = (
                    np.ones(num_symbols, dtype=np.float64) / num_symbols
                )

            cumulative_weight = np.cumsum(current_weight)

            symbol_idx = np.searchsorted(cumulative_weight, value)
            symbol_idx = min(symbol_idx, num_symbols - 1)

            result.append(current_symbols[symbol_idx])

        return result

    def vectors_to_graph(
        self,
        type_vector: npt.NDArray[np.float64],
        connection_vector: npt.NDArray[np.float64],
        rotation_vector: npt.NDArray[np.float64],
    ) -> DiGraph[int]:
        """
        Convert three vectors of values in [0, 1] to a graph.

        Each vector has one value per module except the core, so
        ``num_modules - 1`` values. Module ``k`` (1 to ``num_modules - 1``,
        in order) is decoded as follows:

        * ``type_vector[k - 1]`` picks its type among every type but the core
            (BRICK, HINGE, NONE). A NONE module is left out of the graph.
        * ``connection_vector[k - 1]`` picks one free ``(parent, face)`` slot:
            a face that the type of an already placed module allows and that no
            other module uses. There is always one, since the core has four
            faces and every placed module offers at least as many as it uses.
            The result is always a tree rooted at the core.
        * ``rotation_vector[k - 1]`` picks its rotation among the rotations
            its type allows.

        Parameters
        ----------
        type_vector
            Vector for module types.
        connection_vector
            Vector for connections between modules.
        rotation_vector
            Vector for module rotations.

        Returns
        -------
        DiGraph
            A graph representing the decoded modules and their connections.

        Raises
        ------
        ValueError
            If a vector does not hold ``num_modules - 1`` values, or a value
            is outside [0, 1].
        """
        expected = self.num_modules - 1
        for name, vector in (
            ("type", type_vector),
            ("connection", connection_vector),
            ("rotation", rotation_vector),
        ):
            if len(vector) != expected:
                msg = f"The {name} vector must hold {expected} values."
                msg += f" Got {len(vector)}."
                raise ValueError(msg)

        # Reset the graph
        self._graph = {}
        self.graph = nx.DiGraph()

        self.type_vector = type_vector.copy()
        self.connection_vector = connection_vector.copy()
        self.rotation_vector = rotation_vector.copy()

        self.type_dict = {IDX_OF_CORE: ModuleType.CORE}
        self.rot_dict = {IDX_OF_CORE: ModuleRotationsIdx.DEG_0}
        self.edges: list[tuple[int, int, int]] = []

        if expected > 0:
            self._place_modules()

        self.nodes = set(self.type_dict)
        self.generate_networkx_graph()
        return self.graph

    def _place_modules(self) -> None:
        """Place modules 1 to ``num_modules - 1`` from the stored vectors."""
        types = self.assign_symbols_from_range(
            vector=self.type_vector,
            symbols=[t for t in ModuleType if t is not ModuleType.CORE],
        )
        allowed: list[Sequence[ModuleRotationsIdx]] = [
            ALLOWED_ROTATIONS[t] for t in types
        ]
        rotations: list[ModuleRotationsIdx] = self.assign_symbols_from_range(
            vector=self.rotation_vector,
            symbols=allowed,
        )
        used: set[tuple[int, ModuleFaces]] = set()
        for k, module_type in enumerate(types, start=1):
            if module_type is ModuleType.NONE:
                continue
            slots = [
                (parent, face)
                for parent, parent_type in self.type_dict.items()
                for face in ALLOWED_FACES[parent_type]
                if (parent, face) not in used
            ]
            (slot_idx,) = self.assign_symbols_from_range(
                vector=[self.connection_vector[k - 1]],
                symbols=list(range(len(slots))),
            )
            parent, face = slots[slot_idx]
            used.add((parent, face))
            self.edges.append((parent, k, face.value))
            self.type_dict[k] = module_type
            self.rot_dict[k] = rotations[k - 1]
