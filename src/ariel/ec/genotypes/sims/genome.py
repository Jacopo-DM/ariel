"""A body genotype after Karl Sims (1994): a graph of module templates.

A node is a template for a module: its type, its rotation and its copy
limit, how many copies of it may appear in a row along one chain. A
connection says that a copy of its child grows on one face of every copy of
its parent. A connection may point back at its own parent, or at an earlier
node, so a small graph can stand for a long chain of modules. Node 0 is the
core, the root of every body.

Unfolding the graph into a body is a separate step (goal sims-genotype,
PLAN.md); this module holds the genotype, its checks and its JSON form.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Annotated, Self

from pydantic import BeforeValidator, Field, PlainSerializer, model_validator

from ariel.body_phenotypes import (
    ALLOWED_FACES,
    ALLOWED_ROTATIONS,
    IDX_OF_CORE,
    ModuleFaces,
    ModuleRotationsIdx,
    ModuleType,
)
from ariel.parameters import FrozenModel

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _by_name[E: Enum](enum_cls: type[E]) -> Callable[[object], object]:
    """Return a parser that turns a member name into that member.

    Returns
    -------
    Callable[[object], object]
        Maps a name to its member and passes anything else through, so a
        member given as such is checked by pydantic as usual.
    """

    def parse(value: object) -> object:
        if isinstance(value, str):
            if value not in enum_cls.__members__:
                msg = f"'{value}' is not a {enum_cls.__name__} name"
                raise ValueError(msg)
            return enum_cls[value]
        return value

    return parse


def _name(member: Enum) -> str:
    return member.name


# Enum fields are stored by member name in JSON ("BRICK", not 1)
TypeName = Annotated[
    ModuleType, BeforeValidator(_by_name(ModuleType)), PlainSerializer(_name)
]
RotationName = Annotated[
    ModuleRotationsIdx,
    BeforeValidator(_by_name(ModuleRotationsIdx)),
    PlainSerializer(_name),
]
FaceName = Annotated[
    ModuleFaces, BeforeValidator(_by_name(ModuleFaces)), PlainSerializer(_name)
]


class SimsNode(FrozenModel):
    """A module template: type, rotation and copy limit."""

    type: TypeName
    rotation: RotationName = ModuleRotationsIdx.DEG_0
    copy_limit: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _rotation_allowed(self) -> Self:
        if self.rotation not in ALLOWED_ROTATIONS[self.type]:
            msg = f"{self.type.name} does not allow {self.rotation.name}"
            raise ValueError(msg)
        return self


class SimsConnection(FrozenModel):
    """A child template grown on one face of every copy of its parent.

    ``reflect`` asks for a mirrored twin across the body's centre plane, and
    ``terminal_only`` limits the connection to the last copy of a chain.
    """

    parent: int = Field(ge=0)
    child: int = Field(ge=0)
    face: FaceName
    reflect: bool = False
    terminal_only: bool = False


class SimsGenome(FrozenModel):
    """Templates and their connections; node 0 is the core.

    Connections keep their order: when two start on the same face of the
    same template, unfolding builds the first one. Nodes that no connection
    reaches are allowed; they do nothing until a mutation links them.
    """

    nodes: tuple[SimsNode, ...] = (SimsNode(type=ModuleType.CORE),)
    connections: tuple[SimsConnection, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        self._check_nodes()
        self._check_connections()
        return self

    def _check_nodes(self) -> None:
        if (
            not self.nodes
            or self.nodes[IDX_OF_CORE].type is not ModuleType.CORE
        ):
            msg = f"Node {IDX_OF_CORE} must be the core"
            raise ValueError(msg)
        if self.nodes[IDX_OF_CORE].copy_limit != 1:
            msg = "The core has a copy limit of 1"
            raise ValueError(msg)
        for index, node in enumerate(self.nodes):
            if index == IDX_OF_CORE:
                continue
            if node.type in {ModuleType.CORE, ModuleType.NONE}:
                msg = f"Node {index} is {node.type.name}; only the root is"
                msg += " the core, and NONE is not a module"
                raise ValueError(msg)

    def _check_connections(self) -> None:
        count = len(self.nodes)
        for connection in self.connections:
            parent, child = connection.parent, connection.child
            if parent >= count or child >= count:
                msg = f"Connection {parent} -> {child} names a node that"
                msg += f" does not exist ({count} nodes)"
                raise ValueError(msg)
            if child == IDX_OF_CORE:
                msg = f"Connection {parent} -> {child}: the core is never"
                msg += " a child"
                raise ValueError(msg)
            parent_type = self.nodes[parent].type
            if connection.face not in ALLOWED_FACES[parent_type]:
                msg = f"Connection {parent} -> {child}: {parent_type.name}"
                msg += f" has no {connection.face.name} face"
                raise ValueError(msg)

    def save_json(self, path: Path) -> None:
        """Write the genome to `path` as JSON."""
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> SimsGenome:
        """Read a genome from the JSON file at `path`.

        Returns
        -------
        SimsGenome
            The genome; a file that is not a valid genome raises
            ``pydantic.ValidationError``.
        """
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
