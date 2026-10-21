"""Node gene of a CPPN genome."""

from __future__ import annotations


class Node:
    """A node gene in a NEAT genome."""

    def __init__(
        self, _id: int, typ: str, activation: str | None, bias: float
    ) -> None:
        """Create a node with an ID, type, activation name and bias.

        The activation is a key of ``ACTIVATION_FUNCTIONS``; input nodes
        have none.
        """
        self._id = _id
        self.typ = typ
        self.activation = activation
        self.bias = bias

    def copy(self) -> Node:
        """Return a new Node object with identical values.

        Returns
        -------
        Node
            A copy with the same values.
        """
        return Node(self._id, self.typ, self.activation, self.bias)
