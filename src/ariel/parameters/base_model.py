"""Base classes for the pydantic records of ARIEL.

Every record in the package derives from one of these, so unknown fields are
rejected everywhere and immutability is the default.
"""

# Third-party libraries
from pydantic import BaseModel, ConfigDict


class FrozenModel(BaseModel):
    """An immutable record that rejects unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class MutableModel(BaseModel):
    """A record that may change after creation and rejects unknown fields."""

    model_config = ConfigDict(extra="forbid")
