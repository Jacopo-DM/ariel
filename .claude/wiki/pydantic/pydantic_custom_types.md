---
type: api_reference
tags: [pydantic, types, typeadapter, annotated]
source: https://pydantic.dev/docs/validation/latest/concepts/types/
date_ingested: 2026-10-18
---
# pydantic_custom_types

Reusable constrained types, `TypeAdapter`, and custom types for third-party classes.

## Annotated and TypeAdapter

```python
from typing import Annotated
from pydantic import Field, TypeAdapter, ValidationError

PositiveInt = Annotated[int, Field(gt=0)]
ta = TypeAdapter(PositiveInt)
print(ta.validate_python(1))
```

`annotated-types` constraints such as `Gt(0)` keep the definition independent of Pydantic. Markers such as `AfterValidator`, `PlainSerializer` and `WithJsonSchema` add validation, serialisation or schema behaviour. See [[Field]] and [[pydantic_validators]].

## Named type aliases (v2.11+)

A `type` statement or `TypeAliasType` gives the alias a shared `$defs` JSON Schema entry and allows recursive aliases. Field-specific metadata (`alias`, `default`, `deprecated`) is not supported inside a named alias.

## __get_pydantic_core_schema__

Customises how a class or annotation is validated; described as middleware-like.

```python
class Username(str):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(cls, handler(str))
```

It can sit on the type or on a marker class placed in `Annotated`, which is the way to support a third-party type (for example a marker that accepts ints and converts them to an object the library owns). `GetPydanticSchema` removes boilerplate for simple cases. The page recommends the high-level hooks (`AfterValidator`, `Field`) when they are enough.

Generic custom classes read type parameters through `source_type` with `get_args` and `handler.generate_schema(...)`; "you don't need to use `arbitrary_types_allowed`" when the classmethod is defined.

## Notes

- The page does not list the built-in constrained types; it points to the standard library types docs.
- For an astropy `Quantity` field, a marker class with `__get_pydantic_core_schema__` that checks the unit's physical type is the route (see [[Quantity]] and [[PhysicalType]]). This is an inference from the page, not an example on it.
