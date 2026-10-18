---
type: api_reference
tags: [pydantic, field, computed-field, annotated]
source: https://pydantic.dev/docs/validation/latest/concepts/fields/
date_ingested: 2026-10-18
---
# Field

`pydantic.Field()` sets defaults, constraints and metadata on a model field; `computed_field` adds a property to serialised output. Models: [[BaseModel]].

## Parameters

| Name | Purpose |
|---|---|
| `default`, `default_factory` | default value; since v2.10 `default_factory` can take the already-validated data as one argument, so the field it reads must be defined earlier |
| `validate_default` | validate the default (off by default) |
| `alias`, `validation_alias`, `serialization_alias` | names for both, validation only, serialisation only |
| `gt`, `ge`, `lt`, `le`, `min_length`, `max_length`, `pattern`, `max_digits`, `decimal_places` | constraints (the page shows `gt`, `max_length`, `max_digits`, `decimal_places`) |
| `strict` | strict validation for this field ([[pydantic_strict_mode]]) |
| `frozen` | no reassignment after creation |
| `repr` | show in `repr` |
| `exclude`, `exclude_if` | leave out of dumps (`exclude_if` since v2.12) |
| `discriminator` | union discriminator |
| `deprecated` | message, `@deprecated` instance or `True`; warns on access |
| `title`, `description`, `examples`, `json_schema_extra` | JSON Schema only |
| `init`, `init_var`, `kw_only` | dataclass only |

## Examples

```python
class Model(BaseModel):
    positive: int = Field(gt=0)
    short_str: str = Field(max_length=3)
    precise_decimal: Decimal = Field(max_digits=5, decimal_places=2)


class User(BaseModel):
    email: EmailStr
    username: str = Field(default_factory=lambda data: data["email"])


class Model(BaseModel):
    name: Annotated[str, Field(strict=True), WithJsonSchema({"extra": "data"})]
    int_list: list[Annotated[int, Field(gt=0)]]
```

## computed_field

```python
class Box(BaseModel):
    width: float
    height: float
    depth: float

    @computed_field
    @property
    def volume(self) -> float:
        return self.width * self.height * self.depth
```

A computed field is included in `model_dump()` and in the serialisation JSON Schema (`model_json_schema(mode='serialization')`, marked `readOnly`). It does not validate its return value or invalidate caches. Pass the decorator explicitly for type checkers. `@cached_property` works too.

## Gotchas

- `= Field(...)` does not make a field required; having no default does. The ellipsis is discouraged.
- Metadata placement: `Annotated[int, Field(deprecated=True)] | None` applies to `int` only; put it on the union.
- Constraints on a union with `None` apply to the other members.
- Type checkers do not understand `default`, `default_factory` or `alias` inside `Annotated`; use the assignment form.
- `Optional` and `Any` no longer default to `None` (changed from V1). Defaults are not validated unless `validate_default` is set. Unhashable mutable defaults are deep-copied per instance.
- `model_fields` must be read from the class; deprecated in v2.11, removed in v3.
