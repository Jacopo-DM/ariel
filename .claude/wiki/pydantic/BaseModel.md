---
type: api_reference
tags: [pydantic, model, validation]
source: https://pydantic.dev/docs/validation/latest/concepts/models/
date_ingested: 2026-10-18
---
# BaseModel

Pydantic models subclass `BaseModel` and declare fields as annotated attributes. Fields with defaults are optional.

## Signature

```python
from pydantic import BaseModel, ConfigDict


class User(BaseModel):
    id: int
    name: str = "Jane Doe"

    model_config = ConfigDict(str_max_length=10)
```

## Methods

| Method | Behaviour |
|---|---|
| `Model(**kwargs)` | Python-mode validation, keyword arguments only; raises one `ValidationError` listing every error |
| `model_validate(obj)` | dict or model instance |
| `model_validate_json(data)` | JSON `str` or `bytes` |
| `model_validate_strings(obj)` | nested dict of strings, validated in JSON mode so strings coerce; accepts `strict=True` |
| `model_dump()` / `model_dump_json()` | dict / JSON string; `dict()` does not convert nested models recursively |
| `model_copy(update=..., deep=...)` | copy; `update` is NOT validated; shallow by default, `deep=True` for a deep copy |
| `model_construct()` | skips validation, does not call `__init__`; trusted data only; ignores extra data even with `extra='forbid'` |
| `model_fields_set` | fields explicitly provided at init |
| `model_computed_fields` | maps computed field names to `ComputedFieldInfo` (see [[Field]]) |

## Extra data (`extra`)

`'ignore'` (default) drops it, `'forbid'` raises, `'allow'` stores it in `__pydantic_extra__`. Validation methods take an `extra` argument that overrides the config for one call.

## Frozen models

`model_config = ConfigDict(frozen=True)` (or `class M(BaseModel, frozen=True)`) makes assignment raise a `ValidationError` (`frozen_instance`). Nested mutable objects (a dict field) can still change. Replaces V1 `allow_mutation = False`. Models are mutable by default.

## RootModel

```python
from pydantic import RootModel

Pets = RootModel[list[str]]
print(Pets.model_validate(["dog", "cat"]))
```

Serialisation returns the root value directly.

## Gotchas

- Conversion is lossy by design (`3.000` to `3` for an `int` field); use [[pydantic_strict_mode]].
- `model_copy(update=...)` skips validation: a changed seed in a frozen model is not re-checked unless the copy is validated again (for example `Model.model_validate(m.model_dump() | {...})`).
- Prefer `list[int]` to `Sequence[int]` (faster to validate).
- A custom `__init__` loses strictness and context; use `model_post_init()` or [[pydantic_validators]].
- Recursive models need forward annotations; cyclic input raises `recursion_loop`.
- Lists are copied during validation; an existing model instance passed in is used as is unless `revalidate_instances` is set.
- Avoid `isinstance(obj, Model[int])` on parametrised generics.
- A field named like its type (`int: int | None = None`) breaks validation.
- Config options: [[ConfigDict]].
