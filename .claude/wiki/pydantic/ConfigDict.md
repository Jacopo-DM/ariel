---
type: api_reference
tags: [pydantic, config]
source: https://pydantic.dev/docs/validation/latest/concepts/config/
date_ingested: 2026-10-18
---
# ConfigDict

Model configuration. The page explains how to set it and shows a few options; the full list with defaults is in the `ConfigDict` API reference (`.../api/pydantic/config/`).

## Ways to set it

```python
class Model(BaseModel):
    model_config = ConfigDict(str_max_length=5)
    v: str


class Model2(
    BaseModel, frozen=True
):  # class arguments; type checkers see frozen
    a: str
```

A plain `dict` also works. The V1 `Config` class still works but is deprecated.

## Options shown

| Option | Effect |
|---|---|
| `str_max_length` | limit string length |
| `validate_assignment=True` | validate on assignment |
| `extra='allow'` | keep extra input (see [[BaseModel]]) |
| `str_to_lower=True` | lowercase strings |
| `strict=True` | strict mode ([[pydantic_strict_mode]]); on a stdlib dataclass via `__pydantic_config__` |
| `coerce_numbers_to_str=True` | numbers to strings (shown with `TypeAdapter`) |
| `plugin_settings` | options for plugins such as Logfire |
| `frozen=True` | no assignment |

## Inheritance

Subclass config merges with the parent's: parent `extra='allow'`, `str_to_lower=False` plus child `str_to_lower=True` gives `{'extra': 'allow', 'str_to_lower': True}`.

## Not on this page

`arbitrary_types_allowed`, `populate_by_name`, `str_strip_whitespace` and all defaults.
