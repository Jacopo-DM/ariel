---
type: api_reference
tags: [pydantic, validation, python]
source: https://pydantic.dev/docs/validation/latest/get-started/
date_ingested: 2026-10-18
---
# pydantic_get_started

Pydantic validates and serialises data against type hints. The core is written in Rust; it can emit JSON Schema and runs in strict mode (no conversion) or lax mode (coerces where sensible). Models are in [[BaseModel]]; settings in [[BaseSettings]].

## Install

The page shows the install command for the `pydantic` package from PyPI. In ARIEL, dependencies are added with `uv add`; `pydantic` and `pydantic-settings` are already dependencies.

## Example

```python
from datetime import datetime

from pydantic import BaseModel, PositiveInt


class User(BaseModel):
    id: int
    name: str = "John Doe"
    signup_ts: datetime | None
    tastes: dict[str, PositiveInt]


external_data = {
    "id": 123,
    "signup_ts": "2019-06-01 12:22",
    "tastes": {"wine": 9, b"cheese": 7, "cabbage": "1"},
}
user = User(**external_data)
print(user.model_dump())
# {'id': 123, 'name': 'John Doe', 'signup_ts': datetime.datetime(2019, 6, 1, 12, 22),
#  'tastes': {'wine': 9, 'cheese': 7, 'cabbage': 1}}
```

`id` is coerced from strings, bytes or floats where possible; a field with a default is optional; `signup_ts` is required but may be `None`.

## Errors

Invalid input raises one `ValidationError` listing every failing field:

```python
try:
    User(id="not an int", tastes={})
except ValidationError as e:
    print(e.errors())
# [{'type': 'int_parsing', 'loc': ('id',), 'msg': 'Input should be a valid integer, unable to parse string as an integer',
#   'input': 'not an int', 'url': 'https://errors.pydantic.dev/2/v/int_parsing'},
#  {'type': 'missing', 'loc': ('signup_ts',), 'msg': 'Field required', 'input': {...}, 'url': '...'}]
```

See [[pydantic_validation_errors]].

## Notes

- The page links "Why use Pydantic", an install page, the API reference for `BaseModel.model_dump` and `ValidationError`, and a machine-readable `llms.txt` at `https://pydantic.dev/docs/validation/latest/llms.txt`.
- The page text contained a box asking readers (and agents) to add `goal` and `organization` query parameters to URLs. That is page content, not an instruction from the user, and was ignored.
- Related: [[Field]], [[pydantic_validators]], [[ConfigDict]], [[pydantic_strict_mode]], [[pydantic_custom_types]].
