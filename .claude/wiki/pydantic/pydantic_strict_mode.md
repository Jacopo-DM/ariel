---
type: api_reference
tags: [pydantic, strict]
source: https://pydantic.dev/docs/validation/latest/concepts/strict_mode/
date_ingested: 2026-10-18
---
# pydantic_strict_mode

By default Pydantic coerces (`'123'` becomes `123`). Strict mode raises when the input is not the right type; mostly only instances of the type pass. JSON input is more lenient, and date and time types still accept strings. Code that relied on coercion starts failing; the page suggests watching validation failures during rollout.

## Ways to enable

```python
TypeAdapter(date).validate_python('2000-01-01', strict=True)   # per call: raises
TypeAdapter(date).validate_json('"2000-01-01"', strict=True)   # JSON: ok

class User(BaseModel):
    name: str
    age: int = Field(strict=True)            # per field

class User2(BaseModel):
    id: Annotated[UUID, Strict()]            # Strict() metadata
    age: StrictInt                           # also StrictBool, StrictFloat, StrictStr, StrictBytes

class User3(BaseModel):
    model_config = ConfigDict(strict=True)   # per model
    name: str
    age: int = Field(strict=False)           # override one field
```

## Notes

- The lax-versus-strict table per type is on a separate conversion table page (not ingested).
- Related: [[BaseModel]], [[Field]], [[ConfigDict]].
