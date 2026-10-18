---
type: api_reference
tags: [pydantic, errors]
source: https://pydantic.dev/docs/validation/latest/errors/validation_errors/
date_ingested: 2026-10-18
---
# pydantic_validation_errors

How validation failures are reported. The URL in the plan, `concepts/validation_errors/`, returned 404; this is the page `errors/validation_errors/` of the same docs.

## Reading error types

```python
try:
    Model(x='test')
except ValidationError as exc:
    print(repr(exc.errors()[0]['type']))
    #> 'int_parsing'
```

`exc.errors()` is a list of dicts; the key `'type'` holds the error code. The full key list (`loc`, `msg`, `input`, `ctx`, `url`) is shown in [[pydantic_get_started]], not on this page. Printed form:

```
2 validation errors for Model
a
  Input should be greater than 10 [type=greater_than, input_value=1, input_type=int]
```

## Errors raised in validators

There is no dedicated custom-error API on this page. A `ValueError` raised in a validator is reported as `value_error`; a failing `assert` as `assertion_error` (see [[pydantic_validators]]). `PydanticCustomError` is on the validators page.

## Notes

- The page lists each built-in error code with a triggering example (for instance `default_factory_not_called`).
- Error URLs have the form `https://errors.pydantic.dev/2/v/<type>`.
