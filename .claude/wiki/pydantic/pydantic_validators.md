---
type: api_reference
tags: [pydantic, validators]
source: https://pydantic.dev/docs/validation/latest/concepts/validators/
date_ingested: 2026-10-18
---
# pydantic_validators

Custom validation for fields and whole models. Import from `pydantic` (`field_validator`, `model_validator`) and `pydantic.functional_validators`.

## Field validator kinds

| Mode | Runs | Input |
|---|---|---|
| after (default) | after Pydantic's validation | typed value |
| before | before parsing and coercion | raw input of any type |
| plain | instead of Pydantic's validation; ends validation, no type check | any |
| wrap | receives a handler; can run code before or after, return early, or raise | raw |

## Signatures

```python
@field_validator("number", mode="after")
@classmethod
def is_even(cls, value: int) -> int: ...


@field_validator("my_string", mode="wrap")
@classmethod
def truncate(cls, value: Any, handler: ValidatorFunctionWrapHandler) -> str: ...


@model_validator(mode="after")
def check_passwords_match(self) -> Self: ...


@model_validator(mode="before")
@classmethod
def check_card_number_not_present(cls, data: Any) -> Any: ...
```

`'*'` as the field name applies to all fields; `check_fields=False` skips the existence check. Model validators: after (instance method returning `Self`), before (classmethod, raw input), wrap (`handler: ModelWrapValidatorHandler[Self]`).

## Annotated pattern and order

`Annotated[int, AfterValidator(is_even)]` is reusable and works on parts of a type (list items). Before and wrap validators run right to left; after validators run left to right; decorator validators come after the annotated metadata.

## Raising errors

`ValueError` (most common), `AssertionError` (skipped under `python -O`), or `PydanticCustomError(error_type, message_template, context)`. See [[pydantic_validation_errors]].

## ValidationInfo

An optional argument exposing `data` (earlier fields only, `None` in model validators), `context` (from `model_validate(..., context=...)`, not available through `Model(...)`), `mode` (`'python'`, `'json'`, `'strings'`) and `field_name`.

## Gotchas

- Always return the validated value.
- Plain validators accept any input (`'invalid'` passes an `int` field).
- Defaults are not validated unless configured, so validators do not run on them.
- A subclass model validator with the same name replaces the base one.
- Helpers: `InstanceOf`, `SkipValidation`, `ValidateAs`, `PydanticUseDefault`.
- The page does not cover validating assignment; the option is `validate_assignment` in [[ConfigDict]].
- Related: [[BaseModel]], [[Field]].
