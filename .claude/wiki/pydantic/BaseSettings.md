---
type: api_reference
tags: [pydantic, pydantic-settings, settings, environment]
source: https://pydantic.dev/docs/validation/latest/concepts/pydantic_settings/
date_ingested: 2026-10-18
---
# BaseSettings

`pydantic_settings.BaseSettings` reads fields from the environment and other sources. The package is `pydantic-settings` (already an ARIEL dependency). Unlike [[BaseModel]], it validates default values (`validate_default=False` turns that off). `model_validate()` and friends still load the sources, but `extra`, `strict` and `context` arguments are not applied.

## Basic use

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="my_prefix_")
    auth_key: str = "xxx"  # read from my_prefix_auth_key
```

## SettingsConfigDict options

| Option | Purpose |
|---|---|
| `env_prefix` (default `''`) | prefix for names; also for dotenv and secrets |
| `env_prefix_target` | `'variable'` (default), `'alias'`, `'all'` |
| `case_sensitive` | default off; applies to nested models |
| `env_nested_delimiter`, `env_nested_max_split` | split names into nested models or dicts |
| `env_ignore_empty` | ignore empty variables, use defaults |
| `env_file`, `env_file_encoding`, `env_file_depth` | dotenv |
| `dotenv_filtering` | `'match_prefix'` or `'only_existing'` |
| `extra` | default `'forbid'` (also for dotenv entries) |
| `secrets_dir` | directory of secret files |
| `populate_by_name`, `validate_by_name` | accept field names as well as aliases |
| `enable_decoding` | `False` stops JSON decoding of complex fields |
| `json_file`, `toml_file`, `yaml_file` (+ encodings, `toml_table_header`) | config file sources |
| `cli_parse_args` and other `cli_*` | command line |

Options can also be class arguments: `class Settings(BaseSettings, case_sensitive=True):`.

## Names

- Default: the variable name is the field name. `Field(validation_alias=...)`, `Field(alias=...)` and `AliasChoices('a', 'b')` (first found wins) change it.
- Variables matching no field or alias are ignored, even with `extra='forbid'`: a typo leaves the default in place.
- Complex types (`list`, `set`, `dict`, sub-models) are read as JSON; invalid JSON raises `SettingsError`; `NoDecode` lets you parse yourself.
- Nested models must be `BaseModel`; with `env_nested_delimiter='__'`: `SUB_MODEL__V2=nested-2`. Nested variables beat the top-level JSON.

## Priority (highest first)

CLI arguments (if `cli_parse_args`), initialiser arguments, environment variables, dotenv, secrets directory, field defaults. In `settings_customise_sources` the default order is `init_settings`, `env_settings`, `dotenv_settings`, `file_secret_settings`. Environment variables always beat dotenv values.

```python
@classmethod
def settings_customise_sources(
    cls,
    settings_cls: type[BaseSettings],
    init_settings: PydanticBaseSettingsSource,
    env_settings: PydanticBaseSettingsSource,
    dotenv_settings: PydanticBaseSettingsSource,
    file_secret_settings: PydanticBaseSettingsSource,
) -> tuple[PydanticBaseSettingsSource, ...]:
    return env_settings, CliSettingsSource(settings_cls, cli_parse_args=True)
```

## Files and other sources

- `JsonConfigSettingsSource(json_file=...)`, `TomlConfigSettingsSource(toml_file=..., toml_table_header=(...))`, `YamlConfigSettingsSource(yaml_file=...)`; paths are `str`, `Path` or `Traversable`. A list of TOML files is merged shallowly, later wins; `deep_merge=True` on the source enables deep merging (not available through `SettingsConfigDict`).
- `PyprojectTomlConfigSettingsSource`: default table `('tool', 'pydantic-settings')`, `pyproject_toml_depth` to search parents.
- Dotenv: `env_file=('.env', '.env.prod')` later wins; `_env_file=None` disables loading; unknown entries raise under `extra='forbid'`.
- Secrets: file name is the key, content the value; environment and dotenv beat it. `NestedSecretsSettingsSource` is opt-in. Cloud sources (AWS Secrets Manager, AWS Parameter Store, Azure Key Vault, Google Secret Manager with `SecretVersion`) are added through `settings_customise_sources`.
- CLI: `cli_parse_args=True`, `CliSubCommand`, `CliPositionalArg`, `CliApp.run`; lists accept JSON, repeated options or commas.

## Notes

- Debugging: `PYDANTIC_SETTINGS_DEBUG=1` logs each source's values (can contain secrets).
- Caching: `settings_cached()` returns one shared instance per class; `settings_clear_cache()` clears only the exact class. Fields named `settings_cached*` or `settings_clear_cache*` raise a `UserWarning`.
- Async: load in a thread, `asyncio.to_thread(Settings)`.
- The page has no section on overriding settings in tests; it is long (126608 characters) and was read in two parts.
- Related: [[ConfigDict]], [[Field]], [[pydantic_validators]].
