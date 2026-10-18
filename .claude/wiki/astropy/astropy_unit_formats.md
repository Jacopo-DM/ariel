---
type: api_reference
tags: [astropy, units, formatting, parsing]
source: https://docs.astropy.org/en/stable/units/format.html
date_ingested: 2026-10-18
---
# astropy_unit_formats

Converting units and quantities to and from strings.

## Formats

| Format | Use |
|---|---|
| `generic` (default) | FITS-based syntax, all units including user-defined |
| `fits` | only units defined in the FITS standard |
| `vounit` | VO 1.0 standard |
| `cds` | Vizier tables, VOTable 1.3 and earlier |
| `ogip` | OGIP recommendation |
| `latex`, `latex_inline` | LaTeX; inline uses negative powers |
| `console`, `unicode` | plain text; `fraction="inline"` or `"multiline"`; not parseable |

## Examples

```python
>>> u.Unit("erg.s-1.cm-2", format="cds")
Unit("erg / (s cm2)")
>>> u.Quantity("3m/s")
<Quantity 3. m / s>
>>> q = 10.5 * u.km
>>> f"{q:+.3f}"
'+10.500 km'
>>> f"{q.value:.3f} in {q.unit}"
'10.500 in km'
>>> q.to_string(format="latex", precision=4)
```

## Rules

- Built-in format names (`latex`) apply to the unit and, where possible, the value. Numeric specifiers (`.3f`) apply to the value only. String specifiers (`^20s`) apply to the whole string.
- `ndarray` and non-scalar quantities do not take numeric specifiers; use `numpy.array_str()`.
- `deprecations` argument of `to_string()`: `"raise"`, `"convert"`, `"silent"`, default warns.
- `parse_strict` argument of `Unit`: `'raise'` (default), `'warn'`, `'silent'`. Aliases: `set_enabled_aliases()` and `add_enabled_aliases()`. An `UnrecognizedUnit` keeps its string but cannot be converted.
- Parsing strings is slower than building units from objects such as `u.degree / u.minute`.
- Related: [[Quantity]], [[Unit]].
