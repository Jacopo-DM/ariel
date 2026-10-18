---
type: api_reference
tags: [astropy, units, equivalencies]
source: https://docs.astropy.org/en/stable/units/equivalencies.html
date_ingested: 2026-10-18
---
# astropy_unit_equivalencies

Equivalencies let units that are normally not convertible convert when a physical relationship links them. Pass a list through the `equivalencies` keyword of `Quantity.to()`, `Unit.to()` or `Unit.get_converter()`.

## Examples

```python
>>> (0.8 * u.arcsec).to(u.parsec, equivalencies=u.parallax())
<Quantity 1.25 pc>
>>> (1000 * u.nm).to(u.Hz, equivalencies=u.spectral())
<Quantity  2.99792458e+14 Hz>
>>> with u.set_enabled_equivalencies(u.dimensionless_angles()):
...     phase = 0.5 * u.cycle
...     c = np.exp(1j*phase)
```

Without the context manager, `set_enabled_equivalencies()` stays on until reset (an empty argument turns it off); the page warns this is easy to forget. Lists can be added together, for example `spectral()` plus `spectral_density()`.

## Built-in equivalencies

| Function | Purpose |
|---|---|
| `parallax()` | parallax angle and distance |
| `dimensionless_angles()` | angles to dimensionless units; radians are not cycles, rad/s to Hz needs an explicit equivalency |
| `spectral()` | wavelength, frequency, energy, wave number |
| `doppler_optical()`, `doppler_radio()`, `doppler_relativistic()` | frequency and velocity |
| `spectral_density()` | spectral flux and luminosity density |
| `brightness_temperature()`, `beam_angular_area()`, `thermodynamic_temperature()` | radio brightness units |
| `temperature_energy()` | temperature to energy (eV) |
| `temperature()` | Kelvin, Celsius, Fahrenheit (different zero points) |
| `molar_mass_amu()` | amu and g/mol |
| `pixel_scale()`, `plate_scale()` | angular to pixel and focal-plane scales |
| `zero_point_flux()` | photometric zero points |
| `mass_energy()` | mass and energy |
| `doppler_redshift()` | redshift and radial velocity |
| `magnetic_flux_field()` | flux density B and field strength H |

## Notes

- Incompatible conversion without an equivalency raises `UnitConversionError` (see [[Quantity]]).
- For ARIEL: `dimensionless_angles()` is the one that matters for radians (see [[PhysicalType]] and [[Quantity]]).
