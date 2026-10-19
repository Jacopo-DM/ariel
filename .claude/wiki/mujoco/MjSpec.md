---
type: api_reference
tags: [mujoco, python, class, model-building]
source: https://mujoco.readthedocs.io/en/stable/python.html
date_ingested: 2026-04-13
---

# MjSpec

Programmatic model builder. Allows constructing and editing MuJoCo models in Python without writing XML. Compiles to [[MjModel]].

## Signature

```python
spec = mujoco.MjSpec()  # empty spec
spec = mujoco.MjSpec.from_file("/path/to.xml")  # parse existing XML
spec = mujoco.MjSpec.from_string(xml_str)  # parse XML string
```

## Compilation Methods

```python
model = spec.compile()
# returns MjModel, or None on error (check spec.error)

model, data = spec.recompile(old_model, old_data)
# in-place update preserving simulation state; returns new objects

xml_str = spec.to_xml()
# serialize back to XML string
```

**Error handling:**
```python
model = spec.compile()
if model is None:
    print(spec.error)
```

## Body Tree

```python
spec.worldbody  # root body (MjsBody)

# Adding children to a body:
body = spec.worldbody.add_body()
geom = body.add_geom()
joint = body.add_joint()
site = body.add_site()
camera = body.add_camera()
light = body.add_light()
frame = body.add_frame()

# Free joint shorthand:
joint = body.add_freejoint()
```

## Top-Level Element Addition

```python
actuator = spec.add_actuator()
sensor = spec.add_sensor()
tendon = spec.add_tendon()
material = spec.add_material()
mesh = spec.add_mesh()
texture = spec.add_texture()
key = spec.add_key()
```

## Named Access

```python
spec.geom("my_geom")  # returns element or None
spec.body("my_body")
```

## List Properties

All element collections are accessible as plural properties:

`spec.bodies`, `spec.geoms`, `spec.joints`, `spec.sites`, `spec.cameras`, `spec.lights`, `spec.frames`, `spec.materials`, `spec.meshes`, `spec.tendons`, `spec.actuators`, `spec.sensors`, `spec.textures`, `spec.texts`, `spec.tuples`, `spec.keys`, `spec.numerics`, `spec.excludes`, `spec.pairs`, `spec.equalities`, `spec.plugins`, `spec.skins`, `spec.flexes`, `spec.hfields`

## Body Subtree Methods

```python
body.geoms  # direct children geoms
body.find_all("site")  # or mjtObj.mjOBJ_SITE

spec.delete(spec.geom("my_geom"))  # remove element and all referencing elements
```

## Setting Options

```python
spec.option.timestep = 0.002
spec.option.gravity = [0, 0, -9.81]
```

## Example

```python
import mujoco

spec = mujoco.MjSpec()
spec.option.timestep = 0.002

body = spec.worldbody.add_body()
body.name = "box"
body.pos = [0, 0, 1]

geom = body.add_geom()
geom.type = mujoco.mjtGeom.mjGEOM_BOX
geom.size = [0.1, 0.1, 0.1]
geom.mass = 1.0

joint = body.add_freejoint()

model = spec.compile()
data = mujoco.MjData(model)
```

## Notes

- ⚠️ **`geom.size` always requires exactly 3 elements**, regardless of geom type. The binding enforces shape `[3, 1]`. Always pad unused dimensions with `0`:
  ```python
  # capsule: [radius, half_length, 0]
  geom.size = [0.02, 0.4, 0]
  # sphere: [radius, 0, 0]
  geom.size = [0.08, 0, 0]
  # box: [x, y, z]
  geom.size = [0.1, 0.1, 0.1]
  # plane: [half_x, half_y, 0]
  geom.size = [5, 5, 0]
  ```

- Remove callbacks before recompiling to avoid dangling references:
  ```python
  mujoco.set_mjcb_control(None)
  model = spec.compile()
  mujoco.set_mjcb_control(my_controller)
  ```
- See [[mujoco_callbacks]] for callback management.

## From: Python bindings page, 2026-10-19 refresh

Source: https://mujoco.readthedocs.io/en/stable/python.html, fetched 2026-10-19 (MuJoCo 3.16.0); the calls below exist in the installed 3.16.0 bindings.

* Virtual file system: `vfs = mujoco.MjVfs()`, then `vfs["model.xml"] = some_xml_string.encode("utf-8")`, `del vfs["name"]`, `"name" in vfs`, and `vfs.close()` when done. `MjModel.from_xml_string`, `MjModel.from_xml_path`, `MjSpec.from_string`, `MjSpec.from_file` and `spec.compile()` take an optional `vfs` argument:

```python
vfs = mujoco.MjVfs()
vfs["model.xml"] = some_xml_string.encode("utf-8")
spec = mujoco.MjSpec.from_file("model.xml", vfs=vfs)
spec.compile(vfs=vfs)
vfs.close()
```

* `from_string()` and `from_file()` can only be called at construction time.
* `spec.encode(...)` takes the target filename, an optional compiled model, an optional `vfs` and an optional `content_type`; the format follows the extension (`.xml`, `.mjb`, `.txt`, `.mjz`) or the content type.
* `spec.copy_back(model)` writes values changed in a compiled model back into the spec, so later compiles and saves include them:

```python
model = spec.compile()
model.geom('my_geom').size[0] = 2
spec.copy_back(model)
print(spec.geom('my_geom').size[0])  # 2.0
```

* Serialization with assets: `spec.to_zip(file)` and `MjSpec.from_zip(file)`; `file` is a path or a file object.
* Any element, bodies and frames included, has a `parent` property (for example `site.parent`).
