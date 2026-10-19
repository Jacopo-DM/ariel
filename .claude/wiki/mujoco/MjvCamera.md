---
type: api_reference
tags: [mujoco, python, class, rendering, visualization]
source: https://mujoco.readthedocs.io/en/stable/python.html
date_ingested: 2026-04-13
---

# MjvCamera

Abstract visualizer camera. Used with [[mujoco_rendering_pipeline]] to define the viewpoint.

## Signature

```python
cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)  # initialize to defaults
```

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `type` | `mjtCamera` int | camera mode (FREE=0, TRACKING=1, FIXED=2, USER=3) |
| `fixedcamid` | int | model camera id (used when `type=FIXED`) |
| `trackbodyid` | int | body id to track (used when `type=TRACKING`) |
| `lookat` | `(3,)` numpy | 3D point being viewed |
| `distance` | float | distance from camera to lookat point |
| `azimuth` | float | horizontal angle in degrees |
| `elevation` | float | vertical angle in degrees |

## Camera Types (`mjtCamera`)

| Value | Name | Description |
|-------|------|-------------|
| 0 | `mjCAMERA_FREE` | free-floating camera controlled by azimuth/elevation/distance |
| 1 | `mjCAMERA_TRACKING` | follows a body (set `trackbodyid`) |
| 2 | `mjCAMERA_FIXED` | uses a named model camera (set `fixedcamid`) |
| 3 | `mjCAMERA_USER` | user-controlled |

## Example

```python
cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)

cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:] = [0.0, 0.0, 0.5]
cam.distance = 3.0
cam.azimuth = 90.0
cam.elevation = -30.0
```

## Related Functions

```python
mujoco.mjv_defaultCamera(cam)  # initialize to defaults
mujoco.mjv_defaultFreeCamera(
    model, cam
)  # initialize free camera for model extent
mujoco.mjv_moveCamera(
    model, action, reldx, reldy, scene, cam
)  # interactive move
```

See [[MjvOption]], [[MjvScene]], [[mujoco_rendering_pipeline]].

## From: Visualization page, 2026-10-19 refresh

Source: https://mujoco.readthedocs.io/en/stable/programming/visualization.html, fetched 2026-10-19.

* There are two camera types: the abstract `mjvCamera` (stand-alone struct) and the low-level `mjvGLCamera` (aliased `mjrCamera`), embedded in `mjvScene` or computed directly with `mjv_camera2GLCamera`. When an abstract camera is present, it sets the low-level parameters that the renderer uses; the low-level camera can also be set directly.
* `mjvScene` embeds two `mjvGLCamera`, one per eye. Each has a position, a forward direction (the negative Z axis of the camera frame) and an up direction (the positive Y axis). The frustum stores the average of the left and right edges; the real edges are computed at render time from the viewport aspect ratio, assuming square pixels. The distance between the two positions is the inter-pupillary distance (ipd).
* For free and tracking cameras the ipd and `fovy` come from `mjModel.vis.global.ipd/fovy`; for cameras defined in the model they come from `mjModel.cam_ipd/fovy`.
* `mjv_select` calls the engine function `mj_ray` internally.
