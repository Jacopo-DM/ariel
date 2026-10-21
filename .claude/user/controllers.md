# Controllers — driving the joints

A controller turns the sim state into actuator commands (`data.ctrl`). Two
common routes: a **CPG** (central pattern generator, good for locomotion) or a
**neural network** (e.g. with vision). Both are wired in via
`mujoco.set_mjcb_control(...)` and the `Controller` glue object.

## CPG controllers

Seven CPGs live in `ariel.simulation.controllers.cpg`
([folder](../../src/ariel/simulation/controllers/cpg/)). All are `nn.Module`s
built from an adjacency dict, one oscillator per actuator, and all share the
interface of the base class `CPG`.

```python
from ariel.simulation import HopfCPG, create_fully_connected_adjacency

adj = create_fully_connected_adjacency(model.nu)  # nu = #actuators
cpg = HopfCPG(adj, seed=0)
action = cpg.forward(data.time)  # -> one target angle per joint
```

| Class | Model | Parameter groups |
|---|---|---|
| `HopfCPG` | Hopf oscillators with Kuramoto phase coupling (Righetti and Ijspeert 2006) | `phase`, `w`, `amplitudes`, `offset` |
| `AdaptiveHopfCPG` | the same, frequency learns from `set_input(values)` | `phase`, `w`, `amplitudes`, `offset` |
| `PhaseCPG` | phase oscillators with critically damped amplitudes (Ijspeert 2007, CPG-RL) | `phase`, `w`, `amplitudes`, `offset` |
| `SwingStanceCPG` | Hopf with one frequency per half cycle (Righetti and Ijspeert 2008) | `phase`, `w_swing`, `w_stance`, `amplitudes`, `offset` |
| `NaCPG` | normalized asymmetric CPG (Tong et al. 2022) | `phase`, `w`, `amplitudes`, `ha`, `b` |
| `TegotaeCPG` | phase oscillators steered by the load each limb feels, `set_input(loads)` (Owaki et al.; Herneth et al. 2023) | `phase`, `w`, `amplitudes`, `offset`, `sigma` |
| `LearnableTegotaeCPG` | the same, with the feedback as three small networks shared by all joints (Herneth et al. 2023) | `phase`, `w`, `amplitudes`, `offset`, plus `action`, `reaction`, `gain` weights (51 with `hidden=4`) |

- `forward(time)` follows the simulation clock: it integrates over the time
  since the previous timed call, in sub-steps of at most `dt` (0.01 s). The
  first timed call, or a time earlier than the last, resets the state.
  `forward()` with no time takes one step of `dt`.
- Parameters: `set_flat_params(vector)` (groups in the order of the table, `n`
  values each, except the network weights, sized by `group_sizes`), `set_params_by_group(name, values)`, `set_param_with_dict`,
  `get_flat_params()`, and `parameter_bounds()` for the range each parameter
  is sampled from. Set them before the run, or call `reset_state()` after.
- `angle_limits` (default plus or minus pi/2) clip the output; `clamping_error`
  is the mean overshoot. `angle_tracking=True` keeps `angle_history`.

`tests/functional/cpg_properties.ipynb` shows what each model can and cannot
do (for example, `SwingStanceCPG` only holds in-phase and anti-phase pairs);
`tests/functional/cpg_locomotion.ipynb` compares them on the gecko. For mapping
a genome vector onto the groups, see `map_genotype_to_brain` in
[examples/re_book/2_body_brain_evolution.py](../../examples/re_book/2_body_brain_evolution.py).

## The `Controller` glue + binding

```python
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.tracker import Tracker

tracker = Tracker(mujoco.mjtObj.mjOBJ_BODY, "core", ["xpos"])
ctrl = Controller(
    controller_callback_function=lambda m, d, *a, **k: cpg.forward(d.time),
    tracker=tracker,
)
ctrl.tracker.setup(world.spec, data)
mujoco.set_mjcb_control(lambda m, d: ctrl.set_control(m, d, duration=duration))
```

`Controller` (a `@dataclass`,
[controller.py](../../src/ariel/simulation/controllers/controller.py)) fields:
`controller_callback_function`, `time_steps_per_ctrl_step=50`,
`time_steps_per_save=500`, `alpha=0.5`, `tracker`. It throttles control vs.
save frequency, blends old/new control by `alpha`, and clips to servo bounds.

## Custom / neural controller (the callback pattern)

Any callable `(model, data, *args) -> array of length model.nu` works as the
controller. For a vision NN that reads camera frames + robot state each control
step, see the full `run_vision_simulation` / `Network` in
[examples/re_book/1_brain_evolution.py](../../examples/re_book/1_brain_evolution.py).
Robot proprioceptive state helper:
`from ariel.simulation.controllers.utils.data_get import get_state_from_data`.

## Where this fits

Body → [bodies.md](bodies.md); world/run loop → [simulation.md](simulation.md);
scoring the resulting motion → [tasks_fitness.md](tasks_fitness.md).
