# Parallelising Robot Simulations

Evaluating a population of robots is the main bottleneck in evolutionary robotics: every individual needs to be simulated before selection can proceed, and the simulations are independent of each other.
ARIEL runs them in parallel on one machine with `ariel.simulation.process_pool`, a `concurrent.futures.ProcessPoolExecutor` set up for MuJoCo.

---

## How it works

- `process_pool(workers)` starts worker processes with the `spawn` method on every platform, so Linux, macOS and Windows behave the same, and each worker runs torch on one thread.
- The function a worker runs must live in a `.py` file: a spawned worker imports it by its module name and cannot see functions defined in a notebook cell.
- `pool.map(function, jobs)` returns the results in the order of the jobs. With a deterministic fitness function the results equal a serial run's.
- Open one pool around the whole run: starting it takes a few seconds, because each worker imports torch, MuJoCo and ARIEL.

```python
from ariel.simulation import process_pool
from my_experiment import evaluate  # a function in a .py file

with process_pool(workers=8) as pool:
    for generation in range(GENERATIONS):
        candidates = [optimizer.ask() for _ in range(POPULATION)]
        scores = list(pool.map(evaluate, [c.value for c in candidates]))
        for candidate, score in zip(candidates, scores, strict=True):
            optimizer.tell(candidate, score)
```

Telling a CMA-ES optimizer its scores in the order it asked gives the same search as asking and telling one candidate at a time.

---

## Pages

```{toctree}
:maxdepth: 2

process_pool
```
