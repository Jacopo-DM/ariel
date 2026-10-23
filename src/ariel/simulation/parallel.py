"""A process pool for scoring robots in parallel.

Every worker is a fresh Python process started with the ``spawn`` method,
on every platform, so Linux behaves as macOS does (spawn is the macOS
default). A spawned worker imports the function it runs by its module
name, so that function must live in a ``.py`` file, never in a notebook
cell. Each worker runs torch on one thread, since the pool already uses
one process per core.

Use one pool for many rounds, because starting it costs seconds (each
worker imports torch, MuJoCo and ARIEL)::

    with process_pool(workers=8) as pool:
        for _ in range(generations):
            candidates = [optimizer.ask() for _ in range(size)]
            scores = list(pool.map(score, [c.value for c in candidates]))
            for candidate, value in zip(candidates, scores, strict=True):
                optimizer.tell(candidate, value)

``pool.map`` returns the results in the order of the jobs.
"""

# Standard library
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

# Third-party libraries
import torch

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)


def use_one_thread() -> None:
    """Limit torch and the numeric libraries of this process to one thread.

    The pool runs it once in each worker before its first job. The
    variables reach only libraries that load after it; torch is set
    directly because it is already loaded.
    """
    for name in THREAD_VARIABLES:
        os.environ[name] = "1"
    torch.set_num_threads(1)


def process_pool(workers: int | None = None) -> ProcessPoolExecutor:
    """Start a pool of spawned worker processes, one thread each.

    Parameters
    ----------
    workers
        How many processes; ``None`` means one per logical CPU.

    Returns
    -------
    ProcessPoolExecutor
        The pool; use it in a ``with`` block so its workers stop after.
    """
    return ProcessPoolExecutor(
        max_workers=workers or os.cpu_count() or 1,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=use_one_thread,
    )
