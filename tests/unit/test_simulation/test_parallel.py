"""Test: the spawned process pool keeps job order and uses one thread."""

import multiprocessing
import os

import pytest
import torch

from ariel.simulation import parallel, process_pool

JOBS = [3, 1, 2, 5, 4]


def square(value: int) -> int:
    """Return the square, in a worker process.

    Returns
    -------
    int
        ``value`` squared.
    """
    return value * value


def worker_threads(_job: int) -> tuple[int, str, int]:
    """Report the worker's torch threads, a thread variable and its pid.

    Returns
    -------
    tuple[int, str, int]
        torch's thread count, ``OMP_NUM_THREADS`` and the process id.
    """
    return torch.get_num_threads(), os.environ["OMP_NUM_THREADS"], os.getpid()


def test_the_pool_returns_results_in_job_order_from_other_processes() -> None:
    """Results follow the jobs, and no job runs in this process."""
    with process_pool(workers=2) as pool:
        assert list(pool.map(square, JOBS)) == [9, 1, 4, 25, 16]
        reports = list(pool.map(worker_threads, JOBS))
    assert {threads for threads, _, _ in reports} == {1}
    assert {omp for _, omp, _ in reports} == {"1"}
    assert os.getpid() not in {pid for _, _, pid in reports}


CPUS = 3


class _Recorder:
    """Stands in for ProcessPoolExecutor and keeps its arguments."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, object, object]] = []

    def __call__(
        self, max_workers: int, mp_context: object, initializer: object
    ) -> None:
        self.calls.append((max_workers, mp_context, initializer))


def test_the_pool_spawns_and_defaults_to_one_worker_per_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The spawn context everywhere, and os.cpu_count() workers."""
    recorder = _Recorder()
    monkeypatch.setattr(parallel, "ProcessPoolExecutor", recorder)
    monkeypatch.setattr(os, "cpu_count", lambda: CPUS)
    process_pool()
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    process_pool()
    process_pool(workers=2)
    spawn = multiprocessing.get_context("spawn")
    assert recorder.calls == [
        (CPUS, spawn, parallel.use_one_thread),
        (1, spawn, parallel.use_one_thread),
        (2, spawn, parallel.use_one_thread),
    ]


def test_use_one_thread_sets_torch_and_the_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker set-up, run in this process and undone after."""
    for name in parallel.THREAD_VARIABLES:
        monkeypatch.setenv(name, "4")
    before = torch.get_num_threads()
    try:
        parallel.use_one_thread()
        assert worker_threads(0) == (1, "1", os.getpid())
        assert {os.environ[name] for name in parallel.THREAD_VARIABLES} == {"1"}
        assert square(JOBS[0]) == JOBS[0] ** 2
    finally:
        torch.set_num_threads(before)
