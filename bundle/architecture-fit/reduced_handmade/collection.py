"""Deterministic episode collection backends for reduced-MDP training.

Simulation scheduling is separated from policy updates.  The built-in process
backend runs independent SysML episodes on CPU workers.  A backend that batches
policy inference on another device can implement the same ``EpisodeCollector``
interface without changing checkpoint selection or the training loop.
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
from abc import ABC, abstractmethod
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from oracle import extract_interface

from reduced_handmade.buffered_env import BufferedDiscreteEnv
from reduced_handmade.composite import ProgramShieldComposite
from reduced_handmade.episode import (
    Episode,
    RolloutSummary,
    collect_episode,
    measure_episodes,
    merge_measurements,
    summarize_measurements,
)
from reduced_handmade.oracle_data import collect_oracle_episode
from reduced_handmade.policy import MLPActorCritic


SEED_SCHEME = "numpy_seedsequence_v1_per_episode"


@dataclass(frozen=True)
class CollectionSettings:
    backend: str = "serial"
    workers: int = 1
    start_method: str = "spawn"

    def resolved(self) -> "CollectionSettings":
        workers = max(1, int(self.workers))
        backend = self.backend
        if backend == "auto":
            backend = "process" if workers > 1 else "serial"
        if backend not in {"serial", "process"}:
            raise ValueError(f"unsupported collection backend: {backend}")
        if backend == "serial":
            workers = 1
        if self.start_method not in mp.get_all_start_methods():
            raise ValueError(
                f"unsupported multiprocessing start method: {self.start_method}"
            )
        return CollectionSettings(
            backend=backend,
            workers=workers,
            start_method=self.start_method,
        )

    def to_dict(self) -> dict:
        out = asdict(self.resolved())
        out["seed_scheme"] = SEED_SCHEME
        return out


@dataclass(frozen=True)
class DiscreteRuntime:
    model_path: str
    dt: float
    max_steps: int
    phase: int
    n_obs: int
    n_act: int
    obs_dim: int
    n_actions: int
    hidden_dim: int


@dataclass(frozen=True)
class EpisodeJob:
    index: int
    env_seed: int
    action_seed: int


@dataclass(frozen=True)
class OracleEpisodeJob:
    index: int
    env_seed: int
    include_terminal: bool


def _derived_seed(base_seed: int, domain: int, episode_index: int) -> int:
    words = [
        int(base_seed) & 0xFFFFFFFF,
        int(domain) & 0xFFFFFFFF,
        int(episode_index) & 0xFFFFFFFF,
        (int(episode_index) >> 32) & 0xFFFFFFFF,
    ]
    return int(np.random.SeedSequence(words).generate_state(1, dtype=np.uint32)[0])


def make_episode_jobs(count: int, *, seed_base: int,
                      start_index: int = 0) -> list[EpisodeJob]:
    return [
        EpisodeJob(
            index=index,
            env_seed=_derived_seed(seed_base, 1, index),
            action_seed=_derived_seed(seed_base, 2, index),
        )
        for index in range(start_index, start_index + count)
    ]


def _snapshot_policy(policy) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(value).copy()
        for name, value in policy.parameters().items()
    }


def _load_policy_snapshot(policy, snapshot: dict[str, np.ndarray]) -> None:
    parameters = policy.parameters()
    if set(parameters) != set(snapshot):
        raise RuntimeError("policy snapshot keys do not match worker policy")
    for name, target in parameters.items():
        source = np.asarray(snapshot[name])
        if target.shape != source.shape:
            raise RuntimeError(
                f"policy snapshot shape mismatch for {name}: "
                f"{source.shape} != {target.shape}"
            )
        target[...] = source


def _make_env(runtime: DiscreteRuntime) -> BufferedDiscreteEnv:
    return BufferedDiscreteEnv(
        runtime.model_path,
        dt=runtime.dt,
        max_steps=runtime.max_steps,
        phase=runtime.phase,
        rng_seed=0,
        n_obs=runtime.n_obs,
        n_act=runtime.n_act,
    )


def _partition(items: Sequence, parts: int) -> list[list]:
    if not items:
        return []
    count = min(max(1, parts), len(items))
    return [list(items[offset::count]) for offset in range(count)]


class EpisodeCollector(ABC):
    """Execution interface used by the device-independent training loop."""

    settings: CollectionSettings

    @abstractmethod
    def collect(self, policy, jobs: Sequence[EpisodeJob], *,
                greedy: bool = False) -> list[Episode]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, policy, jobs: Sequence[EpisodeJob]) -> RolloutSummary:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    def __enter__(self) -> "EpisodeCollector":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class SerialEpisodeCollector(EpisodeCollector):
    def __init__(self, runtime: DiscreteRuntime,
                 composite: ProgramShieldComposite):
        self.settings = CollectionSettings().resolved()
        self._env = _make_env(runtime)
        self._composite = composite
        self._closed = False

    def _episode(self, job: EpisodeJob, *, greedy: bool) -> Episode:
        return collect_episode(
            self._env,
            self._composite,
            rng=np.random.default_rng(job.action_seed),
            greedy=greedy,
            reset_seed=job.env_seed,
        )

    def collect(self, policy, jobs: Sequence[EpisodeJob], *,
                greedy: bool = False) -> list[Episode]:
        if self._closed:
            raise RuntimeError("episode collector is closed")
        if self._composite.policy is not policy:
            self._composite.policy = policy
        return [self._episode(job, greedy=greedy).compact() for job in jobs]

    def evaluate(self, policy, jobs: Sequence[EpisodeJob]) -> RolloutSummary:
        if self._closed:
            raise RuntimeError("episode collector is closed")
        if self._composite.policy is not policy:
            self._composite.policy = policy
        measurements = measure_episodes(
            self._episode(job, greedy=True) for job in jobs
        )
        return summarize_measurements(measurements)

    def close(self) -> None:
        if not self._closed:
            self._env.close()
            self._closed = True


class _PolicyWorkerContext:
    def __init__(self, runtime: DiscreteRuntime):
        self.env = _make_env(runtime)
        if self.env.obs_dim != runtime.obs_dim:
            raise RuntimeError(
                f"worker observation dimension changed: "
                f"{self.env.obs_dim} != {runtime.obs_dim}"
            )
        if self.env.n_actions != runtime.n_actions:
            raise RuntimeError(
                f"worker action count changed: "
                f"{self.env.n_actions} != {runtime.n_actions}"
            )
        iface = extract_interface(runtime.model_path)
        self.policy = MLPActorCritic(
            runtime.obs_dim,
            runtime.n_actions,
            runtime.hidden_dim,
            seed=0,
        )
        self.composite = ProgramShieldComposite(
            self.policy, iface["spec_shield"], iface["obs_names"]
        )

    def close(self) -> None:
        self.env.close()


_POLICY_WORKER: _PolicyWorkerContext | None = None


def _init_policy_worker(runtime: DiscreteRuntime) -> None:
    global _POLICY_WORKER
    _POLICY_WORKER = _PolicyWorkerContext(runtime)
    atexit.register(_POLICY_WORKER.close)


def _worker_collect(payload):
    snapshot, jobs, greedy, measurements_only = payload
    if _POLICY_WORKER is None:
        raise RuntimeError("policy worker was not initialized")
    _load_policy_snapshot(_POLICY_WORKER.policy, snapshot)
    episodes = [
        collect_episode(
            _POLICY_WORKER.env,
            _POLICY_WORKER.composite,
            rng=np.random.default_rng(job.action_seed),
            greedy=greedy,
            reset_seed=job.env_seed,
        )
        for job in jobs
    ]
    if measurements_only:
        return measure_episodes(episodes)
    return [(job.index, episode.compact())
            for job, episode in zip(jobs, episodes)]


class ProcessEpisodeCollector(EpisodeCollector):
    def __init__(self, runtime: DiscreteRuntime,
                 settings: CollectionSettings):
        self.settings = settings.resolved()
        if self.settings.backend != "process":
            raise ValueError("process collector requires process settings")
        context = mp.get_context(self.settings.start_method)
        self._executor = ProcessPoolExecutor(
            max_workers=self.settings.workers,
            mp_context=context,
            initializer=_init_policy_worker,
            initargs=(runtime,),
        )
        self._closed = False

    def _payloads(self, policy, jobs: Sequence[EpisodeJob], *, greedy: bool,
                  measurements_only: bool):
        snapshot = _snapshot_policy(policy)
        return [
            (snapshot, chunk, greedy, measurements_only)
            for chunk in _partition(jobs, self.settings.workers)
        ]

    def collect(self, policy, jobs: Sequence[EpisodeJob], *,
                greedy: bool = False) -> list[Episode]:
        if self._closed:
            raise RuntimeError("episode collector is closed")
        payloads = self._payloads(
            policy, jobs, greedy=greedy, measurements_only=False)
        indexed = []
        for chunk in self._executor.map(_worker_collect, payloads):
            indexed.extend(chunk)
        indexed.sort(key=lambda item: item[0])
        return [episode for _, episode in indexed]

    def evaluate(self, policy, jobs: Sequence[EpisodeJob]) -> RolloutSummary:
        if self._closed:
            raise RuntimeError("episode collector is closed")
        payloads = self._payloads(
            policy, jobs, greedy=True, measurements_only=True)
        measurements = list(self._executor.map(_worker_collect, payloads))
        return summarize_measurements(merge_measurements(measurements))

    def close(self) -> None:
        if not self._closed:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._closed = True


def build_episode_collector(settings: CollectionSettings,
                            runtime: DiscreteRuntime,
                            composite: ProgramShieldComposite
                            ) -> EpisodeCollector:
    resolved = settings.resolved()
    if resolved.backend == "serial":
        return SerialEpisodeCollector(runtime, composite)
    return ProcessEpisodeCollector(runtime, resolved)


class _OracleWorkerContext:
    def __init__(self, runtime: DiscreteRuntime):
        self.env = _make_env(runtime)
        self.iface = extract_interface(runtime.model_path)
        self.max_steps = runtime.max_steps

    def close(self) -> None:
        self.env.close()


_ORACLE_WORKER: _OracleWorkerContext | None = None


def _init_oracle_worker(runtime: DiscreteRuntime) -> None:
    global _ORACLE_WORKER
    _ORACLE_WORKER = _OracleWorkerContext(runtime)
    atexit.register(_ORACLE_WORKER.close)


def _worker_oracle_episode(job: OracleEpisodeJob):
    if _ORACLE_WORKER is None:
        raise RuntimeError("oracle worker was not initialized")
    observations, actions = collect_oracle_episode(
        _ORACLE_WORKER.iface,
        _ORACLE_WORKER.env,
        max_steps=_ORACLE_WORKER.max_steps,
        reset_seed=job.env_seed,
        include_terminal=job.include_terminal,
    )
    return job.index, observations, actions


def generate_oracle_data_with_backend(
    runtime: DiscreteRuntime,
    settings: CollectionSettings,
    iface,
    n_samples: int,
    *,
    min_class_count: int = 0,
    max_resets: int = 20000,
    seed_base: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[int, int], int]:
    """Generate an ordered dataset with deterministic per-episode seeds."""
    resolved = settings.resolved()
    observations = []
    actions = []
    by_class: Counter[int] = Counter()
    resets = 0
    next_index = 0

    def need_more() -> bool:
        if sum(len(part) for part in actions) < n_samples:
            return True
        if min_class_count <= 0:
            return False
        return any(count < min_class_count for count in by_class.values())

    serial_env = None
    executor = None
    if resolved.backend == "serial":
        serial_env = _make_env(runtime)
    else:
        context = mp.get_context(resolved.start_method)
        executor = ProcessPoolExecutor(
            max_workers=resolved.workers,
            mp_context=context,
            initializer=_init_oracle_worker,
            initargs=(runtime,),
        )

    try:
        while need_more() and resets < max_resets:
            batch_size = min(
                1 if resolved.backend == "serial" else resolved.workers,
                max_resets - resets,
            )
            jobs = [
                OracleEpisodeJob(
                    index=index,
                    env_seed=_derived_seed(seed_base, 3, index),
                    include_terminal=min_class_count > 0,
                )
                for index in range(next_index, next_index + batch_size)
            ]
            next_index += batch_size
            if executor is None:
                batch = []
                for job in jobs:
                    obs_ep, act_ep = collect_oracle_episode(
                        iface,
                        serial_env,
                        max_steps=runtime.max_steps,
                        reset_seed=job.env_seed,
                        include_terminal=job.include_terminal,
                    )
                    batch.append((job.index, obs_ep, act_ep))
            else:
                batch = list(executor.map(_worker_oracle_episode, jobs))

            batch.sort(key=lambda item: item[0])
            for _, obs_ep, act_ep in batch:
                if not need_more() or resets >= max_resets:
                    break
                if min_class_count <= 0:
                    remaining = n_samples - sum(len(part) for part in actions)
                    obs_ep = obs_ep[:remaining]
                    act_ep = act_ep[:remaining]
                observations.append(np.asarray(obs_ep, dtype=np.float32))
                act_ep = np.asarray(act_ep, dtype=np.int64)
                actions.append(act_ep)
                by_class.update(int(action) for action in act_ep)
                resets += 1
    finally:
        if serial_env is not None:
            serial_env.close()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    if not observations:
        raise RuntimeError("oracle collection produced no samples")
    if need_more():
        raise RuntimeError(
            "oracle coverage target was not reached: "
            f"n={sum(len(part) for part in actions)}, "
            f"class_counts={dict(by_class)}, min_class_count={min_class_count}, "
            f"max_resets={max_resets}"
        )
    return (
        np.concatenate(observations, axis=0),
        np.concatenate(actions, axis=0),
        dict(sorted(by_class.items())),
        resets,
    )
