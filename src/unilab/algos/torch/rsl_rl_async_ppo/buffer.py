"""PPO-specific rollout shared-memory schema."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Mapping
from multiprocessing import shared_memory
from typing import Any

import numpy as np

_SPAWN_CTX = mp.get_context("spawn")


class RslRlPpoRolloutBuffer:
    """Single-schema shared rollout buffer for async RSL-RL PPO.

    Slots are env-major in shared memory: ``[num_envs, num_steps, ...]``.
    Learner-side staging converts them to RSL-RL's time-major storage layout.
    """

    _BASE_FIELDS = {
        "actions": lambda ne, ns, ad, dp: (ne, ns, ad),
        "rewards": lambda ne, ns, ad, dp: (ne, ns),
        "dones": lambda ne, ns, ad, dp: (ne, ns),
        "values": lambda ne, ns, ad, dp: (ne, ns, 1),
        "actions_log_prob": lambda ne, ns, ad, dp: (ne, ns, 1),
        "policy_version_start": lambda ne, ns, ad, dp: (1,),
        "policy_version_end": lambda ne, ns, ad, dp: (1,),
        "rollout_created_time_ns": lambda ne, ns, ad, dp: (1,),
        "truncated": lambda ne, ns, ad, dp: (ne, ns),
        "raw_rewards": lambda ne, ns, ad, dp: (ne, ns),
        "timeout_bootstrap_reward": lambda ne, ns, ad, dp: (ne, ns),
    }

    def __init__(
        self,
        *,
        num_envs: int,
        num_steps: int,
        obs_shapes: Mapping[str, tuple[int, ...]],
        action_dim: int,
        distribution_param_shapes: tuple[tuple[int, ...], ...],
        num_slots: int = 1,
        create: bool = True,
        shm_names: Mapping[str, str] | None = None,
    ) -> None:
        if num_slots != 1:
            raise ValueError("Async RSL-RL PPO v1 supports exactly one in-flight rollout slot")
        self.num_envs = int(num_envs)
        self.num_steps = int(num_steps)
        self.obs_shapes = dict(obs_shapes)
        self.action_dim = int(action_dim)
        self.distribution_param_shapes = tuple(tuple(s) for s in distribution_param_shapes)
        self.num_slots = int(num_slots)

        self._shm_blocks: dict[str, shared_memory.SharedMemory] = {}
        self._arrays: dict[str, np.ndarray] = {}

        for field, shape in self._field_shapes().items():
            nbytes = int(np.prod(shape)) * np.dtype(np.float32).itemsize
            if create:
                shm = shared_memory.SharedMemory(create=True, size=max(nbytes, 1))
            else:
                if shm_names is None:
                    raise ValueError("shm_names is required when create=False")
                shm = shared_memory.SharedMemory(name=shm_names[field], create=False)
            self._shm_blocks[field] = shm
            self._arrays[field] = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)

        if create:
            self._write_ptr = _SPAWN_CTX.Value("l", 0)
            self._read_ptr = _SPAWN_CTX.Value("l", 0)

    def _field_shapes(self) -> dict[str, tuple[int, ...]]:
        fields: dict[str, tuple[int, ...]] = {}
        for name, obs_shape in self.obs_shapes.items():
            fields[f"obs/{name}"] = (self.num_slots, self.num_envs, self.num_steps, *obs_shape)
            fields[f"last_obs/{name}"] = (self.num_slots, self.num_envs, *obs_shape)
        for index, shape in enumerate(self.distribution_param_shapes):
            fields[f"distribution_params/{index}"] = (
                self.num_slots,
                self.num_envs,
                self.num_steps,
                *shape,
            )
        for field, shape_fn in self._BASE_FIELDS.items():
            fields[field] = (
                self.num_slots,
                *shape_fn(
                    self.num_envs,
                    self.num_steps,
                    self.action_dim,
                    self.distribution_param_shapes,
                ),
            )
        return fields

    @property
    def name(self) -> dict[str, str]:
        return {field: shm.name for field, shm in self._shm_blocks.items()}

    @property
    def slot_shapes(self) -> dict[str, tuple[int, ...]]:
        return {field: tuple(arr.shape[1:]) for field, arr in self._arrays.items()}

    def attach_sync_primitives(self, write_ptr: Any, read_ptr: Any) -> None:
        self._write_ptr = write_ptr
        self._read_ptr = read_ptr

    @property
    def write_buffer(self) -> dict[str, np.ndarray]:
        slot = int(self._write_ptr.value) % self.num_slots
        return {field: arr[slot] for field, arr in self._arrays.items()}

    def signal_write_done(self) -> None:
        with self._write_ptr.get_lock():
            self._write_ptr.value += 1

    def available(self) -> int:
        return min(max(0, int(self._write_ptr.value) - int(self._read_ptr.value)), self.num_slots)

    def wait_for_data(self, timeout: float = 60.0) -> bool:
        import time

        deadline = time.monotonic() + timeout
        while self.available() == 0:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.001)
        return True

    def read_numpy_views(self) -> dict[str, np.ndarray]:
        slot = int(self._read_ptr.value) % self.num_slots
        return {field: arr[slot] for field, arr in self._arrays.items()}

    def advance_read(self) -> None:
        with self._read_ptr.get_lock():
            self._read_ptr.value = min(int(self._read_ptr.value) + 1, int(self._write_ptr.value))

    def cleanup(self) -> None:
        for shm in self._shm_blocks.values():
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass

    def close(self) -> None:
        for shm in self._shm_blocks.values():
            try:
                shm.close()
            except Exception:
                pass
