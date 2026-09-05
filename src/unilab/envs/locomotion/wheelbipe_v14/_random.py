"""Source-V14 random draws backed by UniLab's run-level NumPy stream.

WheelBipe's vectorized owner follows the repository-wide training seed
contract, which seeds NumPy's process-global ``RandomState`` before owner
construction/reset.  ``default_rng()`` creates an unrelated entropy-backed
stream, so using it inside a task owner makes both training replay and the
Gym facade's isolated RNG snapshot incomplete.  This tiny adapter exposes the
``Generator`` methods used by the source-semantic owners while delegating
every draw to the seeded legacy stream.

The adapter intentionally carries no state of its own.  Tests may still
replace an owner's stream with a seeded ``numpy.random.Generator`` because the
methods below mirror that small structural interface.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class LegacyNumpyRandomStream:
    """Generator-shaped view of NumPy's process-global seeded stream."""

    def random(self, size: Any = None) -> Any:
        return np.random.random(size=size)

    def uniform(self, low: Any = 0.0, high: Any = 1.0, size: Any = None) -> Any:
        return np.random.uniform(low=low, high=high, size=size)

    def choice(
        self,
        a: Any,
        size: Any = None,
        replace: bool = True,
        p: Any = None,
    ) -> Any:
        return np.random.choice(a=a, size=size, replace=replace, p=p)

    def integers(self, low: Any, high: Any = None, size: Any = None) -> Any:
        return np.random.randint(low=low, high=high, size=size)


GLOBAL_NUMPY_RANDOM = LegacyNumpyRandomStream()


__all__ = ["GLOBAL_NUMPY_RANDOM", "LegacyNumpyRandomStream"]
