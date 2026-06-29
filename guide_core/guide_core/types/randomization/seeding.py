"""The single RNG authority for the framework.

``SeedTree`` is the only place ``numpy`` generators are created. A run has one
master seed (injected, or auto-drawn from system entropy at scene
registration); per-scene / per-episode generators are spawned deterministically
from it, so two branches never share or perturb each other's stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SeedTree:
    """Deterministic generator factory rooted at a single master seed."""

    master: int

    @classmethod
    def create(cls, master: int | None = None) -> "SeedTree":
        """Resolve the master seed once.

        Injected ``master`` wins; otherwise draw a fresh one from OS entropy
        (``np.random.SeedSequence()`` with no argument seeds from ``os.urandom``
        — the modern, higher-quality equivalent of classic time-based seeding).
        The result is concrete and loggable, so even an "unseeded" run is
        reproducible after the fact.
        """
        if master is None:
            master = int(np.random.SeedSequence().entropy)
        return cls(master=int(master))

    def generator(self, *path: int) -> tuple[np.random.Generator, int]:
        """Return ``(generator, seed)`` for the branch identified by ``path``.

        ``path`` is typically ``(scene_id, episode_index)``. The returned
        ``seed`` is the concrete 32-bit seed the generator was built from —
        store it for seed-level reproduction / logging.
        """
        seed_seq = np.random.SeedSequence([self.master, *path])
        seed = int(seed_seq.generate_state(1)[0])
        return np.random.default_rng(seed_seq), seed
