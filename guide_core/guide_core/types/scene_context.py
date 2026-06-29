"""Per-episode reproduction context.

``SceneContext`` is the orchestration-layer carrier that ties a scene/episode
identity to the realized ``RandomizationRecord``. The record itself stays a pure
value object (no scene/episode awareness); identity lives here and is what gets
persisted (e.g. the dataset sidecar filename). Importing the record submodule
directly keeps this dependency-light (NumPy only, no SciPy).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from guide_core.types.randomization.record import RandomizationRecord


@dataclass(slots=True)
class SceneContext:
    scene_id: int
    episode_index: int = 0
    record: Optional[RandomizationRecord] = None

    def to_dict(self) -> dict:
        return {
            "scene_id": int(self.scene_id),
            "episode_index": int(self.episode_index),
            "record": self.record.to_dict() if self.record is not None else None,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict) -> "SceneContext":
        rec = data.get("record")
        return cls(
            scene_id=int(data["scene_id"]),
            episode_index=int(data.get("episode_index", 0)),
            record=RandomizationRecord.from_dict(rec) if rec else None,
        )

    @classmethod
    def from_json(cls, payload: str) -> "SceneContext":
        return cls.from_dict(json.loads(payload))
