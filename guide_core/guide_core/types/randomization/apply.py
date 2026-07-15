"""Apply distributions attached to scene instructions through a Randomizer.

Pure glue between the scene layer and the randomization engine. It walks a list
of instruction dicts and, for each one carrying a ``pose_dist`` (a
``Distribution``), draws it through the single ``Randomizer`` and replaces the
instruction's concrete ``kwargs['pose']`` using an injected ``pose_builder``.

Geometry is deliberately kept out of this module (the caller passes a
``pose_builder`` that turns a realized 7-vector ``[x, y, z, w, x, y, z]`` into a
geometry ``Pose``), so the module stays Isaac/ROS/SciPy-free and unit-testable.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from .engine import Randomizer


def draw_name(instruction: dict) -> str:
    """Stable, human-readable draw key for an instruction.

    Uses the (already scene-prefixed) ``prim_path`` when present so the recorded
    key matches the target it positions; falls back to object identity.
    """
    kwargs = instruction.get("kwargs", {})
    name = kwargs.get("prim_path")
    if isinstance(name, list):
        return ",".join(str(n) for n in name)
    if name:
        return str(name)
    return f"pose@{id(instruction):x}"


def draw_instructions(
    instructions: Iterable[dict],
    randomizer: Randomizer,
    pose_builder: Callable[[Any], Any],
) -> None:
    """Draw every instruction's ``pose_dist`` and write back a concrete pose.

    Mutates each instruction in place: the realized value is captured in the
    randomizer's record under :func:`draw_name`, and ``kwargs['pose']`` is set to
    ``pose_builder(value)``. Instructions without a ``pose_dist`` are left alone.
    """
    for instruction in instructions:
        dist = instruction.get("pose_dist")
        if dist is None:
            continue
        value = randomizer.draw(draw_name(instruction), dist)
        instruction.setdefault("kwargs", {})["pose"] = pose_builder(value)
