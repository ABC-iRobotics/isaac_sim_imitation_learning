from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict


class Layer(IntEnum):
    UTILITY = -1
    PHYSICAL_PROCESS = 0
    PRIMITIVE = 1
    STEP = 2
    SEQUENCE = 3
    SUBTASK = 4
    TASK = 5
    PROCEDURE = 6
    SERVICE = 7


class DemoStatus(Enum):
    PERFECT = "perfect"
    RECOVERY = "recovery"
    FAILURE = "failure"


class ExecutionMode(str, Enum):
    NORMAL = "normal"
    CONDITION = "condition"
    LOOP = "loop"


@dataclass
class ExecutionResult:
    status: DemoStatus
    error_message: str = ""
    outputs: Dict[str, Any] = field(default_factory=dict)
