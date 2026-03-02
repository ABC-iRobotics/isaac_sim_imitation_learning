import isaacsim.core.utils.prims as prims_utils
from isaacsim.util.clash_detection import ClashDetector

from isaac_sim_scene_handler.types.isaac_state import IsaacState

UNINITIALIZED = IsaacState.UNINITIALIZED
INITIALIZING = IsaacState.INITIALIZING
STOPPED = IsaacState.STOPPED
LOADING = IsaacState.LOADING
READY = IsaacState.READY
RUNNING = IsaacState.RUNNING
PAUSED = IsaacState.PAUSED
ERROR = IsaacState.ERROR
SHUTTING_DOWN = IsaacState.SHUTTING_DOWN


def __init_clash_detector(self):

    assert self.state not in [UNINITIALIZED, INITIALIZING, ERROR, SHUTTING_DOWN]

    self._cd = ClashDetector(self._stage, logging=self._debug)


def _cmd_get_scope(self) -> str:
    if self._cd is None:
        __init_clash_detector()

    return self._cd.get_scope()


def _cmd_set_scope(self, searchset: str) -> None:
    if self._cd is None:
        __init_clash_detector()

    self._cd.set_scope(searchset)


def _cmd_is_prim_clashing(self, prim_path: str, scope: str | None = None) -> bool:
    if self._cd is None:
        __init_clash_detector()

    assert self.state in [RUNNING]

    if isinstance(scope, str):
        _cmd_set_scope(scope)

    prim = prims_utils.get_prim_at_path(prim_path)

    return self._cd.is_prim_clashing(prim)
