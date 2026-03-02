from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Tuple

from threading import Thread

import time

from isaac_sim_scene_handler.core.runtime import IsaacSimRuntime
from isaac_sim_scene_handler.core.runtime import import_and_bind
from isaac_sim_scene_handler.scene.scene_manager import SceneManager

from isaac_sim_scene_handler.types.geometry import Pose
from isaac_sim_scene_handler.types.geometry import Point


class GUIDESimulator:

    _sim_id: int
    _sim_path: str

    _runtime: IsaacSimRuntime
    _scene_manager: SceneManager

    def __init__(self, sim_id: int = 0):
        self._sim_id = sim_id
        self._sim_path = f"/Sim_{sim_id}"

    def _import_after_runtime(self):
        import_and_bind("isaac_sim_scene_handler.types.geometry.Pose", namespace=globals())
        import_and_bind("isaac_sim_scene_handler.types.geometry.Point", namespace=globals())
        import_and_bind("isaac_sim_scene_handler.types.geometry.Rotation", namespace=globals())

    # -------------------
    # Runtime functions
    # -------------------
    def init_runtime(self, config: Optional[dict] = None, debug: bool = False):
        self._runtime = IsaacSimRuntime(config=config, debug=debug)
        # self._import_after_runtime()

    def run_runtime_loop(self):
        self._runtime.run_loop()

    def call(self, name: str, timeout: Optional[float] = None, *args: Any, **kwargs: Any) -> Any:
        return self._runtime.call(name, timeout, *args, **kwargs)

    def _add_scene_to_runtime(
        self,
        scene_id: int,
        offset: Tuple[float, float, float] = (0, 0, 0),
        config: Optional[dict] = None,
    ):

        scene_path = f"/Scene_{scene_id}"
        pose: Pose = Pose(position=Point(offset))

        self._runtime.call("add_scene", stage_config=config, root=scene_path)
        self._runtime.call("set_world_poses", prim_path=scene_path, pose=pose)

    # --------------------------
    # Scene manager functions
    # --------------------------
    def init_scene_manager(self):
        self._scene_manager = SceneManager()

    def register_scene(self, package_name: str) -> int:
        id, offset, config = self._add_scene_to_manager(package_name)
        print(offset)
        self._add_scene_to_runtime(
            id,
            offset,
            config,
        )
        # self._runtime.step()
        scene_path = f"/Scene_{id}"
        robots = self._scene_manager._scenes[id].create_robot_graphs()
        cameras = self._scene_manager._scenes[id].create_camera_graphs()
        # time.sleep(10)

        if robots is not None:
            for robot in robots:
                self._runtime.call(
                    "create_robot_control",
                    namespace=f'{self._sim_path}{scene_path}{robot.get("namespace", "/robot")}',
                    articulation_root=f'{scene_path}{robot.get("articulation_root", "/robot")}',
                    path=f'{scene_path}/Graph{robot.get("path", "/robot")}_control_graph',
                )

                self._runtime.call(
                    "create_tf_graph",
                    namespace=f'{self._sim_path}{scene_path}{robot.get("namespace", "/robot")}',  #
                    prim=f'{scene_path}{robot.get("articulation_root", "/robot")}',
                    parent_prim=f'{scene_path}{robot.get("articulation_root", "/robot")}'.rsplit(
                        "/", 1
                    )[0]
                    or "/",
                    path=f'{scene_path}/Graph{robot.get("path", "/robot")}_tf_graph',
                )

        if cameras is not None:
            for camera in cameras:
                self._runtime.call(
                    "create_camera",
                    camera_path=f'{scene_path}{camera.get("camera_path", "/cam")}',
                    path=f'{scene_path}/Graph{camera.get("path", "/cam")}_camera_graph',
                    width=camera.get("width", 640),
                    height=camera.get("height", 480),
                    frame=camera.get("frame", "cam"),
                    namespace=f"{self._sim_path}{scene_path}",
                    topic=camera.get("topic", "/rgb"),
                )

        return id

    def _add_scene_to_manager(self, package_name: str):
        assert package_name is not None

        return self._scene_manager.add_scene(package_name)


def main(sim: GUIDESimulator):
    for _ in range(5):
        try:
            sim.register_scene("/home/user/ros2_ws/src/block_bin_task/src/block_bin_task")
        except Exception as e:
            print(f"Exception: {e}")
    # time.sleep(10)
    sim.call("start")
    time.sleep(300)
    sim.call("shutdown")


if __name__ == "__main__":
    sim = GUIDESimulator()
    sim.init_runtime(debug=False)
    sim.init_scene_manager()

    main_t = Thread(target=main, args=(sim,))
    main_t.start()

    sim.run_runtime_loop()

    main_t.join()
