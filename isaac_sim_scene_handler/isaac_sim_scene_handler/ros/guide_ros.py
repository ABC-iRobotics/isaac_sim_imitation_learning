from __future__ import annotations

import random
from typing import Optional
import argparse
from threading import Thread

import rclpy
from rclpy.node import Node

from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.callback_groups import ReentrantCallbackGroup

from rclpy.executors import ExternalShutdownException
from rclpy.executors import MultiThreadedExecutor

from isaac_sim_scene_handler.core.guide_simulator import GUIDESimulator

from isaac_sim_scene_handler.types.geometry import Pose

from isaac_sim_msgs.srv import Randomize
from isaac_sim_msgs.srv import RegisterScene
from isaac_sim_msgs.srv import Pose as PoseSrv
from isaac_sim_msgs.srv import Attribute
from isaac_sim_msgs.srv import Collision
from isaac_sim_msgs.srv import CheckSuccess


class GUIDEROS2Interface(Node):
    def __init__(self, backend: GUIDESimulator, node_name: Optional[str], namespace: Optional[str]):
        super().__init__(node_name=node_name, namespace=namespace)

        self._backend = backend

        self._reentrant_group = ReentrantCallbackGroup()
        self._mutually_exclusive_group = MutuallyExclusiveCallbackGroup()

        # Randomize Scene
        self._randomize_scene = self.create_service(
            srv_type=Randomize,
            srv_name="/Randomize",
            callback=self._randomize_callback,
            callback_group=self._reentrant_group,
        )

        # Reset Scene
        self._reset_scene = self.create_service(
            srv_type=Randomize,
            srv_name="/Reset",
            callback=self._reset_callback,
            callback_group=self._reentrant_group,
        )

        # Register Scene
        self._register_scene = self.create_service(
            srv_type=RegisterScene,
            srv_name="/Register",
            callback=self._register_callback,
            callback_group=self._mutually_exclusive_group,
        )

        # Pose
        self._pose_request = self.create_service(
            srv_type=PoseSrv,
            srv_name="/PoseRequest",
            callback=self._pose_request_callback,
            callback_group=self._reentrant_group,
        )

        # Attribute
        self._attribute_request = self.create_service(
            srv_type=Attribute,
            srv_name="/AttributeRequest",
            callback=self._attribute_request_callback,
            callback_group=self._reentrant_group,
        )

        # Collision
        self._collision_request = self.create_service(
            srv_type=Collision,
            srv_name="/CollisionRequest",
            callback=self._collision_request_callback,
            callback_group=self._reentrant_group,
        )

        # Success
        self._is_success_request = self.create_service(
            srv_type=CheckSuccess,
            srv_name="/IsSuccess",
            callback=self._is_success_callback,
            callback_group=self._reentrant_group,
        )

    def _randomize_callback(self, request: Randomize.Request) -> Randomize.Response:
        response = Randomize.Response()
        try:
            id = request.id
            seed = request.seed

            if seed == 0 or seed is None:
                seed = random.randint(0, 1e6)
            else:
                seed = None

            success = self._backend._scene_manager.randomize_scene(scene_id=id, seed=seed)

            response.message = ""
            response.success = success
        except Exception as e:
            response.message = e
            response.success = False
        finally:
            return response

    def _reset_callback(self, request: Randomize.Request) -> Randomize.Response:
        response = Randomize.Response()
        try:
            id = request.id
            seed = request.seed

            if seed == 0 or seed is None:
                seed = random.randint(0, 1e6)
            else:
                seed = None

            success = self._backend._scene_manager.reset_scene(scene_id=id, seed=seed)

            response.message = ""
            response.success = success
        except Exception as e:
            response.message = e
            response.success = False
        finally:
            return response

    def _register_callback(self, request: RegisterScene.Request) -> RegisterScene.Response:
        response = RegisterScene.Response()
        try:
            path = request.path

            id, offset, _ = self._backend._scene_manager.add_scene(path)

            response.id = id
            response.offset = list(offset)
            response.message = ""
            response.success = True

        except Exception as e:
            response.id = -1
            response.offset = [0.0, 0.0, 0.0]
            response.message = e
            response.success = False
        finally:
            return response

    def _pose_request_callback(self, request: PoseSrv.Request) -> PoseSrv.Response:
        response = PoseSrv.Response()
        try:
            path = request.path

            pose: Pose = self._backend.call("get_world_poses", prim_path=path)
            if isinstance(pose, bool):
                raise Exception("Getting pose failed!")

            response.pose = pose.to_ros()
            response.message = ""
            response.success = True
        except Exception as e:
            response.message = e
            response.success = False
        finally:
            return response

    def _attribute_request_callback(self, request: Randomize.Request) -> Randomize.Response:
        response = Randomize.Response()
        try:
            path = request.path
            attribute = request.attribute
            value = request.value

            success = self._backend.call(
                "set_prim_attribute_value", prim_path=path, attribute_name=attribute, value=value
            )

            response.message = ""
            response.success = success
        except Exception as e:
            response.message = e
            response.success = False
        finally:
            return response

    def _collision_request_callback(self, request: Randomize.Request) -> Randomize.Response:
        response = Randomize.Response()
        try:
            prim1 = request.prim1
            prim2 = request.prim2

            collision = self._backend.call("is_prim_clashing", prim_path=prim1, scope=prim2)

            response.collision = collision
        except Exception:
            response.collision = False
        finally:
            return response

    def _is_success_callback(self, request: Randomize.Request) -> Randomize.Response:
        response = Randomize.Response()
        try:
            id = request.id

            success = self._backend._scene_manager.is_success(id)

            response.message = ""
            response.success = success
        except Exception as e:
            response.message = e
            response.success = False
        finally:
            return response


def launch_ros_interface(node: GUIDEROS2Interface):
    rclpy.init(args=None)
    executor = MultiThreadedExecutor(num_threads=13)
    try:
        executor.add_node(node)
        executor.spin()

        node.destroy_node()
        rclpy.shutdown()

    except (KeyboardInterrupt, ExternalShutdownException):
        pass


def create_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--id",
        type=int,
        help="Id number of the initialized simulation. Used for namespacing.",
        default=0,
    )
    parser.add_argument("-d", "--debug", type=bool, help="Debug simulator.", default=False)


def ros_entry_point():
    parser = argparse.ArgumentParser(
        description="GUIDE synthetic data generator framework ROS 2 entry point."
    )
    create_arguments(parser)
    args = parser.parse_args()
    sim = GUIDESimulator(sim_id=args.id)
    sim.init_runtime(debug=args.debug)
    sim.init_scene_manager()
    ros_interface = GUIDEROS2Interface(sim, f"GUIDE_{args.id}")
    ros_t = Thread(target=launch_ros_interface, args=(ros_interface,))
    ros_t.start()

    sim.run_runtime_loop()

    ros_t.join()
