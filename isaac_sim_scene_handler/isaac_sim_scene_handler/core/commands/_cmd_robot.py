from isaacsim.ros2.bridge.scripts.og_shortcuts.og_utils import Ros2JointStatesGraph
from isaacsim.ros2.bridge.scripts.og_shortcuts.og_utils import Ros2TfPubGraph
from isaacsim.ros2.bridge.scripts.og_shortcuts.og_rtx_sensors import Ros2CameraGraph

# from isaacsim.sensors.camera import Camera
from isaac_sim_scene_handler.types.geometry import Pose
from isaacsim.sensors.camera import Camera


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


def _cmd_create_robot_control(
    self, namespace: str = "", articulation_root: str = "", path: str | None = None
) -> None:

    assert self.state in [READY, PAUSED, STOPPED]

    robot_control = Ros2JointStatesGraph()
    robot_control._publisher = True
    robot_control._subscriber = True

    robot_control._node_namespace = namespace
    robot_control._art_root_path = articulation_root
    if path is not None:
        robot_control._og_path = path

    print("Creating robot control")
    robot_control.make_graph()


def _cmd_create_camera(
    self,
    pose: Pose | None = None,
    camera_path: str = "/Camera",
    path: str | None = None,
    width: int = 1920,
    height: int = 1080,
    frame: str = "sim_camera",
    namespace: str = "",
    topic: str = "/rgb",
):

    assert self.state in [READY, PAUSED, STOPPED]

    if pose is not None:
        camera = Camera(
            prim_path=camera_path,
            dt=self._dt,
            resolution=(width, height),
            position=pose.position.to_numpy(),
            orientation=pose.orientation.to_numpy_quat(),
        )

    cp = Ros2CameraGraph()
    if path is not None:
        cp._og_path = path
    cp._camera_prim = camera_path
    cp._frame_id = frame
    cp._node_namespace = namespace
    cp._rgb_topic = topic
    cp._depth_pub = False

    print("Creating camera")
    cp.make_graph()


def _cmd_create_tf_graph(
    self,
    prim: str,
    path: str | None = None,
    parent_prim: str = "/World",
    namespace: str = "",
):

    assert self.state in [READY, PAUSED, STOPPED]

    tf_g = Ros2TfPubGraph()
    if path is not None:
        tf_g._og_path = path
    tf_g._node_namespace = namespace
    tf_g._target_prim = prim
    tf_g._parent_prim = parent_prim

    print("Creating tf graph")
    tf_g.make_graph()
