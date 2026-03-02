import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from threading import Event, Lock, Thread

import gymnasium as gym
import rclpy
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.processor.factory import (
    make_default_robot_action_processor,
    make_default_robot_observation_processor,
)
from lerobot.rl.process import ProcessSignalHandler
from lerobot.robots import RobotConfig
from lerobot.utils.hub import HubMixin
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node

# import torch._inductor.config as inductor_config

# inductor_config.triton.cudagraphs = False

# import torch._dynamo
# torch._dynamo.disable()


# global task
# task: str = 'Move a test tube from one rack to another.'

# def to_nchw3(pic: torch.Tensor) -> torch.Tensor:

#     if pic.ndim > 3:
#         pic = pic.squeeze()

#     if pic.ndim == 2:
#         # [H, W]
#         pic = pic.unsqueeze(0)        # [1, H, W]
#         pic = pic.repeat(3, 1, 1)     # [3, H, W]

#     elif pic.ndim == 3:
#         # Lehet [H, W, 3] vagy [3, H, W]
#         if pic.shape[0] == 3:
#             # már [3, H, W], jó így
#             pass
#         elif pic.shape[-1] == 3:
#             # [H, W, 3] -> [3, H, W]
#             pic = pic.permute(2, 0, 1)
#         else:
#             raise ValueError(f"Unknown 3D tensor format: {pic.shape}")

#     else:
#         raise ValueError(f"Expected 2D or 3D tensor, but got: {pic.shape}")

#     pic = pic.unsqueeze(0)   # [1, 3, H, W]
#     return pic


def instructor_query(robot_node: Node, subtasks: list[str]) -> None:
    global task
    task = robot_node.query_subtask(subtasks)
    robot_node.get_logger().info(f"Updated task from Instructor: {task}")


# def main():
#     try:
#         global task
#         rclpy.init(args=None)

#         executor = MultiThreadedExecutor(num_threads=32)

#         robot = gym.make(id="IL_Gym_Env", max_episode_steps=10000, executor=executor)

#         robot.unwrapped.logger.info("Initialized environment")

#         spin_thread = Thread(target=executor.spin)
#         spin_thread.start()

#         global device
#         device = torch.device("cuda")  # or "cuda" or "cpu"
#         model_id = os.path.expanduser("~/Desktop/checkout/pi0_0/checkpoints/030000/pretrained_model")
#         model: PI0Policy = PI0Policy.from_pretrained(model_id)

#         action_features = hw_to_dataset_features(robot.unwrapped.action_features, "action")
#         obs_features = hw_to_dataset_features(robot.unwrapped.observation_features, "observation")

#         dataset_features = {**action_features, **obs_features}

#         preprocess, postprocess = make_pre_post_processors(
#         model.config,
#         model_id,

#         preprocessor_overrides={"device_processor": {"device": str(device)}},
#     )

#         model.config.rtc_config = RTCConfig(
#             enabled=True,
#             execution_horizon=32,  # How many steps to blend with previous chunk
#             max_guidance_weight=100.0,  # How strongly to enforce consistency
#             prefix_attention_schedule=RTCAttentionSchedule.EXP,  # Exponential blend
#         )

#         inference_delay = 16

#         action_queue = ActionQueue(model.config.rtc_config)

#         subtasks = [
#             "Move to home position.",
#             "Orienting above the purple tube.",
#             "Grasp the purple tube and pull it out.",
#             "Move to goal rack and insert the tube.",
#             ]

#         # instructor_thread = robot.unwrapped.create_timer(1.0, lambda: instructor_query(robot.unwrapped, subtasks))
#         # instructor_thread.cancel()

#         global robot_type
#         robot_type = "tm5-900"

#         MAX_EPISODES = 10
#         TRIES_PER_EPISODE = 2
#         STEPS_PER_SUBTASK = 1000
#         MAX_STEPS_PER_EPISODE = TRIES_PER_EPISODE * STEPS_PER_SUBTASK * len(subtasks)
#         LOG_PER_SUBTASK = 50
#         STEPS_PER_LOG = STEPS_PER_SUBTASK // LOG_PER_SUBTASK

#         def get_actions(policy):
#           global stop
#           global obs
#           global robot_type
#           global device
#           global task

#           while not stop:
#             if action_queue.qsize() <= inference_delay:

#               prev_actions = action_queue.get_left_over()
#               obs_frame = build_inference_frame(
#                 observation=obs, ds_features=dataset_features, device=device, task=task, robot_type=robot_type
#                 )

#             obs_processed = preprocess(obs_frame)

#             # Generate actions WITH RTC
#             actions = policy.predict_action_chunk(
#                 obs_processed,
#                 inference_delay=inference_delay,
#                 prev_chunk_left_over=prev_actions,
#             )

#             action_queue.merge(
#                 actions, actions, inference_delay
#             )


#         for _ in range(MAX_EPISODES):
#             terminated = False
#             truncated = False

#             global obs
#             obs, _ = robot.reset()

#             # instructor_thread.reset()
#             # time.sleep(1.0)  # wait for first subtask query to finish
#             global stop

#             stop = False
#             rtc_thread = Thread(target=get_actions, args=([model]))
#             rtc_thread.start()

#             for step in range(MAX_STEPS_PER_EPISODE):

#                 # obs_frame = build_inference_frame(
#                 # observation=obs, ds_features=dataset_features, device=device, task=task, robot_type=robot_type
#                 # )

#                 # obs = preprocess(obs_frame)

#                 # action: PolicyAction = model.select_action(obs)
#                 # action = postprocess(action)
#                 while True:
#                     if action_queue.qsize() > 0:
#                         break
#                     time.sleep(0.01)
#                 action = action_queue.get() # RTC action
#                 if action is None:
#                     continue
#                 robot.unwrapped.logger.info(f"Predicted action: {action}")
#                 action = make_robot_action(action, dataset_features)
#                 if(step % STEPS_PER_LOG == 0): robot.unwrapped.logger.info(f"Task: {task}, Step: {step}, Action: {action}")
#                 obs, _, terminated, truncated, _ = robot.step(action)

#                 if terminated or truncated:
#                     robot.unwrapped.logger.info('Episode terminated or truncated!')
#                     break

#             # instructor_thread.cancel()
#             stop = True
#             print("Episode finished! Starting new episode...")
#             rtc_thread.join()

#     except (KeyboardInterrupt, ExternalShutdownException):
#         pass

#     finally:
#         # if instructor_thread is not None: instructor_thread.cancel()
#         robot.unwrapped.destroy_node()
#         rclpy.shutdown()

# if __name__ == "__main__":
#     main()


class RobotWrapper:
    def __init__(self, robot: gym.Env):
        self._robot = robot
        self.robot: gym.Env = robot.unwrapped
        self.logger = robot.unwrapped.logger
        self._obs = robot.unwrapped.get_obs()
        self.lock = Lock()
        self.terminated = False
        self.truncated = False

    def get_observation(self):
        with self.lock:
            return self._obs

    def send_action(self, action) -> bool:
        with self.lock:
            (self._obs, _, self.terminated, self.truncated, _) = self._robot.step(action)

    def reset(self):
        with self.lock:
            self._obs = (self._robot.reset())[0]

    def observation_features(self):
        with self.lock:
            return self.robot.observation_features

    def action_features(self):
        with self.lock:
            return self.robot.action_features


@dataclass
class RTCDemoConfig(HubMixin):
    """Configuration for RTC demo with action chunking policies and real robots."""

    # Policy configuration
    policy: PreTrainedConfig | None = None

    # Robot configuration
    robot: RobotConfig | None = None

    # RTC configuration
    rtc: RTCConfig = field(
        default_factory=lambda: RTCConfig(
            execution_horizon=8,
            max_guidance_weight=100.0,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
        )
    )

    # Demo parameters
    duration: float = 30.0  # Duration to run the demo (seconds)
    fps: float = 10.0  # Action execution frequency (Hz)

    # Compute device
    device: str | None = None  # Device to run on (cuda, cpu, auto)

    # Get new actions horizon. The amount of executed steps after which will be requested new actions.
    # It should be higher than inference delay + execution horizon.
    action_queue_size_to_get_new_actions: int = 50

    # Task to execute
    task: str = field(default="", metadata={"help": "Task to execute"})

    # Torch compile configuration
    use_torch_compile: bool = field(
        default=False,
        metadata={"help": "Use torch.compile for faster inference (PyTorch 2.0+)"},
    )

    torch_compile_backend: str = field(
        default="inductor",
        metadata={"help": "Backend for torch.compile (inductor, aot_eager, cudagraphs)"},
    )

    torch_compile_mode: str = field(
        default="default",
        metadata={"help": "Compilation mode (default, reduce-overhead, max-autotune)"},
    )

    torch_compile_disable_cudagraphs: bool = field(
        default=True,
        metadata={
            "help": "Disable CUDA graphs in torch.compile. Required due to in-place tensor "
            "operations in denoising loop (x_t += dt * v_t) which cause tensor aliasing issues."
        },
    )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


def get_actions(
    policy: PI05Policy,
    robot: RobotWrapper,
    robot_observation_processor,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: RTCDemoConfig,
):
    """Thread function to request action chunks from the policy.

    Args:
        policy: The policy instance (SmolVLA, Pi0, etc.)
        robot: The robot instance for getting observations
        robot_observation_processor: Processor for raw robot observations
        action_queue: Queue to put new action chunks
        shutdown_event: Event to signal shutdown
        cfg: Demo configuration
    """
    try:
        logger = robot.logger
        logger.info("[GET_ACTIONS] Starting get actions thread")

        latency_tracker = LatencyTracker()  # Track latency of action chunks
        fps = cfg.fps
        time_per_chunk = 1.0 / fps

        dataset_features = hw_to_dataset_features(robot.observation_features(), "observation")
        policy_device = policy.config.device

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=None,  # Will load from pretrained processor files
            preprocessor_overrides={
                "device_processor": {"device": cfg.policy.config.device},
            },
        )

        logger.info(
            "[GET_ACTIONS] Preprocessor/postprocessor loaded successfully with embedded stats"
        )

        get_actions_threshold = cfg.action_queue_size_to_get_new_actions

        if not cfg.rtc.enabled:
            get_actions_threshold = 0

        global task
        # task = cfg.task

        while not shutdown_event.is_set():
            task = robot.robot.task
            if action_queue.qsize() <= get_actions_threshold:
                current_time = time.perf_counter()
                action_index_before_inference = action_queue.get_action_index()
                prev_actions = action_queue.get_left_over()

                inference_latency = latency_tracker.p95()
                inference_delay = math.ceil(inference_latency / time_per_chunk)

                obs = robot.get_observation()

                # Apply robot observation processor
                obs_processed = robot_observation_processor(obs)

                obs_with_policy_features = build_dataset_frame(
                    dataset_features, obs_processed, prefix="observation"
                )

                for name in obs_with_policy_features:
                    obs_with_policy_features[name] = torch.from_numpy(
                        obs_with_policy_features[name]
                    )
                    if "image" in name:
                        obs_with_policy_features[name] = (
                            obs_with_policy_features[name].type(torch.float32) / 255
                        )
                        obs_with_policy_features[name] = (
                            obs_with_policy_features[name].permute(2, 0, 1).contiguous()
                        )
                    obs_with_policy_features[name] = obs_with_policy_features[name].unsqueeze(0)
                    obs_with_policy_features[name] = obs_with_policy_features[name].to(
                        policy_device
                    )

                # Task should be a list, not a string!
                obs_with_policy_features["task"] = [task]
                obs_with_policy_features["robot_type"] = (
                    robot.robot.name if hasattr(robot.robot, "name") else ""
                )

                preproceseded_obs = preprocessor(obs_with_policy_features)

                # Generate actions WITH RTC
                actions = policy.predict_action_chunk(
                    preproceseded_obs,
                    inference_delay=inference_delay,
                    prev_chunk_left_over=prev_actions,
                )

                # After inference, access debug data
                debug_data = policy.rtc_processor.get_all_debug_steps()

                # Visualize denoising steps, corrections, etc.
                from lerobot.policies.rtc.debug_visualizer import RTCDebugVisualizer

                visualizer = RTCDebugVisualizer()

                # Store original actions (before postprocessing) for RTC
                original_actions = actions.squeeze(0).clone()

                postprocessed_actions = postprocessor(actions)

                postprocessed_actions = postprocessed_actions.squeeze(0)

                new_latency = time.perf_counter() - current_time
                new_delay = math.ceil(new_latency / time_per_chunk)
                latency_tracker.add(new_latency)

                if cfg.action_queue_size_to_get_new_actions < cfg.rtc.execution_horizon + new_delay:
                    logger.warning(
                        "[GET_ACTIONS] cfg.action_queue_size_to_get_new_actions Too small, It should be higher than inference delay + execution horizon."
                    )

                action_queue.merge(
                    original_actions,
                    postprocessed_actions,
                    new_delay,
                    action_index_before_inference,
                )
            else:
                # Small sleep to prevent busy waiting
                time.sleep(0.1)

        logger.info("[GET_ACTIONS] get actions thread shutting down")
    except Exception as e:
        logger.error(f"[GET_ACTIONS] Fatal exception in get_actions thread: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def actor_control(
    robot: RobotWrapper,
    robot_action_processor,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: RTCDemoConfig,
):
    """Thread function to execute actions on the robot.

    Args:
        robot: The robot instance
        action_queue: Queue to get actions from
        shutdown_event: Event to signal shutdown
        cfg: Demo configuration
    """
    try:
        robot.logger.info("[ACTOR] Starting actor thread")

        action_count = 0
        action_interval = 1.0 / cfg.fps

        while not shutdown_event.is_set():
            start_time = time.perf_counter()

            # Try to get an action from the queue with timeout
            action = action_queue.get()

            if action is not None:
                action = action.cpu()
                action_dict = {
                    key: action[i].item() for i, key in enumerate(robot.action_features())
                }
                action_processed = robot_action_processor((action_dict, None))
                robot.send_action(action_processed)

                action_count += 1

            dt_s = time.perf_counter() - start_time
            time.sleep(max(0, (action_interval - dt_s) - 0.001))

        robot.logger.info(
            f"[ACTOR] Actor thread shutting down. Total actions executed: {action_count}"
        )
    except Exception as e:
        robot.logger.error(f"[ACTOR] Fatal exception in actor_control thread: {e}")
        robot.logger.error(traceback.format_exc())
        sys.exit(1)


def _apply_torch_compile(policy, cfg: RTCDemoConfig, robot: RobotWrapper):
    """Apply torch.compile to the policy's predict_action_chunk method.

    Args:
        policy: Policy instance to compile
        cfg: Configuration containing torch compile settings

    Returns:
        Policy with compiled predict_action_chunk method
    """

    logger = robot.logger

    # PI models handle their own compilation
    if policy.type == "pi05" or policy.type == "pi0":
        return policy

    try:
        # Check if torch.compile is available (PyTorch 2.0+)
        if not hasattr(torch, "compile"):
            logger.warning(
                f"torch.compile is not available. Requires PyTorch 2.0+. "
                f"Current version: {torch.__version__}. Skipping compilation."
            )
            return policy

        logger.info("Applying torch.compile to predict_action_chunk...")
        logger.info(f"  Backend: {cfg.torch_compile_backend}")
        logger.info(f"  Mode: {cfg.torch_compile_mode}")
        logger.info(f"  Disable CUDA graphs: {cfg.torch_compile_disable_cudagraphs}")

        # Compile the predict_action_chunk method
        # - CUDA graphs disabled to prevent tensor aliasing from in-place ops (x_t += dt * v_t)
        compile_kwargs = {
            "backend": cfg.torch_compile_backend,
            "mode": cfg.torch_compile_mode,
        }

        # Disable CUDA graphs if requested (prevents tensor aliasing issues)
        if cfg.torch_compile_disable_cudagraphs:
            compile_kwargs["options"] = {"triton.cudagraphs": False}

        original_method = policy.predict_action_chunk
        compiled_method = torch.compile(original_method, **compile_kwargs)
        policy.predict_action_chunk = compiled_method
        logger.info("✓ Successfully compiled predict_action_chunk")

    except Exception as e:
        logger.error(f"Failed to apply torch.compile: {e}")
        logger.warning("Continuing without torch.compile")

    return policy


def main():
    try:
        rclpy.init(args=None)
        executor = MultiThreadedExecutor(num_threads=32)

        policy = None
        robot = None
        get_actions_thread = None
        actor_thread = None

        _robot = gym.make(id="IL_Gym_Env", max_episode_steps=10000, executor=executor)
        robot = RobotWrapper(_robot)
        robot.logger.info("Initialized environment")

        global task
        task = "Grab a test tube."

        spin_thread = Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        device = torch.device("cuda")
        # pretrained_path = os.path.expanduser("~/Desktop/checkout/pi0_0/checkpoints/030000/pretrained_model")
        pretrained_path = os.path.expanduser(
            "~/Desktop/checkout/pi05_2/checkpoints/075000/pretrained_model"
        )
        policy: PI05Policy = PI05Policy.from_pretrained(pretrained_path)

        cfg = RTCDemoConfig(
            policy=policy,
            robot=None,
            rtc=policy.rtc_processor.rtc_config,
            duration=1200.0,
            fps=10.0,
            device=str(device),
            task=task,
        )

        # Load config and set compile_model for pi0/pi05 models
        config = policy.config
        if policy.type == "pi05" or policy.type == "pi0":
            policy.config.compile_model = cfg.use_torch_compile
        policy.pretrained_path = pretrained_path

        # Enable debug tracking
        cfg.rtc.debug = True
        cfg.rtc.debug_maxlen = 100

        # Turn on RTC
        policy.config.rtc_config = cfg.rtc

        # Init RTC processort, as by default if RTC disabled in the config
        # The processor won't be created
        policy.init_rtc_processor()

        assert policy.name in [
            "smolvla",
            "pi05",
            "pi0",
        ], "Only smolvla, pi05, and pi0 are supported for RTC"

        policy = policy.to(cfg.device)
        policy.eval()

        action_features = hw_to_dataset_features(robot.robot.action_features, "action")
        obs_features = hw_to_dataset_features(robot.robot.observation_features, "observation")
        dataset_features = {**action_features, **obs_features}

        preprocess, postprocess = make_pre_post_processors(
            policy.config,
            pretrained_path,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )

        robot_type = robot.robot.name if hasattr(robot.robot, "name") else ""

        subtasks = [
            "Move to home position.",
            "Orienting above the purple tube.",
            "Grasp the purple tube and pull it out.",
            "Move to goal rack and insert the tube.",
        ]

        # Setup signal handler for graceful shutdown
        signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
        shutdown_event = signal_handler.shutdown_event

        # Apply torch.compile to predict_action_chunk method if enabled
        if cfg.use_torch_compile:
            policy = _apply_torch_compile(policy, cfg)

        # Create robot observation processor
        robot_observation_processor = make_default_robot_observation_processor()
        robot_action_processor = make_default_robot_action_processor()

        # Create action queue for communication between threads
        action_queue = ActionQueue(cfg.rtc)

        subtasks = [
            "Move to home position.",
            "Orienting above the purple tube.",
            "Grasp the purple tube and pull it out.",
            "Move to goal rack and insert the tube.",
        ]

        # instructor_thread = robot.robot.create_timer(5.0, lambda: instructor_query(robot.robot, subtasks))
        instructor_thread = robot.robot.create_timer(
            5.0, lambda: robot.logger.info(f"Current task: {robot.robot.task}")
        )

        # Start chunk requester thread
        get_actions_thread = Thread(
            target=get_actions,
            args=(policy, robot, robot_observation_processor, action_queue, shutdown_event, cfg),
            daemon=True,
            name="GetActions",
        )
        get_actions_thread.start()
        robot.logger.info("Started get actions thread")

        # Start action executor thread
        actor_thread = Thread(
            target=actor_control,
            args=(robot, robot_action_processor, action_queue, shutdown_event, cfg),
            daemon=True,
            name="Actor",
        )
        actor_thread.start()
        robot.logger.info("Started actor thread")

        robot.logger.info("Started stop by duration thread")

        # Main thread monitors for duration or shutdown
        robot.logger.info(f"Running demo for {cfg.duration} seconds...")
        start_time = time.time()

        robot.reset()

        while not shutdown_event.is_set() and (time.time() - start_time) < cfg.duration:
            time.sleep(10)

            # Log queue status periodically
            if int(time.time() - start_time) % 5 == 0:
                robot.logger.info(f"[MAIN] Action queue size: {action_queue.qsize()}")

            if time.time() - start_time > cfg.duration:
                break

        robot.logger.info("Demo duration reached or shutdown requested")

        # Signal shutdown
        shutdown_event.set()

        # Wait for threads to finish
        if get_actions_thread and get_actions_thread.is_alive():
            robot.logger.info("Waiting for chunk requester thread to finish...")
            get_actions_thread.join()

        if actor_thread and actor_thread.is_alive():
            robot.logger.info("Waiting for action executor thread to finish...")
            actor_thread.join()

        # instructor_thread.cancel()

        robot.logger.info("Cleanup completed")

    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            robot.robot.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()
