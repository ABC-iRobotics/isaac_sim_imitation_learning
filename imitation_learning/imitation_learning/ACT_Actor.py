import os
import time
from threading import Thread

import gymnasium as gym
import IL_Gym_Env
import rclpy
import torch
from IL_Gym_Env import IL_Gym_Env
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node


def main():

    try:
        rclpy.init(args=None)

        executor = MultiThreadedExecutor()
        env_node = Node("gym_env")

        env = gym.make(id="IL_Gym_Env", max_episode_steps=10000, node=env_node)

        executor.add_node(env_node)
        env.unwrapped.logger.info("Initialized environment")

        # env.unwrapped.logger.info((str)(env.observation_space))

        time.sleep(5)
        # train_thread = Thread(target=train, args=[train_config])
        # train_config.start()
        # test = Thread(target=testEnv, args=[env])
        # test.start()

        spin_thread = Thread(target=executor.spin)
        spin_thread.start()

        device = torch.device("cuda")  # or "cuda" or "cpu"
        model_id = os.path.expanduser("~/Desktop/checkout/act_118")
        model = ACTPolicy.from_pretrained(model_id)
        # This only downloads the metadata for the dataset, ~10s of MB even for
        # large-scale datasets
        dataset_metadata = LeRobotDatasetMetadata(
            repo_id="merged_dataset",
            root=os.path.expanduser("~/Desktop/Demonstrations/merged_dataset"),
        )
        preprocess, postprocess = make_pre_post_processors(
            model.config, dataset_stats=dataset_metadata.stats
        )

        # env.unwrapped.logger.info((str)(dataset_metadata.features))
        # env.unwrapped.logger.info((str)(model.config.action_feature))
        # env.unwrapped.logger.info((str)(model.config.env_state_feature))

        MAX_EPISODES = 5
        MAX_STEPS_PER_EPISODE = 200

        for _ in range(MAX_EPISODES):
            terminated = False
            truncated = False

            obs, _ = env.reset()

            for _ in range(MAX_STEPS_PER_EPISODE):
                obs_frame = build_inference_frame(
                    observation=obs, ds_features=dataset_metadata.features, device=device
                )
                obs = preprocess(obs_frame)

                # for key, value in obs.items():
                #     env.unwrapped.logger.info((str)(key) + ': ' + (str)(value))

                dummy = torch.zeros(6, 3, 1216, 1936)
                dummy = dummy.to(device)

                obs["observation.images.cam"] = torch.cat([obs["observation.images.cam"], dummy])

                obs["observation.state"] = obs["observation.state"][0]

                action = model.select_action(obs)
                action = postprocess(action)

                # env.unwrapped.logger.info((str)(action) + '\n' + (str)(action.shape))
                for action_step in action:
                    action_step = make_robot_action(action_step, dataset_metadata.features)

                    obs, _, terminated, truncated, _ = env.step(action_step)

                    if terminated or truncated:
                        break

                env.unwrapped.logger.info("One frame executed")

            print("Episode finished! Starting new episode...")

    except (KeyboardInterrupt, ExternalShutdownException):
        pass

    finally:
        env_node.destroy_node()
        rclpy.shutdown()


def testEnv(env: IL_Gym_Env):
    obs, _ = env.reset()
    env.unwrapped.logger.info((str)(obs))


if __name__ == "__main__":
    main()
