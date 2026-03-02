"""This script demonstrates how to train ACT Policy on a real-world dataset."""

import os
import re

import torch
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]

    return [i / fps for i in delta_indices]


def find_max_checkout_number(path):
    max_number = -1
    pattern = re.compile(r"^act_(\d+)$")
    if not os.path.exists(path):
        return -1
    directory = os.path.expanduser(path)
    for name in os.listdir(directory):
        match = pattern.match(name)
        if match:
            number = int(match.group(1))
            max_number = max(max_number, number)

    return max_number


output_root = os.path.expanduser(f"~/Desktop/checkout")

last_checkout_number = find_max_checkout_number(output_root)

output_directory = os.path.join(output_root, f"act_{last_checkout_number + 1}")

os.makedirs(output_directory, exist_ok=True)

# Select your device
device = torch.device("cuda")  # or "cuda" or "cpu"


dataset_root = os.path.expanduser("~Desktop/Demonstrations")

dataset_id = "video_error_merged_dataset"
dataset_root = os.path.join(dataset_root, dataset_id)

# This specifies the inputs the model will be expecting and the outputs it will produce
# dataset_metadata = LeRobotDatasetMetadata(repo_id='merged_dataset', root=root)
dataset_metadata = LeRobotDatasetMetadata(repo_id=dataset_id, root=dataset_root)
features = dataset_to_policy_features(dataset_metadata.features)

output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
input_features = {key: ft for key, ft in features.items() if key not in output_features}

cfg = ACTConfig(input_features=input_features, output_features=output_features)
policy = ACTPolicy(cfg)
preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

policy.train()
policy.to(device)

# To perform action chunking, ACT expects a given number of actions as targets
delta_timestamps = {
    "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
}

# add image features if they are present
delta_timestamps |= {
    k: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps)
    for k in cfg.image_features
}

# Instantiate the dataset
dataset = LeRobotDataset(
    repo_id=dataset_id, root=dataset_root, tolerance_s=0.05, delta_timestamps=delta_timestamps
)

# Create the optimizer and dataloader for offline training
optimizer = cfg.get_optimizer_preset().build(policy.parameters())
batch_size = 2
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=device.type != "cpu",
    drop_last=True,
)

# Number of training steps and logging frequency
training_steps = 10_000
log_freq = 100
loss_goal = 0.1

# Run training loop
step = 0
done = False
while not done:
    for batch in dataloader:
        batch = preprocessor(batch)
        loss, _ = policy.forward(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % log_freq == 0:
            print(f"step: {step} loss: {loss.item():.3f}")
            policy.save_pretrained(output_directory)
            preprocessor.save_pretrained(output_directory)
            postprocessor.save_pretrained(output_directory)
            last_checkout_number += 1
            output_directory = os.path.join(output_root, f"act_{last_checkout_number + 1}")
        step += 1
        if step >= training_steps or loss.item() < loss_goal:
            done = True
            break

# Save the policy checkpoint, alongside the pre/post processors
policy.save_pretrained(output_directory)
preprocessor.save_pretrained(output_directory)
postprocessor.save_pretrained(output_directory)
print("Checkout export successful!")
