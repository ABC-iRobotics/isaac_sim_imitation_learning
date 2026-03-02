import os
import re
import shutil

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = os.path.expanduser("~/Desktop/Demonstrations/to_merge")

merged_id = "grand_dataset"
merged_root = os.path.join(root, merged_id)

if os.path.exists(merged_root) and os.path.isdir(merged_root):
    shutil.rmtree(merged_root)

dataset_ids = [id for id in os.listdir(root) if re.compile(r"^dataset_(\d+)$").match(id)]
datasets = []

for dataset_id in dataset_ids:
    print(f"Found dataset to merge: {dataset_id}")
    datasets.append(
        LeRobotDataset(repo_id=dataset_id, root=root + "/" + dataset_id, tolerance_s=0.25)
    )

dataset = merge_datasets(datasets=datasets, output_repo_id=merged_id, output_dir=merged_root)
