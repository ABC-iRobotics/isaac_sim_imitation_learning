import argparse
import json
import os
import random
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torchvision.transforms.functional import to_pil_image

parser = argparse.ArgumentParser(
    description="Create a VQA fine-tuning dataset from LeRobotDataset v3.0."
)
parser.add_argument(
    "--dataset", required=True, help="Hugging Face repo ID or local path to dataset directory."
)
parser.add_argument(
    "--num-samples",
    type=int,
    default=10000,
    help="Total number of samples to generate (default 10000).",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="output",
    help="Directory to save output images and JSONL files.",
)
parser.add_argument(
    "--split-ratio",
    type=float,
    default=0.9,
    help="Ratio of data to use for training (default 0.9).",
)
args = parser.parse_args()

# Load dataset
dataset_path = args.dataset
if os.path.exists(dataset_path):
    dataset_path = os.path.abspath(dataset_path)
    repo_id = os.path.basename(dataset_path)
    root_dir = dataset_path
    dataset = LeRobotDataset(repo_id=repo_id, root=root_dir, download_videos=True)
else:
    repo_id = dataset_path
    dataset = LeRobotDataset(repo_id=repo_id, download_videos=True)

QUESTION_TEXT = (
    "Given the following list of subtasks: "
    "1. Move to home position. "
    "2. Orienting above the purple tube. "
    "3. Grasp the purple tube and pull it out. "
    "4. Move to goal rack and insert the tube. "
    "The task is to move one test tube to another rack. "
    "What is the most appropriate subtask to perform next based on the current observation? "
    "If none is appropriate, answer 'Do nothing.'."
)

# Mintavételezés
all_task_indices = dataset.hf_dataset["task_index"]
unique_task_ids = sorted({int(t) for t in all_task_indices})
num_tasks = len(unique_task_ids)
samples_per_task = args.num_samples // num_tasks

task_to_indices = {task_id: [] for task_id in unique_task_ids}
for idx, task_id in enumerate(all_task_indices):
    task_to_indices[int(task_id)].append(idx)

selected_indices = []
for task_id, indices_list in task_to_indices.items():
    random.shuffle(indices_list)
    n = min(samples_per_task, len(indices_list))
    selected_indices.extend(indices_list[:n])

random.shuffle(selected_indices)

# Szétválasztás train/val részre
split_idx = int(len(selected_indices) * args.split_ratio)
split_data = {"train": selected_indices[:split_idx], "val": selected_indices[split_idx:]}

# Mentés
output_dir = Path(args.output_dir)
for split_name, indices in split_data.items():
    split_dir = output_dir / split_name
    os.makedirs(split_dir, exist_ok=True)

    jsonl_path = split_dir / f"{split_name}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
        for count, idx in enumerate(indices, 1):
            sample = dataset[idx]
            cam_key = (
                dataset.meta.camera_keys[0]
                if hasattr(dataset.meta, "camera_keys")
                else next(k for k in sample if k.startswith("observation.images"))
            )
            img_tensor = sample[cam_key]
            img_pil = to_pil_image(img_tensor)

            img_filename = f"{count:04d}.jpg"
            img_path = split_dir / img_filename
            img_pil.save(img_path, format="JPEG")

            entry = {
                "image": str(img_path),
                "question": QUESTION_TEXT,
                "answer": str(sample["task"]),
            }
            jsonl_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Saved {len(indices)} samples to '{split_dir}'.")
