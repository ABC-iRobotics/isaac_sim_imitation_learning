#!/usr/bin/env python3
"""Slice observation.state/action down to their joint-space dimensions.

The GUIDE recorder writes 14-dim state/action: 6 cartesian velocity dims plus the
7 arm joints and the gripper. A joint-space policy only wants the latter 8.
lerobot-edit-dataset can drop a whole feature but not slice dimensions out of
one, so we rewrite the parquet columns and every piece of metadata that mirrors
their width. Videos and tasks are copied untouched.

    python scripts/slice_joint_space.py SRC DST
    python scripts/slice_joint_space.py SRC DST --keep '^joint\\d+\\.pos$'   # no gripper

Indices come from info.json's `names`, not a hardcoded slice: the recorder's
config lists the joints first but writes them last, so position is not reliable.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FEATURES = ("observation.state", "action")
DEFAULT_KEEP = r"^(joint\d+|gripper)\.pos$"


def _slice_fixed_list(col, idx):
    """Slice a fixed_size_list<float>[N] column down to the kept indices."""
    arr = np.stack(col.to_numpy(zero_copy_only=False))[:, idx]
    values = pa.array(arr.ravel(), type=col.type.value_type)
    return pa.FixedSizeListArray.from_arrays(values, len(idx))


def _slice_var_list(col, idx):
    """Slice a variable-length list column (per-episode stats) row by row."""
    return pa.array([[row[i] for i in idx] for row in col.to_pylist()], type=col.type)


def _patch_hf_metadata(metadata, n):
    """Narrow the `length` HF datasets records alongside the parquet schema."""
    md = dict(metadata or {})
    hf = json.loads(md[b"huggingface"])
    for name in FEATURES:
        hf["info"]["features"][name]["length"] = n
    hf.pop("fingerprint", None)  # no longer describes the data
    md[b"huggingface"] = json.dumps(hf).encode()
    return md


def slice_dataset(src: Path, dst: Path, keep: str) -> list[str]:
    info = json.loads((src / "meta" / "info.json").read_text())
    names = info["features"][FEATURES[0]]["names"]
    for name in FEATURES:
        if info["features"][name]["names"] != names:
            raise SystemExit(f"{name} and {FEATURES[0]} have different names; refusing to guess")

    idx = [i for i, n in enumerate(names) if re.match(keep, n)]
    if not idx:
        raise SystemExit(f"--keep {keep!r} matched none of: {names}")
    if len(idx) == len(names):
        raise SystemExit(f"--keep {keep!r} matched every dimension; nothing to slice")

    shutil.copytree(src, dst)

    for path in sorted((dst / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        for name in FEATURES:
            i = table.schema.get_field_index(name)
            table = table.set_column(i, name, _slice_fixed_list(table[name], idx))
        table = table.replace_schema_metadata(_patch_hf_metadata(table.schema.metadata, len(idx)))
        pq.write_table(table, path, compression="snappy")

    # Per-episode stats mirror the feature width, except /count which stays scalar.
    prefixes = tuple(f"stats/{name}/" for name in FEATURES)
    for path in sorted((dst / "meta" / "episodes").rglob("*.parquet")):
        table = pq.read_table(path)
        for i, name in enumerate(table.schema.names):
            if name.startswith(prefixes) and not name.endswith("/count"):
                table = table.set_column(i, name, _slice_var_list(table[name], idx))
        pq.write_table(table, path, compression="snappy")

    kept = [names[i] for i in idx]
    for name in FEATURES:
        info["features"][name]["shape"] = [len(idx)]
        info["features"][name]["names"] = kept
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4) + "\n")

    stats_path = dst / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    for name in FEATURES:
        stats[name] = {
            k: ([v[i] for i in idx] if len(v) == len(names) else v) for k, v in stats[name].items()
        }
    stats_path.write_text(json.dumps(stats, indent=4) + "\n")

    return kept


def verify(dst: Path, kept: list[str]) -> None:
    """Load the result through lerobot itself — the check that proves it trains.

    Reads frames off hf_dataset rather than ds[0] so this stays about the slice:
    ds[0] would also decode video, which needs a working ffmpeg/torchcodec.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(dst.name, root=dst)
    n = len(kept)
    frame = ds.hf_dataset[0]
    # lerobot drops stats/* when loading meta.episodes, so check that parquet directly.
    ep_stats = pq.read_table(next((dst / "meta" / "episodes").rglob("*.parquet")))
    for name in FEATURES:
        assert tuple(ds.meta.features[name]["shape"]) == (n,), ds.meta.features[name]
        assert ds.meta.features[name]["names"] == kept, ds.meta.features[name]
        assert ds.meta.stats[name]["mean"].shape == (n,), ds.meta.stats[name]["mean"].shape
        assert len(frame[name]) == n, len(frame[name])
        assert len(ep_stats[f"stats/{name}/mean"][0]) == n, ep_stats[f"stats/{name}/mean"][0]
    print(f"verified: {len(ds)} frames, {ds.meta.total_episodes} episodes, {n} dims {kept}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    p.add_argument("--keep", default=DEFAULT_KEEP, help=f"regex over feature names (default: {DEFAULT_KEEP})")
    args = p.parse_args()

    if args.dst.exists():
        raise SystemExit(f"{args.dst} already exists")

    kept = slice_dataset(args.src, args.dst, args.keep)
    verify(args.dst, kept)


if __name__ == "__main__":
    main()
