#!/usr/bin/env python
"""Write the committed state-only LeRobot dataset fixture using upstream itself.

Run against upstream `lerobot` at f37be3edbee60f3a09a5183788b91eb19f0c07d1. See
`tools/goldens/README.md`. Not part of `cargo test`.

The point of going through `LeRobotDataset.create` rather than writing parquet by
hand is that the fixture is then upstream's on-disk format by construction --
column names, arrow types (`fixed_size_list<float>[2]` for the state features,
`float` for `timestamp`, `int64` for the four index columns), the `meta/stats.json`
statistics, the `meta/tasks.parquet` index column, and the episode table's
`dataset_from_index` / `dataset_to_index` boundaries all come from upstream's
writer. Rerobot's reader is tested against that, not against a guess at it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ID = "rerobot/state_only_slice"
FPS = 10
OUT = Path(__file__).resolve().parents[2] / "crates/rerobot-train/tests/fixtures/state_only"

FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
    "observation.environment_state": {"dtype": "float32", "shape": (2,), "names": ["gx", "gy"]},
    "action": {"dtype": "float32", "shape": (2,), "names": ["dx", "dy"]},
}

# One episode, four frames. Chosen so that every number is exactly representable
# in binary32 and the per-feature statistics are exact too, which keeps the
# normalization golden readable.
FRAMES = [
    ([0.0, 1.0], [10.0, -1.0], [0.5, -0.5]),
    ([0.25, 0.75], [11.0, -2.0], [0.25, -0.25]),
    ([0.5, 0.5], [12.0, -3.0], [0.0, 0.0]),
    ([1.0, 0.0], [13.0, -4.0], [-0.5, 0.5]),
]
TASK = "reach the target"


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID, fps=FPS, root=OUT, features=FEATURES, use_videos=False
    )
    for state, env_state, action in FRAMES:
        dataset.add_frame(
            {
                "observation.state": np.array(state, dtype=np.float32),
                "observation.environment_state": np.array(env_state, dtype=np.float32),
                "action": np.array(action, dtype=np.float32),
                "task": TASK,
            }
        )
    dataset.save_episode()
    dataset.finalize()

    written = sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file())
    expected = [
        "data/chunk-000/file-000.parquet",
        "meta/episodes/chunk-000/file-000.parquet",
        "meta/info.json",
        "meta/stats.json",
        "meta/tasks.parquet",
    ]
    if written != expected:
        raise SystemExit(f"unexpected fixture layout: {written}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
