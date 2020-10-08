import json
import pathlib
from collections import namedtuple
import numpy as np
import torch
from torch.utils import data

class TrajectoryDataset(data.Dataset):
    """Returns batches of full trajectories.
    dataset[idx] -> a set of snapshots for a full trajectory"""

    Trajectory = namedtuple("Trajectory", ["name", "p", "q", "dp_dt", "dq_dt",
                                           "t", "trajectory_meta"])

    def __init__(self, data_dir):
        super().__init__()
        data_dir = pathlib.Path(data_dir)

        with open(data_dir / "system_meta.json", "r", encoding="utf8") as meta_file:
            metadata = json.load(meta_file)
        self.system = metadata["system"]
        self.system_metadata = metadata["metadata"]
        self._trajectory_meta = metadata["trajectories"]
        self._npz_file = np.load(data_dir / "trajectories.npz")

    def __getitem__(self, idx):
        meta = self._trajectory_meta[idx]
        name = meta["name"]
        # Load arrays
        p = self._npz_file[meta["field_keys"]["p"]]
        q = self._npz_file[meta["field_keys"]["q"]]
        dp_dt = self._npz_file[meta["field_keys"]["dpdt"]]
        dq_dt = self._npz_file[meta["field_keys"]["dqdt"]]
        t = self._npz_file[meta["field_keys"]["t"]]
        # Package and return
        return self.Trajectory(name=name, trajectory_meta=meta,
                               p=p, q=q, dp_dt=dp_dt, dq_dt=dq_dt, t=t)

    def __len__(self):
        return len(self._trajectory_meta)


# TODO: Other dataset wrappers for later (wrap a TrajectoryDataset) to select:
# - Individual samples
# - Rollout-sized chunks
