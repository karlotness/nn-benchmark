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


class SnapshotDataset(data.Dataset):
    Snapshot = namedtuple("Snapshot", ["name", "p", "q", "dp_dt", "dq_dt",
                                       "t", "trajectory_meta"])

    def __init__(self, traj_dataset):
        super().__init__()
        self._traj_dataset = traj_dataset

        self.system = self._traj_dataset.system
        self.system_metadata = self._traj_dataset.system_metadata

        name = []
        p = []
        q = []
        dp_dt = []
        dq_dt = []
        t = []
        traj_meta = []

        for traj_i in range(len(self._traj_dataset)):
            traj = self._traj_dataset[traj_i]
            # Stack the components
            traj_num_steps = traj.p.shape[0]
            name.extend([traj.name] * traj_num_steps)
            p.append(traj.p)
            q.append(traj.q)
            dp_dt.append(traj.dp_dt)
            dq_dt.append(traj.dq_dt)
            t.append(traj.t)
            traj_meta.extend([traj.trajectory_meta] * traj_num_steps)

        # Load each trajectory and join the components
        self._name = name
        self._p = np.concatenate(p)
        self._q = np.concatenate(q)
        self._dp_dt = np.concatenate(dp_dt)
        self._dq_dt = np.concatenate(dq_dt)
        self._t = np.concatenate(t)
        self._traj_meta = traj_meta

    def __getitem__(self, idx):
        return self.Snapshot(name=self._name[idx],
                             trajectory_meta=self._traj_meta[idx],
                             p=self._p[idx], q=self._q[idx],
                             dp_dt=self._dp_dt[idx], dq_dt=self._dq_dt[idx],
                             t=self._t[idx])

    def __len__(self):
        return len(self._traj_meta)


# TODO: Other dataset wrappers for later (wrap a TrajectoryDataset) to select:
# - Rollout-sized chunks
