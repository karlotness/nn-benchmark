import json
import pathlib
from collections import namedtuple
import numpy as np
import torch
from torch.utils import data
import math

Trajectory = namedtuple("Trajectory", ["name", "p", "q", "dp_dt", "dq_dt",
                                       "t", "trajectory_meta",
                                       "p_noiseless", "q_noiseless",
                                       "masses", "edge_index", "vertices",
                                       "fixed_mask_p", "fixed_mask_q",
                                       "extra_fixed_mask", "static_nodes",
                                       ])


class TrajectoryDataset(data.Dataset):
    """Returns batches of full trajectories.
    dataset[idx] -> a set of snapshots for a full trajectory"""

    def __init__(self, data_dir, linearize=False):
        super().__init__()
        data_dir = pathlib.Path(data_dir)

        with open(data_dir / "system_meta.json", "r", encoding="utf8") as meta_file:
            metadata = json.load(meta_file)
        self.system = metadata["system"]
        self.system_metadata = metadata["metadata"]
        self._trajectory_meta = metadata["trajectories"]
        self._npz_file = np.load(data_dir / "trajectories.npz")
        self._linearize = linearize

    def __linearize(self, arr):
        if not isinstance(arr, np.ndarray):
            return arr
        if self._linearize:
            num_steps = arr.shape[0]
            return arr.reshape((num_steps, -1))
        else:
            return arr

    def __getitem__(self, idx):
        meta = self._trajectory_meta[idx]
        name = meta["name"]
        # Load arrays
        p = self._npz_file[meta["field_keys"]["p"]]
        q = self._npz_file[meta["field_keys"]["q"]]
        dp_dt = self._npz_file[meta["field_keys"]["dpdt"]]
        dq_dt = self._npz_file[meta["field_keys"]["dqdt"]]
        t = self._npz_file[meta["field_keys"]["t"]]
        # Handle (possibly missing) noiseless data
        if "p_noiseless" in meta["field_keys"] and "q_noiseless" in meta["field_keys"]:
            # We have explicit noiseless data
            p_noiseless = self._npz_file[meta["field_keys"]["p_noiseless"]]
            q_noiseless = self._npz_file[meta["field_keys"]["q_noiseless"]]
        else:
            # Data must already be noiseless
            p_noiseless = self._npz_file[meta["field_keys"]["p"]]
            q_noiseless = self._npz_file[meta["field_keys"]["q"]]
        # Handle (possibly missing) masses
        if "masses" in meta["field_keys"]:
            masses = self._npz_file[meta["field_keys"]["masses"]]
        else:
            num_particles = p.shape[1]
            masses = np.ones(num_particles)
        if "edge_indices" in meta["field_keys"]:
            edge_index = self._npz_file[meta["field_keys"]["edge_indices"]]
            if edge_index.shape[0] != 2:
                edge_index = edge_index.T
        else:
            edge_index = []
        if "vertices" in meta["field_keys"]:
            vertices = self._npz_file[meta["field_keys"]["vertices"]]
        else:
            vertices = []


        # Handle per-trajectory boundary masks
        if "fixed_mask_p" in meta["field_keys"]:
            fixed_mask_p = np.expand_dims(self._npz_file[meta["field_keys"]["fixed_mask_p"]], 0)
        else:
            fixed_mask_p = [[]]
        if "fixed_mask_q" in meta["field_keys"]:
            fixed_mask_q = np.expand_dims(self._npz_file[meta["field_keys"]["fixed_mask_q"]], 0)
        else:
            fixed_mask_q = [[]]
        if "extra_fixed_mask" in meta["field_keys"]:
            extra_fixed_mask = np.expand_dims(self._npz_file[meta["field_keys"]["extra_fixed_mask"]], 0)
        else:
            extra_fixed_mask = [[]]
        if "enumerated_fixed_mask" in meta["field_keys"]:
            static_nodes = np.expand_dims(self._npz_file[meta["field_keys"]["enumerated_fixed_mask"]], 0)
        else:
            static_nodes = [[]]

        # Package and return
        return Trajectory(name=name, trajectory_meta=meta,
                          p=self.__linearize(p),
                          q=self.__linearize(q),
                          dp_dt=self.__linearize(dp_dt),
                          dq_dt=self.__linearize(dq_dt),
                          t=t,
                          p_noiseless=self.__linearize(p_noiseless),
                          q_noiseless=self.__linearize(q_noiseless),
                          masses=masses,
                          edge_index=edge_index,
                          vertices=vertices,
                          fixed_mask_p=self.__linearize(fixed_mask_p),
                          fixed_mask_q=self.__linearize(fixed_mask_q),
                          extra_fixed_mask=self.__linearize(extra_fixed_mask),
                          static_nodes=self.__linearize(static_nodes),
                          )

    def __len__(self):
        return len(self._trajectory_meta)


NavierStokesSnapshot = namedtuple("NavierStokesSnapshot", ["name", "p", "q", "dp_dt", "dq_dt",
                                                         "t", "trajectory_meta",
                                                         "p_noiseless", "q_noiseless",
                                                         "masses", "edge_index", "vertices",
                                                         "fixed_mask_p", "fixed_mask_q",
                                                         "extra_fixed_mask", "static_nodes",
                                                         ])


class NavierStokesSnapshotDataset(data.Dataset):
    def __init__(self, traj_dataset):
        super().__init__()
        self._traj_dataset = traj_dataset

        self.system = self._traj_dataset.system
        self.system_metadata = self._traj_dataset.system_metadata

        # NOTE: q=press, p=vels
        # NOTE: dq_dt=NEXT_PRESSURE_SNAPSHOT, dp_dt= VELOCITY DERIVATIVE
        name = []
        p = []
        q = []
        dp_dt = []
        dq_dt = []
        t = []
        traj_meta = []
        p_noiseless = []
        q_noiseless = []
        masses = []
        edge_indices = []
        vertices = []
        fixed_mask_p = []
        fixed_mask_q = []
        extra_fixed_mask = []
        static_nodes = []

        for traj_i in range(len(self._traj_dataset)):
            traj = self._traj_dataset[traj_i]
            # Stack the components
            traj_num_steps = traj.p.shape[0] - 1
            name.extend([traj.name] * traj_num_steps)
            p.append(traj.p[:-1])
            q.append(traj.q[:-1])
            dp_dt.append(traj.dp_dt[:-1])
            dq_dt.append(traj.q[1:])
            t.append(traj.t[:-1])
            traj_meta.extend([traj.trajectory_meta] * traj_num_steps)
            p_noiseless.append(traj.p_noiseless[:-1])
            q_noiseless.append(traj.q_noiseless[:-1])
            masses.extend([traj.masses] * traj_num_steps)

            # Ensure edge index has proper shape for old NS datasets
            _edge_index = traj.edge_index
            if _edge_index.shape[0] != 2:
                _edge_index = _edge_index.T

            edge_indices.extend([_edge_index] * traj_num_steps)
            vertices.extend([traj.vertices] * traj_num_steps)

            # Remove fake time dimension added above
            fixed_mask_p.extend([traj.fixed_mask_p[0]] * traj_num_steps)
            fixed_mask_q.extend([traj.fixed_mask_q[0]] * traj_num_steps)
            extra_fixed_mask.extend([traj.extra_fixed_mask[0]] * traj_num_steps)
            static_nodes.extend([traj.static_nodes[0]] * traj_num_steps)

        # Load each trajectory and join the components
        self._name = name
        self._p = np.concatenate(p)
        self._q = np.concatenate(q)
        self._dp_dt = np.concatenate(dp_dt)
        self._dq_dt = np.concatenate(dq_dt)
        self._t = np.concatenate(t)
        self._traj_meta = traj_meta
        self._p_noiseless = np.concatenate(p_noiseless)
        self._q_noiseless = np.concatenate(q_noiseless)
        self._masses = masses
        self._edge_indices = edge_indices
        self._vertices = vertices
        self._fixed_mask_p = fixed_mask_p
        self._fixed_mask_q = fixed_mask_q
        self._extra_fixed_mask = extra_fixed_mask
        self._static_nodes = static_nodes

    def __getitem__(self, idx):
        return NavierStokesSnapshot(name=self._name[idx],
                                   trajectory_meta=self._traj_meta[idx],
                                   p=self._p[idx], q=self._q[idx],
                                   dp_dt=self._dp_dt[idx], dq_dt=self._dq_dt[idx],
                                   t=self._t[idx],
                                   p_noiseless=self._p_noiseless[idx],
                                   q_noiseless=self._q_noiseless[idx],
                                   masses=self._masses[idx],
                                   edge_index=self._edge_indices[idx],
                                   vertices=self._vertices[idx],
                                   fixed_mask_p=self._fixed_mask_p[idx],
                                   fixed_mask_q=self._fixed_mask_q[idx],
                                   extra_fixed_mask=self._extra_fixed_mask[idx],
                                   static_nodes=self._static_nodes[idx],
                                )

    def __len__(self):
        return len(self._traj_meta)


Snapshot = namedtuple("Snapshot", ["name", "p", "q", "dp_dt", "dq_dt",
                                   "t", "trajectory_meta",
                                   "p_noiseless", "q_noiseless",
                                   "masses", "edge_index", "vertices",
                                   "fixed_mask_p", "fixed_mask_q",
                                   "extra_fixed_mask", "enumerated_fixed_mask",
                                   ])

class SnapshotDataset(data.Dataset):

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
        p_noiseless = []
        q_noiseless = []
        masses = []
        edge_indices = []
        vertices = []
        fixed_mask_p = []
        fixed_mask_q = []
        extra_fixed_mask = []
        enumerated_fixed_mask = []

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
            p_noiseless.append(traj.p_noiseless)
            q_noiseless.append(traj.q_noiseless)
            masses.extend([traj.masses] * traj_num_steps)
            edge_indices.extend([traj.edge_index] * traj_num_steps)
            vertices.extend([traj.vertices] * traj_num_steps)
            # Remove fake time dimension added above
            fixed_mask_p.extend([traj.fixed_mask_p[0]] * traj_num_steps)
            fixed_mask_q.extend([traj.fixed_mask_q[0]] * traj_num_steps)
            extra_fixed_mask.extend([traj.extra_fixed_mask[0]] * traj_num_steps)
            enumerated_fixed_mask.extend([traj.enumerated_fixed_mask[0]] * traj_num_steps)

        # Load each trajectory and join the components
        self._name = name
        self._p = np.concatenate(p)
        self._q = np.concatenate(q)
        self._dp_dt = np.concatenate(dp_dt)
        self._dq_dt = np.concatenate(dq_dt)
        self._t = np.concatenate(t)
        self._traj_meta = traj_meta
        self._p_noiseless = np.concatenate(p_noiseless)
        self._q_noiseless = np.concatenate(q_noiseless)
        self._masses = masses
        self._edge_indices = edge_indices
        self._vertices = vertices
        self._fixed_mask_p = fixed_mask_p
        self._fixed_mask_q = fixed_mask_q
        self._extra_fixed_mask = extra_fixed_mask
        self._enumerated_fixed_mask = enumerated_fixed_mask

    def __getitem__(self, idx):
        return Snapshot(name=self._name[idx],
                        trajectory_meta=self._traj_meta[idx],
                        p=self._p[idx], q=self._q[idx],
                        dp_dt=self._dp_dt[idx], dq_dt=self._dq_dt[idx],
                        t=self._t[idx],
                        p_noiseless=self._p_noiseless[idx],
                        q_noiseless=self._q_noiseless[idx],
                        masses=self._masses[idx],
                        edge_index=self._edge_indices[idx],
                        vertices=self._vertices[idx],
                        fixed_mask_p=self._fixed_mask_p[idx],
                        fixed_mask_q=self._fixed_mask_q[idx],
                        extra_fixed_mask=self._extra_fixed_mask[idx],
                        enumerated_fixed_mask=self._enumerated_fixed_mask[idx],
                        )

    def __len__(self):
        return len(self._traj_meta)



StepSnapshot = namedtuple("StepSnapshot",
                          ["name", "p", "q", "dp_dt", "dq_dt",
                           "p_step", "q_step",
                           "t", "trajectory_meta",
                           "p_noiseless", "q_noiseless",
                           "masses", "edge_index", "vertices",
                           "fixed_mask_p", "fixed_mask_q",
                           "extra_fixed_mask", "enumerated_fixed_mask",
                           ])


class StepSnapshotDataset(data.Dataset):

    def __init__(self, traj_dataset, subsample=1, time_skew=1):
        self._traj_dataset = traj_dataset
        self.subsample = subsample
        self.time_skew = time_skew

        self.system = self._traj_dataset.system
        self.system_metadata = self._traj_dataset.system_metadata

        name = []
        p = []
        q = []
        dp_dt = []
        dq_dt = []
        t = []
        traj_meta = []
        p_noiseless = []
        q_noiseless = []
        masses = []
        edge_indices = []
        vertices = []
        fixed_mask_p = []
        fixed_mask_q = []
        extra_fixed_mask = []
        enumerated_fixed_mask = []

        for traj_i in range(len(self._traj_dataset)):
            traj = self._traj_dataset[traj_i]
            # Stack the components
            traj_num_steps = math.ceil((traj.p.shape[0] - time_skew) / subsample)
            name.extend([traj.name] * traj_num_steps)
            p.append(traj.p[:-self.time_skew:self.subsample])
            q.append(traj.q[:-self.time_skew:self.subsample])
            dp_dt.append(traj.p[self.time_skew::self.subsample])
            dq_dt.append(traj.q[self.time_skew::self.subsample])
            t.append(traj.t[:-self.time_skew:self.subsample])
            traj_meta.extend([traj.trajectory_meta] * traj_num_steps)
            p_noiseless.append(traj.p_noiseless[:-self.time_skew:self.subsample])
            q_noiseless.append(traj.q_noiseless[:-self.time_skew:self.subsample])
            masses.extend([traj.masses] * traj_num_steps)
            edge_indices.extend([traj.edge_index] * traj_num_steps)
            vertices.extend([traj.vertices] * traj_num_steps)
            # Remove fake time dimension added above
            fixed_mask_p.extend([traj.fixed_mask_p[0]] * traj_num_steps)
            fixed_mask_q.extend([traj.fixed_mask_q[0]] * traj_num_steps)
            extra_fixed_mask.extend([traj.extra_fixed_mask[0]] * traj_num_steps)
            enumerated_fixed_mask.extend([traj.enumerated_fixed_mask[0]] * traj_num_steps)
            # Check length computation
            assert p[-1].shape[0] == traj_num_steps

        # Load each trajectory and join the components
        self._name = name
        self._p = np.concatenate(p)
        self._q = np.concatenate(q)
        self._dp_dt = np.concatenate(dp_dt)
        self._dq_dt = np.concatenate(dq_dt)
        self._t = np.concatenate(t)
        self._traj_meta = traj_meta
        self._p_noiseless = np.concatenate(p_noiseless)
        self._q_noiseless = np.concatenate(q_noiseless)
        self._masses = masses
        self._edge_indices = edge_indices
        self._vertices = vertices
        self._fixed_mask_p = fixed_mask_p
        self._fixed_mask_q = fixed_mask_q
        self._extra_fixed_mask = extra_fixed_mask
        self._enumerated_fixed_mask = enumerated_fixed_mask

    def __getitem__(self, idx):
        p_step = self._dp_dt[idx]
        q_step = self._dq_dt[idx]
        return StepSnapshot(name=self._name[idx],
                        trajectory_meta=self._traj_meta[idx],
                        p=self._p[idx], q=self._q[idx],
                        dp_dt=p_step, dq_dt=q_step,
                        p_step=p_step, q_step=q_step,
                        t=self._t[idx],
                        p_noiseless=self._p_noiseless[idx],
                        q_noiseless=self._q_noiseless[idx],
                        masses=self._masses[idx],
                        edge_index=self._edge_indices[idx],
                        vertices=self._vertices[idx],
                        fixed_mask_p=self._fixed_mask_p[idx],
                        fixed_mask_q=self._fixed_mask_q[idx],
                        extra_fixed_mask=self._extra_fixed_mask[idx],
                        enumerated_fixed_mask=self._enumerated_fixed_mask[idx],
                        )

    def __len__(self):
        return len(self._traj_meta)


Rollout = namedtuple("Rollout", ["name", "p", "q", "dp_dt", "dq_dt",
                                 "t", "trajectory_meta",
                                 "p_noiseless", "q_noiseless",
                                 "masses"])

class RolloutChunkDataset(data.Dataset):

    def __init__(self, traj_dataset, rollout_length):
        super().__init__()
        self._traj_dataset = traj_dataset

        self.system = self._traj_dataset.system
        self.system_metadata = self._traj_dataset.system_metadata
        self.rollout_length = rollout_length

        name = []
        p = []
        q = []
        dp_dt = []
        dq_dt = []
        t = []
        traj_meta = []
        p_noiseless = []
        q_noiseless = []
        masses = []

        for traj_i in range(len(self._traj_dataset)):
            traj = self._traj_dataset[traj_i]
            traj_num_steps = traj.p.shape[0]
            num_batches = traj_num_steps // self.rollout_length
            for batch in range(num_batches):
                # Slice the batches and append
                slicer = slice(batch * self.rollout_length,
                               (batch + 1) * self.rollout_length)
                name.append(traj.name)
                p.append(traj.p[slicer])
                q.append(traj.q[slicer])
                dp_dt.append(traj.dp_dt[slicer])
                dq_dt.append(traj.dq_dt[slicer])
                t.append(traj.t[slicer])
                traj_meta.append(traj.trajectory_meta)
                p_noiseless.append(traj.p_noiseless[slicer])
                q_noiseless.append(traj.q_noiseless[slicer])
                masses.append(traj.masses)

        # Join components
        self._name = name
        self._p = np.stack(p)
        self._q = np.stack(q)
        self._dp_dt = np.stack(dp_dt)
        self._dq_dt = np.stack(dq_dt)
        self._t = np.stack(t)
        self._traj_meta = traj_meta
        self._p_noiseless = np.stack(p_noiseless)
        self._q_noiseless = np.stack(q_noiseless)
        self._masses = masses

    def __getitem__(self, idx):
        return Rollout(name=self._name[idx],
                       trajectory_meta=self._traj_meta[idx],
                       p=self._p[idx], q=self._q[idx],
                       dp_dt=self._dp_dt[idx], dq_dt=self._dq_dt[idx],
                       t=self._t[idx],
                       p_noiseless=self._p_noiseless[idx],
                       q_noiseless=self._q_noiseless[idx],
                       masses=self._masses[idx])

    def __len__(self):
        return len(self._traj_meta)
