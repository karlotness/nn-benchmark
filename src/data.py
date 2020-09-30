import enum
import pathlib
import re
import numpy as np
import torch
from torch.utils import data


class DatasetSplit(enum.Enum):
    TRAIN = "train"
    VALIDATE = "valid"
    TEST = "test"


class TrajectoryDataset(data.Dataset):
    """Returns batches of full trajectories.
    dataset[idx] -> a set of snapshots for a full trajectory"""

    def __init__(self, data_dir: str, split: DatasetSplit):
        super().__init__()
        self.data_dir = pathlib.Path(data_dir)
        self.split = split.value
        # Determine max index for the split
        self._split_count = 0
        rgx = re.compile(f"^{self.split}\\d{{5}}$")
        for fname in self.data_dir.glob("trajectories/*.npy"):
            if rgx.match(fname.name):
                self._split_count += 1

    def __getitem__(self, idx):
        arr = np.load(self.data_dir / "trajectories" / f"{self.split}{idx:05}.npy")
        return torch.from_numpy(arr)

    def __len__(self):
        return self._split_count

    # Other data set meta can be exposed here (i.e. load from JSON or other)


# TODO: Other dataset wrappers for later (wrap a TrajectoryDataset) to select:
# - Individual samples
# - Rollout-sized chunks
