import pathlib
import methods
import torch
from torch.utils import data
import dataset


def run_phase(base_dir, out_dir, phase_args):
    base_dir = pathlib.Path(base_dir)
    out_dir = pathlib.Path(out_dir)
