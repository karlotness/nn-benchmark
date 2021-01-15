import utils
import argparse
import pathlib
from collections import namedtuple
import itertools
import math
parser = argparse.ArgumentParser(description="Generate run descriptions")
parser.add_argument("base_dir", type=str,
                    help="Base directory for run descriptions")

writable_objects = []

experiment = utils.Experiment("springmesh-test")
mesh_gen = utils.SpringMeshGridGenerator(grid_shape=(5, 5))
init_cond = utils.SpringMeshManualPerturb(mesh_gen, [((2, 2), (-0.2, 0.2))])
dset = utils.SpringMeshDataset(experiment, init_cond, 1,
                               set_type="train",
                               num_time_steps=500, time_step_size=0.1,
                               subsampling=10, noise_sigma=0.0, vel_decay=0.8)
print("INPUT_SIZE", dset.input_size())
writable_objects.append(dset)

if __name__ == "__main__":
    args = parser.parse_args()
    base_dir = pathlib.Path(args.base_dir)
    for obj in writable_objects:
        obj.write_description(base_dir)
