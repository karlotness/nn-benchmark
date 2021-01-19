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
mesh_gen = utils.SpringMeshGridGenerator(grid_shape=(3, 3), fix_particles="top")
init_cond = utils.SpringMeshRowPerturb(mesh_generator=mesh_gen, magnitude=0.25, row=0)
dset = utils.SpringMeshDataset(experiment, init_cond, 3,
                               set_type="train",
                               num_time_steps=1000, time_step_size=0.1,
                               subsampling=10, noise_sigma=0.0, vel_decay=0.8)
print("INPUT_SIZE", dset.input_size())
writable_objects.append(dset)

if __name__ == "__main__":
    args = parser.parse_args()
    base_dir = pathlib.Path(args.base_dir)
    for obj in writable_objects:
        obj.write_description(base_dir)
