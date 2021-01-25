import utils
import argparse
import pathlib
from collections import namedtuple
import itertools
import math
parser = argparse.ArgumentParser(description="Generate run descriptions")
parser.add_argument("base_dir", type=str,
                    help="Base directory for run descriptions")

EPOCHS = 400
GN_EPOCHS = 25
NUM_REPEATS = 3
# Spring base parameters
SPRING_END_TIME = 2 * math.pi
SPRING_DT = 0.3 / 100
SPRING_STEPS = math.ceil(SPRING_END_TIME / SPRING_DT)

writable_objects = []

experiment = utils.Experiment("springmesh-test")
mesh_gen = utils.SpringMeshGridGenerator(grid_shape=(3, 3), fix_particles="top")
init_cond = utils.SpringMeshRowPerturb(mesh_generator=mesh_gen, magnitude=0.25, row=0)
dset = utils.SpringMeshDataset(experiment, init_cond, 1,
                               set_type="train",
                               num_time_steps=SPRING_STEPS, time_step_size=SPRING_DT,
                               subsampling=1, noise_sigma=0.0, vel_decay=0.9)
print("INPUT_SIZE", dset.input_size())
writable_objects.append(dset)

euler_int = utils.BaselineIntegrator(experiment, eval_set=dset, integrator="euler")
leapfrog_int = utils.BaselineIntegrator(experiment, eval_set=dset, integrator="leapfrog")
rk4_int = utils.BaselineIntegrator(experiment, eval_set=dset, integrator="rk4")
writable_objects.extend([euler_int, leapfrog_int, rk4_int])

if __name__ == "__main__":
    args = parser.parse_args()
    base_dir = pathlib.Path(args.base_dir)
    for obj in writable_objects:
        obj.write_description(base_dir)
