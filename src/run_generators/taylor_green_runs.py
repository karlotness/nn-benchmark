import utils
import argparse
import pathlib
from collections import namedtuple
import itertools
import math

parser = argparse.ArgumentParser(description="Generate run descriptions")
parser.add_argument("base_dir", type=str,
                    help="Base directory for run descriptions")

N_GRID = 10

writable_objects = []

experiment = utils.Experiment("experiment")

source = utils.TaylorGreenInitialConditionSource(utils.TaylorGreenGridGenerator((N_GRID, N_GRID)))

dataset = utils.TaylorGreenDataset(experiment, source, 10, n_grid=N_GRID)

gn_train = utils.GN(experiment=experiment, training_set=dataset, validation_set=dataset, epochs=10, batch_size=20)

gn_eval = utils.NetworkEvaluation(experiment=experiment, network=gn_train, eval_set=dataset, integrator="null")

writable_objects.extend([dataset, gn_train, gn_eval])


if __name__ == "__main__":
    args = parser.parse_args()
    base_dir = pathlib.Path(args.base_dir)
    for obj in writable_objects:
        obj.write_description(base_dir)
