import utils
import argparse
import pathlib
from collections import namedtuple
import itertools
import math

parser = argparse.ArgumentParser(description="Generate run descriptions")
parser.add_argument("base_dir", type=str,
                    help="Base directory for run descriptions")

MLP_EPOCHS = 400
GN_EPOCHS = 25
NUM_REPEATS = 3
# Spring base parameters
SPRING_END_TIME = 2 * math.pi
SPRING_DT = 0.3 / 100
SPRING_STEPS = math.ceil(SPRING_END_TIME / SPRING_DT)

experiment_general = utils.Experiment("spring-noise-tests")
experiment_noise = utils.Experiment("spring-noise-tests-noisy")
experiment_plain = utils.Experiment("spring-noise-tests-plain")

writable_objects = []

train_source = utils.SpringInitialConditionSource(radius_range=(0.2, 1))
val_source = utils.SpringInitialConditionSource(radius_range=(0.2, 1))
eval_source = utils.SpringInitialConditionSource(radius_range=(0.2, 1))

data_sets = {}

DatasetKey = namedtuple("DatasetKey", ["type", "dt_factor", "n_traj"])

time_step_size = SPRING_DT
num_steps = SPRING_STEPS
val_set = utils.SpringDataset(experiment=experiment_general,
                              initial_cond_source=val_source,
                              num_traj=5,
                              set_type="val",
                              num_time_steps=num_steps,
                              time_step_size=time_step_size)
eval_set = utils.SpringDataset(experiment=experiment_general,
                               initial_cond_source=eval_source,
                               num_traj=10,
                               set_type="eval",
                               num_time_steps=10 * num_steps,
                               time_step_size=time_step_size)
train_set = utils.SpringDataset(experiment=experiment_general,
                                initial_cond_source=train_source,
                                num_traj=100,
                                set_type="train",
                                num_time_steps=num_steps,
                                time_step_size=time_step_size)
writable_objects.extend([val_set, eval_set, train_set])

for experiment, noise_type, variance in [(experiment_plain, "none", 0),
                                         (experiment_noise, "gn-corrected", 0.025**2)]:
    for _repeat in range(NUM_REPEATS):
        # Emit MLP runs
        for width, depth in [(200, 3), (2048, 2)]:
            mlp_train = utils.MLP(experiment=experiment, training_set=train_set,
                                  hidden_dim=width, depth=depth,
                                  validation_set=val_set, epochs=MLP_EPOCHS,
                                  noise_type=noise_type, noise_variance=variance)
            writable_objects.append(mlp_train)
            for integrator in ["leapfrog", "euler", "rk4", "scipy-RK45"]:
                eval_run = utils.NetworkEvaluation(experiment=experiment,
                                                   network=mlp_train,
                                                   eval_set=eval_set,
                                                   integrator=integrator)
                writable_objects.append(eval_run)
        # Emit GN runs
        gn_train = utils.GN(experiment=experiment,
                            training_set=train_set,
                            validation_set=val_set,
                            epochs=GN_EPOCHS,
                            noise_type=noise_type, noise_variance=variance)
        writable_objects.append(gn_train)
        eval_run = utils.NetworkEvaluation(experiment=experiment,
                                           network=gn_train,
                                           eval_set=eval_set)
        writable_objects.append(eval_run)

# Baselines
for eval_integrator in ["leapfrog", "euler", "rk4", "scipy-RK45", "back-euler", "implicit-rk"]:
    baseline = utils.BaselineIntegrator(experiment=experiment_general,
                                        eval_set=eval_set,
                                        integrator=eval_integrator)
    writable_objects.append(baseline)

if __name__ == "__main__":
    args = parser.parse_args()
    base_dir = pathlib.Path(args.base_dir)
    for obj in writable_objects:
        obj.write_description(base_dir)
