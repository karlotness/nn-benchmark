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
NUM_REPEATS = 3
# Spring base parameters
SPRING_STEPS = 1100
SPRING_DT = 0.3 / 100

experiment_general = utils.Experiment("learn-spring-rerun2")
experiment_physics = utils.Experiment("learn-physics-spring-rerun2")
experiment_integration = utils.Experiment("learn-integration-spring-rerun2")
experiment_integration_small_train = utils.Experiment("learn-integration-spring-small-train")
writable_objects = []

train_source = utils.SpringInitialConditionSource(radius_range=(0.2, 1))
val_source = utils.SpringInitialConditionSource(radius_range=(0.2, 1))
eval_source = utils.SpringInitialConditionSource(radius_range=(0.2, 1))

data_sets = {}

DatasetKey = namedtuple("DatasetKey", ["type", "dt_factor", "n_traj"])

# Generate data sets
for dt_factor in [1, 2, 4, 8, 16]:
    time_step_size = SPRING_DT * dt_factor
    # Generate eval and val sets
    key = DatasetKey(type="val", dt_factor=dt_factor, n_traj=5)
    dset = utils.SpringDataset(experiment=experiment_general,
                               initial_cond_source=val_source,
                               num_traj=5,
                               set_type=f"val-dtfactor{dt_factor}",
                               num_time_steps=SPRING_STEPS,
                               time_step_size=time_step_size)
    data_sets[key] = dset
    key = DatasetKey(type="eval", dt_factor=dt_factor, n_traj=30)
    dset = utils.SpringDataset(experiment=experiment_general,
                               initial_cond_source=eval_source,
                               num_traj=30,
                               set_type=f"eval-dtfactor{dt_factor}",
                               num_time_steps=SPRING_STEPS,
                               time_step_size=time_step_size)
    data_sets[key] = dset
    if dt_factor == 1:
        # Generate the HNN-only long term eval sets
        key = DatasetKey(type="eval-long", dt_factor=dt_factor, n_traj=5)
        dset = utils.SpringDataset(experiment=experiment_general,
                                   initial_cond_source=eval_source,
                                   num_traj=5,
                                   set_type=f"eval-longterm-dtfactor{dt_factor}",
                                   num_time_steps=3 * SPRING_STEPS,
                                   time_step_size=time_step_size)
        data_sets[key] = dset
    # Generate training sets
    for num_traj in [10, 50, 100, 500, 1000, 2500]:
        key = DatasetKey(type="train", dt_factor=dt_factor, n_traj=num_traj)
        dset = utils.SpringDataset(experiment=experiment_general,
                                   initial_cond_source=train_source,
                                   num_traj=num_traj,
                                   set_type=f"train-dtfactor{dt_factor}",
                                   num_time_steps=SPRING_STEPS,
                                   time_step_size=time_step_size)
        data_sets[key] = dset
# Output dataset generation
writable_objects.extend(data_sets.values())

# Emit baseline integrator runs for each evaluation set
for key, dset in data_sets.items():
    if key.type not in {"eval", "eval-long"}:
        continue
    for integrator in ["leapfrog", "euler", "rk4", "scipy-RK45"]:
        integration_run = utils.BaselineIntegrator(experiment=experiment_general,
                                                   eval_set=dset,
                                                   integrator=integrator)
        writable_objects.append(integration_run)

# Emit learning physics training and evaluation
for width, depth in [(200, 3), (2048, 2)]:
    for num_traj in [10, 50, 100, 500, 1000, 2500]:
        train_set_key = DatasetKey(type="train", dt_factor=1, n_traj=num_traj)
        val_set_key = DatasetKey(type="val", dt_factor=1, n_traj=5)
        eval_set_key = DatasetKey(type="eval", dt_factor=1, n_traj=30)
        eval_set_long_key = DatasetKey(type="eval-long", dt_factor=1, n_traj=5)

        train_set = data_sets[train_set_key]
        val_set = data_sets[val_set_key]
        eval_set = data_sets[eval_set_key]
        eval_set_long = data_sets[eval_set_long_key]
        for _repeat in range(NUM_REPEATS):
            # Train the networks
            hnn_train = utils.HNN(experiment=experiment_physics, training_set=train_set,
                                  hidden_dim=width, depth=depth,
                                  validation_set=val_set, epochs=EPOCHS)
            mlp_train = utils.MLP(experiment=experiment_physics, training_set=train_set,
                                  hidden_dim=width, depth=depth,
                                  validation_set=val_set, epochs=EPOCHS)
            writable_objects.extend([hnn_train, mlp_train])
            # Evaluate the networks
            for eval_integrator in ["leapfrog", "euler", "rk4", "scipy-RK45"]:
                hnn_eval = utils.NetworkEvaluation(experiment=experiment_physics,
                                                   network=hnn_train,
                                                   eval_set=eval_set,
                                                   integrator=eval_integrator)
                hnn_eval_large = utils.NetworkEvaluation(experiment=experiment_physics,
                                                         network=hnn_train,
                                                         eval_set=eval_set_long,
                                                         integrator=eval_integrator)
                mlp_eval = utils.NetworkEvaluation(experiment=experiment_physics,
                                                   network=mlp_train,
                                                   eval_set=eval_set,
                                                   integrator=eval_integrator)
                mlp_eval_large = utils.NetworkEvaluation(experiment=experiment_physics,
                                                         network=mlp_train,
                                                         eval_set=eval_set_long,
                                                         integrator=eval_integrator)
                writable_objects.extend([hnn_eval, mlp_eval,
                                         hnn_eval_large, mlp_eval_large])


# Emit learning integration training and evaluation
for dt_factor in [1, 2, 4, 8, 16]:
    for num_traj in [10, 50, 100, 500, 1000, 2500]:
        train_set_key = DatasetKey(type="train", dt_factor=dt_factor, n_traj=num_traj)
        val_set_key = DatasetKey(type="val", dt_factor=dt_factor, n_traj=5)
        eval_set_key = DatasetKey(type="eval", dt_factor=dt_factor, n_traj=30)
        train_set = data_sets[train_set_key]
        val_set = data_sets[val_set_key]
        eval_set = data_sets[eval_set_key]
        for train_integrator in ["leapfrog", "euler", "rk4"]:
            srnn_train = utils.SRNN(experiment_integration, training_set=train_set,
                                    integrator=train_integrator,
                                    hidden_dim=2048, depth=2,
                                    validation_set=val_set,
                                    epochs=EPOCHS)
            srnn_eval_match = utils.NetworkEvaluation(experiment=experiment_integration,
                                                      network=srnn_train,
                                                      eval_set=eval_set,
                                                      integrator=train_integrator,
                                                      gpu=True)
            writable_objects.extend([srnn_train, srnn_eval_match])


# Emit the small training set experiment for SRNN
# Generate special training/eval set
for dt_factor in [1, 2, 4, 8, 16]:
    time_step_size = SPRING_DT * dt_factor
    small_train_set = utils.SpringDataset(experiment=experiment_integration_small_train,
                                          initial_cond_source=train_source,
                                          num_traj=11,
                                          set_type=f"train-dtfactor{dt_factor}",
                                          num_time_steps=SPRING_STEPS,
                                          time_step_size=time_step_size)
    writable_objects.append(small_train_set)
    for train_integrator in ["leapfrog", "euler", "rk4"]:
        small_srnn_train = utils.SRNN(experiment_integration_small_train, training_set=small_train_set,
                                      integrator=train_integrator,
                                      hidden_dim=2048, depth=2,
                                      validation_set=None,
                                      epochs=EPOCHS)
        small_srnn_eval_match = utils.NetworkEvaluation(experiment=experiment_integration_small_train,
                                                        network=small_srnn_train,
                                                        eval_set=small_train_set,
                                                        integrator=train_integrator,
                                                        gpu=True)
        writable_objects.extend([small_srnn_train, small_srnn_eval_match])


if __name__ == "__main__":
    args = parser.parse_args()
    base_dir = pathlib.Path(args.base_dir)
    for obj in writable_objects:
        obj.write_description(base_dir)
