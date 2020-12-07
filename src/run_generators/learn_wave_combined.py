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
# Wave base parameters
WAVE_DT = 0.1 / 250
WAVE_STEPS = 50 * 250
WAVE_SUBSAMPLE = 1000 // 250
WAVE_N_GRID = 125

experiment_general = utils.Experiment("learn-wave")
experiment_physics = utils.Experiment("learn-physics-wave")
experiment_integration = utils.Experiment("learn-integration-wave")
writable_objects = []

train_source = utils.WaveInitialConditionSource()
val_source = utils.WaveInitialConditionSource()
eval_source = utils.WaveInitialConditionSource()

data_sets = {}

DatasetKey = namedtuple("DatasetKey", ["type", "dt_factor", "n_traj"])

DT_FACTORS = [1, 2, 4, 8, 10, 16, 32, 64]
NUM_TRAIN_TRAJS = [10, 25, 50, 75, 100, 500]
ARCHITECTURES = [(200, 3), (2048, 2)]

# Generate data sets
for dt_factor in DT_FACTORS:
    time_step_size = WAVE_DT * dt_factor
    num_time_steps = math.ceil(WAVE_STEPS / dt_factor)
    wave_subsample = math.ceil(WAVE_SUBSAMPLE * dt_factor)
    # Generate eval and val sets
    key = DatasetKey(type="val", dt_factor=dt_factor, n_traj=3)
    dset = utils.WaveDataset(experiment=experiment_general,
                             initial_cond_source=val_source,
                             num_traj=3,
                             set_type=f"val-dtfactor{dt_factor}",
                             n_grid=WAVE_N_GRID,
                             num_time_steps=num_time_steps,
                             time_step_size=time_step_size,
                             wave_speed=0.1,
                             subsampling=wave_subsample)
    data_sets[key] = dset
    key = DatasetKey(type="eval", dt_factor=dt_factor, n_traj=6)
    dset = utils.WaveDataset(experiment=experiment_general,
                             initial_cond_source=eval_source,
                             num_traj=6,
                             set_type=f"eval-dtfactor{dt_factor}",
                             n_grid=WAVE_N_GRID,
                             num_time_steps=num_time_steps,
                             time_step_size=time_step_size,
                             wave_speed=0.1,
                             subsampling=wave_subsample)
    data_sets[key] = dset
    # Generate training sets
    for num_traj in NUM_TRAIN_TRAJS:
        key = DatasetKey(type="train", dt_factor=dt_factor, n_traj=num_traj)
        dset = utils.WaveDataset(experiment=experiment_general,
                                 initial_cond_source=train_source,
                                 num_traj=num_traj,
                                 set_type=f"train-dtfactor{dt_factor}",
                                 n_grid=WAVE_N_GRID,
                                 num_time_steps=num_time_steps,
                                 time_step_size=time_step_size,
                                 wave_speed=0.1,
                                 subsampling=wave_subsample)
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


# Emit learning physics training and evaluation (HNN and MLP)
for width, depth in ARCHITECTURES:
    for num_traj in NUM_TRAIN_TRAJS:
        train_set_key = DatasetKey(type="train", dt_factor=10, n_traj=num_traj)
        val_set_key = DatasetKey(type="val", dt_factor=10, n_traj=3)
        eval_set_key = DatasetKey(type="eval", dt_factor=1, n_traj=6)

        train_set = data_sets[train_set_key]
        val_set = data_sets[val_set_key]
        eval_set = data_sets[eval_set_key]
        for _repeat in range(NUM_REPEATS):
            # Train the networks
            hnn_train = utils.HNN(experiment=experiment_physics,
                                  training_set=train_set,
                                  hidden_dim=width, depth=depth,
                                  validation_set=val_set, epochs=EPOCHS)
            mlp_train = utils.MLP(experiment=experiment_physics,
                                  training_set=train_set,
                                  hidden_dim=width, depth=depth,
                                  validation_set=val_set, epochs=EPOCHS)
            writable_objects.extend([hnn_train, mlp_train])
            # Evaluate the networks
            for eval_integrator in ["leapfrog", "euler", "rk4", "scipy-RK45"]:
                hnn_eval = utils.NetworkEvaluation(experiment=experiment_physics,
                                                   network=hnn_train,
                                                   eval_set=eval_set,
                                                   integrator=eval_integrator)
                mlp_eval = utils.NetworkEvaluation(experiment=experiment_physics,
                                                   network=mlp_train,
                                                   eval_set=eval_set,
                                                   integrator=eval_integrator)
                writable_objects.extend([hnn_eval, mlp_eval])


# Emit learning integration training and evaluation
for dt_factor in DT_FACTORS:
    for width, depth in ARCHITECTURES:
        for num_traj in NUM_TRAIN_TRAJS:
            train_set_key = DatasetKey(type="train", dt_factor=dt_factor, n_traj=num_traj)
            val_set_key = DatasetKey(type="val", dt_factor=dt_factor, n_traj=3)
            eval_set_key = DatasetKey(type="eval", dt_factor=dt_factor, n_traj=6)
            train_set = data_sets[train_set_key]
            val_set = data_sets[val_set_key]
            eval_set = data_sets[eval_set_key]
            for train_integrator in ["leapfrog", "euler", "rk4"]:
                srnn_train = utils.SRNN(experiment_integration, training_set=train_set,
                                        integrator=train_integrator,
                                        hidden_dim=width, depth=depth,
                                        validation_set=val_set,
                                        epochs=EPOCHS)
                srnn_eval_match = utils.NetworkEvaluation(experiment=experiment_integration,
                                                          network=srnn_train,
                                                          eval_set=eval_set,
                                                          integrator=train_integrator,
                                                          gpu=True)
                writable_objects.extend([srnn_train, srnn_eval_match])


if __name__ == "__main__":
    args = parser.parse_args()
    base_dir = pathlib.Path(args.base_dir)
    for obj in writable_objects:
        obj.write_description(base_dir)
