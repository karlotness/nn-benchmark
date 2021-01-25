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
# Wave base parameters
WAVE_END_TIME = 5
WAVE_DT = 0.1 / 250
WAVE_STEPS = math.ceil(WAVE_END_TIME / WAVE_DT)
WAVE_SUBSAMPLE = 1000 // 250
WAVE_N_GRID = 125

DT_FACTORS = [1, 2, 4, 8, 10, 16]
NUM_TRAIN_TRAJS = [10, 25, 50, 75, 100, 500]
ARCHITECTURES = [(200, 3), (2048, 2)]

experiment_general = utils.Experiment("wave-runs")
experiment_noise = utils.Experiment("wave-runs-noise")
experiment_long = utils.Experiment("wave-runs-long")
experiment_outdist = utils.Experiment("wave-runs-outdist")
experiment_easy = utils.Experiment("wave-runs-easy")
writable_objects = []

train_source = utils.WaveInitialConditionSource(height_range=(0.75, 1.25),
                                                width_range=(0.75, 1.25),
                                                position_range=(0.5, 0.5))
val_source = utils.WaveInitialConditionSource(height_range=(0.75, 1.25),
                                              width_range=(0.75, 1.25),
                                              position_range=(0.5, 0.5))
eval_source = utils.WaveInitialConditionSource(height_range=(0.75, 1.25),
                                               width_range=(0.75, 1.25),
                                               position_range=(0.5, 0.5))
eval_outdist_source = utils.WaveDisjointInitialConditionSource(
    height_range=[(0.5, 0.75), (1.25, 1.5)],
    width_range=[(0.5, 0.75), (1.25, 1.5)],
    position_range=[(0.5, 0.5)])

train_source_easy = utils.WaveInitialConditionSource(height_range=(0.95, 1.05),
                                                     width_range=(0.95, 1.05),
                                                     position_range=(0.5, 0.5))
val_source_easy = utils.WaveInitialConditionSource(height_range=(0.95, 1.05),
                                                   width_range=(0.95, 1.05),
                                                   position_range=(0.5, 0.5))
eval_source_easy = utils.WaveInitialConditionSource(height_range=(0.95, 1.05),
                                                    width_range=(0.95, 1.05),
                                                    position_range=(0.5, 0.5))
data_sets = {}

DatasetKey = namedtuple("DatasetKey", ["type", "dt_factor", "n_traj"])

# Generate data sets
# Generate data sets
for dt_factor in DT_FACTORS:
    time_step_size = WAVE_DT * dt_factor
    num_time_steps = math.ceil(WAVE_END_TIME / time_step_size)
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
    key = DatasetKey(type="eval-outdist", dt_factor=dt_factor, n_traj=6)
    dset = utils.WaveDataset(experiment=experiment_outdist,
                             initial_cond_source=eval_outdist_source,
                             num_traj=6,
                             set_type=f"eval-outdist-dtfactor{dt_factor}",
                             n_grid=WAVE_N_GRID,
                             num_time_steps=num_time_steps,
                             time_step_size=time_step_size,
                             wave_speed=0.1,
                             subsampling=wave_subsample)
    data_sets[key] = dset
    if dt_factor == 1:
        # Generate the HNN-only long term eval sets
        key = DatasetKey(type="eval-long", dt_factor=dt_factor, n_traj=3)
        dset = utils.WaveDataset(experiment=experiment_long,
                                 initial_cond_source=eval_source,
                                 num_traj=3,
                                 set_type=f"eval-longterm-dtfactor{dt_factor}",
                                 n_grid=WAVE_N_GRID,
                                 num_time_steps=3 * num_time_steps,
                                 time_step_size=time_step_size,
                                 wave_speed=0.1,
                                 subsampling=wave_subsample)
        data_sets[key] = dset
        key = DatasetKey(type="eval-easy", dt_factor=dt_factor, n_traj=6)
        dset = utils.WaveDataset(experiment=experiment_easy,
                                 initial_cond_source=eval_source_easy,
                                 num_traj=6,
                                 set_type=f"eval-easy-dtfactor{dt_factor}",
                                 n_grid=WAVE_N_GRID,
                                 num_time_steps=num_time_steps,
                                 time_step_size=time_step_size,
                                 wave_speed=0.1,
                                 subsampling=wave_subsample)
        data_sets[key] = dset
        key = DatasetKey(type="val-easy", dt_factor=dt_factor, n_traj=3)
        dset = utils.WaveDataset(experiment=experiment_easy,
                                 initial_cond_source=val_source_easy,
                                 num_traj=3,
                                 set_type=f"val-easy-dtfactor{dt_factor}",
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
        if dt_factor == 1:
            key = DatasetKey(type="train-easy", dt_factor=dt_factor, n_traj=num_traj)
            dset = utils.WaveDataset(experiment=experiment_easy,
                                     initial_cond_source=train_source_easy,
                                     num_traj=num_traj,
                                     set_type=f"train-dtfactor{dt_factor}",
                                     n_grid=WAVE_N_GRID,
                                     num_time_steps=num_time_steps,
                                     time_step_size=time_step_size,
                                     wave_speed=0.1,
                                     subsampling=wave_subsample)
            data_sets[key] = dset
writable_objects.extend(data_sets.values())

# Emit baseline integrator runs for each evaluation set
for key, dset in data_sets.items():
    if key.type not in {"eval", "eval-long", "eval-outdist", "eval-easy"}:
        continue
    for integrator in ["leapfrog", "euler", "rk4",
                       "back-euler", "implicit-rk"]:
        integration_run_float = utils.BaselineIntegrator(experiment=experiment_general,
                                                         eval_set=dset,
                                                         eval_dtype="float",
                                                         integrator=integrator)
        integration_run_double = utils.BaselineIntegrator(experiment=experiment_general,
                                                          eval_set=dset,
                                                          eval_dtype="double",
                                                          integrator=integrator)
        writable_objects.append(integration_run_float)
        writable_objects.append(integration_run_double)

# Emit KNN baselines
for dt_factor in DT_FACTORS:
    for num_traj in NUM_TRAIN_TRAJS:
        train_set_key = DatasetKey(type="train", dt_factor=dt_factor, n_traj=num_traj)
        eval_set_key = DatasetKey(type="eval", dt_factor=dt_factor, n_traj=6)
        eval_set_outdist_key = DatasetKey(type="eval-outdist", dt_factor=dt_factor, n_traj=6)
        train_set = data_sets[train_set_key]
        eval_set = data_sets[eval_set_key]
        eval_set_outdist = data_sets[eval_set_outdist_key]
        if dt_factor == 1:
            eval_set_long_key = DatasetKey(type="eval-long", dt_factor=1, n_traj=3)
            eval_set_long = data_sets[eval_set_long_key]
            train_set_easy_key = DatasetKey(type="train-easy", dt_factor=dt_factor, n_traj=num_traj)
            eval_set_easy_key = DatasetKey(type="eval-easy", dt_factor=dt_factor, n_traj=6)
            train_set_easy = data_sets[train_set_easy_key]
            eval_set_easy = data_sets[eval_set_easy_key]
            knn_predict = utils.KNNPredictorOneshot(experiment_long, training_set=train_set, eval_set=eval_set_long)
            knn_predict_eval_easy = utils.KNNPredictorOneshot(experiment_easy, training_set=train_set_easy, eval_set=eval_set_easy)
            writable_objects.extend([knn_predict, knn_predict_eval_easy])
            for eval_integrator in ["leapfrog", "euler", "rk4"]:
                knn_regressor = utils.KNNRegressorOneshot(experiment_long, training_set=train_set, eval_set=eval_set_long, integrator=eval_integrator)
                knn_regressor_easy = utils.KNNRegressorOneshot(experiment_easy, training_set=train_set_easy, eval_set=eval_set_easy, integrator=eval_integrator)
                writable_objects.extend([knn_regressor, knn_regressor_easy])
        # Write other KNNs
        knn_predict_eval = utils.KNNPredictorOneshot(experiment_general, training_set=train_set, eval_set=eval_set)
        knn_predict_eval_outdist = utils.KNNPredictorOneshot(experiment_outdist, training_set=train_set, eval_set=eval_set_outdist)
        writable_objects.extend([knn_predict_eval, knn_predict_eval_outdist])
        for eval_integrator in ["leapfrog", "euler", "rk4"]:
            knn_regressor = utils.KNNRegressorOneshot(experiment_general, training_set=train_set, eval_set=eval_set, integrator=eval_integrator)
            knn_regressor_outdist = utils.KNNRegressorOneshot(experiment_outdist, training_set=train_set, eval_set=eval_set_outdist, integrator=eval_integrator)
            writable_objects.extend([knn_regressor, knn_regressor_outdist])

# EMIT MLP and NNKERNEL RUNS
for num_traj in NUM_TRAIN_TRAJS:
    train_set_key = DatasetKey(type="train", dt_factor=1, n_traj=num_traj)
    train_set_easy_key = DatasetKey(type="train-easy", dt_factor=1, n_traj=num_traj)
    val_set_key = DatasetKey(type="val", dt_factor=1, n_traj=3)
    val_set_easy_key = DatasetKey(type="val-easy", dt_factor=1, n_traj=3)
    eval_set_key = DatasetKey(type="eval", dt_factor=1, n_traj=6)
    eval_set_easy_key = DatasetKey(type="eval-easy", dt_factor=1, n_traj=6)
    eval_set_outdist_key = DatasetKey(type="eval-outdist", dt_factor=1, n_traj=6)
    eval_set_long_key = DatasetKey(type="eval-long", dt_factor=1, n_traj=3)

    train_set = data_sets[train_set_key]
    train_set_easy = data_sets[train_set_easy_key]
    val_set = data_sets[val_set_key]
    val_set_easy = data_sets[val_set_easy_key]
    eval_set = data_sets[eval_set_key]
    eval_set_easy = data_sets[eval_set_easy_key]
    eval_set_outdist = data_sets[eval_set_outdist_key]
    eval_set_long = data_sets[eval_set_long_key]

    trained_nets = []
    trained_nets_easy = []
    for _repeat in range(NUM_REPEATS):
        # Train the networks
        nn_kern_train = utils.NNKernel(experiment=experiment_general,
                                       training_set=train_set,
                                       learning_rate=0.001, weight_decay=0.0001,
                                       hidden_dim=32768, train_dtype="float",
                                       optimizer="sgd",
                                       batch_size=750, epochs=EPOCHS, validation_set=val_set,
                                       nonlinearity="relu")
        nn_kern_train_easy = utils.NNKernel(experiment=experiment_easy,
                                            training_set=train_set_easy,
                                            learning_rate=0.001, weight_decay=0.0001,
                                            hidden_dim=32768, train_dtype="float",
                                            optimizer="sgd",
                                            batch_size=750, epochs=EPOCHS, validation_set=val_set_easy,
                                            nonlinearity="relu")
        trained_nets.append((experiment_general, nn_kern_train))
        trained_nets_easy.append(nn_kern_train_easy)
        for width, depth in ARCHITECTURES:
            mlp_train = utils.MLP(experiment=experiment_general, training_set=train_set,
                                  hidden_dim=width, depth=depth,
                                  validation_set=val_set, epochs=EPOCHS)
            mlp_train_easy = utils.MLP(experiment=experiment_easy, training_set=train_set_easy,
                                       hidden_dim=width, depth=depth,
                                       validation_set=val_set_easy, epochs=EPOCHS)
            trained_nets.append((experiment_general, mlp_train))
            trained_nets_easy.append(mlp_train_easy)
        cnn_train = utils.CNN(experiment=experiment_general, training_set=train_set,
                              validation_set=val_set, epochs=EPOCHS,
                              chans_inout_kenel=[(None, 32, 5), (32, 64, 5), (64, None, 5)])
        cnn_train_easy = utils.CNN(experiment=experiment_easy, training_set=train_set_easy,
                                   validation_set=val_set_easy, epochs=EPOCHS,
                                   chans_inout_kenel=[(None, 32, 5), (32, 64, 5), (64, None, 5)])
        gn_train = utils.GN(experiment=experiment_general,
                            training_set=train_set,
                            validation_set=val_set,
                            epochs=GN_EPOCHS)
        gn_train_easy = utils.GN(experiment=experiment_easy,
                                 training_set=train_set_easy,
                                 validation_set=val_set_easy,
                                 epochs=GN_EPOCHS)
        trained_nets.extend([(experiment_general, cnn_train),
                             (experiment_general, gn_train)])
        trained_nets_easy.extend([cnn_train_easy, gn_train_easy])
    # Evaluate the networks
    writable_objects.extend([rec for _exp, rec in trained_nets])
    writable_objects.extend(trained_nets_easy)
    for eval_integrator in ["leapfrog", "euler", "rk4"]:
        for experiment, trained_net in trained_nets:
            eval_general = utils.NetworkEvaluation(experiment=experiment,
                                                   network=trained_net,
                                                   eval_set=eval_set,
                                                   integrator=eval_integrator)
            eval_outdist = utils.NetworkEvaluation(experiment=experiment,
                                                   network=trained_net,
                                                   eval_set=eval_set_outdist,
                                                   integrator=eval_integrator)
            eval_long = utils.NetworkEvaluation(experiment=experiment,
                                                network=trained_net,
                                                eval_set=eval_set_long,
                                                integrator=eval_integrator)
            writable_objects.extend([eval_general, eval_outdist, eval_long])
        for trained_net_easy in trained_nets_easy:
            eval_easy = utils.NetworkEvaluation(experiment=experiment_easy,
                                                network=trained_net_easy,
                                                eval_set=eval_set_easy,
                                                integrator=eval_integrator)
            writable_objects.append(eval_easy)

if __name__ == "__main__":
    args = parser.parse_args()
    base_dir = pathlib.Path(args.base_dir)
    for obj in writable_objects:
        obj.write_description(base_dir)
