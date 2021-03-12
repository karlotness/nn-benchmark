import utils
import argparse
import pathlib
from collections import namedtuple
import itertools
import math

parser = argparse.ArgumentParser(description="Generate run descriptions")
parser.add_argument("base_dir", type=str,
                    help="Base directory for run descriptions")

EPOCHS = 250
GN_EPOCHS = 25
NUM_REPEATS = 3
# Wave base parameters
WAVE_END_TIME = 5
WAVE_DT = 0.00049
WAVE_STEPS = math.ceil(WAVE_END_TIME / WAVE_DT)
WAVE_SUBSAMPLE = 2**3
WAVE_N_GRID = 125
EVAL_INTEGRATORS = ["leapfrog", "euler", "rk4"]

experiment_general = utils.Experiment("wave-runs")

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

train_sets = []
val_set = None
eval_sets = []

# Generate data sets
# Generate train set
for num_traj in [10, 25, 50]:
    train_sets.append(
        utils.WaveDataset(experiment=experiment_general,
                          initial_cond_source=train_source,
                          num_traj=num_traj,
                          set_type="train",
                          n_grid=WAVE_N_GRID,
                          num_time_steps=WAVE_STEPS,
                          time_step_size=WAVE_DT,
                          wave_speed=0.1,
                          subsampling=WAVE_SUBSAMPLE))
writable_objects.extend(train_sets)
# Generate val set
val_set = utils.WaveDataset(experiment=experiment_general,
                            initial_cond_source=val_source,
                            num_traj=3,
                            set_type="val",
                            n_grid=WAVE_N_GRID,
                            num_time_steps=WAVE_STEPS,
                            time_step_size=WAVE_DT,
                            wave_speed=0.1,
                            subsampling=WAVE_SUBSAMPLE)
writable_objects.append(val_set)
# Generate eval sets
for source, num_traj, type_key, step_multiplier in [
        (eval_source, 6, "eval", 1),
        (eval_source, 3, "eval-long", 3),
        (eval_outdist_source, 6, "eval-outdist", 1),
        (eval_outdist_source, 3, "eval-outdist-long", 3),
        ]:
    eval_set = []
    for coarse in range(0, 11):
        wave_steps = WAVE_STEPS // (2**coarse)
        eval_set.append(
            utils.WaveDataset(experiment=experiment_general,
                              initial_cond_source=source,
                              num_traj=num_traj,
                              set_type=type_key,
                              n_grid=WAVE_N_GRID,
                              num_time_steps=(step_multiplier * wave_steps),
                              time_step_size=WAVE_DT * (2**coarse),
                              wave_speed=0.1,
                              subsampling=WAVE_SUBSAMPLE * (2**coarse)))
    eval_sets.append(eval_set)
    writable_objects.extend(eval_set)

# Emit baseline integrator runs for each evaluation set
for eval_set, integrator in itertools.product(eval_sets, (EVAL_INTEGRATORS + ["back-euler"])):
    for coarse in range(0, 11):
        integration_run_float = utils.BaselineIntegrator(experiment=experiment_general,
                                                         eval_set=eval_set[coarse],
                                                         eval_dtype="float",
                                                         integrator=integrator)
        integration_run_double = utils.BaselineIntegrator(experiment=experiment_general,
                                                          eval_set=eval_set[coarse],
                                                          eval_dtype="double",
                                                          integrator=integrator)
        writable_objects.append(integration_run_float)
        writable_objects.append(integration_run_double)

# Emit KNN baselines
for train_set, eval_set in itertools.product(train_sets, eval_sets):
    knn_pred = utils.KNNPredictorOneshot(experiment_general,
                                         training_set=train_set,
                                         eval_set=eval_set[0])
    writable_objects.append(knn_pred)
    for integrator in EVAL_INTEGRATORS:
        knn_reg = utils.KNNRegressorOneshot(experiment_general,
                                            training_set=train_set,
                                            eval_set=eval_set[0],
                                            integrator=integrator)
        writable_objects.append(knn_reg)


# Emit MLP, GN, NNkernel runs
for train_set, _repeat in itertools.product(train_sets, range(NUM_REPEATS)):
    # Only one integrator for this one, the "null" integrator
    gn_train = utils.GN(experiment=experiment_general,
                        training_set=train_set,
                        validation_set=val_set,
                        epochs=GN_EPOCHS)
    writable_objects.append(gn_train)
    for eval_set in eval_sets:
        gn_eval = utils.NetworkEvaluation(experiment=experiment_general,
                                          network=gn_train,
                                          eval_set=eval_set[0],
                                          integrator="null")
        writable_objects.append(gn_eval)
    # Other runs work across all integrators
    general_int_nets = []
    nn_kernel = utils.NNKernel(experiment=experiment_general,
                               training_set=train_set,
                               learning_rate=0.001, weight_decay=0.0001,
                               hidden_dim=32768, train_dtype="float",
                               optimizer="sgd",
                               batch_size=750, epochs=EPOCHS, validation_set=val_set,
                               nonlinearity="relu")
    general_int_nets.append(nn_kernel)
    cnn_train = utils.CNN(experiment=experiment_general,
                          training_set=train_set,
                          validation_set=val_set,
                          epochs=EPOCHS,
                          chans_inout_kenel=[(None, 32, 5), (32, 64, 5), (64, None, 5)])
    general_int_nets.append(cnn_train)
    for width, depth in [(200, 3), (2048, 2)]:
        mlp_train = utils.MLP(experiment=experiment_general,
                              training_set=train_set,
                              hidden_dim=width, depth=depth,
                              validation_set=val_set, epochs=EPOCHS)
        general_int_nets.append(mlp_train)
    writable_objects.extend(general_int_nets)
    for trained_net, eval_set, integrator in itertools.product(general_int_nets, eval_sets, EVAL_INTEGRATORS):
        writable_objects.append(
            utils.NetworkEvaluation(experiment=experiment_general,
                                    network=trained_net,
                                    eval_set=eval_set[0],
                                    integrator=integrator))

if __name__ == "__main__":
    args = parser.parse_args()
    base_dir = pathlib.Path(args.base_dir)
    for obj in writable_objects:
        obj.write_description(base_dir)
