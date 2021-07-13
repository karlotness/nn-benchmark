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
NUM_REPEATS = 3
# Wave base parameters
WAVE_END_TIME = 5
WAVE_DT = 0.00049
WAVE_STEPS = math.ceil(WAVE_END_TIME / WAVE_DT)
WAVE_SUBSAMPLE = 2**3
WAVE_N_GRID = 125
EVAL_INTEGRATORS = ["leapfrog", "euler", "rk4"]

TRAIN_SET_SIZES = [10, 25, 50]
COARSE_LEVELS = [1]

experiment_general = utils.Experiment("wave-runs")
experiment_deriv = utils.Experiment("wave-runs-deriv")
experiment_step = utils.Experiment("wave-runs-step")
experiment_coarse_int = utils.Experiment("wave-coarse-int")

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
eval_sets = {}

# Generate data sets
# Generate train set
for num_traj in TRAIN_SET_SIZES:
    _train_set = utils.WaveDataset(
        experiment=experiment_general,
        initial_cond_source=train_source,
        num_traj=num_traj,
        set_type="train",
        n_grid=WAVE_N_GRID,
        num_time_steps=WAVE_STEPS,
        time_step_size=WAVE_DT,
        wave_speed=0.1,
        subsampling=WAVE_SUBSAMPLE
    )
    train_sets.append(_train_set)
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
        #(eval_source, 3, "eval-long", 3),
        (eval_outdist_source, 6, "eval-outdist", 1),
        #(eval_outdist_source, 3, "eval-outdist-long", 3),
        ]:
    for coarse in COARSE_LEVELS:
        _wave_dt = WAVE_DT * coarse
        _wave_steps = step_multiplier * round(WAVE_END_TIME / _wave_dt)
        _wave_subsample = WAVE_SUBSAMPLE * coarse
        _eval_set = utils.WaveDataset(
            experiment=experiment_general,
            initial_cond_source=source,
            num_traj=num_traj,
            set_type=type_key,
            n_grid=WAVE_N_GRID,
            num_time_steps=_wave_steps,
            time_step_size=_wave_dt,
            wave_speed=0.1,
            subsampling=_wave_subsample,
        )
        _eval_set.name_tag = f"cors{coarse}"
        if coarse not in eval_sets:
            eval_sets[coarse] = []
        eval_sets[coarse].append(_eval_set)
writable_objects.extend(itertools.chain.from_iterable(eval_sets.values()))

# Emit baseline integrator runs for each evaluation set
for integrator in (EVAL_INTEGRATORS + ["back-euler", "bdf-2"]):
    for coarse in [1]: #COARSE_LEVELS:
        for eval_set in eval_sets[coarse]:
            integration_run_double = utils.BaselineIntegrator(experiment=experiment_deriv,
                                                              eval_set=eval_set,
                                                              eval_dtype="double",
                                                              integrator=integrator)
            integration_run_double.name_tag = f"cors{coarse}"
            writable_objects.append(integration_run_double)


# Do the evaluation coarsening
for eval_set, integrator, coarse_level in itertools.product(
        eval_sets[1],
        EVAL_INTEGRATORS,
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]):
    integration_run_double = utils.BaselineIntegrator(experiment=experiment_coarse_int,
                                                      eval_set=eval_set,
                                                      eval_dtype="double",
                                                      integrator=integrator,
                                                      coarsening=coarse_level)
    integration_run_double.name_tag = f"int-cors{coarse_level}"
    writable_objects.append(integration_run_double)


# Emit KNN runs
# First, KNN predictors
for coarse, train_set in itertools.product(COARSE_LEVELS, train_sets):
    for eval_set in eval_sets[coarse]:
        knn_pred = utils.KNNPredictorOneshot(experiment_step,
                                             training_set=train_set,
                                             eval_set=eval_set,
                                             step_time_skew=coarse,
                                             step_subsample=1)
        knn_pred.name_tag = f"cors{coarse}"
        writable_objects.append(knn_pred)

# Next, KNN regressors
for train_set, integrator in itertools.product(train_sets, EVAL_INTEGRATORS):
    for eval_set in eval_sets[1]:
        knn_reg =  utils.KNNRegressorOneshot(experiment_deriv,
                                             training_set=train_set,
                                             eval_set=eval_set,
                                             integrator=integrator)
        writable_objects.append(knn_reg)


# DERIVATIVE: Emit MLP, NNkernel runs
for train_set, _repeat in itertools.product(train_sets, range(NUM_REPEATS)):
    # Other networks work for all integrators
    general_int_nets = []
    # NN Kernel
    nn_kernel_small = utils.NNKernel(experiment=experiment_deriv,
                               training_set=train_set,
                               learning_rate=0.001, weight_decay=0.0001,
                               hidden_dim=32768, train_dtype="float",
                               optimizer="sgd",
                               predict_type="deriv",
                               batch_size=375, epochs=EPOCHS, validation_set=val_set,
                               nonlinearity="relu")
    general_int_nets.extend([nn_kernel_small])
    # MLPs
    for width, depth in [(200, 3), (2048, 2), (2048, 5)]:
        mlp_deriv_train = utils.MLP(experiment=experiment_deriv,
                                    training_set=train_set,
                                    batch_size=375,
                                    hidden_dim=width, depth=depth,
                                    learning_rate=(1e-3),
                                    predict_type="deriv",
                                    validation_set=val_set, epochs=EPOCHS)
        general_int_nets.append(mlp_deriv_train)
    # CNNs
    for cnn_arch in [
            [(None, 32, 5), (32, 32, 5), (32, 32, 5), (32, None, 5)],
            [(None, 32, 9), (32, 32, 9), (32, 32, 9), (32, None, 9)],
    ]:
        cnn_deriv_train = utils.CNN(experiment=experiment_deriv,
                                    training_set=train_set,
                                    batch_size=375,
                                    chans_inout_kenel=cnn_arch,
                                    learning_rate=(1e-3),
                                    predict_type="deriv",
                                    validation_set=val_set, epochs=EPOCHS)
        general_int_nets.append(cnn_deriv_train)

    # Eval runs
    writable_objects.extend(general_int_nets)
    for trained_net, eval_set, integrator in itertools.product(general_int_nets, eval_sets[1], EVAL_INTEGRATORS):
        eval_run = utils.NetworkEvaluation(experiment=experiment_deriv,
                                           network=trained_net,
                                           eval_set=eval_set,
                                           integrator=integrator)
        eval_run.name_tag = trained_net.name_tag
        writable_objects.append(eval_run)


# STEP: Emit MLP, NNkernel runs
for coarse, train_set, _repeat in itertools.product(COARSE_LEVELS, train_sets, range(NUM_REPEATS)):
    general_int_nets = []
    nn_kernel_small = utils.NNKernel(experiment=experiment_step,
                               training_set=train_set,
                               learning_rate=0.001, weight_decay=0.0001,
                               hidden_dim=32768, train_dtype="float",
                               optimizer="sgd",
                               predict_type="step",
                               step_time_skew=coarse, step_subsample=1,
                               batch_size=375, epochs=EPOCHS, validation_set=val_set,
                               nonlinearity="relu")
    nn_kernel_small.name_tag = f"cors{coarse}"
    general_int_nets.extend([nn_kernel_small])

    for width, depth in [(200, 3), (2048, 2), (2048, 5)]:
        mlp_step_train = utils.MLP(experiment=experiment_step,
                                    training_set=train_set,
                                    batch_size=375,
                                    hidden_dim=width, depth=depth,
                                    learning_rate=(1e-3),
                                    predict_type="step",
                                    step_time_skew=coarse, step_subsample=1,
                                    validation_set=val_set, epochs=EPOCHS)
        mlp_step_train.name_tag = f"cors{coarse}"
        general_int_nets.append(mlp_step_train)

    # CNNs
    for cnn_arch in [
            [(None, 32, 5), (32, 32, 5), (32, 32, 5), (32, None, 5)],
            [(None, 32, 9), (32, 32, 9), (32, 32, 9), (32, None, 9)],
    ]:
        cnn_step_train = utils.CNN(experiment=experiment_step,
                                   training_set=train_set,
                                   batch_size=375,
                                   chans_inout_kenel=cnn_arch,
                                   learning_rate=(1e-3),
                                   predict_type="step",
                                   step_time_skew=coarse, step_subsample=1,
                                   validation_set=val_set, epochs=EPOCHS)
        cnn_step_train.name_tag = f"cors{coarse}"
        general_int_nets.append(cnn_step_train)

    writable_objects.extend(general_int_nets)
    for trained_net, eval_set  in itertools.product(general_int_nets, eval_sets[coarse]):
        eval_run = utils.NetworkEvaluation(experiment=experiment_step,
                                           network=trained_net,
                                           eval_set=eval_set,
                                           integrator="null")
        eval_run.name_tag = trained_net.name_tag
        writable_objects.append(eval_run)


if __name__ == "__main__":
    args = parser.parse_args()
    base_dir = pathlib.Path(args.base_dir)
    for obj in writable_objects:
        obj.write_description(base_dir)
