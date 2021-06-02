import utils
import argparse
import pathlib
from collections import namedtuple
import itertools
import math
parser = argparse.ArgumentParser(description="Generate run descriptions")
parser.add_argument("base_dir", type=str,
                    help="Base directory for run descriptions")

args = parser.parse_args()
base_dir = pathlib.Path(args.base_dir)

EPOCHS = 800
NUM_REPEATS = 1
# Navier-Stokes base parameters
NS_END_TIME = 0.08 * 65
NS_DT = 0.08
NS_STEPS = math.ceil(NS_END_TIME / NS_DT)
NS_SUBSAMPLE = 1
EVAL_INTEGRATORS = ["leapfrog", "euler", "rk4"]
TRAIN_NOISE_VAR = 1e-3
N_OBSTACLES = 1

COARSE_LEVELS = [1, 4, 16]  # Used for time skew parameter for training & validation
TRAIN_SET_SIZES = [25, 50, 100]

writable_objects = []

experiment_general = utils.Experiment("ns-runs")
experiment_step = utils.Experiment("ns-runs-step")
experiment_deriv = utils.Experiment("ns-runs-deriv")

train_source = utils.NavierStokesMeshInitialConditionSource(velocity_range=(1.0, 1.0), radius_range=(0.05, 0.1), n_obstacles=N_OBSTACLES)
val_source = utils.NavierStokesMeshInitialConditionSource(velocity_range=(1.0, 1.0), radius_range=(0.05, 0.1), n_obstacles=N_OBSTACLES)
eval_source = utils.NavierStokesMeshInitialConditionSource(velocity_range=(1.0, 1.0), radius_range=(0.05, 0.1), n_obstacles=N_OBSTACLES)
eval_outdist_source = utils.NavierStokesMeshInitialConditionSource(velocity_range=(1.0, 1.0), radius_range=(0.025, 0.05), n_obstacles=N_OBSTACLES)

train_sets = []
val_set = None
eval_sets = {}

# Generate data sets
# Generate train set
for num_traj in TRAIN_SET_SIZES:
    _train_set = utils.NavierStokesDataset(experiment=experiment_general,
                                           initial_cond_source=train_source,
                                           set_type="train",
                                           num_traj=num_traj,
                                           subsampling=NS_SUBSAMPLE,
                                           num_time_steps=NS_STEPS,
                                           time_step_size=NS_DT)
    train_sets.append(_train_set)
    _val_set = utils.NavierStokesDataset(experiment=experiment_general,
                                         initial_cond_source=val_source,
                                         set_type="val",
                                         num_traj=2,
                                         subsampling=NS_SUBSAMPLE,
                                         num_time_steps=NS_STEPS,
                                         time_step_size=NS_DT)
    val_set = _val_set
writable_objects.extend(train_sets)
writable_objects.append(val_set)
# Generate eval sets
for source, num_traj, type_key, step_multiplier in [
        (eval_source, 5, "eval", 1),
        #(eval_source, 3, "eval-long", 3),
        (eval_outdist_source, 5, "eval-outdist", 1),
        #(eval_outdist_source, 3, "eval-outdist-long", 3),
        ]:
    for coarse in COARSE_LEVELS:
        _ns_dt = NS_DT * coarse
        _ns_steps = round(NS_END_TIME / _ns_dt)
        _ns_subsample = NS_SUBSAMPLE * coarse
        _eval_set = utils.NavierStokesDataset(experiment=experiment_general,
                                              initial_cond_source=eval_source,
                                              set_type=f"{type_key}-cors{coarse}",
                                              num_traj=num_traj,
                                              num_time_steps=_ns_steps,
                                              subsampling=_ns_subsample,
                                              time_step_size=_ns_dt)
        _eval_set.name_tag = f"cors{coarse}"
        if coarse not in eval_sets:
            eval_sets[coarse] = []
        eval_sets[coarse].append(_eval_set)
writable_objects.extend(itertools.chain.from_iterable(eval_sets.values()))


# Emit baseline integrator runs for each evaluation set
for integrator in (EVAL_INTEGRATORS + ["back-euler", "bdf-2"]):
    for coarse in [1]: #COARSE_LEVELS:
        for eval_set in eval_sets[coarse]:
            # NO BASELINES YET
            pass
            # integration_run_double = utils.BaselineIntegrator(experiment=experiment_deriv,
            #                                                   eval_set=eval_set,
            #                                                   eval_dtype="double",
            #                                                   integrator=integrator)
            # integration_run_double.name_tag = f"cors{coarse}"
            # writable_objects.append(integration_run_double)


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
        pass
        knn_reg =  utils.KNNRegressorOneshot(experiment_deriv,
                                             training_set=train_set,
                                             eval_set=eval_set,
                                             integrator=integrator)
        writable_objects.append(knn_reg)


# DERIVATIVE: Emit MLP, GN, NNkernel runs
for train_set, _repeat in itertools.product(train_sets, range(NUM_REPEATS)):
    # Other networks work for all integrators
    general_int_nets = []
    # NN Kernel
    nn_kernel = utils.NNKernel(experiment=experiment_deriv,
                               training_set=train_set,
                               learning_rate=0.001, weight_decay=0.0001,
                               hidden_dim=32768*2, train_dtype="float",
                               optimizer="sgd",
                               predict_type="deriv",
                               batch_size=375, epochs=EPOCHS, validation_set=val_set,
                               nonlinearity="relu")
    general_int_nets.append(nn_kernel)
    # MLPs
    for width, depth in [(4096, 4), (2048, 5)]:
        mlp_deriv_train = utils.MLP(experiment=experiment_deriv,
                                    training_set=train_set,
                                    batch_size=375,
                                    hidden_dim=width, depth=depth,
                                    learning_rate=(1e-4),
                                    predict_type="deriv",
                                    noise_variance=TRAIN_NOISE_VAR,
                                    validation_set=val_set, epochs=EPOCHS)
        general_int_nets.append(mlp_deriv_train)
    # CNNs
    for cnn_arch in [
            [(None, 32, 9), (32, 32, 9), (32, 32, 9), (32, None, 9)],
            [(None, 64, 9), (64, 64, 9), (64, 64, 9), (64, None, 9)],
    ]:
        cnn_deriv_train = utils.CNN(experiment=experiment_deriv,
                                    training_set=train_set,
                                    batch_size=375,
                                    chans_inout_kenel=cnn_arch,
                                    learning_rate=(1e-4),
                                    predict_type="deriv",
                                    padding_mode="replicate",
                                    noise_variance=TRAIN_NOISE_VAR,
                                    validation_set=val_set, epochs=EPOCHS)
        general_int_nets.append(cnn_deriv_train)

    # U-Net
    unet_deriv_train = utils.UNet(
        experiment=experiment_deriv,
        training_set=train_set,
        learning_rate=0.0004,
        train_dtype="float",
        batch_size=375,
        epochs=EPOCHS,
        validation_set=val_set,
        predict_type="deriv",
        noise_variance=TRAIN_NOISE_VAR,
    )
    general_int_nets.append(unet_deriv_train)

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
    nn_kernel = utils.NNKernel(experiment=experiment_step,
                               training_set=train_set,
                               learning_rate=0.001, weight_decay=0.0001,
                               hidden_dim=32768*2, train_dtype="float",
                               optimizer="sgd",
                               predict_type="step",
                               step_time_skew=coarse, step_subsample=1,
                               batch_size=375, epochs=EPOCHS, validation_set=val_set,
                               nonlinearity="relu")
    nn_kernel.name_tag = f"cors{coarse}"
    general_int_nets.append(nn_kernel)

    for width, depth in [(4096, 4), (2048, 5)]:
        mlp_step_train = utils.MLP(experiment=experiment_step,
                                    training_set=train_set,
                                    batch_size=375,
                                    hidden_dim=width, depth=depth,
                                    learning_rate=(1e-4),
                                    predict_type="step",
                                    step_time_skew=coarse, step_subsample=1,
                                    noise_variance=TRAIN_NOISE_VAR,
                                    validation_set=val_set, epochs=EPOCHS)
        mlp_step_train.name_tag = f"cors{coarse}"
        general_int_nets.append(mlp_step_train)

    # CNNs
    for cnn_arch in [
            [(None, 32, 9), (32, 32, 9), (32, 32, 9), (32, None, 9)],
            [(None, 64, 9), (64, 64, 9), (64, 64, 9), (64, None, 9)],
    ]:
        cnn_step_train = utils.CNN(experiment=experiment_step,
                                   training_set=train_set,
                                   batch_size=375,
                                   chans_inout_kenel=cnn_arch,
                                   learning_rate=(1e-4),
                                   predict_type="step",
                                   step_time_skew=coarse, step_subsample=1,
                                   padding_mode="replicate",
                                   noise_variance=TRAIN_NOISE_VAR,
                                   validation_set=val_set, epochs=EPOCHS)
        cnn_step_train.name_tag = f"cors{coarse}"
        general_int_nets.append(cnn_step_train)

    # U-Net
    unet_step_train = utils.UNet(
        experiment=experiment_step,
        training_set=train_set,
        learning_rate=0.0004,
        train_dtype="float",
        batch_size=375,
        epochs=EPOCHS,
        validation_set=val_set,
        predict_type="step",
        step_time_skew=coarse,
        step_subsample=1,
        noise_variance=TRAIN_NOISE_VAR,
    )
    unet_step_train.name_tag = f"cors{coarse}"
    general_int_nets.append(unet_step_train)

    writable_objects.extend(general_int_nets)
    for trained_net, eval_set in itertools.product(general_int_nets, eval_sets[coarse]):
        eval_run = utils.NetworkEvaluation(experiment=experiment_step,
                                           network=trained_net,
                                           eval_set=eval_set,
                                           integrator="null")
        eval_run.name_tag = trained_net.name_tag
        writable_objects.append(eval_run)

if __name__ == "__main__":
    for obj in writable_objects:
        obj.write_description(base_dir)
