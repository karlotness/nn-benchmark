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
mesh_size = 80

EPOCHS = 400 * 2
GN_EPOCHS = 75 * 2
NUM_REPEATS = 1
# Spring base parameters
SPRING_END_TIME = 2 * math.pi
SPRING_DT = 0.00781
SPRING_STEPS = math.ceil(SPRING_END_TIME / SPRING_DT)
VEL_DECAY = 0.1
SPRING_SUBSAMPLE = 2**7
EVAL_INTEGRATORS = ["leapfrog", "euler", "rk4"]

COARSE_LEVELS = [1, 4, 16, 64]  # Used for time skew parameter for training & validation
TRAIN_SET_SIZES = [12, 25, 50]

writable_objects = []

experiment_general = utils.Experiment(f"springmesh-{mesh_size}-perturball-runs")
experiment_step = utils.Experiment(f"springmesh-{mesh_size}-perturball-runs-step")
experiment_deriv = utils.Experiment(f"springmesh-{mesh_size}-perturball-runs-deriv")

mesh_gen = utils.SpringMeshGridGenerator(grid_shape=(mesh_size, mesh_size), fix_particles="top")


train_source = utils.SpringMeshAllPerturb(mesh_generator=mesh_gen, magnitude_range=(0, 0.35))
val_source = utils.SpringMeshAllPerturb(mesh_generator=mesh_gen, magnitude_range=(0, 0.35))
eval_source = utils.SpringMeshAllPerturb(mesh_generator=mesh_gen, magnitude_range=(0, 0.35))
eval_outdist_source = utils.SpringMeshAllPerturb(mesh_generator=mesh_gen, magnitude_range=(0.35, 0.45))

train_sets = []
val_set = None
eval_sets = {}

# Generate data sets
# Generate train set
for num_traj in TRAIN_SET_SIZES:
    train_sets.append(
        utils.SpringMeshDataset(experiment_general,
                                train_source,
                                num_traj,
                                set_type="train",
                                num_time_steps=SPRING_STEPS,
                                time_step_size=SPRING_DT,
                                subsampling=SPRING_SUBSAMPLE,
                                noise_sigma=0,
                                vel_decay=VEL_DECAY))
writable_objects.extend(train_sets)
# Generate val set
val_set = utils.SpringMeshDataset(experiment_general,
                                  val_source,
                                  3,
                                  set_type="val",
                                  num_time_steps=SPRING_STEPS,
                                  time_step_size=SPRING_DT,
                                  subsampling=SPRING_SUBSAMPLE,
                                  noise_sigma=0,
                                  vel_decay=VEL_DECAY)
writable_objects.append(val_set)
# Generate eval sets
for source, num_traj, type_key, step_multiplier in [
        (eval_source, 5, "eval", 1),
        #(eval_source, 5, "eval-long", 3),
        (eval_outdist_source, 5, "eval-outdist", 1),
        #(eval_outdist_source, 5, "eval-outdist-long", 3),
        ]:
    for coarse in COARSE_LEVELS:
        _spring_dt = SPRING_DT * coarse
        _spring_steps = math.ceil(SPRING_END_TIME / _spring_dt)
        _spring_subsample = SPRING_SUBSAMPLE * coarse
        _eval_set = utils.SpringMeshDataset(experiment_general,
                                            source,
                                            num_traj,
                                            set_type=type_key,
                                            num_time_steps=_spring_steps,
                                            time_step_size=_spring_dt,
                                            subsampling=_spring_subsample,
                                            noise_sigma=0,
                                            vel_decay=VEL_DECAY)
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


# DERIVATIVE: Emit MLP, GN, NNkernel runs
for train_set, _repeat in itertools.product(train_sets, range(NUM_REPEATS)):
    # GN runs
    gn_train = utils.GN(experiment=experiment_deriv,
                        training_set=train_set,
                        validation_set=val_set,
                        batch_size=3,
                        epochs=GN_EPOCHS)
    writable_objects.append(gn_train)
    for eval_set in eval_sets[1]:
        gn_eval = utils.NetworkEvaluation(experiment=experiment_deriv,
                                          network=gn_train,
                                          eval_set=eval_set,
                                          integrator="null")
        writable_objects.append(gn_eval)
    # Other networks work for all integrators
    general_int_nets = []
    # NN Kernel
    nn_kernel = utils.NNKernel(experiment=experiment_deriv,
                               training_set=train_set,
                               learning_rate=0.001, weight_decay=0.0001,
                               hidden_dim=32768, train_dtype="float",
                               optimizer="sgd",
                               predict_type="deriv",
                               batch_size=375, epochs=EPOCHS, validation_set=val_set,
                               nonlinearity="relu")
    general_int_nets.append(nn_kernel)
    # MLPs
    for width, depth in [(200, 3), (2048, 2), (200, 4), (4096, 4)]:
        mlp_deriv_train = utils.MLP(experiment=experiment_deriv,
                                    training_set=train_set,
                                    batch_size=375,
                                    hidden_dim=width, depth=depth,
                                    learning_rate=(1e-3/50),
                                    predict_type="deriv",
                                    validation_set=val_set, epochs=EPOCHS)
        general_int_nets.append(mlp_deriv_train)
    for cnn_arch in [
            [(None, 32, 5), (32, 32, 5), (32, 32, 5), (32, None, 5)],
            [(None, 32, 9), (32, 32, 9), (32, 32, 9), (32, None, 9)],
            [(None, 64, 9), (64, 64, 9), (64, 64, 9), (64, None, 9)],
    ]:
        cnn_deriv_train = utils.CNN(experiment=experiment_deriv,
                                    training_set=train_set,
                                    batch_size=375,
                                    chans_inout_kenel=cnn_arch,
                                    learning_rate=(1e-3/50),
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
    nn_kernel = utils.NNKernel(experiment=experiment_step,
                               training_set=train_set,
                               learning_rate=0.001, weight_decay=0.0001,
                               hidden_dim=32768, train_dtype="float",
                               optimizer="sgd",
                               predict_type="step",
                               step_time_skew=coarse, step_subsample=1,
                               batch_size=375, epochs=EPOCHS, validation_set=val_set,
                               nonlinearity="relu")
    nn_kernel.name_tag = f"cors{coarse}"
    general_int_nets.append(nn_kernel)

    for width, depth in [(200, 3), (2048, 2), (200, 4), (4096, 4)]:
        mlp_step_train = utils.MLP(experiment=experiment_step,
                                    training_set=train_set,
                                    batch_size=375,
                                    hidden_dim=width, depth=depth,
                                    learning_rate=(1e-3/50),
                                    predict_type="step",
                                    step_time_skew=coarse, step_subsample=1,
                                    validation_set=val_set, epochs=EPOCHS)
        mlp_step_train.name_tag = f"cors{coarse}"
        general_int_nets.append(mlp_step_train)

    # CNNs
    for cnn_arch in [
            [(None, 32, 5), (32, 32, 5), (32, 32, 5), (32, None, 5)],
            [(None, 32, 9), (32, 32, 9), (32, 32, 9), (32, None, 9)],
            [(None, 64, 9), (64, 64, 9), (64, 64, 9), (64, None, 9)],
    ]:
        cnn_step_train = utils.CNN(experiment=experiment_step,
                                   training_set=train_set,
                                   batch_size=375,
                                   chans_inout_kenel=cnn_arch,
                                   learning_rate=(1e-3/50),
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
    for obj in writable_objects:
        obj.write_description(base_dir)
