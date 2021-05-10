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

writable_objects = []

experiment_general = utils.Experiment(f"springmesh-{mesh_size}-perturball-runs")
mesh_gen = utils.SpringMeshGridGenerator(grid_shape=(mesh_size, mesh_size), fix_particles="top")
train_source = utils.SpringMeshAllPerturb(mesh_generator=mesh_gen, magnitude_range=(0, 0.35))
val_source = utils.SpringMeshAllPerturb(mesh_generator=mesh_gen, magnitude_range=(0, 0.35))
eval_source = utils.SpringMeshAllPerturb(mesh_generator=mesh_gen, magnitude_range=(0, 0.35))
eval_outdist_source = utils.SpringMeshAllPerturb(mesh_generator=mesh_gen, magnitude_range=(0.35, 0.45))

train_sets = []
val_set = None
eval_sets = []

# Generate data sets
# Generate train set
for num_traj in [50]:
    train_sets.append(
        utils.SpringMeshDataset(experiment_general,
                                train_source,
                                num_traj,
                                set_type="train",
                                num_time_steps=SPRING_STEPS,
                                time_step_size=SPRING_DT,
                                subsampling=SPRING_SUBSAMPLE,
                                noise_sigma=0.0,
                                vel_decay=VEL_DECAY))
writable_objects.extend(train_sets)
# Generate val set
val_set = utils.SpringMeshDataset(experiment_general,
                                  val_source,
                                  5,
                                  set_type="val",
                                  num_time_steps=SPRING_STEPS,
                                  time_step_size=SPRING_DT,
                                  subsampling=SPRING_SUBSAMPLE,
                                  noise_sigma=0.0,
                                  vel_decay=VEL_DECAY)
writable_objects.append(val_set)
# Generate eval sets
for source, num_traj, type_key, step_multiplier in [
        #(eval_source, 15, "eval", 1),
        (eval_source, 5, "eval-long", 3),
        #(eval_outdist_source, 15, "eval-outdist", 1),
        (eval_outdist_source, 5, "eval-outdist-long", 3),
        ]:
    eval_set = []
    for coarse in range(0, 7):
        spring_steps = SPRING_STEPS // (2**coarse)
        eval_set.append(
            utils.SpringMeshDataset(experiment_general,
                                    source,
                                    num_traj,
                                    set_type=type_key,
                                    num_time_steps=(step_multiplier * spring_steps),
                                    time_step_size=SPRING_DT * (2**coarse),
                                    subsampling=SPRING_SUBSAMPLE * (2**coarse),
                                    noise_sigma=0.0,
                                    vel_decay=VEL_DECAY))
    eval_sets.append(eval_set)
    writable_objects.extend(eval_set)

# Emit baseline integrator runs for each evaluation set
for eval_set, integrator in itertools.product(eval_sets, EVAL_INTEGRATORS + ["back-euler"]):
    for coarse in range(0, 7):
        integration_run_double = utils.BaselineIntegrator(experiment=experiment_general,
                                                          eval_set=eval_set[coarse],
                                                          eval_dtype="double",
                                                          integrator=integrator)
        writable_objects.append(integration_run_double)

# Emit KNN baselines
for train_set, eval_set in itertools.product(train_sets, eval_sets):
    pass
    # knn_pred = utils.KNNPredictorOneshot(experiment_general,
    #                                      training_set=train_set,
    #                                      eval_set=eval_set[0])
    # writable_objects.append(knn_pred)
    # for integrator in EVAL_INTEGRATORS:
    #     knn_reg = utils.KNNRegressorOneshot(experiment_general,
    #                                         training_set=train_set,
    #                                         eval_set=eval_set[0],
    #                                         integrator=integrator)
    #     writable_objects.append(knn_reg)


# Emit MLP, GN, NNkernel runs
for train_set, _repeat in itertools.product(train_sets, range(NUM_REPEATS)):
    # Only one integrator for this one, the "null" integrator
    for hidden in [64, 128, 256, 512, 1024]:
        gn_train = utils.GN(experiment=experiment_general,
                            training_set=train_set,
                            validation_set=val_set,
                            hidden_dim=hidden,
                            batch_size=3,
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
    for hidden in [32768, 2048, 8192]:
        nn_kernel = utils.NNKernel(experiment=experiment_general,
                                   training_set=train_set,
                                   learning_rate=0.001, weight_decay=0.0001,
                                   hidden_dim=hidden, train_dtype="float",
                                   optimizer="sgd",
                                   batch_size=375, epochs=EPOCHS, validation_set=val_set,
                                   nonlinearity="relu")
        general_int_nets.append(nn_kernel)
    for width, depth in itertools.product([200, 1024, 2048, 4096],
                                          [2, 3, 4]):
        mlp_deriv_train = utils.MLP(experiment=experiment_general,
                                   training_set=train_set,
                                   batch_size=375,
                                   hidden_dim=width, depth=depth,
                                   validation_set=val_set, epochs=EPOCHS)
        general_int_nets.append(mlp_deriv_train)

    writable_objects.extend(general_int_nets)
    for trained_net, eval_set, integrator in itertools.product(general_int_nets, eval_sets, EVAL_INTEGRATORS):
        writable_objects.append(
            utils.NetworkEvaluation(experiment=experiment_general,
                                    network=trained_net,
                                    eval_set=eval_set[0],
                                    integrator=integrator))

if __name__ == "__main__":
    for obj in writable_objects:
        obj.write_description(base_dir)
