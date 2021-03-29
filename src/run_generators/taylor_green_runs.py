import utils
import argparse
import pathlib
from collections import namedtuple
import itertools
import math

parser = argparse.ArgumentParser(description="Generate run descriptions")
parser.add_argument("base_dir", type=str,
                    help="Base directory for run descriptions")



EVAL_INTEGRATORS = ["leapfrog", "euler", "rk4"]
EPOCHS = 150
GN_EPOCHS = 20
NUM_REPEATS = 1
N_GRID = 20

TG_END_TIME = 1
TG_DT = 0.002
TG_STEPS = math.ceil(TG_END_TIME / TG_DT)
TG_SUBSAMPLE = 2**3
EVAL_INTEGRATORS = ["leapfrog", "euler", "rk4"]
COARSE_STEPS = [0]

experiment_general = utils.Experiment("tg-runs")

writable_objects = []

grid_gen = utils.TaylorGreenGridGenerator((N_GRID, N_GRID))
train_source = utils.TaylorGreenInitialConditionSource(mesh_generator=grid_gen,
                                                       viscosity_range=(0.5,1.5),
                                                       density_range=(1.0, 1.0))
val_source = utils.TaylorGreenInitialConditionSource(mesh_generator=grid_gen,
                                                     viscosity_range=(0.5,1.5),
                                                     density_range=(1.0, 1.0))
eval_source = utils.TaylorGreenInitialConditionSource(mesh_generator=grid_gen,
                                                      viscosity_range=(0.5,1.5),
                                                      density_range=(1.0, 1.0))

train_sets = []
val_set = None
eval_sets = []

# Generate data sets
# Generate train set
for num_traj in [30]:
    train_sets.append(
        utils.TaylorGreenDataset(experiment=experiment_general,
                                 initial_cond_source=train_source,
                                 set_type="train",
                                 num_traj=num_traj,
                                 n_grid=N_GRID,
                                 num_time_steps=TG_STEPS,
                                 time_step_size=TG_DT))
writable_objects.extend(train_sets)
# Generate val set
val_set = utils.TaylorGreenDataset(experiment=experiment_general,
                                   initial_cond_source=val_source,
                                   set_type="val",
                                   num_traj=3,
                                   n_grid=N_GRID,
                                   num_time_steps=TG_STEPS,
                                   time_step_size=TG_DT)
writable_objects.append(val_set)
# Generate eval sets
for source, num_traj, type_key, step_multiplier in [
        (eval_source, 6, "eval", 1),
        ]:
    eval_set = []
    for coarse in COARSE_STEPS:
        tg_steps = TG_STEPS // (2**coarse)
        eval_set.append(utils.TaylorGreenDataset(experiment=experiment_general,
                                                 initial_cond_source=source,
                                                 set_type=type_key,
                                                 num_traj=num_traj,
                                                 n_grid=N_GRID,
                                                 num_time_steps=(step_multiplier * tg_steps),
                                                 time_step_size=TG_DT * (2**coarse)))
    eval_sets.append(eval_set)
    writable_objects.extend(eval_set)

# Emit baseline integrator runs for each evaluation set
for eval_set, integrator in itertools.product(eval_sets, (EVAL_INTEGRATORS)):
    for coarse in COARSE_STEPS:
        integration_run_double = utils.BaselineIntegrator(experiment=experiment_general,
                                                          eval_set=eval_set[coarse],
                                                          eval_dtype="double",
                                                          integrator=integrator)
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
