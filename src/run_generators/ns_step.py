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
NUM_REPEATS = 1
# Wave base parameters
NS_DT = 0.08
NS_STEPS = 65
NS_INTEGRATORS = ["leapfrog", "euler", "rk4"]

COARSE_LEVELS = [1, 2, 4, 8, 16, 32, 64]  # Used for time skew parameter for training & validation
TRAIN_SET_SIZES = [25]

experiment_general = utils.Experiment("ns-step")

writable_objects = []

train_source = utils.NavierStokesInitialConditionSource(velocity_range=(1.25, 1.75))
val_source = utils.NavierStokesInitialConditionSource(velocity_range=(1.25, 1.75))
eval_source = utils.NavierStokesInitialConditionSource(velocity_range=(1.25, 1.75))
eval_outdist_source = utils.NavierStokesInitialConditionSource(velocity_range=(1.75, 2.0))

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
                                           num_time_steps=NS_STEPS,
                                           time_step_size=NS_DT)
    train_sets.append(_train_set)
    _val_set = utils.NavierStokesDataset(experiment=experiment_general,
                                           initial_cond_source=val_source,
                                           set_type="val",
                                           num_traj=2,
                                           num_time_steps=NS_STEPS,
                                           time_step_size=NS_DT)
    val_set = _val_set
writable_objects.extend(train_sets)
writable_objects.append(val_set)
# Generate eval sets
for source, num_traj, type_key, step_multiplier in [
        (eval_source, 4, "eval", 1),
        #(eval_source, 3, "eval-long", 3),
        (eval_outdist_source, 4, "eval-outdist", 1),
        #(eval_outdist_source, 3, "eval-outdist-long", 3),
        ]:
    for coarse in COARSE_LEVELS:
        ns_steps = math.ceil(NS_STEPS / coarse)
        _eval_set = utils.NavierStokesDataset(experiment=experiment_general,
                                              initial_cond_source=eval_source,
                                              set_type=f"{type_key}-cors{coarse}",
                                              num_traj=num_traj,
                                              num_time_steps=ns_steps,
                                              time_step_size=NS_DT * coarse)
        if coarse not in eval_sets:
            eval_sets[coarse] = []
        eval_sets[coarse].append(_eval_set)
writable_objects.extend(itertools.chain.from_iterable(eval_sets.values()))

# Emit baseline integrator runs for each evaluation set
# for integrator in (EVAL_INTEGRATORS + ["back-euler", "bdf-2"]):
#     for coarse in COARSE_LEVELS:
#         for eval_set in eval_sets[coarse]:
#             pass
#             integration_run_double = utils.BaselineIntegrator(experiment=experiment_general,
#                                                               eval_set=eval_set,
#                                                               eval_dtype="double",
#                                                               integrator=integrator)
#             writable_objects.append(integration_run_double)

# Emit KNN baselines
for coarse, train_set in itertools.product(COARSE_LEVELS, train_sets):
    for eval_set in eval_sets[coarse]:
        knn_pred = utils.KNNPredictorOneshot(experiment_general,
                                             training_set=train_set,
                                             eval_set=eval_set,
                                             step_time_skew=coarse,
                                             step_subsample=1)
        knn_pred.name_tag = f"cors{coarse}"
        writable_objects.append(knn_pred)

# Emit MLP, GN, NNkernel runs
for coarse, train_set, _repeat in itertools.product(COARSE_LEVELS, train_sets, range(NUM_REPEATS)):
    # Only one integrator for this one, the "null" integrator
    # gn_train = utils.GN(experiment=experiment_general,
    #                     training_set=train_set,
    #                     validation_set=val_sets[coarse],
    #                     epochs=GN_EPOCHS)
    # writable_objects.append(gn_train)
    # for eval_set in eval_sets[coarse]:
    #     gn_eval = utils.NetworkEvaluation(experiment=experiment_general,
    #                                       network=gn_train,
    #                                       eval_set=eval_set,
    #                                       integrator="null")
    #     writable_objects.append(gn_eval)
    # Other runs work across all integrators
    # nn_kernel = utils.NNKernel(experiment=experiment_general,
    #                            training_set=train_set,
    #                            learning_rate=0.001, weight_decay=0.0001,
    #                            hidden_dim=32768, train_dtype="float",
    #                            optimizer="sgd",
    #                            batch_size=750, epochs=EPOCHS, validation_set=val_sets[coarse],
    #                            nonlinearity="relu")
    # general_int_nets.append(nn_kernel)
    for width, depth in [(200, 3), (2048, 2)]:
        mlp_step_train = utils.MLP(experiment=experiment_general,
                                   training_set=train_set,
                                   hidden_dim=width, depth=depth,
                                   batch_size=400,
                                   validation_set=val_set, epochs=EPOCHS,
                                   predict_type="step",
                                   step_time_skew=coarse, step_subsample=1)
        mlp_step_train.name_tag = f"cors{coarse}"
        writable_objects.append(mlp_step_train)
        for eval_set in eval_sets[coarse]:
            mlp_step_eval = utils.NetworkEvaluation(experiment=experiment_general,
                                              network=mlp_step_train,
                                              eval_set=eval_set,
                                              integrator="null")
            writable_objects.append(mlp_step_eval)


if __name__ == "__main__":
    args = parser.parse_args()
    base_dir = pathlib.Path(args.base_dir)
    for obj in writable_objects:
        obj.write_description(base_dir)
