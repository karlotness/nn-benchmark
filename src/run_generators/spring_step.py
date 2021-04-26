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
# Spring base parameters
SPRING_END_TIME = 2 * math.pi
SPRING_DT = 0.00781
SPRING_STEPS = math.ceil(SPRING_END_TIME / SPRING_DT)
SPRING_SUBSAMPLE = 2**7
EVAL_INTEGRATORS = ["leapfrog", "euler", "rk4"]

COARSE_LEVELS = [1, 2, 4, 8, 16]
TRAIN_SET_SIZES = [1000]

experiment_general = utils.Experiment("spring-step")
writable_objects = []

train_source = utils.SpringInitialConditionSource(radius_range=(0.2, 1))
val_source = utils.SpringInitialConditionSource(radius_range=(0.2, 1))
eval_source = utils.SpringInitialConditionSource(radius_range=(0.2, 1))
eval_outdist_source = utils.SpringInitialConditionSource(radius_range=(1, 1.2))

train_sets = []
val_sets = {}
eval_sets = {}

# Generate data sets
# Generate train set
for num_traj in TRAIN_SET_SIZES:
    for coarse in COARSE_LEVELS:
        spring_steps = SPRING_STEPS // coarse
        _train_set = (utils.SpringDataset(experiment=experiment_general,
                                          initial_cond_source=train_source,
                                          num_traj=num_traj,
                                          set_type=f"train-cors{coarse}",
                                          num_time_steps=spring_steps,
                                          subsampling=SPRING_SUBSAMPLE * coarse,
                                          time_step_size=SPRING_DT * coarse))
        train_sets.append((coarse, _train_set))
        # Generate val set
        _val_set = utils.SpringDataset(experiment=experiment_general,
                                       initial_cond_source=val_source,
                                       num_traj=5,
                                       set_type=f"val-cors{coarse}",
                                       num_time_steps=spring_steps,
                                       subsampling=SPRING_SUBSAMPLE * coarse,
                                       time_step_size=SPRING_DT * coarse)
        val_sets[coarse] = _val_set
writable_objects.extend(s for _c, s in train_sets)
writable_objects.extend(val_sets.values())
# Generate eval sets
for source, num_traj, type_key, step_multiplier in [
        (eval_source, 30, "eval", 1),
        (eval_source, 5, "eval-long", 3),
        (eval_outdist_source, 30, "eval-outdist", 1),
        (eval_outdist_source, 5, "eval-outdist-long", 3),
        ]:
    for coarse in COARSE_LEVELS:
        spring_steps = SPRING_STEPS // coarse
        _eval_set = utils.SpringDataset(experiment=experiment_general,
                                        initial_cond_source=source,
                                        num_traj=num_traj,
                                        set_type=f"{type_key}-cors{coarse}",
                                        num_time_steps=(step_multiplier * spring_steps),
                                        subsampling=SPRING_SUBSAMPLE * coarse,
                                        time_step_size=SPRING_DT * coarse)
        if coarse not in eval_sets:
            eval_sets[coarse] = []
        eval_sets[coarse].append(_eval_set)
writable_objects.extend(itertools.chain.from_iterable(eval_sets.values()))

# Emit baseline integrator runs for each evaluation set
for integrator in (EVAL_INTEGRATORS + ["back-euler", "bdf-2"]):
    for coarse in COARSE_LEVELS:
        for eval_set in eval_sets[coarse]:
            integration_run_double = utils.BaselineIntegrator(experiment=experiment_general,
                                                              eval_set=eval_set,
                                                              eval_dtype="double",
                                                              integrator=integrator)
            writable_objects.append(integration_run_double)

# Emit KNN baselines
for coarse, train_set in train_sets:
    for eval_set in eval_sets[coarse]:
        knn_pred = utils.KNNPredictorOneshot(experiment_general,
                                             training_set=train_set,
                                             eval_set=eval_set)
        writable_objects.append(knn_pred)

# Emit MLP, GN, NNkernel runs
for (coarse, train_set), _repeat in itertools.product(train_sets, range(NUM_REPEATS)):
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
    general_int_nets = []
    # nn_kernel = utils.NNKernel(experiment=experiment_general,
    #                            training_set=train_set,
    #                            learning_rate=0.001, weight_decay=0.0001,
    #                            hidden_dim=4096, train_dtype="float",
    #                            optimizer="sgd",
    #                            batch_size=750, epochs=EPOCHS, validation_set=val_sets[coarse],
    #                            nonlinearity="relu")
    # general_int_nets.append(nn_kernel)
    for width, depth in [(200, 3), (2048, 2)]:
        mlp_step_train = utils.MLP(experiment=experiment_general,
                                   training_set=train_set,
                                   hidden_dim=width, depth=depth,
                                   validation_set=val_sets[coarse], epochs=EPOCHS,
                                   predict_type="step")
        writable_objects.append(mlp_step_train)
        for eval_set in eval_sets[coarse]:
            mlp_step_eval = utils.NetworkEvaluation(experiment=experiment_general,
                                              network=mlp_step_train,
                                              eval_set=eval_set,
                                              integrator="null")
            writable_objects.append(mlp_step_eval)

    writable_objects.extend(general_int_nets)
    for trained_net, eval_set, integrator in itertools.product(general_int_nets, eval_sets[coarse], EVAL_INTEGRATORS):
        writable_objects.append(
            utils.NetworkEvaluation(experiment=experiment_general,
                                    network=trained_net,
                                    eval_set=eval_set,
                                    integrator=integrator))

if __name__ == "__main__":
    args = parser.parse_args()
    base_dir = pathlib.Path(args.base_dir)
    for obj in writable_objects:
        obj.write_description(base_dir)
