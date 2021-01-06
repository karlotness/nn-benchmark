import utils
import argparse
import pathlib
import itertools
import math

parser = argparse.ArgumentParser(description="Generate run descriptions")
parser.add_argument("base_dir", type=str,
                    help="Base directory for run descriptions")

EPOCHS = 250

# Spring base parameters
SPRING_STEPS = 1100
SPRING_DT = 0.3 / 100

writable_objects = []

experiment = utils.Experiment("test-nnkernel")

initial_condition_sources = {
    "spring-train": utils.SpringInitialConditionSource(radius_range=(0.2, 1)),
    "spring-val": utils.SpringInitialConditionSource(radius_range=(0.2, 1)),
    "spring-eval": utils.SpringInitialConditionSource(radius_range=(0.2, 1)),
    "spring-eval-outdist": utils.SpringInitialConditionSource(radius_range=(1, 1.8)),
}


eval_sets = {
    "spring": utils.SpringDataset(experiment=experiment,
                                  initial_cond_source=initial_condition_sources["spring-eval"],
                                  num_traj=30,
                                  set_type="eval",
                                  num_time_steps=SPRING_STEPS,
                                  time_step_size=SPRING_DT),
}
writable_objects.extend(eval_sets.values())

# Small validation sets for use during training
val_sets = {
    "spring": utils.SpringDataset(experiment=experiment,
                                  initial_cond_source=initial_condition_sources["spring-val"],
                                  num_traj=5,
                                  set_type="val",
                                  num_time_steps=SPRING_STEPS,
                                  time_step_size=SPRING_DT),
}
writable_objects.extend(val_sets.values())

system = "spring"

# Traditional integrator baselines
for integrator in ["leapfrog", "euler", "rk4", "scipy-RK45"]:
    integration_run = utils.BaselineIntegrator(experiment=experiment,
                                               eval_set=eval_sets[system],
                                               integrator=integrator)
    writable_objects.append(integration_run)

for num_traj in [10, 100, 500]:
    val_set = val_sets[system]
    eval_set = eval_sets[system]
    # Construct training sets
    train_set = utils.SpringDataset(experiment=experiment,
                                    initial_cond_source=initial_condition_sources["spring-train"],
                                    num_traj=num_traj,
                                    set_type="train",
                                    num_time_steps=SPRING_STEPS,
                                    time_step_size=SPRING_DT)
    writable_objects.append(train_set)
    # CASE: all others
    train_runs = []
    for learning_rate in [1e-3, 1e-4, 1e-5, 1e-6]:
        for weight_decay in [0.1, 0.01, 0.001, 0.0001]:
            for retry in range(3):
                nn_kern_train = utils.NNKernel(experiment=experiment,
                                               training_set=train_set,
                                               learning_rate=learning_rate, weight_decay=weight_decay,
                                               hidden_dim=4096, train_dtype="float",
                                               optimizer="sgd",
                                               batch_size=750, epochs=EPOCHS, validation_set=val_set,
                                               nonlinearity="relu")
                train_runs.append(nn_kern_train)
    for retry in range(3):
        mlp_train = utils.MLP(experiment=experiment, training_set=train_set,
                              hidden_dim=200, depth=3,
                              validation_set=val_set, epochs=EPOCHS)
        train_runs.append(mlp_train)
    writable_objects.extend(train_runs)
    for eval_integrator in ["leapfrog", "euler", "rk4", "scipy-RK45"]:
        for train_run in train_runs:
            eval_run = utils.NetworkEvaluation(experiment=experiment,
                                               network=train_run,
                                               eval_set=eval_set,
                                               integrator=eval_integrator)
            writable_objects.append(eval_run)


if __name__ == "__main__":
    args = parser.parse_args()
    base_dir = pathlib.Path(args.base_dir)

    for obj in writable_objects:
        obj.write_description(base_dir)
