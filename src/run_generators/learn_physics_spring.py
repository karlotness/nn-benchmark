import utils
import argparse
import pathlib
import itertools
import math

parser = argparse.ArgumentParser(description="Generate run descriptions")
parser.add_argument("base_dir", type=str,
                    help="Base directory for run descriptions")

EPOCHS = 400

# Spring base parameters
SPRING_STEPS = 1100
SPRING_DT = 0.3 / 100

writable_objects = []

experiment = utils.Experiment("learn-physics-spring")

initial_condition_sources = {
    "spring-train": utils.SpringInitialConditionSource(radius_range=(0.2, 1)),
    "spring-val": utils.SpringInitialConditionSource(radius_range=(0.2, 1)),
    "spring-eval": utils.SpringInitialConditionSource(radius_range=(0.2, 1)),
    "spring-eval-outdist": utils.SpringInitialConditionSource(radius_range=(1, 1.8)),
}


# Small validation sets for use during training
eval_sets = {
    "spring": utils.SpringDataset(experiment=experiment,
                                  initial_cond_source=initial_condition_sources["spring-eval"],
                                  num_traj=30,
                                  set_type="eval",
                                  num_time_steps=SPRING_STEPS,
                                  time_step_size=SPRING_DT),
    "spring-outdist": utils.SpringDataset(experiment=experiment,
                                          initial_cond_source=initial_condition_sources["spring-eval-outdist"],
                                          num_traj=30,
                                          set_type="eval-outdist",
                                          num_time_steps=SPRING_STEPS,
                                          time_step_size=SPRING_DT),
}
writable_objects.extend(eval_sets.values())

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

for num_traj in [10, 100, 500, 1000, 2500]:
    val_set = val_sets[system]
    eval_set = eval_sets[system]
    eval_set_outdist = eval_sets[system + "-outdist"]
    # Construct training sets
    train_set = utils.SpringDataset(experiment=experiment,
                                    initial_cond_source=initial_condition_sources["spring-train"],
                                    num_traj=num_traj,
                                    set_type="train",
                                    num_time_steps=SPRING_STEPS,
                                    time_step_size=SPRING_DT)
    writable_objects.append(train_set)
    # CASE: all others
    hnn_train = utils.HNN(experiment=experiment, training_set=train_set,
                          hidden_dim=200, depth=3,
                          validation_set=val_set, epochs=EPOCHS)
    mlp_train = utils.MLP(experiment=experiment, training_set=train_set,
                          hidden_dim=200, depth=3,
                          validation_set=val_set, epochs=EPOCHS)
    writable_objects.extend([hnn_train, mlp_train])
    for eval_integrator in ["leapfrog", "euler", "rk4", "scipy-RK45"]:
        hnn_eval = utils.NetworkEvaluation(experiment=experiment,
                                           network=hnn_train,
                                           eval_set=eval_set,
                                           integrator=eval_integrator)
        hnn_eval_outdist = utils.NetworkEvaluation(experiment=experiment,
                                                   network=hnn_train,
                                                   eval_set=eval_set_outdist,
                                                   integrator=eval_integrator)
        mlp_eval = utils.NetworkEvaluation(experiment=experiment,
                                           network=mlp_train,
                                           eval_set=eval_set,
                                           integrator=eval_integrator)
        mlp_eval_outdist = utils.NetworkEvaluation(experiment=experiment,
                                                   network=mlp_train,
                                                   eval_set=eval_set_outdist,
                                                   integrator=eval_integrator)
        writable_objects.extend([hnn_eval, mlp_eval, hnn_eval_outdist, mlp_eval_outdist])


if __name__ == "__main__":
    args = parser.parse_args()
    base_dir = pathlib.Path(args.base_dir)

    for obj in writable_objects:
        obj.write_description(base_dir)