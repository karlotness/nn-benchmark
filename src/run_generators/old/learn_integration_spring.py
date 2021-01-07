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

experiment = utils.Experiment("learn-integration-spring")

initial_condition_sources = {
    "spring-train": utils.SpringInitialConditionSource(radius_range=(0.2, 1)),
    "spring-val": utils.SpringInitialConditionSource(radius_range=(0.2, 1)),
    "spring-eval": utils.SpringInitialConditionSource(radius_range=(0.2, 1)),
}

for dt_factor in [1, 2, 4, 8, 16]:
    # Generate evaluation and validation set
    time_step_size = SPRING_DT * dt_factor
    eval_set = utils.SpringDataset(experiment=experiment,
                                   initial_cond_source=initial_condition_sources["spring-eval"],
                                   num_traj=30,
                                   set_type=f"eval-dtfactor{dt_factor}",
                                   num_time_steps=SPRING_STEPS,
                                   time_step_size=time_step_size)
    val_set = utils.SpringDataset(experiment=experiment,
                                  initial_cond_source=initial_condition_sources["spring-val"],
                                  num_traj=5,
                                  set_type=f"val-dtfactor{dt_factor}",
                                  num_time_steps=SPRING_STEPS,
                                  time_step_size=time_step_size)
    writable_objects.extend([eval_set, val_set])
    # Issue baseline integrator evaluation jobs
    for integrator in ["leapfrog", "euler", "rk4", "scipy-RK45"]:
        integration_run = utils.BaselineIntegrator(experiment=experiment,
                                                   eval_set=eval_set,
                                                   integrator=integrator)
        writable_objects.append(integration_run)
    for num_traj in [10, 100, 500, 1000, 2500]:
        # Generate training set
        train_set = utils.SpringDataset(experiment=experiment,
                                        initial_cond_source=initial_condition_sources["spring-train"],
                                        num_traj=num_traj,
                                        set_type=f"train-dtfactor{dt_factor}",
                                        num_time_steps=SPRING_STEPS,
                                        time_step_size=time_step_size)
        writable_objects.append(train_set)
        # Issue SRNN training and evaluation runs
        for train_integrator in ["leapfrog", "euler", "rk4"]:
            srnn_train = utils.SRNN(experiment, training_set=train_set,
                                    integrator=train_integrator,
                                    hidden_dim=2048, depth=2,
                                    validation_set=val_set,
                                    epochs=EPOCHS)
            srnn_eval_match = utils.NetworkEvaluation(experiment=experiment,
                                                      network=srnn_train,
                                                      eval_set=eval_set,
                                                      integrator=train_integrator)
            writable_objects.extend([srnn_train, srnn_eval_match])


if __name__ == "__main__":
    args = parser.parse_args()
    base_dir = pathlib.Path(args.base_dir)

    for obj in writable_objects:
        obj.write_description(base_dir)
