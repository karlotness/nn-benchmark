import utils
import argparse
import pathlib

parser = argparse.ArgumentParser(description="Generate run descriptions")
parser.add_argument("base_dir", type=str,
                    help="Base directory for run descriptions")

EPOCHS = 400
NUM_REPEATS = 3
# Spring base parameters
SPRING_STEPS = 1100
SPRING_DT = 0.3 / 100

experiment_integration_small_train = utils.Experiment("learn-integration-spring-easy")
writable_objects = []

train_source = utils.SpringInitialConditionSource(radius_range=(0.3, 0.4))
eval_source = utils.SpringInitialConditionSource(radius_range=(0.3, 0.4))

# Emit the small training set experiment for SRNN
# Generate special training/eval set
for dt_factor in [1, 2, 4, 8, 16]:
    time_step_size = SPRING_DT * dt_factor
    small_train_set = utils.SpringDataset(experiment=experiment_integration_small_train,
                                          initial_cond_source=train_source,
                                          num_traj=11,
                                          set_type=f"train-dtfactor{dt_factor}",
                                          num_time_steps=SPRING_STEPS,
                                          time_step_size=time_step_size)
    small_eval_set = utils.SpringDataset(experiment=experiment_integration_small_train,
                                         initial_cond_source=eval_source,
                                         num_traj=11,
                                         set_type=f"eval-dtfactor{dt_factor}",
                                         num_time_steps=SPRING_STEPS,
                                         time_step_size=time_step_size)
    writable_objects.extend([small_train_set, small_eval_set])
    for train_integrator in ["leapfrog", "euler", "rk4"]:
        small_srnn_train = utils.SRNN(experiment_integration_small_train, training_set=small_train_set,
                                      integrator=train_integrator,
                                      hidden_dim=2048, depth=2,
                                      validation_set=None,
                                      epochs=EPOCHS)
        small_srnn_eval_match = utils.NetworkEvaluation(experiment=experiment_integration_small_train,
                                                        network=small_srnn_train,
                                                        eval_set=small_train_set,
                                                        integrator=train_integrator,
                                                        gpu=True)
        small_srnn_eval_match_2 = utils.NetworkEvaluation(experiment=experiment_integration_small_train,
                                                          network=small_srnn_train,
                                                          eval_set=small_eval_set,
                                                          integrator=train_integrator,
                                                          gpu=True)
        integration_run_train = utils.BaselineIntegrator(experiment=experiment_integration_small_train,
                                                         eval_set=small_train_set,
                                                         integrator=train_integrator)
        integration_run_eval = utils.BaselineIntegrator(experiment=experiment_integration_small_train,
                                                        eval_set=small_eval_set,
                                                        integrator=train_integrator)
        writable_objects.extend([small_srnn_train, small_srnn_eval_match, small_srnn_eval_match_2,
                                 integration_run_eval, integration_run_train])


if __name__ == "__main__":
    args = parser.parse_args()
    base_dir = pathlib.Path(args.base_dir)
    for obj in writable_objects:
        obj.write_description(base_dir)
