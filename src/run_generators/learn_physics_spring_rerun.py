import utils
import argparse
import pathlib
import itertools
import math

# Spring base parameters
SPRING_STEPS = 1100
SPRING_DT = 0.3 / 100
EPOCHS = 400
NUM_REPEATS = 3

parser = argparse.ArgumentParser(description="Generate run descriptions")
parser.add_argument("base_dir", type=str,
                    help="Base directory for run descriptions")
args = parser.parse_args()
base_dir = pathlib.Path(args.base_dir)

experiment = utils.Experiment("learn-physics-spring-rerun")

data_descr_path = base_dir / "descr"/ "data_gen"
writable_objects = []

train_sets = {
    10: utils.ExistingDataset(data_descr_path / "learn-physics-spring_train-spring-n10-t1100-n0.0_00001.json"),
    100: utils.ExistingDataset(data_descr_path / "learn-physics-spring_train-spring-n100-t1100-n0.0_00001.json"),
    1000: utils.ExistingDataset(data_descr_path / "learn-physics-spring_train-spring-n1000-t1100-n0.0_00001.json"),
}
val_set = utils.ExistingDataset(data_descr_path / "learn-physics-spring_val-spring-n5-t1100-n0.0_00001.json")
eval_set = utils.ExistingDataset(data_descr_path / "learn-physics-spring_eval-spring-n30-t1100-n0.0_00001.json")
eval_set_large_src = utils.SpringInitialConditionSource(radius_range=(0.2, 1))
eval_set_large = utils.SpringDataset(experiment=experiment,
                                     initial_cond_source=eval_set_large_src,
                                     num_traj=10,
                                     set_type="eval-longterm",
                                     num_time_steps=4 * SPRING_STEPS,
                                     time_step_size=SPRING_DT)

writable_objects.append(eval_set_large)

# Traditional integrator baselines
for integrator in ["leapfrog", "euler", "rk4", "scipy-RK45"]:
    integration_run = utils.BaselineIntegrator(experiment=experiment,
                                               eval_set=eval_set,
                                               integrator=integrator)
    integration_run_large = utils.BaselineIntegrator(experiment=experiment,
                                                     eval_set=eval_set_large,
                                                     integrator=integrator)
    writable_objects.extend([integration_run, integration_run_large])

for width, depth in [(200, 3), (2048, 2)]:
    for train_set in train_sets.values():
        for _repeat in range(NUM_REPEATS):
            # Train the networks
            hnn_train = utils.HNN(experiment=experiment, training_set=train_set,
                                  hidden_dim=width, depth=depth,
                                  validation_set=val_set, epochs=EPOCHS)
            mlp_train = utils.MLP(experiment=experiment, training_set=train_set,
                                  hidden_dim=width, depth=depth,
                                  validation_set=val_set, epochs=EPOCHS)
            writable_objects.extend([hnn_train, mlp_train])
            # Evaluate the networks
            for eval_integrator in ["leapfrog", "euler", "rk4", "scipy-RK45"]:
                hnn_eval = utils.NetworkEvaluation(experiment=experiment,
                                                   network=hnn_train,
                                                   eval_set=eval_set,
                                                   integrator=eval_integrator)
                hnn_eval_large = utils.NetworkEvaluation(experiment=experiment,
                                                         network=hnn_train,
                                                         eval_set=eval_set_large,
                                                         integrator=eval_integrator)
                mlp_eval = utils.NetworkEvaluation(experiment=experiment,
                                                   network=mlp_train,
                                                   eval_set=eval_set,
                                                   integrator=eval_integrator)
                mlp_eval_large = utils.NetworkEvaluation(experiment=experiment,
                                                         network=mlp_train,
                                                         eval_set=eval_set_large,
                                                         integrator=eval_integrator)
                writable_objects.extend([hnn_eval, mlp_eval, hnn_eval_large, mlp_eval_large])


# Write outputs
for obj in writable_objects:
    obj.write_description(base_dir)
