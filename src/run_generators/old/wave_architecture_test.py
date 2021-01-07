import utils
import argparse
import pathlib
import itertools
import math

parser = argparse.ArgumentParser(description="Generate run descriptions")
parser.add_argument("base_dir", type=str,
                    help="Base directory for run descriptions")

EPOCHS = 400

# Wave base parameters
WAVE_DT = 0.1 / 25
WAVE_STEPS = 100 * 25
WAVE_SUBSAMPLE = 1000 // 25
WAVE_N_GRID = 125

WAVE_EVAL_DT = 0.1 / 250
WAVE_EVAL_STEPS = 100 * 250
WAVE_EVAL_SUBSAMPLE = 1000 // 250

writable_objects = []

experiment = utils.Experiment("wave-architecture-test")

initial_condition_sources = {
    "wave-train": utils.WaveInitialConditionSource(),
    "wave-val": utils.WaveInitialConditionSource(),
    "wave-eval": utils.WaveInitialConditionSource(),
}

eval_set = utils.WaveDataset(experiment=experiment,
                             initial_cond_source=initial_condition_sources["wave-eval"],
                             num_traj=3,
                             set_type="eval",
                             n_grid=WAVE_N_GRID,
                             num_time_steps=WAVE_EVAL_STEPS,
                             time_step_size=WAVE_EVAL_DT,
                             wave_speed=0.1,
                             subsampling=WAVE_EVAL_SUBSAMPLE)
train_set = utils.WaveDataset(experiment=experiment,
                              initial_cond_source=initial_condition_sources["wave-train"],
                              num_traj=500,
                              set_type="train",
                              n_grid=WAVE_N_GRID,
                              num_time_steps=WAVE_STEPS,
                              time_step_size=WAVE_DT,
                              wave_speed=0.1,
                              subsampling=WAVE_SUBSAMPLE)
val_set = utils.WaveDataset(experiment=experiment,
                            initial_cond_source=initial_condition_sources["wave-val"],
                            num_traj=5,
                            set_type="val",
                            n_grid=WAVE_N_GRID,
                            num_time_steps=WAVE_STEPS,
                            time_step_size=WAVE_DT,
                            wave_speed=0.1,
                            subsampling=WAVE_SUBSAMPLE)

writable_objects.extend([eval_set, train_set, val_set])

# Traditional integrator baselines
for integrator in ["leapfrog", "euler", "rk4", "scipy-RK45"]:
    integration_run = utils.BaselineIntegrator(experiment=experiment,
                                               eval_set=eval_set,
                                               integrator=integrator)
    writable_objects.append(integration_run)

for depth, width in itertools.product([2, 3, 4],
                                      [200, 1024, 2048, 4096]):
    hnn_train = utils.HNN(experiment=experiment, training_set=train_set,
                          hidden_dim=width, depth=depth,
                          validation_set=val_set, epochs=EPOCHS)
    writable_objects.append(hnn_train)
    for eval_integrator in ["leapfrog", "euler", "rk4", "scipy-RK45"]:
        hnn_eval = utils.NetworkEvaluation(experiment=experiment,
                                           network=hnn_train,
                                           eval_set=eval_set,
                                           integrator=eval_integrator)
        writable_objects.append(hnn_eval)
    # Issue SRNN training and evaluation runs
    for train_integrator in ["leapfrog", "euler", "rk4"]:
        srnn_train = utils.SRNN(experiment, training_set=train_set,
                                integrator=train_integrator,
                                hidden_dim=width, depth=depth,
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
