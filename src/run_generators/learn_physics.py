import utils
import argparse
import pathlib
import itertools
import math

parser = argparse.ArgumentParser(description="Generate run descriptions")
parser.add_argument("base_dir", type=str,
                    help="Base directory for run descriptions")

# Wave base parameters
WAVE_DT = 0.1 / 250
WAVE_STEPS = 100 * 250
WAVE_SUBSAMPLE = 1000 // 250

# Spring base parameters
SPRING_STEPS = 1100
SPRING_DT = 0.3 / 100

# Particle base parameters
PARTICLE_STEPS = 500
PARTICLE_DT = 0.01

writable_objects = []

experiment_base = utils.Experiment("learn-physics")
experiment_hard = utils.Experiment("learn-physics-hard")

initial_condition_sources = {
    "spring-train": utils.SpringInitialConditionSource(),
    "spring-val": utils.SpringInitialConditionSource(),
    "spring-eval": utils.SpringInitialConditionSource(),
    "wave-train": utils.WaveInitialConditionSource(),
    "wave-val": utils.WaveInitialConditionSource(),
    "wave-eval": utils.WaveInitialConditionSource(),
    "particle-train": utils.ParticleInitialConditionSource(n_particles=2, n_dim=2),
    "particle-val": utils.ParticleInitialConditionSource(n_particles=2, n_dim=2),
    "particle-eval": utils.ParticleInitialConditionSource(n_particles=2, n_dim=2),
    "wave-hard-train": utils.WaveInitialConditionSource(position_range=(0.35, 0.65)),
    "wave-hard-val": utils.WaveInitialConditionSource(position_range=(0.35, 0.65)),
    "wave-hard-eval": utils.WaveInitialConditionSource(position_range=(0.35, 0.65)),
}



# Small validation sets for use during training
eval_sets = {
    "wave": utils.WaveDataset(experiment=experiment_base,
                              initial_cond_source=initial_condition_sources["wave-eval"],
                              num_traj=30,
                              set_type="eval", n_grid=250,
                              num_time_steps=WAVE_STEPS,
                              time_step_size=WAVE_DT,
                              wave_speed=0.1,
                              subsampling=WAVE_SUBSAMPLE),
    "wave-hard": utils.WaveDataset(experiment=experiment_hard,
                                   initial_cond_source=initial_condition_sources["wave-hard-eval"],
                                   num_traj=30,
                                   set_type="eval", n_grid=250,
                                   num_time_steps=WAVE_STEPS,
                                   time_step_size=WAVE_DT,
                                   wave_speed=0.1,
                                   subsampling=WAVE_SUBSAMPLE),
    "spring": utils.SpringDataset(experiment=experiment_base,
                                  initial_cond_source=initial_condition_sources["spring-eval"],
                                  num_traj=30,
                                  set_type="eval",
                                  num_time_steps=SPRING_STEPS,
                                  time_step_size=SPRING_DT),
    "particle": utils.ParticleDataset(experiment=experiment_base,
                                      initial_cond_source=initial_condition_sources["particle-eval"],
                                      num_traj=30,
                                      set_type="eval", n_dim=2, n_particles=2,
                                      num_time_steps=PARTICLE_STEPS,
                                      time_step_size=PARTICLE_DT)
}
writable_objects.extend(eval_sets.values())

val_sets = {
    "wave": utils.WaveDataset(experiment=experiment_base,
                              initial_cond_source=initial_condition_sources["wave-val"],
                              num_traj=5,
                              set_type="val", n_grid=250,
                              num_time_steps=WAVE_STEPS,
                              time_step_size=WAVE_DT,
                              wave_speed=0.1,
                              subsampling=WAVE_SUBSAMPLE),
    "wave-hard": utils.WaveDataset(experiment=experiment_hard,
                                   initial_cond_source=initial_condition_sources["wave-hard-val"],
                                   num_traj=5,
                                   set_type="val", n_grid=250,
                                   num_time_steps=WAVE_STEPS,
                                   time_step_size=WAVE_DT,
                                   wave_speed=0.1,
                                   subsampling=WAVE_SUBSAMPLE),
    "spring": utils.SpringDataset(experiment=experiment_base,
                                  initial_cond_source=initial_condition_sources["spring-val"],
                                  num_traj=5,
                                  set_type="val",
                                  num_time_steps=SPRING_STEPS,
                                  time_step_size=SPRING_DT),
    "particle": utils.ParticleDataset(experiment=experiment_base,
                                      initial_cond_source=initial_condition_sources["particle-val"],
                                      num_traj=5,
                                      set_type="val", n_dim=2, n_particles=2,
                                      num_time_steps=PARTICLE_STEPS,
                                      time_step_size=PARTICLE_DT),
}
writable_objects.extend(val_sets.values())

# Traditional integrator baselines
for integrator in ["leapfrog", "euler", "rk4", "scipy-RK45"]:
    for system in ["wave", "spring", "particle", "wave-hard"]:
        if system == "wave-hard":
            experiment = experiment_hard
        else:
            experiment = experiment_base
        integration_run = utils.BaselineIntegrator(experiment=experiment,
                                                   eval_set=eval_sets[system],
                                                   integrator=integrator)
        writable_objects.append(integration_run)

for num_traj, step_factor in itertools.product([10, 25, 50, 100],
                                               [0.25, 0.5, 1]):
    for system in ["wave", "spring", "particle", "wave-hard"]:
        if system == "wave-hard":
            experiment = experiment_hard
        else:
            experiment = experiment_base
        val_set = val_sets[system]
        eval_set = eval_sets[system]
        # Construct training sets
        if system == "wave":
            num_steps = math.ceil(step_factor * WAVE_STEPS)
            train_set = utils.WaveDataset(experiment=experiment,
                                          initial_cond_source=initial_condition_sources["wave-train"],
                                          num_traj=num_traj,
                                          set_type="train", n_grid=250,
                                          num_time_steps=num_steps,
                                          time_step_size=WAVE_DT,
                                          wave_speed=0.1,
                                          subsampling=WAVE_SUBSAMPLE)
        elif system == "wave-hard":
            num_steps = math.ceil(step_factor * WAVE_STEPS)
            train_set = utils.WaveDataset(experiment=experiment,
                                          initial_cond_source=initial_condition_sources["wave-hard-train"],
                                          num_traj=num_traj,
                                          set_type="train", n_grid=250,
                                          num_time_steps=num_steps,
                                          time_step_size=WAVE_DT,
                                          wave_speed=0.1,
                                          subsampling=WAVE_SUBSAMPLE)
        elif system == "spring":
            num_steps = math.ceil(step_factor * SPRING_STEPS)
            train_set = utils.SpringDataset(experiment=experiment,
                                            initial_cond_source=initial_condition_sources["spring-train"],
                                            num_traj=num_traj,
                                            set_type="train",
                                            num_time_steps=num_steps,
                                            time_step_size=SPRING_DT)
        elif system == "particle":
            num_steps = math.ceil(step_factor * PARTICLE_STEPS)
            train_set = utils.ParticleDataset(experiment=experiment,
                                              initial_cond_source=initial_condition_sources["particle-train"],
                                              num_traj=num_traj,
                                              set_type="train", n_dim=2, n_particles=2,
                                              num_time_steps=num_steps,
                                              time_step_size=PARTICLE_DT)
        writable_objects.append(train_set)
        # Build networks for training
        # CASE: KNN-Predictor
        knn_pred_train = utils.KNNPredictor(experiment=experiment,
                                            training_set=train_set)
        knn_pred_eval = utils.NetworkEvaluation(experiment=experiment,
                                                network=knn_pred_train,
                                                eval_set=eval_set)
        writable_objects.extend([knn_pred_train, knn_pred_eval])
        # CASE: all others
        hnn_train = utils.HNN(experiment=experiment, training_set=train_set, validation_set=val_set)
        mlp_train = utils.MLP(experiment=experiment, training_set=train_set, validation_set=val_set)
        hogn_train = utils.HOGN(experiment=experiment, training_set=train_set, validation_set=val_set)
        knn_train = utils.KNNRegressor(experiment=experiment, training_set=train_set)
        writable_objects.extend([hnn_train, mlp_train, hogn_train, knn_train])
        for eval_integrator in ["leapfrog", "euler", "rk4", "scipy-RK45"]:
            hnn_eval = utils.NetworkEvaluation(experiment=experiment,
                                               network=hnn_train,
                                               eval_set=eval_set,
                                               integrator=eval_integrator)
            mlp_eval = utils.NetworkEvaluation(experiment=experiment,
                                               network=mlp_train,
                                               eval_set=eval_set,
                                               integrator=eval_integrator)
            hogn_eval = utils.NetworkEvaluation(experiment=experiment,
                                                network=hogn_train,
                                                eval_set=eval_set,
                                                integrator=eval_integrator)
            knn_eval = utils.NetworkEvaluation(experiment=experiment,
                                               network=knn_train,
                                               eval_set=eval_set,
                                               integrator=eval_integrator)
            writable_objects.extend([hnn_eval, mlp_eval, hogn_eval, knn_eval])


if __name__ == "__main__":
    args = parser.parse_args()
    base_dir = pathlib.Path(args.base_dir)

    for obj in writable_objects:
        obj.write_description(base_dir)
