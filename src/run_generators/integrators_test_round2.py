import utils
import argparse
import pathlib

parser = argparse.ArgumentParser(description="Generate run descriptions")
parser.add_argument("base_dir", type=str,
                    help="Base directory for run descriptions")


writable_objects = []

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
}

for step_factor in [1, 10]:
    experiment = utils.Experiment(f"int-test-step{step_factor}x-round2")
    eval_sets = {
        "wave": utils.WaveDataset(experiment=experiment,
                                  initial_cond_source=initial_condition_sources["wave-eval"],
                                  num_traj=30,
                                  set_type="eval", n_grid=250,
                                  num_time_steps=1000, time_step_size=0.1 / step_factor,
                                  wave_speed=0.1,
                                  subsampling=1000 // step_factor, noise_sigma=0.0),
        "spring": utils.SpringDataset(experiment=experiment,
                                      initial_cond_source=initial_condition_sources["spring-eval"],
                                      num_traj=30,
                                      set_type="eval",
                                      num_time_steps=1000,
                                      time_step_size=0.3 / step_factor,
                                      rtol=1e-10,
                                      noise_sigma=0.0),
        "particle": utils.ParticleDataset(experiment=experiment,
                                          initial_cond_source=initial_condition_sources["particle-eval"],
                                          num_traj=30,
                                          set_type="eval", n_dim=2, n_particles=2,
                                          num_time_steps=500, time_step_size=0.1, noise_sigma=0.0)
    }
    val_sets = {
        "wave": utils.WaveDataset(experiment=experiment,
                                  initial_cond_source=initial_condition_sources["wave-val"],
                                  num_traj=10,
                                  set_type="val", n_grid=250,
                                  num_time_steps=200 * step_factor, time_step_size=0.1 / step_factor,
                                  wave_speed=0.1,
                                  subsampling=1000 // step_factor, noise_sigma=0.0),
        "spring": utils.SpringDataset(experiment=experiment,
                                      initial_cond_source=initial_condition_sources["spring-val"],
                                      num_traj=10,
                                      set_type="val",
                                      num_time_steps=300,
                                      time_step_size=0.3 / step_factor,
                                      rtol=1e-10,
                                      noise_sigma=0.0),
        "particle": utils.ParticleDataset(experiment=experiment,
                                          initial_cond_source=initial_condition_sources["particle-val"],
                                          num_traj=10,
                                          set_type="val", n_dim=2, n_particles=2,
                                          num_time_steps=500, time_step_size=0.1 / step_factor, noise_sigma=0.0),
    }
    writable_objects.extend(eval_sets.values())
    writable_objects.extend(val_sets.values())
    # Add traditional integrator evaluation
    for integrator in ["leapfrog", "euler", "scipy-RK45"]:
        for system in ["wave", "spring", "particle"]:
            integration_run = utils.BaselineIntegrator(experiment=experiment,
                                                       eval_set=eval_sets[system],
                                                       integrator=integrator)
            writable_objects.append(integration_run)
    # Generate conventional tests
    for system in ["spring", "wave", "particle"]:
        for traj_count in [30, 100, 250, 500]:
            if system == "wave" and traj_count == 500:
                # Skip 500 trajectory count for wave
                continue
            val_set = val_sets[system]
            if system == "wave":
                train_set = utils.WaveDataset(experiment=experiment,
                                              initial_cond_source=initial_condition_sources["wave-train"],
                                              num_traj=traj_count,
                                              set_type="train", n_grid=250,
                                              num_time_steps=200 * step_factor, time_step_size=0.1 / step_factor,
                                              wave_speed=0.1,
                                              subsampling=1000 // step_factor, noise_sigma=0.0)
            elif system == "spring":
                train_set = utils.SpringDataset(experiment=experiment,
                                                initial_cond_source=initial_condition_sources["spring-train"],
                                                num_traj=traj_count,
                                                set_type="train",
                                                num_time_steps=300,
                                                time_step_size=0.3 / step_factor,
                                                rtol=1e-10,
                                                noise_sigma=0.0)
            elif system == "particle":
                train_set = utils.ParticleDataset(experiment=experiment,
                                                  initial_cond_source=initial_condition_sources["particle-train"],
                                                  num_traj=traj_count,
                                                  set_type="train", n_dim=2, n_particles=2,
                                                  num_time_steps=500, time_step_size=0.1 / step_factor, noise_sigma=0.0)
            writable_objects.append(train_set)
            # Generate training and evaluation runs
            # CASE: KNN-Predictor
            knn_pred_train = utils.KNNPredictor(experiment=experiment, training_set=train_set)
            knn_pred_eval = utils.NetworkEvaluation(experiment=experiment, network=knn_pred_train, eval_set=eval_sets[system])
            writable_objects.extend([knn_pred_train, knn_pred_eval])
            # CASE: SRNN
            for train_integrator in ["leapfrog", "euler"]:
                srnn_train = utils.SRNN(experiment, training_set=train_set,
                                        integrator=train_integrator,
                                        validation_set=val_set)
                srnn_eval_match = utils.NetworkEvaluation(experiment=experiment,
                                                          network=srnn_train,
                                                          eval_set=eval_sets[system],
                                                          integrator=train_integrator)
                srnn_eval_rk = utils.NetworkEvaluation(experiment=experiment,
                                                       network=srnn_train,
                                                       eval_set=eval_sets[system],
                                                       integrator="scipy-RK45")
                writable_objects.extend([srnn_train, srnn_eval_match, srnn_eval_rk])
            # CASE: all others
            hnn_train = utils.HNN(experiment=experiment, training_set=train_set, validation_set=val_set)
            mlp_train = utils.MLP(experiment=experiment, training_set=train_set, validation_set=val_set)
            hogn_train = utils.HOGN(experiment=experiment, training_set=train_set, validation_set=val_set)
            knn_train = utils.KNNRegressor(experiment=experiment, training_set=train_set)
            writable_objects.extend([hnn_train, mlp_train, hogn_train, knn_train])
            for eval_integrator in ["leapfrog", "euler", "scipy-RK45"]:
                hnn_eval = utils.NetworkEvaluation(experiment=experiment,
                                                   network=hnn_train,
                                                   eval_set=eval_sets[system],
                                                   integrator=eval_integrator)
                mlp_eval = utils.NetworkEvaluation(experiment=experiment,
                                                   network=mlp_train,
                                                   eval_set=eval_sets[system],
                                                   integrator=eval_integrator)
                hogn_eval = utils.NetworkEvaluation(experiment=experiment,
                                                    network=hogn_train,
                                                    eval_set=eval_sets[system],
                                                    integrator=eval_integrator)
                knn_eval = utils.NetworkEvaluation(experiment=experiment,
                                                   network=knn_train,
                                                   eval_set=eval_sets[system],
                                                   integrator=eval_integrator)
                writable_objects.extend([hnn_eval, mlp_eval, hogn_eval, knn_eval])


if __name__ == "__main__":
    args = parser.parse_args()
    base_dir = pathlib.Path(args.base_dir)

    for obj in writable_objects:
        obj.write_description(base_dir)
