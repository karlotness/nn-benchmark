import utils
import argparse
import pathlib

parser = argparse.ArgumentParser(description="Generate run descriptions")
parser.add_argument("base_dir", type=str,
                    help="Base directory for run descriptions")

experiment = utils.Experiment("test-knn-oneshot")

train_source = utils.SpringInitialConditionSource(radius_range=(0.2, 1))
dset = utils.SpringDataset(experiment=experiment,
                           initial_cond_source=train_source,
                           num_traj=5,
                           set_type=f"train",
                           num_time_steps=100,
                           time_step_size=0.3)

# Knn runs
knn_predict = utils.KNNPredictorOneshot(experiment, training_set=dset, eval_set=dset)
knn_regressor_leapfrog = utils.KNNRegressorOneshot(experiment, training_set=dset, eval_set=dset, integrator='leapfrog')
knn_regressor_euler = utils.KNNRegressorOneshot(experiment, training_set=dset, eval_set=dset, integrator='euler')

writable_objects = [dset, knn_predict, knn_regressor_leapfrog, knn_regressor_euler]

if __name__ == "__main__":
    args = parser.parse_args()
    base_dir = pathlib.Path(args.base_dir)
    for obj in writable_objects:
        obj.write_description(base_dir)
