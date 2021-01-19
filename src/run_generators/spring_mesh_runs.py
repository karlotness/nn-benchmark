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
SPRING_DT = 0.3 / 100
SPRING_STEPS = math.ceil(SPRING_END_TIME / SPRING_DT)
VEL_DECAY = 0.9

NUM_TRAJ = 100

writable_objects = []

experiment_general = utils.Experiment("springmesh-test")
experiment_outdist = utils.Experiment("springmesh-test-outdist")
mesh_gen = utils.SpringMeshGridGenerator(grid_shape=(3, 3), fix_particles="top")
train_source = utils.SpringMeshRowPerturb(mesh_generator=mesh_gen, magnitude=0.25, row=0)
val_source = utils.SpringMeshRowPerturb(mesh_generator=mesh_gen, magnitude=0.25, row=0)
eval_source = utils.SpringMeshRowPerturb(mesh_generator=mesh_gen, magnitude=0.25, row=0)
eval_outdist_source = utils.SpringMeshRowPerturb(mesh_generator=mesh_gen, magnitude=0.35, row=0)

data_sets = {}

DatasetKey = namedtuple("DatasetKey", ["type", "dt_factor", "n_traj"])

# Generate data sets
for dt_factor in [1]:
    time_step_size = SPRING_DT * dt_factor
    num_steps = math.ceil(SPRING_END_TIME / time_step_size)
    # Generate eval and val sets
    key = DatasetKey(type="val", dt_factor=dt_factor, n_traj=5)
    dset = utils.SpringMeshDataset(experiment_general, val_source, 5,
                                   set_type="val",
                                   num_time_steps=SPRING_STEPS, time_step_size=time_step_size,
                                   subsampling=1, noise_sigma=0.0, vel_decay=VEL_DECAY)
    data_sets[key] = dset
    key = DatasetKey(type="eval", dt_factor=dt_factor, n_traj=15)
    dset = utils.SpringMeshDataset(experiment_general, eval_source, 15,
                                   set_type="eval",
                                   num_time_steps=SPRING_STEPS, time_step_size=time_step_size,
                                   subsampling=1, noise_sigma=0.0, vel_decay=VEL_DECAY)
    data_sets[key] = dset
    key = DatasetKey(type="eval-outdist", dt_factor=dt_factor, n_traj=15)
    dset = utils.SpringMeshDataset(experiment_outdist, eval_outdist_source, 15,
                                   set_type="eval-outdist",
                                   num_time_steps=SPRING_STEPS, time_step_size=time_step_size,
                                   subsampling=1, noise_sigma=0.0, vel_decay=VEL_DECAY)
    data_sets[key] = dset
    # Generate training sets
    for num_traj in [NUM_TRAJ]:
        key = DatasetKey(type="train", dt_factor=dt_factor, n_traj=num_traj)
        dset = utils.SpringMeshDataset(experiment_general, train_source, num_traj,
                                       set_type="train",
                                       num_time_steps=SPRING_STEPS, time_step_size=time_step_size,
                                       subsampling=1, noise_sigma=0.0, vel_decay=VEL_DECAY)
        data_sets[key] = dset
writable_objects.extend(data_sets.values())

# Emit baseline integrator runs for each evaluation set
for key, dset in data_sets.items():
    if key.type not in {"eval", "eval-long", "eval-outdist", "eval-easy"}:
        continue
    for integrator in ["leapfrog", "euler", "rk4", "scipy-RK45",
                       "back-euler", "implicit-rk"]:
        integration_run_float = utils.BaselineIntegrator(experiment=experiment_general,
                                                         eval_set=dset,
                                                         eval_dtype="float",
                                                         integrator=integrator)
        integration_run_double = utils.BaselineIntegrator(experiment=experiment_general,
                                                          eval_set=dset,
                                                          eval_dtype="double",
                                                          integrator=integrator)
        writable_objects.append(integration_run_float)
        writable_objects.append(integration_run_double)

# Emit KNN baselines
for dt_factor in [1]:
    for num_traj in [NUM_TRAJ]:
        train_set_key = DatasetKey(type="train", dt_factor=dt_factor, n_traj=num_traj)
        eval_set_key = DatasetKey(type="eval", dt_factor=dt_factor, n_traj=15)
        eval_set_outdist_key = DatasetKey(type="eval-outdist", dt_factor=dt_factor, n_traj=15)
        train_set = data_sets[train_set_key]
        eval_set = data_sets[eval_set_key]
        eval_set_outdist = data_sets[eval_set_outdist_key]
        # Write other KNNs
        knn_predict_eval = utils.KNNPredictorOneshot(experiment_general, training_set=train_set, eval_set=eval_set)
        knn_predict_eval_outdist = utils.KNNPredictorOneshot(experiment_outdist, training_set=train_set, eval_set=eval_set_outdist)
        writable_objects.extend([knn_predict_eval, knn_predict_eval_outdist])
        for eval_integrator in ["leapfrog", "euler", "rk4", "scipy-RK45"]:
            knn_regressor = utils.KNNRegressorOneshot(experiment_general, training_set=train_set, eval_set=eval_set, integrator=eval_integrator)
            knn_regressor_outdist = utils.KNNRegressorOneshot(experiment_outdist, training_set=train_set, eval_set=eval_set_outdist, integrator=eval_integrator)
            writable_objects.extend([knn_regressor, knn_regressor_outdist])

# EMIT MLP and NNKERNEL RUNS
for num_traj in [NUM_TRAJ]:
        train_set_key = DatasetKey(type="train", dt_factor=1, n_traj=num_traj)
        val_set_key = DatasetKey(type="val", dt_factor=1, n_traj=5)
        eval_set_key = DatasetKey(type="eval", dt_factor=1, n_traj=15)
        eval_set_outdist_key = DatasetKey(type="eval-outdist", dt_factor=1, n_traj=15)

        train_set = data_sets[train_set_key]
        val_set = data_sets[val_set_key]
        eval_set = data_sets[eval_set_key]
        eval_set_outdist = data_sets[eval_set_outdist_key]

        # SPECIAL CASE: GN, only one integrator
        for _repeat in range(NUM_REPEATS):
            gn_train = utils.GN(experiment=experiment_general,
                                training_set=train_set,
                                validation_set=val_set,
                                epochs=GN_EPOCHS)
            eval_gn_general = utils.NetworkEvaluation(experiment=experiment_general,
                                                      network=gn_train,
                                                      eval_set=eval_set,
                                                      integrator="null")
            eval_gn_outdist = utils.NetworkEvaluation(experiment=experiment_outdist,
                                                      network=gn_train,
                                                      eval_set=eval_set_outdist,
                                                      integrator="null")
            writable_objects.extend([gn_train,
                                     eval_gn_general, eval_gn_outdist])


        trained_nets = []
        for _repeat in range(NUM_REPEATS):
            # Train the networks
            nn_kern_train = utils.NNKernel(experiment=experiment_general,
                                           training_set=train_set,
                                           learning_rate=0.001, weight_decay=0.0001,
                                           hidden_dim=8192, train_dtype="float",
                                           optimizer="sgd",
                                           batch_size=750, epochs=EPOCHS, validation_set=val_set,
                                           nonlinearity="relu")
            trained_nets.append(nn_kern_train)
            for width, depth in [(200, 3), (2048, 2)]:
                mlp_train = utils.MLP(experiment=experiment_general, training_set=train_set,
                                      hidden_dim=width, depth=depth,
                                      validation_set=val_set, epochs=EPOCHS)
                trained_nets.append(mlp_train)
        # Evaluate the networks
        writable_objects.extend(trained_nets)
        for eval_integrator in ["leapfrog", "euler", "rk4", "scipy-RK45"]:
            for trained_net in trained_nets:
                eval_general = utils.NetworkEvaluation(experiment=experiment_general,
                                                       network=trained_net,
                                                       eval_set=eval_set,
                                                       integrator=eval_integrator)
                eval_outdist = utils.NetworkEvaluation(experiment=experiment_outdist,
                                                       network=trained_net,
                                                       eval_set=eval_set_outdist,
                                                       integrator=eval_integrator)
                writable_objects.extend([eval_general, eval_outdist])

if __name__ == "__main__":
    args = parser.parse_args()
    base_dir = pathlib.Path(args.base_dir)
    for obj in writable_objects:
        obj.write_description(base_dir)
