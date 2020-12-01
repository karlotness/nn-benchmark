import utils
import argparse
import pathlib

parser = argparse.ArgumentParser(description="Generate run descriptions")
parser.add_argument("base_dir", type=str,
                    help="Base directory for run descriptions")

writable_objects = []

experiment = utils.Experiment(name="gpu-speed-test")
experiment_gpu = utils.Experiment(name="gpu-speed-test-gpu")
experiment_cpu = utils.Experiment(name="gpu-speed-test-cpu")
wave_source = utils.WaveInitialConditionSource()
spring_source = utils.SpringInitialConditionSource()

# Wave integration tests
for mult in [250]:
    factor = 1000 // mult
    data = utils.WaveDataset(experiment=experiment,
                             initial_cond_source=wave_source,
                             num_traj=100,
                             set_type="train", n_grid=250,
                             num_time_steps=100 * mult,
                             time_step_size=0.1 / mult,
                             subsampling=factor)
    hnn_gpu = utils.HNN(experiment=experiment_gpu, training_set=data, gpu=True, epochs=100)
    hnn_cpu = utils.HNN(experiment=experiment_cpu, training_set=data, gpu=False, epochs=100)
    mlp_gpu = utils.MLP(experiment=experiment_gpu, training_set=data, gpu=True, epochs=100)
    mlp_cpu = utils.MLP(experiment=experiment_cpu, training_set=data, gpu=False, epochs=100)
    srnn_gpu = utils.SRNN(experiment=experiment_gpu, training_set=data, gpu=True, epochs=100)
    srnn_cpu = utils.SRNN(experiment=experiment_cpu, training_set=data, gpu=False, epochs=100)
    writable_objects.extend([data, hnn_gpu, hnn_cpu, mlp_gpu, mlp_cpu, srnn_gpu, srnn_cpu])

# Spring integration tests
for mult in [100]:
    data = utils.SpringDataset(experiment=experiment,
                               initial_cond_source=spring_source,
                               num_traj=100,
                               set_type="train",
                               num_time_steps=300 * mult,
                               time_step_size=0.3 / mult)
    hnn_gpu = utils.HNN(experiment=experiment_gpu, training_set=data, gpu=True, epochs=100)
    hnn_cpu = utils.HNN(experiment=experiment_cpu, training_set=data, gpu=False, epochs=100)
    mlp_gpu = utils.MLP(experiment=experiment_gpu, training_set=data, gpu=True, epochs=100)
    mlp_cpu = utils.MLP(experiment=experiment_cpu, training_set=data, gpu=False, epochs=100)
    srnn_gpu = utils.SRNN(experiment=experiment_gpu, training_set=data, gpu=True, epochs=100)
    srnn_cpu = utils.SRNN(experiment=experiment_cpu, training_set=data, gpu=False, epochs=100)
    writable_objects.extend([data, hnn_gpu, hnn_cpu, mlp_gpu, mlp_cpu, srnn_gpu, srnn_cpu])

if __name__ == "__main__":
    args = parser.parse_args()
    base_dir = pathlib.Path(args.base_dir)

    for obj in writable_objects:
        obj.write_description(base_dir)
