import utils
import argparse
import pathlib

parser = argparse.ArgumentParser(description="Generate run descriptions")
parser.add_argument("base_dir", type=str,
                    help="Base directory for run descriptions")

writable_objects = []

experiment = utils.Experiment(name="dt-testing")
wave_source = utils.WaveInitialConditionSource()
spring_source = utils.SpringInitialConditionSource()

# Wave integration tests
for mult in [500, 250, 125, 100, 10]:
    factor = 1000 // mult
    data = utils.WaveDataset(experiment=experiment,
                             initial_cond_source=wave_source,
                             num_traj=3,
                             set_type="train", n_grid=250,
                             num_time_steps=100 * mult,
                             time_step_size=0.1 / mult,
                             subsampling=factor)
    euler = utils.BaselineIntegrator(experiment=experiment,
                                     eval_set=data,
                                     integrator="euler")
    leapfrog = utils.BaselineIntegrator(experiment=experiment,
                                        eval_set=data,
                                        integrator="leapfrog")
    rk45 = utils.BaselineIntegrator(experiment=experiment,
                                    eval_set=data,
                                    integrator="scipy-RK45")
    rk4 = utils.BaselineIntegrator(experiment=experiment,
                                   eval_set=data,
                                   integrator="rk4")
    writable_objects.extend([data, euler, leapfrog, rk45, rk4])

# Spring integration tests
for mult in [1, 10, 50, 100, 200]:
    data = utils.SpringDataset(experiment=experiment,
                               initial_cond_source=spring_source,
                               num_traj=3,
                               set_type="train",
                               num_time_steps=300 * mult,
                               time_step_size=0.3 / mult)
    euler = utils.BaselineIntegrator(experiment=experiment,
                                     eval_set=data,
                                     integrator="euler")
    leapfrog = utils.BaselineIntegrator(experiment=experiment,
                                        eval_set=data,
                                        integrator="leapfrog")
    rk45 = utils.BaselineIntegrator(experiment=experiment,
                                    eval_set=data,
                                    integrator="scipy-RK45")
    rk4 = utils.BaselineIntegrator(experiment=experiment,
                                   eval_set=data,
                                   integrator="rk4")
    writable_objects.extend([data, euler, leapfrog, rk45, rk4])

if __name__ == "__main__":
    args = parser.parse_args()
    base_dir = pathlib.Path(args.base_dir)

    for obj in writable_objects:
        obj.write_description(base_dir)
