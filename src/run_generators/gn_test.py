import utils
import argparse
import pathlib
import itertools
import math

parser = argparse.ArgumentParser(description="Generate run descriptions")
parser.add_argument("base_dir", type=str,
                    help="Base directory for run descriptions")

EPOCHS = 500

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

experiment = utils.Experiment("test-gn-spring")

initial_condition_sources = {
    "spring-train": utils.SpringInitialConditionSource(),
    "spring-val": utils.SpringInitialConditionSource(),
    "spring-eval": utils.SpringInitialConditionSource(),
}


# Small validation sets for use during training
eval_sets = {
    "spring": utils.SpringDataset(experiment=experiment,
                                  initial_cond_source=initial_condition_sources["spring-eval"],
                                  num_traj=30,
                                  set_type="eval",
                                  num_time_steps=SPRING_STEPS,
                                  time_step_size=SPRING_DT),
}
writable_objects.extend(eval_sets.values())

val_sets = {
    "spring": utils.SpringDataset(experiment=experiment,
                                  initial_cond_source=initial_condition_sources["spring-val"],
                                  num_traj=5,
                                  set_type="val",
                                  num_time_steps=SPRING_STEPS,
                                  time_step_size=SPRING_DT),
}
writable_objects.extend(val_sets.values())

for num_traj, step_factor in itertools.product([10], [0.25]):
    for system in ["spring"]:
        val_set = val_sets[system]
        eval_set = eval_sets[system]
        # Construct training sets
        if system == "spring":
            num_steps = math.ceil(step_factor * SPRING_STEPS)
            train_set = utils.SpringDataset(experiment=experiment,
                                            initial_cond_source=initial_condition_sources["spring-train"],
                                            num_traj=num_traj,
                                            set_type="train",
                                            num_time_steps=num_steps,
                                            time_step_size=SPRING_DT)
        writable_objects.append(train_set)
        # Build networks for training
        gn_train = utils.GN(experiment=experiment,
                            training_set=train_set,
                            validation_set=val_set,
                            # validation_set=train_set,
                            epochs=EPOCHS,
                            # scheduler="exponential",
                            )
        writable_objects.extend([gn_train])
        for eval_integrator in ["null"]:
            gn_eval = utils.NetworkEvaluation(experiment=experiment,
                                              network=gn_train,
                                              eval_set=eval_set,
                                              # eval_set=train_set,
                                              integrator=eval_integrator,
                                              system=system)
            writable_objects.extend([gn_eval])


# Traditional integrator baselines
for integrator in ["rk4"]:
  for system in ["spring"]:
    integration_run = utils.BaselineIntegrator(experiment=experiment,
        eval_set=eval_sets[system],
        # eval_set=train_set,
        integrator=integrator)
    writable_objects.append(integration_run)



if __name__ == "__main__":
    args = parser.parse_args()
    base_dir = pathlib.Path(args.base_dir)

    for obj in writable_objects:
        obj.write_description(base_dir)
