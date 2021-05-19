import utils
import argparse
import pathlib
from collections import namedtuple
import itertools
import math

parser = argparse.ArgumentParser(description="Generate run descriptions")
parser.add_argument("base_dir", type=str,
                    help="Base directory for run descriptions")

experiment_general = utils.Experiment("test-ns")
writable_objects = []

train_source = utils.NavierStokesInitialConditionSource(velocity_range=(0.25, 1.75))
val_source = utils.NavierStokesInitialConditionSource(velocity_range=(0.25, 1.75))
eval_source = utils.NavierStokesInitialConditionSource(velocity_range=(0.25, 1.75))

train_set = utils.NavierStokesDataset(experiment=experiment_general,
                                      initial_cond_source=train_source,
                                      num_traj=1,
                                      num_time_steps=10,
                                      time_step_size=0.08)
val_set = utils.NavierStokesDataset(experiment=experiment_general,
                                      initial_cond_source=val_source,
                                      num_traj=3,
                                      num_time_steps=3,
                                      time_step_size=0.08)
eval_set = utils.NavierStokesDataset(experiment=experiment_general,
                                     initial_cond_source=eval_source,
                                     num_traj=4,
                                     num_time_steps=3,
                                     time_step_size=0.08)
writable_objects.extend([train_set]) # , val_set, eval_set])

# mlp_deriv_train = utils.MLP(experiment=experiment_general,
#                                    training_set=train_set,
#                                    hidden_dim=200, depth=2,
#                                    validation_set=val_set, epochs=2,
#                                    predict_type="deriv")
# mlp_deriv_eval = utils.NetworkEvaluation(experiment=experiment_general,
#                                          network=mlp_deriv_train,
#                                          eval_set=eval_set,
#                                          integrator="euler")
# writable_objects.extend([mlp_deriv_train, mlp_deriv_eval])
#
# nn_kern_step_train = utils.NNKernel(experiment=experiment_general,
#                                    training_set=train_set,
#                                    learning_rate=0.001, weight_decay=0.0001,
#                                    hidden_dim=32768, train_dtype="float",
#                                    optimizer="sgd",
#                                    batch_size=375,
#                                    validation_set=val_set, epochs=2,
#                                    predict_type="step", nonlinearity="relu")
# nn_kern_step_eval = utils.NetworkEvaluation(experiment=experiment_general,
#                                          network=nn_kern_step_train,
#                                          eval_set=eval_set,
#                                          integrator="null")
# writable_objects.extend([nn_kern_step_train, nn_kern_step_eval])


gn_train = utils.GN(experiment=experiment_general,
                        training_set=train_set,
                        validation_set=train_set,
                        batch_size=5,
                        epochs=10)
gn_eval = utils.NetworkEvaluation(experiment=experiment_general,
                                  network=gn_train,
                                  eval_set=train_set,
                                  integrator="null")
writable_objects.extend([gn_train, gn_eval])


if __name__ == "__main__":
    args = parser.parse_args()
    base_dir = pathlib.Path(args.base_dir)
    for obj in writable_objects:
        obj.write_description(base_dir)

