import utils
import argparse
import pathlib
import itertools
import math

WAVE_N_GRID = 125
WAVE_DT = 0.1 / 250
WAVE_STEPS = 50 * 250
WAVE_SUBSAMPLE = 1000 // 250
WAVE_N_GRID = 125

parser = argparse.ArgumentParser(description="Generate run descriptions")
parser.add_argument("base_dir", type=str,
                    help="Base directory for run descriptions")
parser.add_argument("--out_dir", type=str, default=None,
                    help="Directory for new descriptions")

args = parser.parse_args()
base_dir = pathlib.Path(args.base_dir)
if args.out_dir is None:
    out_dir = base_dir
else:
    out_dir = pathlib.Path(args.out_dir)

existing_networks = []
writable_objects = []

for net_path in (base_dir / "descr" / "train").glob("*.json"):
    net = utils.ExistingNetwork(net_path, base_dir)
    if net.method == "hnn":
        existing_networks.append(net)

# Next, request generation of out of distribution wave data
experiment = utils.Experiment("backfill-wave-outdist")
eval_source = utils.WaveDisjointInitialConditionSource(height_range=[(0.5, 0.75), (1.25, 1.5)],
                                                       width_range=[(0.5, 0.75), (1.25, 1.5)],)
time_step_size = WAVE_DT * 1
num_time_steps = math.ceil(WAVE_STEPS / 1)
wave_subsample = math.ceil(WAVE_SUBSAMPLE * 1)
eval_set = utils.WaveDataset(experiment=experiment,
                             initial_cond_source=eval_source,
                             num_traj=6,
                             set_type="eval-dtfactor1-outdist",
                             n_grid=WAVE_N_GRID,
                             num_time_steps=num_time_steps,
                             time_step_size=time_step_size,
                             wave_speed=0.1,
                             subsampling=wave_subsample)
writable_objects.append(eval_set)

for net in existing_networks:
    # Request evaluation
    for eval_integrator in ["leapfrog", "euler", "rk4", "scipy-RK45"]:
        hnn_eval = utils.NetworkEvaluation(experiment=experiment,
                                           network=net,
                                           eval_set=eval_set,
                                           integrator=eval_integrator)
        writable_objects.append(hnn_eval)

for obj in writable_objects:
    obj.write_description(out_dir)
