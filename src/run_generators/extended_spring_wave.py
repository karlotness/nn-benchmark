import argparse
import pathlib
import json
import numpy as np
import itertools
import copy


parser = argparse.ArgumentParser(description="Generate run descriptions")
parser.add_argument("base_dir", type=str,
                    help="Base directory for run descriptions")


EXPERIMENT_NAME_BASE = "extended-spring-wave"

WAVE_HEIGHT_RANGE = (0.5, 1.5)
WAVE_WIDTH_RANGE = (0.5, 1.5)
WAVE_SPEED = 0.1
WAVE_N_GRID = 250
WAVE_DELTA_T = 0.1
WAVE_SUBSAMPLE = 10
WAVE_POSITION_RANGE = (0.35, 0.65)

SPRING_RADIUS_RANGE = (0.2, 2)
SPRING_DELTA_T = 0.1

TRAIN_SET_TRAJ_COUNTS = [25, 100, 250, 500, 750, 1000]
EVAL_SET_NUM_TRAJ = 50
ROLLOUT_STEPS = 1000

# (depth, hidden)
NET_DEPTHS_CONFIGS = [(2, 256), (2, 512), (2, 1024),
                      (3, 256), (3, 512), (3, 1024),
                      (5, 256), (5, 512), (5, 1024)]

LEARNING_RATE = 1e-3
EPOCHS = 1000
TRY_GPU = True
HNN_BATCH = 750
SRNN_BATCH = 750
SRNN_ROLLOUT_LENGTH = 10


def sample_ring_uniform(inner_r, outer_r, num_pts):
    theta = np.random.uniform(0, 2*np.pi, num_pts)
    r = np.sqrt(np.random.uniform(size=num_pts) * (outer_r**2 - inner_r**2) + inner_r**2)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack((x, y), axis=-1)


def create_spring_trajectories():
    max_traj = max(TRAIN_SET_TRAJ_COUNTS) + EVAL_SET_NUM_TRAJ
    initial_conds = sample_ring_uniform(SPRING_RADIUS_RANGE[0],
                                        SPRING_RADIUS_RANGE[1],
                                        max_traj)
    # Wrap the initial conditions in spring metadata
    traj_descrs = []
    for i in range(max_traj):
        traj_descrs.append({
            "initial_condition": list(initial_conds[i]),
            "num_time_steps": ROLLOUT_STEPS,
            "time_step_size": SPRING_DELTA_T,
        })
    # Split into test & eval and return
    train_traj = traj_descrs[:-EVAL_SET_NUM_TRAJ]
    eval_traj = traj_descrs[-EVAL_SET_NUM_TRAJ:]
    return train_traj, eval_traj

def create_spring_training_sets():
    train_traj, eval_traj = create_spring_trajectories()
    template = {
        "out_dir": None,
        "exp_name": EXPERIMENT_NAME_BASE,
        "run_name": None,
        "phase": "data_gen",
        "phase_args": {
            "system": "spring",
            "system_args": {
                "trajectory_defs": None
            }
        },
        "slurm_args": {
            "gpu": False,
            "time": "00:30:00",
            "cpus": 8,
            "mem": 8,
        },
    }
    train_sets = []
    # Create training templates
    for train_size in TRAIN_SET_TRAJ_COUNTS:
        out_dir = f"run/data_gen/{EXPERIMENT_NAME_BASE}_train-spring-{train_size}"
        traj_defs = train_traj[:train_size]
        data = copy.deepcopy(template)
        data["out_dir"] = out_dir
        data["run_name"] = f"{EXPERIMENT_NAME_BASE}-train-spring-{train_size}"
        data["phase_args"]["system_args"]["trajectory_defs"] = traj_defs
        train_sets.append((f"descr/data_gen/{EXPERIMENT_NAME_BASE}_train-spring-{train_size}.json", data, out_dir))

    # Create eval template
    out_dir = f"run/data_gen/{EXPERIMENT_NAME_BASE}_eval-spring"
    data = copy.deepcopy(template)
    data["out_dir"] = out_dir
    data["run_name"] = f"{EXPERIMENT_NAME_BASE}-eval-spring"
    data["phase_args"]["system_args"]["trajectory_defs"] = eval_traj
    eval_set = (f"descr/data_gen/{EXPERIMENT_NAME_BASE}_eval-spring.json", data, out_dir)

    return train_sets, eval_set


def create_wave_trajectories():
    max_traj = max(TRAIN_SET_TRAJ_COUNTS) + EVAL_SET_NUM_TRAJ
    traj_descrs = []
    for i in range(max_traj):
        width = np.random.uniform(WAVE_WIDTH_RANGE[0], WAVE_WIDTH_RANGE[1])
        height = np.random.uniform(WAVE_HEIGHT_RANGE[0], WAVE_HEIGHT_RANGE[1])
        position = np.random.uniform(WAVE_POSITION_RANGE[0], WAVE_POSITION_RANGE[1])
        traj_desc = {
            "start_type": "cubic_splines",
            "start_type_args": {
                "width": width,
                "height": height,
                "position": position
            },
            "wave_speed": WAVE_SPEED,
            "num_time_steps": ROLLOUT_STEPS,
            "time_step_size": WAVE_DELTA_T,
            "subsample": WAVE_SUBSAMPLE,
        }
        traj_descrs.append(traj_desc)
    # Split into test & eval and return
    train_traj = traj_descrs[:-EVAL_SET_NUM_TRAJ]
    eval_traj = traj_descrs[-EVAL_SET_NUM_TRAJ:]
    return train_traj, eval_traj


def create_wave_training_sets():
    train_traj, eval_traj = create_wave_trajectories()
    template = {
        "out_dir": None,
        "exp_name": EXPERIMENT_NAME_BASE,
        "run_name": None,
        "phase": "data_gen",
        "phase_args": {
            "system": "wave",
            "system_args": {
                "space_max": 1,
                "n_grid": WAVE_N_GRID,
                "trajectory_defs": None
            }
        },
        "slurm_args": {
            "gpu": False,
            "time": "03:00:00",
            "cpus": 8,
            "mem": 32,
        },
    }
    train_sets = []
    # Create training templates
    for train_size in TRAIN_SET_TRAJ_COUNTS:
        out_dir = f"run/data_gen/{EXPERIMENT_NAME_BASE}_train-wave-{train_size}"
        traj_defs = train_traj[:train_size]
        data = copy.deepcopy(template)
        data["out_dir"] = out_dir
        data["run_name"] = f"{EXPERIMENT_NAME_BASE}-train-wave-{train_size}"
        data["phase_args"]["system_args"]["trajectory_defs"] = traj_defs
        train_sets.append((f"descr/data_gen/{EXPERIMENT_NAME_BASE}_train-wave-{train_size}.json", data, out_dir))

    # Create eval template
    out_dir = f"run/data_gen/{EXPERIMENT_NAME_BASE}_eval-wave"
    data = copy.deepcopy(template)
    data["out_dir"] = out_dir
    data["run_name"] = f"{EXPERIMENT_NAME_BASE}-eval-wave"
    data["phase_args"]["system_args"]["trajectory_defs"] = eval_traj
    eval_set = (f"descr/data_gen/{EXPERIMENT_NAME_BASE}_eval-wave.json", data, out_dir)

    return train_sets, eval_set


def create_training_run(train_data_dir, training_type, net_architecture, name_key):
    out_dir = f"run/train/{EXPERIMENT_NAME_BASE}_train-{name_key}"
    out_file = f"descr/train/{EXPERIMENT_NAME_BASE}_train-{name_key}.json"
    train_type_args = {}
    dataset_type = "snapshot"
    dataset_args = {}
    loader_args = {
        "batch_size": HNN_BATCH,
        "shuffle": True,
    }
    if training_type == "srnn":
        train_type_args = {
            "rollout_length": SRNN_ROLLOUT_LENGTH,
            "integrator": "leapfrog",
        }
        dataset_type = "rollout-chunk"
        dataset_args = {
            "rollout_length": SRNN_ROLLOUT_LENGTH
        }
        loader_args = {
            "batch_size": SRNN_BATCH,
            "shuffle": True,
        }
    template = {
        "out_dir": out_dir,
        "exp_name": EXPERIMENT_NAME_BASE,
        "run_name": f"{EXPERIMENT_NAME_BASE}_train-{name_key}",
        "phase": "train",
        "phase_args": {
            "network": net_architecture,
            "training": {
                "optimizer": "adam",
                "optimizer_args": {
                    "learning_rate": LEARNING_RATE,
                },
                "max_epochs": EPOCHS,
                "try_gpu": (TRY_GPU and training_type != "knn-regressor"),
                "train_dtype": "float",
                "train_type": training_type,
                "train_type_args": train_type_args,
            },
            "train_data": {
                "data_dir": train_data_dir,
                "dataset": dataset_type,
                "dataset_args": dataset_args,
                "loader": loader_args,
            }
        },
        "slurm_args": {
            "gpu": (training_type != "knn-regressor"),  # GPU only if not KNN run
            "time": "08:00:00",
            "cpus": 8,
            "mem": 32,
        },
    }
    return out_file, template, out_dir


def create_mlp_net(input_dim, hidden_dim, output_dim, depth):
    return {
        "arch": "mlp",
        "arch_args": {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "depth": depth,
            "nonlinearity": "tanh",
        }
    }


def create_hnn_net(input_dim, hidden_dim, output_dim, depth):
    return {
        "arch": "hnn",
        "arch_args": {
            "base_model": "mlp",
            "input_dim": input_dim,
            "base_model_args": {
                "hidden_dim": hidden_dim,
                "output_dim": output_dim,
                "depth": depth,
                "nonlinearity": "tanh",
            },
            "hnn_args": {
                "field_type": "solenoidal",
            }
        }
    }


def create_srnn_net(input_dim, hidden_dim, output_dim, depth):
    return {
        "arch": "srnn",
        "arch_args": {
            "base_model": "mlp",
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "depth": depth,
            "nonlinearity": "tanh",
        }
    }


def create_knn_regressor_net(input_dim, output_dim):
    return {
        "arch": "knn-regressor",
        "arch_args": {
            "input_dim": input_dim,
            "output_dim": output_dim,
        }
    }


def create_train_runs(train_sets):
    types = {
        "srnn": create_srnn_net,
        "hnn": create_hnn_net,
        "mlp": create_mlp_net,
        "knn-regressor": create_knn_regressor_net,
    }
    train_sets = list(train_sets)
    train_runs = []
    for net_type in ["srnn", "hnn", "mlp", "knn-regressor"]:
        net_func = types[net_type]
        for _dest_file, _contents, train_out_dir in train_sets:
            train_set_size = int(train_out_dir.split("-")[-1])
            net_configs = NET_DEPTHS_CONFIGS
            if net_type == "knn-regressor":
                net_configs = [(None, None)]
            for depth, hidden in net_configs:
                base_data_name = str(train_out_dir).split("_")[-1]
                system_type = "spring"
                input_dim = 2
                if "wave" in str(base_data_name):
                    system_type = "wave"
                    input_dim = 2 * WAVE_N_GRID
                if net_type == "srnn":
                    input_dim = input_dim // 2
                output_dim = 2
                if net_type == "mlp":
                    output_dim = input_dim
                if net_type == "knn-regressor":
                    net = net_func(input_dim=input_dim,
                                   output_dim=output_dim)
                    name_key = f"{system_type}-{net_type}-{train_set_size}"
                else:
                    net = net_func(input_dim=input_dim,
                                   hidden_dim=hidden,
                                   output_dim=output_dim,
                                   depth=depth)
                    name_key = f"{system_type}-{net_type}-d{depth}-h{hidden}-{train_set_size}"
                run = create_training_run(train_data_dir=train_out_dir,
                                          training_type=net_type,
                                          net_architecture=net,
                                          name_key=name_key)
                train_runs.append(run)
    return train_runs


def create_eval_run(trained_net, eval_set, eval_type):
    name_suffix = "-".join(trained_net.split("_")[1].split("-")[1:])
    file_name = f"{EXPERIMENT_NAME_BASE}_eval-{name_suffix}"
    template = {
        "out_dir": f"run/eval/{EXPERIMENT_NAME_BASE}_eval-{name_suffix}",
        "run_name": f"{EXPERIMENT_NAME_BASE}_eval-{name_suffix}",
        "exp_name": EXPERIMENT_NAME_BASE,
        "phase": "eval",
        "phase_args": {
            "eval_net": trained_net,
            "eval_data": {
                "data_dir": eval_set,
            },
            "eval": {
                "eval_type": eval_type,
                "integrator": "leapfrog",
                "eval_dtype": "float",
                "try_gpu": False
            }
        },
        "slurm_args": {
            "gpu": False,  # GPU only if not KNN run
            "time": "02:00:00",
            "cpus": 8,
            "mem": 32,
        },
    }
    return file_name, template


def write_json_file(base_dir, dest_file, contents):
    dest_file = pathlib.Path(dest_file)
    full_path = base_dir / dest_file
    # Create directory
    full_path.parent.mkdir(parents=True, exist_ok=True)
    with open(full_path, "w", encoding="utf8") as out_file:
        json.dump(contents, out_file)


if __name__ == "__main__":
    args = parser.parse_args()
    base_dir = pathlib.Path(args.base_dir)

    np.random.seed(1)
    # Generate data tasks
    wave_train_sets, wave_eval_set = create_wave_training_sets()
    spring_train_sets, spring_eval_set = create_spring_training_sets()

    # Save data tasks
    for dest_file, contents, _out_dir in itertools.chain(wave_train_sets, spring_train_sets,
                                                         [wave_eval_set, spring_eval_set]):
        write_json_file(base_dir, dest_file, contents)

    # Generate network training tasks
    # Vary network architectures (MLP hidden dim and depth. All combinations)
    training_tasks = create_train_runs(itertools.chain(wave_train_sets,
                                                       spring_train_sets))

    # Save network training tasks
    for dest_file, contents, _out_dir in training_tasks:
        write_json_file(base_dir, dest_file, contents)

    # Generate evaluation tasks
    # Evaluate all networks on the single evaluation set for each system
    for _dest_file, contents, out_dir in training_tasks:
        trained_net = out_dir
        base_data_name = str(out_dir).split("_")[-1]
        eval_type = contents["phase_args"]["training"]["train_type"]
        system_type = "spring"
        if "wave" in str(base_data_name):
            system_type = "wave"

        if system_type == "spring":
            eval_set = spring_eval_set[2]
        elif system_type == "wave":
            eval_set = wave_eval_set[2]
        file_name, eval_contents = create_eval_run(trained_net, eval_set, eval_type)
        write_json_file(base_dir, "descr/eval/"+file_name+".json", eval_contents)

    # Save evaluation training tasks
