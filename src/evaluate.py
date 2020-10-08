import logging
import pathlib
import json
import time
import torch
from torch import utils
import numpy as np
from train import select_device, TRAIN_DTYPES
import methods
import dataset
import integrators


METHOD_HNET = 5


def load_network(net_dir, base_logger):
    logger = base_logger.getChild("load_network")
    # Load metadata
    meta_path = net_dir / "model.json"
    with open(meta_path, "r", encoding="utf8") as meta_file:
        metadata = json.load(meta_file)
    logger.info(f"Loaded network description from {meta_path}")
    # Create network
    net = methods.build_network(metadata)
    # Load weights
    weight_path = net_dir / "model.pt"
    net.load_state_dict(torch.load(net_dir / "model.pt", map_location="cpu"))
    logger.info(f"Loaded weights from {weight_path}")
    return net


def create_dataset(base_dir, data_args):
    data_dir = base_dir / data_args["data_dir"]
    data_set = dataset.TrajectoryDataset(data_dir=data_dir)
    loader = utils.DataLoader(data_set, batch_size=1, shuffle=False)
    return data_set, loader


def run_phase(base_dir, out_dir, phase_args):
    logger = logging.getLogger("eval")
    base_dir = pathlib.Path(base_dir)
    out_dir = pathlib.Path(out_dir)
    eval_args = phase_args["eval"]

    # Choose the device and dtype
    device = select_device(try_gpu=eval_args["try_gpu"], base_logger=logger)
    eval_dtype = TRAIN_DTYPES[eval_args.get("train_dtype", "float")]

    # Load the network
    net = load_network(phase_args["eval_net"], base_logger=logger)
    net = net.to(device, dtype=eval_dtype)

    # Load the evaluation data
    eval_dataset, eval_loader = create_dataset(base_dir, phase_args["eval_data"])

    # Integrate each trajectory, compute stats and store
    logger.info("Starting evaluation")
    eval_type = eval_args["eval_type"]
    integrator_type = eval_args["integrator"]

    if eval_type == "hnn":
        time_deriv_func = net
    elif eval_type == "srnn":
        def model_hamiltonian(p, q):
            stacked_input = torch.stack([p, q], dim=-1)
            return net(stacked_input, split=False)
        time_deriv_func = model_hamiltonian
    else:
        logger.error(f"Invalid evaluation type: {eval_type}")
        raise ValueError(f"Invalid evaluation type: {eval_type}")

    trajectory_results = []

    for traj_num, trajectory in enumerate(eval_loader):
        traj_name = trajectory.name[0]
        p = trajectory.p.to(device, dtype=eval_dtype)
        q = trajectory.q.to(device, dtype=eval_dtype)
        num_time_steps = trajectory.trajectory_meta["num_time_steps"][0]
        time_step_size = trajectory.trajectory_meta["time_step_size"][0]
        p0 = p[0].unsqueeze(0)
        q0 = q[0].unsqueeze(0)
        integrate_start = time.perf_counter()
        int_res = integrators.numerically_integrate(
            integrator_type,
            p0,
            q0,
            model=time_deriv_func,
            method=METHOD_HNET,
            T=num_time_steps,
            dt=time_step_size,
            volatile=True,
            device=device,
            coarsening_factor=1).permute(1, 0, 2)
        integrate_elapsed = time.perf_counter() - integrate_start
        trajectory_results.append((traj_name, int_res, integrate_elapsed))
        logger.info(f"Integrated trajectory: {traj_num} ({traj_name}) in {integrate_elapsed} sec")

    # Process for save
    results_meta = {"integration_stats": []}
    saved_trajectories = {}
    for traj_name, int_res, int_elapsed_time in trajectory_results:
        saved_trajectories[traj_name] = int_res
        results_meta["integration_stats"].append({
            "name": traj_name,
            "elapsed_time": int_elapsed_time,
        })

    # Save integration results
    int_traj_file = out_dir / "integrated_trajectories.npz"
    np.savez(int_traj_file, **saved_trajectories)
    logger.info(f"Saved integrated trajectories to {int_traj_file}")

    # Save metadata
    int_meta_file = out_dir / "results_meta.json",
    with open(int_meta_file, "w", encoding="utf8") as meta_file:
        json.dump(results_meta, meta_file)
    logger.info(f"Saved trajectory integration metadata to {int_meta_file}")
