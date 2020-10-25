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
from systems import spring, wave
import joblib


METHOD_HNET = 5
METHOD_DIRECT_DERIV = 1


def load_network(net_dir, base_dir, eval_type, base_logger):
    logger = base_logger.getChild("load_network")
    # Load metadata
    net_dir = base_dir / pathlib.Path(net_dir)
    meta_path = net_dir / "model.json"
    with open(meta_path, "r", encoding="utf8") as meta_file:
        metadata = json.load(meta_file)
    logger.info(f"Loaded network description from {meta_path}")
    # Create network
    net = methods.build_network(metadata)
    # Load weights
    weight_path = net_dir / "model.pt"
    if eval_type == "knn-regressor":
        with open(net_dir / "model.pt", "rb") as model_file:
            net = joblib.load(model_file)
    else:
        net.load_state_dict(torch.load(net_dir / "model.pt", map_location="cpu"))
    logger.info(f"Loaded weights from {weight_path}")
    return net


def create_dataset(base_dir, data_args):
    data_dir = base_dir / data_args["data_dir"]
    data_set = dataset.TrajectoryDataset(data_dir=data_dir)
    loader = utils.data.DataLoader(data_set, batch_size=1, shuffle=False)
    return data_set, loader


def raw_err(approx, true, norm=2):
    err = np.linalg.norm(approx - true, ord=norm, axis=1)
    return err

def rel_err(approx, true, norm=2):
    num = np.linalg.norm(approx - true, ord=norm, axis=1)
    denom = np.linalg.norm(true, ord=norm, axis=1)
    return num / denom


def run_phase(base_dir, out_dir, phase_args):
    logger = logging.getLogger("eval")
    base_dir = pathlib.Path(base_dir)
    out_dir = pathlib.Path(out_dir)
    eval_args = phase_args["eval"]

    # Choose the device and dtype
    device = select_device(try_gpu=eval_args["try_gpu"], base_logger=logger)
    eval_type = eval_args["eval_type"]
    eval_dtype, eval_dtype_np = TRAIN_DTYPES[eval_args.get("eval_dtype", "float")]

    # Load the network
    net = load_network(phase_args["eval_net"], base_dir=base_dir, eval_type=eval_type, base_logger=logger)
    if eval_type != "knn-regressor":
        net = net.to(device, dtype=eval_dtype)

    # Load the evaluation data
    eval_dataset, eval_loader = create_dataset(base_dir, phase_args["eval_data"])

    # Integrate each trajectory, compute stats and store
    logger.info("Starting evaluation")
    integrator_type = eval_args["integrator"]

    if eval_type == "srnn":
        time_deriv_func = net
        time_deriv_method = METHOD_HNET
        hamiltonian_func = net
    elif eval_type == "hnn":
        def model_time_deriv(x):
            res = net.time_derivative(x)
            return res
        def model_hamiltonian(p, q):
            x = torch.cat([p, q], dim=-1)
            hamilt = net(x)
            return hamilt[0] + hamilt[1]
        time_deriv_func = model_time_deriv
        time_deriv_method = METHOD_DIRECT_DERIV
        hamiltonian_func = model_hamiltonian
    elif eval_type == "mlp":
        # Use the time_derivative
        def model_time_deriv(x):
            # x ordered (p, q)
            return net(x)
        time_deriv_func = model_time_deriv
        time_deriv_method = METHOD_DIRECT_DERIV
    elif eval_type == "knn-regressor":
        # Use the time_derivative
        def model_time_deriv(x):
            # x ordered (p, q)
            x = x.detach().cpu().numpy()
            return torch.from_numpy(net.predict(x))
        time_deriv_func = model_time_deriv
        time_deriv_method = METHOD_DIRECT_DERIV
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
        p0 = p[:, 0]
        q0 = q[:, 0]
        integrate_start = time.perf_counter()
        int_res = integrators.numerically_integrate(
            integrator_type,
            p0,
            q0,
            model=time_deriv_func,
            method=time_deriv_method,
            T=num_time_steps,
            dt=time_step_size,
            volatile=True,
            device=device,
            coarsening_factor=1).permute(1, 0, 2)
        # Remove extraneous batch dimension
        int_res = int_res[0].detach().cpu().numpy()
        integrate_elapsed = time.perf_counter() - integrate_start
        # Split the integration result
        int_p, int_q = np.split(int_res, 2, axis=1)

        # Compute errors and other statistics
        true = torch.cat([p, q], axis=-1)[0].detach().cpu().numpy()
        raw_l2 = raw_err(approx=int_res, true=true, norm=2)
        rel_l2 = rel_err(approx=int_res, true=true, norm=2)

        # Compute hamiltonians
        # Construct systems
        if eval_dataset.system == "spring":
            system = spring.SpringSystem()
        elif eval_dataset.system == "wave":
            n_grid = eval_dataset.system_metadata["n_grid"]
            space_max = eval_dataset.system_metadata["space_max"]
            wave_speed = trajectory.trajectory_meta["wave_speed"][0].item()
            system = wave.WaveSystem(n_grid=n_grid,
                                     space_max=space_max,
                                     wave_speed=wave_speed)
        else:
            raise ValueError(f"Unknown system type {eval_dataset.system}")

        # Compute true hamiltonians
        true_hamilt_true_traj = system.hamiltonian(true).squeeze()
        true_hamilt_net_traj = system.hamiltonian(int_res).squeeze()
        # Compute network hamiltonians
        net_hamilt_true_traj, net_hamilt_net_traj = None, None
        if eval_type in ("srnn", "hnn"):
            net_hamilt_true_traj = hamiltonian_func(p=p[0], q=q[0]).sum(axis=-1).detach().cpu().numpy()
            int_p_torch, int_q_torch = np.split(int_res, 2, axis=-1)
            int_p_torch = torch.from_numpy(int_p_torch).to(device, dtype=eval_dtype)
            int_q_torch = torch.from_numpy(int_q_torch).to(device, dtype=eval_dtype)
            net_hamilt_net_traj = hamiltonian_func(p=int_p_torch, q=int_q_torch).sum(axis=-1).detach().cpu().numpy()

        # All combinations true/net hamiltonian on true/net trajectories

        int_numpy_arrays = {
            traj_name: int_res,
            f"{traj_name}_relerr_l2": rel_l2,
            f"{traj_name}_raw_l2": raw_l2,
            f"{traj_name}_p": int_p,
            f"{traj_name}_q": int_q,
            f"{traj_name}_true_hamilt_true_traj": true_hamilt_true_traj,
            f"{traj_name}_true_hamilt_net_traj": true_hamilt_net_traj,
        }

        int_stats = {
            "name": traj_name,
            "file_names": {
                "traj": traj_name,
                "relerr_l2": f"{traj_name}_relerr_l2",
                "raw_l2": f"{traj_name}_raw_l2",
                "p": f"{traj_name}_p",
                "q": f"{traj_name}_q",
                "true_hamilt_true_traj": f"{traj_name}_true_hamilt_true_traj",
                "true_hamilt_net_traj": f"{traj_name}_true_hamilt_net_traj",
                "net_hamilt_true_traj": None,
                "net_hamilt_net_traj": None,
            },
            "timing": {
                "integrate_elapsed": integrate_elapsed
            }
        }

        if net_hamilt_true_traj is not None and net_hamilt_net_traj is not None:
            # We computed neural net hamiltonians, update the arrays and metadata
            int_numpy_arrays.update({
                f"{traj_name}_net_hamilt_true_traj": net_hamilt_true_traj,
                f"{traj_name}_net_hamilt_net_traj": net_hamilt_net_traj,
            })
            int_stats["file_names"]["net_hamilt_true_traj"] = f"{traj_name}_net_hamilt_true_traj"
            int_stats["file_names"]["net_hamilt_net_traj"] = f"{traj_name}_net_hamilt_net_traj"

        trajectory_results.append((traj_name, int_numpy_arrays, int_stats))
        logger.info(f"Integrated trajectory: {traj_num} ({traj_name}) in {integrate_elapsed} sec")

    # Process for save
    results_meta = {"integration_stats": []}
    saved_trajectories = {}
    for _traj_name, int_res, int_stats in trajectory_results:
        saved_trajectories.update(int_res)
        results_meta["integration_stats"].append(int_stats)

    # Save integration results
    int_traj_file = out_dir / "integrated_trajectories.npz"
    np.savez(int_traj_file, **saved_trajectories)
    logger.info(f"Saved integrated trajectories to {int_traj_file}")

    # Save metadata
    int_meta_file = out_dir / "results_meta.json"
    with open(int_meta_file, "w", encoding="utf8") as meta_file:
        json.dump(results_meta, meta_file)
    logger.info(f"Saved trajectory integration metadata to {int_meta_file}")
