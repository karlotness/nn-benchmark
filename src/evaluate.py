import logging
import pathlib
import json
import time
import torch
from torch import utils
import numpy as np
from train import select_device, TRAIN_DTYPES
from train import create_dataset as train_create_dataset
import methods
import dataset
import integrators
from systems import spring, wave, particle
import joblib
from collections import namedtuple


METHOD_HNET = integrators.IntegrationScheme.HAMILTONIAN
METHOD_DIRECT_DERIV = integrators.IntegrationScheme.DIRECT_OUTPUT


def load_network(net_dir, base_dir, base_logger):
    logger = base_logger.getChild("load_network")
    if net_dir is None:
        logger.info("Skipping network load")
        return None
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
    if metadata["arch"] in {"knn-regressor", "knn-predictor"}:
        with open(net_dir / "model.pt", "rb") as model_file:
            net = joblib.load(model_file)
    else:
        net.load_state_dict(torch.load(net_dir / "model.pt", map_location="cpu"))
    logger.info(f"Loaded weights from {weight_path}")
    return net


def create_dataset(base_dir, data_args):
    dataset_type = data_args.get("dataset", None)
    if dataset_type is None:
        data_dir = base_dir / data_args["data_dir"]
        linearize = data_args.get("linearize", False)
        data_set = dataset.TrajectoryDataset(data_dir=data_dir, linearize=linearize)
        loader = utils.data.DataLoader(data_set, batch_size=1, shuffle=False)
        return data_set, loader
    else:
        # Non-default data set for evaluation
        # Use this for creating more complex datasets, like pytorch-geometric
        train_dataset, train_loader, _val_dataset, _val_loader = train_create_dataset(base_dir=base_dir, data_args=data_args)
        return train_dataset, train_loader


def raw_err(approx, true, norm=2):
    err = np.linalg.norm(approx - true, ord=norm, axis=1)
    return err


def rel_err(approx, true, norm=2):
    num = np.linalg.norm(approx - true, ord=norm, axis=1)
    denom = np.linalg.norm(true, ord=norm, axis=1)
    return num / denom


def mean_square_err(approx, true):
    diff = (approx - true)**2
    return np.mean(diff, axis=1)


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
    net = load_network(phase_args["eval_net"], base_dir=base_dir, base_logger=logger)
    if isinstance(net, (torch.nn.Module, torch.Tensor)):
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
        def model_hamiltonian(p, q):
            hamilt = net(p=p, q=q)
            return hamilt[0] + hamilt[1]

        def hnn_model_time_deriv(p, q):
            return net.time_derivative(p=p, q=q, create_graph=False)
        time_deriv_func = hnn_model_time_deriv
        time_deriv_method = METHOD_DIRECT_DERIV
        hamiltonian_func = model_hamiltonian
    elif eval_type == "mlp":
        time_deriv_func = net
        time_deriv_method = METHOD_DIRECT_DERIV
    elif eval_type == "knn-regressor":
        # Use the time_derivative
        KNNDerivative = namedtuple("KNNDerivative", ["dq_dt", "dp_dt"])
        def model_time_deriv(p, q):
            x = torch.cat([p, q], axis=-1).detach().cpu().numpy()
            ret = net.predict(x)
            dpdt, dqdt = np.split(ret, 2, axis=-1)
            dpdt = torch.from_numpy(dpdt).to(device, dtype=eval_dtype)
            dqdt = torch.from_numpy(dqdt).to(device, dtype=eval_dtype)
            return KNNDerivative(dq_dt=dqdt, dp_dt=dpdt)
        time_deriv_func = model_time_deriv
        time_deriv_method = METHOD_DIRECT_DERIV
    elif eval_type == "knn-predictor":
        KNNPrediction = namedtuple("KNNPrediction", ["q", "p"])
        def model_next_step(p, q):
            x = torch.cat([p, q], axis=-1).detach().cpu().numpy()
            ret = net.predict(x)
            next_p, next_q = np.split(ret, 2, axis=-1)
            next_p = torch.from_numpy(next_p).to(device, dtype=eval_dtype)
            next_q = torch.from_numpy(next_q).to(device, dtype=eval_dtype)
            return KNNPrediction(q=next_q, p=next_p)
        time_deriv_func = model_next_step
        time_deriv_method = METHOD_DIRECT_DERIV
        if integrator_type != "null":
            raise ValueError(f"KNN predictions to not work with integrator {integrator_type}")
    elif eval_type == "integrator-baseline":
        # Use the time_derivative
        SystemDerivative = namedtuple("SystemDerivative", ["dq_dt", "dp_dt"])
        def system_derivative(p, q):
            p = p.detach().cpu().numpy()
            q = q.detach().cpu().numpy()
            derivative = system.derivative(p=p, q=q)
            dp_dt = torch.from_numpy(derivative.p).to(device, dtype=eval_dtype)
            dq_dt = torch.from_numpy(derivative.q).to(device, dtype=eval_dtype)
            return SystemDerivative(dp_dt=dp_dt, dq_dt=dq_dt)
        time_deriv_func = system_derivative
        time_deriv_method = METHOD_DIRECT_DERIV
    elif eval_type == "hogn":
        # Lazy import to avoid pytorch-geometric if possible
        from methods import hogn
        import dataset_geometric

        package_args = eval_args["package_args"]
        HognMockDataset = namedtuple("HognMockDataset", ["p", "q", "dp_dt", "dq_dt", "masses"])

        def hogn_time_deriv_func(masses):
            def model_time_deriv(p, q):
                mocked = HognMockDataset(p=p, q=q, masses=masses,
                                         dp_dt=None, dq_dt=None)
                bundled = dataset_geometric(dataset=[mocked],
                                            package_args=package_args)[0]
                derivs = net.just_derivative(bundled)
                unbundled = hogn.unpackage_time_derivative(input_data=bundled,
                                                           deriv=derivs)
                return unbundled
            return model_time_deriv

        def hogn_hamiltonian_func(masses):
            def model_hamiltonian(p, q):
                hamilts = []
                for i in range(p.shape[0]):
                    # Break out batches separately
                    mocked = HognMockDataset(p=p[i], q=q[i], masses=masses[i],
                                             dp_dt=None, dq_dt=None)
                    bundled = dataset_geometric(dataset=[mocked],
                                                package_args=package_args)[0]
                    h = net(bundled).sum()
                    hamilts.append(h)
                return np.array(hamilts)
            return model_hamiltonian

        time_deriv_method = METHOD_DIRECT_DERIV
    else:
        logger.error(f"Invalid evaluation type: {eval_type}")
        raise ValueError(f"Invalid evaluation type: {eval_type}")

    trajectory_results = []

    for traj_num, trajectory in enumerate(eval_loader):
        traj_name = trajectory.name[0]
        p = trajectory.p.to(device, dtype=eval_dtype)
        q = trajectory.q.to(device, dtype=eval_dtype)
        p_noiseless = trajectory.p_noiseless.to(device, dtype=eval_dtype)
        q_noiseless = trajectory.q_noiseless.to(device, dtype=eval_dtype)
        masses = trajectory.masses.to(device, dtype=eval_dtype)
        num_time_steps = trajectory.trajectory_meta["num_time_steps"][0]
        time_step_size = trajectory.trajectory_meta["time_step_size"][0]

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
        elif eval_dataset.system == "particle":
            n_dim = eval_dataset.system_metadata["n_dim"]
            n_particles = eval_dataset.system_metadata["n_particles"]
            g = eval_dataset.system_metadata["g"]
            system = particle.ParticleSystem(n_particles=n_particles,
                                             n_dim=n_dim, g=g)
        else:
            raise ValueError(f"Unknown system type {eval_dataset.system}")

        if eval_type == "hogn":
            # Pull out masses for HOGN
            time_deriv_func = hogn_time_deriv_func(masses=masses)
            hamiltonian_func = hogn_hamiltonian_func(masses=masses)

        p0 = p[:, 0]
        q0 = q[:, 0]
        integrate_start = time.perf_counter()
        int_res_raw = integrators.numerically_integrate(
            integrator_type,
            p_0=p0,
            q_0=q0,
            model=time_deriv_func,
            method=time_deriv_method,
            T=num_time_steps,
            dt=time_step_size,
            volatile=True,
            device=device,
            coarsening_factor=1)
        # Remove extraneous batch dimension
        integrate_elapsed = time.perf_counter() - integrate_start
        # Split the integration result
        int_p = int_res_raw.p
        int_q = int_res_raw.q

        # Compute errors and other statistics
        int_res = torch.cat([int_p, int_q], axis=-1)[0].detach().cpu().numpy()
        true = torch.cat([p_noiseless, q_noiseless], axis=-1)[0].detach().cpu().numpy()
        raw_l2 = raw_err(approx=int_res, true=true, norm=2)
        rel_l2 = rel_err(approx=int_res, true=true, norm=2)
        mse_err = mean_square_err(approx=int_res, true=true)

        int_p = int_res_raw.p[0].detach().cpu().numpy()
        int_q = int_res_raw.q[0].detach().cpu().numpy()

        # Compute true hamiltonians
        additional_hamilt_args = {}
        if isinstance(system, particle.ParticleSystem):
            additional_hamilt_args = {
                "masses": masses.detach().cpu().numpy(),
            }
        true_hamilt_true_traj = system.hamiltonian(p=p_noiseless.cpu().numpy()[0],
                                                   q=q_noiseless.cpu().numpy()[0],
                                                   **additional_hamilt_args).squeeze()
        true_hamilt_net_traj = system.hamiltonian(p=int_p, q=int_q,
                                                  **additional_hamilt_args).squeeze()
        # Compute network hamiltonians
        net_hamilt_true_traj, net_hamilt_net_traj = None, None
        if eval_type in {"srnn", "hnn", "hogn"}:
            net_hamilt_true_traj = hamiltonian_func(p=p[0], q=q[0]).sum(axis=-1).detach().cpu().numpy()
            int_p_torch = torch.from_numpy(int_p).to(device, dtype=eval_dtype)
            int_q_torch = torch.from_numpy(int_q).to(device, dtype=eval_dtype)
            net_hamilt_net_traj = hamiltonian_func(p=int_p_torch, q=int_q_torch).sum(axis=-1).detach().cpu().numpy()

        # All combinations true/net hamiltonian on true/net trajectories

        int_numpy_arrays = {
            traj_name: int_res,
            f"{traj_name}_relerr_l2": rel_l2,
            f"{traj_name}_raw_l2": raw_l2,
            f"{traj_name}_mse": mse_err,
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
                "mse": f"{traj_name}_mse",
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
            int_stats["file_names"].update({
                "net_hamilt_true_traj": f"{traj_name}_net_hamilt_true_traj",
                "net_hamilt_net_traj": f"{traj_name}_net_hamilt_net_traj",
            })

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
