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
from systems import spring, wave, spring_mesh, navier_stokes
from collections import namedtuple
import math


class TrajTimeCollector:
    def __init__(self):
        self.times = {}
        self.current_traj = None

    def start_traj(self, traj_num):
        self.current_traj = traj_num
        if self.current_traj not in self.times:
            self.times[self.current_traj] = 0

    def accumulate_time(self, time):
        self.times[self.current_traj] += time

    def get_time(self, traj_num):
        return self.times.get(traj_num)


def load_network(net_dir, base_dir, base_logger, model_file_name="model.pt"):
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
    weight_path = net_dir / model_file_name
    net.load_state_dict(torch.load(weight_path, map_location="cpu"))
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


def train_knn(net, eval_args, base_dir, base_logger):
    # Load the training data
    logger = base_logger.getChild("knn_train")
    eval_type = eval_args["eval_type"]
    if eval_type == "knn-predictor-oneshot":
        assert eval_args["train_data"]["dataset"] == "step-snapshot"
        assert eval_args["train_data"]["loader"].get("type", "pytorch") == "pytorch"
        assert eval_args["train_data"]["loader"]["batch_size"] == 1

    train_dataset, train_loader = create_dataset(base_dir, eval_args["train_data"])
    # Train the KNN
    if eval_type == "knn-regressor-oneshot":
        logger.info("Starting fitting of dataset for KNN Regressor.")
        data_x = []
        data_y = []
        for batch in train_loader:
            data_x.append(np.concatenate([batch.p, batch.q], axis=-1))
            data_y.append(np.concatenate([batch.dp_dt, batch.dq_dt], axis=-1))
        data_x = np.concatenate(data_x, axis=0)
        data_y = np.concatenate(data_y, axis=0)
        net.fit(data_x, data_y)
        logger.info("Finished fitting of dataset for KNN Regressor.")
    elif eval_type == "knn-predictor-oneshot":
        logger.info("Starting fitting of dataset for KNN Predictor.")
        data_x = []
        data_y = []
        for batch in train_loader:
            assert batch.p.shape[0] == 1
            assert batch.q.shape[0] == 1
            assert batch.p_noiseless.shape[0] == 1
            assert batch.q_noiseless.shape[0] == 1
            data_x.append(np.concatenate([batch.p[0], batch.q[0]], axis=-1))
            data_y.append(np.concatenate([batch.p_step[0], batch.q_step[0]], axis=-1))
        data_x = np.stack(data_x, axis=0)
        data_y = np.stack(data_y, axis=0)
        net.fit(data_x, data_y)
        logger.info("Finished fitting of dataset for KNN Predictor.")
    # Return the KNN
    return net


def run_phase(base_dir, out_dir, phase_args):
    logger = logging.getLogger("eval")
    base_dir = pathlib.Path(base_dir)
    out_dir = pathlib.Path(out_dir)
    eval_args = phase_args["eval"]
    eval_only_time_collector = TrajTimeCollector()

    # Choose the device and dtype
    device = select_device(try_gpu=eval_args["try_gpu"], base_logger=logger)
    eval_type = eval_args["eval_type"]
    eval_dtype, eval_dtype_np = TRAIN_DTYPES[eval_args.get("eval_dtype", "float")]
    baseline_stride = int(eval_args.get("coarsening", 1))

    # Load the network
    if eval_type in {"knn-regressor-oneshot", "knn-predictor-oneshot"}:
        logger.info(f"Doing \"oneshot\" KNN training: {eval_type}")
        net = methods.build_network(net_args={"arch": eval_type, "arch_args": None})
    else:
        # Normal network load process
        net = load_network(phase_args["eval_net"], base_dir=base_dir, base_logger=logger,
                           model_file_name=phase_args.get("eval_net_file", "model.pt"))

    # Send network to correct device if PyTorch
    if isinstance(net, (torch.nn.Module, torch.Tensor)):
        net = net.to(device, dtype=eval_dtype)
        net.eval()

    # Handle possible knn "oneshot" training before evaluation
    if eval_type in {"knn-regressor-oneshot", "knn-predictor-oneshot"}:
        net = train_knn(net=net, eval_args=eval_args,
                        base_dir=base_dir, base_logger=logger)

    # Load the evaluation data
    eval_dataset, eval_loader = create_dataset(base_dir, phase_args["eval_data"])

    # Integrate each trajectory, compute stats and store
    logger.info("Starting evaluation")
    integrator_type = eval_args["integrator"]

    if eval_type in {"mlp-deriv", "nn-kernel-deriv"}:
        def mlp_time_deriv_func(batch):
            MLPDerivative = namedtuple("MLPDerivative", ["dq_dt", "dp_dt"])

            extra_data = batch.extra_fixed_mask
            if torch.is_tensor(extra_data):
                extra_data = extra_data[0].to(device, dtype=eval_dtype)
                extra_data.requires_grad = False
            else:
                extra_data = None

            def net_no_grad(q, p, dt=1.0, t=0):
                with torch.no_grad():
                    q = torch.from_numpy(q).to(device, dtype=eval_dtype)
                    p = torch.from_numpy(p).to(device, dtype=eval_dtype)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_start = time.perf_counter()
                    ret = net(q=q, p=p, extra_data=extra_data)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_end = time.perf_counter()
                    eval_only_time_collector.accumulate_time(t_end - t_start)
                    dq = ret.dq_dt.detach().cpu().numpy()
                    dp = ret.dp_dt.detach().cpu().numpy()
                    return MLPDerivative(dq_dt=dq, dp_dt=dp)

            return net_no_grad

    elif eval_type in {"mlp-step", "nn-kernel-step"}:
        def mlp_time_deriv_func(batch):
            MLPPrediction = namedtuple("MLPPrediction", ["q", "p"])

            extra_data = batch.extra_fixed_mask
            if torch.is_tensor(extra_data):
                extra_data = extra_data[0].to(device, dtype=eval_dtype)
                extra_data.requires_grad = False
            else:
                extra_data = None

            def model_next_step(q, p, dt=1.0, t=0):
                with torch.no_grad():
                    q = torch.from_numpy(q).to(device, dtype=eval_dtype)
                    p = torch.from_numpy(p).to(device, dtype=eval_dtype)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_start = time.perf_counter()
                    ret = net(q=q, p=p, extra_data=extra_data)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_end = time.perf_counter()
                    eval_only_time_collector.accumulate_time(t_end - t_start)
                    next_q = ret.q.detach().cpu().numpy()
                    next_p = ret.p.detach().cpu().numpy()
                    return MLPPrediction(q=next_q, p=next_p)

            return model_next_step
        if integrator_type != "null":
            raise ValueError(f"mlp-step predictions do not work with integrator {integrator_type}")
    elif eval_type in {"cnn-deriv", "unet-deriv"}:
        def cnn_time_deriv_func(batch):
            CNNDerivative = namedtuple("CNNDerivative", ["dq_dt", "dp_dt"])

            p_full_shape = (1, ) + batch.p[0, 0].shape
            q_full_shape = (1, ) + batch.q[0, 0].shape

            if len(q_full_shape) < 3:
                q_full_shape = q_full_shape + (1, )
            if len(p_full_shape) < 3:
                p_full_shape = p_full_shape + (1, )

            extra_data = batch.extra_fixed_mask
            if torch.is_tensor(extra_data):
                extra_data = extra_data[0].to(device, dtype=eval_dtype)
                if extra_data.ndim < 3:
                    extra_data = extra_data.unsqueeze(-1)
                extra_data.requires_grad = False
            else:
                extra_data = None

            def cnn_time_deriv(q, p, dt=1.0, t=0):
                with torch.no_grad():
                    q_orig_shape = q.shape
                    p_orig_shape = p.shape
                    q = torch.from_numpy(q).to(device, dtype=eval_dtype).reshape(q_full_shape)
                    p = torch.from_numpy(p).to(device, dtype=eval_dtype).reshape(p_full_shape)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_start = time.perf_counter()
                    res = net(p=p, q=q, extra_data=extra_data)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_end = time.perf_counter()
                    eval_only_time_collector.accumulate_time(t_end - t_start)
                    dq_dt = res.dq_dt.reshape(q_orig_shape).cpu().numpy()
                    dp_dt = res.dp_dt.reshape(p_orig_shape).cpu().numpy()
                    return CNNDerivative(dq_dt=dq_dt, dp_dt=dp_dt)

            return cnn_time_deriv

    elif eval_type in {"cnn-step", "unet-step"}:
        def cnn_time_deriv_func(batch):
            CNNPrediction = namedtuple("CNNPrediction", ["q", "p"])

            p_full_shape = (1, ) + batch.p[0, 0].shape
            q_full_shape = (1, ) + batch.q[0, 0].shape

            if len(q_full_shape) < 3:
                q_full_shape = q_full_shape + (1, )
            if len(p_full_shape) < 3:
                p_full_shape = p_full_shape + (1, )

            extra_data = batch.extra_fixed_mask
            if torch.is_tensor(extra_data):
                extra_data = extra_data[0].to(device, dtype=eval_dtype)
                if extra_data.ndim < 3:
                    extra_data = extra_data.unsqueeze(-1)
                extra_data.requires_grad = False
            else:
                extra_data = None

            def model_next_step(q, p, dt=1.0, t=0):
                with torch.no_grad():
                    q_orig_shape = q.shape
                    p_orig_shape = p.shape
                    q = torch.from_numpy(q).to(device, dtype=eval_dtype).reshape(q_full_shape)
                    p = torch.from_numpy(p).to(device, dtype=eval_dtype).reshape(p_full_shape)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_start = time.perf_counter()
                    res = net(p=p, q=q, extra_data=extra_data)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_end = time.perf_counter()
                    eval_only_time_collector.accumulate_time(t_end - t_start)
                    q_next = res.q.reshape(q_orig_shape).cpu().numpy()
                    p_next = res.p.reshape(p_orig_shape).cpu().numpy()
                    return CNNPrediction(q=q_next, p=p_next)

            return model_next_step

        if integrator_type != "null":
            raise ValueError(f"cnn-step predictions do not work with integrator {integrator_type}")

    elif eval_type in {"knn-regressor", "knn-regressor-oneshot"}:
        # Use the time_derivative
        KNNDerivative = namedtuple("KNNDerivative", ["dq_dt", "dp_dt"])
        def model_time_deriv(q, p, dt=1.0, t=0):
            p_len = p.shape[-1]
            x = np.concatenate([p, q], axis=-1)
            t_start = time.perf_counter()
            ret = net.predict(x)
            t_end = time.perf_counter()
            eval_only_time_collector.accumulate_time(t_end - t_start)
            dpdt, dqdt = np.split(ret, [p_len], axis=-1)
            return KNNDerivative(dq_dt=dqdt, dp_dt=dpdt)
        time_deriv_func = model_time_deriv
    elif eval_type in {"knn-predictor", "knn-predictor-oneshot"}:
        KNNPrediction = namedtuple("KNNPrediction", ["q", "p"])
        def model_next_step(q, p, dt=1.0, t=0):
            p_len = p.shape[-1]
            x = np.concatenate([p, q], axis=-1)
            t_start = time.perf_counter()
            ret = net.predict(x)
            t_end = time.perf_counter()
            eval_only_time_collector.accumulate_time(t_end - t_start)
            next_p, next_q = np.split(ret, [p_len], axis=-1)
            return KNNPrediction(q=next_q, p=next_p)
        time_deriv_func = model_next_step
        if integrator_type != "null":
            raise ValueError(f"KNN predictions do not work with integrator {integrator_type}")
    elif eval_type == "integrator-baseline":
        # Use the time_derivative
        SystemDerivative = namedtuple("SystemDerivative", ["dq_dt", "dp_dt"])
        def system_derivative(q, p, dt=1.0, t=0):
            if torch.is_tensor(dt):
                dt = dt.item()
            t_start = time.perf_counter()
            dq_dt, dp_dt = system.derivative(p=p, q=q)
            t_end = time.perf_counter()
            eval_only_time_collector.accumulate_time(t_end - t_start)
            return SystemDerivative(dp_dt=dp_dt, dq_dt=dq_dt)
        time_deriv_func = system_derivative
    else:
        logger.error(f"Invalid evaluation type: {eval_type}")
        raise ValueError(f"Invalid evaluation type: {eval_type}")

    trajectory_results = []

    for traj_num, trajectory in enumerate(eval_loader):
        logger.info(f"Starting trajectory number {traj_num}")
        eval_only_time_collector.start_traj(traj_num)
        traj_name = trajectory.name[0]
        p = trajectory.p.to(device, dtype=eval_dtype)
        q = trajectory.q.to(device, dtype=eval_dtype)
        p_noiseless = trajectory.p_noiseless.to(device, dtype=eval_dtype)

        q_noiseless = trajectory.q_noiseless.to(device, dtype=eval_dtype)
        masses = trajectory.masses.to(device, dtype=eval_dtype)
        num_time_steps = int(trajectory.trajectory_meta["num_time_steps"][0].detach().cpu().item())
        time_step_size = float(trajectory.trajectory_meta["time_step_size"][0].detach().cpu().item())
        edges = None
        vertices = None
        boundary_cond_func = None
        static_nodes = None

        # Compute hamiltonians
        # Construct systems
        if eval_dataset.system == "spring":
            system = spring.system_from_records()
        elif eval_dataset.system == "wave":
            n_grid = eval_dataset.system_metadata["n_grid"]
            space_max = eval_dataset.system_metadata["space_max"]
            wave_speed = trajectory.trajectory_meta["wave_speed"][0].item()
            system = wave.system_from_records(n_grid=n_grid,
                                              space_max=space_max,
                                              wave_speed=wave_speed)
        elif eval_dataset.system == "spring-mesh":
            n_dim = eval_dataset.system_metadata["n_dim"]
            particles = eval_dataset.system_metadata["particles"]
            n_particles = len(eval_dataset.system_metadata["particles"])
            edges_dict = eval_dataset.system_metadata["edges"]
            edges = np.array([(e["a"], e["b"]) for e in edges_dict] +
                             [(e["b"], e["a"]) for e in edges_dict], dtype=np.int64).T
            vel_decay = eval_dataset.system_metadata["vel_decay"]
            system = spring_mesh.system_from_records(n_dim, particles, edges_dict,
                                                     vel_decay=vel_decay)
            boundary_cond_func = spring_mesh.make_enforce_boundary_function(trajectory=trajectory)
        elif eval_dataset.system == "navier-stokes":
            grid_resolution = eval_dataset.system_metadata["grid_resolution"]
            in_velocity = trajectory.trajectory_meta["in_velocity"][0].item()
            viscosity = trajectory.trajectory_meta["viscosity"][0].item()
            vertices = eval_dataset[0].vertices.tolist()
            vertices_array = eval_dataset[0].vertices
            vertices_array.setflags(write=False)
            edges = eval_dataset[0].edge_index.tolist()
            system = navier_stokes.system_from_records(grid_resolution=grid_resolution, viscosity=viscosity)

            fixed_mask_solutions = trajectory.fixed_mask_p[0].cpu().numpy()
            fixed_mask_solutions.setflags(write=False)
            fixed_mask_pressures = trajectory.fixed_mask_q[0].cpu().numpy()
            fixed_mask_pressures.setflags(write=False)
            boundary_cond_func = navier_stokes.make_enforce_boundary_function(
                in_velocity=in_velocity,
                vertex_coords=vertices_array,
                fixed_mask_solutions=fixed_mask_solutions,
                fixed_mask_pressures=fixed_mask_pressures,
            )
            static_nodes = trajectory.static_nodes.numpy().squeeze()
        else:
            raise ValueError(f"Unknown system type {eval_dataset.system}")

        if eval_type in {"cnn-step", "cnn-deriv", "unet-step", "unet-deriv"}:
            time_deriv_func = cnn_time_deriv_func(batch=trajectory)
        elif eval_type in {"mlp", "mlp-step", "mlp-deriv", "nn-kernel-step", "nn-kernel-deriv"}:
            time_deriv_func = mlp_time_deriv_func(batch=trajectory)

        p0 = p[:, 0].detach().cpu().numpy()
        q0 = q[:, 0].detach().cpu().numpy()
        t0 = trajectory.t[0, 0].item()
        q0, p0 = q0, p0
        integrate_start = time.perf_counter()
        if not getattr(eval_dataset, "_linearize", True):
            q0 = q0.reshape((1, -1))
            p0 = p0.reshape((1, -1))

        if eval_type == "integrator-baseline" and baseline_stride != 1:
            num_time_steps = math.ceil(num_time_steps / baseline_stride)
            time_step_size = time_step_size * baseline_stride
            logger.info(f"Trajectory coarsened to {time_step_size} steps")

        int_res_raw = integrators.numerically_integrate(
            integrator=integrator_type,
            q0=q0,
            p0=p0,
            t0=t0,
            num_steps=num_time_steps,
            dt=time_step_size,
            deriv_func=time_deriv_func,
            boundary_cond_func=boundary_cond_func,
            system=system)

        # Remove extraneous batch dimension
        integrate_elapsed = time.perf_counter() - integrate_start
        # Split the integration result
        int_p = int_res_raw.p
        int_q = int_res_raw.q

        if not getattr(eval_dataset, "_linearize", True):
            n_noiseless_steps = p_noiseless.shape[1]
            p_noiseless = p_noiseless.reshape((1, n_noiseless_steps, -1))
            q_noiseless = q_noiseless.reshape((1, n_noiseless_steps, -1))

        # Compute errors and other statistics
        int_res = np.concatenate([int_p, int_q], axis=-1)
        true = torch.cat([p_noiseless, q_noiseless], axis=-1)[0].detach().cpu().numpy()

        if eval_type == "integrator-baseline" and baseline_stride != 1:
            true = true[::baseline_stride]

        raw_l2 = raw_err(approx=int_res, true=true, norm=2)
        rel_l2 = rel_err(approx=int_res, true=true, norm=2)
        mse_err = mean_square_err(approx=int_res, true=true)

        mean_raw_l2 = np.mean(raw_l2)
        mean_mse = np.mean(mse_err)
        logger.info(f"Mean raw l2: {mean_raw_l2}")
        logger.info(f"Mean MSE: {mean_mse}")

        int_p = int_res_raw.p
        int_q = int_res_raw.q

        # All combinations true/net hamiltonian on true/net trajectories

        int_numpy_arrays = {
            traj_name: int_res,
            f"{traj_name}_relerr_l2": rel_l2,
            f"{traj_name}_raw_l2": raw_l2,
            f"{traj_name}_mse": mse_err,
            f"{traj_name}_p": int_p,
            f"{traj_name}_q": int_q,
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
            },
            "timing": {
                "integrate_elapsed": integrate_elapsed,
                "eval_only": eval_only_time_collector.get_time(traj_num),
            }
        }

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
