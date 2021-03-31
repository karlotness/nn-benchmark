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
from systems import spring, wave, particle, spring_mesh, taylor_green
import joblib
from collections import namedtuple


DecoratedIntegrationResult = namedtuple("DecoratedIntegrationResult", ["q", "p"])

class NullEvalDecorator:
    def __init__(self, integrator):
        pass

    def decorate_initial_cond(self, q0, p0):
        return q0, p0

    def decorate_deriv_func(self, func):
        return func

    def process_results(self, int_res):
        return int_res


class TaylorGreenEvalDecorator:
    def __init__(self, integrator):
        self.pressure_steps = []
        if integrator == "rk4":
            self.stride = 4
        elif integrator == "leapfrog":
            self.stride = 2
        else:
            self.stride = 1

    def decorate_initial_cond(self, q0, p0):
        press = q0
        vels = p0
        self.pressure_steps.append(press)
        nq0, np0 = np.split(vels, 2, axis=-1)
        return nq0, np0

    def decorate_deriv_func(self, func):
        def tg_wrapped(q, p, dt=1.0, t=0):
            press = self.pressure_steps[-1]
            vel = np.concatenate([q, p], axis=-1)
            dq, dp = func(press, vel, dt=dt, t=t)
            dq = dq.reshape((1, -1))
            dp = dp.reshape((1, -1))
            x = np.concatenate([dq, dp], axis=-1)
            n = x.shape[-1]//3
            new_press = x[..., :n]
            nq0, np0 = np.split(x[..., n:], 2, axis=-1)
            self.pressure_steps.append(new_press)
            return nq0, np0

        return tg_wrapped

    def process_results(self, int_res):
        new_p = np.concatenate([int_res.q, int_res.p], axis=-1)
        press_stride = self.stride
        press_steps = self.pressure_steps[::press_stride][:-1]
        new_q = np.vstack(press_steps)
        return DecoratedIntegrationResult(q=new_q, p=new_p)


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
    if metadata["arch"] in {"knn-regressor", "knn-predictor"}:
        with open(weight_path, "rb") as model_file:
            net = joblib.load(model_file)
    else:
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
        logger.warning("Overriding loader args for KNN predictor")
        eval_args["train_data"]["dataset"] = "trajectory"
        eval_args["train_data"]["loader"]["type"] = "pytorch"
        eval_args["train_data"]["loader"]["batch_size"] = 1

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
            data_x.append(np.concatenate([batch.p[0], batch.q[0]], axis=-1)[:-1, ...])
            data_y.append(np.concatenate([batch.p_noiseless[0], batch.q_noiseless[0]], axis=-1)[1:, ...])
        data_x = np.concatenate(data_x, axis=0)
        data_y = np.concatenate(data_y, axis=0)
        net.fit(data_x, data_y)
        logger.info("Finished fitting of dataset for KNN Predictor.")
    # Return the KNN
    return net


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

    make_eval_decorator = NullEvalDecorator
    if eval_dataset.system == "taylor-green" and eval_type != "gn":
        make_eval_decorator = TaylorGreenEvalDecorator

    # Integrate each trajectory, compute stats and store
    logger.info("Starting evaluation")
    integrator_type = eval_args["integrator"]

    if eval_type == "srnn":
        time_deriv_func = net
        hamiltonian_func = net
    elif eval_type == "hnn":
        def model_hamiltonian(q, p, dt=1.0):
            hamilt = net(p=p, q=q)
            return hamilt[0] + hamilt[1]

        def hnn_model_time_deriv(q, p, dt=1.0, t=0):
            return net.time_derivative(p=p, q=q, create_graph=False)
        time_deriv_func = hnn_model_time_deriv
        hamiltonian_func = model_hamiltonian
    elif eval_type in {"mlp-deriv", "nn-kernel"}:
        MLPDerivative = namedtuple("MLPDerivative", ["dq_dt", "dp_dt"])
        def net_no_grad(q, p, dt=1.0, t=0):
            with torch.no_grad():
                q = torch.from_numpy(q).to(device, dtype=eval_dtype)
                p = torch.from_numpy(p).to(device, dtype=eval_dtype)
                ret = net(q=q, p=p)
                dq = ret.dq_dt.detach().cpu().numpy()
                dp = ret.dp_dt.detach().cpu().numpy()
                return MLPDerivative(dq_dt=dq, dp_dt=dp)
        time_deriv_func = net_no_grad
    elif eval_type in {"mlp-step"}:
        MLPPrediction = namedtuple("MLPPrediction", ["q", "p"])
        def model_next_step(q, p, dt=1.0, t=0):
            with torch.no_grad():
                q = torch.from_numpy(q).to(device, dtype=eval_dtype)
                p = torch.from_numpy(p).to(device, dtype=eval_dtype)
                ret = net(q=q, p=p)
                next_q = ret.q.detach().cpu().numpy()
                next_p = ret.p.detach().cpu().numpy()
                return MLPPrediction(q=next_q, p=next_p)
        time_deriv_func = model_next_step
        if integrator_type != "null":
            raise ValueError(f"mlp-step predictions do not work with integrator {integrator_type}")
    elif eval_type == "cnn-deriv":
        CNNDerivative = namedtuple("CNNDerivative", ["dq_dt", "dp_dt"])
        def cnn_time_deriv(q, p, dt=1.0, t=0):
            with torch.no_grad():
                unsqueezed = False
                if len(p.shape) < 3:
                    # Unsqueeze all tensors
                    p = p.unsqueeze(1)
                    q = q.unsqueeze(1)
                    unsqueezed = True
                res = net(p=p, q=q)
                if unsqueezed:
                    res = CNNDerivative(
                        dq_dt=res.dq_dt[:, 0],
                        dp_dt=res.dp_dt[:, 0],
                    )
                return res
        time_deriv_func = cnn_time_deriv
    elif eval_type == "cnn-step":
        CNNPrediction = namedtuple("CNNPrediction", ["q", "p"])
        def model_next_step(q, p, dt=1.0, t=0):
            with torch.no_grad():
                unsqueezed = False
                if len(p.shape) < 3:
                    # Unsqueeze all tensors
                    p = p.unsqueeze(1)
                    q = q.unsqueeze(1)
                    unsqueezed = True
                res = net(p=p, q=q)
                if unsqueezed:
                    res = CNNPrediction(
                        q=res.q[:, 0],
                        p=res.p[:, 0],
                    )
                return res
        time_deriv_func = model_next_step
        if integrator_type != "null":
            raise ValueError(f"cnn-step predictions do not work with integrator {integrator_type}")
    elif eval_type in {"knn-regressor", "knn-regressor-oneshot"}:
        # Use the time_derivative
        KNNDerivative = namedtuple("KNNDerivative", ["dq_dt", "dp_dt"])
        def model_time_deriv(q, p, dt=1.0, t=0):
            x = np.concatenate([p, q], axis=-1)
            ret = net.predict(x)
            dpdt, dqdt = np.split(ret, 2, axis=-1)
            return KNNDerivative(dq_dt=dqdt, dp_dt=dpdt)
        time_deriv_func = model_time_deriv
    elif eval_type in {"knn-predictor", "knn-predictor-oneshot"}:
        KNNPrediction = namedtuple("KNNPrediction", ["q", "p"])
        def model_next_step(q, p, dt=1.0, t=0):
            x = np.concatenate([p, q], axis=-1)
            ret = net.predict(x)
            next_p, next_q = np.split(ret, 2, axis=-1)
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
            dq_dt, dp_dt = system.derivative(p=p, q=q)
            return SystemDerivative(dp_dt=dp_dt, dq_dt=dq_dt)
        time_deriv_func = system_derivative
        if eval_dataset.system == "taylor-green":
            # Special case for TG derivatives
            def tg_system_derivative(q, p, dt=1.0, t=0):
                # q is pressure
                # p is velocity
                d_vel, d_press = system.derivative(p=p, q=q, t=t)
                new_press = system.pressure(t=t)
                return SystemDerivative(dq_dt=new_press, dp_dt=d_vel)
            time_deriv_func = tg_system_derivative
    elif eval_type == "hogn":
        # Lazy import to avoid pytorch-geometric if possible
        from methods import hogn
        import dataset_geometric

        package_args = eval_args["package_args"]
        HognMockDataset = namedtuple("HognMockDataset", ["p", "q", "dp_dt", "dq_dt", "masses"])

        def hogn_time_deriv_func(masses):
            def model_time_deriv(q, p, dt=1.0, t=0):
                mocked = HognMockDataset(p=p, q=q, masses=masses,
                                         dp_dt=None, dq_dt=None)
                bundled = dataset_geometric.package_data(dataset=[mocked],
                                                         package_args=package_args)[0]
                derivs = net.just_derivative(bundled)
                unbundled = hogn.unpackage_time_derivative(input_data=bundled,
                                                           deriv=derivs)
                return unbundled
            return model_time_deriv

        def hogn_hamiltonian_func(masses):
            def model_hamiltonian(q, p):
                hamilts = []
                for i in range(p.shape[0]):
                    # Break out batches separately
                    mocked = HognMockDataset(p=p[i], q=q[i], masses=masses[i],
                                             dp_dt=None, dq_dt=None)
                    bundled = dataset_geometric.package_data(dataset=[mocked],
                                                             package_args=package_args,
                                                             system=dataset.system)[0]
                    h = net(bundled).sum()
                    hamilts.append(h)
                return np.array(hamilts)
            return model_hamiltonian

    elif eval_type == "gn":
        # Lazy import to avoid pytorch-geometric if possible
        from methods import gn
        import dataset_geometric

        package_args = eval_args["package_args"]
        GnMockDataset = namedtuple("GnMockDataset", ["p", "q", "dp_dt", "dq_dt", "masses", "edge_index", "vertices"])

        GNPrediction = namedtuple("GNPrediction", ["q", "p"])
        def gn_time_deriv_func(masses, edges, n_particles, vertices):
            def model_next_step(q, p, dt=1.0, t=0):
                with torch.no_grad():
                    time_step_size = dt
                    p_orig_shape = p.shape
                    q_orig_shape = q.shape
                    batch_size = p.shape[0]
                    assert batch_size == 1
                    p = p.reshape((n_particles, -1))
                    q = q.reshape((n_particles, -1))
                    mocked = GnMockDataset(p=p, q=q,
                        masses=masses, dp_dt=None, dq_dt=None, edge_index=edges, vertices=vertices)
                    bundled = dataset_geometric.package_data([mocked],
                        package_args=package_args, system=eval_dataset.system)[0]

                    accel = net(torch.unsqueeze(bundled.pos, 0),
                        torch.unsqueeze(bundled.x, 0),
                        torch.unsqueeze(bundled.edge_index, 0))
                    accel = gn.unpack_results(accel, eval_dataset.system)

                    # Prediction for Taylor Green
                    if eval_dataset.system == "taylor-green":
                        accel = accel.squeeze().detach().cpu().numpy()

                        p_next = p + time_step_size * accel[..., :2]
                        q_next = accel[..., -1]
                    else:
                        accel = accel.reshape(p.shape).detach().cpu().numpy()

                        p_next = p + time_step_size * accel
                        q_next = q + time_step_size * p_next

                    return GNPrediction(p=p_next.reshape(p_orig_shape),
                                        q=q_next.reshape(q_orig_shape))
            return model_next_step

        if integrator_type != "null":
          raise ValueError(f"GN predictions do not work with integrator {integrator_type}")
    else:
        logger.error(f"Invalid evaluation type: {eval_type}")
        raise ValueError(f"Invalid evaluation type: {eval_type}")

    trajectory_results = []

    for traj_num, trajectory in enumerate(eval_loader):
        logger.info(f"Starting trajectory number {traj_num}")
        traj_name = trajectory.name[0]
        p = trajectory.p.to(device, dtype=eval_dtype)
        q = trajectory.q.to(device, dtype=eval_dtype)
        p_noiseless = trajectory.p_noiseless.to(device, dtype=eval_dtype)

        q_noiseless = trajectory.q_noiseless.to(device, dtype=eval_dtype)
        masses = trajectory.masses.to(device, dtype=eval_dtype)
        num_time_steps = trajectory.trajectory_meta["num_time_steps"][0]
        time_step_size = trajectory.trajectory_meta["time_step_size"][0]
        edges = None
        vertices = None

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
        elif eval_dataset.system == "particle":
            n_dim = eval_dataset.system_metadata["n_dim"]
            n_particles = eval_dataset.system_metadata["n_particles"]
            g = eval_dataset.system_metadata["g"]
            system = particle.ParticleSystem(n_particles=n_particles,
                                             n_dim=n_dim, g=g)
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
        elif eval_dataset.system == "taylor-green":
            n_grid = eval_dataset.system_metadata["n_grid"]
            space_scale = trajectory.trajectory_meta["space_scale"][0].item()
            viscosity = trajectory.trajectory_meta["viscosity"][0].item()
            density = trajectory.trajectory_meta["density"][0].item()
            vertices = eval_dataset.system_metadata["vertices"]
            edges_dict = eval_dataset.system_metadata["edges"]
            edges = np.array([(e["a"], e["b"]) for e in edges_dict] +
                             [(e["b"], e["a"]) for e in edges_dict], dtype=np.int64).T
            system = taylor_green.system_from_records(n_grid=n_grid, space_scale=space_scale, viscosity=viscosity, density=density, vertices=vertices, edges=edges_dict)
        else:
            raise ValueError(f"Unknown system type {eval_dataset.system}")

        if eval_type == "hogn":
            # Pull out masses for HOGN
            time_deriv_func = hogn_time_deriv_func(masses=masses)
            hamiltonian_func = hogn_hamiltonian_func(masses=masses)
        elif eval_type == "gn":
            n_particles = net.static_nodes.shape[0]
            if eval_dataset.system == "spring":
                n_particles = 1
            time_deriv_func = gn_time_deriv_func(masses=masses, edges=edges, n_particles=n_particles, vertices=vertices)
            num_traj = p.shape[0]
            traj_steps = p.shape[1]
            p = p.reshape((num_traj, traj_steps, -1))
            q = q.reshape((num_traj, traj_steps, -1))
            p_noiseless = p_noiseless.reshape((num_traj, traj_steps, -1))
            q_noiseless = q_noiseless.reshape((num_traj, traj_steps, -1))

        p0 = p[:, 0].detach().cpu().numpy()
        q0 = q[:, 0].detach().cpu().numpy()
        eval_decorator = make_eval_decorator(integrator=integrator_type)
        q0, p0 = eval_decorator.decorate_initial_cond(q0, p0)
        wrapped_time_deriv_func = eval_decorator.decorate_deriv_func(time_deriv_func)
        integrate_start = time.perf_counter()
        int_res_raw = integrators.numerically_integrate(
            integrator=integrator_type,
            q0=q0,
            p0=p0,
            num_steps=int(num_time_steps.detach().cpu().item()),
            dt=float(time_step_size.detach().cpu().item()),
            deriv_func=wrapped_time_deriv_func,
            system=system)
        int_res_raw = eval_decorator.process_results(int_res_raw)

        # Remove extraneous batch dimension
        integrate_elapsed = time.perf_counter() - integrate_start
        # Split the integration result
        int_p = int_res_raw.p
        int_q = int_res_raw.q

        # Compute errors and other statistics
        int_res = np.concatenate([int_p, int_q], axis=-1)
        true = torch.cat([p_noiseless, q_noiseless], axis=-1)[0].detach().cpu().numpy()
        raw_l2 = raw_err(approx=int_res, true=true, norm=2)
        rel_l2 = rel_err(approx=int_res, true=true, norm=2)
        mse_err = mean_square_err(approx=int_res, true=true)

        mean_raw_l2 = np.mean(raw_l2)
        mean_mse = np.mean(mse_err)
        logger.info(f"Mean raw l2: {mean_raw_l2}")
        logger.info(f"Mean MSE: {mean_mse}")

        int_p = int_res_raw.p
        int_q = int_res_raw.q

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
