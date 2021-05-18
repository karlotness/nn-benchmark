import pathlib
import methods
import torch
from torch import utils
import dataset
import logging
import json
import time
import joblib
import integrators
import numpy as np
from collections import namedtuple
import copy


TRAIN_DTYPES = {
    "float": (torch.float, np.float32),
    "double": (torch.double, np.float64)
}


class NoNoise:
    def __init__(self):
        pass

    def process_batch(self, batch):
        return batch


class RandomCorrectedNoise:
    NoiseBatch = namedtuple("NoiseBatch", ["name", "p", "q", "dp_dt", "dq_dt",
                                           "t", "trajectory_meta",
                                           "p_noiseless", "q_noiseless",
                                           "masses", "edge_index"])

    def __init__(self, variance, gamma=0.1):
        self.variance = variance
        self.gamma = gamma

    def process_batch(self, batch):
        noise_sigma = np.sqrt(self.variance)
        if hasattr(batch, "dp_dt"):
            noise_p = noise_sigma * torch.randn(*batch.p.shape,
                                                dtype=batch.p.dtype,
                                                device=batch.p.device)
            noise_q = noise_sigma * torch.randn(*batch.q.shape,
                                                dtype=batch.q.dtype,
                                                device=batch.q.device)
            return self.NoiseBatch(
                name=batch.name,
                p=batch.p + noise_p,
                q=batch.q + noise_q,
                dp_dt=batch.dp_dt - noise_p,
                dq_dt=batch.dq_dt - noise_q,
                t=batch.t,
                trajectory_meta=batch.trajectory_meta,
                p_noiseless=batch.p_noiseless,
                q_noiseless=batch.q_noiseless,
                masses=batch.masses,
                edge_index=batch.edge_index)
        else:
            noise_pos = noise_sigma * torch.randn(*batch.pos.shape,
                                                  dtype=batch.pos.dtype,
                                                  device=batch.pos.device)
            batch.pos += noise_pos
            batch.x += noise_pos
            batch.y -= (self.gamma * (2 * noise_pos)) + (
                (1 - self.gamma) * noise_pos)
            return batch


def create_live_noise(noise_args, base_logger):
    logger = base_logger.getChild("live-noise")
    noise_type = noise_args.get("type", "none")
    if noise_type == "none":
        logger.info("No noise added during training")
        return NoNoise()
    elif noise_type == "gn-corrected":
        variance = noise_args["variance"]
        logger.info(f"Adding GN-style corrected noise variance={variance}")
        return RandomCorrectedNoise(variance=variance)


def save_network(net, network_args, train_type, out_dir, base_logger,
                 model_file_name="model.pt"):
    logger = base_logger.getChild("save_network")
    logger.info("Saving network")

    if train_type in {"knn-regressor", "knn-predictor"}:
        joblib.dump(net, out_dir / model_file_name)
    else:
        torch.save(net.state_dict(), out_dir / model_file_name)

    with open(out_dir / "model.json", "w", encoding="utf8") as model_file:
        json.dump(network_args, model_file)
    logger.info("Saved network")


def create_optimizer(net, optimizer, optim_args, base_logger=None):
    if base_logger:
        logger = base_logger.getChild("optim")
    else:
        logger = logging.getLogger("optim")
    lr = optim_args["learning_rate"]
    weight_decay = optim_args.get("weight_decay", 0)
    logger.info(f"Using optimizer {optimizer} lr={lr} decay={weight_decay}")
    if optimizer == "adam":
        return torch.optim.Adam(net.parameters(),
                                lr=lr,
                                weight_decay=weight_decay)
    elif optimizer == "sgd":
        return torch.optim.SGD(net.parameters(),
                               lr=lr,
                               weight_decay=weight_decay)
    else:
        raise ValueError(f"Invalid optimizer {optimizer}")


class SchedulerWrapper:
    def __init__(self, scheduler, step_period, logger):
        self.scheduler = scheduler
        self.step_period = step_period
        self.logger = logger

    def step_batch(self, *args, **kwargs):
        if self.scheduler is not None and self.step_period == "batch":
            self.logger.debug("Stepping scheduler")
            self.scheduler.step(*args, **kwargs)

    def step_epoch(self, *args, **kwargs):
        if self.scheduler is not None and self.step_period == "epoch":
            self.logger.debug("Stepping scheduler")
            self.scheduler.step(*args, **kwargs)


def create_scheduler(optimizer, scheduler_type, scheduler_step, scheduler_args,
                     base_logger=None):
    if base_logger:
        logger = base_logger.getChild("sched")
    else:
        logger = logging.getLogger("sched")
    if scheduler_type == "none":
        logger.info("No scheduler")
        return SchedulerWrapper(scheduler=None, step_period=None,
                                logger=logger)
    if scheduler_type == "exponential":
        scheduler_cls = torch.optim.lr_scheduler.ExponentialLR
    else:
        raise ValueError(f"Invalid scheduler {scheduler_type}")
    logger.info(f"Using scheduler {scheduler_type} with step unit {scheduler_step}")
    scheduler = scheduler_cls(optimizer, **scheduler_args)
    return SchedulerWrapper(scheduler=scheduler, step_period=scheduler_step,
                            logger=logger)


def create_dataset(base_dir, data_args):
    def _create_dataset_inner(data_dir, base_dir, data_args):
        linearize = data_args.get("linearize", False)
        base_data_set = dataset.TrajectoryDataset(data_dir=data_dir,
                                                  linearize=linearize)
        dataset_type = data_args["dataset"]
        if dataset_type == "trajectory":
            data_set = base_data_set
        elif dataset_type == "snapshot":
            data_set = dataset.SnapshotDataset(traj_dataset=base_data_set)
        elif dataset_type == "step-snapshot":
            time_skew = int(data_args["dataset_args"].get("time-skew", 1))
            subsample = int(data_args["dataset_args"].get("subsample", 1))
            data_set = dataset.StepSnapshotDataset(traj_dataset=base_data_set, subsample=subsample, time_skew=time_skew)
        elif dataset_type == "navier-stokes":
            data_set = dataset.NavierStokesSnapshotDataset(traj_dataset=base_data_set)
        elif dataset_type == "rollout-chunk":
            rollout_length = int(data_args["dataset_args"]["rollout_length"])
            data_set = dataset.RolloutChunkDataset(traj_dataset=base_data_set,
                                                   rollout_length=rollout_length)
        else:
            raise ValueError(f"Invalid dataset type {dataset_type}")

        loader_args = data_args["loader"]
        loader_type = loader_args.get("type", "pytorch")
        pin_memory = loader_args.get("pin_memory", False)
        num_workers = loader_args.get("num_workers", 0)
        if loader_type == "pytorch":
            loader = utils.data.DataLoader(data_set,
                                           batch_size=loader_args["batch_size"],
                                           shuffle=loader_args["shuffle"],
                                           pin_memory=pin_memory,
                                           num_workers=num_workers)
        elif loader_type == "pytorch-geometric":
            # Lazy import to avoid pytorch-geometric if possible
            import dataset_geometric
            from torch_geometric import data as geometric_data
            package_args = loader_args["package_args"]
            loader = geometric_data.DenseDataLoader(
                dataset=dataset_geometric.package_data(data_set,
                                                       package_args=package_args,
                                                       system=data_set.system),
                batch_size=loader_args["batch_size"],
                shuffle=loader_args["shuffle"])
        else:
            raise ValueError(f"Invalid loader type {loader_type}")
        return data_set, loader
    # End of inner dataset
    data_dir = base_dir / data_args["data_dir"]
    train_data_set, train_loader = _create_dataset_inner(data_dir, base_dir, data_args)
    if "val_data_dir" in data_args and data_args["val_data_dir"] is not None:
        val_data_dir = base_dir / data_args["val_data_dir"]
        val_data_set, val_loader = _create_dataset_inner(val_data_dir, base_dir, data_args)
    else:
        val_data_set, val_loader = None, None
    return train_data_set, train_loader, val_data_set, val_loader


def add_auxiliary_data(trajectory_dataset, network_args):
    if trajectory_dataset.system == "navier-stokes":
        network_args["arch_args"]["mesh_coords"] = trajectory_dataset[0].vertices.tolist()
        network_args["arch_args"]["static_nodes"] = trajectory_dataset._traj_dataset._npz_file["fixed_mask"].tolist()
    else:
        pass


def select_device(try_gpu, base_logger):
    logger = base_logger.getChild("select_device")
    device = torch.device("cpu")
    if try_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    elif try_gpu:
        logger.warning("Using CPU despite trying for GPU")
    logger.info(f"Using device {device}")
    return device


class TorchTypeConverter:
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype

    def __call__(self, x):
        return x.to(self.device, dtype=self.dtype)


def shape_product(shape):
    prod = 1
    for i in shape:
        prod *= i
    return prod


TrainLossResult = namedtuple("TrainLossResult",
                             ["loss", "total_loss_denom_incr"])


def train_hnn(net, batch, loss_fn, train_type_args, tensor_converter):
    # Extract values from batch
    p = tensor_converter(batch.p)
    q = tensor_converter(batch.q)
    dp_dt = tensor_converter(batch.dp_dt)
    dq_dt = tensor_converter(batch.dq_dt)
    # Perform training
    # Assume snapshot dataset (shape [batch_size, n_grid])
    deriv_pred = net.time_derivative(p=p, q=q)
    dx_dt = torch.cat([dp_dt, dq_dt], dim=-1)
    dx_dt_pred = torch.cat([deriv_pred.dp_dt, deriv_pred.dq_dt], dim=-1)
    loss = loss_fn(dx_dt_pred, dx_dt)
    return TrainLossResult(loss=loss,
                           total_loss_denom_incr=shape_product(p.shape))


def train_srnn(net, batch, loss_fn, train_type_args, tensor_converter):
    # Extract values from batch
    p = tensor_converter(batch.p)
    q = tensor_converter(batch.q)
    p_noiseless = tensor_converter(batch.p_noiseless)
    q_noiseless = tensor_converter(batch.q_noiseless)
    trajectory_meta = batch.trajectory_meta
    # Perform training
    # Assume rollout dataset (shape [batch_size, dataset rollout_length, n_grid])
    training_steps = train_type_args["rollout_length"]
    time_step_size = float(trajectory_meta["time_step_size"][0])
    # Check that all time step sizes are equal
    if not torch.all(trajectory_meta["time_step_size"] == time_step_size):
        raise ValueError("Inconsistent time step sizes in batch")
    int_res = integrators.numerically_integrate(
        train_type_args["integrator"],
        p_0=p[:, 0],
        q_0=q[:, 0],
        model=net,
        method=integrators.IntegrationScheme.HAMILTONIAN,
        T=training_steps,
        dt=time_step_size,
        volatile=False,
        device=tensor_converter.device,
        coarsening_factor=1)
    x = torch.cat([p_noiseless, q_noiseless], dim=-1)
    x_pred = torch.cat([int_res.p, int_res.q], dim=-1)
    loss = loss_fn(x_pred, x)
    return TrainLossResult(loss=loss,
                           total_loss_denom_incr=shape_product(p.shape))


def train_mlp(net, batch, loss_fn, train_type_args, tensor_converter, predict_type="deriv"):
    # Extract values from batch
    p = tensor_converter(batch.p)
    q = tensor_converter(batch.q)
    pred = net(p=p, q=q)

    # Perform training
    # Assume snapshot dataset (shape [batch_size, n_grid])
    if predict_type == "deriv":
        dp_dt = tensor_converter(batch.dp_dt)
        dq_dt = tensor_converter(batch.dq_dt)
        true = torch.cat([dq_dt, dp_dt], dim=-1)
        pred = torch.cat([pred.dq_dt, pred.dp_dt], dim=-1)
    elif predict_type == "step":
        p_step = tensor_converter(batch.p_step)
        q_step = tensor_converter(batch.q_step)
        true = torch.cat([q_step, p_step], dim=-1)
        pred = torch.cat([pred.q, pred.p], dim=-1)
    else:
        raise ValueError(f"Invalid predict type {predict_type}")

    loss = loss_fn(pred, true)
    return TrainLossResult(loss=loss,
                           total_loss_denom_incr=shape_product(p.shape))


def train_cnn(net, batch, loss_fn, train_type_args, tensor_converter, predict_type="deriv", extra_data=None):
    # Extract values from batch
    p = tensor_converter(batch.p)
    q = tensor_converter(batch.q)
    if extra_data:
        if not torch.is_tensor(extra_data):
            extra_data = torch.from_numpy(extra_data)
        # Ensure we have a batch dimension
        repeat_sizes = tuple([p.shape[0]] + [1 for _ in range(extra_data.ndim)])
        extra_data = tensor_converter(extra_data).unsqueeze(0).repeat(*repeat_sizes)

    # Perform training
    # Assume snapshot dataset (shape [batch_size, n_grid])
    pred = net(p=p, q=q, extra_data=extra_data)

    if predict_type == "deriv":
        dp_dt = tensor_converter(batch.dp_dt)
        dq_dt = tensor_converter(batch.dq_dt)

        true = torch.cat([dq_dt, dp_dt], dim=-1)
        pred = torch.cat([pred.dq_dt, pred.dp_dt], dim=-1)
    elif predict_type == "step":
        p_step = tensor_converter(batch.p_step)
        q_step = tensor_converter(batch.q_step)

        true = torch.cat([q_step, p_step], dim=-1)
        pred = torch.cat([pred.q, pred.p], dim=-1)
    else:
        raise ValueError(f"Invalid predict type {predict_type}")

    loss = loss_fn(pred, true)
    return TrainLossResult(loss=loss,
                           total_loss_denom_incr=shape_product(p.shape))


def train_hogn(net, batch, loss_fn, train_type_args, tensor_converter):
    # Extract values from batch
    graph_batch = batch
    graph_batch.x = tensor_converter(graph_batch.x)
    graph_batch.y = tensor_converter(graph_batch.y)
    graph_batch.edge_index = graph_batch.edge_index.to(tensor_converter.device)
    # Perform training
    # HOGN training with graph_batch
    loss = net.loss(graph_batch)
    return TrainLossResult(loss=loss,
                           total_loss_denom_incr=graph_batch.num_graphs)


def train_gn(net, batch, loss_fn, train_type_args, tensor_converter):
    # Extract values from batch
    graph_batch = batch
    graph_batch.x = tensor_converter(graph_batch.x)
    graph_batch.pos = tensor_converter(graph_batch.pos)
    graph_batch.y = tensor_converter(graph_batch.y)
    graph_batch.edge_index = graph_batch.edge_index.to(tensor_converter.device)

    accel_pred = net(graph_batch.pos, graph_batch.x, graph_batch.edge_index)
    accel = graph_batch.y

    loss = loss_fn(accel_pred, accel)
    return TrainLossResult(loss=loss,
                           total_loss_denom_incr=shape_product(accel.shape))


TRAIN_FUNCTIONS = {
    "hnn": train_hnn,
    "srnn": train_srnn,
    "mlp-deriv": lambda *args, **kwargs: train_mlp(*args, **kwargs, predict_type="deriv"),
    "mlp-step": lambda *args, **kwargs: train_mlp(*args, **kwargs, predict_type="step"),
    "nn-kernel-deriv": lambda *args, **kwargs: train_mlp(*args, **kwargs, predict_type="deriv"),
    "nn-kernel-step": lambda *args, **kwargs: train_mlp(*args, **kwargs, predict_type="step"),
    "cnn-deriv": lambda *args, **kwargs: train_cnn(*args, **kwargs, predict_type="deriv"),
    "cnn-step": lambda *args, **kwargs: train_cnn(*args, **kwargs, predict_type="step"),
    "hogn": train_hogn,
    "gn": train_gn,
}


def run_phase(base_dir, out_dir, phase_args):
    logger = logging.getLogger("train")
    base_dir = pathlib.Path(base_dir)
    out_dir = pathlib.Path(out_dir)
    training_args = phase_args["training"]

    # Load the data
    logger.info("Constructing dataset")
    train_dataset, train_loader, val_dataset, val_loader = create_dataset(base_dir, phase_args["train_data"])
    add_auxiliary_data(train_dataset, phase_args["network"])
    network_args = phase_args["network"]

    # Construct the network
    logger.info("Building network")
    net = methods.build_network(network_args)

    # Misc training parameters
    max_epochs = training_args["max_epochs"]
    device = select_device(try_gpu=training_args["try_gpu"], base_logger=logger)
    train_dtype, train_dtype_np = TRAIN_DTYPES[training_args.get("train_dtype", "float")]
    logger.info(f"Training in dtype {train_dtype}")
    train_type = training_args["train_type"]
    train_type_args = training_args["train_type_args"]

    # Set up noise injection
    noise_injector = create_live_noise(
        noise_args=training_args.get("noise", {}),
        base_logger=logger)

    # If training a knn, this is all we need.
    if train_type == "knn-regressor":
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

        total_epoch_count = 1
        epoch_stats = {}
    elif train_type == "knn-predictor":
        logger.info("Starting fitting of dataset for KNN Predictor.")

        data_x = []
        data_y = []
        for batch in train_loader:
            data_x.append(np.concatenate([batch.p, batch.q], axis=-1))
            data_y.append(np.concatenate([batch.p_noiseless, batch.q_noiseless], axis=-1))
        data_x = np.concatenate(data_x, axis=0)
        data_y = np.concatenate(data_y, axis=0)

        net.fit(data_x[:-1, ...], data_y[1:, ...])

        logger.info("Finished fitting of dataset for KNN Predictor.")

        total_epoch_count = 1
        epoch_stats = {}

    else:
        # Construct the optimizer
        logger.info("Creating optimizer")
        optimizer = training_args["optimizer"]
        optim_args = training_args["optimizer_args"]
        scheduler_type = training_args.get("scheduler", "none")
        scheduler_step = training_args.get("scheduler_step", "epoch")
        scheduler_args = training_args.get("scheduler_args", {})
        optim = create_optimizer(net, optimizer, optim_args, base_logger=logger)
        sched = create_scheduler(optim,
                                 scheduler_type=scheduler_type,
                                 scheduler_step=scheduler_step,
                                 scheduler_args=scheduler_args,
                                 base_logger=logger)
        torch_converter = TorchTypeConverter(device=device, dtype=train_dtype)

        extra_train_args = {}
        if train_type in {"cnn-step", "cnn-deriv", "cnn"} and (getattr(train_dataset, "fixed_mask", None) is not None):
            logger.info("Providing fixed mask as extra data")
            _extra_data = torch_converter(torch.from_numpy(train_dataset.fixed_mask))
            _extra_data.requires_grad = False
            extra_train_args = {"extra_data": _extra_data}

        # Move network to device and convert to dtype
        net = torch_converter(net)

        # Declare loss and training functions
        loss_fn = torch.nn.MSELoss()
        try:
            train_fn = TRAIN_FUNCTIONS[train_type]
        except KeyError as exc:
            raise ValueError(f"Invalid train type: {train_type}") from exc

        # Run training epochs
        logger.info("Starting training")
        epoch_stats = []
        min_val_loss = np.inf
        for epoch in range(max_epochs):
            logger.info(f"Epoch {epoch} of {max_epochs}")
            this_epoch_stats = {}
            this_epoch_timing = {}
            total_forward_time = 0
            total_backward_time = 0
            time_epoch_start = time.perf_counter()
            total_loss = 0
            total_loss_denom = 0
            time_train_start = time.perf_counter()
            # Do training
            net.train()
            for batch_num, batch in enumerate(train_loader):
                batch = noise_injector.process_batch(batch)

                optim.zero_grad()
                time_forward_start = time.perf_counter()
                train_result = train_fn(net=net, batch=batch, loss_fn=loss_fn,
                                        train_type_args=train_type_args,
                                        tensor_converter=torch_converter, **extra_train_args)
                total_forward_time += time.perf_counter() - time_forward_start
                loss = train_result.loss
                total_loss_denom += train_result.total_loss_denom_incr
                # Training step
                time_backward_start = time.perf_counter()
                loss.backward()
                optim.step()
                total_backward_time += time.perf_counter() - time_backward_start
                total_loss += loss.item() * train_result.total_loss_denom_incr
                # Step optimizer for batch
                sched.step_batch()
            total_train_time = time.perf_counter() - time_train_start
            avg_loss = total_loss / total_loss_denom

            this_epoch_stats.update({
                "num_batches": batch_num + 1,
                "avg_loss": avg_loss,
                "train_total_loss": total_loss,
                "train_loss_denom": total_loss_denom,
                "saved_min_val_net": False,
            })
            this_epoch_timing.update({
                "total_forward": total_forward_time,
                "total_backward": total_backward_time,
                "total_train": total_train_time,
            })

            optim.zero_grad()

            # Do validation, if we have a validation loader
            avg_val_loss = None
            if val_loader:
                val_total_loss = 0
                val_total_loss_denom = 0
                time_val_start = time.perf_counter()
                net.eval()
                for val_batch_num, val_batch in enumerate(val_loader):
                    val_result = train_fn(net=net, batch=val_batch,
                                          loss_fn=loss_fn,
                                          train_type_args=train_type_args,
                                          tensor_converter=torch_converter, **extra_train_args)
                    val_loss = val_result.loss
                    val_total_loss_denom += val_result.total_loss_denom_incr
                    val_total_loss += val_loss.item() * val_result.total_loss_denom_incr
                total_val_time = time.perf_counter() - time_val_start
                avg_val_loss = val_total_loss / val_total_loss_denom
                # Store validation results
                this_epoch_stats.update({
                    "val_batches": val_batch_num + 1,
                    "val_total_loss": val_total_loss,
                    "val_loss_denom": val_total_loss_denom,
                })
                this_epoch_timing.update({
                    "total_val": total_val_time,
                })
                # Update minimum validation loss
                if val_total_loss < min_val_loss:
                    # New minimum validation loss, save network
                    min_val_loss = val_total_loss
                    logger.info("New minimum validation loss. Saving network")
                    save_network(net=net, network_args=network_args, train_type=train_type,
                                 out_dir=out_dir, base_logger=logger,
                                 model_file_name="model_min_val.pt")
                    this_epoch_stats["saved_min_val_net"] = True

            # Compute total epoch time
            total_epoch_time = time.perf_counter() - time_epoch_start

            # Report epoch statistics
            logger.info(f"Epoch complete. Avg loss: {avg_loss}, time: {total_epoch_time}, val loss: {avg_val_loss}")
            # Compute per-epoch statistics and store
            this_epoch_timing.update({
                "total_epoch": total_epoch_time,
            })
            this_epoch_stats.update({
                "timing": this_epoch_timing,
            })
            epoch_stats.append(this_epoch_stats)
            # Step optimizer for epoch
            sched.step_epoch()

        logger.info("Training done")
        total_epoch_count = epoch + 1

    # Save the network
    save_network(net=net, network_args=network_args, train_type=train_type,
                 out_dir=out_dir, base_logger=logger)

    # Save the run statistics
    with open(out_dir / "train_stats.json", "w", encoding="utf8") as stats_file:
        stats = {
            "num_epochs": total_epoch_count,
            "epoch_stats": epoch_stats,
        }
        json.dump(stats, stats_file)
