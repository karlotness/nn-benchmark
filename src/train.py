import pathlib
import methods
import torch
from torch import utils
import dataset
import logging
import json
import time
import integrators
import numpy as np
from collections import namedtuple
import copy
import functools


TRAIN_DTYPES = {
    "float": (torch.float, np.float32),
    "double": (torch.double, np.float64)
}


class NoiseInjector(torch.utils.data.Dataset):
    def __init__(self, data_set):
        super().__init__()
        self.__dataset = data_set

    def inject_noise(self, batch):
        raise NotImplementedError("Subclass to implement")

    def __getattr__(self, name):
        return getattr(self.__dataset, name)

    def __len__(self):
        return len(self.__dataset)

    def __getitem__(self, idx):
        return self.inject_noise(self.__dataset[idx])


class StepSnapshotNoiseInjector(NoiseInjector):
    def __init__(self, data_set, variance):
        super().__init__(data_set=data_set)
        self.variance = variance
        self.noise_sigma = np.sqrt(self.variance)

    def __apply_noise(self, t, fixed_mask):
        noise = self.noise_sigma * np.random.randn(*t.shape)
        noise[fixed_mask] = 0
        return t + noise

    def inject_noise(self, batch):
        # We need to inject noise into the input q, p (and not on the boundary)
        # Assume these are namedtuples
        q = self.__apply_noise(batch.q, batch.fixed_mask_q)
        p = self.__apply_noise(batch.p, batch.fixed_mask_p)
        return batch._replace(q=q, p=p)


class SnapshotCorrectedNoiseInjector(NoiseInjector):
    def __init__(self, data_set, variance):
        super().__init__(data_set=data_set)
        self.variance = variance
        self.noise_sigma = np.sqrt(self.variance)

    def __generate_noise_mask(self, t, fixed_mask):
        noise = self.noise_sigma * np.random.randn(*t.shape)
        noise[fixed_mask] = 0
        return noise

    def inject_noise(self, batch):
        # We need to apply noise to q, p. And subtract it from dq, dp scaled by gamma
        q_noise = self.__generate_noise_mask(batch.q, batch.fixed_mask_q)
        p_noise = self.__generate_noise_mask(batch.p, batch.fixed_mask_p)
        new_q = batch.q + q_noise
        new_p = batch.p + p_noise
        new_dq = batch.dq_dt - q_noise
        new_dp = batch.dp_dt - p_noise
        return batch._replace(
            q=new_q,
            p=new_p,
            dq_dt=new_dq,
            dp_dt=new_dp,
        )


def create_live_noise(noise_args, base_logger):
    logger = base_logger.getChild("live-noise")
    noise_type = noise_args.get("type", "none")
    if noise_type == "none":
        logger.info("No noise added during training")
        return None
    elif noise_type == "deriv-corrected":
        variance = noise_args["variance"]
        logger.info(f"Adding GN-style derivative corrected noise with variance={variance}")
        return functools.partial(SnapshotCorrectedNoiseInjector, variance=variance)
    elif noise_type == "step-corrected":
        variance = noise_args["variance"]
        logger.info(f"Adding noise to input steps with variance={variance}")
        return functools.partial(StepSnapshotNoiseInjector, variance=variance)
    else:
        raise ValueError(f"Unknown live training noise type {noise_type}")


def save_network(net, network_args, train_type, out_dir, base_logger,
                 model_file_name="model.pt"):
    logger = base_logger.getChild("save_network")
    logger.info("Saving network")

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


def create_loss_fn(loss_type, device):
    if loss_type == "mse":
        loss_fn = torch.nn.MSELoss()
    elif loss_type == "l1":
        loss_fn = torch.nn.L1Loss()
    else:
        raise ValueError(f"Unknown loss type {loss_type}")
    return loss_fn.to(device)


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


def create_dataset(base_dir, data_args, train_noise_wrapper=None):
    def _create_dataset_inner(data_dir, base_dir, data_args, noise_wrapper):
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
        else:
            raise ValueError(f"Invalid dataset type {dataset_type}")

        if noise_wrapper is not None:
            data_set = noise_wrapper(data_set)

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
        else:
            raise ValueError(f"Invalid loader type {loader_type}")
        return data_set, loader
    # End of inner dataset
    data_dir = base_dir / data_args["data_dir"]
    train_data_set, train_loader = _create_dataset_inner(data_dir, base_dir, data_args, noise_wrapper=train_noise_wrapper)
    if "val_data_dir" in data_args and data_args["val_data_dir"] is not None:
        val_data_dir = base_dir / data_args["val_data_dir"]
        val_data_set, val_loader = _create_dataset_inner(val_data_dir, base_dir, data_args, noise_wrapper=None)
    else:
        val_data_set, val_loader = None, None
    return train_data_set, train_loader, val_data_set, val_loader


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


def train_mlp(net, batch, loss_fn, train_type_args, tensor_converter, predict_type="deriv", extra_data=None):
    # Extract values from batch
    p = tensor_converter(batch.p)
    q = tensor_converter(batch.q)
    if extra_data is not None:
        extra_data = tensor_converter(extra_data)
    pred = net(p=p, q=q, extra_data=extra_data)

    # Perform training
    # Assume snapshot dataset (shape [batch_size, n_grid])
    if predict_type == "deriv":
        pred_q_shape = pred.dq_dt.shape
        pred_p_shape = pred.dp_dt.shape
        dp_dt = tensor_converter(batch.dp_dt)
        dq_dt = tensor_converter(batch.dq_dt)
        true = torch.cat([dq_dt, dp_dt], dim=-1)
        pred = torch.cat([pred.dq_dt, pred.dp_dt], dim=-1)
    elif predict_type == "step":
        pred_q_shape = pred.q.shape
        pred_p_shape = pred.p.shape
        p_step = tensor_converter(batch.p_step)
        q_step = tensor_converter(batch.q_step)
        true = torch.cat([q_step, p_step], dim=-1)
        pred = torch.cat([pred.q, pred.p], dim=-1)
    else:
        raise ValueError(f"Invalid predict type {predict_type}")

    # Handle the possible fixed mask
    if torch.is_tensor(batch.fixed_mask_p):
        fm_q = batch.fixed_mask_q.to(device=tensor_converter.device).reshape(pred_q_shape)
        fm_p = batch.fixed_mask_p.to(device=tensor_converter.device).reshape(pred_p_shape)
        mask = torch.logical_not(torch.cat([fm_q, fm_p], dim=-1).reshape(pred.shape))
        true = torch.masked_select(true, mask)
        pred = torch.masked_select(pred, mask)

    loss = loss_fn(pred, true)
    return TrainLossResult(loss=loss,
                           total_loss_denom_incr=shape_product(p.shape))


def _ensure_cnn_dims(t):
    if t.ndim < 3:
        return t.unsqueeze(-1)
    else:
        return t


def train_cnn(net, batch, loss_fn, train_type_args, tensor_converter, predict_type="deriv", extra_data=None):
    # Extract values from batch
    p = _ensure_cnn_dims(tensor_converter(batch.p))
    q = _ensure_cnn_dims(tensor_converter(batch.q))
    if extra_data is not None:
        # Ensure we have a batch dimension
        extra_data = _ensure_cnn_dims(tensor_converter(extra_data))

    # Perform training
    # Assume snapshot dataset (shape [batch_size, n_grid])
    pred = net(p=p, q=q, extra_data=extra_data)
    n_batch = p.shape[0]

    if predict_type == "deriv":
        pred_q_shape = pred.dq_dt.shape
        pred_p_shape = pred.dp_dt.shape
        dp_dt = _ensure_cnn_dims(tensor_converter(batch.dp_dt))
        dq_dt = _ensure_cnn_dims(tensor_converter(batch.dq_dt))

        true = torch.cat([dq_dt, dp_dt], dim=-1)
        pred = torch.cat([pred.dq_dt, pred.dp_dt], dim=-1)
    elif predict_type == "step":
        pred_q_shape = pred.q.shape
        pred_p_shape = pred.p.shape
        p_step = _ensure_cnn_dims(tensor_converter(batch.p_step))
        q_step = _ensure_cnn_dims(tensor_converter(batch.q_step))

        true = torch.cat([q_step, p_step], dim=-1)
        pred = torch.cat([pred.q, pred.p], dim=-1)
    else:
        raise ValueError(f"Invalid predict type {predict_type}")

    # Handle the possible fixed mask
    if torch.is_tensor(batch.fixed_mask_p):
        fm_q = batch.fixed_mask_q.to(device=tensor_converter.device).reshape(pred_q_shape)
        fm_p = batch.fixed_mask_p.to(device=tensor_converter.device).reshape(pred_p_shape)
        mask = torch.logical_not(torch.cat([fm_q, fm_p], dim=-1).reshape(pred.shape))
        true = torch.masked_select(true, mask)
        pred = torch.masked_select(pred, mask)

    loss = loss_fn(pred, true)
    return TrainLossResult(loss=loss,
                           total_loss_denom_incr=shape_product(p.shape))


TRAIN_FUNCTIONS = {
    "mlp-deriv": functools.partial(train_mlp, predict_type="deriv"),
    "mlp-step": functools.partial(train_mlp, predict_type="step"),
    "nn-kernel-deriv": functools.partial(train_mlp, predict_type="deriv"),
    "nn-kernel-step": functools.partial(train_mlp, predict_type="step"),
    "cnn-deriv": functools.partial(train_cnn, predict_type="deriv"),
    "cnn-step": functools.partial(train_cnn, predict_type="step"),
    "unet-deriv": functools.partial(train_cnn, predict_type="deriv"),
    "unet-step": functools.partial(train_cnn, predict_type="step"),
}


def run_phase(base_dir, out_dir, phase_args):
    logger = logging.getLogger("train")
    base_dir = pathlib.Path(base_dir)
    out_dir = pathlib.Path(out_dir)
    training_args = phase_args["training"]

    # Set up noise injection
    noise_wrapper = create_live_noise(
        noise_args=training_args.get("noise", {}),
        base_logger=logger)

    # Load the data
    logger.info("Constructing dataset")
    train_dataset, train_loader, val_dataset, val_loader = create_dataset(base_dir, phase_args["train_data"], train_noise_wrapper=noise_wrapper)

    # Construct the network
    network_args = phase_args["network"]
    logger.info("Building network")
    net = methods.build_network(network_args)

    # Misc training parameters
    max_epochs = training_args["max_epochs"]
    device = select_device(try_gpu=training_args["try_gpu"], base_logger=logger)
    train_dtype, train_dtype_np = TRAIN_DTYPES[training_args.get("train_dtype", "float")]
    logger.info(f"Training in dtype {train_dtype}")
    train_type = training_args["train_type"]
    train_type_args = training_args["train_type_args"]

    # If training a knn, this is all we need.
    if train_type in {"knn-regressor", "knn-predictor"}:
        raise ValueError("Storing trained KNNs is no longer supported. Use oneshot mode instead.")
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

        # Move network to device and convert to dtype
        net = torch_converter(net)

        # Declare loss and training functions
        loss_fn = create_loss_fn(training_args.get("loss_type", "mse"), device=device)
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

                extra_train_args = {}
                if train_type in {"cnn-step", "cnn-deriv", "cnn", "unet-step", "unet-deriv", "mlp", "mlp-deriv", "mlp-step"} and torch.is_tensor(batch.extra_fixed_mask):
                    if epoch == 0 and batch_num == 0:
                        logger.info("Providing fixed mask as extra data")
                    _extra_data = torch_converter(batch.extra_fixed_mask)
                    _extra_data.requires_grad = False
                    extra_train_args = {"extra_data": _extra_data}


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
                with torch.no_grad():
                    for val_batch_num, val_batch in enumerate(val_loader):
                        extra_val_args = {}
                        if train_type in {"cnn-step", "cnn-deriv", "cnn", "unet-step", "unet-deriv", "mlp", "mlp-deriv", "mlp-step"} and torch.is_tensor(val_batch.extra_fixed_mask):
                            _val_extra_data = torch_converter(val_batch.extra_fixed_mask)
                            _val_extra_data.requires_grad = False
                            extra_val_args = {"extra_data": _val_extra_data}
                        # Run validation batch
                        val_result = train_fn(net=net, batch=val_batch,
                                              loss_fn=loss_fn,
                                              train_type_args=train_type_args,
                                              tensor_converter=torch_converter, **extra_val_args)
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
