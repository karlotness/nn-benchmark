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


TRAIN_DTYPES = {
    "float": (torch.float, np.float32),
    "double": (torch.double, np.float64)
}


def save_network(net, network_args, train_type, out_dir, base_logger):
    logger = base_logger.getChild("save_network")
    logger.info("Saving network")

    if train_type == "knn-regressor":
        joblib.dump(net, out_dir / "model.pt")
    else:
        torch.save(net.state_dict(), out_dir / "model.pt")

    with open(out_dir / "model.json", "w", encoding="utf8") as model_file:
        json.dump(network_args, model_file)
    logger.info("Saved network")


def create_optimizer(net, optimizer, optim_args):
    if optimizer == "adam":
        return torch.optim.Adam(net.parameters(),
                                lr=optim_args["learning_rate"])
    else:
        raise ValueError(f"Invalid optimizer {optimizer}")


def create_dataset(base_dir, data_args):
    data_dir = base_dir / data_args["data_dir"]
    base_data_set = dataset.TrajectoryDataset(data_dir=data_dir)
    dataset_type = data_args["dataset"]
    if dataset_type == "trajectory":
        data_set = base_data_set
    elif dataset_type == "snapshot":
        data_set = dataset.SnapshotDataset(traj_dataset=base_data_set)
    elif dataset_type == "rollout-chunk":
        rollout_length = int(data_args["dataset_args"]["rollout_length"])
        data_set = dataset.RolloutChunkDataset(traj_dataset=base_data_set,
                                               rollout_length=rollout_length)
    else:
        raise ValueError(f"Invalid dataset type {dataset_type}")
    loader_args = data_args["loader"]
    loader_type = loader_args.get("type", "pytorch")
    if loader_type == "pytorch":
        loader = utils.data.DataLoader(data_set,
                                       batch_size=loader_args["batch_size"],
                                       shuffle=loader_args["shuffle"])
    elif loader_type == "pytorch-geometric":
        # Lazy import to avoid pytorch-geometric if possible
        import dataset_geometric
        from torch_geometric import data as geometric_data
        package_args = loader_args["package_args"]
        loader = geometric_data.DataLoader(
            dataset=dataset_geometric.package_data(data_set,
                                                   package_args=package_args),
            batch_size=loader_args["batch_size"],
            shuffle=loader_args["shuffle"])
    else:
        raise ValueError(f"Invalid loader type {loader_type}")
    return data_set, loader


def select_device(try_gpu, base_logger):
    logger = base_logger.getChild("select_device")
    device = torch.device("cpu")
    if try_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    elif try_gpu:
        logger.warning("Using CPU despite trying for GPU")
    logger.info(f"Using device {device}")
    return device


def run_phase(base_dir, out_dir, phase_args):
    logger = logging.getLogger("train")
    base_dir = pathlib.Path(base_dir)
    out_dir = pathlib.Path(out_dir)
    training_args = phase_args["training"]

    # Construct the network
    logger.info("Building network")
    network_args = phase_args["network"]
    net = methods.build_network(network_args)

    # Misc training parameters
    max_epochs = training_args["max_epochs"]
    device = select_device(try_gpu=training_args["try_gpu"], base_logger=logger)
    train_dtype, train_dtype_np = TRAIN_DTYPES[training_args.get("train_dtype", "float")]
    logger.info(f"Training in dtype {train_dtype}")
    train_type = training_args["train_type"]
    train_type_args = training_args["train_type_args"]

    # Load the data
    logger.info("Constructing dataset")
    train_dataset, train_loader = create_dataset(base_dir, phase_args["train_data"])

    # If training a knn-regressor, this is all we need.
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

    else:
        # Construct the optimizer
        logger.info("Creating optimizer")
        optimizer = training_args["optimizer"]
        optim_args = training_args["optimizer_args"]
        optim = create_optimizer(net, optimizer, optim_args)

        # Move network to device and convert to dtype
        net = net.to(device, dtype=train_dtype)

        # Declare loss function
        loss_fn = torch.nn.MSELoss()

        # Run training epochs
        logger.info("Starting training")
        epoch_stats = []
        for epoch in range(max_epochs):
            logger.info(f"Epoch {epoch} of {max_epochs}")
            total_forward_time = 0
            total_backward_time = 0
            time_epoch_start = time.perf_counter()
            total_loss = 0
            total_loss_denom = 0
            for batch_num, batch in enumerate(train_loader):
                p = batch.p.to(device, dtype=train_dtype)
                q = batch.q.to(device, dtype=train_dtype)
                p_noiseless = batch.p_noiseless.to(device, dtype=train_dtype)
                q_noiseless = batch.q_noiseless.to(device, dtype=train_dtype)
                dp_dt = batch.dp_dt.to(device, dtype=train_dtype)
                dq_dt = batch.dq_dt.to(device, dtype=train_dtype)
                trajectory_meta = batch.trajectory_meta

                # Reset optimizer
                optim.zero_grad()

                time_forward_start = time.perf_counter()
                if train_type == "hnn":
                    # Assume snapshot dataset (shape [batch_size, n_grid])
                    deriv_pred = net.time_derivative(p=p, q=q)
                    dx_dt = torch.cat([dp_dt, dq_dt], dim=-1)
                    dx_dt_pred = torch.cat([deriv_pred.dp_dt, deriv_pred.dq_dt], dim=-1)
                    loss = loss_fn(dx_dt_pred, dx_dt)
                    total_loss_denom += p.shape[0]
                elif train_type == "srnn":
                    # Assume rollout dataset (shape [batch_size, dataset rollout_length, n_grid])
                    method_hnet = 5
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
                        method=method_hnet,
                        T=training_steps,
                        dt=time_step_size,
                        volatile=False,
                        device=device,
                        coarsening_factor=1)
                    x = torch.cat([p_noiseless, q_noiseless], dim=-1)
                    x_pred = torch.cat([int_res.p, int_res.q], dim=-1)
                    loss = loss_fn(x_pred, x)
                    total_loss_denom += p.shape[0] * p.shape[1]
                elif train_type == "mlp":
                    # Assume snapshot dataset (shape [batch_size, n_grid])
                    deriv_pred = net(p=p, q=q)
                    dx_dt = torch.cat([dp_dt, dq_dt], dim=-1)
                    dx_dt_pred = torch.cat([deriv_pred.dp_dt, deriv_pred.dq_dt], dim=-1)
                    loss = loss_fn(dx_dt_pred, dx_dt)
                    total_loss_denom += p.shape[0]
                else:
                    raise ValueError(f"Invalid train type: {train_type}")
                total_forward_time += time.perf_counter() - time_forward_start

                # Training step
                time_backward_start = time.perf_counter()
                loss.backward()
                optim.step()
                total_backward_time += time.perf_counter() - time_backward_start
                total_loss += loss.item()

            total_epoch_time = time.perf_counter() - time_epoch_start
            avg_loss = total_loss / total_loss_denom
            logger.info(f"Epoch complete. Avg loss: {avg_loss}, time: {total_epoch_time}")
            # Compute per-epoch statistics
            epoch_stats.append({
                "num_batches": batch_num + 1,
                "avg_loss": avg_loss,
                "timing": {
                    "total_forward": total_forward_time,
                    "total_backward": total_backward_time,
                    "total_epoch": total_epoch_time,
                }
            })

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
