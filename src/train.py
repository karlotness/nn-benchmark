import pathlib
import methods
import torch
from torch import utils
import dataset
import logging
import json
import time
from sklearn import neighbors
import joblib
import integrators
import time


TRAIN_DTYPES = {
    "float": torch.float,
    "double": torch.double,
}


def save_network(net, network_args, train_type, out_dir, base_logger):
    logger = base_logger.getChild("save_network")
    logger.info("Saving network")

    if train_type == "knn":
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
    loader = utils.data.DataLoader(data_set,
                                   batch_size=loader_args["batch_size"],
                                   shuffle=loader_args["shuffle"])
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

    # Load the data
    logger.info("Constructing dataset")
    train_dataset, train_loader = create_dataset(base_dir, phase_args["train_data"])

    # If training a knn_regressor, this is all we need.
    if train_type == "knn_regressor":
        logger.info("Starting fitting of dataset for KNN Regressor.")

        data = np.stack([np.stack([batch.p, batch.q, batch.dp, batch.dq], axis=-1)
                      for batch in train_loader], axis=0)
        print(data.shape)
        net.fit(data[..., 0:2], data[..., 2:4])

        logger.info("Finished fitting of dataset for KNN Regressor.")

        # Save the network
        save_network(net=net, network_args=network_args, train_type=train_type,
                     out_dir=out_dir, base_logger=logger)

        # Save the run statistics

        return

    # Construct the optimizer
    logger.info("Creating optimizer")
    optimizer = training_args["optimizer"]
    optim_args = training_args["optimizer_args"]
    optim = create_optimizer(net, optimizer, optim_args)

    # Misc training parameters
    max_epochs = training_args["max_epochs"]
    device = select_device(try_gpu=training_args["try_gpu"], base_logger=logger)
    train_dtype = TRAIN_DTYPES[training_args.get("train_dtype", "float")]
    logger.info(f"Training in dtype {train_dtype}")
    train_type = training_args["train_type"]
    train_type_args = training_args["train_type_args"]
    
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
            time_start = time.perf_counter()

            p = batch.p.to(device, dtype=train_dtype)
            q = batch.q.to(device, dtype=train_dtype)
            dp_dt = batch.dp_dt.to(device, dtype=train_dtype)
            dq_dt = batch.dq_dt.to(device, dtype=train_dtype)
            trajectory_meta = batch.trajectory_meta

            # Reset optimizer
            optim.zero_grad()

            time_forward_start = time.perf_counter()
            if train_type == "hnn":
                # Assume snapshot dataset (shape [batch_size, n_grid])
                x = torch.cat([p, q], dim=-1)
                dx_dt = torch.cat([dp_dt, dq_dt], dim=-1)
                dx_dt_pred = net.time_derivative(x)
                loss = loss_fn(dx_dt_pred, dx_dt)
                total_loss_denom += p.shape[0]
            elif train_type == "srnn":
                # Assume rollout dataset (shape [batch_size, dataset rollout_length, n_grid])
                x = torch.cat([p, q], dim=-1)
                method_hnet = 5
                training_steps = train_type_args["rollout_length"]
                time_step_size = float(trajectory_meta["time_step_size"][0])
                # Check that all time step sizes are equal
                if not torch.all(trajectory_meta["time_step_size"] == time_step_size):
                    raise ValueError("Inconsistent time step sizes in batch")
                int_res = integrators.numerically_integrate(
                    train_type_args["integrator"],
                    p[:, 0],
                    q[:, 0],
                    model=net,
                    method=method_hnet,
                    T=training_steps,
                    dt=time_step_size,
                    volatile=False,
                    device=device,
                    coarsening_factor=1).permute(1, 0, 2)
                loss = loss_fn(int_res, x)
                total_loss_denom += p.shape[0] * p.shape[1]
            elif train_type == "mlp":
                # Assume snapshot dataset (shape [batch_size, n_grid])
                x = torch.cat([p, q], dim=-1)
                dx_dt = torch.cat([dp_dt, dq_dt], dim=-1)
                dx_dt_pred = net(x)
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

            logger.info("Batch {} inference running time: {}".format(
                batch_num, time.perf_counter() - time_start))
            time_start = time.perf_counter()

            loss.backward()
            optim.step()
            optim.zero_grad()

            logger.info("Batch {} optimization running time: {}".format(
                batch_num, time.perf_counter() - time_start))

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
