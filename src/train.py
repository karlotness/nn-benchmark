import pathlib
import methods
import torch
from torch import utils
import dataset
import logging
import json
import time
from sklearn import neighbors
from sklearn.externals import joblib


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
    data_set = dataset.TrajectoryDataset(data_dir=data_dir)
    loader_args = data_args["loader"]
    loader = utils.DataLoader(data_set,
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

    # Construct the optimizer
    logger.info("Creating optimizer")
    optimizer = training_args["optimizer"]
    optim_args = training_args["optimizer_args"]
    optim = create_optimizer(net, optimizer, optim_args)

    # Load the data
    logger.info("Constructing dataset")
    train_dataset, train_loader = create_dataset(base_dir, phase_args["train_data"])

    # Misc training parameters
    max_epochs = training_args["max_epochs"]
    device = select_device(try_gpu=training_args["try_gpu"], base_logger=logger)
    train_dtype = TRAIN_DTYPES[training_args.get("train_dtype", "float")]
    logger.info(f"Training in dtype {train_dtype}")
    train_type = training_args["train_type"]  # hnn or srnn. TODO: handle
    train_type_args = training_args["train_type_args"]

    if train_type == "knn_regressor":
        net = neighbors.KNeighborsRegressor(n_neighbors=5)

        logger.info("Starting fitting of dataset for KNN Regressor.")

        data = np.stack([np.stack([batch.p, batch.q, batch.dp, batch.dq], axis=-1)
                      for batch in train_loader], axis=0)
        net.fit(data[..., 0:2], data[..., 2:4])

        logger.info("Finished fitting of dataset for KNN Regressor.")

        # Save the network
        save_network(net=net, network_args=network_args, train_type=train_type,
                     out_dir=out_dir, base_logger=logger)

        # Save the run statistics

        return


    # Move network to device and convert to dtype
    net = net.to(device, dtype=train_dtype)

    # Declare loss function
    loss_fn = torch.nn.MSELoss()

    # Run training epochs
    logger.info("Starting training")
    for epoch in range(max_epochs):
        for batch_num, batch in enumerate(train_loader):
            time_start = time.perf_counter()

            p = batch.p.to(device, dtype=train_dtype)
            q = batch.q.to(device, dtype=train_dtype)
            dp_dt = batch.dp_dt.to(device, dtype=train_dtype)
            dq_dt = batch.dq_dt.to(device, dtype=train_dtype)
            trajectory_meta = batch.trajectory_meta
            if train_type == "hnn":
                x = torch.stack([p, q], dim=-1)
                # TODO(arvi): Fix this for different dimensions
                x = x.reshape([-1, 2])
                dx_dt = torch.stack([dp_dt, dq_dt], dim=-1)
                # TODO(arvi): Fix this for different dimensions
                dx_dt = dx_dt.reshape([-1, 2])
                dx_dt_pred = net.time_derivative(x)
                loss = loss_fn(dx_dt_pred, dx_dt)
            elif train_type == "srnn":
                method_hnet = 5
                training_steps = train_type_args["rollout_length"]
                # TODO(arvi): Make sure that all of the following are equal
                time_step_size = trajectory_meta["time_step_size"][0]
                int_res = integrators.numerically_integrate(
                    train_type_args["integrator"],
                    # TODO(arvi): Fix this for different dimensions
                    p[:, 0, :],
                    q[:, 0, :],
                    model=net,
                    method=method_hnet,
                    T=training_steps,
                    dt=time_step_size,
                    volatile=False,
                    device=device,
                    coarsening_factor=1).permute(1, 0, 2)
                loss = loss_fn(int_res, x)
            elif train_type == "mlp":
                x = torch.stack([p, q], dim=-1)
                x = x.reshape([-1, 2])
                dx_dt = torch.stack([dp_dt, dq_dt], dim=-1)
                dx_dt = dx_dt.reshape([-1, 2])
                dx_dt_pred = net(x)
                loss = loss_fn(dx_dt_pred, dx_dt)
            else:
                raise ValueError(f"Invalid train type: {train_type}")

            logger.info("Batch {} inference running time: {}".format(
                batch_num, time.perf_counter() - time_start))
            time_start = time.perf_counter()

            loss.backward()
            optim.step()
            optim.zero_grad()

            logger.info("Batch {} optimization running time: {}".format(
                batch_num, time.perf_counter() - time_start))

    logger.info("Training done")

    # Save the network
    save_network(net=net, network_args=network_args, train_type=train_type,
                 out_dir=out_dir, base_logger=logger)

    # Save the run statistics
