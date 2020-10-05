import pathlib
import methods
import torch
from torch import utils
import dataset
import logging
import json


TRAIN_DTYPES = {
    "float": torch.float,
    "double": torch.double,
}


def save_network(net, network_args, out_dir, base_logger):
    logger = base_logger.getChild("save_network")
    logger.info("Saving network")
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

    # Move network to device and convert to dtype
    net = net.to(device, dtype=train_dtype)

    # Run training epochs
    logger.info("Starting training")
    for epoch in range(max_epochs):
        for batch_num, batch in enumerate(train_loader):
            p = batch.p.to(device, dtype=train_dtype)
            q = batch.q.to(device, dtype=train_dtype)
            dp_dt = batch.dp_dt.to(device, dtype=train_dtype)
            dq_dt = batch.dq_dt.to(device, dtype=train_dtype)
            if train_type == "hnn":
                pass
            elif train_type == "srnn":
                pass
            else:
                raise ValueError(f"Invalid train type: {train_type}")

    logger.info("Training done")

    # Save the network
    save_network(net=net, network_args=network_args,
                 out_dir=out_dir, base_logger=logger)

    # Save the run statistics
