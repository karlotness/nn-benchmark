import argparse
import pathlib
import numpy as np
import torch
from methods import hnn, srnn
import data
import integrators
import utils

parser = argparse.ArgumentParser()
parser.add_argument("out_dir", type=str,
                    help="Directory to store results")
parser.add_argument("data_dir", type=str,
                    help="Directory from which to load data")
parser.add_argument("net_dir", type=str,
                    help="Directory from which to load the neural net")
parser.add_argument("--architecture", type=str, default="hnn",
                    choices=["hnn", "srnn"],
                    help="Neural net architecture type")
parser.add_argument("--rollout_length", type=int, default=15,
                    help="The number of time steps to integrate through")
parser.add_argument('--log', type=str, default="INFO",
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    help="Minimum Python log level")

INPUT_DIM = 2
HIDDEN_DIM = 200
OUTPUT_DIM = 2
DEPTH = 3
NONLINEARITY = torch.nn.Tanh
METHOD_HNET = 5
TIME_STEP_SIZE = 1e-1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(out_dir, data_dir, net_dir, architecture):
    out_dir = pathlib.Path(out_dir)
    data_dir = pathlib.Path(data_dir)
    net_dir = pathlib.Path(net_dir)
    # Load network
    if architecture == "hnn":
        _inner_model = hnn.MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, depth=DEPTH,
                               nonlinearity=NONLINEARITY)
        net = hnn.HNN(INPUT_DIM, _inner_model)
    elif architecture == "srnn":
        net = srnn.SRNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, depth=DEPTH,
                        nonlinearity=NONLINEARITY)
    net.load_state_dict(torch.load(data_dir / 'model.pt'))
    net = net.to(device)
    integrator = "leapfrog"
    # Load data
    dataset = data.TrajectoryDataset(data_dir=data_dir,
                                     split=data.DatasetSplit.EVALUATE)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                         pin_memory=(device.type == 'cuda'))

    batch_evaluations = []
    # Run evaluation loop
    for batch in loader:
        # Pass the batch through the net + integrator combo
        # Store out the steps that result and report statistics
        initial_step = batch[0]
        p_0s = initial_step[0]
        q_0s = initial_step[1]
        int_res = integrators.numerically_integrate(integrator, p_0s, q_0s, model=net, method=METHOD_HNET, T=batch.shape[0],
                                                    dt=TIME_STEP_SIZE, volatile=True, device=device, coarsening_factor=1)\
                      .permute(1, 0, 2)
        batch_evaluations.append(int_res)

    # Save the evaluation trajectories
    trajectories = {}
    for i, batch_eval in enumerate(batch_evaluations):
        trajectories[f"test{i:05}"] = batch_eval
    np.savez(out_dir / 'learned_trajectories.npz', **trajectories)


if __name__ == "__main__":
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    utils.set_up_logging(args.log, out_file=out_dir / "run.log")
    evaluate(out_dir=out_dir, net_dir=args.net_dir, data_dir=args.data_dir,
             architecture=args.architecture)
