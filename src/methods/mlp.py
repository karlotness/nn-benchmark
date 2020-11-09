from collections import namedtuple
import torch
from .defs import NONLINEARITIES

TimeDerivative = namedtuple("TimeDerivative", ["dq_dt", "dp_dt"])

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth=3,
                 nonlinearity=torch.nn.Tanh):
        super().__init__()
        assert depth >= 2
        layers = [torch.nn.Linear(input_dim, hidden_dim), nonlinearity()]
        for i in range(depth - 2):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(nonlinearity())
        layers.append(torch.nn.Linear(hidden_dim, output_dim))
        self.ops = torch.nn.Sequential(*layers)

    def forward(self, q, p):
        x = torch.cat([p, q], dim=-1)
        ret = self.ops(x)
        split_size = ret.shape[-1] // 2
        dp, dq = torch.split(ret, split_size, dim=-1)
        return TimeDerivative(dq_dt=dq, dp_dt=dp)


def build_network(arch_args):
    input_dim = arch_args["input_dim"]
    hidden_dim = arch_args["hidden_dim"]
    output_dim = arch_args["output_dim"]
    depth = arch_args["depth"]
    nonlinearity = NONLINEARITIES[arch_args["nonlinearity"]]
    mlp = MLP(input_dim=input_dim, hidden_dim=hidden_dim,
              output_dim=output_dim, depth=depth,
              nonlinearity=nonlinearity)
    return mlp
