import torch
from torch.autograd import grad
from .defs import NONLINEARITIES


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth=3, nonlinearity=torch.nn.Tanh):
        super().__init__()
        assert depth >= 2
        layers = [torch.nn.Linear(input_dim, hidden_dim), nonlinearity()]
        for i in range(depth - 2):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(nonlinearity())
        layers.append(torch.nn.Linear(hidden_dim, output_dim))
        self.ops = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.ops(x)


class SRNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth=3, nonlinearity=torch.nn.Tanh):
        super().__init__()
        self.k_mlp = MLP(input_dim, hidden_dim, output_dim, depth=depth, nonlinearity=nonlinearity)
        self.p_mlp = MLP(input_dim, hidden_dim, output_dim, depth=depth, nonlinearity=nonlinearity)

    def kinetic_energy(self, p):
        return self.k_mlp(p)

    def potential_energy(self, q):
        return self.p_mlp(q)

    def forward(self, p, q):
        return self.kinetic_energy(p) + self.potential_energy(q)


def build_network(arch_args):
    input_dim = arch_args["input_dim"]
    hidden_dim = arch_args["hidden_dim"]
    output_dim = arch_args["output_dim"]
    depth = arch_args["depth"]
    nonlinearity = NONLINEARITIES[arch_args["nonlinearity"]]
    return SRNN(input_dim=input_dim, hidden_dim=hidden_dim,
                output_dim=output_dim, depth=depth, nonlinearity=nonlinearity)
