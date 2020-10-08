import torch
from .defs import NONLINEARITIES


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

    def forward(self, x):
        return self.ops(x)


def build_mlp(input_dim, base_model_args):
    hidden_dim = base_model_args["hidden_dim"]
    output_dim = base_model_args["output_dim"]
    depth = base_model_args["depth"]
    nonlinearity = NONLINEARITIES[base_model_args["nonlinearity"]]
    return MLP(input_dim=input_dim, hidden_dim=hidden_dim,
               output_dim=output_dim, depth=depth,
               nonlinearity=nonlinearity)


def build_network(arch_args):
    input_dim = arch_args["input_dim"]
    base_model_args = arch_args["base_model_args"]
    mlp = build_mlp(input_dim=input_dim, base_model_args=base_model_args)

    return mlp
