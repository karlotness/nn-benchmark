from collections import namedtuple
import torch
from .defs import NONLINEARITIES

TimeDerivative = namedtuple("TimeDerivative", ["dq_dt", "dp_dt"])


class NNKernel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 nonlinearity=torch.nn.Tanh):
        super().__init__()
        frozen_linear = torch.nn.Linear(input_dim, hidden_dim, bias=False)
        frozen_linear.weight.requires_grad_(False)
        nonlinear_op = nonlinearity()
        unfrozen_linear = torch.nn.Linear(hidden_dim, output_dim)
        self.ops = torch.nn.Sequential(frozen_linear,
                                       nonlinear_op,
                                       unfrozen_linear)

    def forward(self, q, p):
        x = torch.cat([q, p], dim=-1)
        ret = self.ops(x)
        split_size = ret.shape[-1] // 2
        dq, dp = torch.split(ret, split_size, dim=-1)
        return TimeDerivative(dq_dt=dq, dp_dt=dp)


def build_network(arch_args, predict_type):
    input_dim = arch_args["input_dim"]
    hidden_dim = arch_args["hidden_dim"]
    output_dim = arch_args["output_dim"]
    nonlinearity = NONLINEARITIES[arch_args["nonlinearity"]]
    kern = NNKernel(input_dim=input_dim, hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    nonlinearity=nonlinearity,
                    predict_type=predict_type)
    return kern
