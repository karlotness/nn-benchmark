from collections import namedtuple
import torch
from .defs import NONLINEARITIES

TimeDerivative = namedtuple("TimeDerivative", ["dq_dt", "dp_dt"])
StepPrediction = namedtuple("StepPrediction", ["q", "p"])


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth=3,
                 nonlinearity=torch.nn.Tanh, predict_type="deriv"):
        super().__init__()
        assert depth >= 2
        layers = [torch.nn.Linear(input_dim, hidden_dim), nonlinearity()]
        for i in range(depth - 2):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(nonlinearity())
        layers.append(torch.nn.Linear(hidden_dim, output_dim))
        self.ops = torch.nn.Sequential(*layers)
        self.predict_type = predict_type

    def forward(self, q, p):
        x = torch.cat([p, q], dim=-1)
        ret = self.ops(x)
        split_size = [p.shape[-1], q.shape[-1]]
        if self.predict_type == "deriv":
            dp, dq = torch.split(ret, split_size, dim=-1)
            result = TimeDerivative(dq_dt=dq, dp_dt=dp)
        elif self.predict_type == "step":
            p, q = torch.split(ret, split_size, dim=-1)
            result = StepPrediction(q=q, p=p)
        else:
            raise ValueError(f"Invalid predict type {self.predict_type}")

        return result


def build_network(arch_args, predict_type):
    input_dim = arch_args["input_dim"]
    hidden_dim = arch_args["hidden_dim"]
    output_dim = arch_args["output_dim"]
    depth = arch_args["depth"]
    nonlinearity = NONLINEARITIES[arch_args["nonlinearity"]]
    mlp = MLP(input_dim=input_dim, hidden_dim=hidden_dim,
              output_dim=output_dim, depth=depth,
              nonlinearity=nonlinearity, predict_type=predict_type)
    return mlp
