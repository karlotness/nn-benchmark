import torch
from .defs import NONLINEARITIES, TimeDerivative, StepPrediction


class NNKernel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 nonlinearity=torch.nn.Tanh, predict_type="deriv"):
        super().__init__()
        frozen_linear = torch.nn.Linear(input_dim, hidden_dim, bias=False)
        frozen_linear.weight.requires_grad_(False)
        nonlinear_op = nonlinearity()
        unfrozen_linear = torch.nn.Linear(hidden_dim, output_dim)
        self.ops = torch.nn.Sequential(frozen_linear,
                                       nonlinear_op,
                                       unfrozen_linear)
        self.predict_type = predict_type

    def forward(self, q, p, extra_data=None):
        # Ignore extra data
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
    nonlinearity = NONLINEARITIES[arch_args["nonlinearity"]]
    kern = NNKernel(input_dim=input_dim, hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    nonlinearity=nonlinearity,
                    predict_type=predict_type)
    return kern
