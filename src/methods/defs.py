from collections import namedtuple
import torch

NONLINEARITIES = {
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
}

TimeDerivative = namedtuple("TimeDerivative", ["dq_dt", "dp_dt"])
StepPrediction = namedtuple("StepPrediction", ["q", "p"])
