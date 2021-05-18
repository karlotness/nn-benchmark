from collections import namedtuple
import torch
import math
from .defs import NONLINEARITIES


TimeDerivative = namedtuple("TimeDerivative", ["dq_dt", "dp_dt"])
StepPrediction = namedtuple("StepPrediction", ["q", "p"])
LayerDef = namedtuple("LayerDef", ["kernel_size", "in_chans", "out_chans"])


class CNN(torch.nn.Module):
    def __init__(self, layer_defs,
                 nonlinearity=torch.nn.ReLU, predict_type="deriv", dim=1, spatial_reshape=None):
        super().__init__()
        layers = []
        self.dim = dim
        assert layer_defs[0].in_chans == layer_defs[-1].out_chans
        for layer_def in layer_defs:
            kern_size = layer_def.kernel_size
            pad = (kern_size - 1) // 2
            _conv = self._make_conv(
                in_chans=layer_def.in_chans,
                out_chans=layer_def.out_chans,
                kern_size=kern_size,
                pad=pad,
                dim=self.dim,
            )
            layers.append(_conv)
            layers.append(nonlinearity())
        # Remove final nonlinearity
        layers = layers[:-1]
        self.ops = torch.nn.Sequential(*layers)
        self.predict_type = predict_type
        self.spatial_reshape = spatial_reshape

    @staticmethod
    def _make_conv(in_chans, out_chans, kern_size, pad, dim=1):
        if dim == 1:
            return torch.nn.Conv1d(
                in_channels=in_chans,
                out_channels=out_chans,
                kernel_size=kern_size,
                padding=pad,
            )
        elif dim == 2:
            return torch.nn.Conv2d(
                in_channels=in_chans,
                out_channels=out_chans,
                kernel_size=kern_size,
                padding=pad,
            )
        else:
            raise ValueError(f"Invalid convolution dimension {dim}")

    def _spatial_reshape(self, t):
        if not self.spatial_reshape:
            return t
        # Reshape input
        n_batch = t.shape[0]
        target_shape = (n_batch, ) + self.spatial_reshape + (t.shape[2: ] or (1, ))
        return t.view(target_shape)

    def forward(self, q, p, extra_data=None):
        # Preprocess inputs for shape
        orig_q_shape = q.shape
        orig_p_shape = p.shape
        q = self._spatial_reshape(q)
        p = self._spatial_reshape(p)
        if extra_data is not None:
            extra_data = (self._spatial_reshape(extra_data), )
            extra_chans = extra_data[0].shape[-1]
        else:
            extra_chans = 0
            extra_data = ()

        # Concatenate input
        # Pass through operations
        # Split input
        x = torch.movedim(torch.cat((q, p) + extra_data, dim=-1), -1, 1)
        split_size = [q.shape[-1], p.shape[-1], extra_chans]
        y = torch.movedim(self.ops(x), 1, -1)
        dq, dp, _extra = torch.split(y, split_size, dim=-1)
        dq = dq.view(orig_q_shape)
        dp = dp.view(orig_p_shape)

        # Package result value
        if self.predict_type == "deriv":
            result = TimeDerivative(dq_dt=dq, dp_dt=dp)
        elif self.predict_type == "step":
            result = StepPrediction(q=dq, p=dp)
        else:
            raise ValueError(f"Invalid predict type {self.predict_type}")

        return result


def build_network(arch_args, predict_type):
    nonlinearity = NONLINEARITIES[arch_args.get("nonlinearity", "relu")]
    dim = int(arch_args.get("dim", 1))
    spatial_reshape = None
    if "spatial_reshape" in arch_args:
        spatial_reshape = tuple(arch_args["spatial_reshape"])
    layer_defs = []
    for record in arch_args["layer_defs"]:
        layer_def = LayerDef(
            kernel_size=record["kernel_size"],
            in_chans=record["in_chans"],
            out_chans=record["out_chans"],
        )
        layer_defs.append(layer_def)
    cnn = CNN(layer_defs=layer_defs,
              nonlinearity=nonlinearity,
              predict_type=predict_type,
              dim=dim,
              spatial_reshape=spatial_reshape)
    return cnn
