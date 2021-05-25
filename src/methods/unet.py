from collections import namedtuple
import torch
import math
from .defs import NONLINEARITIES


TimeDerivative = namedtuple("TimeDerivative", ["dq_dt", "dp_dt"])
StepPrediction = namedtuple("StepPrediction", ["q", "p"])

RELU_LEAK = 0.2
BASE_CHANS = 64


class UNetBlock(torch.nn.Module):
    def __init__(self, in_chans, out_chans, transposed=False, bn=True, relu=True, size=4, pad=1):
        super().__init__()
        batch_norm = bn
        relu_leak = None if relu else RELU_LEAK
        kern_size = size
        ops = []
        # First the activation
        if relu_leak is None or relu_leak == 0:
            ops.append(torch.nn.ReLU(inplace=True))
        else:
            ops.append(torch.nn.LeakyReLU(negative_slope=relu_leak, inplace=True))
        # Next, the actual conv op
        if not transposed:
            # Regular conv
            ops.append(
                torch.nn.Conv2d(
                    in_channels=in_chans,
                    out_channels=out_chans,
                    kernel_size=kern_size,
                    stride=2,
                    padding=pad,
                    bias=True,
                )
            )
        else:
            # Upsample and transpose conv
            ops.append(torch.nn.Upsample(scale_factor=2, mode="bilinear"))
            ops.append(
                torch.nn.Conv2d(
                    in_channels=in_chans,
                    out_channels=out_chans,
                    kernel_size=(kern_size-1),
                    stride=1,
                    padding=pad,
                    bias=True,
                )
            )
        # Finally, optional batch norm
        if batch_norm:
            ops.append(torch.nn.BatchNorm2d(out_chans))
        # Bundle ops into Sequential
        self.ops = torch.nn.Sequential(*ops)

    def forward(self, x):
        return self.ops(x)


class UNet(torch.nn.Module):
    def __init__(
            self,
            predict_system,
            predict_type="deriv",
            spatial_reshape=None,
    ):
        super().__init__()
        self.predict_system = predict_system
        self.predict_type = predict_type
        self.spatial_reshape = spatial_reshape

        if self.predict_system == "navier-stokes":
            self.in_channels = 4
            self.out_channels = 3
            self.upsampler = torch.nn.Upsample(size=(256, 256), mode="bilinear")
        else:
            raise ValueError(f"Unsupported system {self.predict_system}")

        # Build network operations
        # ENCODER LAYERS
        self.input_ops = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=BASE_CHANS,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=True
                ),
            ),
            UNetBlock(BASE_CHANS, BASE_CHANS * 2, transposed=False, bn=True, relu=False),
            UNetBlock(BASE_CHANS * 2, BASE_CHANS * 2, transposed=False, bn=True, relu=False),
            UNetBlock(BASE_CHANS * 2, BASE_CHANS * 4, transposed=False, bn=True, relu=False),
            UNetBlock(BASE_CHANS * 4, BASE_CHANS * 8, transposed=False, bn=True, relu=False, size=4),
            UNetBlock(BASE_CHANS * 8, BASE_CHANS * 8, transposed=False, bn=True, relu=False, size=2, pad=0),
            UNetBlock(BASE_CHANS * 8, BASE_CHANS * 8, transposed=False, bn=False, relu=False, size=2, pad=0),
        ])

        # DECODER LAYERS
        self.output_ops = torch.nn.ModuleList([
            UNetBlock(BASE_CHANS * 8, BASE_CHANS * 8, transposed=True, bn=True, relu=True, size=2, pad=0),
            UNetBlock(BASE_CHANS * 8 * 2, BASE_CHANS * 8, transposed=True, bn=True, relu=True, size=2, pad=0),
            UNetBlock(BASE_CHANS * 8 * 2, BASE_CHANS * 4, transposed=True, bn=True, relu=True),
            UNetBlock(BASE_CHANS * 4 * 2, BASE_CHANS * 2, transposed=True, bn=True, relu=True),
            UNetBlock(BASE_CHANS * 2 * 2, BASE_CHANS * 2, transposed=True, bn=True, relu=True),
            UNetBlock(BASE_CHANS * 2 * 2, BASE_CHANS, transposed=True, bn=True, relu=True),
            torch.nn.Sequential(
                torch.nn.ReLU(inplace=True),
                torch.nn.ConvTranspose2d(
                    in_channels=BASE_CHANS * 2,
                    out_channels=self.out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=True,
                ),
            )
        ])

        # Initialize weights
        self.apply(self.__init_weights)

    @staticmethod
    def __init_weights(module):
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            module.weight.data.normal_(0.0, 0.02)
        elif isinstance(module, torch.nn.BatchNorm2d):
            module.weight.data.normal_(1.0, 0.02)
            module.bias.data.fill_(0)

    def _spatial_reshape(self, t):
        if not self.spatial_reshape:
            return t
        # Reshape input
        n_batch = t.shape[0]
        target_shape = (n_batch, ) + self.spatial_reshape + (t.shape[2: ] or (1, ))
        return t.view(target_shape)

    def _apply_ops(self, x):
        skip_connections = []
        # Encoder ops
        for op in self.input_ops:
            x = op(x)
            skip_connections.append(x)
        # Decoder ops
        x = skip_connections.pop()
        for op in self.output_ops:
            x = op(x)
            if skip_connections:
                x = torch.cat([x, skip_connections.pop()], dim=1)
        return x

    def forward(self, q, p, extra_data=None):
        # Preprocess inputs for shape
        orig_q_shape = q.shape
        orig_p_shape = p.shape

        # For Navier-Stokes: q=pressures, p=solutions
        if self.predict_system == "navier-stokes":
            # For Navier-Stokes: q=pressures, p=solutions
            # We only use p as input and require extra data
            p = self._spatial_reshape(p)
            extra_data = self._spatial_reshape(extra_data)
            split_size = [q.shape[-1], p.shape[-1]]
            x = torch.cat((p, extra_data), dim=-1)
            x = torch.movedim(x, -1, 1)

        # Apply operations
        orig_x_shape = x.shape[-2:]
        x = self.upsampler(x)
        y = self._apply_ops(x)
        y = torch.nn.functional.interpolate(y, size=orig_x_shape, mode="bilinear")
        y = torch.movedim(y, 1, -1)

        # Package output
        dq, dp = torch.split(y, split_size, dim=-1)
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
    predict_system = arch_args["predict_system"]
    spatial_reshape = None
    if "spatial_reshape" in arch_args:
        spatial_reshape = tuple(arch_args["spatial_reshape"])
    unet = UNet(
        predict_system=predict_system,
        predict_type=predict_type,
        spatial_reshape=spatial_reshape
    )
    return unet
