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

    def forward(self, q, p, extra_data=None):
        pass


def build_network(arch_args, predict_type):
    pass
