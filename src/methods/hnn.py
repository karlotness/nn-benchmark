import torch
from .defs import NONLINEARITIES


def permutation_tensor(n):
    mat = torch.eye(n)
    return torch.cat((mat[n//2:], -mat[:n//2]))


class HNN(torch.nn.Module):
    def __init__(self, input_dim, base_model, field_type="solenoidal"):
        super().__init__()
        self.base_model = base_model
        self._permute_mat = permutation_tensor(input_dim)
        self.field_type = field_type

    def forward(self, x):
        y = self.base_model(x)
        return y.split(1, 1)

    def time_derivative(self, x):
        f1, f2 = self.forward(x)

        if self.field_type != "solenoidal":
            d_f1 = torch.autograd.grad(f1.sum(), x, create_graph=True)[0]
            conservative_field = d_f1 @ torch.eye(*self._permute_mat.shape)
        else:
            conservative_field = torch.zeros_like(x)

        if self.field_type != "conservative":
            d_f2 = torch.autograd.grad(f2.sum(), x, create_graph=True)[0]
            solenoidal_field = d_f2 @ self._permute_mat.t()
        else:
            solenoidal_field = torch.zeros_like(x)

        return conservative_field + solenoidal_field


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


def build_hnn(input_dim, base_model, hnn_args):
    field_type = hnn_args["field_type"]
    return HNN(input_dim=input_dim, base_model=base_model,
               field_type=field_type)


def build_network(arch_args):
    base_model = arch_args["base_model"]
    input_dim = arch_args["input_dim"]
    base_model_args = arch_args["base_model_args"]
    if base_model == "mlp":
        base_model = build_mlp(input_dim=input_dim,
                               base_model_args=base_model_args)
    else:
        raise ValueError(f"Invalid inner model for HNN: {base_model}")

    hnn = build_hnn(input_dim=input_dim, base_model=base_model,
                    hnn_args=arch_args["hnn_args"])
    return hnn
