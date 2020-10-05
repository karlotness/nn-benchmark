import torch


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
