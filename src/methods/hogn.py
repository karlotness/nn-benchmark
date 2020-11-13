import torch
from torch.nn import Sequential as Seq, Linear as Lin, Softplus
from torch_geometric.nn import MessagePassing
from torch.autograd import grad


class HGN(MessagePassing):
    def __init__(self, n_f, ndim, hidden=300):
        super().__init__(aggr='add')  # "Add" aggregation.
        self.pair_energy = Seq(
            Lin(2*n_f, hidden),
            Softplus(),
            Lin(hidden, hidden),
            Softplus(),
            Lin(hidden, hidden),
            Softplus(),
            Lin(hidden, 1)
        )

        self.self_energy = Seq(
            Lin(n_f, hidden),
            Softplus(),
            Lin(hidden, hidden),
            Softplus(),
            Lin(hidden, hidden),
            Softplus(),
            Lin(hidden, 1)
        )
        self.ndim = ndim

    def forward(self, x, edge_index):
        # x is [n, n_f]
        x = x
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i, x_j):
        # x_i has shape [n_e, n_f]; x_j has shape [n_e, n_f]
        tmp = torch.cat([x_i, x_j], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.pair_energy(tmp)

    def update(self, aggr_out, x=None):
        # aggr_out has shape [n, msg_dim]

        sum_pair_energies = aggr_out
        self_energies = self.self_energy(x)
        return sum_pair_energies + self_energies

    def just_derivative(self, g, augment=False, augmentation=3):
        # x is [n, n_f]f
        x = g.x
        ndim = self.ndim
        if augment:
            augmentation = torch.randn(1, ndim)*augmentation
            augmentation = augmentation.repeat(len(x), 1).to(x.device)
            x = x.index_add(1, torch.arange(ndim).to(x.device), augmentation)

        #Make momenta:
        x = torch.cat((x[:, :ndim], x[:, ndim:2*ndim]*x[:, [-1]*ndim], x[:, 2*ndim:]), dim=1)
        x.requires_grad_()

        edge_index = g.edge_index
        total_energy = self.propagate(
                edge_index, size=(x.size(0), x.size(0)),
                x=x).sum()

        dH = grad(total_energy, x, create_graph=True)[0]
        dH_dq = dH[:, :ndim]
        dH_dp = dH[:, ndim:2*ndim]

        dq_dt = dH_dp
        dp_dt = -dH_dq
        dv_dt = dp_dt/x[:, [-1]*ndim]
        return torch.cat((dq_dt, dv_dt), dim=1)

    def loss(self, g, augment=True, square=False, reg=True, augmentation=3, **kwargs):
        all_derivatives = self.just_derivative(g, augment=augment, augmentation=augmentation)
        ndim = self.ndim
        dv_dt = all_derivatives[:, self.ndim:]

        if reg:
            ## If predicting dq_dt too, the following regularization is important:
            edge_index = g.edge_index
            x = g.x
            #make momenta:
            x = torch.cat((x[:, :ndim], x[:, ndim:2*ndim]*x[:, [-1]*ndim], x[:, 2*ndim:]), dim=1)
            x.requires_grad_()

            self_energies = self.self_energy(x)
            total_energy = self.propagate(
                    edge_index, size=(x.size(0), x.size(0)),
                    x=x)
            #pair_energies = total_energy - self_energies
            #regularization = 1e-3 * torch.sum((pair_energies)**2)
            dH = grad(total_energy.sum(), x, create_graph=True)[0]
            dH_dother = dH[2*ndim:]
            #Punish total energy and gradient with respect to other variables:
            regularization = 1e-6 * (torch.sum((total_energy)**2) + torch.sum((dH_dother)**2))
            return torch.sum(torch.abs(g.y - dv_dt)) + regularization
        else:
            return torch.sum(torch.abs(g.y - dv_dt))
