import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn import MetaLayer
from torch.autograd import grad
from torch_geometric.data import data
from torch_scatter import scatter_mean, scatter_sum
import numpy as np
from collections import namedtuple


TimeDerivative = namedtuple("TimeDerivative", ["dq_dt", "dp_dt"])


def package_batch(p, q, dp_dt, dq_dt, masses, edge_index):
    # convert momenta to velocities
    v = p / masses
    x = torch.from_numpy(np.concatenate((q, v, masses), axis=-1))
    if dp_dt is not None and dq_dt is not None:
        dv_dt = dp_dt / masses
        y = torch.from_numpy(np.concatenate((dq_dt, dv_dt), axis=-1))
    else:
        y = None
    ret = data.Data(x=x, edge_index=edge_index, y=y)
    return ret


def unpackage_time_derivative(input_data, deriv):
    # format: deriv = torch.cat((dq_dt, dv_dt), dim=1)
    masses = input_data.x[..., -1]
    dq_dt, dv_dt = np.split(deriv, 2, axis=1)
    dp_dt = dv_dt * masses
    return TimeDerivative(dq_dt=dq_dt.reshape((1, -1)),
                          dp_dt=dp_dt.reshape((1, -1)))


class Residual(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


class EdgeModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden=128):
        super(EdgeModel, self).__init__()
        self.edge_mlp = Residual(Seq(Lin(input_dim, hidden),
                                     ReLU(), Lin(hidden, output_dim)))

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        return self.edge_mlp(out)


class NodeModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden=128):
        super(NodeModel, self).__init__()
        # self.node_mlp_1 = Residual(Seq(Lin(..., hidden),
        #                                ReLU(), Lin(hidden, ...)))
        self.node_mlp_2 = Residual(Seq(Lin(input_dim, hidden),
                                       ReLU(), Lin(hidden, output_dim)))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index

        # Default
        # out = torch.cat([x[row], edge_attr], dim=1)
        # out = self.node_mlp_1(out)
        # out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        # out = torch.cat([x, out, u[batch]], dim=1)

        # Custom
        edge_attr_sum = scatter_sum(edge_attr, row, dim=0) #, dim_size=x.size(0))
        out = torch.cat([x, edge_attr_sum], dim=1)
        return self.node_mlp_2(out)


class GN:
    def __init__(self, num_v_features, num_e_features, mesh_coords,
                 static_nodes, weights, hidden=128):
        super().__init__()

        self.mesh_coords = mesh_coords
        self.static_nodes = torch.nn.functional.one_hot(static_nodes, 2)
        self.weights = weights

        self.process = MetaLayer(
            EdgeModel(num_e_features, num_e_features, hidden),
            NodeModel(num_v_features, num_v_features, hidden),
            None)
        self.decode = Seq(Lin(num_v_features, hidden), ReLU(),
                          Lin(hidden, num_v_features))

    def encode(self, world_coords, prev_world_coords, edge_index):
        vertices = torch.cat([self.static_nodes,
                              world_coords - prev_world_coords], dim=-1)
        if self.weights:
            vertices = torch.cat([vertices, self.weights], dim=-1)

        row, col = edge_index
        edge_vectors = world_coords[col] - world_coords[row]
        edge_vectors_norm = torch.norm(edge_vectors, dim=-1, keepdims=True)
        edge_attr = torch.cat([edge_vectors, edge_vectors_norm], dim=-1)
        if self.mesh_coords:
            mesh_vectors = self.mesh_coords[col] - self.mesh_coords[row]
            mesh_vectors_norm = torch.norm(mesh_vectors, dim=-1, keepdim=True)
            edge_attr = torch.cat([edge_attr, mesh_vectors, mesh_vectors_norm],
                                  dim=-1)

        return vertices, edge_attr

    def forward(self, world_coords, prev_world_coords, edge_index):
        # x is [n, n_f]
        vertices, edge_attr = self.encode(world_coords, prev_world_coords,
                                          edge_index)
        L = 10
        for i in range(L):
            vertices, edge_attr, _ = self.op(vertices, edge_index, edge_attr)
        x = self.decode(vertices)
        return x


def build_network(arch_args):
    # Input dim controls the actual vector shape accepted by the network
    # ndim is the number of positional dimensions for the system
    # Input dim = ndim + extra features (certainly at least one particle mass)
    input_dim = arch_args["input_dim"]
    ndim = arch_args["ndim"]
    hidden_dim = arch_args["hidden_dim"]
    assert input_dim >= 2 * ndim + 1

    net = GN(n_f=input_dim, ndim=ndim)
    return net
