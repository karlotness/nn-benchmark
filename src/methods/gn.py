import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn import MetaLayer
from torch.autograd import grad
from torch_geometric.data import data
from torch_scatter import scatter_sum
import numpy as np
from collections import namedtuple


TimeDerivative = namedtuple("TimeDerivative", ["dq_dt", "dp_dt"])


def package_batch(p, q, dp_dt, dq_dt, masses, edge_index, boundary_vertices):
    vertices = torch.linspace(0, 1, step=p.shape[-1])
    x = torch.from_numpy(np.concatenate((vertices, q), axis=-1))

    if boundary_vertices is not None:
        p = torch.nn.functional.pad(p, (1, 1), "constant", 0)
        boundary_vertices = torch.unsqueeze(boundary_vertices, 0).repeat_interleave(x.shape[0], dim=0)
        x = torch.concatenate((boundary_vertices[:, 0, :], x, boundary_vertices[:, 1, :]), axis=-2)

    # convert momenta to velocities
    print(masses)
    masses = np.ones_like(p)
    v = p / masses
    v = torch.from_numpy(np.concatenate((torch.zeros_like(v), v), axis=-1))
    ret = data.Data(x=v, edge_index=edge_index, pos=x)
    return ret


def unpackage_time_derivative(input_data, deriv):
    # format: deriv = torch.cat((dq_dt, dv_dt), dim=1)
    masses = input_data.x[..., -1]
    dq_dt, dv_dt = np.split(deriv, 2, axis=1)
    dp_dt = dv_dt * masses
    return TimeDerivative(dq_dt=dq_dt.reshape((1, -1)),
                          dp_dt=dp_dt.reshape((1, -1)))


class EdgeModel(torch.nn.Module):
    def __init__(self, hidden=128):
        super(EdgeModel, self).__init__()
        self.edge_mlp = Seq(Lin(3 * hidden, hidden),
                            ReLU(), Lin(hidden, hidden))

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr], dim=1)
        out = self.edge_mlp(out)
        out += edge_attr
        return out


class NodeModel(torch.nn.Module):
    def __init__(self, hidden=128):
        super(NodeModel, self).__init__()
        # self.node_mlp_1 = Residual(Seq(Lin(..., hidden),
        #                                ReLU(), Lin(hidden, ...)))
        self.node_mlp_2 = Seq(Lin(2 * hidden, hidden),
                              ReLU(), Lin(hidden, hidden))

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
        edge_attr_sum = scatter_sum(edge_attr, row, dim=0, dim_size=x.size(0))
        out = torch.cat([x, edge_attr_sum], dim=1)
        out = self.node_mlp_2(out)
        out += x
        return out


class GN:
    def __init__(self, v_features, e_features, mesh_coords,
                 static_nodes, constant_vertex_features=None, hidden=128):
        super().__init__()

        self.mesh_coords = mesh_coords
        self.static_nodes = torch.nn.functional.one_hot(static_nodes,
                                                        2).float()
        self.constant_vertex_features = constant_vertex_features

        self.encode_mlps = {"vertex" : Seq(Lin(v_features, hidden), ReLU(), Lin(hidden, hidden)),
            "edge" : Seq(Lin(e_features, hidden), ReLU(), Lin(hidden, hidden))}
        self.process = MetaLayer(
            EdgeModel(hidden),
            NodeModel(hidden),
            None)
        self.decode = Seq(Lin(hidden, hidden), ReLU(),
                          Lin(hidden, v_features))

    def encode(self, world_coords, vertex_features, edge_index):
        vertices = torch.cat([self.static_nodes,
                              vertex_features], dim=-1)
        if self.constant_vertex_features is not None:
            vertices = torch.cat([vertices, self.constant_vertex_features],
                                 dim=-1)

        row, col = edge_index
        edge_vectors = world_coords[col] - world_coords[row]
        edge_vectors_norm = torch.norm(edge_vectors, dim=-1, keepdim=True)
        edge_attr = torch.cat([edge_vectors, edge_vectors_norm], dim=-1)
        if self.mesh_coords is not None:
            mesh_vectors = self.mesh_coords[col] - self.mesh_coords[row]
            mesh_vectors_norm = torch.norm(mesh_vectors, dim=-1, keepdim=True)
            edge_attr = torch.cat([edge_attr, mesh_vectors, mesh_vectors_norm],
                                  dim=-1)

        vertices = self.encode_mlps["vertex"].forward(vertices)
        edge_attr = self.encode_mlps["edge"].forward(edge_attr)
        return vertices, edge_attr

    def forward(self, world_coords, vertex_features, edge_index):
        # x is [n, n_f]
        vertices, edge_attr = self.encode(world_coords, vertex_features,
                                          edge_index)
        L = 10
        for i in range(L):
            print(i)
            vertices, edge_attr, _ = self.process(vertices, edge_index,
                                                  edge_attr)
        x = self.decode(vertices)
        return x

    def loss(self, pred_batch, true_batch):
        return


def build_network(arch_args):
    # Input dim controls the actual vector shape accepted by the network
    # ndim is the number of positional dimensions for the system
    # Input dim = ndim + extra features (certainly at least one particle mass)
    v_features = arch_args["v_features"]
    e_features = arch_args["e_features"]
    hidden_dim = arch_args["hidden_dim"]
    mesh_coords = torch.tensor(arch_args["mesh_coords"])
    static_nodes = torch.tensor(arch_args["static_nodes"], dtype=torch.int)

    net = GN(v_features=v_features, e_features=e_features,
             hidden_dim=hidden_dim, mesh_coords=mesh_coords,
             static_nodes=static_nodes)
    return net
