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
    vertices = torch.unsqueeze(torch.linspace(0, 1, p.shape[1]), 0).repeat(p.shape[0], p.shape[1])
    x = torch.from_numpy(np.concatenate((vertices, q), axis=-1))

    if boundary_vertices is not None:
        p = np.pad(p, ((1, 1), (1, 0)), "constant", constant_values=0)
        boundary_vertices = torch.unsqueeze(torch.tensor(boundary_vertices), 0).repeat_interleave(x.shape[0], dim=0)
        x = torch.cat((boundary_vertices[:, 0, :], x, boundary_vertices[:, 1, :]), axis=-2)

    # convert momenta to velocities
    masses = np.ones_like(p)
    v = p / masses
    v = torch.from_numpy(v)

    # Training or eval.
    if dp_dt is not None:
        masses = np.ones_like(dp_dt)
        acceleration = torch.from_numpy(dp_dt / masses)
    else:
        acceleration = dp_dt

    # v = torch.from_numpy(np.concatenate((np.zeros_like(v), v), axis=-1))
    ret = data.Data(x=v, edge_index=edge_index, pos=x, y=acceleration)
    return ret


def unpack_results(result, package_args):
    return result


def index(input_tensor, index_tensor):
    return torch.gather(input_tensor, 1, index_tensor.permute(0, 2, 1).repeat(1, 1, input_tensor.shape[-1]))


class EdgeModel(torch.nn.Module):
    def __init__(self, hidden=128):
        super(EdgeModel, self).__init__()
        self.edge_mlp = Seq(Lin(3 * hidden, hidden),
                            ReLU(), Lin(hidden, hidden))
        self.layer_norm = torch.nn.LayerNorm(hidden)

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr], dim=-1)
        out = self.edge_mlp(out)
        out += edge_attr

        out = self.layer_norm(out)

        return out


class NodeModel(torch.nn.Module):
    def __init__(self, hidden=128):
        super(NodeModel, self).__init__()
        # self.node_mlp_1 = Residual(Seq(Lin(..., hidden),
        #                                ReLU(), Lin(hidden, ...)))
        self.node_mlp_2 = Seq(Lin(2 * hidden, hidden),
                              ReLU(), Lin(hidden, hidden))
        self.hidden = hidden
        self.layer_norm = torch.nn.LayerNorm(hidden)

    def forward(self, x, edge_index, edge_attr, u=None, batch=None):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = torch.split(edge_index, [1, 1], dim=1)

        # Default
        # out = torch.cat([x[row], edge_attr], dim=1)
        # out = self.node_mlp_1(out)
        # out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        # out = torch.cat([x, out, u[batch]], dim=1)

        # Custom
        # edge_attr_sum = scatter_sum(edge_attr, row, dim=0, dim_size=x.size(0))
        edge_attr_sum = scatter_sum(edge_attr, row.permute(0, 2, 1).repeat(1, 1, self.hidden), dim=1, dim_size=x.size(1))

        out = torch.cat([x, edge_attr_sum], dim=-1)
        out = self.node_mlp_2(out)
        out += x

        out = self.layer_norm(out)

        return out


class CustomMetaLayer(torch.nn.Module):
    def __init__(self, edge_model, node_model, global_model):
        super(CustomMetaLayer, self).__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()


    def forward(self, x, edge_index, edge_attr=None):
        """"""
        row, col = torch.split(edge_index, [1, 1], dim=1)

        edge_attr = self.edge_model(index(x, row), index(x, col), edge_attr)

        x = self.node_model(x, edge_index, edge_attr)

        return x, edge_attr, None


    def __repr__(self):
        return ('{}(\n'
                '    edge_model={},\n'
                '    node_model={},\n'
                ')').format(self.__class__.__name__, self.edge_model,
                            self.node_model)


class GN(torch.nn.Module):
    def __init__(self, v_features, e_features, mesh_coords,
                 static_nodes, constant_vertex_features=None, hidden=128):
        super().__init__()

        self.mesh_coords = mesh_coords
        self.static_nodes = torch.nn.functional.one_hot(static_nodes,
                                                        2).float()
        self.constant_vertex_features = constant_vertex_features

        self.vertex_encode_mlp = Seq(Lin(v_features, hidden), ReLU(), Lin(hidden, hidden))
        self.edge_encode_mlp = Seq(Lin(e_features, hidden), ReLU(), Lin(hidden, hidden))
        self.process = CustomMetaLayer(
            EdgeModel(hidden),
            NodeModel(hidden),
            None)
        self.decode = Seq(Lin(hidden, hidden), ReLU(), Lin(hidden, v_features - self.static_nodes.shape[-1]))
        self.layer_norm = torch.nn.LayerNorm(hidden)

    def _apply(self, fn):
        super()._apply(fn)

        self.mesh_coords = fn(self.mesh_coords)
        self.static_nodes = fn(self.static_nodes)
        if self.constant_vertex_features is not None:
            self.constant_vertex_features = fn(self.constant_vertex_features)

        return self

    def encode(self, world_coords, vertex_features, edge_index):
        static_nodes_batch = torch.unsqueeze(self.static_nodes, 0).repeat(world_coords.shape[0], 1, 1)
        vertices = torch.cat([static_nodes_batch,
                              vertex_features], dim=-1)
        if self.constant_vertex_features is not None:
            vertices = torch.cat([vertices, self.constant_vertex_features],
                                 dim=-1)

        row, col = torch.split(edge_index, [1, 1], dim=1)
        edge_vectors = index(world_coords, col) - index(world_coords, row)
        edge_vectors_norm = torch.norm(edge_vectors, dim=-1, keepdim=True)
        edge_attr = torch.cat([edge_vectors, edge_vectors_norm], dim=-1)
        if self.mesh_coords is not None:
            mesh_coords_batch = torch.unsqueeze(self.mesh_coords, 0).repeat(world_coords.shape[0], 1, 1)
            mesh_vectors = index(mesh_coords_batch, col) - index(mesh_coords_batch, row)
            mesh_vectors_norm = torch.norm(mesh_vectors, dim=-1, keepdim=True)
            edge_attr = torch.cat([edge_attr, mesh_vectors, mesh_vectors_norm],
                                  dim=-1)

        # edge_attr = torch.squeeze(edge_attr)

        vertices = self.vertex_encode_mlp.forward(vertices)
        vertices = self.layer_norm(vertices)
        edge_attr = self.edge_encode_mlp.forward(edge_attr)
        edge_attr = self.layer_norm(edge_attr)

        return vertices, edge_attr

    def forward(self, world_coords, vertex_features, edge_index):
        # x is [n, n_f]
        vertices, edge_attr = self.encode(world_coords, vertex_features,
                                          edge_index)
        L = 15
        for i in range(L):
            vertices, edge_attr, _ = self.process(vertices, edge_index,
                                                  edge_attr)
        x = self.decode(vertices)

        return x


def build_network(arch_args):
    # Input dim controls the actual vector shape accepted by the network
    # ndim is the number of positional dimensions for the system
    # Input dim = ndim + extra features (certainly at least one particle mass)
    v_features = arch_args["v_features"]
    e_features = arch_args["e_features"]
    hidden_dim = arch_args["hidden_dim"]
    mesh_coords = torch.tensor(arch_args["mesh_coords"])
    static_nodes = torch.tensor(arch_args["static_nodes"]).long()

    net = GN(v_features=v_features, e_features=e_features,
             hidden=hidden_dim, mesh_coords=mesh_coords,
             static_nodes=static_nodes)
    return net
