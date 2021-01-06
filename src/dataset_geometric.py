from collections import namedtuple
import numpy as np
import torch
from scipy.linalg import circulant
from methods import hogn, gn


ProcessedParticles = namedtuple("ProcessedParticles", ["p", "q", "dp_dt", "dq_dt", "masses"])


def get_edge_index(connection_args):
    conn_type = connection_args["type"]
    if conn_type == "fully-connected":
        # Connect all vertices to each other
        # No self loop
        dim = connection_args["dimension"]
        a = np.ones((dim, dim), dtype=np.int8)
        b = np.eye(dim, dtype=np.int8)
        adj = torch.from_numpy(np.array(np.where(a - b)))
        return adj
    elif conn_type == "circular-local":
        # Connect neighboring vertices to each other
        # Distance controlled by "degree" on either side
        # No self loops
        dim = connection_args["dimension"]
        degree = connection_args["degree"]
        template = np.zeros(dim, dtype=np.int8)
        template[1:degree+1] = 1
        template[-degree:] = 1
        adj = torch.from_numpy(np.array(np.where(circulant(template))))
        return adj
    elif conn_type == "regular-grid":
        dim = connection_args["dimension"]
        boundary_cond = connection_args["boundary_conditions"]
        if boundary_cond == "periodic":
            idx = torch.arange(dim)
            adj = torch.stack((
                torch.cat((torch.tensor([0., dim - 1.]), idx[1:], idx[:-1])),
                torch.cat((torch.tensor([dim - 1., 0.]), idx[:-1], idx[1:]))),
                              dim=0)
        elif boundary_cond == "fixed":
            idx = torch.arange(dim + 2)
            adj = torch.stack((
                torch.cat((idx[1:], idx[:-1])),
                torch.cat((idx[:-1], idx[1:]))), dim=0)
        return adj
    else:
        raise ValueError(f"Unknown connection type {conn_type}")


def particle_type_one_dim(p, q, dp_dt, dq_dt, masses):
    def process_elem(ev):
        # Plain vector, pivot to have one dimension
        if ev is not None:
            return ev.reshape((-1, 1))
        else:
            return None
    return ProcessedParticles(
        p=process_elem(p),
        q=process_elem(q),
        dp_dt=process_elem(dp_dt),
        dq_dt=process_elem(dq_dt),
        masses=process_elem(masses))


def particle_type_identity(p, q, dp_dt, dq_dt, masses):
    return ProcessedParticles(
        p=p,
        q=q,
        dp_dt=dp_dt,
        dq_dt=dq_dt,
        masses=masses.reshape((-1, 1)))


def package_data(data_set, package_args):
    particle_process_type = package_args["particle_processing"]
    package_type = package_args["package_type"]
    adjacency_args = package_args["adjacency_args"]
    edge_index = get_edge_index(adjacency_args)

    if package_type == "hogn":
        package_func = hogn.package_batch
        boundary_vertices = None
    elif package_type == "gn":
        package_func = gn.package_batch
        boundary_vertices = adjacency_args["boundary_vertices"]
    else:
        raise ValueError(f"Unknown package type {package_type}")

    if particle_process_type == "one-dim":
        particle_process_func = particle_type_one_dim
    elif particle_process_type == "identity":
        particle_process_func = particle_type_identity
    else:
        raise ValueError(f"Unknown particle processing type {particle_process_type}")

    data_elems = []

    for batch in data_set:
        p = batch.p
        q = batch.q
        dp_dt = batch.dp_dt
        dq_dt = batch.dq_dt
        masses = batch.masses
        proc_part = particle_process_func(p=p, q=q,
                                          dp_dt=dp_dt, dq_dt=dq_dt,
                                          masses=masses)
        packaged = package_func(p=proc_part.p, q=proc_part.q,
                                dp_dt=proc_part.dp_dt, dq_dt=proc_part.dq_dt,
                                masses=proc_part.masses, edge_index=edge_index,
                                boundary_vertices=boundary_vertices)
        data_elems.append(packaged)

    return data_elems
