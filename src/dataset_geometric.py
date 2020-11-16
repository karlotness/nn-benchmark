from collections import namedtuple
import numpy as np
import torch
from scipy.linalg import circulant
from methods import hogn


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


def package_data(data_set, package_args):
    particle_process_type = package_args["particle_processing"]
    package_type = package_args["package_type"]
    adjacency_args = package_args["adjacency_args"]
    edge_index = get_edge_index(adjacency_args)

    if package_type == "hogn":
        package_func = hogn.package_batch
    else:
        raise ValueError(f"Unknown package type {package_type}")

    if particle_process_type == "one-dim":
        particle_process_func = particle_type_one_dim
    else:
        raise ValueError(f"Unknown particle processing type {particle_process_type}")

    data_elems = []

    for batch in data_set:
        p = batch.p
        q = batch.q
        dp_dt = batch.dp_dt
        dq_dt = batch.dq_dt
        if hasattr(batch, "masses") and batch["masses"] is not None:
            masses = batch.masses
        else:
            masses = np.ones(p.shape[1])
        proc_part = particle_process_func(p=p, q=q,
                                          dp_dt=dp_dt, dq_dt=dq_dt,
                                          masses=masses)
        packaged = package_func(p=proc_part.p, q=proc_part.q,
                                dp_dt=proc_part.dp_dt, dq_dt=proc_part.dq_dt,
                                masses=proc_part.masses, edge_index=edge_index)
        data_elems.append(packaged)

    return data_elems
