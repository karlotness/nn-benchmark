import numpy as np
from scipy.linalg import lu_factor, lu_solve
from .defs import System, TrajectoryResult, SystemResult, StatePair, SystemCache
from collections import namedtuple
import logging
import time
from numba import jit


taylor_green_cache = SystemCache()

Edge = namedtuple("Edge", ["a", "b"])

MeshTrajectoryResult = namedtuple("MeshTrajectoryResult",
                                      ["q", "p",
                                       "dq_dt", "dp_dt",
                                       "t_steps",
                                       "p_noiseless", "q_noiseless",
                                       "masses", "edge_indices", "vertices"])

class TaylorGreenSystem(System):
    def __init__(self, n_grid, vertices, edges, space_scale=2, viscosity=1, density=1):
        super().__init__()
        self.n_grid = n_grid
        self.space_scale = 2
        self.viscosity = viscosity
        self.density = density
        space_steps = np.linspace(0, self.space_scale * np.pi, self.n_grid)

        self.edges = edges
        self.edge_indices = np.array([(e.a, e.b) for e in edges] +
                                     [(e.b, e.a) for e in edges], dtype=np.int64).T
        self.edge_indices.setflags(write=False)
        self.vertices = vertices

        self.x, self.y = np.meshgrid(space_steps, space_steps)

    def derivative(self, q, p, t):
        n_steps = q.shape[0]
        df_t = -2 * self.viscosity * np.exp(-2 * self.viscosity * t).reshape((n_steps, 1, 1))
        # Velocity derivative
        u = np.cos(self.x) * np.sin(self.y) * df_t
        v = -1 * np.sin(self.x) * np.cos(self.y) * df_t
        vel = np.stack((u, v), axis=-1)
        # Pressure derivative
        pr_df_t = self.viscosity * np.exp(-4 * self.viscosity * t).reshape((n_steps, 1, 1))
        press = self.density * (np.cos(2 * self.x) + np.cos(2 * self.y)) * pr_df_t
        return vel, press

    def vel(self, t):
        n_steps = t.shape[0]
        f_t = np.exp(-2 * self.viscosity * t).reshape((n_steps, 1, 1))
        u = np.cos(self.x) * np.sin(self.y) * f_t
        v = -1 * np.sin(self.x) * np.cos(self.y) * f_t
        return np.stack((u, v), axis=-1)

    def pressure(self, t):
        n_steps = t.shape[0]
        f_t = np.exp(-2 * self.viscosity * t).reshape((n_steps, 1, 1))
        press = (-self.density / 4) * (np.cos(2 * self.x) + np.cos(2 * self.y)) * (f_t ** 2)
        return press

    def generate_trajectory(self, num_time_steps, time_step_size):
        t = np.arange(num_time_steps) * time_step_size
        vels = self.vel(t).reshape((num_time_steps, self.n_grid ** 2, 2))
        press = self.pressure(t).reshape((num_time_steps, self.n_grid ** 2, 1))
        d_vel, d_press = self.derivative(vels, press, t)
        d_vel = d_vel.reshape((num_time_steps, self.n_grid ** 2, 2))
        d_press = d_press.reshape((num_time_steps, self.n_grid ** 2, 1))
        return MeshTrajectoryResult(q=press, p=vels, dq_dt=d_press, dp_dt=d_vel,
                                    t_steps=t,
                                    q_noiseless=press, p_noiseless=vels,
                                    edge_indices=self.edge_indices,
                                    vertices=self.vertices,
                                    masses=np.ones_like(self.vertices))

    def hamiltonian(self, q, p):
        return np.zeros(q.shape[0], dtype=q.dtype)

    def _args_compatible(self, n_grid, vertices, edges, space_scale, viscosity, density):
        return (self.n_grid == n_grid and
                set(self.edges) == set(edges) and
                self.space_scale == space_scale and
                self.viscosity == viscosity and
                self.density == density)

def system_from_records(n_grid, space_scale, viscosity, density, vertices, edges):
    edges = [Edge(a=e["a"], b=e["b"]) for e in edges]
    cached_sys = taylor_green_cache.find(n_grid=n_grid,
                                         space_scale=space_scale,
                                         viscosity=viscosity,
                                         density=density,
                                         vertices=vertices,
                                         edges=edges)
    if cached_sys is not None:
        return cached_sys
    else:
        new_sys = TaylorGreenSystem(n_grid=n_grid,
                                    vertices=vertices,
                                    edges=edges,
                                    space_scale=space_scale,
                                    viscosity=viscosity,
                                    density=density)
        taylor_green_cache.insert(new_sys)
        return new_sys


def generate_data(system_args, base_logger=None):
    if base_logger:
        logger = base_logger.getChild("taylor-green")
    else:
        logger = logging.getLogger("taylor-green")

    # Global system parameters
    n_grid = system_args["n_grid"]

    trajectory_metadata = []
    trajectories = {}
    trajectory_defs = system_args["trajectory_defs"]
    for i, traj_def in enumerate(trajectory_defs):
        traj_name = f"traj_{i:05}"
        logger.info(f"Generating trajectory {traj_name}")

        # Generate starting conditions
        viscosity=traj_def["viscosity"]
        space_scale=int(traj_def.get("space_scale", 2))
        density=traj_def.get("density", 1.0)
        num_time_steps = traj_def["num_time_steps"]
        time_step_size = traj_def["time_step_size"]
        noise_sigma = 0.0
        vertices = traj_def["vertices"]
        edges_dict = traj_def["edges"]
        edges = [Edge(a=e["a"], b=e["b"]) for e in edges_dict]


        system = taylor_green_cache.find(n_grid=n_grid,
                                         vertices=vertices,
                                         edges=edges,
                                         space_scale=space_scale,
                                         viscosity=viscosity,
                                         density=density)
        if system is None:
            system = TaylorGreenSystem(n_grid=n_grid,
                                       vertices=vertices,
                                       edges=edges,
                                       space_scale=space_scale,
                                       viscosity=viscosity,
                                       density=density)
            taylor_green_cache.insert(system)

        traj_gen_start = time.perf_counter()
        traj_result = system.generate_trajectory(num_time_steps=num_time_steps,
                                                 time_step_size=time_step_size)
        traj_gen_elapsed = time.perf_counter() - traj_gen_start
        logger.info(f"Generated {traj_name} in {traj_gen_elapsed} sec")

        # Store trajectory data
        trajectories.update({
            f"{traj_name}_p": traj_result.p,
            f"{traj_name}_q": traj_result.q,
            f"{traj_name}_dqdt": traj_result.dq_dt,
            f"{traj_name}_dpdt": traj_result.dp_dt,
            f"{traj_name}_t": traj_result.t_steps,
            f"{traj_name}_p_noiseless": traj_result.p_noiseless,
            f"{traj_name}_q_noiseless": traj_result.q_noiseless,
            f"{traj_name}_edge_indices": traj_result.edge_indices,
            f"{traj_name}_vertices": traj_result.vertices,
        })

        # Store per-trajectory metadata
        trajectory_metadata.append(
            {"name": traj_name,
             "num_time_steps": num_time_steps,
             "time_step_size": time_step_size,
             "viscosity": viscosity,
             "space_scale": space_scale,
             "density": density,
             "noise_sigma": noise_sigma,
             "field_keys": {
                 "p": f"{traj_name}_p",
                 "q": f"{traj_name}_q",
                 "dpdt": f"{traj_name}_dpdt",
                 "dqdt": f"{traj_name}_dqdt",
                 "t": f"{traj_name}_t",
                 "p_noiseless": f"{traj_name}_p_noiseless",
                 "q_noiseless": f"{traj_name}_q_noiseless",
                 "edge_indices": f"{traj_name}_edge_indices",
                 "vertices": f"{traj_name}_vertices",
             },
             "timing": {
                 "traj_gen_time": traj_gen_elapsed
             }})

    logger.info("Done generating trajectories")

    return SystemResult(trajectories=trajectories,
                        metadata={
                            "n_grid": n_grid,
                            "space_scale": space_scale,
                            "viscosity": viscosity,
                            "density": density,
                            "vertices": vertices,
                            "edges": edges_dict,
                        },
                        trajectory_metadata=trajectory_metadata)
