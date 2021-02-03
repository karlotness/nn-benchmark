import numpy as np
from scipy.linalg import lu_factor, lu_solve
from .defs import System, TrajectoryResult, SystemResult, StatePair
from collections import namedtuple
import logging
import time
import torch
from scipy.sparse import coo_matrix


Particle = namedtuple("Particle", ["mass", "is_fixed"])
# Particle numbers are zero indexed
Edge = namedtuple("Edge", ["a", "b", "spring_const", "rest_length"])

ParticleTrajectoryResult = namedtuple("ParticleTrajectoryResult",
                                      ["q", "p",
                                       "dq_dt", "dp_dt",
                                       "t_steps",
                                       "p_noiseless", "q_noiseless",
                                       "masses", "edge_indices", "fixed_mask"])


class SpringMeshSystem(System):
    def __init__(self, n_dims, particles, edges, vel_decay):
        super().__init__()
        self.particles = particles
        self.edges = edges
        self.n_dims = n_dims
        assert self.n_dims == 2
        self.n_particles = len(particles)
        self.vel_decay = vel_decay
        self.masses = np.array([p.mass for p in self.particles], dtype=np.float64)
        self.fixed_mask = np.array([p.is_fixed for p in self.particles], dtype=np.bool)
        # Gather other data
        self.edge_indices = np.array([(e.a, e.b) for e in self.edges] +
                                     [(e.b, e.a) for e in self.edges], dtype=np.int64).T
        self.spring_consts = np.array([e.spring_const for e in self.edges] +
                                      [e.spring_const for e in self.edges], dtype=np.float64)
        self.rest_lengths = np.array([e.rest_length for e in self.edges] +
                                     [e.rest_length for e in self.edges], dtype=np.float64)
        # Compute the update matrices
        mass_matrix = np.diag(np.tile(np.expand_dims(self.masses, 1), (1, self.n_dims)).reshape((-1,)))
        assert mass_matrix.shape == (self.n_particles * self.n_dims, self.n_particles * self.n_dims)
        # Build the stiffness matrix
        stiff_mat_parts = [[np.zeros((self.n_dims, self.n_dims)) for _ in range(self.n_particles)] for _ in range(self.n_particles)]
        for edge in self.edges:
            m_aa = np.diag([edge.spring_const] * self.n_dims)
            m_ab = np.diag([-1 * edge.spring_const] * self.n_dims)
            m_ba = m_ab
            m_bb = m_aa
            stiff_mat_parts[edge.a][edge.a] += m_aa
            stiff_mat_parts[edge.a][edge.b] += m_ab
            stiff_mat_parts[edge.b][edge.a] += m_ba
            stiff_mat_parts[edge.b][edge.b] += m_bb
        stiff_mat = np.block(stiff_mat_parts)
        assert stiff_mat.shape == (self.n_particles * self.n_dims, self.n_particles * self.n_dims)
        # Compute the selection matrix for non-fixed vertices
        n_non_fixed = self.n_particles - np.count_nonzero(self.fixed_mask)
        unfixed_mask_parts = [[np.zeros((self.n_dims, self.n_dims), dtype=np.bool) for _ in range(self.n_particles)] for _ in range(n_non_fixed)]
        for i, j in enumerate(np.nonzero(np.logical_not(self.fixed_mask))[0]):
            unfixed_mask_parts[i][j] = np.eye(self.n_dims, dtype=np.bool)
        unfixed_mask_mat = np.block(unfixed_mask_parts)
        assert unfixed_mask_mat.shape == (n_non_fixed * self.n_dims, self.n_particles * self.n_dims)
        # Store system matrices
        self._mass_matrix = mass_matrix
        self._select_matrix = unfixed_mask_mat
        self._stiff_mat = stiff_mat

    def hamiltonian(self, q, p):
        return torch.zeros(q.shape[0], q.shape[1])

    def compute_forces(self, q):
        assert q.shape[0] == 1
        q = q.reshape((self.n_particles, self.n_dims))
        # Compute length of each spring and "diff" directions of the forces
        diffs = q[self.edge_indices[0], :] - q[self.edge_indices[1], :]
        lengths = np.linalg.norm(diffs, ord=2, axis=-1)
        # Compute forces
        spring_consts = self.spring_consts
        rest_lengths = self.rest_lengths
        edge_forces = np.expand_dims(-1 * spring_consts * (lengths - rest_lengths) / lengths, axis=-1) * diffs
        # Gather forces for each of their "lead" particles
        # Stitch together lists of coordinates
        row_coords = np.concatenate([self.edge_indices[0], self.edge_indices[0]])
        col_coords = np.concatenate([np.repeat(0, edge_forces.shape[0]),
                                     np.repeat(1, edge_forces.shape[0])])
        data = np.concatenate([edge_forces[:, 0], edge_forces[:, 1]])
        forces_coo = coo_matrix((data, (row_coords, col_coords)), shape=(self.n_particles, self.n_dims))
        # Mask forces on fixed particles
        forces = forces_coo.todense()
        forces[self.fixed_mask, :] = 0
        return np.expand_dims(forces, axis=0)

    def derivative(self, q, p, dt=1):
        step_vel_decay = self.vel_decay ** dt
        orig_q_shape = q.shape
        orig_p_shape = p.shape
        q = q.reshape((-1, self.n_particles, self.n_dims))
        p = p.reshape((-1, self.n_particles, self.n_dims))
        # Compute action of forces on each particle
        forces = self.compute_forces(q=q)
        # Update positions
        masses = np.expand_dims(self.masses, axis=(0, -1))
        pos = (1 / masses) * p
        pos[:, self.fixed_mask, :] = 0
        q_out = (step_vel_decay * pos).reshape(orig_q_shape)
        p_out = forces.reshape(orig_p_shape)
        return StatePair(q=q_out, p=p_out)

    def _compute_next_step(self, q, q_dot, time_step_size, mat_unknown_factors, step_vel_decay=1.0):
        # Input states are (n_particle, n_dim)
        forces_orig = self.compute_forces(q=q)[0]
        forces = forces_orig.reshape((-1,))
        q = q.reshape((-1, ))
        q_dot = q_dot.reshape((-1, ))
        known = self._select_matrix @ (self._mass_matrix @ q_dot) + (time_step_size * (self._select_matrix @ forces))
        # Two of the values to return
        q_dot_hat_next = step_vel_decay * lu_solve(mat_unknown_factors, known)
        q_next = q + time_step_size * (self._select_matrix.T @ q_dot_hat_next)
        # Reshape
        q_dot_next = (self._select_matrix.T @ q_dot_hat_next).reshape((self.n_particles, self.n_dims))
        q_next = q_next.reshape((self.n_particles, self.n_dims))
        # Get the p values to return
        p = np.zeros_like(q_dot, shape=(self.n_particles, self.n_dims))
        for i, part in enumerate(self.particles):
            if part.is_fixed:
                continue
            p[i] = part.mass * q_dot_next[i]
        return q_next, q_dot_next, p, forces_orig

    def generate_trajectory(self, q0, p0, num_time_steps, time_step_size,
                            subsample=1, noise_sigma=0.0):
        # Check shapes of inputs
        if (q0.shape != (self.n_particles, self.n_dims)) or (p0.shape != (self.n_particles, self.n_dims)):
            raise ValueError("Invalid input shape for particle system")

        t_eval = np.arange(num_time_steps) * time_step_size

        # Process arguments for subsampling
        num_steps = num_time_steps * subsample
        orig_time_step_size = time_step_size
        time_step_size = time_step_size / subsample
        step_vel_decay = self.vel_decay ** time_step_size

        # Compute updates using explicit Euler
        # compute update matrices
        mat_unknown = self._select_matrix @ (self._mass_matrix - (time_step_size ** 2) * self._stiff_mat) @ self._select_matrix.T
        mat_unknown_factors = lu_factor(mat_unknown)

        init_vel = np.zeros_like(q0)
        for i, part in enumerate(self.particles):
            init_vel[i] = (1/part.mass) * p0[i]

        qs = [q0]
        q_dots = [init_vel]
        ps = [p0]
        p_dots = [self.compute_forces(q=q0)[0]]
        q = q0.copy()
        q_dot = p0.copy()

        for i, part in enumerate(self.particles):
            q_dot[i] /= part.mass
        for step_idx in range(1, num_steps):
            q, q_dot, p, _p_dot_next = self._compute_next_step(q=q, q_dot=q_dot, time_step_size=time_step_size,
                                                               mat_unknown_factors=mat_unknown_factors,
                                                               step_vel_decay=step_vel_decay)
            if step_idx % subsample == 0:
                p_dot = self.compute_forces(q=q)[0]
                qs.append(q)
                q_dots.append(q_dot)
                ps.append(p)
                p_dots.append(p_dot)

        qs = np.stack(qs).reshape(num_time_steps, self.n_particles, self.n_dims)
        ps = np.stack(ps).reshape(num_time_steps, self.n_particles, self.n_dims)
        dq_dt = np.stack(q_dots).reshape(num_time_steps, self.n_particles, self.n_dims)
        dp_dt = np.stack(p_dots).reshape(num_time_steps, self.n_particles, self.n_dims)

        # Add configured noise
        noise_ps = noise_sigma * np.random.randn(*ps.shape)
        noise_qs = noise_sigma * np.random.randn(*qs.shape)

        qs_noisy = qs + noise_qs
        ps_noisy = ps + noise_ps

        # Gather other data
        edge_indices = np.array([(e.a, e.b) for e in self.edges] +
                                [(e.b, e.a) for e in self.edges], dtype=np.int64).T

        return ParticleTrajectoryResult(
            q=qs_noisy,
            p=ps_noisy,
            dq_dt=dq_dt,
            dp_dt=dp_dt,
            t_steps=t_eval,
            q_noiseless=qs,
            p_noiseless=ps,
            masses=self.masses,
            edge_indices=edge_indices,
            fixed_mask=self.fixed_mask)


def system_from_records(n_dims, particles, edges, vel_decay):
    parts = []
    edgs = []
    for pdef in particles:
        parts.append(
            Particle(mass=pdef["mass"],
                     is_fixed=pdef["is_fixed"]))
    for edef in edges:
        edgs.append(
            Edge(a=edef["a"],
                 b=edef["b"],
                 spring_const=edef["spring_const"],
                 rest_length=edef["rest_length"]))
    return SpringMeshSystem(n_dims=n_dims,
                            particles=parts,
                            edges=edgs,
                            vel_decay=vel_decay)


def generate_data(system_args, base_logger=None):
    if base_logger:
        logger = base_logger.getChild("spring-mesh")
    else:
        logger = logging.getLogger("spring-mesh")

    trajectory_metadata = []
    trajectories = {}
    vel_decay = system_args.get("vel_decay", 1.0)
    trajectory_defs = system_args["trajectory_defs"]
    for i, traj_def in enumerate(trajectory_defs):
        traj_name = f"traj_{i:05}"
        logger.info(f"Generating trajectory {traj_name}")

        # Create the trajectory
        particle_defs = traj_def["particles"]
        spring_defs = traj_def["springs"]
        num_time_steps = traj_def["num_time_steps"]
        time_step_size = traj_def["time_step_size"]
        noise_sigma = traj_def.get("noise_sigma", 0.0)
        subsample = int(traj_def.get("subsample", 1))

        # Split particles and springs into components
        q0 = []
        particles = []
        edges = []
        for pdef in particle_defs:
            particles.append(
                Particle(mass=pdef["mass"],
                         is_fixed=pdef["is_fixed"]))
            q0.append(np.array(pdef["position"]))
        for edef in spring_defs:
            edges.append(
                Edge(a=edef["a"],
                     b=edef["b"],
                     spring_const=edef["spring_const"],
                     rest_length=edef["rest_length"]))
        q0 = np.stack(q0).astype(np.float64)
        p0 = np.zeros_like(q0)

        n_dims = q0.shape[-1]
        n_particles = len(particle_defs)
        system = SpringMeshSystem(n_dims=n_dims, particles=particles,
                                  edges=edges, vel_decay=vel_decay)

        traj_gen_start = time.perf_counter()
        traj_result = system.generate_trajectory(q0=q0,
                                                 p0=p0,
                                                 num_time_steps=num_time_steps,
                                                 time_step_size=time_step_size,
                                                 subsample=subsample,
                                                 noise_sigma=noise_sigma)
        traj_gen_elapsed = time.perf_counter() - traj_gen_start
        logger.info(f"Generating {traj_name} in {traj_gen_elapsed} sec")

        # Store trajectory data
        trajectories.update({
            f"{traj_name}_p": traj_result.p,
            f"{traj_name}_q": traj_result.q,
            f"{traj_name}_dqdt": traj_result.dq_dt,
            f"{traj_name}_dpdt": traj_result.dp_dt,
            f"{traj_name}_t": traj_result.t_steps,
            f"{traj_name}_p_noiseless": traj_result.p_noiseless,
            f"{traj_name}_q_noiseless": traj_result.q_noiseless,
            f"{traj_name}_masses": traj_result.masses,
            f"{traj_name}_edge_indices": traj_result.edge_indices,
            f"{traj_name}_fixed_mask": traj_result.fixed_mask,
        })

        # Store per-trajectory metadata
        trajectory_metadata.append(
            {"name": traj_name,
             "num_time_steps": num_time_steps,
             "time_step_size": time_step_size,
             "noise_sigma": noise_sigma,
             "field_keys": {
                 "p": f"{traj_name}_p",
                 "q": f"{traj_name}_q",
                 "dpdt": f"{traj_name}_dpdt",
                 "dqdt": f"{traj_name}_dqdt",
                 "t": f"{traj_name}_t",
                 "p_noiseless": f"{traj_name}_p_noiseless",
                 "q_noiseless": f"{traj_name}_q_noiseless",
                 "masses": f"{traj_name}_masses",
                 "edge_indices": f"{traj_name}_edge_indices",
                 "fixed_mask": f"{traj_name}_fixed_mask",
             },
             "timing": {
                 "traj_gen_time": traj_gen_elapsed
             }})

    logger.info("Done generating trajectories")

    particle_records = []
    edge_records = []
    for part in trajectory_defs[0]["particles"]:
        particle_records.append({
            "mass": part["mass"],
            "is_fixed": part["is_fixed"],
        })
    for edge in trajectory_defs[0]["springs"]:
        edge_records.append({
            "a": edge["a"],
            "b": edge["b"],
            "spring_const": edge["spring_const"],
            "rest_length": edge["rest_length"],
        })

    return SystemResult(trajectories=trajectories,
                        metadata={
                            "n_grid": n_dims,
                            "n_dim": n_dims,
                            "n_particles": n_particles,
                            "system_type": "spring-mesh",
                            "particles": particle_records,
                            "edges": edge_records,
                            "vel_decay": vel_decay,
                        },
                        trajectory_metadata=trajectory_metadata)
