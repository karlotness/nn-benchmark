import numpy as np
from scipy.linalg import lu_factor, lu_solve
from .defs import System, TrajectoryResult, SystemResult, StatePair, SystemCache
from collections import namedtuple
import logging
import time
from numba import jit


Particle = namedtuple("Particle", ["mass", "is_fixed"])
# Particle numbers are zero indexed
Edge = namedtuple("Edge", ["a", "b", "spring_const", "rest_length"])

ParticleTrajectoryResult = namedtuple("ParticleTrajectoryResult",
                                      ["q", "p",
                                       "dq_dt", "dp_dt",
                                       "t_steps",
                                       "p_noiseless", "q_noiseless",
                                       "masses", "edge_indices", "fixed_mask"])


spring_mesh_cache = SystemCache()


@jit(nopython=True)
def batch_outer(v, w):
    res = np.empty((v.shape[0], v.shape[1], w.shape[1]), dtype=v.dtype)
    for i in range(v.shape[0]):
        res[i] = np.outer(v[i], w[i])
    return res


class SpringMeshSystem(System):
    def __init__(self, n_dims, particles, edges, vel_decay):
        super().__init__()
        self.particles = particles
        self.edges = edges
        self.n_dims = n_dims
        assert self.n_dims == 2
        self.n_particles = len(particles)
        self.masses = np.array([p.mass for p in self.particles], dtype=np.float64)
        self.masses.setflags(write=False)
        self.fixed_mask = np.array([p.is_fixed for p in self.particles], dtype=np.bool)
        self.fixed_mask.setflags(write=False)
        self.viscosity_constant = vel_decay
        # Gather other data
        self.edge_indices = np.array([(e.a, e.b) for e in self.edges] +
                                     [(e.b, e.a) for e in self.edges], dtype=np.int64).T
        self.edge_indices.setflags(write=False)
        self.spring_consts = np.expand_dims(np.array([e.spring_const for e in self.edges] +
                                                     [e.spring_const for e in self.edges], dtype=np.float64),
                                            0)
        self.spring_consts.setflags(write=False)
        self.rest_lengths = np.expand_dims(np.array([e.rest_length for e in self.edges] +
                                                    [e.rest_length for e in self.edges], dtype=np.float64),
                                           0)
        self.rest_lengths.setflags(write=False)
        self.row_coords = np.concatenate([self.edge_indices[0], self.edge_indices[0]])
        self.col_coords = np.concatenate([np.repeat(0, 2 * len(edges)),
                                          np.repeat(1, 2 * len(edges))])
        # Compute the update matrices
        mass_matrix = np.diag(np.tile(np.expand_dims(self.masses, 1), (1, self.n_dims)).reshape((-1,)))
        mass_matrix.setflags(write=False)
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
        stiff_mat.setflags(write=False)
        assert stiff_mat.shape == (self.n_particles * self.n_dims, self.n_particles * self.n_dims)
        # Compute the selection matrix for non-fixed vertices
        n_non_fixed = self.n_particles - np.count_nonzero(self.fixed_mask)
        unfixed_mask_parts = [[np.zeros((self.n_dims, self.n_dims), dtype=np.bool) for _ in range(self.n_particles)] for _ in range(n_non_fixed)]
        for i, j in enumerate(np.nonzero(np.logical_not(self.fixed_mask))[0]):
            unfixed_mask_parts[i][j] = np.eye(self.n_dims, dtype=np.bool)
        unfixed_mask_mat = np.block(unfixed_mask_parts)
        unfixed_mask_mat.setflags(write=False)
        assert unfixed_mask_mat.shape == (n_non_fixed * self.n_dims, self.n_particles * self.n_dims)
        # Store system matrices
        self._mass_matrix = mass_matrix
        self._select_matrix = unfixed_mask_mat
        self._stiff_mat = stiff_mat
        # Set up support functions for computing derivatives
        edge_indices = self.edge_indices
        n_particles = self.n_particles
        viscosity_constant = self.viscosity_constant
        n_dims = self.n_dims
        spring_consts = self.spring_consts
        rest_lengths = self.rest_lengths
        fixed_mask = self.fixed_mask
        masses = self.masses
        masses_expanded = np.expand_dims(masses, axis=(0, -1))
        masses_repeats = np.tile(np.expand_dims(masses, -1), (1, n_dims)).reshape((-1, ))
        M_inv = np.reciprocal(masses_repeats)
        @jit(nopython=True, fastmath=False)
        def gather_forces(edge_forces, out):
            for i in range(edge_indices.shape[1]):
                a = edge_indices[0, i]
                out[:, a] += edge_forces[:, i]
        @jit(nopython=True, fastmath=False)
        def compute_forces(q, q_dot):
            q = q.reshape((-1, n_particles, n_dims))
            q_dot = q_dot.reshape((-1, n_particles, n_dims))
            # Compute length of each spring and "diff" directions of the forces
            diffs = q[:, edge_indices[0], :] - q[:, edge_indices[1], :]
            diffs_vel = q_dot[:, edge_indices[0], :] - q_dot[:, edge_indices[1], :]
            lengths = np.sqrt((diffs ** 2).sum(axis=-1))
            # Compute forces
            edge_forces = ((np.expand_dims(-1 * spring_consts * (lengths - rest_lengths) / lengths, axis=-1) * diffs)
                           - viscosity_constant * diffs_vel)
            # Gather forces for each of their "lead" particles
            forces = np.zeros(shape=(q.shape[0], n_particles, n_dims), dtype=q.dtype)
            gather_forces(edge_forces=edge_forces, out=forces)
            # Mask forces on fixed particles
            forces[:, fixed_mask, :] = 0
            return forces
        self.compute_forces = compute_forces
        # Set up free derivative function
        @jit(nopython=True, fastmath=False)
        def derivative(q, p):
            orig_q_shape = q.shape
            orig_p_shape = p.shape
            q_dot = M_inv * p
            q = q.reshape((-1, n_particles, n_dims))
            p = p.reshape((-1, n_particles, n_dims))
            # Compute action of forces on each particle
            forces = compute_forces(q=q, q_dot=q_dot)
            # Update positions
            pos = (1 / masses_expanded) * p
            pos[:, fixed_mask, :] = 0
            q_out = pos.reshape(orig_q_shape)
            p_out = forces.reshape(orig_p_shape)
            return q_out, p_out
        self.derivative = derivative

        # Configure functions for Newton iteration
        jac_idx_arr = np.arange(n_particles * n_dims).reshape((n_particles, n_dims))
        arr_size = n_particles * n_dims
        eye_n_dims = np.tile(np.expand_dims(np.eye(n_dims), axis=0), (2*len(self.edges), 1, 1))
        jac_b = np.zeros((arr_size, arr_size))
        I = np.eye(n_dims * n_particles)
        zeromat = np.zeros_like(I)
        I_zero = np.block([I, zeromat])
        zero_I = np.block([zeromat, I])
        for edge in self.edges:
            for a, b in [(edge.a, edge.b), (edge.b, edge.a)]:
                jac_b[jac_idx_arr[a], jac_idx_arr[b]] += viscosity_constant
                jac_b[jac_idx_arr[a], jac_idx_arr[a]] -= viscosity_constant
        jac_idx_arr.setflags(write=False)
        eye_n_dims.setflags(write=False)
        jac_b.setflags(write=False)
        I.setflags(write=False)
        zeromat.setflags(write=False)
        I_zero.setflags(write=False)
        zero_I.setflags(write=False)

        @jit(nopython=True)
        def store_jac(term_ab, jac):
            for i in range(term_ab.shape[0]):
                # Store term
                a = edge_indices[0, i]
                b = edge_indices[1, i]
                jac[jac_idx_arr[a][0]:jac_idx_arr[a][1]+1, jac_idx_arr[b][0]:jac_idx_arr[b][1]+1] = term_ab[i]

        @jit(nopython=True)
        def _force_grad(q, p):
            q = q.reshape((n_particles, n_dims))
            p = p.reshape((n_particles, n_dims))
            jac = np.zeros((arr_size, arr_size))

            diff = q[edge_indices[0], :] - q[edge_indices[1], :]
            norm2 = np.sqrt((diff ** 2).sum(axis=-1))
            lead_coeffs = np.expand_dims(np.expand_dims(spring_consts[0], axis=-1), axis=-1)

            term1_scalar = np.expand_dims(np.expand_dims(norm2 - rest_lengths[0], axis=-1), axis=-1)
            term1_deriv_vec = diff / np.expand_dims(norm2, axis=-1)
            term2_vec = diff / np.expand_dims(norm2, axis=-1)
            outer_prod = batch_outer(diff, diff)

            term2_a = eye_n_dims / np.expand_dims(np.expand_dims(norm2, axis=-1), axis=-1)
            term2_b = outer_prod/np.expand_dims(np.expand_dims(norm2**3, axis=-1), axis=-1)
            term2_deriv_mat = term2_a - term2_b
            term_ab = (lead_coeffs * (batch_outer(term1_deriv_vec, term2_vec)) +
                       lead_coeffs * (term1_scalar * term2_deriv_mat))
            # Store results from term_ab
            store_jac(term_ab, jac)
            res = np.concatenate((jac, jac_b), axis=-1)
            res[np.repeat(fixed_mask, n_dims)] = 0
            return res

        @jit(nopython=True)
        def _newton_func_val(q_prev, q_dot_prev, q_next, q_dot_next, dt):
            forces = compute_forces(q_next, q_dot_next).reshape((-1, ))
            q_val = q_next - q_prev - dt * q_dot_prev - dt**2 * M_inv * forces
            q_dot_val = q_dot_next - q_dot_prev - dt * M_inv * forces
            return np.concatenate((q_val, q_dot_val), axis=-1)

        @jit(nopython=True)
        def _newton_func_jac(q_prev, q_dot_prev, q_next, q_dot_next, dt):
            minv_jf = np.diag(M_inv) @ _force_grad(q_next, q_dot_next)
            q_rows = I_zero - dt**2 * minv_jf
            q_dot_rows = zero_I - dt * minv_jf
            return np.concatenate((q_rows, q_dot_rows), axis=0)

        # Do the newton iterations
        @jit(nopython=True)
        def compute_next_step(q_prev, q_dot_prev, dt):
            split_idx = n_dims * n_particles
            q_prev = q_prev.reshape((-1, ))
            q_dot_prev = q_dot_prev.reshape((-1, ))
            q_next = q_prev.copy()
            q_dot_next = q_dot_prev.copy()
            val = _newton_func_val(q_prev, q_dot_prev, q_next, q_dot_next, dt)
            num_iter = 0
            while np.linalg.norm(val) > 1e-12:
                val = _newton_func_val(q_prev, q_dot_prev, q_next, q_dot_next, dt)
                jac = _newton_func_jac(q_prev, q_dot_prev, q_next, q_dot_next, dt)
                jac_inv_prod = np.linalg.solve(jac, val)
                q_incr = jac_inv_prod[:split_idx]
                q_dot_incr = jac_inv_prod[split_idx:]
                q_next = q_next - q_incr
                q_dot_next = q_dot_next - q_dot_incr
                num_iter += 1
                if num_iter > 50:
                    break
            return q_next, q_dot_next

        @jit(nopython=True)
        def back_euler(q0, p0, dt, out_q, out_p):
            q = q0.reshape((-1, ))
            q_dot = M_inv * p0.reshape((-1, ))
            for i in range(out_q.shape[0]):
                out_q[i] = q
                out_p[i] = masses_repeats * q_dot
                q, q_dot = compute_next_step(q, q_dot, dt)
        self.back_euler = back_euler

    def _args_compatible(self, n_dims, particles, edges, vel_decay):
        return (self.n_dims == n_dims and
                self.particles == particles and
                set(self.edges) == set(edges) and
                self.viscosity_constant == vel_decay)

    def hamiltonian(self, q, p):
        return np.zeros(q.shape[0], q.shape[1])

    def _compute_next_step(self, q, q_dot, time_step_size, mat_unknown_factors):
        # Input states are (n_particle, n_dim)
        forces_orig = self.compute_forces(q=q, q_dot=q_dot)[0]
        forces = forces_orig.reshape((-1,))
        q = q.reshape((-1, ))
        q_dot = q_dot.reshape((-1, ))
        known = self._select_matrix @ (self._mass_matrix @ q_dot) + (time_step_size * (self._select_matrix @ forces))
        # Two of the values to return
        q_dot_hat_next = lu_solve(mat_unknown_factors, known)
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
        p_dots = [self.compute_forces(q=q0, q_dot=p0)[0]]
        q = q0.copy()
        q_dot = p0.copy()

        for i, part in enumerate(self.particles):
            q_dot[i] /= part.mass
        for step_idx in range(1, num_steps):
            q, q_dot, p, _p_dot_next = self._compute_next_step(q=q, q_dot=q_dot, time_step_size=time_step_size,
                                                               mat_unknown_factors=mat_unknown_factors)
            if step_idx % subsample == 0:
                p_dot = self.compute_forces(q=q, q_dot=p)[0]
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
    cached_sys = spring_mesh_cache.find(n_dims=n_dims,
                                        particles=parts,
                                        edges=edgs,
                                        vel_decay=vel_decay)
    if cached_sys is not None:
        return cached_sys
    else:
        new_sys = SpringMeshSystem(n_dims=n_dims,
                                   particles=parts,
                                   edges=edgs,
                                   vel_decay=vel_decay)
        spring_mesh_cache.insert(new_sys)
        return new_sys


def generate_data(system_args, base_logger=None):
    if base_logger:
        logger = base_logger.getChild("spring-mesh")
    else:
        logger = logging.getLogger("spring-mesh")

    trajectory_metadata = []
    trajectories = {}
    vel_decay = system_args.get("vel_decay", 0.0)
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

        system = spring_mesh_cache.find(n_dims=n_dims, particles=particles,
                                        edges=edges, vel_decay=vel_decay)
        if system is None:
            system = SpringMeshSystem(n_dims=n_dims, particles=particles,
                                      edges=edges, vel_decay=vel_decay)
            spring_mesh_cache.insert(system)

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
