import numpy as np
from .defs import System, TrajectoryResult, SystemResult, StatePair
from collections import namedtuple
import logging
import time
import torch


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
    def __init__(self, n_dims, particles, edges):
        super().__init__()
        self.particles = particles
        self.edges = edges
        self.n_dims = n_dims
        self.n_particles = len(particles)

    def hamiltonian(self, q, p):
        return 0

    def _dynamics(self, time, coord):
        q, p = np.split(coord, 2, axis=-1)
        deriv = self.derivative(q=q, p=p)
        return np.concatenate((deriv.q.reshape((-1, )),
                               deriv.p.reshape((-1, ))),
                              axis=-1)

    def compute_forces(self, q):
        q = q.reshape((-1, self.n_particles, self.n_dims))
        forces = np.zeros_like(q)
        for edge in self.edges:
            length = np.expand_dims(np.linalg.norm(q[:, edge.a] - q[:, edge.b], ord=2, axis=-1), axis=-1)
            for a, b in [(edge.a, edge.b), (edge.b, edge.a)]:
                diff = q[:, a] - q[:, b]
                forces[:, a] += -1 * edge.spring_const * (length - edge.rest_length) / length * diff
        return forces

    def derivative(self, q, p):
        q = q.reshape((-1, self.n_particles, self.n_dims))
        p = p.reshape((-1, self.n_particles, self.n_dims))
        # Compute action of forces on each particle
        forces = self.compute_forces(q=q)
        # Update positions
        pos = np.zeros_like(q)
        for i, particle in enumerate(self.particles):
            if particle.is_fixed:
                forces[:, i] = 0
                pos[:, i] = 0
            else:
                pos[:, i] = (1/particle.mass) * p[:, i]
        return StatePair(q=pos, p=forces)

    def _compute_next_step(self, q, q_dot, time_step_size, step_vel_decay=1.0):
        forces = self.compute_forces(q=q)[0]
        vel = np.zeros_like(q_dot)
        for i, part in enumerate(self.particles):
            if part.is_fixed:
                continue
            vel[i] = q_dot[i] + time_step_size * (1/part.mass) * forces[i]
        vel *= step_vel_decay
        pos = q + time_step_size * vel
        return pos, vel

    def generate_trajectory(self, q0, p0, num_time_steps, time_step_size,
                            subsample=1, noise_sigma=0.0, vel_decay=1.0):
        # Check shapes of inputs
        if (q0.shape != (self.n_particles, self.n_dims)) or (p0.shape != (self.n_particles, self.n_dims)):
            raise ValueError("Invalid input shape for particle system")

        t_eval = np.arange(num_time_steps) * time_step_size

        # Process arguments for subsampling
        num_steps = num_time_steps * subsample
        orig_time_step_size = time_step_size
        time_step_size = time_step_size / subsample
        step_vel_decay = vel_decay ** time_step_size

        # Compute updates using explicit Euler
        # TODO: Implicit Euler
        qs = [q0]
        q = q0.copy()
        q_dot = p0.copy()

        for i, part in enumerate(self.particles):
            q_dot[i] /= part.mass
        for step_idx in range(1, num_steps):
            q, q_dot = self._compute_next_step(q=q, q_dot=q_dot, time_step_size=time_step_size, step_vel_decay=step_vel_decay)
            if step_idx % subsample == 0:
                qs.append(q)
        ps = [self.compute_forces(q) for q in qs]

        qs = np.stack(qs).reshape(num_time_steps, self.n_particles, self.n_dims)
        ps = np.stack(ps).reshape(num_time_steps, self.n_particles, self.n_dims)

        # Compute derivatives
        derivs = self.derivative(q=qs, p=ps)
        dq_dt = derivs.q
        dp_dt = derivs.p

        # Add configured noise
        noise_ps = noise_sigma * np.random.randn(*ps.shape)
        noise_qs = noise_sigma * np.random.randn(*qs.shape)

        qs_noisy = qs + noise_qs
        ps_noisy = ps + noise_ps

        # Gather other data
        masses = np.array([p.mass for p in self.particles], dtype=np.float64)
        fixed_mask = np.array([p.is_fixed for p in self.particles], dtype=np.bool)
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
            masses=masses,
            edge_indices=edge_indices,
            fixed_mask=fixed_mask)


def system_from_records(n_dims, particles, edges):
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
                            edges=edgs)


def generate_data(system_args, base_logger=None):
    if base_logger:
        logger = base_logger.getChild("spring-mesh")
    else:
        logger = logging.getLogger("spring-mesh")

    trajectory_metadata = []
    trajectories = {}
    trajectory_defs = system_args["trajectory_defs"]
    for i, traj_def in enumerate(trajectory_defs):
        traj_name = f"traj_{i:05}"
        logger.info(f"Generating trajectory {traj_name}")

        # Create the trajectory
        particle_defs = traj_def["particles"]
        spring_defs = traj_def["springs"]
        num_time_steps = traj_def["num_time_steps"]
        time_step_size = traj_def["time_step_size"]
        vel_decay = traj_def.get("vel_decay", 1.0)
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
                                  edges=edges)

        traj_gen_start = time.perf_counter()
        traj_result = system.generate_trajectory(q0=q0,
                                                 p0=p0,
                                                 num_time_steps=num_time_steps,
                                                 time_step_size=time_step_size,
                                                 subsample=subsample,
                                                 vel_decay=vel_decay,
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
                        },
                        trajectory_metadata=trajectory_metadata)
