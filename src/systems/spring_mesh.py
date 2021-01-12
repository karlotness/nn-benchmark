import numpy as np
from scipy.integrate import solve_ivp
from .defs import System, TrajectoryResult, SystemResult, StatePair
from collections import namedtuple
import logging
import time
import torch


Particle = namedtuple("Particle", ["mass", "is_fixed"])
# Particle numbers are zero indexed
Edge = namedtuple("Edge", ["a", "b", "spring_const", "rest_length"])


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

    def derivative(self, q, p):
        q = q.reshape((-1, self.n_particles, self.n_dims))
        p = p.reshape((-1, self.n_particles, self.n_dims))
        # Compute action of forces on each particle
        forces = np.zeros_like(p)
        for edge in self.edges:
            length = np.expand_dims(np.linalg.norm(q[:, edge.a] - q[:, edge.b], ord=2, axis=-1), axis=-1)
            for a, b in [(edge.a, edge.b), (edge.b, edge.a)]:
                diff = q[:, a] - q[:, b]
                forces[:, a] += -1 * edge.spring_const * (length - edge.rest_length) / length * diff
        # Update positions
        pos = np.zeros_like(q)
        for i, particle in enumerate(self.particles):
            if particle.is_fixed:
                forces[:, i] = 0
                pos[:, i] = 0
            else:
                pos[:, i] = (1/particle.mass) * p[:, i]
        return StatePair(q=pos, p=forces)

    def generate_trajectory(self, q0, p0, num_time_steps, time_step_size, rtol=1e-10,
                            noise_sigma=0.0):
        # Check shapes of inputs
        if (q0.shape != (self.n_particles, self.n_dims)) or (p0.shape != (self.n_particles, self.n_dims)):
            raise ValueError("Invalid input shape for particle system")
        x0 = np.concatenate((q0.reshape((-1, )), p0.reshape((-1, ))), axis=-1).reshape((-1, ))
        # Run the solver
        t_span = (0, num_time_steps * time_step_size)
        t_eval = np.arange(num_time_steps) * time_step_size
        particle_ivp = solve_ivp(fun=self._dynamics, t_span=t_span, y0=x0,
                                 t_eval=t_eval, rtol=rtol)
        # Extract results
        res = particle_ivp['y']
        res = np.moveaxis(res, 0, -1)
        qs, ps = np.split(res, 2, axis=-1)
        qs = qs.reshape(num_time_steps, self.n_particles, self.n_dims)
        ps = ps.reshape(num_time_steps, self.n_particles, self.n_dims)

        # Compute derivatives
        derivs = self.derivative(q=qs, p=ps)
        dq_dt = derivs.q
        dp_dt = derivs.p

        # Add configured noise
        noise_ps = noise_sigma * np.random.randn(*ps.shape)
        noise_qs = noise_sigma * np.random.randn(*qs.shape)

        qs_noisy = qs + noise_qs
        ps_noisy = ps + noise_ps

        return TrajectoryResult(
            q=qs_noisy,
            p=ps_noisy,
            dq_dt=dq_dt,
            dp_dt=dp_dt,
            t_steps=t_eval,
            q_noiseless=qs,
            p_noiseless=ps)

def generate_data(system_args, base_logger=None):
    if base_logger:
        logger = base_logger.getChild("spring-mesh")
    else:
        logger = logging.getLogger("spring-mesh")

    system = SpringMeshSystem()
