import numpy as np
from .defs import System, TrajectoryResult, SystemResult, StatePair
import time
import logging
import itertools
from collections import namedtuple
from scipy.integrate import solve_ivp


ParticleTrajectoryResult = namedtuple("ParticleTrajectoryResult",
                                      ["q", "p",
                                       "dq_dt", "dp_dt",
                                       "t_steps",
                                       "p_noiseless", "q_noiseless",
                                       "masses"])


class ParticleSystem(System):
    def __init__(self, n_particles, n_dim, g=1):
        super().__init__()
        self.n_particles = n_particles
        self.n_dim = n_dim
        self.g = g

    def hamiltonian(self, q, p, masses, epsilon=1e-2):
        q = q.reshape((-1, self.n_particles, self.n_dim))
        p = p.reshape((-1, self.n_particles, self.n_dim))
        masses = masses.reshape((self.n_particles, ))
        assert q.shape == p.shape
        num_steps = q.shape[0]
        # Initialize with ke
        hamilts = np.sum(1/(2 * masses.reshape(1, self.n_particles)) * np.sum(p ** 2, axis=-1), axis=-1)
        for step in range(num_steps):
            for i, j in itertools.combinations(range(self.n_particles), 2):
                diff = q[step, j] - q[step, i]
                dist = np.linalg.norm(diff, ord=2) + epsilon
                pe = -1 * self.g * masses[i] * masses[j] / dist
                hamilts[step] += pe
        return hamilts

    def derivative(self, q, p, masses):
        q = q.reshape((-1, self.n_particles, self.n_dim))
        p = p.reshape((-1, self.n_particles, self.n_dim))
        masses = masses.reshape((self.n_particles, ))
        assert q.shape == p.shape
        num_steps = q.shape[0]
        dq_dt = p / masses.reshape((1, self.n_particles, 1))
        dp_dt = np.zeros((num_steps, self.n_particles, self.n_dim))
        # Compute change in momentum
        for step in range(num_steps):
            for i, j in itertools.combinations(range(self.n_particles), 2):
                fqa, fqb = self._pairwise_force(qa=q[step, i], ma=masses[i],
                                                qb=q[step, j], mb=masses[j])
                dp_dt[step, i] += fqa
                dp_dt[step, j] += fqb
        return StatePair(q=dq_dt, p=dp_dt)

    def _pairwise_force(self, qa, qb, ma, mb, epsilon=1e-2):
        # Shapes each (n_dim) for position
        diff = qb - qa
        dist = np.linalg.norm(diff, ord=2) + epsilon
        mass_prod = ma * mb
        denom = dist**3
        fqa = self.g * mass_prod * (diff / denom)
        fqb = -1 * fqa
        return fqa, fqb

    def _dynamics(self, _time, coord):
        # Coord shape is n_particles * (n_dim: q, n_dim: v, 1: mass)
        orig_shape = coord.shape
        coord = coord.reshape((self.n_particles, 2 * self.n_dim + 1))
        qs = coord[:, :self.n_dim]
        vs = coord[:, self.n_dim:2*self.n_dim]
        masses = coord[:, -1]
        # Accumulate forces
        force_acc = np.zeros((self.n_particles, self.n_dim))
        for i, j in itertools.combinations(range(self.n_particles), 2):
            fqa, fqb = self._pairwise_force(qa=qs[i], ma=masses[i],
                                            qb=qs[j], mb=masses[j])
            force_acc[i] += fqa
            force_acc[j] += fqb
        # Compute updates
        accel = force_acc / masses.reshape((self.n_particles, 1))
        res = np.concatenate((vs, accel, np.zeros((self.n_particles, 1))), axis=-1)
        # Return result
        return res.reshape(orig_shape)

    def generate_trajectory(self, q0, p0, num_time_steps, time_step_size,
                            masses, rtol=1e-10, noise_sigma=0.0):
        # Check shapes of inputs
        if ((q0.shape != (self.n_particles, self.n_dim)) or (p0.shape != (self.n_particles, self.n_dim)) or (masses.shape != (self.n_particles,))):
            raise ValueError("Invalid input shape for particle system")
        m0 = masses.reshape((self.n_particles, 1))
        v0 = p0 / m0
        x0 = np.concatenate((q0, v0, m0), axis=-1).reshape((-1, ))
        # Run the solver
        t_span = (0, num_time_steps * time_step_size)
        t_eval = np.arange(num_time_steps) * time_step_size
        particle_ivp = solve_ivp(fun=self._dynamics, t_span=t_span, y0=x0,
                                 t_eval=t_eval, rtol=rtol)
        # Extract results
        res = particle_ivp['y']
        res = np.moveaxis(res, 0, -1).reshape((num_time_steps, self.n_particles, 2 * self.n_dim + 1))
        qs = res[:, :, :self.n_dim]
        vs = res[:, :, self.n_dim:2*self.n_dim]
        ps = vs * masses.reshape((1, self.n_particles, 1))

        # Compute derivatives
        derivs = self.derivative(q=qs, p=ps, masses=masses)
        dq_dt = derivs.q
        dp_dt = derivs.p

        # Add configured noise
        noise_ps = noise_sigma * np.random.randn(*ps.shape)
        noise_qs = noise_sigma * np.random.randn(*qs.shape)

        qs_noisy = qs + noise_qs
        ps_noisy = ps + noise_ps

        return ParticleTrajectoryResult(
            q=qs_noisy,
            p=ps_noisy,
            dq_dt=dq_dt,
            dp_dt=dp_dt,
            t_steps=t_eval,
            q_noiseless=qs,
            p_noiseless=ps,
            masses=masses)


def generate_data(system_args, base_logger=None):
    if base_logger:
        logger = base_logger.getChild("particle")
    else:
        logger = logging.getLogger("particle")

    # Global system parameters
    n_particles = system_args["n_particles"]
    n_dim = system_args["n_dim"]
    g = system_args.get("g", 1.0)

    system = ParticleSystem(n_particles=n_particles,
                            n_dim=n_dim,
                            g=g)

    trajectory_metadata = []
    trajectories = {}
    trajectory_defs = system_args["trajectory_defs"]
    for i, traj_def in enumerate(trajectory_defs):
        traj_name = f"traj_{i:05}"
        logger.info(f"Generating trajectory {traj_name}")

        q0 = np.array(traj_def["q0"]).astype(np.float64)
        p0 = np.array(traj_def["p0"]).astype(np.float64)
        masses = np.array(traj_def["masses"]).astype(np.float64)
        num_time_steps = traj_def["num_time_steps"]
        time_step_size = traj_def["time_step_size"]
        rtol = traj_def.get("rtol", 1e-10)
        noise_sigma = traj_def.get("noise_sigma", 0.0)

        # Generate trajectory
        traj_gen_start = time.perf_counter()
        traj_result = system.generate_trajectory(
            q0=q0, p0=p0, masses=masses,
            num_time_steps=num_time_steps, time_step_size=time_step_size,
            rtol=rtol, noise_sigma=noise_sigma)
        traj_gen_end = time.perf_counter()

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
             },
             "timing": {
                 "traj_gen_time": traj_gen_end - traj_gen_start
             }})

    logger.info("Done generating trajectories")

    return SystemResult(trajectories=trajectories,
                        metadata={
                            "n_grid": n_dim,
                            "n_dim": n_dim,
                            "n_particles": n_particles,
                            "g": g,
                            "system_type": "particle",
                        },
                        trajectory_metadata=trajectory_metadata)
