import numpy as np
from scipy.linalg import circulant, lu_factor, lu_solve
from .defs import System, TrajectoryResult, SystemResult, StatePair
import time
import logging
import torch


def _build_update_matrices(n_grid, space_max, wave_speed, time_step):
    # Build update matrices
    delta_x = space_max / n_grid
    I_2n = np.identity(2 * n_grid)
    I_n = np.identity(n_grid)
    zero_n = np.zeros((n_grid, n_grid))
    stencil = np.zeros(n_grid)
    stencil[:3] = [1, -2, 1]
    stencil = (1 / delta_x**2) * np.roll(stencil, -1)
    d_xx = circulant(stencil)
    K = np.block([[zero_n, I_n], [wave_speed**2 * d_xx, zero_n]])

    # Produce "equation" matrices
    eqn_known = (I_2n + (time_step / 2) * K)
    eqn_unknown = (I_2n - (time_step / 2) * K)
    return eqn_known, eqn_unknown


def _get_k(n_grid, space_max, wave_speed):
    # Build update matrices
    delta_x = space_max / n_grid
    I_n = np.identity(n_grid)
    zero_n = np.zeros((n_grid, n_grid))
    stencil = np.zeros(n_grid)
    stencil[:3] = [1, -2, 1]
    stencil = (1 / delta_x**2) * np.roll(stencil, -1)
    d_xx = circulant(stencil)
    return np.block([[zero_n, I_n], [wave_speed**2 * d_xx, zero_n]])


class WaveSystem(System):
    def __init__(self, n_grid, space_max, wave_speed):
        super().__init__()
        self.n_grid = n_grid
        self.space_max = space_max
        self.wave_speed = wave_speed
        self.d_x = self.space_max / self.n_grid
        self.k = _get_k(n_grid=n_grid, space_max=space_max, wave_speed=wave_speed)

    def implicit_matrix_package(self, q, p):
        return torch.cat((q, p), dim=-1)

    def implicit_matrix_unpackage(self, x):
        return StatePair(q=x[..., :self.n_grid], p=x[..., self.n_grid:])

    def implicit_matrix(self, x):
        return torch.from_numpy(self.k)

    def hamiltonian(self, q, p):
        denom = 4 * self.d_x**2
        q_m1 = np.roll(q, shift=1, axis=-1)
        q_p1 = np.roll(q, shift=-1, axis=-1)

        t1 = 0.5 * p**2
        t2 = self.wave_speed**2 * (q_p1 - q)**2 / denom
        t3 = self.wave_speed**2 * (q - q_m1)**2 / denom

        return self.d_x * np.sum((t1 + t2 + t3), axis=-1)

    def derivative(self, q, p, dt=1.0):
        coord = np.concatenate((q, p), axis=-1)
        deriv_coord = coord @ self.k.T
        dqdt = deriv_coord[..., :self.n_grid]
        dpdt = deriv_coord[..., self.n_grid:]
        return StatePair(q=dqdt, p=dpdt)

    def generate_trajectory(self, q0, p0, num_time_steps, time_step_size,
                            subsample=1, noise_sigma=0.0):
        # Process arguments for subsampling
        num_steps = num_time_steps * subsample
        orig_time_step_size = time_step_size
        time_step_size = time_step_size / subsample

        eqn_known, eqn_unknown = _build_update_matrices(n_grid=self.n_grid, space_max=self.space_max,
                                                        wave_speed=self.wave_speed, time_step=time_step_size)
        eqn_unknown_factor = lu_factor(eqn_unknown)

        x0 = np.stack((q0, p0))
        steps = [x0]
        step = x0
        for step_idx in range(1, num_steps):
            step = self._compute_next_step(step, eqn_known, eqn_unknown_factor)
            if step_idx % subsample == 0:
                steps.append(step)
        steps = np.stack(steps)

        q = steps[:, 0]
        p = steps[:, 1]

        derivatives = self.derivative(q=q, p=p)

        noise_p = noise_sigma * np.random.randn(*p.shape)
        noise_q = noise_sigma * np.random.randn(*q.shape)

        p_noisy = p + noise_p
        q_noisy = q + noise_q

        dqdt = derivatives.q
        dpdt = derivatives.p
        t_steps = (np.arange(num_time_steps) * orig_time_step_size).astype(np.float64)

        return TrajectoryResult(q=q_noisy, p=p_noisy, dq_dt=dqdt, dp_dt=dpdt,
                                t_steps=t_steps,
                                q_noiseless=q, p_noiseless=p)

    def _compute_next_step(self, prev_step, eqn_known, eqn_unknown_factor):
        orig_shape = prev_step.shape
        prev_step = prev_step.reshape((-1))
        known = eqn_known @ prev_step
        new_step = lu_solve(eqn_unknown_factor, known)
        return new_step.reshape(orig_shape)


class WaveStartGenerator:
    def __init__(self, space_max, n_grid):
        self.n_grid = n_grid
        self.space_max = space_max
        self.n_grid = n_grid
        self.coords = np.linspace(0, self.space_max, self.n_grid)

    def __s(self, x, center, width):
        return 10 * np.abs((1/width) * (x - center))

    def __h(self, s, height):
        mask_1 = (0 <= s) & (s <= 1)
        mask_2 = (1 < s) & (s <= 2)
        mask_3 = s > 2
        s_clone = s.copy()
        s_clone[mask_1] = 1 - (3 / 2) * np.power(s_clone[mask_1], 2) + \
            (3 / 4) * np.power(s_clone[mask_1], 3)
        s_clone[mask_2] = (1 / 4) * np.power(2 - s_clone[mask_2], 3)
        s_clone[mask_3] = 0
        return height * s_clone

    def gen_start(self, height, width, position):
        q_init = np.zeros(self.n_grid, dtype='float64')
        steps = np.copy(self.coords)
        wave = self.__h(self.__s(steps, position, width), height)
        q_init += wave
        p_init = np.zeros_like(q_init)
        return StatePair(q=q_init, p=p_init)


def generate_cubic_spline_start(space_max, n_grid, start_type_args):
    start_gen = WaveStartGenerator(space_max=space_max, n_grid=n_grid)
    init_cond = start_gen.gen_start(height=start_type_args["height"],
                                    width=start_type_args["width"],
                                    position=start_type_args["position"])
    return init_cond


def generate_data(system_args, base_logger=None):
    if base_logger:
        logger = base_logger.getChild("wave")
    else:
        logger = logging.getLogger("wave")

    # Global system parameters
    space_max = system_args["space_max"]
    n_grid = system_args["n_grid"]

    trajectory_metadata = []
    trajectories = {}
    trajectory_defs = system_args["trajectory_defs"]
    for i, traj_def in enumerate(trajectory_defs):
        traj_name = f"traj_{i:05}"
        logger.info(f"Generating trajectory {traj_name}")

        # Generate starting conditions
        start_type = traj_def["start_type"]
        start_type_args = traj_def["start_type_args"]
        if start_type == "cubic_splines":
            init_cond = generate_cubic_spline_start(space_max=space_max,
                                                    n_grid=n_grid,
                                                    start_type_args=start_type_args)
        else:
            logger.error(f"Unknown start type: {start_type}")
            raise ValueError(f"Unknown start type: {start_type}")

        # Create the trajectory
        wave_speed = traj_def["wave_speed"]
        num_time_steps = traj_def["num_time_steps"]
        time_step_size = traj_def["time_step_size"]
        noise_sigma = traj_def.get("noise_sigma", 0.0)
        subsample = int(traj_def.get("subsample", 1))
        system = WaveSystem(n_grid=n_grid, space_max=space_max,
                            wave_speed=wave_speed)

        traj_gen_start = time.perf_counter()
        traj_result = system.generate_trajectory(q0=init_cond.q,
                                                 p0=init_cond.p,
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
        })

        # Store per-trajectory metadata
        trajectory_metadata.append(
            {"name": traj_name,
             "num_time_steps": num_time_steps,
             "time_step_size": time_step_size,
             "wave_speed": wave_speed,
             "noise_sigma": noise_sigma,
             "field_keys": {
                 "p": f"{traj_name}_p",
                 "q": f"{traj_name}_q",
                 "dpdt": f"{traj_name}_dpdt",
                 "dqdt": f"{traj_name}_dqdt",
                 "t": f"{traj_name}_t",
                 "p_noiseless": f"{traj_name}_p_noiseless",
                 "q_noiseless": f"{traj_name}_q_noiseless",
             },
             "timing": {
                 "traj_gen_time": traj_gen_elapsed
             }})

    logger.info("Done generating trajectories")

    return SystemResult(trajectories=trajectories,
                        metadata={
                            "n_grid": n_grid,
                            "space_max": space_max,
                        },
                        trajectory_metadata=trajectory_metadata)
