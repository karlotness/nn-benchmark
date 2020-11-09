import numpy as np
from scipy.integrate import solve_ivp
from .defs import System, TrajectoryResult, SystemResult
import logging
import time


class SpringSystem(System):
    def __init__(self):
        super().__init__()

    def hamiltonian(self, coord):
        q, p = np.split(coord.T, 2)
        return q**2 + p**2

    def _hamiltonian_grad(self, coord):
        return 2 * coord

    def _dynamics(self, time, coord):
        return self.derivative(coord)

    def derivative(self, coord):
        grads = self._hamiltonian_grad(coord)
        dqdt, dpdt = np.split(grads, 2)
        return np.concatenate([dpdt, -dqdt], axis=-1)

    def generate_trajectory(self, x0, t_span, time_step_size, rtol=1e-10,
                            noise_sigma=0.0):
        t_min, t_max = t_span
        assert t_min < t_max
        num_steps = np.ceil((t_max - t_min) / time_step_size)
        t_eval = np.arange(num_steps) * time_step_size + t_min

        spring_ivp = solve_ivp(fun=self._dynamics, t_span=t_span, y0=x0,
                               t_eval=t_eval, rtol=rtol)
        q, p = spring_ivp['y'][0], spring_ivp['y'][1]
        dydt = [self._dynamics(None, y) for y in spring_ivp['y'].T]
        dydt = np.stack(dydt).T
        dqdt, dpdt = np.split(dydt,2)

        noise_p = noise_sigma * np.random.randn(*p.shape)
        noise_q = noise_sigma * np.random.randn(*q.shape)

        p_noisy = p + noise_p
        q_noisy = q + noise_q

        return TrajectoryResult(q=q_noisy[:, np.newaxis],
                                p=p_noisy[:, np.newaxis],
                                dq_dt=dqdt[0, :, np.newaxis],
                                dp_dt=dpdt[0, :, np.newaxis],
                                t_steps=t_eval,
                                q_noiseless=q[:, np.newaxis],
                                p_noiseless=p[:, np.newaxis])


def generate_data(system_args, base_logger=None):
    if base_logger:
        logger = base_logger.getChild("spring")
    else:
        logger = logging.getLogger("spring")

    system = SpringSystem()

    trajectory_metadata = []
    trajectories = {}
    trajectory_defs = system_args["trajectory_defs"]
    for i, traj_def in enumerate(trajectory_defs):
        traj_name = f"traj_{i:05}"
        logger.info(f"Generating trajectory {traj_name}")

        # Extract parameters
        x0 = np.array(traj_def["initial_condition"], dtype=np.float64)
        num_time_steps = traj_def["num_time_steps"]
        time_step_size = traj_def["time_step_size"]
        noise_sigma = traj_def.get("noise_sigma", 0.0)
        t_span = (0, num_time_steps * time_step_size)
        rtol = traj_def.get("rtol", 1e-10)

        # Generate trajectory
        traj_gen_start = time.perf_counter()
        traj_result = system.generate_trajectory(x0=x0,
                                                 t_span=t_span,
                                                 time_step_size=time_step_size,
                                                 rtol=rtol,
                                                 noise_sigma=noise_sigma)
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
             },
             "timing": {
                 "traj_gen_time": traj_gen_end - traj_gen_start
             }})
    logger.info("Done generating trajectories")

    return SystemResult(trajectories=trajectories,
                        metadata={
                            "n_grid": 1,
                        },
                        trajectory_metadata=trajectory_metadata)
