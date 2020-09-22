import numpy as np
from scipy.integrate import solve_ivp
from .defs import System, TrajectoryResult


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

    def generate_trajectory(self, x0, t_span, time_step_size, rtol=1e-10):
        t_min, t_max = t_span
        assert t_min < t_max
        num_steps = np.ceil((t_max - t_min) / time_step_size)
        t_eval = np.linspace(t_min, t_max, int(num_steps))

        spring_ivp = solve_ivp(fun=self._dynamics, t_span=t_span, y0=x0,
                               t_eval=t_eval, rtol=rtol)
        q, p = spring_ivp['y'][0], spring_ivp['y'][1]
        dydt = [self._dynamics(None, y) for y in spring_ivp['y'].T]
        dydt = np.stack(dydt).T
        dqdt, dpdt = np.split(dydt,2)

        return TrajectoryResult(q=q, p=p, dq_dt=dqdt, dp_dt=dpdt, t_steps=t_eval)
