import numpy as np
from scipy.linalg import circulant
from .defs import System


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

    def hamiltonian(self, coord):
        if len(coord.shape) == 2:
            q, p = coord[0], coord[1]
        else:
            q, p = coord[:, 0], coord[:, 1]

        denom = 4 * self.d_x**2
        q_m1 = np.roll(q, shift=1, axis=-1)
        q_p1 = np.roll(q, shift=-1, axis=-1)

        t1 = 0.5 * p**2
        t2 = self.wave_speed**2 * (q_p1 - q)**2 / denom
        t3 = self.wave_speed**2 * (q - q_m1)**2 / denom

        return self.d_x * np.sum((t1 + t2 + t3), axis=-1)

    def derivative(self, coord):
        orig_shape = coord.shape
        return (coord.reshape((-1, 2 * self.n_grid)) @ self.k.T).reshape(orig_shape)

    def generate_trajectory(self, x0, t_span, time_step_size):
        t_min, t_max = t_span
        assert t_min < t_max
        num_steps = int(np.ceil((t_max - t_min) / time_step_size))
        eqn_known, eqn_unknown = _build_update_matrices(n_grid=self.n_grid, space_max=self.space_max,
                                                        wave_speed=self.wave_speed, time_step=time_step_size)
        steps = [x0]
        step = x0
        for _ in range(num_steps):
            step = self._compute_next_step(step, eqn_known, eqn_unknown)
            steps.append(step)

        return np.stack(steps)

    def _compute_next_step(self, prev_step, eqn_known, eqn_unknown):
        orig_shape = prev_step.shape
        prev_step = prev_step.reshape((-1))
        known = eqn_known @ prev_step
        new_step = np.linalg.solve(eqn_unknown, known)
        return new_step.reshape(orig_shape)


class WaveStartGenerator:
    def __init__(self, space_max, n_grid):
        self.n_grid = n_grid
        self.space_max = space_max
        self.n_grid = n_grid
        self.coords = np.linspace(0, self.space_max, self.n_grid)
        self.position = 0.5

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

    def gen_start(self, height, width):
        y_matrix = np.zeros(2 * self.n_grid)
        q_init = np.zeros(self.n_grid, dtype='float64')
        steps = np.copy(self.coords)
        wave = self.__h(self.__s(steps, self.position, width), height)
        q_init += wave
        y_matrix[:self.n_grid] = q_init
        return y_matrix.reshape((2, self.n_grid))
