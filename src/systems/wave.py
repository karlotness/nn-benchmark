import numpy as np
from scipy.linalg import circulant
from .defs import System



        raw_k = utils.get_k(n_grid, space_max, wave_speed)
        self.np_k = raw_k
        self.k = utils.to_torch_sparse(raw_k).float().to(tensor_device)
        self.k.requires_grad_(False)
        self._wave_matrix_cache = None
        if isinstance(n_grid, tuple):
            raise ValueError("This system does not support multi-dimensional grids")

def get_k(n_grid, space_max, wave_speed):
    # Build update matrices
    delta_x = space_max / n_grid
    I_n = np.identity(n_grid)
    zero_n = np.zeros((n_grid, n_grid))
    stencil = np.zeros(n_grid)
    stencil[:3] = [1, -2, 1]
    stencil = (1 / delta_x**2) * np.roll(stencil, -1)
    d_xx = linalg.circulant(stencil)
    return np.block([[zero_n, I_n], [wave_speed**2 * d_xx, zero_n]])


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
        q_m1 = np.roll(a, shift=1, axis=-1)
        q_p1 = np.roll(a, shift=-1, axis=-1)

        t1 = 0.5 * p**2
        t2 = self.wave_speed**2 * (q_p1 - q)**2 / denom
        t3 = self.wave_speed**2 * (q - q_m1)**2 / denom

        return self.d_x * np.sum((t1 + t2 + t3), axis=-1)

    def derivative(self, coord):
        orig_shape = coord.shape
        return (coord.reshape((-1, 2 * self.n_grid)) @ self.k.T).reshape(orig_shape)

    def generate_trajectory(self, x0, t_span, time_step_size):
        eqn_known, eqn_unknown = utils.build_update_matrices(n_grid=self.n_grid, space_max=self.space_max,
                                                             wave_speed=self.wave_speed, time_step=time_step_size)
        raise NotImplementedError("Subclass this")

    def _compute_next_step(self, prev_step, eqn_known, eqn_unknown):
        orig_shape = prev_step.shape
        prev_step = prev_step.reshape((-1))
        known = eqn_known @ prev_step
        new_step = np.linalg.solve(eqn_unknown, known)
        return new_step.reshape(orig_shape)
