from collections import namedtuple

TrajectoryResult = namedtuple("TrajectoryResult", ["q", "p", "dq_dt", "dp_dt", "t_steps", "p_noiseless", "q_noiseless"])
SystemResult = namedtuple("SystemResult",
                          ["trajectories", "trajectory_metadata", "metadata"])

class System:
    def __init__(self):
        pass

    def hamiltonian(self, coord):
        """Compute hamiltonian for each coord. Input [batch, 2, n_grid] or [2, n_grid]. Return [batch] or scalar"""
        raise NotImplementedError("Subclass this")

    def derivative(self, coord):
        """Return local q, p time derivative for each of coord. Input shape [batch, 2, n_grid] or [2, n_grid]"""
        raise NotImplementedError("Subclass this")

    def generate_trajectory(self, x0, t_span, time_step_size, **kwargs):
        """x0 is initial state, t_span = (t_min, t_max), time_step_size with
        t_span determines # steps
        """
        raise NotImplementedError("Subclass this")
