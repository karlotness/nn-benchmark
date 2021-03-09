from collections import namedtuple, deque
import logging

TrajectoryResult = namedtuple("TrajectoryResult", ["q", "p", "dq_dt", "dp_dt", "t_steps", "p_noiseless", "q_noiseless"])
SystemResult = namedtuple("SystemResult",
                          ["trajectories", "trajectory_metadata", "metadata"])
StatePair = namedtuple("StatePair", ["q", "p"])

class System:
    def __init__(self):
        pass

    def hamiltonian(self, q, p):
        """Compute hamiltonian for each coord. Input [batch, 2, n_grid] or [2, n_grid]. Return [batch] or scalar"""
        raise NotImplementedError("Subclass this")

    def derivative(self, q, p):
        """Return local q, p time derivative for each of coord. Input shape [batch, 2, n_grid] or [2, n_grid]"""
        raise NotImplementedError("Subclass this")

    def generate_trajectory(self, q0, p0, num_time_steps, time_step_size, **kwargs):
        """x0 is initial state, t_span = (t_min, t_max), time_step_size with
        t_span determines # steps
        """
        raise NotImplementedError("Subclass this")


class SystemCache:
    def __init__(self, size=5):
        self._deque = deque(maxlen=size)
        self._logger = logging.getLogger("syscache")

    def find(self, *args, **kwargs):
        for system in self._deque:
            if system._args_compatible(*args, **kwargs):
                self._logger.info("Found compatible cached system")
                return system
        self._logger.info("No cached system found")
        return None

    def insert(self, system):
        self._logger.info("System added to cache")
        self._deque.append(system)
