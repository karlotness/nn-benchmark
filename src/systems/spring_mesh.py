import numpy as np
from scipy.integrate import solve_ivp
from .defs import System, TrajectoryResult, SystemResult, StatePair
import logging
import time
import torch


class SpringSystem(System):
    def __init__(self):
        super().__init__()

    def hamiltonian(self, q, p):
        pass

    def _hamiltonian_grad(self, q, p):
        pass

    def _dynamics(self, time, coord):
        pass

    def derivative(self, q, p):
        pass

    def implicit_matrix_package(self, q, p):
        pass

    def implicit_matrix_unpackage(self, x):
        pass

    def implicit_matrix(self, x):
        pass

    def generate_trajectory(self, q0, p0, t_span, time_step_size, rtol=1e-10,
                            noise_sigma=0.0):
        pass


def generate_data(system_args, base_logger=None):
    if base_logger:
        logger = base_logger.getChild("spring-mesh")
    else:
        logger = logging.getLogger("spring-mesh")

    system = SpringMeshSystem()
