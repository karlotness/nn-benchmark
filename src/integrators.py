from collections import namedtuple
import numpy as np
import numba

IntegrationResult = namedtuple("IntegrationResult", ["q", "p"])

@numba.jit(nopython=True)
def euler(q0, p0, dt, func, out_q, out_p):
    q = q0
    p = p0
    for i in range(out_q.shape[0]):
        out_q[i] = q
        out_p[i] = p
        dq, dp = func(q, p)
        q = q + dt * dq
        p = p + dt * dp


@numba.jit(nopython=True)
def leapfrog(q0, p0, dt, func, out_q, out_p):
    q = q0
    p = p0
    dqdt, dpdt = func(q, p)
    for i in range(out_q.shape[0]):
        p_half = p + dpdt * (dt / 2)
        out_q[i] = q
        out_p[i] = p
        dqdt, dpdt = func(q, p_half)
        q_next = q + dqdt * dt
        dqdt, dpdt = func(q_next, p_half)
        p_next = p_half + dpdt * (dt / 2)
        p = p_next
        q = q_next


@numba.jit(nopython=True)
def rk4(q0, p0, dt, func, out_q, out_p):
    q = q0
    p = p0
    for i in range(out_q.shape[0]):
        out_q[i] = q
        out_p[i] = p
        q_k1, p_k1 = func(q, p)
        q_k2, p_k2 = func(q + 0.5*dt*q_k1, p + 0.5*dt*p_k1)
        q_k3, p_k3 = func(q + 0.5*dt*q_k2, p + 0.5*dt*p_k2)
        q_k4, p_k4 = func(q + dt*q_k3, p + dt*p_k3)
        p_next = p + (1./6.) * dt * (p_k1 + 2 * p_k2 + 2 * p_k3 + p_k4)
        q_next = q + (1./6.) * dt * (q_k1 + 2 * q_k2 + 2 * q_k3 + q_k4)
        p = p_next
        q = q_next


def null_integrator(q0, p0, dt, func, out_q, out_p):
    q = q0
    p = p0
    for i in range(out_q.shape[0]):
        out_q[i] = q
        out_p[i] = p
        q, p = func(q, p)


@numba.jit(nopython=True)
def backward_euler(x0, dt, func, out_x, deriv_mat):
    x = x0
    deriv_eye = np.eye(x.shape[-1], dtype=x0.dtype)
    unknown_mat = np.expand_dims(deriv_eye - dt * deriv_mat, 0)
    for i in range(out_x.shape[0]):
        out_x[i] = x
        x = np.linalg.solve(unknown_mat, x)


INTEGRATORS = {
    "euler": (euler, None),
    "leapfrog": (leapfrog, None),
    "rk4": (rk4, None),
    "null": (null_integrator, None),
    "back-euler": (backward_euler, "back_euler"),
    "implicit-rk": (None, "implicit_rk"),
}


def numerically_integrate(integrator, q0, p0, num_steps, dt, deriv_func, system=None):
    try:
        # Find the integrator function
        int_func, implicit_attr = INTEGRATORS[integrator]
        if (not isinstance(deriv_func, numba.core.dispatcher.Dispatcher)
            and isinstance(int_func, numba.core.dispatcher.Dispatcher)):
            # We weren't passed a JIT function, unwrap the integrator and call directly
            int_func = int_func.py_func
    except KeyError:
        raise ValueError(f"Unknown integrator {integrator}")
    if implicit_attr and hasattr(system, implicit_attr):
        # Call system's implicit integrator directly
        out_shape = (num_steps, p0.shape[1])
        out_q = np.empty_like(q0, shape=out_shape)
        out_p = np.empty_like(p0, shape=out_shape)
        getattr(system, implicit_attr)(q0, p0, dt, out_q, out_p)
    elif implicit_attr:
        # This is an implicit integrator, set up the extra context
        # In this case, we require `system` to be provided
        x0 = system.implicit_matrix_package(q=q0, p=p0)
        out_x = np.empty_like(q0, shape=out_shape)
        deriv_mat = system.implicit_matrix(x0)
        int_func(x0, dt, deriv_func, out_x, deriv_mat)
        ret_split = system.implicit_matrix_unpackage(out_x)
        out_q = ret_split.q
        out_p = ret_split.p
    else:
        # Allocate output array
        out_shape = (num_steps, p0.shape[1])
        out_q = np.empty_like(q0, shape=out_shape)
        out_p = np.empty_like(p0, shape=out_shape)
        int_func(q0, p0, dt, deriv_func, out_q, out_p)
    return IntegrationResult(q=out_q, p=out_p)
