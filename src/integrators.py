from collections import namedtuple
import numpy as np
import numba

IntegrationResult = namedtuple("IntegrationResult", ["q", "p"])

@numba.jit(nopython=True)
def no_boundary_condition(q, p, t):
    q = q.copy()
    p = p.copy()
    return q, p

@numba.jit(nopython=True)
def euler(q0, p0, dt, func, bc_func, out_q, out_p, t0):
    q = q0
    p = p0
    t = t0
    for i in range(out_q.shape[0]):
        out_q[i] = q
        out_p[i] = p
        dq, dp = func(q, p, dt, t)
        q = q + dt * dq
        p = p + dt * dp
        t += dt


@numba.jit(nopython=True)
def leapfrog(q0, p0, dt, func, bc_func, out_q, out_p, t0):
    q = q0
    p = p0
    t = t0
    dqdt, dpdt = func(q, p, dt, t)
    for i in range(out_q.shape[0]):
        p_half = p + dpdt * (dt / 2)
        out_q[i] = q
        out_p[i] = p
        t += 0.5 * dt
        dqdt, dpdt = func(q, p_half, dt, t)
        q_next = q + dqdt * dt
        t += 0.5 * dt
        dqdt, dpdt = func(q_next, p_half, dt, t)
        p_next = p_half + dpdt * (dt / 2)
        p = p_next
        q = q_next


@numba.jit(nopython=True)
def rk4(q0, p0, dt, func, bc_func, out_q, out_p, t0):
    q = q0
    p = p0
    t = t0
    for i in range(out_q.shape[0]):
        out_q[i] = q
        out_p[i] = p
        q_k1, p_k1 = func(q, p, dt, t)
        q_k2, p_k2 = func(q + 0.5*dt*q_k1, p + 0.5*dt*p_k1, dt, t + 0.5*dt)
        q_k3, p_k3 = func(q + 0.5*dt*q_k2, p + 0.5*dt*p_k2, dt, t + 0.5*dt)
        q_k4, p_k4 = func(q + dt*q_k3, p + dt*p_k3, dt, t + dt)
        p_next = p + (1./6.) * dt * (p_k1 + 2 * p_k2 + 2 * p_k3 + p_k4)
        q_next = q + (1./6.) * dt * (q_k1 + 2 * q_k2 + 2 * q_k3 + q_k4)
        t += dt
        p = p_next
        q = q_next


def null_integrator(q0, p0, dt, func, bc_func, out_q, out_p, t0):
    q = q0
    p = p0
    t = t0
    for i in range(out_q.shape[0]):
        out_q[i] = q
        out_p[i] = p
        q, p = func(q, p, dt, t)
        t += dt

        # Reset boundary conditions.
        q, p = bc_func(q, p, t)


@numba.jit(nopython=True)
def backward_euler(x0, dt, func, bc_func, out_x, deriv_mat, t0):
    x = x0
    deriv_eye = np.eye(x.shape[-1], dtype=x0.dtype)
    unknown_mat = np.expand_dims(deriv_eye - dt * deriv_mat, 0)
    for i in range(out_x.shape[0]):
        out_x[i] = x
        x = np.linalg.solve(unknown_mat, x)


@numba.jit(nopython=True)
def bdf_2(x0, dt, func, bc_func, out_x, deriv_mat, t0):
    x = x0
    deriv_eye = np.eye(x.shape[-1], dtype=x0.dtype)

    # Perform one step of backward euler to get second point.
    # Since backward euler is second order in one step error,
    # this should not change the order of the whole method.
    unknown_mat = np.expand_dims(deriv_eye - dt * deriv_mat, 0)
    x_prev = x.copy()
    out_x[0] = x
    x = np.linalg.solve(unknown_mat, x)

    unknown_mat = np.expand_dims(deriv_eye - (2/3) * dt * deriv_mat, 0)
    for i in range(1, out_x.shape[0]):
        out_x[i] = x
        tmp = x.copy()
        x = np.linalg.solve(unknown_mat, - (1/3) * x_prev + (4/3) * x)
        x_prev = tmp


INTEGRATORS = {
    "euler": (euler, None),
    "leapfrog": (leapfrog, None),
    "rk4": (rk4, None),
    "null": (null_integrator, None),
    "back-euler": (backward_euler, "back_euler"),
    "bdf-2": (bdf_2, "bdf_2"),
}


def numerically_integrate(integrator, q0, p0, num_steps, dt, deriv_func, system=None, boundary_cond_func=None, t0=0.0):
    if boundary_cond_func is None:
        boundary_cond_func = no_boundary_condition
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
        out_shape = (num_steps, p0.shape[-1])
        out_q = np.empty_like(q0, shape=out_shape)
        out_p = np.empty_like(p0, shape=out_shape)
        getattr(system, implicit_attr)(q0, p0, dt, out_q, out_p)
    elif implicit_attr:
        # This is an implicit integrator, set up the extra context
        # In this case, we require `system` to be provided
        x0 = system.implicit_matrix_package(q=q0, p=p0)
        out_shape = (num_steps, x0.shape[-1])
        out_x = np.empty_like(q0, shape=out_shape)
        deriv_mat = system.implicit_matrix(x0)
        int_func(x0, dt, deriv_func, boundary_cond_func, out_x, deriv_mat, t0)
        ret_split = system.implicit_matrix_unpackage(out_x)
        out_q = ret_split.q
        out_p = ret_split.p
    else:
        # Allocate output array
        out_shape_q = (num_steps, q0.shape[-1])
        out_shape_p = (num_steps, p0.shape[-1])
        out_q = np.empty_like(q0, shape=out_shape_q)
        out_p = np.empty_like(p0, shape=out_shape_p)
        int_func(q0, p0, dt, deriv_func, boundary_cond_func, out_q, out_p, t0)
    return IntegrationResult(q=out_q, p=out_p)
