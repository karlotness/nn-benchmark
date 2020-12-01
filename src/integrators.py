from collections import namedtuple
import torch
from torch.autograd import grad
import enum
from scipy import integrate
import numpy as np

IntegrationResult = namedtuple("IntegrationResult", ["q", "p"])


class IntegrationScheme(enum.IntEnum):
    DIRECT_OUTPUT = 1
    HAMILTONIAN = 5


def leapfrog(p_0, q_0, Func, T, dt, volatile=True, is_Hamilt=True, device='cpu'):

    trajectories = torch.empty((T, p_0.shape[0], 2 * p_0.shape[1]), requires_grad=False).to(device)

    p = p_0
    q = q_0
    p.requires_grad_()
    q.requires_grad_()

    range_of_for_loop = range(T)

    if is_Hamilt:
        hamilt = Func(p=p, q=q)
        dpdt = -grad(hamilt.sum(), q, create_graph=not volatile)[0]

        for i in range_of_for_loop:
            p_half = p + dpdt * (dt / 2)

            if volatile:
                trajectories[i, :, :p_0.shape[1]] = p.detach()
                trajectories[i, :, p_0.shape[1]:] = q.detach()
            else:
                trajectories[i, :, :p_0.shape[1]] = p
                trajectories[i, :, p_0.shape[1]:] = q

            hamilt = Func(p=p_half, q=q)
            dqdt = grad(hamilt.sum(), p, create_graph=not volatile)[0]

            q_next = q + dqdt * dt

            hamilt = Func(p=p_half, q=q_next)
            dpdt = -grad(hamilt.sum(), q_next, create_graph=not volatile)[0]

            p_next = p_half + dpdt * (dt / 2)

            p = p_next
            q = q_next

    else:
        dim = p_0.shape[1]
        time_drvt = Func(q=q, p=p)
        dpdt = time_drvt.dp_dt

        for i in range_of_for_loop:
            p_half = p + dpdt * (dt / 2)

            if volatile:
                trajectories[i, :, :dim] = p.detach()
                trajectories[i, :, dim:] = q.detach()
            else:
                trajectories[i, :, :dim] = p
                trajectories[i, :, dim:] = q

            time_drvt = Func(p=p_half, q=q)
            dqdt = time_drvt.dq_dt

            q_next = q + dqdt * dt

            time_drvt = Func(p=p_half, q=q_next)
            dpdt = time_drvt.dp_dt

            p_next = p_half + dpdt * (dt / 2)

            p = p_next
            q = q_next

    trajectories = trajectories.permute(1, 0, 2)
    n = p_0.shape[1]
    ret_p = trajectories[:, :, :n]
    ret_q = trajectories[:, :, n:]
    return IntegrationResult(q=ret_q, p=ret_p)


def euler(p_0, q_0, Func, T, dt, volatile=True, is_Hamilt=True, device='cpu'):

    trajectories = torch.empty((T, p_0.shape[0], 2 * p_0.shape[1]), requires_grad=False).to(device)

    p = p_0
    q = q_0
    p.requires_grad_()
    q.requires_grad_()

    range_of_for_loop = range(T)

    if is_Hamilt:

        for i in range_of_for_loop:

            if volatile:
                trajectories[i, :, :p_0.shape[1]] = p.detach()
                trajectories[i, :, p_0.shape[1]:] = q.detach()
            else:
                trajectories[i, :, :p_0.shape[1]] = p
                trajectories[i, :, p_0.shape[1]:] = q

            hamilt = Func(p=p, q=q)
            dpdt = -grad(hamilt.sum(), q, create_graph=not volatile)[0]
            dqdt = grad(hamilt.sum(), p, create_graph=not volatile)[0]

            p_next = p + dpdt * dt
            q_next = q + dqdt * dt

            p = p_next
            q = q_next

    else:
        dim = p_0.shape[1]

        for i in range_of_for_loop:

            if volatile:
                trajectories[i, :, :dim] = p.detach()
                trajectories[i, :, dim:] = q.detach()
            else:
                trajectories[i, :, :dim] = p
                trajectories[i, :, dim:] = q

            time_drvt = Func(p=p, q=q)
            dpdt = time_drvt.dp_dt
            dqdt = time_drvt.dq_dt

            p_next = p + dpdt * dt
            q_next = q + dqdt * dt

            p = p_next
            q = q_next

    trajectories = trajectories.permute(1, 0, 2)
    n = p_0.shape[1]
    ret_p = trajectories[:, :, :n]
    ret_q = trajectories[:, :, n:]
    return IntegrationResult(q=ret_q, p=ret_p)


def rk4(p_0, q_0, Func, T, dt, volatile=True, is_Hamilt=True, device='cpu'):

    trajectories = torch.empty((T, p_0.shape[0], 2 * p_0.shape[1]), requires_grad=False).to(device)

    p = p_0
    q = q_0
    p.requires_grad_()
    q.requires_grad_()

    range_of_for_loop = range(T)

    if is_Hamilt:

        for i in range_of_for_loop:

            if volatile:
                trajectories[i, :, :p_0.shape[1]] = p.detach()
                trajectories[i, :, p_0.shape[1]:] = q.detach()
            else:
                trajectories[i, :, :p_0.shape[1]] = p
                trajectories[i, :, p_0.shape[1]:] = q

            def dpdt_dqdt(p_, q_):
                hamilt = Func(p=p_, q=q_)
                dpdt = -grad(hamilt.sum(), q_, create_graph=not volatile)[0]
                dqdt = grad(hamilt.sum(), p_, create_graph=not volatile)[0]
                return dpdt, dqdt

            p_k1, q_k1 = dpdt_dqdt(p, q)
            p_k2, q_k2 = dpdt_dqdt(p + 0.5*dt*p_k1, q + 0.5*dt*q_k1)
            p_k3, q_k3 = dpdt_dqdt(p + 0.5*dt*p_k2, q + 0.5*dt*q_k2)
            p_k4, q_k4 = dpdt_dqdt(p + dt*p_k3, q + dt*q_k3)

            p_next = p + (1./6.) * dt * (p_k1 + 2 * p_k2 + 3 * p_k3 + p_k4)
            q_next = q + (1./6.) * dt * (q_k1 + 2 * q_k2 + 3 * q_k3 + q_k4)

            p = p_next
            q = q_next

    else:
        dim = p_0.shape[1]

        for i in range_of_for_loop:

            if volatile:
                trajectories[i, :, :dim] = p.detach()
                trajectories[i, :, dim:] = q.detach()
            else:
                trajectories[i, :, :dim] = p
                trajectories[i, :, dim:] = q

            def dpdt_dqdt(p_, q_):
                time_drvt = Func(p=p_, q=q_)
                dpdt = time_drvt.dp_dt
                dqdt = time_drvt.dq_dt
                return dpdt, dqdt

            p_k1, q_k1 = dpdt_dqdt(p, q)
            p_k2, q_k2 = dpdt_dqdt(p + 0.5*dt*p_k1, q + 0.5*dt*q_k1)
            p_k3, q_k3 = dpdt_dqdt(p + 0.5*dt*p_k2, q + 0.5*dt*q_k2)
            p_k4, q_k4 = dpdt_dqdt(p + dt*p_k3, q + dt*q_k3)

            p_next = p + (1./6.) * dt * (p_k1 + 2 * p_k2 + 3 * p_k3 + p_k4)
            q_next = q + (1./6.) * dt * (q_k1 + 2 * q_k2 + 3 * q_k3 + q_k4)

            p = p_next
            q = q_next

    trajectories = trajectories.permute(1, 0, 2)
    n = p_0.shape[1]
    ret_p = trajectories[:, :, :n]
    ret_q = trajectories[:, :, n:]
    return IntegrationResult(q=ret_q, p=ret_p)


def null_integrator(p_0, q_0, Func, T, dt, volatile=True, is_Hamilt=False, device='cpu'):
    # Integrator performs no actual integration, function provides next states
    if not volatile or is_Hamilt:
        raise ValueError("Null integrator cannot create graph and does not support Hamiltonian")

    p = p_0
    q = q_0

    trajectories = torch.empty((T, p_0.shape[0], 2 * p_0.shape[1]), requires_grad=False).to(device)

    for i in range(T):
        trajectories[i, :, :p_0.shape[1]] = p.detach()
        trajectories[i, :, p_0.shape[1]:] = q.detach()

        res = Func(p=p, q=q)

        if hasattr(res, "p"):
            p = res.p
            q = res.q
        else:
            p = res.dp_dt
            q = res.dq_dt

    trajectories = trajectories.permute(1, 0, 2)
    n = p_0.shape[1]
    ret_p = trajectories[:, :, :n]
    ret_q = trajectories[:, :, n:]
    return IntegrationResult(q=ret_q, p=ret_p)


def scipy_integrator(p_0, q_0, Func, T, dt, volatile=True, is_Hamilt=True, device='cpu', method="RK45"):
    if not volatile:
        raise ValueError("SciPy integrators cannot create graph")
    if len(p_0.shape) > 2 or (len(p_0.shape) > 1 and p_0.shape[0] != 1):
        raise ValueError("SciPy integrators do not support batching")

    dt = dt.item()
    T = T.item()
    torch_dtype = p_0.dtype
    p_0 = p_0.detach().cpu().numpy()
    q_0 = q_0.detach().cpu().numpy()

    def scipy_wrapper(_time, y):
        p, q = np.split(y, 2, axis=-1)
        p = torch.from_numpy(p).to(device, dtype=torch_dtype)
        q = torch.from_numpy(q).to(device, dtype=torch_dtype)
        if len(p.shape) < 2:
            p = p.unsqueeze(0)
            q = q.unsqueeze(0)
        if is_Hamilt:
            # Do backprop for hamiltonian
            p.requires_grad_()
            q.requires_grad_()
            hamilt = Func(p=p, q=q)
            dpdt = -grad(hamilt.sum(), q, create_graph=not volatile)[0]
            dqdt = grad(hamilt.sum(), p, create_graph=not volatile)[0]
        else:
            # Do direct integration
            deriv = Func(p=p, q=q)
            dpdt = deriv.dp_dt
            dqdt = deriv.dq_dt
        # Process return values
        dpdt = dpdt.detach().cpu().numpy()
        dqdt = dqdt.detach().cpu().numpy()
        if len(dpdt.shape) > 1:
            dpdt = dpdt[0]
            dqdt = dqdt[0]
        return np.concatenate((dpdt, dqdt), axis=-1)

    t_span = (0, dt * T)
    t_eval = np.arange(T).astype(np.float64) * dt
    y0 = np.concatenate((p_0, q_0), axis=-1)[0]

    ivp_res = integrate.solve_ivp(scipy_wrapper,
                                  t_span=t_span,
                                  y0=y0,
                                  method=method,
                                  t_eval=t_eval)

    # Read out results
    y = np.moveaxis(ivp_res["y"], 0, -1)
    ps, qs = np.split(y, 2, axis=-1)
    ps = torch.from_numpy(ps).to(device, dtype=torch_dtype).unsqueeze(0)
    qs = torch.from_numpy(qs).to(device, dtype=torch_dtype).unsqueeze(0)
    return IntegrationResult(q=qs, p=ps)


def numerically_integrate(integrator, p_0, q_0, model, method, T, dt, volatile, device, coarsening_factor=1):
    if (coarsening_factor > 1):
        fine_trajectory = numerically_integrate(integrator, p_0, q_0, model, method, T * coarsening_factor, dt / coarsening_factor, volatile, device)
        trajectory_simulated = fine_trajectory[np.arange(T) * coarsening_factor, :, :]
        return trajectory_simulated
    if (method == IntegrationScheme.HAMILTONIAN):
        if (integrator == 'leapfrog'):
            trajectory_simulated = leapfrog(p_0, q_0, model, T, dt, volatile=volatile, device=device)
        elif (integrator == 'euler'):
            trajectory_simulated = euler(p_0, q_0, model, T, dt, volatile=volatile, device=device)
        elif integrator.startswith("scipy-"):
            # Handle SciPy integration
            method = integrator.split("-")[1]
            trajectory_simulated = scipy_integrator(p_0, q_0, model, T, dt, volatile=volatile, device=device, method=method)
        else:
            raise ValueError(f"Unknown integrator {integrator}")
    elif (method == IntegrationScheme.DIRECT_OUTPUT):
        if (integrator == 'leapfrog'):
            trajectory_simulated = leapfrog(p_0, q_0, model, T, dt, volatile=volatile, is_Hamilt=False, device=device)
        elif (integrator == 'euler'):
            trajectory_simulated = euler(p_0, q_0, model, T, dt, volatile=volatile, is_Hamilt=False, device=device)
        elif integrator == 'null':
            trajectory_simulated = null_integrator(p_0, q_0, model, T, dt, volatile=volatile, is_Hamilt=False, device=device)
        elif integrator.startswith("scipy-"):
            # Handle SciPy integration
            method = integrator.split("-")[1]
            trajectory_simulated = scipy_integrator(p_0, q_0, model, T, dt, volatile=volatile, is_Hamilt=False, device=device, method=method)
        else:
            raise ValueError(f"Unknown integrator {integrator}")
    else:
        trajectory_simulated = model(torch.cat([p_0, q_0], dim=1), T)
    return trajectory_simulated
