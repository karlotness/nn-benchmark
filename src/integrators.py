import torch
from torch.autograd import grad


def leapfrog(p_0, q_0, Func, T, dt, volatile=True, is_Hamilt=True, device='cpu'):

    trajectories = torch.empty((T, p_0.shape[0], 2 * p_0.shape[1]), requires_grad=False).to(device)

    p = p_0
    q = q_0
    p.requires_grad_()
    q.requires_grad_()

    range_of_for_loop = range(T)

    if is_Hamilt:
        hamilt = Func(p, q)
        dpdt = -grad(hamilt.sum(), q, create_graph=not volatile)[0]

        for i in range_of_for_loop:
            p_half = p + dpdt * (dt / 2)

            if volatile:
                trajectories[i, :, :p_0.shape[1]] = p.detach()
                trajectories[i, :, p_0.shape[1]:] = q.detach()
            else:
                trajectories[i, :, :p_0.shape[1]] = p
                trajectories[i, :, p_0.shape[1]:] = q

            hamilt = Func(p_half, q)
            dqdt = grad(hamilt.sum(), p, create_graph=not volatile)[0]

            q_next = q + dqdt * dt

            hamilt = Func(p_half, q_next)
            dpdt = -grad(hamilt.sum(), q_next, create_graph=not volatile)[0]

            p_next = p_half + dpdt * (dt / 2)

            p = p_next
            q = q_next

    else:
        dim = p_0.shape[1]
        time_drvt = Func(torch.cat((p, q), 1))
        dpdt = time_drvt[:, :dim]

        for i in range_of_for_loop:
            p_half = p + dpdt * (dt / 2)

            if volatile:
                trajectories[i, :, :dim] = p.detach()
                trajectories[i, :, dim:] = q.detach()
            else:
                trajectories[i, :, :dim] = p
                trajectories[i, :, dim:] = q

            time_drvt = Func(torch.cat((p_half, q), 1))
            dqdt = time_drvt[:, dim:]

            q_next = q + dqdt * dt

            time_drvt = Func(torch.cat((p_half, q_next), 1))
            dpdt = time_drvt[:, :dim]

            p_next = p_half + dpdt * (dt / 2)

            p = p_next
            q = q_next

    return trajectories


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

            hamilt = Func(p, q)
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

            time_drvt = Func(torch.cat((p, q), 1))
            dpdt = time_drvt[:, :dim]
            dqdt = time_drvt[:, dim:]

            p_next = p + dpdt * dt
            q_next = q + dqdt * dt

            p = p_next
            q = q_next

    return trajectories


def numerically_integrate(integrator, p_0, q_0, model, method, T, dt, volatile, device, coarsening_factor=1):
    if (coarsening_factor > 1):
        fine_trajectory = numerically_integrate(integrator, p_0, q_0, model, method, T * coarsening_factor, dt / coarsening_factor, volatile, device)
        trajectory_simulated = fine_trajectory[np.arange(T) * coarsening_factor, :, :]
        return trajectory_simulated
    if (method == 5):
        if (integrator == 'leapfrog'):
            trajectory_simulated = leapfrog(p_0, q_0, model, T, dt, volatile=volatile, device=device)
        elif (integrator == 'euler'):
            trajectory_simulated = euler(p_0, q_0, model, T, dt, volatile=volatile, device=device)
    elif (method == 1):
        if (integrator == 'leapfrog'):
            trajectory_simulated = leapfrog(p_0, q_0, model, T, dt, volatile=volatile, is_Hamilt=False, device=device)
        elif (integrator == 'euler'):
            trajectory_simulated = euler(p_0, q_0, model, T, dt, volatile=volatile, is_Hamilt=False, device=device)
    else:
        trajectory_simulated = model(torch.cat([p_0, q_0], dim=1), T)
    return trajectory_simulated
