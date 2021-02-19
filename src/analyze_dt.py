import numpy as np
import integrators
from systems import spring_mesh, wave
from run_generators import utils as run_utils
import torch
from collections import namedtuple
import matplotlib.pyplot as plt
import multiprocessing
import itertools

SystemInitialization = namedtuple("SystemInitialization", ["system", "q0", "p0", "end_time", "dt_array", "dim"])

def setup_spring_mesh(n=5):
    PERTURB_COORDS = [
        (0, 0),
        (4, 0),
        (0, 3),
        (4, 3),
        (2, 2),
    ]

    mesh_gen = run_utils.SpringMeshGridGenerator(grid_shape=(n, n), fix_particles="top")
    train_source = run_utils.SpringMeshInterpolatePerturb(mesh_generator=mesh_gen, coords=PERTURB_COORDS, magnitude_range=(0, 0.75))
    init_cond = train_source.sample_initial_conditions(1)[0]

    sm_sys = spring_mesh.system_from_records(n_dims=2, particles=init_cond["particles"], edges=init_cond["springs"])

    q0 = []
    particles = []
    edges = []
    for pdef in init_cond["particles"]:
        q0.append(np.array(pdef["position"]))
    q0 = np.stack(q0).astype(np.float64)
    p0 = np.zeros_like(q0)

    sm_end = 2 * np.pi

    sm_dts = 2.**-np.arange(2, 14)

    dim = (n**2, 2, )

    return SystemInitialization(
        system=sm_sys,
        q0=q0,
        p0=p0,
        end_time=sm_end,
        dt_array=sm_dts,
        dim=dim)


def setup_wave(n_grid=125):
    train_source = run_utils.WaveInitialConditionSource(
        height_range=(0.75, 1.25), width_range=(0.75, 1.25), position_range=(0.5, 0.5))
    init_cond = wave.generate_cubic_spline_start(
        space_max=1, n_grid=n_grid, start_type_args=train_source.sample_initial_conditions(1)[0]["start_type_args"])

    sm_sys = wave.WaveSystem(n_grid=n_grid, space_max=1, wave_speed=0.1)

    sm_end = 5

    sm_dts = 2.**-np.arange(2, 14)

    dim = (n_grid, )

    return SystemInitialization(
        system=sm_sys,
        q0=init_cond.q,
        p0=init_cond.p,
        end_time=sm_end,
        dt_array=sm_dts,
        dim=dim)

SYSTEMS = {
    "wave": setup_wave,
    "spring-mesh": setup_spring_mesh,
}


for k, v in SYSTEMS.items():
    print(f"System: {k}")
    v = v()
    trajs = []
    for dt in v.dt_array:
        steps = int(np.ceil(v.end_time / dt))
        traj = v.system.generate_trajectory(q0=v.q0, p0=v.p0, num_time_steps=steps, time_step_size=dt, subsample=1)
        trajs.append(traj)

    convergence = []
    smallest_error = None
    print("Ground Truth Data")
    for i in range(1, len(trajs)):
        subsample_factor = 2
        error = np.linalg.norm(trajs[i].q[::subsample_factor, ...][:trajs[i-1].q.shape[0]] - trajs[i-1].q, axis=-1).mean()
        print(f"{np.round(2*np.pi/trajs[i-1].q.shape[0], 5)} -> {np.round(2*np.pi/trajs[i].q.shape[0], 5)}\t{error}")
        smallest_error = error


    SystemDerivative = namedtuple("SystemDerivative", ["dq_dt", "dp_dt"])
    system = v.system
    eval_dtype = torch.double
    device = torch.device("cpu")
    def system_derivative(p, q, dt=1.0):
        with torch.no_grad():
            if torch.is_tensor(dt):
                dt = dt.item()
            p = p.detach().cpu().numpy()
            q = q.detach().cpu().numpy()
            derivative = system.derivative(p=p, q=q, dt=dt)
            dp_dt = torch.from_numpy(derivative.p).to(device, dtype=eval_dtype)
            dq_dt = torch.from_numpy(derivative.q).to(device, dtype=eval_dtype)
            return SystemDerivative(dp_dt=dp_dt, dq_dt=dq_dt)
    time_deriv_func = system_derivative
    time_deriv_method = integrators.IntegrationScheme.DIRECT_OUTPUT

    for integrator in [integrators.euler, integrators.rk4, integrators.leapfrog]:
        print("integrator: ", integrator)
        for i, dt in enumerate(v.dt_array):
            steps = int(np.ceil(v.end_time / dt))

            int_q0 = torch.from_numpy(v.q0.reshape((1, -1,)))
            int_p0 = torch.from_numpy(v.p0.reshape((1, -1,)))

            euler_traj = integrator(p_0=int_p0, q_0=int_q0, Func=system_derivative, T=steps, dt=dt, volatile=True, is_Hamilt=False)
            euler_q = euler_traj.q.reshape((-1, *v.dim)).detach().cpu().numpy()
            euler_p = euler_traj.p.reshape((-1, *v.dim)).detach().cpu().numpy()

            factor = int(np.ceil(trajs[-1].q.shape[0] / euler_q.shape[0]))
            error = np.linalg.norm(euler_q - trajs[-1].q[::factor, ...], axis=-1).mean()

            print(f"{np.round(dt, 5)}\t{error}")

            if np.allclose(smallest_error, error, rtol=0.2) or (error < smallest_error):
                break

