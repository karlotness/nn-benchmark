import numpy as np
import integrators
from systems import spring_mesh, wave, spring
from run_generators import utils as run_utils
import torch
from collections import namedtuple
import itertools
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_boolean("verbose", False, "Whether to print verbose output of errors per timestep.")

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

    sm_sys = spring_mesh.system_from_records(n_dims=2, particles=init_cond["particles"], edges=init_cond["springs"], vel_decay=init_cond["vel_decay"])

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

def setup_spring():
    train_source = run_utils.SpringInitialConditionSource(radius_range=(0.2, 1))
    init_cond = train_source.sample_initial_conditions(1)[0]["initial_condition"]

    sm_sys = spring.SpringSystem()

    sm_end = 2 * np.pi

    sm_dts = 2.**-np.arange(2, 14)

    dim = (1, )

    return SystemInitialization(
        system=sm_sys,
        q0=np.array([init_cond["q"]]),
        p0=np.array([init_cond["p"]]),
        end_time=sm_end,
        dt_array=sm_dts,
        dim=dim)


def analyze_system(system_name, system):
    print(f"System: {system_name}")
    trajs = []
    for dt in system.dt_array:
        steps = int(np.ceil(system.end_time / dt))
        traj = system.system.generate_trajectory(q0=system.q0, p0=system.p0, num_time_steps=steps, time_step_size=dt, subsample=1)
        trajs.append(traj.q.reshape((-1, *system.dim)))

    convergence = []
    smallest_error = None
    print("Ground Truth Data")
    for i in range(1, len(trajs)):
        subsample_factor = 2
        error = np.linalg.norm(trajs[i][::subsample_factor, ...][:trajs[i-1].shape[0]] - trajs[i-1], axis=-1).mean()
        if FLAGS.verbose:
            print(f"{system.dt_array[i-1]:.5f} -> {system.dt_array[i]:.5f}\t{error}")
        smallest_error = error
    if not FLAGS.verbose:
        print(f"{system.dt_array[-2]:.5f} -> {system.dt_array[-1]:.5f}\t{smallest_error}")


    SystemDerivative = namedtuple("SystemDerivative", ["dq_dt", "dp_dt"])
    physical_system = system.system
    eval_dtype = torch.double
    device = torch.device("cpu")
    def system_derivative(p, q, dt):
        derivative = physical_system.derivative(p=p, q=q)
        dp_dt, dq_dt = derivative
        return SystemDerivative(dp_dt=dp_dt, dq_dt=dq_dt)
    time_deriv_func = system_derivative

    all_integrators = (
        "euler",
        "rk4",
        "leapfrog",
        "back-euler",
        "implicit-rk",
    )

    for integrator_name in all_integrators:
        print(f"integrator: {integrator_name}")
        method_failed = False
        for i, dt in enumerate(system.dt_array):
            steps = int(np.ceil(system.end_time / dt))

            int_q0 = system.q0.reshape((1, -1,))
            int_p0 = system.p0.reshape((1, -1,))

            try:
                int_traj = integrators.numerically_integrate(integrator=integrator_name, p0=int_p0, q0=int_q0, num_steps=steps, dt=dt, deriv_func=time_deriv_func, system=system.system)
            except Exception as e:
                print("The implicit matrix is probably not implemented.")
                print(e)
                method_failed = True
                break

            int_q = int_traj.q.reshape((-1, *system.dim)) # .detach().cpu().numpy()

            factor = int(np.ceil(trajs[-1].shape[0] / int_q.shape[0]))
            error = np.linalg.norm(int_q - trajs[-1][::factor, ...], axis=-1).mean()

            if FLAGS.verbose:
                print(f"{dt:.5f}\t{error}")

            # Save for non verbose printing below
            dt_curr = dt
            error_curr = error

            if np.allclose(smallest_error, error, rtol=0.2) or (error < smallest_error):
                break

        if not FLAGS.verbose and not method_failed:
            print(f"{dt_curr:.5f}\t{error_curr}")


SYSTEMS = {
    "spring": setup_spring,
    "wave": setup_wave,
    "spring-mesh": setup_spring_mesh,
}


def main(argv):
    for k, v in SYSTEMS.items():
        analyze_system(k, v())


for k, v in SYSTEMS.items():
    if __name__ == "__main__":
        app.run(main)
