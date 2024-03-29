import logging
import codecs
import subprocess
import json
import os
import pathlib
import shutil
import tempfile
import lzma
import time
from collections import namedtuple
from .defs import System, SystemCache, SystemResult
import numpy as np
import concurrent.futures
from numba import jit
import triangle
import dataclasses
import tarfile
import io


IN_MESH_PATH = pathlib.Path(__file__).parent / "resources" / "mesh.obj.xz"
OBSTACLE_STEPS = 18
navier_stokes_cache = SystemCache()
NavierStokesTrajectoryResult = namedtuple("NavierStokesTrajectoryResult",
                                          ["grids",
                                           "solutions",
                                           "grads",
                                           "pressures",
                                           "pressures_grads",
                                           "t",
                                           "fixed_mask",
                                           "fixed_mask_pressures",
                                           "fixed_mask_solutions",
                                           "extra_fixed_mask",
                                           "enumerated_fixed_mask",
                                           "log_compressed",
                                           ])
MeshDefinition = dataclasses.make_dataclass("MeshDefinition", ["radius", "center"])

MESH_SIZE = (221, 42)


class NavierStokesSystem(System):
    def __init__(self, grid_resolution=0.01, viscosity=0.001):
        super().__init__()
        self.grid_resolution = grid_resolution
        self.viscosity = viscosity

    def _args_compatible(self, grid_resolution, viscosity):
        return (self.grid_resolution == grid_resolution and
                self.viscosity == viscosity)

    def hamiltonian(self, q, p):
        return -1 * np.ones_like(q, shape=(q.shape[0]))

    def _gen_config(self, mesh_file, num_time_steps, time_step_size, in_velocity):
        t_end = (num_time_steps + 1) * time_step_size
        return {
            "mesh": mesh_file,
            "normalize_mesh": False,
            "has_neumann": True,
            "discr_order": 2,
            "pressure_discr_order": 1,
            "BDF_order": 3,
            "n_refs": 0,
            "tend": t_end,
            "time_steps": num_time_steps + 1,
            "vismesh_rel_area": 0.0001,
            "problem_params": {
                "U": in_velocity,
                "time_dependent": True,
            },
            "params": {
                "viscosity": self.viscosity,
            },
            "solver_params": {
                "gradNorm": 1e-8,
                "nl_iterations": 20,
            },
            "export": {
                "sol_on_grid": self.grid_resolution,
            },
            "problem": "FlowWithObstacle",
            "tensor_formulation": "NavierStokes",
        }

    @staticmethod
    def _replace_nan(mat):
        mat[np.isnan(mat)] = 0.0
        return mat

    def _generate_obstacle(self, pos, r, n_steps, trafo=None, edge_skew=4):
        if trafo is None:
            trafo = np.eye(2)
        return (np.array([ [np.cos(2*np.pi*i/n_steps), np.sin(2*np.pi*i/n_steps)] for i in range(n_steps)])*r@trafo + pos,
                np.array([ [i+edge_skew, (i+1)%n_steps+edge_skew] for i in range(n_steps)]))

    def _generate_mesh(self, mesh_args):
        if mesh_args is None:
            # Use the old static mesh
            with lzma.open(IN_MESH_PATH, "rt", encoding="utf8") as in_mesh_file:
                return in_mesh_file.read()
        # Generate a dynamic mesh according to the input parameters
        # Generate the boundary points and edges
        square = np.array([[0, 0], [2.2, 0], [2.2, 0.41], [0, 0.41]], dtype=np.float64)
        square_e = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
        pts = [square]
        edges = [square_e]
        centers = []
        # Generate obstacle points and edges
        for mesh_arg in mesh_args:
            edge_skew = np.max(edges[-1]) + 1
            center = np.array(mesh_arg.center).reshape((1, 2))
            radius = mesh_arg.radius
            obst, obst_e = self._generate_obstacle(pos=center, r=radius, n_steps=OBSTACLE_STEPS, trafo=None, edge_skew=edge_skew)
            pts.append(obst)
            edges.append(obst_e)
            centers.append(center)
        # Combine obstacle with boundary
        pts = np.concatenate(pts)
        edges = np.concatenate(edges)
        centers = np.concatenate(centers)
        # Triangulate
        t = triangle.triangulate({"vertices": pts, "segments": edges, "holes": centers}, "qpa0.0003")
        v = t['vertices']
        f = t['triangles']
        # Produce mesh string
        str_segments = []
        for p in v:
            str_segments.append(f"v {p[0]} {p[1]} 0\n")
        for t in f:
            str_segments.append(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")
        return "".join(str_segments)

    def _find_polyfem(self):
        # Check POLYFEM_BIN_DIR (if set) and PATH
        prog = shutil.which(str(pathlib.Path(os.environ.get("POLYFEM_BIN_DIR", "")) / "PolyFEM_bin"))
        if prog:
            return prog
        # Check current directory
        prog = pathlib.Path("PolyFEM_bin")
        if prog.is_file():
            return str(prog.resolve())
        return None

    def generate_trajectory(self, num_time_steps, time_step_size, in_velocity, subsample=1, mesh_args=None, traj_name=""):
        orig_time_step_size = time_step_size
        orig_num_time_steps = num_time_steps

        time_step_size = time_step_size / subsample
        num_time_steps = num_time_steps * subsample

        # Create temporary directory
        with tempfile.TemporaryDirectory(prefix=f"polyfem-{traj_name}-") as _tmp_dir:
            tmp_dir = pathlib.Path(_tmp_dir)

            # Write the mesh file
            with open(tmp_dir / "mesh.obj", "w", encoding="utf8") as mesh_file:
                mesh_file.write(self._generate_mesh(mesh_args))

            # Write the config file
            with open(tmp_dir / "config.json", "w", encoding="utf8") as config_file:
                config = self._gen_config(
                    mesh_file="mesh.obj",
                    num_time_steps=num_time_steps,
                    time_step_size=time_step_size,
                    in_velocity=in_velocity,
                )
                json.dump(config, config_file)

            # Run PolyFEM
            prog = self._find_polyfem()
            if not prog:
                raise ValueError("Cannot find PolyFEM; please install and set POLYFEM_BIN_DIR")
            cmdline = [prog, "--json", "config.json", "--cmd"]
            new_env = dict(os.environ)
            if "OMP_NUM_THREADS" not in new_env:
                new_env["OMP_NUM_THREADS"] = "2"
            polyfem_log_path = tmp_dir / "polyfem_run.log"
            with open(polyfem_log_path, "wb") as polyfem_log_file:
                proc = subprocess.run(cmdline, stdout=polyfem_log_file, env=new_env, cwd=tmp_dir, check=True)

            # Load log text
            with open(polyfem_log_path, "rb") as polyfem_log_file:
                log_compressed = lzma.compress(polyfem_log_file.read())

            # Gather results
            grids = []
            solutions = []
            pressures = []
            grads = []
            pressures_grads = []
            ts = []
            for i in range(subsample, num_time_steps + 1, subsample):
                i_next = i + 1
                grids.append(np.loadtxt(tmp_dir / f"step_{i}.vtu_grid.txt"))
                solutions.append(np.loadtxt(tmp_dir / f"step_{i}.vtu_sol.txt"))
                pressures.append(np.loadtxt(tmp_dir / f"step_{i}.vtu_p_sol.txt"))

                _grad = (1/time_step_size) * (np.loadtxt(tmp_dir / f"step_{i_next}.vtu_sol.txt") - np.loadtxt(tmp_dir / f"step_{i}.vtu_sol.txt"))
                _press_grad = (1/time_step_size) * (np.loadtxt(tmp_dir / f"step_{i_next}.vtu_p_sol.txt") - np.loadtxt(tmp_dir / f"step_{i}.vtu_p_sol.txt"))
                grads.append(_grad)
                pressures_grads.append(_press_grad)
                ts.append(i * time_step_size)

        grids = np.stack(grids)
        solutions = np.stack(solutions)
        pressures = np.stack(pressures)
        grads = np.stack(grads)
        pressures_grads = np.stack(pressures_grads)

        obstacle_mask = np.any(np.isnan(solutions[0]), axis=1)

        fixed_mask = obstacle_mask.copy().reshape(MESH_SIZE)
        fixed_mask[:, 0] = True
        fixed_mask[:, -1] = True
        fixed_mask[0, :] = True

        fixed_mask_pressures = obstacle_mask.copy().reshape(pressures.shape[1:])
        fixed_mask_solutions = np.tile(np.expand_dims(fixed_mask, axis=-1), (1, 1, 2)).reshape(solutions.shape[1:])
        fixed_mask = fixed_mask.reshape(obstacle_mask.shape)
        extra_fixed_mask = np.stack([fixed_mask, obstacle_mask], axis=-1)
        enumerated_fixed_mask = obstacle_mask.copy().reshape(MESH_SIZE).astype(np.int32)
        enumerated_fixed_mask[:, 0] = 1
        enumerated_fixed_mask[:, -1] = 1
        enumerated_fixed_mask[0, :] = 2
        enumerated_fixed_mask[-1, :] = 3
        enumerated_fixed_mask = enumerated_fixed_mask.reshape(obstacle_mask.shape)

        # Repackage results
        t = np.array(ts)
        return NavierStokesTrajectoryResult(
            grids=grids,
            solutions=self._replace_nan(solutions),
            grads=self._replace_nan(grads),
            pressures=self._replace_nan(pressures),
            pressures_grads=self._replace_nan(pressures_grads),
            t=t,
            fixed_mask=fixed_mask,
            fixed_mask_pressures=fixed_mask_pressures,
            fixed_mask_solutions=fixed_mask_solutions,
            extra_fixed_mask=extra_fixed_mask,
            enumerated_fixed_mask=enumerated_fixed_mask,
            log_compressed=log_compressed,
        )


def make_enforce_boundary_function(in_velocity, vertex_coords, fixed_mask_solutions, fixed_mask_pressures):
    vertex_y = vertex_coords.reshape((MESH_SIZE + (2, )))[0, :, 1]
    vertex_y.setflags(write=False)
    fixed_mask_solutions = fixed_mask_solutions.reshape((-1, ))
    fixed_mask_pressures = fixed_mask_pressures.reshape((-1, ))

    left_boundary_indexing = np.zeros(MESH_SIZE + (2, ), dtype=bool)
    left_boundary_indexing[0, :, 0] = True
    left_boundary_indexing = left_boundary_indexing.reshape((-1, ))
    left_boundary_indexing.setflags(write=False)

    @jit(nopython=True)
    def get_left_boundary(t):
        return in_velocity * 4 * vertex_y * (0.41 - vertex_y) / (0.41 * 0.41) * (1 - np.exp(-5 * t))

    @jit(nopython=True)
    def ns_boundary_condition(q, p, t):
        left_boundary = np.expand_dims(get_left_boundary(t), axis=0)
        q = q.copy()
        p = p.copy()
        q[:, fixed_mask_pressures] = 0
        p[:, fixed_mask_solutions] = 0
        p[:, left_boundary_indexing] = left_boundary
        return q, p

    return ns_boundary_condition


def compute_edge_indices(vtx_coords):
    # Determine grid size
    # Grid indexing here referred to as [a, b]
    max_coords = np.max(vtx_coords, axis=0)
    num_b = int(max_coords[-1] / vtx_coords[1, -1]) + 1
    num_a = vtx_coords.shape[0] // num_b
    assert num_a * num_b == vtx_coords.shape[0]
    edges = []
    for a in range(num_a):
        for b in range(num_b):
            curr_idx = num_b * a + b
            for da, db in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                na = a + da
                nb = b + db
                if not (0 <= na < num_a and 0 <= nb < num_b):
                    continue
                next_idx = num_b * na + nb
                edges.append((curr_idx, next_idx))
    edges = np.array(edges, dtype=np.int64)

    # Check the results
    db = vtx_coords[1, -1] - vtx_coords[0, -1]
    da = vtx_coords[num_b, 0] - vtx_coords[0, 0]
    max_diff = 1.1 * np.expand_dims(np.array([da, db], dtype=np.float64), 0)
    assert np.all(np.abs(vtx_coords[edges[:, 0]] - vtx_coords[edges[:, 1]]) <= max_diff)

    return edges.T


def system_from_records(grid_resolution, viscosity):
    cached_sys = navier_stokes_cache.find(
        grid_resolution=grid_resolution,
        viscosity=viscosity,
    )
    if cached_sys is not None:
        return cached_sys
    else:
        new_sys = NavierStokesSystem(
            grid_resolution=grid_resolution,
            viscosity=viscosity,
        )
        navier_stokes_cache.insert(new_sys)
        return new_sys


def _generate_data_worker(i, traj_def, grid_resolution):
    traj_name = f"traj_{i:05}"
    base_logger = logging.getLogger("navier-stokes")
    base_logger.info(f"Generating trajectory {traj_name}")

    # Get trajectory definitions
    num_time_steps = traj_def["num_time_steps"]
    time_step_size = traj_def["time_step_size"]
    in_velocity = traj_def["in_velocity"]
    viscosity = traj_def.get("viscosity", 0.001)
    subsample = int(traj_def.get("subsample", 1))

    if "mesh" in traj_def:
        raw_mesh_args = traj_def["mesh"]
        if not isinstance(raw_mesh_args, list):
            raw_mesh_args = [raw_mesh_args]
        mesh_args = []
        for mesh_arg in raw_mesh_args:
            mesh_args.append(
                MeshDefinition(
                    radius=mesh_arg["radius"],
                    center=tuple(mesh_arg["center"]),
                )
            )
    else:
        mesh_args = None

    # Create system
    system = navier_stokes_cache.find(grid_resolution=grid_resolution, viscosity=viscosity)
    if system is None:
        system = NavierStokesSystem(grid_resolution=grid_resolution, viscosity=viscosity)
        navier_stokes_cache.insert(system)

    traj_gen_start = time.perf_counter()
    traj_result = system.generate_trajectory(
        num_time_steps=num_time_steps,
        time_step_size=time_step_size,
        in_velocity=in_velocity,
        subsample=subsample,
        mesh_args=mesh_args,
        traj_name=traj_name,
    )
    traj_gen_elapsed = time.perf_counter() - traj_gen_start
    base_logger.info(f"Generated {traj_name} in {traj_gen_elapsed} sec")

    trajectories_update = {
        f"{traj_name}_solutions": traj_result.solutions,
        f"{traj_name}_grads": traj_result.grads,
        f"{traj_name}_pressures": traj_result.pressures,
        f"{traj_name}_pressures_grads": traj_result.pressures_grads,
        f"{traj_name}_t": traj_result.t,
        # Fixed masks
        f"{traj_name}_fixed_mask": traj_result.fixed_mask,
        f"{traj_name}_extra_fixed_mask": traj_result.extra_fixed_mask,
        f"{traj_name}_fixed_mask_solutions": traj_result.fixed_mask_solutions,
        f"{traj_name}_fixed_mask_pressures": traj_result.fixed_mask_pressures,
        f"{traj_name}_enumerated_fixed_mask": traj_result.enumerated_fixed_mask,
    }

    one_time_results = {
        "vertices": traj_result.grids[0],
    }

    trajectory_metadata = {
        "name": traj_name,
        "num_time_steps": num_time_steps,
        "time_step_size": time_step_size,
        "in_velocity": in_velocity,
        "viscosity": viscosity,
        "noise_sigma": 0,
        "field_keys": {
            # Plain names
            "solutions": f"{traj_name}_solutions",
            "grads": f"{traj_name}_grads",
            "pressures": f"{traj_name}_pressures",
            "pressures_grads": f"{traj_name}_pressures_grads",
            # Mapped names
            "p": f"{traj_name}_solutions",
            "q": f"{traj_name}_pressures",
            "dpdt": f"{traj_name}_grads",
            "dqdt": f"{traj_name}_pressures_grads",
            "t": f"{traj_name}_t",
            "p_noiseless": f"{traj_name}_solutions",
            "q_noiseless": f"{traj_name}_pressures",
            "fixed_mask_p": f"{traj_name}_fixed_mask_solutions",
            "fixed_mask_q": f"{traj_name}_fixed_mask_pressures",
            # Grid information (fixed)
            "edge_indices": "edge_indices",
            "vertices": "vertices",
            # Fixed masks
            "fixed_mask": f"{traj_name}_fixed_mask",
            "extra_fixed_mask": f"{traj_name}_extra_fixed_mask",
            "fixed_mask_solutions": f"{traj_name}_fixed_mask_solutions",
            "fixed_mask_pressures": f"{traj_name}_fixed_mask_pressures",
            "enumerated_fixed_mask": f"{traj_name}_enumerated_fixed_mask",
        },
        "timing": {
            "traj_gen_time": traj_gen_elapsed
        }
    }

    return trajectories_update, one_time_results, (i, trajectory_metadata), traj_result.log_compressed



def generate_data(system_args, out_dir, base_logger=None):
    if base_logger:
        logger = base_logger.getChild("navier-stokes")
    else:
        logger = logging.getLogger("navier-stokes")

    trajectory_metadata = []
    trajectories = {}

    grid_resolution = system_args.get("grid_resolution", 0.01)
    trajectory_defs = system_args["trajectory_defs"]

    # Determine number of cores accessible from this job
    num_cores = int(os.environ.get("SLURM_JOB_CPUS_PER_NODE",
                                   len(os.sched_getaffinity(0))))
    # Limit workers to at most the number of trajectories
    num_tasks = min(num_cores, len(trajectory_defs))

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_tasks) as executor:
        futures = []
        for i, traj_def in enumerate(trajectory_defs):
            futures.append(executor.submit(_generate_data_worker, i=i, traj_def=traj_def, grid_resolution=grid_resolution))
        with tarfile.open(out_dir / "polyfem_logs.tar", "w") as log_output_tar:
            for future in concurrent.futures.as_completed(futures):
                trajectories_update, one_time_results, traj_meta_update, log_compressed = future.result()
                # Write logs
                tinfo = tarfile.TarInfo(name=traj_meta_update[1]["name"] + ".log.xz")
                tinfo.size = len(log_compressed)
                log_output_tar.addfile(tinfo, fileobj=io.BytesIO(log_compressed))
                trajectories.update(trajectories_update)
                trajectory_metadata.append(traj_meta_update)
                # Process one-time results
                if "edge_indices" not in trajectories or "vertices" not in trajectories:
                    trajectories["edge_indices"] = compute_edge_indices(one_time_results["vertices"])
                    trajectories["vertices"] = one_time_results["vertices"]

    # Perform final processing of output
    logger.info("Done generating trajectories")
    trajectory_metadata.sort()
    trajectory_metadata = [d for _i, d in trajectory_metadata]

    viscosity = trajectory_defs[0].get("viscosity", 0.001)

    return SystemResult(
        trajectories=trajectories,
        metadata={
            "grid_resolution": grid_resolution,
            "viscosity": viscosity,
        },
        trajectory_metadata=trajectory_metadata)
