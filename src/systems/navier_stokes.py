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


IN_MESH_PATH = pathlib.Path(__file__).parent / "resources" / "mesh.obj.xz"
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
                                           "fixed_mask_solutions"])

MESH_SIZE = (221, 42)


class NavierStokesSystem(System):
    def __init__(self, grid_resolution=0.01, viscosity=0.001, base_logger=None):
        super().__init__()
        self.grid_resolution = grid_resolution
        self.viscosity = viscosity
        if base_logger:
            self.logger = base_logger.getChild("ns-system")
        else:
            self.logger = logging.getLogger("ns-system")

    def _args_compatible(self, grid_resolution, viscosity, base_logger=None):
        return (self.grid_resolution == grid_resolution and
                self.viscosity == viscosity)

    def hamiltonian(self, q, p):
        return -1 * np.ones_like(q, shape=(q.shape[0]))

    def _gen_config(self, mesh_file, num_time_steps, time_step_size, in_velocity):
        t_end = (num_time_steps + 1) * time_step_size
        return {
            "mesh": mesh_file,
            "normalize_mesh": False,
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

    def generate_trajectory(self, num_time_steps, time_step_size, in_velocity, subsample=1, logger_override=None):
        orig_time_step_size = time_step_size
        orig_num_time_steps = num_time_steps

        time_step_size = time_step_size / subsample
        num_time_steps = num_time_steps * subsample

        if logger_override is not None:
            logger = logger_override.getChild("ns-system")
        else:
            logger = self.logger

        # Create temporary directory
        with tempfile.TemporaryDirectory(prefix="polyfem-") as _tmp_dir:
            logger.info(f"Generating trajectory in {_tmp_dir}")
            tmp_dir = pathlib.Path(_tmp_dir)

            # Write the mesh file
            with open(tmp_dir / "mesh.obj", "wb") as mesh_file:
                with lzma.open(IN_MESH_PATH, "r") as in_mesh_file:
                    mesh_file.write(in_mesh_file.read())

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
            polyfem_logger = logger.getChild("polyfem")
            prog = shutil.which(str(pathlib.Path(os.environ.get("POLYFEM_BIN_DIR", "")) / "PolyFEM_bin"))
            if not prog:
                raise ValueError("Cannot find PolyFEM; please install and set POLYFEM_BIN_DIR")
            cmdline = [prog, "--json", "config.json", "--cmd"]
            new_env = dict(os.environ)
            if "OMP_NUM_THREADS" not in new_env:
                new_env["OMP_NUM_THREADS"] = "2"
            logger.info(f"Launching PolyFEM: {cmdline}")
            with subprocess.Popen(cmdline, stdout=subprocess.PIPE, env=new_env, cwd=tmp_dir) as proc:
                for line in proc.stdout:
                    level = logging.DEBUG
                    str_line = codecs.decode(line, encoding="utf-8").strip()
                    if "[info]" in str_line:
                        # Crop the line before logging
                        str_line = str_line[str_line.index("[info]") + 7:]
                        level = logging.INFO
                    polyfem_logger.log(level, str_line)
            if proc.returncode != 0:
                raise RuntimeError(f"PolyFEM failed: {proc.returncode}")
            else:
                logger.info("PolyFEM finished")

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
        )


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


def system_from_records(grid_resolution, viscosity, base_logger=None):
    cached_sys = navier_stokes_cache.find(
        grid_resolution=grid_resolution,
        viscosity=viscosity,
        base_logger=base_logger
    )
    if cached_sys is not None:
        return cached_sys
    else:
        new_sys = NavierStokesSystem(
            grid_resolution=grid_resolution,
            viscosity=viscosity,
            base_logger=base_logger
        )
        navier_stokes_cache.insert(new_sys)
        return new_sys


def generate_data(system_args, base_logger=None):
    if base_logger:
        logger = base_logger.getChild("navier-stokes")
    else:
        logger = logging.getLogger("navier-stokes")

    trajectory_metadata = []
    trajectories = {}

    grid_resolution = system_args.get("grid_resolution", 0.01)
    trajectory_defs = system_args["trajectory_defs"]

    def generate_trajectory(system, metadata):
        # Generate trajectory
        traj_gen_start = time.perf_counter()
        traj_result = system.generate_trajectory(
            num_time_steps=metadata["num_time_steps"],
            time_step_size=metadata["time_step_size"],
            in_velocity=metadata["in_velocity"],
            subsample=metadata["subsample"],
            logger_override=logger.getChild(metadata["traj_name"])
        )
        traj_gen_elapsed = time.perf_counter() - traj_gen_start
        metadata["traj_result"] = traj_result
        metadata["traj_gen_elapsed"] = traj_gen_elapsed

        return metadata

    # Determine number of cores accessible from this job
    num_cores = int(os.environ.get("SLURM_JOB_CPUS_PER_NODE",
                                   len(os.sched_getaffinity(0))))
    # Limit workers to at most the number of trajectories
    num_tasks = min(num_cores, len(trajectory_defs))

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_tasks) as executor:
        futures = []
        for i, traj_def in enumerate(trajectory_defs):
            traj_name = f"traj_{i:05}"
            logger.info(f"Generating trajectory {traj_name}")

            num_time_steps = traj_def["num_time_steps"]
            time_step_size = traj_def["time_step_size"]
            in_velocity = traj_def["in_velocity"]
            viscosity = traj_def.get("viscosity", 0.001)
            subsample = int(traj_def.get("subsample", 1))

            # Create system
            system = navier_stokes_cache.find(grid_resolution=grid_resolution, viscosity=viscosity, base_logger=logger)
            if system is None:
                system = NavierStokesSystem(grid_resolution=grid_resolution, viscosity=viscosity, base_logger=logger)
                navier_stokes_cache.insert(system)

            metadata = {
                "traj_name": traj_name,
                "traj_num": i,
                "num_time_steps": num_time_steps,
                "time_step_size": time_step_size,
                "in_velocity": in_velocity,
                "viscosity": viscosity,
                "subsample": subsample,
            }

            futures.append(executor.submit(generate_trajectory, system=system, metadata=metadata))

        for future in concurrent.futures.as_completed(futures):
            result = future.result()

            traj_num = result["traj_num"]
            traj_name = result["traj_name"]
            num_time_steps = result["num_time_steps"]
            time_step_size = result["time_step_size"]
            in_velocity = result["in_velocity"]
            viscosity = result["viscosity"]
            subsample = result["subsample"]
            traj_result = result["traj_result"]
            traj_gen_elapsed = result["traj_gen_elapsed"]

            logger.info(f"Generated {traj_name} in {traj_gen_elapsed} sec")

            # Store trajectory data
            trajectories.update({
                f"{traj_name}_solutions": traj_result.solutions,
                f"{traj_name}_grads": traj_result.grads,
                f"{traj_name}_pressures": traj_result.pressures,
                f"{traj_name}_pressures_grads": traj_result.pressures_grads,
                f"{traj_name}_t": traj_result.t,
            })

            # Store one copy of each fixed result
            if "edge_indices" not in trajectories:
                trajectories["edge_indices"] = compute_edge_indices(traj_result.grids[0])
            if "vertices" not in trajectories:
                trajectories["vertices"] = traj_result.grids[0]
            if "fixed_mask" not in trajectories:
                trajectories["fixed_mask"] = traj_result.fixed_mask
            if "fixed_mask_solutions" not in trajectories:
                trajectories["fixed_mask_solutions"] = traj_result.fixed_mask_solutions
            if "fixed_mask_pressures" not in trajectories:
                trajectories["fixed_mask_pressures"] = traj_result.fixed_mask_pressures

            # Store per-trajectory metadata
            trajectory_metadata.append(
                (traj_num,
                 {"name": traj_name,
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
                      "fixed_mask_p": "fixed_mask_solutions",
                      "fixed_mask_q": "fixed_mask_pressures",
                      # Grid information (fixed)
                      "edge_indices": "edge_indices",
                      "vertices": "vertices",
                      "fixed_mask": "fixed_mask",
                      "fixed_mask_solutions": "fixed_mask_solutions",
                      "fixed_mask_pressures": "fixed_mask_pressures",
                  },
                  "timing": {
                      "traj_gen_time": traj_gen_elapsed
                  }}))

    logger.info("Done generating trajectories")

    trajectory_metadata.sort()
    trajectory_metadata = [d for _i, d in trajectory_metadata]

    return SystemResult(
        trajectories=trajectories,
        metadata={
            "grid_resolution": grid_resolution,
            "viscosity": viscosity,
        },
        trajectory_metadata=trajectory_metadata)
