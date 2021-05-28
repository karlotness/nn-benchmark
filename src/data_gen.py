import logging
import pathlib
import numpy as np
import json


def run_phase(base_dir, out_dir, phase_args):
    logger = logging.getLogger("data_gen")
    base_dir = pathlib.Path(base_dir)
    out_dir = pathlib.Path(out_dir)

    logger.info("Starting data generation")
    system = phase_args["system"]
    system_args = phase_args["system_args"]
    if system == "spring":
        from systems import spring
        sys_result = spring.generate_data(system_args=system_args,
                                          base_logger=logger)
    elif system == "wave":
        from systems import wave
        sys_result = wave.generate_data(system_args=system_args,
                                        base_logger=logger)
    elif system == "particle":
        from systems import particle
        sys_result = particle.generate_data(system_args=system_args,
                                            base_logger=logger)
    elif system == "spring-mesh":
        from systems import spring_mesh
        sys_result = spring_mesh.generate_data(system_args=system_args,
                                               base_logger=logger)
    elif system == "taylor-green":
        from systems import taylor_green
        sys_result = taylor_green.generate_data(system_args=system_args,
                                                base_logger=logger)
    elif system == "navier-stokes":
        from systems import navier_stokes
        sys_result = navier_stokes.generate_data(system_args=system_args,
                                                 base_logger=logger)
    else:
        raise ValueError(f"Invalid system: {system}")

    # Save system trajectories
    logger.info("Saving system results")
    results_file = out_dir / "trajectories.npz"
    np.savez(results_file, **sys_result.trajectories)
    logger.info(f"Saved trajectories to: {results_file}")

    # Save system metadata
    with open(out_dir / "system_meta.json", "w", encoding="utf8") as meta_file:
        json.dump({"system": system,
                   "system_args": system_args,
                   "metadata": sys_result.metadata,
                   "trajectories": sys_result.trajectory_metadata},
                  meta_file)
    logger.info("Saved system metadata")

    logger.info("Data generation done")
