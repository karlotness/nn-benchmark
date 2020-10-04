import logging
import pathlib
import numpy as np
from systems import spring
import json


def run_phase(base_dir, out_dir, phase_args):
    logger = logging.getLogger("data_gen")
    base_dir = pathlib.Path(base_dir)
    out_dir = pathlib.Path(out_dir)

    logger.info("Starting data generation")
    system = phase_args["system"]
    system_args = phase_args["system_args"]
    if system == "spring":
        sys_result = spring.generate_data(system_args=system_args,
                                          base_logger=logger)
    elif system == "wave":
        sys_result = None
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
