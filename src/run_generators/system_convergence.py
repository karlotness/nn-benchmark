import argparse
import pathlib
import json
import numpy as np
import itertools
import copy


parser = argparse.ArgumentParser(description="Generate run descriptions")
parser.add_argument("base_dir", type=str,
                    help="Base directory for run descriptions")


EXPERIMENT_NAME_BASE = "system-convergence"


def wave_description(subsample, num_steps=1000, full_step_size=0.1):
    run_name = f"{EXPERIMENT_NAME_BASE}_wave-{subsample}"
    descr = {
        "out_dir": f"run/data_gen/{run_name}",
        "exp_name": EXPERIMENT_NAME_BASE,
        "run_name": run_name,
        "phase": "data_gen",
        "phase_args": {
            "system": "wave",
            "system_args": {
                "space_max": 1,
                "n_grid": 250,
                "trajectory_defs": [
                    {
                        "start_type": "cubic_splines",
                        "start_type_args": {
                            "width": 0.917022004702574,
                            "height": 1.220324493442158,
                            "position": 0.5
                        },
                        "wave_speed": 0.1,
                        "num_time_steps": num_steps,
                        "time_step_size": full_step_size,
                        "subsample": subsample,
                    },
                    {
                        "start_type": "cubic_splines",
                        "start_type_args": {
                            "width": 0.8023325726318398,
                            "height": 0.646755890817113,
                            "position": 0.37770157843063934
                        },
                        "wave_speed": 0.1,
                        "num_time_steps": num_steps,
                        "time_step_size": full_step_size,
                        "subsample": subsample,
                    }
                ]
            }
        },
        "slurm_args": {
            "gpu": False,
            "time": "08:00:00",
            "cpus": 16,
            "mem": 32
        }
    }
    return descr


def spring_description(subsampling):
    pass


if __name__ == "__main__":
    args = parser.parse_args()
    base_dir = pathlib.Path(args.base_dir)

    for i in range(1, 13):
        subsample = 2**i
        descr = wave_description(subsample=subsample)
        rname = descr["run_name"] + ".json"
        out_path = base_dir / "descr" / "data_gen" / rname
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf8") as out_file:
            json.dump(descr, out_file)
