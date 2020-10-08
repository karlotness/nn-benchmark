#! /usr/bin/env python3
import argparse
import shutil
import pathlib
import subprocess
import enum
import json
import utils

PHASES = ["data_gen", "train", "eval"]

# Root argument parser
parser = argparse.ArgumentParser(description="Report on state of JSON-description runs and launch")
subparsers = parser.add_subparsers(title="commands",
                                   description="valid operations",
                                   dest="command",
                                   help="Operation to run: scan or launch")
# Launch arguments
launch_args = subparsers.add_parser("launch", description="Launch new runs")
launch_args.add_argument("root_directory", type=str,
                         help="Path to directory tree of run descriptions")
launch_args.add_argument("phase", type=str,
                         choices=PHASES,
                         help="Select the phase to run. Chooses subdirectory of the root to search for JSON files.")
launch_args.add_argument("--launch_type", type=str, default="auto",
                         choices=["slurm", "local", "auto"],
                         help="Choice of launch type: Slurm submission or a local run. Default chooses automatically.")
# Scan arguments
scan_args = subparsers.add_parser("scan", description="Report on the state of all runs")
scan_args.add_argument("root_directory", type=str,
                       help="Path to directory tree of run descriptions")


class RunState(enum.Enum):
    OUTSTANDING = enum.auto()  # No output directory exists, (ready to launch)
    NO_MATCH = enum.auto()  # Run description does not match (manual resolution needed)
    INCOMPLETE = enum.auto()  # Either failed or in progress (manual resolution needed)
    FINISHED = enum.auto()  # Run finished successfully (skip, launch proceeds)


def get_run_state(run_file, root_directory):
    root_directory = pathlib.Path(root_directory)
    pass


def select_launch_method(user_preference):
    slurm_available = shutil.which("sbatch") is not None
    if user_preference == "auto":
        # No user preference, choose automatically
        return "slurm" if slurm_available else "local"
    elif user_preference == "slurm":
        # User wants Slurm, make sure it's available
        if not slurm_available:
            raise ValueError("Slurm launch requested, but unavailable")
        else:
            return "slurm"
    elif user_preference == "local":
        # User wants a local launch
        return "local"
    else:
        # Invalid
        raise ValueError(f"Invalid run type: {user_preference}")


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    command = args.command
    root_directory = pathlib.Path(args.root_directory)

    if command == "scan":
        # Scan for states of all run descriptions
        pass
    elif command == "launch":
        # Check states and launch outstanding runs
        phase = args.phase
        launch_method = select_launch_method(args.launch_type)
    else:
        # Invalid
        raise ValueError(f"Unknown command {command}")

    # Scan root_directory / "descr" / <phase> for JSON description files
    # Scan root_directory / <run's out_dir> for outputs
    # Determine which runs are outstanding, incomplete, finished, bad_match
    # Also check for duplicate output directories
    #   Should refuse to launch if any runs have problems (incomplete or description doesn't match, etc.)
    #   Otherwise, launch only the runs that are outstanding
