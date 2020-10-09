#! /usr/bin/env python3
import argparse
import shutil
import pathlib
import subprocess
import enum
import json
import collections
import os
import utils

PHASES = ["data_gen", "train", "eval"]

# Root argument parser
parser = argparse.ArgumentParser(description="Report on state of JSON-description runs and launch")
subparsers = parser.add_subparsers(title="commands",
                                   description="valid operations",
                                   dest="command",
                                   help="Operation to run")
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


def get_run_state(run_file_path, root_directory):
    root_directory = pathlib.Path(root_directory)
    # Load the run file
    with open(run_file_path, "r", encoding="utf8") as run_file:
        run_descr = json.load(run_file)
    # Identify output directory
    relative_out_dir = pathlib.Path(run_descr["out_dir"])
    out_dir = root_directory / relative_out_dir

    # Inspect the state of the run
    if not out_dir.is_dir():
        # Output directory doesn't exist, run outstanding
        return RunState.OUTSTANDING

    # If the directory exists we check the run description there
    out_run_desc_path = out_dir / "launch" / "run_description.json"
    if not out_run_desc_path.is_file():
        # Run description didn't get copied
        return RunState.NO_MATCH

    # Load the output run description
    with open(out_run_desc_path, "r", encoding="utf8") as out_run_file:
        out_run_descr = json.load(out_run_file)

    # Make sure we match
    if run_descr != out_run_descr:
        return RunState.NO_MATCH

    # Check whether the run finished
    done_token_path = out_dir / "launch" / "done_token.txt"
    if done_token_path.is_file():
        # The run matches and finished
        return RunState.FINISHED
    else:
        # Run matches, but isn't finished
        return RunState.INCOMPLETE


def get_out_dir_conflicts(root_directory):
    out_dir_runs = {}
    for phase in PHASES:
        # Iterate over all the run descriptions, get out_dir
        for run_descr_path in (root_directory / "descr" / phase).glob("*.json"):
            # Get the output directory
            with open(run_descr_path, "r", encoding="utf8") as run_file:
                run_descr = json.load(run_file)
            out_dir = root_directory / pathlib.Path(run_descr["out_dir"])
            if out_dir not in out_dir_runs:
                out_dir_runs[out_dir] = []
            out_dir_runs[out_dir].append(run_descr_path)
    # Now check for conflicts
    conflicts = []
    for out_dir, run_descr_paths in out_dir_runs.items():
        if len(run_descr_paths) > 1:
            # This is a conflict
            conflicts.append((out_dir, run_descr_paths))
    return conflicts


def do_scan(root_directory):
    if not root_directory.is_dir():
        raise ValueError(f"{root_directory} is not an existing directory")

    # Gather conflicting directories
    conflicts = get_out_dir_conflicts(root_directory)
    # Get run states
    run_states = {}
    for phase in PHASES:
        for run_descr_path in (root_directory / "descr" / phase).glob("*.json"):
            run_state = get_run_state(run_descr_path, root_directory)
            run_states[run_descr_path] = run_state

    # Report conflicts
    if conflicts:
        print("------ Conflicting Output Directories ------")
        for out_dir, run_descr_paths in conflicts:
            print(f"Output directory: {out_dir}")
            for path in run_descr_paths:
                print(f"  {path}")
        print("")

    # Report runs with error states
    incomplete_runs = [k for k, v in run_states.items() if v == RunState.INCOMPLETE]
    no_match_runs = [k for k, v in run_states.items() if v == RunState.NO_MATCH]

    if incomplete_runs:
        print("------ Incomplete Runs ------")
        for ir in incomplete_runs:
            print(ir)
        print("")
    if no_match_runs:
        print("------ Runs with Mismatched Descriptions ------")
        for nmr in no_match_runs:
            print(nmr)
        print("")

    # Final summary
    counts = collections.Counter(run_states.values())
    print("------ Run State Summary ------")
    print("{:5} runs outstanding".format(counts[RunState.OUTSTANDING]))
    print("{:5} runs finished".format(counts[RunState.FINISHED]))
    print("{:5} runs with mismatched descriptions".format(counts[RunState.NO_MATCH]))
    print("{:5} runs incomplete (still running or failed)".format(counts[RunState.INCOMPLETE]))
    print("")
    print("Found {} output directories with conflicts".format(len(conflicts)))


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


def do_local_launch(run_descr, root_directory):
    shortname = run_descr.relative_to(root_directory)
    print(f"Launching {shortname}")
    try:
        subprocess.run(["python3", "main.py", run_descr, root_directory], check=True)
    except subprocess.CalledProcessError:
        print("Run FAILED")


def do_launch(root_directory, phase, launch_method):
    print(f"Performing launch with {launch_method}")
    # Check for conda env
    if os.environ.get("CONDA_DEFAULT_ENV", None) != "nn-benchmark":
        # Conda env seems not to be loaded, warn
        print("WARNING: The Conda environment seems not to be loaded")
        print("         Consider canceling this launch to load it")

    # Check for a clean Git worktree
    git_info = utils.get_git_info()
    if git_info is None or not git_info.clean_worktree:
        # Dirty worktree or error
        print("WARNING: The Git worktree seems to be dirty")
        print("         Suggest committing your changes before proceeding")

    # List outstanding runs
    outstanding_runs = []
    for run_descr_path in (root_directory / "descr" / phase).glob("*.json"):
        run_state = get_run_state(run_descr_path, root_directory)
        if run_state == RunState.FINISHED:
            # Run finished already, skip
            continue
        elif run_state == RunState.OUTSTANDING:
            # Launch this one
            outstanding_runs.append(run_descr_path)
        else:
            # This is an error
            print("ERROR: A run in invalid state has been detected")
            print("       Use the scan mode to fix issues")
            print("       Runs should only be complete or outstanding")
            raise ValueError(f"Invalid run state {run_state} for run {run_descr_path}")

    # Check if user wants to continue
    if not outstanding_runs:
        print("No runs to launch")
        return

    print("Will launch {} runs".format(len(outstanding_runs)))
    try:
        yn = input("Continue? (yes/no) ")
    except EOFError:
        yn = "no"
    if yn.lower() != "yes":
        print("Canceling run")
        return

    # Launch the runs
    for run_descr in outstanding_runs:
        if launch_method == "local":
            do_local_launch(run_descr.resolve(), root_directory.resolve())
        elif launch_method == "slurm":
            raise NotImplementedError("Implement slurm launch")
        else:
            raise ValueError(f"Invalid launch type {launch_method}")


if __name__ == "__main__":
    args = parser.parse_args()

    command = args.command
    root_directory = pathlib.Path(args.root_directory)

    if command == "scan":
        # Scan for states of all run descriptions
        do_scan(root_directory)
    elif command == "launch":
        # Check states and launch outstanding runs
        phase = args.phase
        launch_method = select_launch_method(args.launch_type)
        do_launch(root_directory, phase, launch_method)
    else:
        # Invalid
        raise ValueError(f"Unknown command {command}")

    # Scan root_directory / "descr" / <phase> for JSON description files
    # Scan root_directory / <run's out_dir> for outputs
    # Determine which runs are outstanding, incomplete, finished, bad_match
    # Also check for duplicate output directories
    #   Should refuse to launch if any runs have problems (incomplete or description doesn't match, etc.)
    #   Otherwise, launch only the runs that are outstanding
