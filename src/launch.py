#! /usr/bin/env python3
import argparse
import json
import utils
import pathlib
import logging
import shutil
import os
import platform
import time


parser = argparse.ArgumentParser(description="Launch runs from JSON descriptions")
parser.add_argument("run_description", type=str,
                    help="Path to run description file")


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.run_description, 'r', encoding="utf8") as run_file:
        run_description = json.load(run_file)

    # Set up logging
    out_dir = pathlib.Path(run_description["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_level = run_description.get("log_level", default="INFO")
    utils.set_up_logging(level=log_level, out_file=out_dir / 'run.log')
    logger = logging.getLogger("launch")

    # Create directory for launch data
    launch_data_dir = out_dir / "launch"
    launch_data_dir.mkdir(parents=True, exist_ok=True)

    # Copy the run description
    shutil.copy(args.run_description, launch_data_dir / "run_description.json")

    # Store environment details in json
    logger.info(f"Running on description: {args.run_description}")
    env_details = {"run_description": args.run_description,
                   "hostname": platform.node()}
    git_info = utils.get_git_info(base_logger=logger)
    if git_info:
        env_details["git_info"] = git_info.asdict()
    else:
        env_details["git_info"] = None
    env_details["envvars"] = os.environ
    with open(launch_data_dir / "env_details.json", 'w', encoding='utf8') as env_file:
        json.dump(env_file, env_details)

    # Dispatch to correct phase (directly in Python)
    phase = run_description["phase"]
    logger.info(f"Starting main phase {phase}")
    phase_start = time.perf_counter()
    try:
        if phase == "data_gen":
            pass
        elif phase == "train":
            pass
        elif phase == "eval":
            pass
    except Exception as e:
        logger.exception("Encountered exception during run")
        # Quit so we don't store the "done" flag
        raise e
    phase_end = time.perf_counter()
    logger.info("Main phase finished")

    # Save "done" file token
    total_phase_time = phase_end - phase_start
    with open(launch_data_dir / "done_token.txt", 'w', encoding='utf8') as done_token:
        done_token.write(f"done {total_phase_time}\n")
    logger.info(f"Total run time: {total_phase_time}")
    logger.info("Run finished")
