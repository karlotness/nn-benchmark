import numpy as np
import os
import pandas
import json
from pathlib import Path
import matplotlib.pyplot as plt
from absl import app
from absl import flags
import pickle
import multiprocessing
import itertools

FLAGS = flags.FLAGS
flags.DEFINE_string("root_dir",
                    "/home/karl/benchmark-project/finished-runs",
                    "Root directory that contains the logged data.")
flags.DEFINE_string("output_dir", None, "Directory to output the dataframe.")
flags.DEFINE_integer("processes", 16, "Number of processes to use.")

def get_aggregate_data_dict():
    return {
        "experiment_name" : None,
        "system_name" : None,
        "method_name" : None,
        "integrator_name" : None,
        "precision_type" : None,
        "network_hidden_dim" : None,
        "network_depth" : None,
        "num_train_trajectories" : None,
        "num_eval_trajectories" : None,
        "num_epochs" : None,
        "inference_time" : None,
    }

def get_ground_truth_data_dict(experiment_name=None, trajectory_number=None, timestep_number=None):
    return {
            "experiment_name" : experiment_name,
            "trajectory_number" : trajectory_number,
            "timestep_number" : timestep_number,
            "ground_truth_data" : None,
    }

def get_inferred_data_dict(experiment_name=None, trajectory_number=None, timestep_number=None):
    return {
            "experiment_name" : experiment_name,
            "trajectory_number" : trajectory_number,
            "timestep_number" : timestep_number,
            "inferred_data" : None,
            "relerr_l2" : None,
    }


def build_experiment_dataframe(input_args):
    path, run_description = input_args

    aggregate_data_df = []
    ground_truth_data_df = []
    inferred_data_df = []

    try:
        with run_description.open() as file_:
            metadata = json.load(file_)
        with (run_description.parent.parent / "results_meta.json").open() as file_:
            results_metadata = json.load(file_)
        with (path / metadata["phase_args"]["eval_data"]["data_dir"] / "system_meta.json").open() as file_:
            system_metadata = json.load(file_)

        if metadata["phase_args"]["eval_net"]:
            # A model was trained.
            with (path / metadata["phase_args"]["eval_net"] / "train_stats.json").open() as file_:
                train_stats = json.load(file_)
            with (path / metadata["phase_args"]["eval_net"] / "model.json").open() as file_:
                model_config = json.load(file_)
            with (path / metadata["phase_args"]["eval_net"] / "launch" / "run_description.json").open() as file_:
                train_run_description = json.load(file_)
            with (path / train_run_description["phase_args"]["train_data"]["data_dir"] / "system_meta.json").open() as file_:
                train_system_metadata = json.load(file_)
        else:
            # No model was trained; this is a baseline.
            train_stats = None
            model_config = None
            train_run_description = None
            train_system_metadata = None

    except FileNotFoundError as e:
        return [], [], []

    df_row_dict = get_aggregate_data_dict()

    df_row_dict["experiment_name"] = metadata["out_dir"]
    df_row_dict["system_name"] = system_metadata["system"]
    df_row_dict["method_name"] = metadata["phase_args"]["eval"]["eval_type"]
    df_row_dict["integrator_name"] = metadata["phase_args"]["eval"]["integrator"]
    df_row_dict["precision_type"] = metadata["phase_args"]["eval"]["eval_dtype"]

    # Store model hyperparameters and configurations if neural network.
    if df_row_dict["method_name"] == "hnn":
        df_row_dict["network_hidden_dim"] = model_config["arch_args"]["base_model_args"]["hidden_dim"]
        df_row_dict["network_depth"] = model_config["arch_args"]["base_model_args"]["depth"]
    elif df_row_dict["method_name"] in ["knn-regressor", "knn-predictor", "integrator-baseline"]:
        df_row_dict["network_hidden_dim"] = None
        df_row_dict["network_depth"] = None
    else:
        df_row_dict["network_hidden_dim"] = model_config["arch_args"]["hidden_dim"]
        df_row_dict["network_depth"] = model_config["arch_args"]["depth"]

    # Store training stats if trained.
    if df_row_dict["method_name"] == "integrator-baseline":
        df_row_dict["num_train_trajectories"] = None
        df_row_dict["num_epochs"] = None
    else:
        df_row_dict["num_train_trajectories"] = len(train_system_metadata["system_args"]["trajectory_defs"])
        df_row_dict["num_epochs"] = train_stats["num_epochs"]

    df_row_dict["num_eval_trajectories"] = len(system_metadata["system_args"]["trajectory_defs"])
    df_row_dict["inference_time"] = np.mean([
        i["timing"]["integrate_elapsed"]
        for i in results_metadata["integration_stats"]
    ])

    ground_truth_trajectories = np.load(
        (path / metadata["phase_args"]["eval_data"]["data_dir"] / "trajectories.npz"))
    for trajectory_index, trajectory in enumerate(system_metadata["trajectories"]):
        data = np.stack([
            ground_truth_trajectories[trajectory["field_keys"]["p"]],
            ground_truth_trajectories[trajectory["field_keys"]["q"]],
        ], axis=-1)
        for time_index in range(data.shape[0]):
            row = get_ground_truth_data_dict(df_row_dict["experiment_name"], trajectory_index, time_index)
            row["ground_truth_data"] = data[time_index, ...]
            ground_truth_data_df.append(row)

    inferred_trajectories = np.load(run_description.parent.parent / "integrated_trajectories.npz")
    for trajectory_index, trajectory in enumerate(results_metadata["integration_stats"]):
        data = np.stack([
            inferred_trajectories[trajectory["file_names"]["p"]],
            inferred_trajectories[trajectory["file_names"]["q"]]],
            axis=-1)
        for time_index in range(data.shape[0]):
            row = get_inferred_data_dict(df_row_dict["experiment_name"], trajectory_index, time_index)
            row["inferred_data"] = data[time_index, ...]
            row["relerr_l2"] = inferred_trajectories[trajectory["file_names"]["relerr_l2"]][time_index]
            inferred_data_df.append(row)

    aggregate_data_df.append(df_row_dict)
    print("Processed example at {}".format(run_description))

    return aggregate_data_df, ground_truth_data_df, inferred_data_df



def build_dataframe(dir_prefix, processes):
    aggregate_data_df = pandas.DataFrame(columns=get_aggregate_data_dict().keys())
    ground_truth_data_df = pandas.DataFrame(columns=get_ground_truth_data_dict().keys())
    inferred_data_df = pandas.DataFrame(columns=get_inferred_data_dict().keys())

    path = Path(dir_prefix)

    with multiprocessing.Pool(processes) as pool:
        results = pool.map(build_experiment_dataframe, itertools.product([path], path.glob("run/eval/*/launch/run_description.json")))

    aggregate_data_df = aggregate_data_df.from_records(itertools.chain(*[agg for agg, _, _ in results]))
    ground_truth_data_df = ground_truth_data_df.from_records(itertools.chain(*[ground for _, ground, _ in results]))
    inferred_data_df = inferred_data_df.from_records(itertools.chain(*[inferred for _, _, inferred in results]))

    return {"aggregate_data" : aggregate_data_df,
            "ground_truth_data" : ground_truth_data_df,
            "inferred_data" : inferred_data_df}

def main(argv):
    df_dict = build_dataframe(FLAGS.root_dir, FLAGS.processes)

    print("Aggregate dataframe has {} entries".format(len(df_dict["aggregate_data"])))

    if FLAGS.output_dir:
        for k, v in df_dict.items():
            v.to_pickle(os.path.join(FLAGS.output_dir, k + ".pkl"))

if __name__ == "__main__":
    app.run(main)
