import numpy as np
import os
import sys
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
flags.DEFINE_integer("decimate_factor", 100, "Decimation factor on the timestep.")

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
        "integrator_timestep_size" : None,
        "train_loss" : None,
        "val_loss" : None,
        "relerr_l2" : None,
        "raw_l2" : None,
        "mse" : None,
        "ground_truth_data" : None,
        "inferred_data" : None,
    }


def build_experiment_dataframe(input_args):
    path, run_description, decimate_factor = input_args

    aggregate_data_df = []

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
        return None

    df_row_dict = get_aggregate_data_dict()

    df_row_dict["experiment_name"] = metadata["out_dir"]
    df_row_dict["system_name"] = system_metadata["system"]
    df_row_dict["method_name"] = metadata["phase_args"]["eval"]["eval_type"]
    df_row_dict["integrator_name"] = metadata["phase_args"]["eval"]["integrator"]
    df_row_dict["precision_type"] = metadata["phase_args"]["eval"]["eval_dtype"]
    df_row_dict["integrator_timestep_size"] = system_metadata["trajectories"][0]["time_step_size"]

    # Store model hyperparameters and configurations if neural network.
    if df_row_dict["method_name"] in ["hnn"]:
        df_row_dict["network_hidden_dim"] = model_config["arch_args"]["base_model_args"]["hidden_dim"]
        df_row_dict["network_depth"] = model_config["arch_args"]["base_model_args"]["depth"]
    elif df_row_dict["method_name"] in ["knn-regressor", "knn-predictor", "integrator-baseline", "gn", "knn-predictor-oneshot", "knn-regressor-oneshot"]:
        df_row_dict["network_hidden_dim"] = None
        df_row_dict["network_depth"] = None
    elif df_row_dict["method_name"] in ["nn-kernel"]:
        df_row_dict["network_hidden_dim"] = model_config["arch_args"]["hidden_dim"]
        df_row_dict["network_depth"] = None
    else:
        df_row_dict["network_hidden_dim"] = model_config["arch_args"]["hidden_dim"]
        df_row_dict["network_depth"] = model_config["arch_args"]["depth"]

    # Store training stats if trained.
    if df_row_dict["method_name"] in ["integrator-baseline", "knn-regressor-oneshot", "knn-predictor-oneshot"]:
        df_row_dict["num_train_trajectories"] = None
        df_row_dict["num_epochs"] = None
        df_row_dict["training_loss"] = None
        df_row_dict["validation_loss"] = None
    else:
        df_row_dict["num_train_trajectories"] = len(train_system_metadata["system_args"]["trajectory_defs"])
        df_row_dict["num_epochs"] = train_stats["num_epochs"]
        df_row_dict["train_loss"] = [epoch["train_total_loss"]/epoch["train_loss_denom"] for epoch in train_stats["epoch_stats"]]
        df_row_dict["val_loss"] = [epoch["val_total_loss"]/epoch["val_loss_denom"] for epoch in train_stats["epoch_stats"]]

    df_row_dict["num_eval_trajectories"] = len(system_metadata["system_args"]["trajectory_defs"])

    # Load ground truth data.
    df_row_dict["ground_truth_data"] = []
    ground_truth_trajectories = np.load(
        (path / metadata["phase_args"]["eval_data"]["data_dir"] / "trajectories.npz"))
    for trajectory_index, trajectory in enumerate(system_metadata["trajectories"]):
        ground_truth_data = np.stack([
            ground_truth_trajectories[trajectory["field_keys"]["p"]],
            ground_truth_trajectories[trajectory["field_keys"]["q"]],
        ], axis=-1)
        df_row_dict["ground_truth_data"].append(ground_truth_data[::decimate_factor, ...].tolist())


    # Load inferred data.
    df_row_dict["inferred_data"] = []
    inferred_trajectories = np.load(run_description.parent.parent / "integrated_trajectories.npz")
    errors = [[], [], []]
    integration_time = []
    for trajectory_index, trajectory in enumerate(results_metadata["integration_stats"]):
        timesteps = inferred_trajectories[trajectory["file_names"]["p"]].shape[0]
        inferred_data = np.stack([
            inferred_trajectories[trajectory["file_names"]["p"]].reshape([timesteps, -1, 2]),
            inferred_trajectories[trajectory["file_names"]["q"]].reshape([timesteps, -1, 2])],
            axis=-1)
        error_relerr_l2 = [inferred_trajectories[trajectory["file_names"]["relerr_l2"]][time_index]
                 for time_index in range(inferred_data.shape[0])]
        error_raw_l2 = [inferred_trajectories[trajectory["file_names"]["raw_l2"]][time_index]
                 for time_index in range(inferred_data.shape[0])]
        error_mse = [inferred_trajectories[trajectory["file_names"]["mse"]][time_index]
                 for time_index in range(inferred_data.shape[0])]
        errors[0].append(error_relerr_l2)
        errors[1].append(error_raw_l2)
        errors[2].append(error_mse)
        integration_time.append([trajectory["timing"]["integrate_elapsed"]])
        df_row_dict["inferred_data"].append(inferred_data[::decimate_factor, ...].tolist())
    errors = np.array(errors).mean(axis=0)
    integration_time = np.array(integration_time).mean()
    df_row_dict["relerr_l2"] = errors[0].tolist()
    df_row_dict["raw_l2"] = errors[1].tolist()
    df_row_dict["mse"] = errors[2].tolist()
    df_row_dict["integration_time"] = integration_time

    # ground_truth_trajectories = np.load(
    #     (path / metadata["phase_args"]["eval_data"]["data_dir"] / "trajectories.npz"))
    # for trajectory_index, trajectory in enumerate(system_metadata["trajectories"]):
    #     data = np.stack([
    #         ground_truth_trajectories[trajectory["field_keys"]["p"]],
    #         ground_truth_trajectories[trajectory["field_keys"]["q"]],
    #     ], axis=-1)
    #     for time_index in range(data.shape[0]):
    #         row = get_ground_truth_data_dict(df_row_dict["experiment_name"], trajectory_index, time_index)
    #         row["ground_truth_data"] = data[time_index, ...]
    #         ground_truth_data_df.append(row)

    # inferred_trajectories = np.load(run_description.parent.parent / "integrated_trajectories.npz")
    # for trajectory_index, trajectory in enumerate(results_metadata["integration_stats"]):
    #     data = np.stack([
    #         inferred_trajectories[trajectory["file_names"]["p"]],
    #         inferred_trajectories[trajectory["file_names"]["q"]]],
    #         axis=-1)
    #     for time_index in range(data.shape[0]):
    #         row = get_inferred_data_dict(df_row_dict["experiment_name"], trajectory_index, time_index)
    #         row["inferred_data"] = data[time_index, ...]
    #         row["relerr_l2"] = inferred_trajectories[trajectory["file_names"]["relerr_l2"]][time_index]
    #         inferred_data_df.append(row)

    print("Processed example at {}".format(run_description))

    return df_row_dict


def build_json(dir_prefix, processes):
    path = Path(dir_prefix)

    with multiprocessing.Pool(processes) as pool:
        results = pool.map(build_experiment_dataframe, itertools.product([path], path.glob("run/eval/*/launch/run_description.json"), [FLAGS.decimate_factor]))

    return [result for result in results if result is not None]


def main(argv):
    df_dict = build_json(FLAGS.root_dir, FLAGS.processes)

    if FLAGS.output_dir:
        print("Aggregate data has {} entries".format(len(df_dict)))
        data_string = json.dumps(df_dict, indent=4, sort_keys=True)
        data_string = "var nn_benchmark_data = \n" + data_string
        size_mb = sys.getsizeof(data_string) // (1024 * 1024)
        num_files = int(np.ceil(size_mb / 95.))
        num_entries_per_file = len(df_dict) // num_files
        collection_data_string = "var nn_benchmark_data = [];\n"
        for i in range(num_files):
            data_string = json.dumps(df_dict[i*num_entries_per_file:(i+1)*num_entries_per_file], indent=4, sort_keys=True)
            data_string = "var nn_benchmark_data_{} = \n".format(i) + data_string
            collection_data_string += "nn_benchmark_data = nn_benchmark_data.concat(nn_benchmark_data_{});\n".format(i)
            with open(os.path.join(FLAGS.output_dir, "nn_benchmark_data_{}.js".format(i)), "w") as file_:
                file_.write(data_string)
        with open(os.path.join(FLAGS.output_dir, "nn_benchmark_data.js"), "w") as file_:
            file_.write(collection_data_string)

if __name__ == "__main__":
    app.run(main)
