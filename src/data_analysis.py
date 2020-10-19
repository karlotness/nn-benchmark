import numpy as np
import os
import pandas
import json
from pathlib import Path
import matplotlib.pyplot as plt
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("root_dir", 
                    "/home/karl/benchmark-project/finished-runs", 
                    "Root directory that contains the logged data.")
flags.DEFINE_string("output_dir", None, "Directory to output the dataframe.")

df_columns=[
        "experiment_name",
        "system_name",
        "method_name",
        "integrator_name",
        "precision_type",
        "num_train_trajectories",
        "num_eval_trajectories",
        "num_epochs",
        "inference_time",
        "ground_truth_trajectory",
        "inferred_trajectory",
        "inferred_trajectory_error",
]


def build_dataframe(dir_prefix):
    df = pandas.DataFrame(columns=df_columns)

    path = Path(dir_prefix)

    for run_index, run_description in enumerate(path.glob("run/eval/*/launch/run_description.json")):
        with run_description.open() as file_:
            metadata = json.load(file_)
        with (run_description.parent.parent / "results_meta.json").open() as file_:
            results_metadata = json.load(file_)
        with (path / metadata["phase_args"]["eval_data"]["data_dir"] / "system_meta.json").open() as file_:
            system_metadata = json.load(file_)
        with (path / metadata["phase_args"]["eval_net"] / "train_stats.json").open() as file_:
            train_stats = json.load(file_)
        with (path / metadata["phase_args"]["eval_net"] / "model.json").open() as file_:
            model_config = json.load(file_)
        with (path / metadata["phase_args"]["eval_net"] / "launch" / "run_description.json").open() as file_:
            train_run_description = json.load(file_)
        with (path / train_run_description["phase_args"]["train_data"]["data_dir"] / "system_meta.json").open() as file_:
            train_system_metadata = json.load(file_)
            
        experiment_name = metadata["out_dir"]
        system_name = system_metadata["system"]
        method_name = metadata["phase_args"]["eval"]["eval_type"]
        integrator_name = metadata["phase_args"]["eval"]["integrator"]
        precision_type = metadata["phase_args"]["eval"]["eval_dtype"]
        num_train_trajectories = len(train_system_metadata["system_args"]["trajectory_defs"])
        num_eval_trajectories = len(system_metadata["system_args"]["trajectory_defs"])
        num_epochs = train_stats["num_epochs"]
        inference_time = np.mean([
            i["timing"]["integrate_elapsed"] 
            for i in results_metadata["integration_stats"]
        ])

        ground_truth_trajectories = np.load(
            (path / metadata["phase_args"]["eval_data"]["data_dir"] / "trajectories.npz"))
        ground_truth_trajectory = np.stack([
            np.stack([
                ground_truth_trajectories[trajectory["field_keys"]["p"]],
                ground_truth_trajectories[trajectory["field_keys"]["q"]]
            ], axis=-1)
            for trajectory in system_metadata["trajectories"]
        ])

        inferred_trajectories = np.load(run_description.parent.parent / "integrated_trajectories.npz")
        inferred_trajectory = np.stack([
            inferred_trajectories[integration["file_names"]["traj"]] 
            for integration in results_metadata["integration_stats"]])
        inferred_trajectory_error = np.stack([
            inferred_trajectories[integration["file_names"]["relerr_l2"]] 
            for integration in results_metadata["integration_stats"]])

        row_entry = [
            experiment_name,
            system_name,
            method_name,
            integrator_name,
            precision_type,
            num_train_trajectories,
            num_eval_trajectories,
            num_epochs,
            inference_time,
            ground_truth_trajectory,
            inferred_trajectory,
            inferred_trajectory_error,
        ]

        df.loc[run_index] = row_entry
    
    return df

def main(argv):
    df = build_dataframe(FLAGS.root_dir)
    
    print("Dataframe has {} entries".format(len(df)))
    
    if FLAGS.output_dir:
        df.to_pickle(os.path.join(FLAGS.output_dir, "dataframe.pkl"))

if __name__ == "__main__":
    app.run(main)