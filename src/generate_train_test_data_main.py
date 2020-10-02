# System level imports
import os
import json
import hashlib
from absl import app
from absl import flags
import numpy as np

# Package level imports
from systems.spring import SpringSystem

FLAGS = flags.FLAGS

# Flag definitions
flags.DEFINE_string('data_dir', 'data', 'Directory to save the training and validation data.')
flags.DEFINE_bool('delete_existing', False, 'Whether to delete the existing data in FLAGS.data_dir.')

def serialize_numpy(list_of_arrays, data_dir):
    fnames = []
    for array in list_of_arrays:
        fname = hashlib.sha1(array).hexdigest() + ".npy"
        np.save(os.path.join(data_dir, fname), array)
        fnames.append(fname)
    return fnames

def generate_spring_1d(data_dir):
    system = SpringSystem()
    x0 = np.array([1, 1], dtype=np.float64)
    t_span = np.array([0, 3], dtype=np.float64)
    timestep = 1e-1
    trajectory_data = [[], [], [], [], []]
    initial_conditions = []
    for alpha in np.linspace(1, 5, 100):
        single_trajectory = system.generate_trajectory(alpha*x0, t_span, timestep)
        for i in range(5):
            trajectory_data[i].append(single_trajectory[i].squeeze())
        initial_conditions.append(alpha*x0)
    for i in range(5):
        trajectory_data[i] = np.stack(trajectory_data[i])
    initial_conditions = np.stack(initial_conditions)
    [init_cond_fname,
     p_fname,
     q_fname,
     dpdt_fname,
     dqdt_fname,
     t_fname] = serialize_numpy([initial_conditions, *trajectory_data], data_dir)
    data = {
        "t_start": t_span[0],
        "t_end": t_span[1],
        "timestep": timestep,
        "dimensions": 1,
        "num_timesteps" : trajectory_data[0].shape[1],
        "trajectories" : trajectory_data[0].shape[0],
        "initial_conditions": init_cond_fname,
        "t" : t_fname,
        "p" : p_fname,
        "q" : q_fname,
        "dpdt" : dpdt_fname,
        "dqdt" : dqdt_fname,
    }
    return data

def generate_all_data(data_dir):
    metadata = {
        "spring_1d": generate_spring_1d(data_dir),
    }

    with open(os.path.join(data_dir, "metadata.json"), "w") as metadata_file:
        json.dump(metadata, metadata_file)


def main(argv):
    existing_data = os.listdir(FLAGS.data_dir)
    if not FLAGS.delete_existing and existing_data:
        raise AssertionError("Existing data found. To delete, use flag --delete_existing.")

    # First, clear FLAGS.data_dir.
    for fname in existing_data:
        fpath = os.path.join(FLAGS.data_dir, fname)
        os.unlink(fpath)

    generate_all_data(FLAGS.data_dir)
    return 0

if __name__ == "__main__":
    app.run(main)
