import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import numpy as np
from typing import List


parser = argparse.ArgumentParser(description="Run from JSON descriptions")
parser.add_argument("base_dir", type=str,
                    help="Base directory at which all paths will be rooted")
parser.add_argument("key", type=str,
                    help="Key for which animations will be plotted")


def plot_animation(ground_truth, inference, edge_indices):
    timesteps = ground_truth.shape[0]
    n_particles = int(np.sqrt(ground_truth.shape[1]))
    # Shift right to plot side by side
    ground_truth[..., 0] += n_particles

    fig = plt.figure(figsize=(8, 6))
    plt.xlim(-1, 2*n_particles + 0.5)
    plt.ylim(-1, n_particles + 0.5)

    skiptime = 10
    edge_lines_inf = []
    edge_lines_gt = []

    for e in range(edge_indices.shape[1]):
        a, = plt.plot([], [], "b", animated=True)
        b, = plt.plot([], [], "b", animated=True)
        edge_lines_inf.append(a)
        edge_lines_gt.append(b)

    vertices_inf, = plt.plot([], [], "or", animated=True)
    vertices_gt, = plt.plot([], [], "or", animated=True)

    def update_data(i, trajectory_inf, trajectory_gt, edges, v_line_inf, v_line_gt, e_lines_inf, e_lines_gt, skip):
        i *= skip
        v_line_inf.set_xdata(trajectory_inf[i, :, 0])
        v_line_inf.set_ydata(trajectory_inf[i, :, 1])
        v_line_gt.set_xdata(trajectory_gt[i, :, 0])
        v_line_gt.set_ydata(trajectory_gt[i, :, 1])
        for idx in range(edge_indices.shape[1]):
            line_inf = e_lines_inf[idx]
            line_gt = e_lines_gt[idx]

            j, k = edge_indices[:, idx]

            line_inf.set_xdata(trajectory_inf[i, [j, k], 0])
            line_inf.set_ydata(trajectory_inf[i, [j, k], 1])
            line_gt.set_xdata(trajectory_gt[i, [j, k], 0])
            line_gt.set_ydata(trajectory_gt[i, [j, k], 1])
        return v_line_inf, v_line_gt, *e_lines_inf, *e_lines_gt

    line_ani = FuncAnimation(fig, 
            update_data, 
            timesteps//skiptime, 
            fargs=(inference, 
                ground_truth, 
                edge_indices, 
                vertices_inf, 
                vertices_gt, 
                edge_lines_inf, 
                edge_lines_gt, 
                skiptime), 
            interval=50, 
            blit=True)

    # line_ani.save("animation.mp4")
    plt.show()

    return line_ani


def parse_data_tree(base_dir: str, keys: List[str]):
    ground_truth = None
    inferences = []
    for root, dirs, files in os.walk(base_dir):
        if "descr" in root:
            continue
        for dir_ in dirs:
            if "data_gen" in root and "eval" in dir_:
                ground_truth = np.load(os.path.join(root, dir_, "trajectories.npz"))
            if "eval" in root and dir_ in keys:
                inferences.append(np.load(os.path.join(root, dir_, "integrated_trajectories.npz")))

    assert ground_truth is not None
    assert len(inferences) != 0

    for inference in inferences:
        idx = 0
        while True:
            try:
                traj_inference = inference["traj_000{:02d}_q".format(idx)]
                num_steps = traj_inference.shape[0]
                num_dims = 2
                traj_inference = traj_inference.reshape([num_steps, -1, num_dims])
                gt_inference = ground_truth["traj_000{:02d}_q".format(idx)] 
                edge_indices = ground_truth["traj_000{:02d}_edge_indices".format(idx)]
                animation = plot_animation(gt_inference, traj_inference, edge_indices)
            except KeyError:
                break
            idx += 1


    return


if __name__ == "__main__":
    args = parser.parse_args()

    parse_data_tree(args.base_dir, [args.key])
 
