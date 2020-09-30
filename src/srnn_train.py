from systems import spring
from methods import srnn
import numpy as np
import torch
import time

TRAINING_TRAJECTORIES = 1000
TRAINING_STEPS = 30
T_SPAN = (0, 3)
TIME_STEP_SIZE = T_SPAN[1] / TRAINING_STEPS
LEARNING_RATE = 0.001
EPOCHS = 1000
BATCH_SIZE = 32

METHOD_HNET = 5

#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')

system = spring.SpringSystem()
net = srnn.SRNN(input_dim=1, hidden_dim=200, output_dim=1, depth=3).float().to(device)


# Generate training set
def sample_initial_cond(radius=0.7):
    x0 = np.random.rand(2) * 2 - 1
    return (x0 / np.sqrt(x0**2).sum()) * radius

initial_conditions = [sample_initial_cond() for i in range(TRAINING_TRAJECTORIES)]
np_train_trajectories = [system.generate_trajectory(ic, t_span=T_SPAN, time_step_size=TIME_STEP_SIZE) for ic in initial_conditions]
initial_conditions = [torch.from_numpy(ic).float().to(device) for ic in initial_conditions]
train_trajectories = []
for traj in np_train_trajectories:
    p, q = traj.p, traj.q
    traj = np.stack((p, q), axis=1)
    train_trajectories.append(torch.from_numpy(traj).float().to(device))

optim = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
loss_func = torch.nn.MSELoss()

train_idxs = np.arange(len(initial_conditions))

num_batches = int(np.ceil(len(train_trajectories) / BATCH_SIZE))

for epoch in range(EPOCHS):
    np.random.shuffle(train_idxs)
    tot_loss = 0
    time_batches, time_gather_data, time_integrate, time_backprop = 0, 0, 0, 0
    for batch in range(num_batches):
        start_batch = time.perf_counter()
        start_gather_data = time.perf_counter()
        batch_idxs = train_idxs[batch * BATCH_SIZE: (batch + 1) * BATCH_SIZE]
        # Gather batch and stack
        x0 = torch.stack([initial_conditions[bi] for bi in batch_idxs])
        # Gather targets
        y0 = torch.stack([train_trajectories[bi] for bi in batch_idxs])
        end_gather_data = time.perf_counter()
        # Split the components
        p_0s = torch.unsqueeze(x0[:, 0], 1)
        q_0s = torch.unsqueeze(x0[:, 1], 1)
        start_integrate = time.perf_counter()
        int_res = srnn.numerically_integrate('leapfrog', p_0s, q_0s, model=net, method=METHOD_HNET, T=TRAINING_STEPS,
                                             dt=TIME_STEP_SIZE, volatile=False, device=device, coarsening_factor=1)\
                                             .permute(1, 0, 2)
        end_integrate = time.perf_counter()
        # Compute loss and perform updates
        start_backprop = time.perf_counter()
        loss = loss_func(int_res, y0)
        loss.backward()
        optim.step()
        end_backprop = time.perf_counter()
        tot_loss += float(loss)
        end_batch = time.perf_counter()
        # Update timing counters
        time_batches += end_batch - start_batch
        time_gather_data += end_gather_data - start_gather_data
        time_integrate += end_integrate - start_integrate
        time_backprop += end_backprop - start_backprop
    print("Epoch {} -> avg loss {}".format(epoch, tot_loss / len(initial_conditions)))
    print("  Total batch time: {}".format(time_batches))
    print("    Total gather time: {}".format(time_gather_data))
    print("    Total integrate time: {}".format(time_integrate))
    print("    Total backprop time: {}".format(time_backprop))
    print("  Unaccounted time: {}".format(time_batches - time_gather_data - time_integrate - time_backprop))

torch.save(net.state_dict(), "srnn_model.pt")
