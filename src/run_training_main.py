# System level imports
import os
import random
from absl import app
from absl import flags
import torch
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import DataLoader
import numpy as np

# Package level imports
from methods.hnn import HNN
from methods.hnn import MLP
from methods.srnn import SRNN
import integrators
from systems.spring import SpringSystem
from dataset import DatasetCollection

FLAGS = flags.FLAGS

# Flag definitions
flags.DEFINE_string('data_dir', 'data', 'Directory that contains the training and validation data.')
flags.DEFINE_string('model_save_dir', 'checkpoints', 'Directory that contains the trained models.')
flags.DEFINE_string('system_name', 'spring_1d', 'Name of the system to run training on.')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to run training for.')
flags.DEFINE_float('learning_rate', 1e-3, 'General learning rate for model training.')
flags.DEFINE_enum('architecture', 'hnn', ['hnn', 'srnn'], 'Architecture to use.')
flags.DEFINE_enum('precision', 'float32', ['float32', 'float64'], 'Floating point precision to use.')
flags.DEFINE_bool('cuda', False, 'Whether to accelerate training using CUDA. Not available for float64 precision.')


def configure_model(architecture, device):
    input_dim = 1
    hidden_dim = 200
    output_dim = 2
    depth = 3
    weight_decay = 1e-4
    nonlinearity = torch.nn.Tanh
    if architecture== 'hnn':
        nn_model = MLP(2*input_dim, hidden_dim, output_dim, depth=depth, nonlinearity=nonlinearity)
        model = HNN(2*input_dim, nn_model)
    elif architecture == 'srnn':
        model = SRNN(input_dim, hidden_dim, output_dim, depth=depth, nonlinearity=nonlinearity)
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), FLAGS.learning_rate, weight_decay=1e-4)

    return model, optim

def load_dataset(data_dir, system_name):
    dataset_collection = DatasetCollection(data_dir)
    return dataset_collection[system_name]

def data_batcher(batch_size, x, dxdt, initial_conditions):
    num_timeseries, num_timesteps, _ = x.shape
    total_batches = set(range(num_timeseries))
    while len(total_batches) > 0:
        batch_size = min(batch_size, len(total_batches))
        sample_batches = random.sample(total_batches, batch_size)
        total_batches -= set(sample_batches)
        sample_batches = tuple(sample_batches)
        yield x[sample_batches, ...], dxdt[sample_batches, ...], initial_conditions[sample_batches, ...]

def train_model(model, device, optim, dataset, architecture, precision):
    loader = DataLoader(dataset, batch_size=2)
    # q, p, dqdt, dpdt, t = data

    # x = torch.from_numpy(np.stack([p, q], axis=-1).astype(np.float32))
    # x.requires_grad = True

    # dxdt_ref = torch.from_numpy(np.stack([dpdt, dqdt], axis=-1).squeeze().astype(np.float32))

    # initial_conditions = torch.from_numpy(initial_conditions)

    # x = x.to(device)
    # dxdt_ref = dxdt_ref.to(device)
    # initial_conditions = initial_conditions.to(device)

    loss_fn = MSELoss()

    for epoch in range(FLAGS.epochs):
        #for x_batch, dxdt_ref_batch, initial_conditions_batch in data_batcher(10, x, dxdt_ref, initial_conditions):
        for init_cond, t, p, q, dpdt, dqdt in loader:
            x = torch.stack([p, q], dim=-1).float().to(device)
            dxdt = torch.stack([dpdt, dqdt], dim=-1).float().to(device)
            init_cond = init_cond.float().to(device)
            if architecture == 'hnn':
                x = x.reshape([-1, 2])
                x.requires_grad = True
                dxdt = dxdt.reshape([-1, 2])
                dxdt_pred = model.time_derivative(x)
                loss = loss_fn(dxdt_pred, dxdt)
            elif architecture == 'srnn':
                method_hnet = 5
                training_steps = 30
                time_step_size = 3./training_steps
                p0, q0 = torch.split(init_cond, [1, 1], dim=1)
                int_res = integrators.numerically_integrate(
                    'leapfrog',
                    p0,
                    q0,
                    model=model,
                    method=method_hnet,
                    T=training_steps,
                    dt=time_step_size,
                    volatile=False,
                    device=device,
                    coarsening_factor=1).permute(1, 0, 2)
                loss = loss_fn(int_res, x[:, :training_steps, ...])

            loss.backward()
            optim.step()
            optim.zero_grad()
        print("Epoch {} Loss {}".format(epoch, loss.detach().cpu().numpy()))


def main(argv):
    if FLAGS.cuda and FLAGS.precison == 'float64':
        raise AssertionError('Cannot run using both CUDA and float64.')

    if not os.path.exists(FLAGS.data_dir):
        raise AssertionError('Cannot find the specified data directory.')
    if not os.path.exists(FLAGS.model_save_dir):
        try:
            os.mkdir(FLAGS.model_save_dir)
        except:
            OSError('Error creating the model save directory.')

    device = torch.device('cuda') if torch.cuda.is_available() and FLAGS.cuda else torch.device('cpu')

    model, optim = configure_model(FLAGS.architecture, device)
    dataset = load_dataset(FLAGS.data_dir, FLAGS.system_name)
    train_model(model, device, optim, dataset, FLAGS.architecture, FLAGS.precision)
    return 0

if __name__ == '__main__':
    app.run(main)
