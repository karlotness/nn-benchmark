# System level imports
from absl import app
from absl import flags
import torch
from torch.optim import Adam
from torch.nn import MSELoss
import numpy as np

# Package level imports
from methods.hnn import HNN
from methods.hnn import MLP
from systems.spring import SpringSystem

FLAGS = flags.FLAGS

# Flag definitions
flags.DEFINE_integer('iterations', 1000, 'Number of iterations to run training for.')
flags.DEFINE_float('learning_rate', 1e-3, 'General learning rate for model training.')


def configure_hnn():
    input_dim = 2
    hidden_dim = 200
    output_dim = 2
    weight_decay = 1e-4
    nonlinearity = torch.nn.Tanh
    nn_model = MLP(input_dim, hidden_dim, output_dim, nonlinearity=nonlinearity)
    model = HNN(input_dim, nn_model)
    optim = torch.optim.Adam(model.parameters(), FLAGS.learning_rate, weight_decay=1e-4)

    return model, optim

def generate_spring_data():
    system = SpringSystem()
    x0 = np.array([1, 1], dtype=np.float32)
    t_span = np.array([0, 10], dtype=np.float32)
    time_step_size = 1e-3
    trajectory_data = system.generate_trajectory(x0, t_span, time_step_size)
    return trajectory_data

def train_hnn(model, optim, data):
    q, p, dqdt, dpdt, t = data
    x = torch.from_numpy(np.stack([p, q]).T.astype(np.float32))
    x.requires_grad = True
    dxdt_reference = torch.from_numpy(np.stack([dpdt, dqdt]).T.squeeze().astype(np.float32))
    loss_fn = MSELoss()
    for iteration in range(FLAGS.iterations):
        dxdt = model.time_derivative(x)
        loss = loss_fn(dxdt, dxdt_reference)

        loss.backward()
        optim.step()
        optim.zero_grad()
        print("Iteration {} Loss {}".format(iteration, loss.detach().cpu().numpy()))


def main(argv):
    model, optim = configure_hnn()
    data = generate_spring_data()
    train_hnn(model, optim, data)
    return 0

if __name__ == '__main__':
    app.run(main)
