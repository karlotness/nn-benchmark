# NN_physics
NN for learning physics simulations

## Model Architectures
models.py file has baseline models:

MLP: with 4 hidden layers, each with 512 neurouns. softplus activation function.

RNN:
Elman Network(1 layer)
LSTM(1 layer)

All networks have approximately 1 million parameters.

MLP approximates the function f: f(x_0, t) -> f(x_t)
while RNN, LSTM approximates f: f(x_n) -> f(x_{n+1})

## Datasets
10,000 trajectories; each with 10 sample points. Note: may need to double this.
MLPs are fed with data with irregular time stamps since t is an input to the network
RNNs are fed with data with constant dt

Initial conditions of the simulated trajectories for a physics system are sampled randomly in the phase space and within a predifiend range in energy (E1, E2); 
so that hopefully the network learns how to simulate a Hamiltonian system, NOT just work on some initial condition(a trained ML model should know how to work on all situation the model is intended for. See https://arxiv.org/pdf/2006.02619.pdf for more discussion)


## Next:
Test other models and compare.
