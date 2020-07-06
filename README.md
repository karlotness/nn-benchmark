# NN_physics
NN for learning physics simulations

## Baseline Neural Net Models
models.py file has baseline models:

1. MLP: with 4 hidden layers, each with 512 neurouns. softplus activation function.

2. RNNs:
Elman Network(1 layer)
LSTM(1 layer)

All networks have approximately 1 million parameters.

MLP approximates the function f: f(x_0, t) -> f(x_t)

RNNs approximate f: f(x_n) -> f(x_{n+1})

## Baseline Traditional Methods
### First-order methods:
Explicit, Implicit, Symplectic Euler
### Higher-order non-symplectic methods:
RK4
### Symplectic methods:
Symplectic Euler
Leapfrog, Verlet, 
and other higher-order symplectic integrator


## Datasets
10,000 trajectories; each with 10 sample points. 10% for validation 90% for training.
MLPs are fed with data with irregular time stamps since t is an input to the network
RNNs are fed with data with constant dt

Initial conditions of the simulated trajectories for a physics system are sampled randomly in the phase space and within a predifiend range in energy (E1, E2); 
so that hopefully the network learns how to simulate a Hamiltonian system, NOT just work on some initial condition(a trained ML model should know how to work on all situation the model is intended for. See https://arxiv.org/pdf/2006.02619.pdf for more discussion)

## Training
Adam optimizer w/ the reduceOnPlateau scheduler. Initial learning rate = 5e-3, factor = 0.9, patience 100.

## Evalutation Metrics

## Next:
Test other models and compare.
