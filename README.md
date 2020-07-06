# NN_physics
NN for learning physics simulations

# Physics
### Pendulum

### Nbody Simulation
Gravitational N-body
Spring System

# Baselines
## Naive Application of Neural Net Models
models.py file has baseline models:

1. MLP: with 4 hidden layers, each with 512 neurouns. softplus activation function.

2. RNNs:
Elman Network(1 layer)
LSTM(1 layer)

All networks have approximately 1 million parameters.

MLP approximates the function f: f(x_0, t) -> f(x_t)

RNNs approximate f: f(x_n) -> f(x_{n+1})
## Numerical Methods
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


### Simple datasets:
## Spring System with rest-length = 0
## Gravitational N-body with circular motions.

## Training
Adam optimizer w/ the reduceOnPlateau scheduler. Initial learning rate = 5e-3, factor = 0.9, patience 100.

## Variables
1. Inference Time: Compare with numerical integrators

2. Dataset Size: Data needed for the model to learn successfully.

## Deep Learning Models with Physics Constraints
# Models that work on few-body systems:
### 1. Newton vs the machine: solving the chaotic three-body problem using deep neural networks

https://arxiv.org/pdf/1910.07291.pdf

A brute force solution-using relu network with 10 hidden layers with 128 hidden nodes. 


### 2.SympNets: Intrinsic structure-preserving symplectic networks for identifying Hamiltonian systems

https://arxiv.org/pdf/2001.03750.pdf

Code Available: No; but basically the same as what I have been doing.

Experiments: Gravitational 3-body, Pendulum, Double Pendulum

### 3. Symplectic Recurrent Neural Networks

https://arxiv.org/abs/1909.13334

Code Available: Yes

Experiments: Spring Mass Chain (1D), Bouncing Ball with Gravity(2D)





# Models that scale:
## Mostly Graph NN based.

### 1. Neural Relational Inference for Interacting Systems 
(Interaction Network basically; but with the ability to classify the type of interaction)

https://arxiv.org/pdf/1802.04687.pdf

Code Available: Yes

Experiments: Spring Mass(2D), Charged Particle(2D), Kuramoto(1D)

### 2. Deep Potential Molecular Dynamics: a scalable model with the accuracy of quantum mechanics

https://arxiv.org/pdf/1707.09571.pdf

Code Available:No. But perhaps worth reprocducing


