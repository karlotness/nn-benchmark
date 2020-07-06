import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib import animation
from IPython.display import HTML
import torch
import torch.nn as nn
import os
import scipy.optimize

import argparse
import pickle




# physical parameter
g = 1.
l = 1.

#  first order methods
def forwardEuler(x0, dt, g, l):
    q0, p0 = x0[0], x0[1]
    qt = q0 + dt * p0
    pt = p0 - dt * g/l * np.sin(q0)
    return np.array([qt, pt])

def nonlinearproblem(x, x0, dt, g, l):
    x1, x2 = x[0], x[1]
    q0, p0 = x0[0], x0[1]
    y1 = x1 - dt * x2 - q0
    y2 = x2 + dt * g/l * np.sin(x1) - p0
    
    return np.array([y1, y2])
def backwardEuler(x0, dt, g, l):
    
    # the intial guess will just be set to x0
    return scipy.optimize.fsolve(nonlinearproblem, x0, (x0,dt,g,l), xtol = 1e-12)

def symplecticEuler(x0, dt, g, l):
    q0, p0 = x0[0], x0[1]
    pt = p0 - dt * g/l * np.sin(q0)
    qt = q0 + dt * pt
    return np.array([qt, pt])


## second order methods
def stormer_verlet(x0, dt, g, l):
    q0, p0 = x0[0], x0[1]
    pt_ = p0 - dt/2. * g/l * np.sin(q0)
    qt = q0 + dt * pt_
    pt = pt_ - dt/2. * g/l * np.sin(qt)
    return np.array([qt, pt])



def integrate(integrator, x0, T, g, l, Nsteps):
    dt = T/Nsteps
    x = x0
    trajectory = np.zeros((Nsteps+1, 2))
    trajectory[0] = x0
    for i in range(Nsteps):
        x = integrator(x, dt, g, l)
        trajectory[i+1] = x
    return trajectory



def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", metavar='size of dataset', type=int, required = True)

    args = parser.parse_args()

    
    Ndata = args.size
    interval = 0.1

    X = [] # input
    y = [] # output
    for i in range(Ndata):
        x0 = 2 * np.random.normal(size = 2)
        xt = integrate(stormer_verlet, x0, interval, g, l, 200)[-1]
        X.append(x0)
        y.append(xt)

    X = np.vstack(X)
    y = np.vstack(y)
    data = {'X': X, 'y': y}
    with open('data/Pendulum'+str(args.size)+'.pickle', 'wb') as f:
        pickle.dump(data, f)



if __name__ == "__main__":
	main()







