import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib import animation
from IPython.display import HTML
import torch
import torch.nn as nn
import os
from IPython.display import clear_output
import argparse
import pickle
# use for non-zero rest length
def Force(p_n, edges, edge_rest_lengths, K_ij):

    F = np.zeros(p_n.shape)
    for e, erl in zip(edges, edge_rest_lengths):
        p_j_m_p_i = p_n[e[1]] - p_n[e[0]]
        p_j_m_p_i_norm = np.linalg.norm(p_j_m_p_i)
        
        k = K_ij[e[0]][e[1]]
        
        F[e[0]] += k * (p_j_m_p_i - erl * p_j_m_p_i/p_j_m_p_i_norm)
        F[e[1]] += - k * (p_j_m_p_i - erl * p_j_m_p_i/p_j_m_p_i_norm)
    
    return F

def forward_euler(p_n, v_n, edges, edge_rest_lengths, k, dt):

    a_n = Force(p_n, edges, edge_rest_lengths, k)
    v_n_p_1 =  v_n + dt * a_n
    p_n_p_1 =  p_n + dt * v_n

    return p_n_p_1, v_n_p_1

def implicit_euler(p_n, v_n, edges, edge_rest_lengths,k, dt):
    # TODO
    return

def symplectic_euler(p_n, v_n, edges, edge_rest_lengths, k, dt):
    a_n = Force(p_n, edges, edge_rest_lengths, k)
    v_n_p_1 = v_n + dt * a_n
    p_n_p_1 = p_n + dt * v_n_p_1

    return p_n_p_1, v_n_p_1

def verlet(p_n, v_n, edges, edge_rest_lengths, k, dt):

    a_n = Force(p_n, edges, edge_rest_lengths, k)
    p_n_p_1 = p_n + dt * v_n + 1/2. * a_n * dt **2
    v_n_p_1 = v_n + dt * 1/2. * (a_n + Force(p_n_p_1, edges, edge_rest_lengths, k))
    
    return p_n_p_1, v_n_p_1

def integrate(p_n, v_n, T, Npoints, n_object, n_dim, edges, edge_rest_lengths, k, integrator = verlet):
    steps = Npoints - 1
    data = np.zeros((Npoints, 2,  n_object, n_dim))
    dt = T/steps
    data[0] = p_n, v_n
    for i in range(steps):
        p_n, v_n = integrator(p_n, v_n, edges, edge_rest_lengths, k, dt)
        data[i+1] = p_n, v_n
    return data[:,0,:,:], data[:,1,:,:]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", metavar='size of dataset', type=int, required = True)
    parser.add_argument("--n", metavar='number of objects', type=int, required = True)
    parser.add_argument("--r", metavar='restlength', type=int, required = True)

    args = parser.parse_args()
    n_object = args.n
    Ndata = args.size
    # with pairwise interaction
    EDGES = np.array(list(combinations(range(n_object), 2)))
    R = float(args.r)
    # uniform restlength
    EDGE_REST_LENGTHS = np.array([R] * 12)
    # and spring constants randomly drawn from U(0.5, 1)
    #K = np.random.uniform(low = 0.5, high = 1, size = (n_object))
    #np.outer(K,K)
    K_ij = np.ones((n_object, n_object))

    interval = 0.1

    X = [] # input
    y = [] # output

    q0 = 1 * np.random.normal(size = (n_object, 2))
    p0 = 1 * np.random.normal(size = (n_object, 2))

    p0 -= np.average(p0, axis = 0) # so that the system doesn't drift; galileo invariance
    # time to integrate

    for i in range(Ndata):
        q0 = 1 * np.random.normal(size = (n_object, 2))
        p0 = 1 * np.random.normal(size = (n_object, 2))

        p0 -= np.average(p0, axis = 0) # so that the system doesn't drift; galileo invariance

        qhist, phist = integrate(q0, p0, interval, int(interval/5e-4) + 1, n_object, 2, EDGES, EDGE_REST_LENGTHS, K_ij)
        x0 = np.hstack((q0.reshape(-1), p0.reshape(-1)))
        xt = np.hstack((qhist[-1].reshape(-1), phist[-1].reshape(-1)))

        X.append(x0)
        y.append(xt)

    X = np.vstack(X)
    y = np.vstack(y)
    print(X.shape)
    
    data = {'X': X, 'y': y}
    with open('data/SpringMass2d'+'n'+str(args.n) + 'r' + str(args.r) +'data'+str(args.size)+'.pickle', 'wb') as f:
        pickle.dump(data, f)



if __name__ == "__main__":
    main()

