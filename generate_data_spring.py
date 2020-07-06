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
    for e in edges:
        p_j_m_p_i = p_n[e[1]] - p_n[e[0]]
        p_j_m_p_i_norm = np.linalg.norm(p_j_m_p_i)
        
        k = K_ij[e[0]][e[1]]
        erl = edge_rest_lengths[e[0]][e[1]]
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

# compute energy;. q.shape = p.shape = [N x n_object x n_dim]
def compute_total_energy(q, p, K_ij, edges, edge_rest_lengths):
	KE = 0
	PE = 0
	KE = 1/2. * np.sum(p[:,:,0]**2 + p[:,:,1]**2, axis = 1)
	for i, j in edges:
	    erl = edge_rest_lengths[i][j]
	    k = K_ij[i][j]
	    PE -= 1/2 * k * (np.linalg.norm(q[:,i,:] - q[:,j,:], axis = 1) - erl)**2
	    
	return KE, PE


def integrate(p_n, v_n, T, Npoints, n_object, n_dim, edges, edge_rest_lengths, k, randomize, NSamplePoints = -1, integrator = verlet):
    if NSamplePoints == -1:
        NSamplePoints = Npoints
    steps = Npoints
    data = np.zeros((Npoints+1, 2,  n_object, n_dim))
    dt = T/steps
    data[0] = p_n, v_n
    for i in range(steps):
        p_n, v_n = integrator(p_n, v_n, edges, edge_rest_lengths, k, dt)
        data[i+1] = p_n, v_n

    # option 1: timestamps randomized
    if randomize == 1:
        indices = np.random.choice(np.arange(Npoints)+1, size = NSamplePoints, replace = False)
    # option 2: constant time step
    else:
        indices = (np.arange(Npoints+1))[int(Npoints/NSamplePoints):][::int(Npoints/NSamplePoints)]
    Ts = np.arange(0, Npoints+1) * dt
    return data[indices,0,:,:], data[indices,1,:,:], Ts[indices]

def main():

	np.random.seed(0)

	parser = argparse.ArgumentParser()
	parser.add_argument("--nTraj", metavar='number of trajectories', type=int, default = 10000)
	parser.add_argument("--n_object", metavar='number of objects', type=int, default = 3)
	parser.add_argument("--interval", metavar='interval of the trajectory', type=float, default = 4)
	#parser.add_argument("--epsilon", metavar='param of soft clipping the spring potential', type=float, default = 0.02)
	parser.add_argument("--pointsPerTraj", metavar='# of points for one trajectory', type=int, default = 10)
	parser.add_argument("--randomize", metavar = 'irregular time stamps', type=int, default = 1)
	#parser.add_argument("--uniformMass", metavar = 'unit mass if flagged 1; otherwise sampled from U(0.8, 1.2)', type=int, default = 1)

	args = parser.parse_args()

	# number of objects
	n_object = args.n_object
	randomize = args.randomize
	# with pairwise interaction
	EDGES = np.array(list(combinations(range(n_object), 2)))
	# uniform restlength

	vec = np.array([1.] * n_object)
	EDGE_REST_LENGTHS = np.outer(vec, vec)
	K = np.ones(n_object)#np.random.uniform(low = 0.8, high = 1.2, size = (n_object))
	K_ij = np.outer(K,K)
	Ndata = args.nTraj

	interval = args.interval # sub sampled step size; T = 4, 8, 12 for example.
	#epsilon = args.epsilon

	# data save path
	if randomize == 0:
	    filename = 'data/spring'+str(n_object)+'body_discrete_time.pickle'
	else:
	    filename = 'data/spring'+str(n_object)+'body_continuous_time.pickle'


	NSamplePoints = args.pointsPerTraj # = 100 for example.

	X = np.zeros((Ndata, NSamplePoints, n_object * 2 * 2))
	y = np.zeros((Ndata, NSamplePoints, n_object * 2 * 2)) # output
	timestamps = np.zeros((Ndata, NSamplePoints))

	for i in range(Ndata):
	    E = np.array([1.0])
	    while E[0]> -1 or E[0]<-2: # reasoning behind this: sample from energy within this range.
	        q0 = 1.0 * np.random.normal(size = (n_object, 2))
	        p0 = 1.0 * np.random.normal(size = (n_object, 2))

	        q0 -= np.average(q0, axis = 0) # restrict center of mass to be AT the center
	        p0 -= np.average(p0, axis = 0) # so that the system doesn't drift; galileo invariance

	        # filter by energy
	        KE, PE = compute_total_energy(np.expand_dims(q0,0), np.expand_dims(p0,0), K_ij, EDGES, EDGE_REST_LENGTHS)
	        E = KE + PE
	    qhist, phist, dT = integrate(q0,  p0, interval, int(interval/5e-4), n_object, 2, EDGES, EDGE_REST_LENGTHS, K_ij, randomize, NSamplePoints, integrator = verlet)
	    
	    X[i] = np.repeat(np.hstack((q0.reshape(-1), p0.reshape(-1))).reshape(1,-1), NSamplePoints, axis = 0)
	    y[i] = np.hstack((qhist.reshape(NSamplePoints, -1), phist.reshape(NSamplePoints, -1)))
	    timestamps[i] = dT
	    if((i+1)%100 == 0):
	        data = {'X0': X, 't':timestamps, 'Xt': y}
	        with open(filename, 'wb') as f:
	            pickle.dump(data, f)

	data = {'X0': X, 't':timestamps, 'Xt': y}
	with open(filename, 'wb') as f:
	    pickle.dump(data, f)

if __name__ == "__main__":
    main()

