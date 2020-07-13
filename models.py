# models
import numpy as numpy
import torch
import torch.nn as nn
from itertools import combinations
class MLP(nn.Module):
    def __init__(self, numOfHiddenLayers, Nhidden, n_object, n_dim):
        super(MLP, self).__init__()
        self.skip = nn.Linear(2 * n_dim * n_object+1, 2 * n_dim * n_object)
        self.fc1 = nn.Linear(2 * n_dim * n_object+1, Nhidden)
        
        self.hiddenlayers = torch.nn.ModuleList()
        for i in range(numOfHiddenLayers):
            self.hiddenlayers.append(nn.Linear(Nhidden, Nhidden))
        self.out = nn.Linear(Nhidden, 4 * n_object)
        
    def forward(self, X):
        residual = self.skip(X)
        act = torch.nn.Softplus() #torch.relu
        X = act(self.fc1(X))
        for i,l in enumerate(self.hiddenlayers):
            X = act(l(X))
        X = self.out(X) + residual
        return X


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.rnn = nn.RNNCell(self.input_size, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, output_size)
        
    def forward(self, x_0, T, volatile=True):
        batch_size = x_0.shape[0]
        trajectory_predicted = torch.zeros(T, batch_size, self.output_size, requires_grad = not volatile).to(next(self.parameters()).device)
        hidden_init = torch.zeros(batch_size, self.hidden_size, requires_grad = not volatile).to(next(self.parameters()).device)
        cell_init = torch.zeros(batch_size, self.hidden_size, requires_grad = not volatile).to(next(self.parameters()).device)

        trajectory_predicted[0, :, :] = x_0
        x_input = x_0
        hidden = hidden_init
        cell = cell_init

        for t in range(T-1):
            
            hidden = self.rnn(x_input, hidden)
            output = self.fc(hidden)
            trajectory_predicted[t+1, :, :] = output
            x_input = output

        return trajectory_predicted



class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, output_size)
        
    def forward(self, x_0, T, volatile=True):
        batch_size = x_0.shape[0]
        trajectory_predicted = torch.zeros(T, batch_size, self.output_size, requires_grad = not volatile).to(next(self.parameters()).device)
        hidden_init = torch.zeros(batch_size, self.hidden_size, requires_grad = not volatile).to(next(self.parameters()).device)
        cell_init = torch.zeros(batch_size, self.hidden_size, requires_grad = not volatile).to(next(self.parameters()).device)

        trajectory_predicted[0, :, :] = x_0
        x_input = x_0
        hidden = hidden_init
        cell = cell_init

        for t in range(T-1):
            
            cell, hidden = self.lstm(x_input, (cell, hidden))
            output = self.fc(hidden)
            trajectory_predicted[t+1, :, :] = output
            x_input = output

        return trajectory_predicted



class many_to_one(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super(many_to_one, self).__init__()

        self.hiddenlayers = torch.nn.ModuleList()
        self.input = nn.Linear(input_size, hidden_size)
        
        self.layernorms = torch.nn.ModuleList()
        
        for i in range(n_layers):
            self.hiddenlayers.append(nn.Linear(hidden_size, hidden_size))
            
            self.layernorms.append(nn.LayerNorm(hidden_size))
            
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, X):
        act = torch.nn.Softplus() #torch.relu
        X = act(self.input(X))
        for i,(l,ln) in enumerate(zip(self.hiddenlayers, self.layernorms)):
            X = act(l(X))
            X = ln(X)
        X = self.out(X) 
        return X
## RK2, RK4
class HNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super(HNN, self).__init__()
        self.H = many_to_one(input_size, hidden_size, n_layers)
    def forward(self, q, p):
        X = torch.cat((q,p), -1)
        return self.H(X)

class SRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super(SRNN, self).__init__()

        self.KE = many_to_one(int(input_size/2), hidden_size, n_layers)
        self.PE = many_to_one(int(input_size/2), hidden_size, n_layers)

    def forward(self, q, p):
        #X = torch.cat((q,p), -1)
        return self.KE(p) + self.PE(q)

    
class HOGN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super(HOGN, self).__init__()
        self.n_object = int(input_size/4)
        # KE of each object i; pairwise PE of objects i,j
        self.KE = many_to_one(2, hidden_size, n_layers)
        self.PE = many_to_one(4, hidden_size, n_layers)
    def forward(self, q, p):
        q_n2 = q.reshape(-1, self.n_object, 2)
        p_n2 = p.reshape(-1, self.n_object, 2)
        batch_size = q.shape[0]
        KE_tot = torch.zeros((batch_size,1), requires_grad = False).to(next(self.parameters()).device)
        PE_tot = torch.zeros((batch_size,1), requires_grad = False).to(next(self.parameters()).device)

        for i in range(self.n_object):
            KE_tot += self.KE(p_n2[:,i,:])
            
        for i,j in combinations(range(self.n_object), 2):
            PE_tot += self.PE(torch.cat((q_n2[:,i,:], q_n2[:,j,:]),-1))
        
        return KE_tot + PE_tot


def Euler(q_0, p_0, HNet, T, dt, device, training_mode):
    q, p = q_0, p_0

    q.requires_grad_()
    p.requires_grad_()

    batch_size = q_0.shape[0]
    output_size = q_0.shape[-1] + p_0.shape[-1]

    trajectory_predicted = torch.zeros(T, batch_size, output_size, requires_grad = False).to(device)
    
    for i in range(T):
        H_hat = HNet(q, p)
        dhdq = torch.autograd.grad(H_hat.sum(), q, create_graph = True)[0]
        dhdp = torch.autograd.grad(H_hat.sum(), p, create_graph = True)[0]
        dqdt, dpdt = dhdp, - dhdq
        # Euler step.
        q_p1 = q + dt * dqdt
        p_p1 = p + dt * dpdt
        trajectory_predicted[i] = torch.cat((q_p1, p_p1), -1)
        q, p = q_p1, p_p1
    return trajectory_predicted



def RK2(q_0, p_0, HNet, T, dt, device, training_mode):
    q, p = q_0, p_0

    q.requires_grad_()
    p.requires_grad_()

    batch_size = q_0.shape[0]
    output_size = q_0.shape[-1] + p_0.shape[-1]

    trajectory_predicted = torch.zeros(T, batch_size, output_size, requires_grad = False).to(device)
    
    for i in range(T):
        H_hat = HNet(q, p)
        dhdq = torch.autograd.grad(H_hat.sum(), q, create_graph = training_mode)[0]
        dhdp = torch.autograd.grad(H_hat.sum(), p, create_graph = training_mode)[0]
        k1 = (dt * dhdp, - dt * dhdq)
        q_half, p_half = q + 1/2. * k1[0], p + 1/2. * k1[1]
        H_hat_half = HNet(q_half, p_half)
        dhdq_half = torch.autograd.grad(H_hat_half.sum(), q_half, create_graph = training_mode)[0]
        dhdp_half = torch.autograd.grad(H_hat_half.sum(), p_half, create_graph = training_mode)[0]
        k2 = (dt * dhdp_half, - dt * dhdq_half)
        q_p1, p_p1 = q + 1/2. * (k1[0]+k2[0]), p + 1/2. * (k1[1]+k2[1])

        # Euler step.
        trajectory_predicted[i] = torch.cat((q_p1, p_p1), -1)
        q, p = q_p1, p_p1
    return trajectory_predicted




def Leapfrog(q_0, p_0, HNet, T, dt, device, training_mode):
    q, p = q_0, p_0

    q.requires_grad_()
    p.requires_grad_()

    batch_size = q_0.shape[0]
    output_size = q_0.shape[-1] + p_0.shape[-1]

    trajectory_predicted = torch.zeros(T, batch_size, output_size, requires_grad = False).to(device)
    for i in range(T):
        H_hat = HNet(q, p)
        dpdt = - torch.autograd.grad(H_hat.sum(), q, create_graph = training_mode)[0]
        
        p_half = p + dpdt * (dt / 2)
        H_hat = HNet(q, p_half)
        dqdt_half = torch.autograd.grad(H_hat.sum(), p_half, create_graph = training_mode)[0]
        
        # LF step
        q_p1 = q + dt * dqdt_half
        H_hat = HNet(q_p1, p_half) # p_value shouldn't affect things.
        dpdt_p1 = - torch.autograd.grad(H_hat.sum(), q_p1, create_graph = training_mode)[0]
        p_p1 = p_half + dt/2. * dpdt_p1
        trajectory_predicted[i] = torch.cat((q_p1, p_p1), -1)
        q, p = q_p1, p_p1

    return trajectory_predicted


def SE(q_0, p_0, HNet, T, dt, device, training_mode):
    q, p = q_0, p_0

    q.requires_grad_()
    p.requires_grad_()

    batch_size = q_0.shape[0]
    output_size = q_0.shape[-1] + p_0.shape[-1]

    trajectory_predicted = torch.zeros(T, batch_size, output_size, requires_grad = False).to(device)
    
    for i in range(T):
        H_hat = HNet(q, p)
        dhdp = torch.autograd.grad(H_hat.sum(), p, create_graph = training_mode)[0]
        dqdt = dhdp
        # Euler step.
        q_p1 = q + dt * dqdt

        H_hat = HNet(q_p1, p)
        dhdq_p1 = torch.autograd.grad(H_hat.sum(), q_p1, create_graph = training_mode)[0]
        dpdt_p1 = - dhdq_p1

        p_p1 = p + dt * dpdt_p1
        trajectory_predicted[i] = torch.cat((q_p1, p_p1), -1)
        q, p = q_p1, p_p1
    return trajectory_predicted

                          
def numerically_integrate(integrator, model, x_0, T, dt, device, training_mode = True):
    # methods = ['RK2', 'RK4', 'LF', 'S4']
    # models = [HNN, SRNN, HOGN]
    
    q0, p0 = torch.split((x_0), int(x_0.shape[-1]/2), -1)
    predicted_trajectories = integrator(q0, p0, model, T, dt, device, training_mode)
    return predicted_trajectories

'''


# models
import numpy as numpy
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, numOfHiddenLayers, Nhidden, n_object, n_dim):
        super(MLP, self).__init__()
        self.skip = nn.Linear(2 * n_dim * n_object+1, 2 * n_dim * n_object)
        self.fc1 = nn.Linear(2 * n_dim * n_object+1, Nhidden)
        
        self.hiddenlayers = torch.nn.ModuleList()
        for i in range(numOfHiddenLayers):
            self.hiddenlayers.append(nn.Linear(Nhidden, Nhidden))
        self.out = nn.Linear(Nhidden, 4 * n_object)
        
    def forward(self, X):
        residual = self.skip(X)
        act = torch.nn.Softplus() #torch.relu
        X = act(self.fc1(X))
        for i,l in enumerate(self.hiddenlayers):
            X = act(l(X))
        X = self.out(X) + residual
        return X


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.rnn = nn.RNNCell(self.input_size, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, output_size)
        
    def forward(self, x_0, T, volatile=True):
        batch_size = x_0.shape[0]
        trajectory_predicted = torch.zeros(T, batch_size, self.output_size, requires_grad = not volatile).to(next(self.parameters()).device)
        hidden_init = torch.zeros(batch_size, self.hidden_size, requires_grad = not volatile).to(next(self.parameters()).device)
        cell_init = torch.zeros(batch_size, self.hidden_size, requires_grad = not volatile).to(next(self.parameters()).device)

        trajectory_predicted[0, :, :] = x_0
        x_input = x_0
        hidden = hidden_init
        cell = cell_init

        for t in range(T-1):
            
            hidden = self.rnn(x_input, hidden)
            output = self.fc(hidden)
            trajectory_predicted[t+1, :, :] = output
            x_input = output





class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, output_size)
        
    def forward(self, x_0, T, volatile=True):
        batch_size = x_0.shape[0]
        trajectory_predicted = torch.zeros(T, batch_size, self.output_size, requires_grad = not volatile).to(next(self.parameters()).device)
        hidden_init = torch.zeros(batch_size, self.hidden_size, requires_grad = not volatile).to(next(self.parameters()).device)
        cell_init = torch.zeros(batch_size, self.hidden_size, requires_grad = not volatile).to(next(self.parameters()).device)

        trajectory_predicted[0, :, :] = x_0
        x_input = x_0
        hidden = hidden_init
        cell = cell_init

        for t in range(T-1):
            
            cell, hidden = self.lstm(x_input, (cell, hidden))
            output = self.fc(hidden)
            trajectory_predicted[t+1, :, :] = output
            x_input = output

        return trajectory_predicted


# with layernorm op
class many_to_one(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super(many_to_one, self).__init__()

        self.hiddenlayers = torch.nn.ModuleList()
        self.input = nn.Linear(input_size, hidden_size)
        
        self.layernorms = torch.nn.ModuleList()
        
        for i in range(n_layers):
            self.hiddenlayers.append(nn.Linear(hidden_size, hidden_size))
            
            self.layernorms.append(nn.LayerNorm(hidden_size))
            
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, X):
        act = torch.nn.Softplus() #torch.relu
        X = act(self.input(X))
        for i,(l,ln) in enumerate(zip(self.hiddenlayers, self.layernorms)):
            X = act(l(X))
            X = ln(X)
        X = self.out(X) 
        return X

## RK1, RK2, RK4
class HNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super(HNN, self).__init__()
        self.H = many_to_one(input_size, hidden_size, n_layers)
    def forward(self, q, p):
        X = torch.cat((q,p), -1)
        return self.H(X)

## separable hamiltonian;
## SE, Leapfrog, S4
class SRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super(SRNN, self).__init__()

        self.KE = many_to_one(int(input_size/2), hidden_size, n_layers)
        self.PE = many_to_one(int(input_size/2), hidden_size, n_layers)

    def forward(self, q, p):
        #X = torch.cat((q,p), -1)
        return self.KE(p) + self.PE(q)

class HOGN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super(HOGN, self).__init__()
        self.n_object = int(input_size/4)
        # KE of each object i; pairwise PE of objects i,j
        self.KE = many_to_one(2, hidden_size, n_layers)
        self.PE = many_to_one(4, hidden_size, n_layers)
    def forward(self, q, p):
        q_n2 = q.reshape(-1, self.n_object, 2)
        p_n2 = p.reshape(-1, self.n_object, 2)
        batch_size = q.shape[0]
        KE_tot = torch.zeros(batch_size, requires_grad = False).to(next(self.parameters()).device)
        PE_tot = torch.zeros(batch_size, requires_grad = False).to(next(self.parameters()).device)

        for i in range(self.n_object):
            KE_tot += self.KE(p_n2[:,i,:])
        for i,j in combinations(range(self.n_object), 2):
            PE_tot += self.PE(torch.cat((q_n2[:,i,:], q_n2[:,j,:]),-1))
        return KE_tot + PE_tot


def Euler(q_0, p_0, HNet, T, dt, device):
    q, p = q_0, p_0

    q.requires_grad_()
    p.requires_grad_()

    batch_size = q_0.shape[0]
    output_size = q_0.shape[-1] + p_0.shape[-1]

    trajectory_predicted = torch.zeros(T, batch_size, output_size, requires_grad = False).to(device)
    
    for i in range(T):
        H_hat = HNet(q, p)
        dhdq = torch.autograd.grad(H_hat.sum(), q, create_graph = True)[0]
        dhdp = torch.autograd.grad(H_hat.sum(), p, create_graph = True)[0]
        dqdt, dpdt = dhdp, - dhdq
        # Euler step.
        q_p1 = q + dt * dqdt
        p_p1 = p + dt * dpdt
        trajectory_predicted[i] = torch.cat((q_p1, p_p1), -1)
        q, p = q_p1, p_p1
    return trajectory_predicted


def RK2(q_0, p_0, HNet, T, dt, device):
    q, p = q_0, p_0

    q.requires_grad_()
    p.requires_grad_()

    batch_size = q_0.shape[0]
    output_size = q_0.shape[-1] + p_0.shape[-1]

    trajectory_predicted = torch.zeros(T, batch_size, output_size, requires_grad = False).to(device)
    
    for i in range(T):
        H_hat = HNet(q, p)
        dhdq = torch.autograd.grad(H_hat.sum(), q, create_graph = True)[0]
        dhdp = torch.autograd.grad(H_hat.sum(), p, create_graph = True)[0]
        k1 = (dt * dhdp, - dt * dhdq)
        q_half, p_half = q + 1/2. * k1[0], p + 1/2. * k1[1]
        H_hat_half = HNet(q_half, p_half)
        dhdq_half = torch.autograd.grad(H_hat_half.sum(), q_half, create_graph = True)[0]
        dhdp_half = torch.autograd.grad(H_hat_half.sum(), p_half, create_graph = True)[0]
        k2 = (dt * dhdp_half, - dt * dhdq_half)
        q_p1, p_p1 = q + k2[0], p + k2[1]

        # Euler step.
        trajectory_predicted[i] = torch.cat((q_p1, p_p1), -1)
        q, p = q_p1, p_p1
    return trajectory_predicted

# Symplectic Euler
def SE(q_0, p_0, HNet, T, dt, device):
    q, p = q_0, p_0

    q.requires_grad_()
    p.requires_grad_()

    batch_size = q_0.shape[0]
    output_size = q_0.shape[-1] + p_0.shape[-1]

    trajectory_predicted = torch.zeros(T, batch_size, output_size, requires_grad = False).to(device)
    
    for i in range(T):
        H_hat = HNet(q, p)
        dhdp = torch.autograd.grad(H_hat.sum(), p, create_graph = True)[0]
        dqdt = dhdp
        # Euler step.
        q_p1 = q + dt * dqdt

        H_hat = HNet(q_p1, p)
        dhdq_p1 = torch.autograd.grad(H_hat.sum(), q_p1, create_graph = True)[0]
        dpdt_p1 = - dhdq_p1

        p_p1 = p + dt * dpdt_p1
        trajectory_predicted[i] = torch.cat((q_p1, p_p1), -1)
        q, p = q_p1, p_p1
    return trajectory_predicted


def Leapfrog(q_0, p_0, HNet, T, dt, device):
    q, p = q_0, p_0

    q.requires_grad_()
    p.requires_grad_()

    batch_size = q_0.shape[0]
    output_size = q_0.shape[-1] + p_0.shape[-1]

    trajectory_predicted = torch.zeros(T, batch_size, output_size, requires_grad = False).to(device)
    H_hat = HNet(q, p)
    dpdt = - torch.autograd.grad(H_hat.sum(), q, create_graph = True)[0]
    for i in range(T):
        p_half = p + dpdt * (dt / 2)
        H_hat = HNet(q, p_half)
        dqdt_half = torch.autograd.grad(H_hat.sum(), p_half, create_graph = True)[0]
        
        # LF step
        q_p1 = q + dt * dqdt_half
        H_hat = HNet(q_p1, p_half) # p_value shouldn't affect things.
        dpdt_p1 = - torch.autograd.grad(H_hat.sum(), q_p1, create_graph = True)[0]
        p_p1 = p_half + dt/2. * dpdt_p1
        trajectory_predicted[i] = torch.cat((q_p1, p_p1), -1)
        q, p = q_p1, p_p1

    return trajectory_predicted



                          
def numerically_integrate(integrator, model, x_0, T, dt, device):
    # methods = ['RK1', RK2', 'RK4',  'SE','LF', 'S4']
    # models = [HNN, SRNN, HOGN]
    
    q0, p0 = torch.split((x_0), int(x_0.shape[-1]/2), -1)
    predicted_trajectories = integrator(q0, p0, model, T, dt, device)
    return predicted_trajectories


'''


