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


