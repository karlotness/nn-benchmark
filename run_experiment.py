# run experiment
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
import random
import pickle
import sys
from os import path
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.optimize import minimize
from utils import *
from models import *

import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--physical_system_type', type = str, help = "'massspring' or 'gravitational'")
    parser.add_argument('--n_object', type = int, help = "number of interacting objects")
    parser.add_argument('--model_type', type = str, default = 'MLP', help = "one of [the set of papers]" )
    parser.add_argument('--T', type = int, default = 10, help = "number of time steps per trajectory")
    # for MLP/continuous time data, dt is not constant.
    parser.add_argument('--dt', type=float, default=0.1, help='size of time-step')
    #parser.add_argument('--T_test', type=int, default=50, help='number of time-steps of testing trajectories')
    parser.add_argument('--dataset_size', type = int, default = 10000, help ='size of train+val data set; max = 10000')
    parser.add_argument('--train_test_split', type=float, default = 0.9, help = 'train-test split ratio')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size for training')
    parser.add_argument('--n_epochs', type=int, default=1000, help='number of training epochs')
    parser.add_argument('--n_hidden', type=int, default=512, help='number of hidden units in the MLP/RNN/LSTM')
    parser.add_argument('--n_layers', type=int, default = 4, help='number of layers in the MLP network')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')

    args = parser.parse_args()

    physical_system_type = args.physical_system_type
    n_object = args.n_object
    model_type = args.model_type
    T = args.T
    dt = args.dt
    #T_test = args.T_test
    dataset_size = args.dataset_size
    train_test_split = args.train_test_split
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    n_hidden = args.n_hidden
    n_layers = args.n_layers
    lr = args.lr

    ## load data set
    if model_type == 'MLP':
        dataset = pickle.load(open('data/'+str(physical_system_type) + str(n_object) + 'body_continuous_time.pickle', 'rb'))
    else: # RNN or LSTM requires data at discrete time steps
        dataset = pickle.load(open('data/'+str(physical_system_type) + str(n_object) + 'body_discrete_time.pickle', 'rb'))
    X0 = dataset['X0'][:dataset_size]
    t = dataset['t'][:dataset_size]
    Xt = dataset['Xt'][:dataset_size]
    
    if model_type in ['RNN', 'LSTM']:
        # X = X0 for RNN, LSTM
        X = X0
        y = Xt
    else:
        # otherwise, for feed forward network or irregular time stampe:
        X = np.hstack((X0.reshape(-1,4*n_object), t.reshape(-1,1)))
        y = Xt.reshape(-1,4*n_object)


    PATH = os.getcwd()
    print(PATH)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using GPU')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    torch.set_default_tensor_type(torch.DoubleTensor)

    X_tensor = torch.Tensor(X).to(device = device)
    y_tensor = torch.Tensor(y).to(device = device)

    Xtrain = X_tensor[0:int(train_test_split*len(X_tensor))]
    Xtest = X_tensor[int(train_test_split*len(X_tensor)):]

    ytrain = y_tensor[0:int(train_test_split*len(y_tensor))]
    ytest = y_tensor[int(train_test_split*len(y_tensor)):]


    models = {'RNN':RNN, 'LSTM':LSTM, 'MLP': MLP}
    if model_type == 'MLP':
        model = MLP(n_layers, n_hidden, n_object, n_dim = 2).to(device = device)
    elif model_type == 'LSTM':
        model = LSTM(4 * n_object, n_hidden, 4 * n_object).to(device = device)
    elif model_type == 'RNN':
        model = RNN(4 * n_object, n_hidden, 4 * n_object).to(device = device)
        
    print('Number of model parameters = '+str(count_parameters(model)))

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose = True, mode='min', factor = 0.9, patience = 100)
    MSELoss = torch.nn.MSELoss()
    loss_hist = []
    validation_hist = []

    try:
        checkpoint = torch.load('trained_models/'+physical_system_type+str(args.n_object)+'objects'+'model'+model_type)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        loss_hist = checkpoint['loss_hist']
        validation_hist = checkpoint['validation_hist']
    except:
        pass

    for epoch in range(n_epochs):
        permutation = torch.randperm(Xtrain.size()[0])
        '''
        for i in range(0,Xtrain.size()[0], batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i+batch_size]
            batch_x, batch_y = Xtrain[indices], ytrain[indices]

            if model_type == 'MLP':
                outputs = model(batch_x)
                validation_error = MSELoss(model(Xtest), ytest)

            else:
                outputs = model(batch_x[:,0,:], T).permute(1,0,2)
                validation_error = MSELoss(model(Xtest[:,0,:], T).permute(1,0,2), ytest)

            loss = MSELoss(outputs,batch_y)
            loss_hist.append(float(loss))
            scheduler.step(loss)

            validation_hist.append(float(validation_error))
            loss.backward()
            optimizer.step()
        '''

        for i in range(0,Xtrain.size()[0], batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i+batch_size]
            batch_x, batch_y = Xtrain[indices], ytrain[indices]

            if model_type == 'MLP':
                outputs = model(batch_x)
                validation_error = MSELoss(model(Xtest), ytest)
            else:
                outputs = model(batch_x[:,0,:], 10).permute(1,0,2)
                validation_error = MSELoss(model(Xtest[:,0,:], 10).permute(1,0,2), ytest)
            loss = MSELoss(outputs,batch_y)
            loss_hist.append(float(loss))


            #validation_error = MSELoss(baseline_Net(Xtest[:,0,:], 10).permute(1,0,2), ytest)
            scheduler.step(loss)

            validation_hist.append(float(validation_error))
            loss.backward()
            optimizer.step()

            
        if (epoch+1)%10 == 0:
            data = {'train_loss': loss_hist, 'val_loss': validation_hist}
            with open('logs/'+physical_system_type+str(args.n_object)+'objects'+'model'+model_type+'.pickle', 'wb') as f:
                pickle.dump(data, f)

            # plot loss function.
            plt.figure(figsize = (10,7))
            plt.title('MSE Loss')
            plt.semilogy(loss_hist, label = 'training')
            plt.semilogy(validation_hist, label = 'validation')
            plt.legend()
            plt.savefig('logs/'+physical_system_type+str(args.n_object)+'objects'+'model'+model_type+'.png')
            plt.close()
            '''
            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss_hist': loss_hist,
            'validation_hist': validation_hist,
            }, 'trained_models/'+physical_system_type+str(args.n_object)+'objects'+'model'+model_type)
            '''



if __name__ == "__main__":
    main()



