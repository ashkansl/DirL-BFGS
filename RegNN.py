import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from opzer.dirl_bfgs_nn import DirLBFGS
from opzer.main_lbfgs_nn import LBFGS
from opzer.bfgs import BFGS
import time

# ------------------------------------------------------------------

def lbfgs_imp(opzer, lr, hs, npp, time_stop=0):

    data = fetch_california_housing()
    X, y = data.data, data.target

    X_train_raw, X_test_raw, y_train, y_test = X, X, y, y
    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
   


    X_train = torch.tensor(X_train, dtype=torch.float32, device='cuda')
    y_train = torch.tensor(y_train, dtype=torch.float32,  device='cuda').reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)


    torch.manual_seed(10)
    
    model = nn.Sequential(
        nn.Linear(8, 24),
        nn.Sigmoid(),
        nn.Linear(24, 12),
        nn.Sigmoid(),
        nn.Linear(12, 6),
        nn.ReLU(),
        nn.Linear(6, 1)
    )
    
   
    model.cuda()
    for name, param in model.named_parameters():
        print(param)
    

        
    # loss function and optimizer
    loss_fn = nn.MSELoss()  # mean square error

    early_stop = True
    history_size = hs
    l_r = lr
    line_search = None 
    optimizer = None
    tt1 = 0
    if opzer == 'lbfgs':
        optimizer = LBFGS(model.parameters(), lr=l_r, history_size=history_size, max_iter=1, line_search_fn=line_search, return_time=True) # no wolfe
    elif opzer == 'bfgs':
        optimizer = BFGS(model.parameters(), lr=l_r, history_size=2, max_iter=1, line_search_fn=line_search, return_time=True, dtype=torch.float) # no wolfe
    else:
        optimizer = DirLBFGS(model.parameters(), lr=l_r, 
                         history_size=history_size, preallocate_memory=history_size+10, max_iter=1, 
                         fast_version=True, update_gamma=True, 
                         line_search_fn=line_search, restart=True, return_time=True) # no wolfe with fast version


    n_epochs = npp   # number of epochs to run
    batch_size = 50000 # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)

    # Hold the best model
    best_mse = np.inf   # init to infinity
    best_weights = None
    history = []
    timelist = []
    ptlist = []
    acctime = 0

    t1 = time.time()
    if time_stop>0:
        n_epochs = 1000000000000000
    counter = 0
    for epoch in range(n_epochs):
        counter = counter + 1
        model.train()
        iter = 0
        for start in batch_start:
            iter = iter + 1
            
            # take a batch
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
   
            def closure():
                optimizer.zero_grad()
                y_pred = model(X_batch)
                
                objective = loss_fn(y_pred, y_batch)
                objective.backward()
                return objective
            # update weights
            loss, tott = optimizer.step(closure)
            acctime = acctime + tott
            
          
        timelist.append(time.time()-t1)
        ptlist.append(acctime)
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_train)
        mse = loss_fn(y_pred, y_train)
        mse = float(mse)
        if not torch.isnan(torch.tensor(mse)):
            history.append(mse)
        print(f'{epoch}: {mse} {len(optimizer.state[optimizer._params[0]]["old_dirs"])}')
        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())
        if timelist[-1] > time_stop and time_stop > 0:
            break

    # restore model and return best accuracy
    name = ''
    if time_stop > 0:
        name = str(opzer) +  '_lr' + str(lr) + '_hs' + str(hs) + '_epchs'+ str(counter) + '_time' + str(f'{timelist[-1]:0.3f}') + '_pime' + str(f'{ptlist[-1]:0.3f}')  + str(f'_loss{best_mse:.3f}')
    elif time_stop == False:
        name = str(opzer) + '_epchs'+ str(n_epochs) +  '_lr' + str(lr) + '_hs' + str(hs) + '_time' + str(f'{timelist[-1]:0.3f}') + '_pime' + str(f'{ptlist[-1]:0.3f}')  + str(f'_loss{best_mse:.3f}')
    torch.save(history, name + '_hist.pt')
    torch.save(timelist, name + '_time.pt')
    torch.save(ptlist, name + '_pime.pt')
    
    
    
    model.load_state_dict(best_weights)
    print("MSE: %.5f" % best_mse)
    print("RMSE: %.5f" % np.sqrt(best_mse))
    namee = opzer + '.png'




if __name__ == "__main__":

    opzer= ['bfgs']
    lrlist = [1.0]
    hslist = [10000]
    n_epochs = [2000]    
  
    for opz,lr,hs,npp in zip(opzer,lrlist,hslist,n_epochs):
        lbfgs_imp(opzer=opz, lr=lr, hs = hs, npp = npp, time_stop=0)
    
