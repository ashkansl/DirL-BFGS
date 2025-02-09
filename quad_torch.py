import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import os
import sys

from opzer.dirl_bfgs import DirLBFGS

from opzer.main_lbfgs import LBFGS
from opzer.bfgs import BFGS 



import time
import glob

out_fig_path = './out_figs/'
if not os.path.exists(out_fig_path):
    os.mkdir(out_fig_path)
    
USE_CUDA = True
if USE_CUDA:
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

def w(v):
    if USE_CUDA:
        return v.cuda()
    return v
 
def cpu(v):
    if v is None:
        return v
    elif isinstance(v, int):
        return v
    elif isinstance(v, float):
        return v
    return v.cpu()


def check_symmetric(a, rtol=1e-05, atol=1e-05):
    return torch.allclose(a, a.T, rtol=rtol, atol=atol)




w_name = "w5_1000d_1000s_1000cn.pt"
y_name = "y5_1000d_1000s_1000cn.pt"


opt_it = 30000  # vvv
history_size = 5000


quad_size = int(w_name.split('d')[0].split('_')[-1])
quad_samples = int(w_name.split('s')[0].split('_')[-1])

load_q = 1

quad_path = './data/'
w_name = quad_path + w_name
y_name = quad_path + y_name
allw = w(Variable(torch.load(w_name)))
ally = w(Variable(torch.load(y_name)))



q_counter = 1


class QuadraticLoss:
    def __init__(self, **kwargs):
        if load_q == 0:
            self.W = w(Variable(torch.randn(quad_samples, quad_size)))  # TODO comment
            self.y = w(Variable(torch.randn(quad_samples)))
        elif load_q == 1:
            global q_counter
            self.W = allw[q_counter,:,:]
            self.y = ally[q_counter,:]
            if self.W.shape[0] != quad_samples or self.W.shape[1] != quad_size:
                    raise Exception("error: size mismatch!")
            q_counter = q_counter + 1

        self.W_inv = torch.linalg.inv(self.W)
        self.answer = self.W_inv @ self.y
        debug = 0

    def get_loss(self, theta):
      
        t1 = torch.sum((self.W.double().matmul(theta.double()) - self.y.double()).pow(2))
        return t1


    def func(self, x):
        return (x-self.npy)**2

    def get_derive(self, theta):
        grad = 2 * (self.W.double().matmul(torch.from_numpy(theta).double()) - self.y.double()) @ self.W.double()
        return grad
    
   
    def visualize(self, thetas, qc, loss, name):
        fig = plt.figure('optimizer path')
        plt.clf()
        plt.scatter(self.answer[0].cpu(), self.answer[1].cpu(), marker="*", c='red', s=200)
        nt = np.zeros((len(thetas), 2))
        for i in range(len(thetas)):
            nt[i][0] = thetas[i][0]
            nt[i][1] = thetas[i][1]
        nt = nt.transpose()
  
        plt.scatter(nt[0][0], nt[1][0], marker="o", c='black', s=50)
        plt.plot(nt[0], nt[1], marker='.')
        plt.title(f'{qc}: loss={loss}')
       
        save_name = str(qc) + '_' + name + '_curve.png'
        plt.savefig(out_fig_path+save_name)
        plt.pause(0.1)


def two_f():
   
    opt2 = []
    tl2= []
    it2 = []
    timlist = []
    endqc = 1
    for qc in range(0, endqc):
        torch.cuda.empty_cache()
        print(DEVICE)
        global q_counter
        q_counter = qc
        print(f'{qc}-----------------------------------------------------------------------------------------')

        ql = QuadraticLoss()
    
        
        def closure2():
            optimizer.zero_grad()
            # print(x_lbfgs)
            objective = ql.get_loss(x_lbfgs2)
            objective.backward()
            return objective


        x_lbfgs2 = nn.Parameter(torch.ones(quad_size, device=DEVICE, dtype=torch.double))
        x_lbfgs2.requires_grad = True
 

       
        lr = 1
        line_search = "strong_wolfe"
       
        
        
        optimizer = DirLBFGS([x_lbfgs2], 
                    lr=1, 
                    history_size=history_size,
                    preallocate_memory=history_size+2, 
                    line_search_fn=line_search, 
                    tolerance_grad=1e-20,
                    tolerance_change=1e-20,
                    tolerance_change_gtd = False, 
                    restart=True, 
                    max_iter=1,
                    return_time=True,
                    debug_time=True)
        
        # optimizer = LBFGS([x_lbfgs2], 
        #         lr=1, 
        #         history_size=history_size,
        #         line_search_fn=line_search, 
        #         max_iter=1, 
        #         tolerance_grad=1e-20,
        #         tolerance_change=1e-20)


        # optimizer = BFGS([x_lbfgs2], 
        #                 lr=1, 
        #                 line_search_fn=line_search, 
        #                 max_iter=1, 
        #                 tolerance_grad=1e-20,
        #                 tolerance_change=1e-20)

       
        loss_specific = 10e-7

        t1 = time.time()
        ll = []
        l2 = 100000000
        for i in range(opt_it):
            l2,ttt = optimizer.step(closure2)
            # ll.append(l2.item())
            if l2 < loss_specific:
                break
            # if i % 100 == 0:
            #     print(f'{i}: {l2}')
            # print(l2)
            #----------------------------- time vs iter
            # if i == stop_iter:
            #     tim0 = time.time() 
            # if i == stop_iter + 20:
            #     tim1 = time.time() 
            #     print(f'time sum = {tim1 - tim0}')
            #     print(f'time avg = {(tim1 - tim0)/20}')
            #     break
            #-----------------------------
            
            
        print(i)
        new_time = time.time() - t1
        timlist.append(new_time)
        print(f'iters = {i}')
        print(f'Fast Time = {new_time}')
        print(f'Loss = {l2}')
        


    sumtime = 0
    for item in timlist:
        sumtime = item + sumtime
    print(f'avg time {sumtime/len(timlist)}')
      

        
    
   
    


if __name__ == "__main__":
    two_f()




