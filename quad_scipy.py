
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
import os
import time

from scipy.optimize import minimize





out_fig_path = './out_figs/'
if not os.path.exists(out_fig_path):
    os.mkdir(out_fig_path)
    
USE_CUDA = False
if USE_CUDA:
    DEVICE = torch.device('cuda:0')
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

history_size = 5


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

    def get_loss(self, theta):
        t1 = torch.sum((self.W.double().matmul(torch.from_numpy(theta).double()) - self.y.double()).pow(2))
        return t1
    
   
    def get_derive(self, theta):
        grad = 2 * (self.W.double().matmul(torch.from_numpy(theta).double()) - self.y.double()) @ self.W.double()
        return grad


    def func(self, x):
        return (x-self.npy)**2

    

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


ql = QuadraticLoss()
time_list = []
for i in range(1):
    x_lbfgs = torch.ones(quad_size, device=DEVICE)
  
    t1 = time.time()
    result = minimize(ql.get_loss, x_lbfgs, jac=ql.get_derive, tol=10e-200, method='L-BFGS-B',
                      options={'iprint':10, 'maxcor':history_size, 'ftol':10e-200, 'gtol':10e-200, 'maxiter':9999999999, 'maxfun':9999999999})
    t2 = time.time()
  
    print('Status : %s' % result['message'])
    print('Total Evaluations: %d' % result['nfev'])
    solution = result['x']
    print(result)
    time_list.append(t2-t1)
 
print(time_list)
sumtime = 0
for item in time_list:
    sumtime = item + sumtime
print(f'avg time {sumtime/len(time_list)}')