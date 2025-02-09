
import numpy as np
import torch
import opzer.python_lbfgsb_quad as lbfgsb_gpu
import time
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
import os
import time







out_fig_path = './out_figs/'
if not os.path.exists(out_fig_path):
    os.mkdir(out_fig_path)
    
USE_CUDA = True
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




w_name = "w5_400d_400s_100cn.pt"
y_name = "y5_400d_400s_100cn.pt"

history_size = 100


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
       
        t1 = torch.sum((self.W.double().matmul(theta.double()) - self.y.double()).pow(2))
        return t1
    
   
    def get_derive(self, theta):
        grad = 2 * (self.W.double().matmul(theta.double()) - self.y.double()) @ self.W.double()
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
    l = torch.tensor([-np.inf], dtype=torch.double, device=DEVICE)
    u = torch.tensor([np.inf], dtype=torch.double, device=DEVICE)
   
    t0 = time.time()
   
    result, tt, iters = lbfgsb_gpu.L_BFGS_B(f=ql.get_loss, x0=x_lbfgs, df=ql.get_derive, m=history_size, l=l, u=u, max_iter = 700000, lsrch='scipy')
    ttt = time.time()-t0
    print('iters: ', iters)
    print('result: ', result)
    print('time:', tt)
    print(f'total time: {ttt}')
    print(f'conv. time: {lbfgsb_gpu.CONVTIME}')
    print(f'main time: {ttt - lbfgsb_gpu.CONVTIME}')

