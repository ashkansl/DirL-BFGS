import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
from scipy.interpolate import make_interp_spline
# import seaborn
# from new import NewLBFGS
from dirlbfgs import NewLBFGS as FAST
# from newlbfgs_local import NewLBFGS as NewLBFGS0
# from newlbfgs_local_debug_lbfgs import NewLBFGS as NewLBFGS0
# from new_fast import NewLBFGS as NewLBFGS_fast
# from new_4 import NewLBFGS
# from new_2hist import NewLBFGS as NewLBFGS_2hist
# from new_nopast import NewLBFGS as NewLBFGS_nopast
# from new_undopast import NewLBFGS as NewLBFGS_undopast
# from newlbfgs_local_nogamma import NewLBFGS as NewLBFGS_nogamma

from main_lbfgs import LBFGS
# from lbfgs2 import LBFGS as LBFGS2


import time
import glob

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

# temcpu = []
# for varr in item:
    # itemcpu.append(varr.cpu())

def check_symmetric(a, rtol=1e-05, atol=1e-05):
    return torch.allclose(a, a.T, rtol=rtol, atol=atol)
# w_name = "w1000_5d.pt"
# y_name = "y1000_5d.pt"

# w_name = "w1000_5d_5s.pt"
# y_name = "y1000_5d_5s.pt"

# w_name = "w1000_2d_2s.pt"
# y_name = "y1000_2d_2s.pt"

# w_name = "w100_2d_2s_r1.pt"
# y_name = "y100_2d_2s_r1.pt"

# w_name = "w500_50d_200s.pt"
# y_name = "y500_50d_200s.pt"

# w_name = "w500_200d_600s.pt"
# y_name = "y500_200d_600s.pt"


# w_name = "w10_5000d_10000s.pt"
# y_name = "y10_5000d_10000s.pt"

# w_name = "w20_1000d_1100s.pt"
# y_name = "y20_1000d_1100s.pt"

# w_name == "w100_1000d_3000s.pt"
# y_name = "y100_1000d_3000s.pt"

# w_name = "w10_2000d_10000s.pt"
# y_name = "y10_2000d_10000s.pt"

# w_name = "w10_2000d_5000s.pt"
# y_name = "y10_2000d_5000s.pt"


# w_name = "w10_2000d_2200s.pt"
# y_name = "y10_2000d_2200s.pt"



w_name = "w2_100d_110s.pt"
y_name = "y2_100d_110s.pt"

# quad_size = 50  # 1000 # 5000  # d quad_size >= quad_samples
# quad_samples = 200  # 1500 # 10000 # s
quad_size = int(w_name.split('d')[0].split('_')[-1])
quad_samples = int(w_name.split('s')[0].split('_')[-1])
opt_it = 1000000  # vvv
load_q = 1

quad_path = './data/quad/'
w_name = quad_path + w_name
y_name = quad_path + y_name
allw = w(Variable(torch.load(w_name)))
ally = w(Variable(torch.load(y_name)))

# w_name = "w20_500d_2000s.pt"
# y_name = "y20_500d_2000s.pt"

# w_name = "w500_4d_20s.pt"
# y_name = "y500_4d_20s.pt"

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

        self.W_inv = torch.linalg.pinv(self.W)
        self.answer = self.W_inv @ self.y

    def get_loss(self, theta):
        t1 = torch.sum((self.W.double().matmul(theta.double()) - self.y.double()).pow(2))
        t2 = torch.sum((self.W.double().matmul(self.answer.double()) - self.y.double()).pow(2))
        return torch.abs(t1 - t2)
        # t1 = torch.sum((self.W.double().matmul(theta.double()) - self.y.double()).pow(2))
        # t2 = torch.sum((self.W.double().matmul(self.answer.double()) - self.y.double()).pow(2))
        # return (t1 - t2).pow(2)

    def func(self, x):
        return (x-self.npy)**2

    def get_grad(self, theta):
        grad = 2 * (self.W.double().matmul(theta.double()) - self.y.double()) @ self.W.double()
        return grad

    def visualize(self, thetas, qc, loss, name):
        fig = plt.figure('optimizer path')
        # ax = plt.axes(projection='3d', elev=50, azim=-50)
        plt.clf()
        plt.scatter(self.answer[0].cpu(), self.answer[1].cpu(), marker="*", c='red', s=200)
        nt = np.zeros((len(thetas), 2))
        for i in range(len(thetas)):
            nt[i][0] = thetas[i][0]
            nt[i][1] = thetas[i][1]
        nt = nt.transpose()
        # T = np.linspace(0, 1, np.size(thetas)) ** 2
        # for i in range(len(thetas)):
        #     plt.plot(nt[0][i], nt[1][i], marker='.',color=(0.0,0.5,T[i]))
        plt.scatter(nt[0][0], nt[1][0], marker="o", c='black', s=50)
        plt.plot(nt[0], nt[1], marker='.')
        plt.title(f'{qc}: loss={loss}')
        # plt.title(str(opnet.mult_module.l1.weight.data[0]) + str(final_loss))
        # plt.title(f'weights: {str(opnet.mult_module.l1.weight.data[0])} \n '
        #           f'{str(opnet.mult_module.l2.weight.data[0])} \n loss: {final_loss}')
        # plt.plot(nt[-1][0], nt[-1][1], marker='+', c='black')
        save_name = str(qc) + '_' + name + '_curve.png'
        plt.savefig(out_fig_path+save_name)
        plt.pause(0.1)
        # dddd = 0


def two_f():
    total_time = 0
    for qc in range(0, 1000):
        print(DEVICE)
        global q_counter
        q_counter = qc
        print(f'{qc}-----------------------------------------------------------------------------------------')

        ql = QuadraticLoss()
        matrix_rank = torch.linalg.matrix_rank(ql.W)
        matrix_cond = torch.linalg.cond(ql.W)
        print(f'matrix rank: {matrix_rank}, matrix condition number: {matrix_cond}')

        # if ~(matrix_cond > 5 and matrix_cond < 10):
        if ~(matrix_cond > 0):
            continue

        def f(x):
            ls = ql.get_loss(x)
            return ls

        def closure():
            lbfgs.zero_grad()
            # print(x_lbfgs)
            objective = f(x_lbfgs)
            objective.backward()
            return objective

        def closure2():
            lbfgs2.zero_grad()
            # print(x_lbfgs)
            objective = f(x_lbfgs2)
            objective.backward()
            return objective


        line_search = None # "strong_wolfe"
        # history_size = [5,200,300,400]
        # history_size = [10,20,50,100,200,500,700,1000]
        history_size = [20,500,1000,1500,2000,2500]
        loop = 50
        
        
        # loop = 50
        lr = 0.0000001
        x_lbfgs = nn.Parameter(torch.ones(quad_size, device=DEVICE) * 10)
        x_lbfgs.requires_grad = True
        lbfgs = LBFGS([x_lbfgs], lr=lr, history_size=history_size[-1], max_iter=1) # no wolfe
        x_lbfgs2 = nn.Parameter(torch.ones(quad_size, device=DEVICE) * 10)
        x_lbfgs2.requires_grad = True
        lbfgs2 = FAST([x_lbfgs2], lr=lr, history_size=history_size[-1], max_iter=1, fast_version=True, update_gamma=True) # no wolfe with fast wersion

        history_lbfgs = []
        history_lbfgs_log = []
        time_list1 = []
        iter_list1 = []
        history_lbfgs2 = []
        history_lbfgs_log2 = []
        time_list2 = []
        iter_list2 = []
        
        l1list = []
        l2list = []
        early_stop = 100
        early_counter = 0
        lmin = 10e10
        
        j = 0
        t1 = time.time()
        # log_flag = 0
        lpc = 100000000000000
        for i in tqdm(range(opt_it)):
            tft = time.time()
            history_lbfgs.append(f(x_lbfgs).item())
            history_lbfgs_log.append(np.log10(history_lbfgs[-1]))
            l1 = lbfgs.step(closure)
            l1list.append(l1)
            if lpc < loop:
                iter_list1.append(time.time() - tft)
                lpc = lpc + 1
            if j < len(history_size) and i > history_size[j]:
                time_list1.append(time.time() - t1)
                j = j+1
                lpc = 0
            if i > history_size[-1]+loop:
                break
     
        j = 0
        t1 = time.time()
        # log_flag = 0
        lpc = 100000000000000
        for i in tqdm(range(opt_it)):
            tft = time.time()
            history_lbfgs2.append(f(x_lbfgs2).item())
            history_lbfgs_log2.append(np.log10(history_lbfgs2[-1]))
            l2 = lbfgs2.step(closure2)
            l2list.append(l2)
            if lpc < loop:
                iter_list2.append(time.time() - tft)
                lpc = lpc + 1
            if j < len(history_size) and i > history_size[j]:
                time_list2.append(time.time() - t1)
                j = j+1
                lpc = 0
            if i > history_size[-1]+loop:
                break
        
        iter_list11 = []
        iter_list22 = []
        tmp1, tmp2 = 0, 0
        for i in range(len(iter_list1)):
            tmp1 = tmp1 + iter_list1[i]
            tmp2 = tmp2 + iter_list2[i]
            if (i+1)%loop==0:
                iter_list11.append(tmp1/loop)
                iter_list22.append(tmp2/loop)
                tmp1, tmp2 = 0, 0 
            
        torch.save(time_list1,'time_list1.pt')
        torch.save(time_list2,'time_list2.pt')
        torch.save(iter_list11,'iter_list1.pt')
        torch.save(iter_list22,'iter_list2.pt')
        torch.save(history_size,'history_size.pt')
        
        X1_Y1_Spline = make_interp_spline(time_list1, history_size)
        X1_ = np.linspace(min(time_list1), max(time_list1), 100)
        Y1_ = X1_Y1_Spline(X1_)
        X2_Y2_Spline = make_interp_spline(time_list2, history_size)
        X2_ = np.linspace(min(time_list2), max(time_list2), 100)
        Y2_ = X2_Y2_Spline(X2_)

        l1 = 'L-BFGS'
        l2 = 'Proposed Method'
        wname = w_name.replace("\\", "/").split("/")[-1].split('.')[0]
        fig, axs = plt.subplots(1)
        axs.plot(X1_, Y1_, label=l1, c='black', linestyle='--', linewidth=1.5)
        axs.plot(X2_, Y2_, label=l2, c='black', linewidth=1.5)
        axs.set_ylabel('history size')
        axs.set_xlabel('all iterations time (s)')
        fig.suptitle(f'{qc}:{wname}, m={history_size}, lr={lr}, line_search={line_search}', size='small')
        plt.legend()
        axs.grid(True, which="both", color='grey', linestyle=':', linewidth=0.7)
        save_name = wname + '_' + str(qc) + '_all_time_plot.png'
        plt.savefig(out_fig_path+save_name, dpi=500)
        
        plt.clf()
        X1_Y1_Spline = make_interp_spline(iter_list11, history_size)
        X1_ = np.linspace(min(iter_list11), max(iter_list11), 100)
        Y1_ = X1_Y1_Spline(X1_)
        X2_Y2_Spline = make_interp_spline(iter_list22, history_size)
        X2_ = np.linspace(min(iter_list22), max(iter_list22), 100)
        Y2_ = X2_Y2_Spline(X2_)

        l1 = 'L-BFGS'
        l2 = 'Proposed Method'
        wname = w_name.replace("\\", "/").split("/")[-1].split('.')[0]
        fig, axs = plt.subplots(1)
        axs.plot(X1_, Y1_, label=l1, c='black', linestyle='--', linewidth=1.5)
        axs.plot(X2_, Y2_, label=l2, c='black', linewidth=1.5)
        axs.set_ylabel('history size')
        axs.set_xlabel('each iteration time (s)')
        fig.suptitle(f'{qc}:{wname}, m={history_size}, lr={lr}, line_search={line_search}', size='small')
        plt.legend()
        axs.grid(True, which="both", color='grey', linestyle=':', linewidth=0.7)
        save_name = wname + '_' + str(qc) + '_iter_time_plot.png'
        plt.savefig(out_fig_path+save_name, dpi=500)
        # plt.show(block=False)
        break
        
        # plt.show(block=False)
        # plt.pause(0)
        
        # plt.plot(history_lbfgs)
        # plt.show()
        # plt.scatter(time_list1, history_lbfgs)
        # plt.show()
        


def load_and_plot():
    path = './experiments/quad/5000d_100000s/*'
    new_data_list = []
    lbf_data_list = []
    new_time_list = []
    lbf_time_list = []
    for item in glob.glob(path):
        item = item.replace('\\', '/')
        if item.find('new') > 0 and item.find('time') > 0:
            new_time_list.append(torch.load(item))
        elif item.find('new') > 0 and item.find('hist') > 0:
            new_data_list.append(torch.load(item))
        elif item.find('lbfgs') > 0 and item.find('time') > 0:
            lbf_time_list.append(torch.load(item))
        elif item.find('lbfgs') > 0 and item.find('hist') > 0:
            lbf_data_list.append(torch.load(item))
    for new_data, new_time, lbf_data, lbf_time in zip(new_data_list, new_time_list, lbf_data_list, lbf_time_list):
        print(f'{new_time[-1]} {lbf_time[-1]} {new_data[-1]} {lbf_data[-1]}')
        plt.semilogy(new_time, new_data, label='new')
        plt.semilogy(lbf_time, lbf_data, label='L-BFGS')
        plt.legend()
        plt.show(block=False)
        plt.pause(0)


if __name__ == "__main__":
    # one_f()
    two_f()
    # three_f()
    # load_and_plot()



