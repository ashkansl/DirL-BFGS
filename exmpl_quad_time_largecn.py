import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
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

# from lbfgs2 import LBFGS as LBFGS2
from main_lbfgs import LBFGS
from bfgs import LBFGS as BFGS



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

#--------------------------------------- good experiments


# w_name = "w100_100d_100s_10cn.pt"
# y_name = "y100_100d_100s_10cn.pt"

w_name = "w100_100d_100s_500cn.pt"
y_name = "y100_100d_100s_500cn.pt"
#--------------------------------------- 

# quad_size = 50  # 1000 # 5000  # d quad_size >= quad_samples
# quad_samples = 200  # 1500 # 10000 # s
quad_size = int(w_name.split('d')[0].split('_')[-1])
quad_samples = int(w_name.split('s')[0].split('_')[-1])

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

        self.W_inv = torch.linalg.inv(self.W)
        self.answer = self.W_inv @ self.y
        debug = 0

    def get_loss(self, theta):
         # t0 = torch.sum((self.W.double().matmul(theta.double()) - self.y.double()).pow(2))
        # t1 = self.W.double().matmul(theta.double()) - self.y.double() 
        # t2 = self.W.double().matmul(self.answer.double()) - self.y.double()
        # loss = torch.sum((t1 - t2).pow(2))
        # # print(f'{t0}, {torch.sum((t1 - t2).pow(2))}, {t0 - torch.sum((t1 - t2).pow(2))}')
        # return loss
        #------------------------------------------------------------------------------------------------
        # t1 = torch.sum((self.W.double().matmul(theta.double()) - self.y.double()).pow(2))
        # t2 = torch.sum((self.W.double().matmul(self.answer.double()) - self.y.double()).pow(2))
        # # print(t1-t2)
        # return t1 - t2    
        #------------------------------------------------------------------------------------------------
        t1 = torch.sum((self.W.double().matmul(theta.double()) - self.y.double()).pow(2))
        # print(t1)
        return t1


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
    opt1 = []
    tl1 = []
    it1 = []
    opt2 = []
    tl2= []
    it2 = []
    opt3 = []
    tl3 = []
    it3 = []
    endqc = 7
    time_iter_sl = np.zeros((endqc,6))
    for qc in range(0, endqc):
        torch.cuda.empty_cache()
        print(DEVICE)
        global q_counter
        q_counter = qc
        print(f'{qc}-----------------------------------------------------------------------------------------')

        ql = QuadraticLoss()
        matrix_rank = torch.linalg.matrix_rank(ql.W)
        matrix_cond = torch.linalg.cond(ql.W)
        print(f'matrix rank: {matrix_rank}, matrix condition number: {matrix_cond}')

        # if ~(matrix_cond > 100 and matrix_cond < 150):
        #     continue
        # if (matrix_cond > 1000):
        #     continue

        def f(x):
            ls = ql.get_loss(x)
            return ls

        def closure():
            lbfgs.zero_grad()
            # print(x_lbfgs)
            objective = f(x_lbfgs)
            objective.backward()
            return objective
        
        def closure3():
            lbfgs3.zero_grad()
            # print(x_lbfgs)
            objective = f(x_lbfgs3)
            objective.backward()
            return objective

        def closure2():
            lbfgs2.zero_grad()
            # print(x_lbfgs)
            objective = f(x_lbfgs2)
            objective.backward()
            return objective

        x_lbfgs = nn.Parameter(torch.ones(quad_size, device=DEVICE))
        x_lbfgs.requires_grad = True

        
        x_lbfgs3 = nn.Parameter(torch.ones(quad_size, device=DEVICE))
        x_lbfgs3.requires_grad = True
   

        x_lbfgs2 = nn.Parameter(torch.ones(quad_size, device=DEVICE))
        x_lbfgs2.requires_grad = True
 

        opt_it = 1500  # vvv
        
        history_size = 200
        lr = 1
        line_search = "strong_wolfe" #  # # "strong_wolfe" # "strong_wolfe" # "strong_wolfe" # "strong_wolfe"
        # lbfgs = LBFGS([x_lbfgs], lr=lr, history_size=history_size, max_iter=1, line_search_fn=line_search) # main
        lbfgs = LBFGS([x_lbfgs], 
                      lr=lr, 
                      history_size=history_size, 
                      max_iter=1, 
                      line_search_fn=line_search,
                      tolerance_grad=1e-12, 
                      tolerance_change=1e-12) # no wolfe
      
        # history_size = 2
        lbfgs3 = BFGS([x_lbfgs3], 
                      lr=lr, 
                      history_size=2, 
                      max_iter=1, 
                      line_search_fn=line_search,
                      tolerance_grad=1e-12, 
                      tolerance_change=1e-12) # no wolfe
        
        history_size2 = history_size
        lbfgs2 = FAST([x_lbfgs2], 
                      lr=lr, 
                      history_size=history_size2, 
                      max_iter=1, 
                      fast_version=True, 
                      update_gamma=True, 
                      line_search_fn=line_search,
                      tolerance_grad=1e-12, 
                      tolerance_change=1e-12,
                      restart=True) # no wolfe) # no wolfe with fast wersion

        

        history_lbfgs = []
        history_lbfgs_log = []
        time_list1 = []
        itl1 = []
        itl2 = []
        itl3 = []
        history_lbfgs2 = []
        history_lbfgs_log2 = []
        time_list2 = []
        l1list = []
        l2list = []
        early_stop = 7
        early_counter = 0
        break_falg = False
        lmin = 1e-6
        loss_specific = 1e-6
        sflag = 1
        t1 = time.time()
        #lllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllll
        for i in tqdm(range(opt_it)):
            history_lbfgs.append(f(x_lbfgs).item())
            history_lbfgs_log.append(np.log10(history_lbfgs[-1]))
            l1 = lbfgs.step(closure)
            l1list.append(l1)
            time_list1.append(time.time() - t1)
            itl1.append(i)
            if l1 < loss_specific and sflag == 1:
                time_iter_sl[qc, 0] = time_list1[-1]
                time_iter_sl[qc, 1] = i
                sflag = 0
            # if break_falg:
                # break
            if break_falg:
                break
            if l1 < lmin:
                break_falg = True
            if i>10 and l1list[-2]-l1 <= 1e-12:
                early_stop = early_stop - 1
                if early_stop==0:
                    temp = 0
                    # break
                
            #     lmin = l1
            #     early_counter = 0
            # if early_counter > early_stop:
            #     break
            # early_counter += 1
            # if l1 < 10e-7:
            #     break
        lbfgs_time = time.time() - t1
        
        #bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
        history_lbfgs3 = []
        history_lbfgs_log3 = []
        l3list = []
        early_stop = 15
        time_list3 = []
        sflag = 1
        break_falg = False
        t1 = time.time()
        for i in tqdm(range(opt_it)):
            history_lbfgs3.append(f(x_lbfgs3).item())
            history_lbfgs_log3.append(np.log10(history_lbfgs3[-1]))
            l1 = lbfgs3.step(closure3)
            l3list.append(l1)
            time_list3.append(time.time() - t1)
            itl3.append(i)
            if l1 < loss_specific and sflag == 1:
                time_iter_sl[qc, 4] = time_list3[-1]
                time_iter_sl[qc, 5] = i
                sflag = 0
            if break_falg:
                break
            if l1 < lmin:
                break_falg = True
        bfgs_time = time.time() - t1
                
                
        #newwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
        early_counter = 0
        early_stop = 15
        break_falg = False
        # lmin = 10e7
        t1 = time.time()
        sflag = 1
        for i in tqdm(range(opt_it)):
            history_lbfgs2.append(f(x_lbfgs2).item())
            history_lbfgs_log2.append(np.log10(history_lbfgs2[-1]))
            l2 = lbfgs2.step(closure2)
            l2list.append(l2)
            time_list2.append(time.time() - t1)
            itl2.append(i)
            if l2 < loss_specific and sflag == 1:
                time_iter_sl[qc, 2] = time_list2[-1]
                time_iter_sl[qc, 3] = i
                sflag = 0
            # if break_falg:
            #     break
            if break_falg:
                break
            if l2 < lmin:
                break_falg = True
            if i>10 and l2list[-2]-l2 <= 1e-12:
                early_stop = early_stop - 1
                if early_stop==0:
                    temp = 0
                    # break
                
            # if early_counter > early_stop:
                
            # early_counter += 1
            # if l2 < 10e-7:
            #     break
        new_time = time.time() - t1
        # print(time_list1)

        #---------------------------------------------------------------------------------- time and step
        l1 = 'lbfgs'
        l3 = 'bfgs'
        l2 = 'new'
        wname = w_name.replace("\\", "/").split("/")[-1].split('.')[0]
        fig, axs = plt.subplots(2)
        axs[0].semilogy(history_lbfgs, label=l1)
        axs[0].semilogy(history_lbfgs3, label=l3)
        axs[0].semilogy(history_lbfgs2, label=l2)
        axs[0].set_ylabel('loss')
        axs[0].set_xlabel('step')
        axs[1].semilogy(time_list1, history_lbfgs, label=l1)
        axs[1].semilogy(time_list3, history_lbfgs3, label=l3)
        axs[1].semilogy(time_list2, history_lbfgs2, label=l2)
        axs[1].set_ylabel('loss')
        axs[1].set_xlabel('time')
        fig.suptitle(f'{qc}:{wname}, m={history_size}, lr={lr}, line_search={line_search}, condition={matrix_cond},\
                     \n lbfgs={history_lbfgs[-1]},  new={history_lbfgs2[-1]}, \
                     \n lbfgs_time={lbfgs_time}, new_time={new_time}', \
                     size='small')
        plt.legend()
        save_name = wname + '_' + str(qc) + '_plot.png'
        
        for i in range(len(history_lbfgs3)):
            if np.isnan(history_lbfgs3[i]):
                history_lbfgs3[i] = history_lbfgs3[i-1]
                
        plt.savefig(out_fig_path+save_name)
        opt1.append(history_lbfgs)
        opt3.append(history_lbfgs3)
        opt2.append(history_lbfgs2)
        tl1.append(time_list1)
        tl3.append(time_list3)
        tl2.append(time_list2)
        it1.append(itl1)
        it2.append(itl2)
        it3.append(itl3)
        
        
        
        
        #---------------------------------------------------------------------------------- time
    totl1 = 0
    for item in opt1:
        totl1 = totl1 + len(item)
    al1 = totl1 / len(opt1)
    totl2 = 0
    for item in opt2:
        totl2 = totl2 + len(item)
    al2 = totl2 / len(opt2)
    
    max_len = 0
    for l in opt1:
        if len(l) > max_len:
            max_len = len(l)
    for l,t,i in zip(opt1, tl1, it1):
        while len(l) < max_len:
            l.append(l[-1])
            t.append(t[-1])
            i.append(i[-1])
    
    max_len = 0   
    for l in opt2:
        if len(l) > max_len:
            max_len = len(l)
    for l,t,i in zip(opt2, tl2, it2):
        while len(l) < max_len:
            l.append(l[-1])
            t.append(t[-1])
            i.append(i[-1])
    
    max_len = 0    
    for l in opt3:
        if len(l) > max_len:
            max_len = len(l)
    for l,t,i in zip(opt3, tl3, it3):
        while len(l) < max_len:
            l.append(l[-1])
            t.append(t[-1])
            i.append(i[-1])
    
    
    
    avg1 = []
    for m in range(len(opt1[0])):
        sum = 0
        for l in range(len(opt1)):
            sum = sum + opt1[l][m]
        avg1.append(sum/np.float64(len(opt1)))
    avg2 = []
    for m in range(len(opt2[0])):
        sum = 0
        for l in range(len(opt2)):
            sum = sum + opt2[l][m]
        avg2.append(sum/np.float64(len(opt2)))
    avg3 = []
    for m in range(len(opt3[0])):
        sum = 0
        for l in range(len(opt3)):
            sum = sum + opt3[l][m]
        avg3.append(sum/np.float64(len(opt3)))    
        
    tvg1 = []
    for m in range(len(tl1[0])):
        sum = 0
        for l in range(len(tl1)):
            sum = sum + tl1[l][m]
        tvg1.append(sum/np.float64(len(tl1)))
    tvg2 = []
    for m in range(len(tl2[0])):
        sum = 0
        for l in range(len(tl2)):
            sum = sum + tl2[l][m]
        tvg2.append(sum/np.float64(len(tl2)))
    tvg3 = []
    for m in range(len(tl3[0])):
        sum = 0
        for l in range(len(tl3)):
            sum = sum + tl3[l][m]
        tvg3.append(sum/np.float64(len(tl3)))   
    
    ivg1 = []
    for m in range(len(it1[0])):
        sum = 0
        for l in range(len(it1)):
            sum = sum + it1[l][m]
        ivg1.append(sum/np.float64(len(it1)))
    ivg2 = []
    for m in range(len(it2[0])):
        sum = 0
        for l in range(len(it2)):
            sum = sum + it2[l][m]
        ivg2.append(sum/np.float64(len(it2)))
    ivg3 = []
    for m in range(len(it3[0])):
        sum = 0
        for l in range(len(it3)):
            sum = sum + it3[l][m]
        ivg3.append(sum/np.float64(len(it3)))   
           
    
    std_loss1 = 0
    for l in opt1:
        std_loss1 = std_loss1 + np.power((l[-1] - avg1[-1]),2.0)
    std_loss1 = np.sqrt(std_loss1)
    std_loss2 = 0
    for l in opt2:
        std_loss2 = std_loss2 + np.power((l[-1] - avg2[-1]),2.0)
    std_loss2 = np.sqrt(std_loss2)
    std_loss3 = 0
    for l in opt3:
        std_loss3 = std_loss3 + np.power((l[-1] - avg3[-1]),2.0)
    std_loss3 = np.sqrt(std_loss3)
    
    
    std_time1 = 0
    for l in tl1:
        std_time1 = std_time1 + np.power((l[-1] - tvg1[-1]),2.0)
    std_time1 = np.sqrt(std_time1)
    std_time2 = 0
    for l in tl2:
        std_time2 = std_time2 + np.power((l[-1] - tvg2[-1]),2.0)
    std_time2 = np.sqrt(std_time2)
    std_time3 = 0
    for l in tl3:
        std_time3 = std_time3 + np.power((l[-1] - tvg3[-1]),2.0)
    std_time3 = np.sqrt(std_time3)
   
    def min_loss_time_cal(op,tl):
        min_loss_time = 0
        counter = 0
        for l,t in zip(op,tl):
            minval = l[-1]
            for val,ti in zip(l,t):
                counter = counter + 1
                if val==minval:
                    min_loss_time = ti + min_loss_time
                    break
        min_loss_time_avg = min_loss_time / len(opt1)
        min_loss_iter_avg = counter / len(opt1)
        
        min_loss_time = 0
        min_loss_iter_std = 0
        counter = 0
        for l,t in zip(op,tl):
            counter = 0
            minval = l[-1]
            for val,ti in zip(l,t):
                counter = counter + 1
                if val==minval:
                    min_loss_iter_std = min_loss_iter_std + np.power((counter - min_loss_iter_avg),2.0) 
                    min_loss_time = min_loss_time + np.power((ti - min_loss_time_avg),2.0) 
                    break
        min_loss_time_std = np.sqrt(min_loss_time)
        min_loss_iter_std = np.sqrt(min_loss_iter_std)
        
        return min_loss_time_avg, min_loss_time_std, min_loss_iter_avg, min_loss_iter_std
    
    min_loss_time_avg1, min_loss_time_std1, min_loss_iter_avg1, min_loss_iter_std1 = min_loss_time_cal(opt1,tl1)
    min_loss_time_avg2, min_loss_time_std2, min_loss_iter_avg2, min_loss_iter_std2 = min_loss_time_cal(opt2,tl2)
    min_loss_time_avg3, min_loss_time_std3, min_loss_iter_avg3, min_loss_iter_std3 = min_loss_time_cal(opt3,tl3)
    
    # torch.save()
    
    
    
    l1 = 'L-BFGS'
    l3 = 'BFGS'
    l2 = 'Proposed Method'
    wname = w_name.replace("\\", "/").split("/")[-1].split('.')[0]
    fig, axs = plt.subplots(1)
    axs.semilogy(tvg1, avg1, label=l1, c='black', linestyle='--', linewidth=1.5)
    axs.semilogy(tvg3, avg3, label=l3, c='black', linestyle='-.', linewidth=1.5)
    axs.semilogy(tvg2, avg2, label=l2, c='black', linewidth=1.5)
    axs.set_ylabel('Loss', fontsize = 12)
    axs.set_xlabel('Time', fontsize = 12)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    # plt.ylim(0, 30000)
    plt.xlim(-1.0, 50.0)
    
    fig.suptitle(f'{qc}:{wname}, m_lbfgs={history_size}, m_new={history_size2}, lr={lr}, line_search={line_search}\
                    \n lbfgs={avg1[-1]}, new={avg2[-1]}, lbfgs_iter={al1}, new_iter={al2} \
                    \n lbfgs_time={lbfgs_time:0.3f}/{lbfgs_time/al1:0.7f}, new_time={new_time:0.3f}/{new_time/al2:0.7f}', \
                    size='small')
    axs.grid(True, which="both", color='grey', linestyle=':', linewidth=0.7)
    plt.legend()
    # fig.set_size_inches(18.5, 10.5)
    # fig.savefig('test2png.png', dpi=100)
    save_name = wname + '_ls' + str(line_search) + '_iter' + str(opt_it) + '_n' + str(qc) + '_lr' + str(lr) + '_hs' + str(history_size) +  '_avg_time.png'
    plt.savefig(out_fig_path+save_name, dpi=500)
    
    
    # ------------------------------------------------------------------- iterations
    l1 = 'L-BFGS'
    l3 = 'BFGS'
    l2 = 'Proposed Method'
    wname = w_name.replace("\\", "/").split("/")[-1].split('.')[0]
    fig, axs = plt.subplots(1)
    axs.semilogy(ivg1, avg1, label=l1, c='black', linestyle='--', linewidth=1.5)
    axs.semilogy(ivg3, avg3, label=l3, c='black', linestyle='-.', linewidth=1.5)
    axs.semilogy(ivg2, avg2, label=l2, c='black', linewidth=1.5)
    axs.set_ylabel('Loss', fontsize = 12)
    axs.set_xlabel('Iterations', fontsize = 12)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    # plt.ylim(0, 30000)
    # plt.xlim(-0.02, 2.0)
    
    fig.suptitle(f'{qc}:{wname}, m_lbfgs={history_size}, m_new={history_size2}, lr={lr}, line_search={line_search}\
                    \n lbfgs={avg1[-1]}, new={avg2[-1]}, lbfgs_iter={al1}, new_iter={al2} \
                    \n lbfgs_time={lbfgs_time:0.3f}/{lbfgs_time/al1:0.7f}, new_time={new_time:0.3f}/{new_time/al2:0.7f}', \
                    size='small')
    axs.grid(True, which="both", color='grey', linestyle=':', linewidth=0.7)
    plt.legend()
    # fig.set_size_inches(18.5, 10.5)
    # fig.savefig('test2png.png', dpi=100)
    save_name = wname + '_ls' + str(line_search) + '_iter' + str(opt_it) + '_n' + str(qc) + '_lr' + str(lr) + '_hs' + str(history_size) +  '_avg_iter.png'
    plt.savefig(out_fig_path+save_name, dpi=500)
    
    save_name = save_name.split('.')[0]
    f = open(out_fig_path+save_name+'.txt', "w")
    
    f.write('L-BFGS \n')
    f.write(f'avg loss: {avg1[-1]}\n')
    f.write(f'std loss: {std_loss1}\n')
    f.write(f'avg time: {tvg1[-1]}\n')
    f.write(f'std time: {std_time1}\n')
    f.write(f'avg time 1e-6: {np.average(time_iter_sl[:,0])}\n')
    f.write(f'std time 1e-6: {np.std(time_iter_sl[:,0])}\n')
    f.write(f'avg iter 1e-6: {np.average(time_iter_sl[:,1])}\n')
    f.write(f'std iter 1e-6: {np.std(time_iter_sl[:,1])}\n\n')
    
    f.write('BFGS \n')
    f.write(f'avg loss: {avg3[-1]}\n')
    f.write(f'std loss: {std_loss3}\n')
    f.write(f'avg time: {tvg3[-1]}\n')
    f.write(f'std time: {std_time3}\n')
    f.write(f'avg time 1e-6: {np.average(time_iter_sl[:,4])}\n')
    f.write(f'std time 1e-6: {np.std(time_iter_sl[:,4])}\n')
    f.write(f'avg iter 1e-6: {np.average(time_iter_sl[:,5])}\n')
    f.write(f'std iter 1e-6: {np.std(time_iter_sl[:,5])}\n\n')
    
    f.write('New \n')
    f.write(f'avg loss: {avg2[-1]}\n')
    f.write(f'std loss: {std_loss2}\n')
    f.write(f'avg time: {tvg2[-1]}\n')
    f.write(f'std time: {std_time2}\n')
    f.write(f'avg time 1e-6: {np.average(time_iter_sl[:,2])}\n')
    f.write(f'std time 1e-6: {np.std(time_iter_sl[:,2])}\n')
    f.write(f'avg iter 1e-6: {np.average(time_iter_sl[:,3])}\n')
    f.write(f'std iter 1e-6: {np.std(time_iter_sl[:,3])}\n\n')
    
    f.close()
    if history_size > opt_it:
         history_size = 'Infinity'
    save_name = 'iter' + str(opt_it) + '_n' + str(qc) + '_hs' + str(history_size) +  '.pt'
    torch.save(tvg1, 'lbfgs_time_'+save_name)
    torch.save(avg1, 'lbfgs_hist_'+save_name)
    torch.save(ivg1, 'lbfgs_iter_'+save_name)
    
    
    torch.save(tvg3, 'bfgs_time_'+save_name)
    torch.save(avg3, 'bfgs_hist_'+save_name)
    torch.save(ivg3, 'bfgs_iter_'+save_name)
    
    
    torch.save(tvg2, 'new_time_'+save_name)
    torch.save(avg2, 'new_hist_'+save_name)
    torch.save(ivg2, 'new_iter_'+save_name)
    
    

        
        
        
        


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



