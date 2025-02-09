import torch
torch.backends.cudnn.benchmark = True
torch.nn.parallel.DistributedDataParallel = True
torch.jit.enable_onednn_fusion(True)
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile = False
torch.autograd.gradgradcheck = False
torch.set_float32_matmul_precision

import time
import torch.optim
import pycutest

from cute_utils import CUTEstProblem
from opzer.dirl_bfgs import DirLBFGS
from opzer.main_lbfgs import LBFGS
from opzer.bfgs import BFGS


import cute_utils

def wrapper_fn():
    cute_utils.CONVTIME1 = 0
    cute_utils.CONVTIME2 = 0
    
  
    problemName = 'EIGENALS'
    Ns_int = 100000
    history_size = 1000
    preallocate_memory = history_size + 2
    max_iter = 10000000



    Ns = {problemName: Ns_int}
    pycutest.print_available_sif_params(problemName)
    print(pycutest.problem_properties(problemName))

    if problemName in Ns:
        sifParams = {'N': Ns[problemName]}
    else:
        sifParams = {}
    problem = pycutest.import_problem(problemName)
    ndim = len(problem.vartype)
    print('number of dimensions=',ndim)

    
    model = CUTEstProblem(problem)
    

    line_search = 'strong_wolfe'

    optimizer = DirLBFGS(model.parameters(), 
                    lr=1, 
                    history_size=history_size,
                    preallocate_memory=preallocate_memory, 
                    line_search_fn=line_search, 
                    tolerance_grad=10e-4,
                    tolerance_change=10e-100,
                    restart=False, 
                    max_iter=1,
                    return_time=False,
                    debug_time=True)
    
    
    # optimizer = LBFGS(model.parameters(), 
    #                 lr=1, 
    #                 history_size=history_size,
    #                 line_search_fn=line_search, 
    #                 max_iter=1, 
    #                 tolerance_grad=10e-4,
    #                 tolerance_change=10e-20)
    
    # optimizer = torch.optim.LBFGS(model.parameters(), 
    #                                 lr=1, 
    #                                 history_size=history_size,
    #                                 line_search_fn=line_search, 
    #                                 max_iter=1, 
    #                                 tolerance_grad=10e-4,
    #                                 tolerance_change=10e-200)

    # optimizer = BFGS(model.parameters(), 
    #                 lr=1, 
    #                 line_search_fn=line_search, 
    #                 max_iter=1, 
    #                 tolerance_grad=1e-20,
    #                 tolerance_change=1e-20)



    loss_list = []
    t1 = time.time()
    non_gpu_time = 0
    stop_loss = 10e-7
    for n_iter in range(max_iter):
        def closure():
            optimizer.zero_grad()
            loss_fn = model()
           
            loss_fn.backward()
            return loss_fn

        obj, sc = optimizer.step(closure)
     
        if n_iter % 50 == 0:
            print(f'{n_iter}: {obj}')
        if sc:
            break
        # if obj < stop_loss:
        #     break
        # if n_iter > 345:
        #     break
        # if gtd < 10e-5:
            # break

        
    ttime = time.time() - t1    
    print(f'loss cal. time: {cute_utils.CONVTIME1}')
    convtime = cute_utils.CONVTIME2 - cute_utils.CONVTIME1
    print(f'conv. time: {convtime}')
    print(f'main time: {ttime-convtime}')
    print(f'problem: {problemName}, N: {problem.n}')
    print(f'final loss: {obj}')
    print(f'dir eval counter: {cute_utils.DIR_EVAL_COUNTER}\n')
    print(f'total time: {ttime}')
    print(f'iterations: {n_iter}')
    
    return convtime, ttime
   


i = 0
sum_tt = 0
sum_ct = 0
iterations = 1
while i < iterations:
    
    ct , tt = wrapper_fn()
    sum_tt = sum_tt + tt
    sum_ct = sum_ct + ct
    i = i + 1  
    print(i,100*'-')
    
print('avg. total time=',sum_tt / iterations)
print('avg. convert time=',sum_ct / iterations)
print('avg. optimization time=',(sum_tt-sum_ct) / iterations)
