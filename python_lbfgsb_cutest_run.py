
import numpy as np
import torch
import opzer.python_lbfgsb_cutest as lbfgsb_gpu
import pycutest
import time

if lbfgsb_gpu.USE_CUDA :
    DEVICE = torch.device('cuda:0')
else:
    DEVICE = torch.device('cpu')





problemName = 'OSCIGRAD'
Ns_int = 100000
history_size = 40

Ns = {problemName: Ns_int}
pycutest.print_available_sif_params(problemName)
print(pycutest.problem_properties(problemName))
if problemName in Ns:
    sifParams = {'N': Ns[problemName]}
else:
    sifParams = {}
problem = pycutest.import_problem(problemName, sifParams=sifParams)

l = torch.tensor([-np.inf], dtype=torch.double, device=DEVICE)
u = torch.tensor([np.inf], dtype=torch.double, device=DEVICE)
x0 = torch.tensor(problem.x0, dtype=torch.double, device=DEVICE)
t0 = time.time()
# scipy
# strong_wolfe
result, tt, iters = lbfgsb_gpu.L_BFGS_B(f=problem.obj, x0=x0, df=problem.grad, m=history_size, l=l, u=u, max_iter = 700000, lsrch='scipy')
ttt = time.time()-t0
print('iters: ', iters)
print(result)
print('time:', tt)
print(f'total time: {ttt}')
print(f'conv. time: {lbfgsb_gpu.CONVTIME}')
print(f'main time: {ttt - lbfgsb_gpu.CONVTIME}')

