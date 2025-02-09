import numpy as np
import time
import pycutest
from scipy.optimize import minimize



  
problemName = 'EIGENALS'
Ns_int = 100000
history_size = 1000



Ns = {problemName: Ns_int}
pycutest.print_available_sif_params(problemName)
print(pycutest.problem_properties(problemName))

if problemName in Ns:
    sifParams = {'N': Ns[problemName]}
else:
    sifParams = {}
problem = pycutest.import_problem(problemName)
ndim = len(problem.vartype)
print(f'{problemName} ndim={ndim}')

line_search = 'strong_wolfe'

boundes = []
for i in range(len(problem.x0)):
    boundes.append((-np.inf, np.inf))


t1 = time.time()
result = minimize(problem.obj, problem.x0, bounds=boundes, method='L-BFGS-B', jac=problem.grad, 
                  tol=1e-10000, options={'iprint':10, 'gtol':10e-4, 'ftol':10e-100000, 'maxcor':history_size})



print('histoty size=',history_size)
print(f'func.:{result['fun']}')
print(f'iter.:{result['nit']}')
print('time:',time.time()-t1)
ndim = len(problem.vartype)
print(f'{problemName} ndim={ndim}')
