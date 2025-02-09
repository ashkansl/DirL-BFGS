import numpy as np
from scipy.optimize import minpack2
from collections import deque
import time
import torch
CONVTIME = 0
USE_CUDA = True
if USE_CUDA:
    DEVICE = torch.device('cuda:0')
else:
    DEVICE = torch.device('cpu')
def compute_Cauchy_point(x, g, l, u, W, M, theta):
   
    eps_f_sec = 1e-30 
    t = torch.ones(len(x)) * torch.inf
    d = -g
    x_cp = x.clone()
    W = W.type(torch.DoubleTensor).to(DEVICE)
   
    F = torch.arange(len(x))

    t_old = 0
    F_i = 0
    b=F[0]
    t_min = t[b]
    Dt = t_min
    
    p = W.transpose(0,1) @ d.reshape(-1,1)
    c = torch.zeros(len(p), dtype=torch.float64, device=DEVICE)
    f_prime = -d.dot(d)
    f_second = -theta*f_prime- p.transpose(0,1) @ M @ p
    f_sec0 = f_second
    Dt_min = -f_prime/f_second

    while Dt_min>=Dt and F_i<len(F):
        if d[b]>0:
            x_cp[b] = u[b]
        elif d[b]<0:
            x_cp[b] = l[b]
        x_bcp = x_cp[b]
        
        zb = x_bcp - x[b]
        c += Dt*p
        W_b = W[b,:]
        g_b = g[b]
        
        f_prime += Dt*f_second+ g_b*(g_b+theta*zb-W_b.dot(M.dot(c)))
        f_second -= g_b*(g_b*theta+W_b.dot(M.dot(2*p+g_b*W_b)))
        f_second = min(f_second, eps_f_sec*f_sec0)
        
        Dt_min = -f_prime/f_second
        
        p += g_b*W_b
        d[b] = 0
        t_old = t_min
        F_i+=1
        
        if F_i<len(F):
            b=F[F_i]
            t_min = t[b]
            Dt = t_min-t_old
        else:
            t_min = np.inf

    Dt_min = 0 if Dt_min<0 else Dt_min
    t_old += Dt_min
    
    
    x_cp = x + t_old*d
    
    F = []
            
    c += Dt_min[0]*p[:,0]
    return {'xc':x_cp, 'c':c, 'F':F}

def minimize_model(x, xc, c, g, l, u, W, M, theta):
   
    invThet = 1.0/theta
    
    
    n = len(xc[0])

    
    free_vars = torch.arange(n)
        
    if len(free_vars) == 0:
        return {'xbar':xc}
    
    
    WTZ = W.T

    
    rHat = g + theta*(xc[0]-x) - W@(M@c)

    v = WTZ @ rHat
    v = M @ v
    
    N = invThet*WTZ @ W
    N = torch.eye(N.shape[0], dtype=torch.float64, device=DEVICE)- M@N
    v = torch.linalg.solve(N, v) #????N^-1@v
    
    dHat = -invThet * (rHat + invThet * (W @ v))
    
    alpha_star = 1
  
    d_star = alpha_star*dHat
    xbar = xc + d_star
   
    return {'xbar':xbar}


def max_allowed_steplength(x, d, l, u, max_steplength):
    
    max_stpl = max_steplength
    
    
    return max_stpl

def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    # ported from https://github.com/torch/optim/blob/master/polyinterp.lua
    # Compute bounds of interpolation area
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

    # Code for most common case: cubic interpolation of 2 points
    #   w/ function and derivative values for both
    # Solution in this case (where x2 is the farthest point):
    #   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
    #   d2 = sqrt(d1^2 - g1*g2);
    #   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
    #   t_new = min(max(min_pos,xmin_bound),xmax_bound);
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1 ** 2 - g1 * g2
    if d2_square >= 0:
        d2 = d2_square.sqrt()
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, xmin_bound), xmax_bound)
    else:
        return (xmin_bound + xmax_bound) / 2.


def _strong_wolfe(obj_func,
                  dobj,
                  x,
                  t,
                  d,
                  f,
                  g,
                  gtd,
                  c1=1e-4,
                  c2=0.9,
                  tolerance_change=1e-9,
                  max_ls=25):
    # ported from https://github.com/torch/optim/blob/master/lswolfe.lua
    d_norm = d.abs().max()
    g = g.clone(memory_format=torch.contiguous_format)
    # evaluate objective and gradient using initial step
    f_new, g_new = obj_func(x, t, d)
    # print(f'fnew: {f_new}, gnew: {g_new}')
    ls_func_evals = 1
    gtd_new = g_new.dot(d)

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0
    while ls_iter < max_ls:
        # print(f'lsiter:{ls_iter}')
        # check conditions
        if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if abs(gtd_new) <= -c2 * gtd:
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break

        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        # interpolate
        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = _cubic_interpolate(
            t_prev,
            f_prev,
            gtd_prev,
            t,
            f_new,
            gtd_new,
            bounds=(min_step, max_step))

        # next step
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        gtd_prev = gtd_new
        f_new, g_new = obj_func(x, t, d)
        # print(f'(first while) fnew: {f_new}, gnew: {g_new}')
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

    # reached max number of iterations?
    if ls_iter == max_ls:
        bracket = [0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]

    # zoom phase: we now have a point satisfying the criteria, or
    # a bracket around it. We refine the bracket until we find the
    # exact point satisfying the criteria
    insuf_progress = False
    # find high and low points in bracket
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
    while not done and ls_iter < max_ls:
        # line-search bracket is so small
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
            break

        # compute new trial value
        t = _cubic_interpolate(bracket[0], bracket_f[0], bracket_gtd[0],
                               bracket[1], bracket_f[1], bracket_gtd[1])

        # test that we are making sufficient progress:
        # in case `t` is so close to boundary, we mark that we are making
        # insufficient progress, and if
        #   + we have made insufficient progress in the last step, or
        #   + `t` is at one of the boundary,
        # we will move `t` to a position which is `0.1 * len(bracket)`
        # away from the nearest boundary point.
        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            # interpolation close to boundary
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                # evaluate at 0.1 away from boundary
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # Evaluate new point
        f_new, g_new = obj_func(x, t, d)
        # print(f'(second while) fnew: {f_new}, gnew: {g_new}')
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd:
                # Wolfe conditions satisfied
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                # old high becomes new low
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            # new point becomes new low
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[low_pos] = gtd_new

    # return stuff
    t = bracket[low_pos]
    f_new = bracket_f[low_pos]
    g_new = bracket_g[low_pos]
    return f_new, g_new, t, ls_func_evals

def line_search(x0, f0, g0, d, above_iter, max_steplength,\
                fct_f, fct_grad,\
                alpha = 1e-4, beta = 0.9,\
                xtol_minpack = 1e-5, max_iter = 30):
    """
        Finds a step that satisfies a sufficient decrease condition and a curvature condition.

        The algorithm is designed to find a step that satisfies the sufficient decrease condition 
        
              f(x0+stp*d) <= f(x0) + alpha*stp*\langle f'(x0),d\rangle,
        
        and the curvature condition
        
              abs(f'(x0+stp*d)) <= beta*abs(\langle f'(x0),d\rangle).
        
        If alpha is less than beta and if, for example, the functionis bounded below, then
        there is always a step which satisfies both conditions. 

    :param x0: starting point 
    :type x0: np.array
    
    :param f0: f(x0) 
    :type f0: float
    
    :param g0: f'(x0), gradient 
    :type g0: np.array
    
    :param d: search direction
    :type d: np.array
    
    :param above_iter: current iteration in optimization process
    :type above_iter: integer
    
    :param max_steplength: maximum steplength allowed 
    :type max_steplength: float 
    
    :param fct_f: callable, function f(x) 
    :type fct_f: function returning float
    
    :param fct_grad: callable, function f'(x) 
    :type fct_grad: function returning np.array
    
    :param alpha, beta: parameters of the decrease and curvature conditions 
    :type alpha, beta: floats
    
    :param xtol_minpack: tolerance used in minpack2.dcsrch
    :type xtol_minpack: float
    
    :param max_iter: number of iteration allowed for finding a steplength
    :type max_iter: integer
    
    :return: optimal steplength meeting both decrease and curvature condition 
    :rtype: float
    
    
    

    .. seealso:: 
        
       [minpack] scipy.optimize.minpack2.dcsrch

       [1] R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, ``A limited
       memory algorithm for bound constrained optimization'',
       SIAM J. Scientific Computing 16 (1995), no. 5, pp. 1190--1208.

       [2] C. Zhu, R.H. Byrd, P. Lu, J. Nocedal, ``L-BFGS-B: FORTRAN
       Subroutines for Large Scale Bound Constrained Optimization''
       Tech. Report, NAM-11, EECS Department, Northwestern University,
       1994.
    """
    
    steplength_0 = 1 if max_steplength > 1 else 0.5*max_steplength
    f_m1 = f0
    # dphi = g0.dot(d)
    dphi = g0 @ d.T
    dphi_m1 = dphi
    i = 0

    if(above_iter == 0):
        max_steplength = 1.0
        steplength_0 = min(1.0/torch.sqrt(d@d.T), 1.0)

    isave = np.zeros((2,), np.intc)
    dsave = np.zeros((13,), float)
    task = b'START'
    if type(steplength_0) == int or type(steplength_0) == float:
        steplength_0 = steplength_0
    else:
        steplength_0 = steplength_0.item()
    dphi_m1 = dphi_m1.item()
    
    global CONVTIME
    ctim = time.time()
    x00 = x0
    dd = d[0]
    CONVTIME = CONVTIME + time.time() - ctim
    
    
    while i<max_iter:
        steplength, f0, dphi, task = minpack2.dcsrch(steplength_0, f_m1, dphi_m1,
                                                   alpha, beta, xtol_minpack, task,
                                                   0, max_steplength, isave, dsave)
        if task[:2] == b'FG':
            steplength_0 = steplength
            f_m1 = fct_f(x00 + steplength*(dd))
            dphi_m1 = fct_grad(x00 + steplength*dd).dot(dd)
        else:
            break
        i = i + 1 
    else:
        # max_iter reached, the line search did not converge
        steplength = None
    
    # print('iiiiiiiiii=',i)
    if task[:5] == b'ERROR' or task[:4] == b'WARN':
        if task[:21] != b'WARNING: STP = STPMAX':
            print(task)
            steplength = None  # failed
        
    return steplength


def update_SY(sk, yk, S, Y, m,\
              W, M, thet,\
              eps = 2.2e-16):
    """
        Update lists S and Y, and form the L-BFGS Hessian approximation thet, W and M.

    :param sk: correction in x = new_x - old_x 
    :type sk: np.array
    
    :param yk: correction in gradient = f'(new_x) - f'(old_x) 
    :type yk: np.array
    
    :param S, Y: lists defining the L-BFGS matrices, updated during process (IN/OUT)
    :type S, Y: list
    
    :param m: Maximum size of lists S and Y: keep in memory only m previous iterations
    :type m: integer
    
    :param W, M: L-BFGS matrices 
    :type W, M: np.array
    
    :param thet: L-BFGS float parameter 
    :type thet: float
    
    :param eps: Positive stability parameter for accepting current step for updating matrices.
    :type eps: float >0
    
    :return: updated [W, M, thet]
    :rtype: tuple 

    .. seealso:: 

       [1] R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, ``A limited
       memory algorithm for bound constrained optimization'',
       SIAM J. Scientific Computing 16 (1995), no. 5, pp. 1190--1208.

       [2] C. Zhu, R.H. Byrd, P. Lu, J. Nocedal, ``L-BFGS-B: FORTRAN
       Subroutines for Large Scale Bound Constrained Optimization''
       Tech. Report, NAM-11, EECS Department, Northwestern University,
       1994.
    """
    ddd = len(sk)
    sTy = sk.dot(yk)
    yTy = yk.dot(yk)
    if (sTy > eps*yTy):
        S.append(sk.reshape(-1,1))
        Y.append(yk.reshape(-1,1))
        if len(S) > m :
            S.pop(0)
            Y.pop(0)
        Sarray = torch.cat(S,1).reshape(ddd,-1)
        Yarray = torch.cat(Y,1).reshape(ddd,-1)
        STS = Sarray.T @ Sarray
        L = Sarray.T @ Yarray
        D = torch.diag(-torch.diag(L))
        L = torch.tril(L, -1)
        
        thet = yTy/sTy
        W = torch.hstack([Yarray, thet*Sarray])
        M = torch.linalg.inv(torch.hstack([torch.vstack([D, L]), torch.vstack([L.T, thet*STS])]))

    return [W, M, thet]


def L_BFGS_B(x0, f, df, l, u, m=10,\
             epsg = 1e-20, epsf = 1e-20, max_iter = 50,\
             alpha_linesearch = 1e-4, beta_linesearch = 0.9,\
             max_steplength = 1e8,\
             xtol_minpack = 1e-5, max_iter_linesearch = 30, eps_SY = 2.2e-16, lsrch='strong_wolfe'):
    
    # debug_time = []
    """
       Solves bound constrained optimization problems by using the compact formula 
       of the limited memory BFGS updates. 

    :param x0: initial guess
    :type sk: np.array
    
    :param f: cost function to optimize f(x)
    :type f: function returning float
    
    :param df: gradient of cost function to optimize f'(x)
    :type df: function returning np.array
    
    :param l: the lower bound of x 
    :type l: np.array
    
    :param u: the upper bound of x 
    :type u: np.array
    
    :param m: Maximum size of lists for L-BFGS Hessian approximation 
    :type m: integer
    
    :param epsg: Tolerance on projected gradient: programs converges when
                P(x-g, l, u)<epsg.
    :type epsg: float
    
    :param epsf: Tolerance on function change: programs ends when (f_k-f_{k+1})/max(|f_k|,|f_{k+1}|,1) < epsf * epsmch, where 
                epsmch is the machine precision. 
    :type epsf: float
    
    :param alpha_linesearch, beta_linesearch: Parameters for linesearch. 
                                              See ``alpha`` and ``beta`` in :func:`line_search`
    :type alpha_linesearch, beta_linesearch: float 

    :param max_steplength: Maximum steplength allowed. See ``max_steplength`` in :func:`max_allowed_steplength`
    :type max_steplength: float
    
    :param xtol_minpack: Tolerence used by minpack2. See ``xtol_minpack`` in :func:`line_search`
    :type xtol_minpack: float
    
    :param max_iter_linesearch: Maximum number of trials for linesearch. 
                                See ``max_iter_linesearch`` in :func:`line_search`
    :type max_iter_linesearch: integer 
    
    :param eps_SY: Parameter used for updating the L-BFGS matrices. See ``eps`` in :func:`update_SY`
    :type eps_SY: float
    
    :return: dict containing:
            - 'x': optimal point
            - 'f': optimal value at x
            - 'df': gradient f'(x)
    :rtype: dict 


    ..todo Check matrices update and different safeguards may be missing
    
    .. seealso:: 
       Function tested on Rosenbrock and Beale function with different starting points. All tests passed. 
        
       [1] R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, ``A limited
       memory algorithm for bound constrained optimization'',
       SIAM J. Scientific Computing 16 (1995), no. 5, pp. 1190--1208.

       [2] C. Zhu, R.H. Byrd, P. Lu, J. Nocedal, ``L-BFGS-B: FORTRAN
       Subroutines for Large Scale Bound Constrained Optimization''
       Tech. Report, NAM-11, EECS Department, Northwestern University,
       1994.
    """
    global CONVTIME
    n = len(x0)
   
    x = torch.clip(x0, l[0], u[0])
    k=0
    S = []
    Y = []
    W = torch.zeros([n, 1], dtype=torch.double, device=DEVICE)
    M = torch.zeros([1, 1], dtype=torch.double, device=DEVICE)
    theta = 1
    epsmch = np.finfo(1.0).resolution
    
    ctim = time.time()
    xcpu = x
    CONVTIME = CONVTIME + time.time() - ctim
    
    f0 = f(xcpu)
    g = df(xcpu)

    ctim = time.time()
    g = g
    CONVTIME = CONVTIME + time.time() - ctim
    
    i=0
    timel = torch.zeros(10)
    while torch.max(torch.abs(torch.clip(x-g,l[0],u[0])-x))>epsg and i<max_iter:
        oldf0 = f0
        oldx = x.clone()
        oldg = g.clone()

        t0 = time.time()
        dictCP = compute_Cauchy_point(x, g, l, u, W, M, theta)
        timel[0] = timel[0] + (time.time() - t0)

        t0 = time.time()
        dictMinMod = minimize_model(x, dictCP['xc'], dictCP['c'], g, l, u, W, M, theta)
        timel[1] = timel[1] + (time.time() - t0)
        
        d = dictMinMod['xbar'] - x

        t0 = time.time()
        max_stpl = max_steplength # max_allowed_steplength(x, d, l, u, max_steplength) TODO
        timel[2] = timel[2] + (time.time() - t0)

        def obj_func(x, t, d):
            global CONVTIME
            xnew = x + t*d
            
            ctim = time.time()
            xnew_nd = xnew
            CONVTIME = CONVTIME + time.time() - ctim
            
            fnew = f(xnew_nd)
            dfnew = df(xnew_nd)
            
            ctim = time.time()
            dfnew = dfnew
            CONVTIME = CONVTIME + time.time() - ctim
            
            return fnew, dfnew
        
        t0 = time.time()
        if lsrch == 'scipy':
            steplength = line_search(x, f0, g, d, i, max_stpl, f, df,\
                    alpha_linesearch, beta_linesearch,\
                    xtol_minpack, max_iter_linesearch)
        
        elif lsrch == 'strong_wolfe':
            gtd = oldg.dot(d[0]) 
            steplength_0 = 1 if max_stpl > 1 else 0.5*max_stpl
            if(i == 0):
                max_steplength = 1.0
                steplength_0 = min(1.0/torch.sqrt(d@d.T), 1.0)[0][0]
            f_new, g_new, steplength, ls_func_evals = _strong_wolfe(obj_func=obj_func, 
                                                                    dobj=df, 
                                                                    x=x, 
                                                                    t=steplength_0, 
                                                                    d=d[0], 
                                                                    f=f0, 
                                                                    g=oldg, 
                                                                    gtd=gtd)
    
        timel[3] = timel[3] + (time.time() - t0)
        
        if steplength==None:
            if len(S)==0:
                #Hessian already rebooted: abort.
                print("Error: can not compute new steplength : abort")
                return {'x':x, 'f':f(x), 'df':df(x)}
            else:
                #Reboot BFGS-Hessian:
                S.clear()
                Y.clear()
                W = np.zeros([n, 1])
                M = np.zeros([1, 1])
                theta = 1
        else:
            x += steplength*d[0]
            x = x.double()
            
            ctim = time.time()
            xcpu = x
            CONVTIME = CONVTIME + time.time() - ctim
        
            f0 = f(xcpu)
            g = df(xcpu)

            ctim = time.time()
            g = g
            CONVTIME = CONVTIME + time.time() - ctim
            
            [W, M, theta] = update_SY(x-oldx, g-oldg, S, Y, m,\
                           W, M, theta, eps_SY)
            timel[4] = timel[4] + (time.time() - ctim)
            if i%1000==0:
                print(f'iter{i+1}: {f0}')
            
            # print("Iteration #%d (max: %d): ||x||=%.3e, f(x)=%.3e, ||df(x)||=%.3e, cdt_arret=%.3e (eps=%.3e)"%\
            #       (i, max_iter, torch.linalg.norm(x, torch.inf), f0, torch.linalg.norm(torch.from_numpy(g).to(DEVICE), torch.inf),\
            #        torch.max(torch.abs(torch.clip(x-torch.from_numpy(g).to(DEVICE),l[0],u[0])-x)), epsg))
            # if((oldf0-f0)/max(abs(oldf0),abs(f0),1)<epsmch * epsf):
            #     print("Relative reduction of f below tolerence: abort.")
            #     break

            i += 1
        
        if f0 < 10e-7:
            return f0, timel, i
        
    if i==max_iter:
        print("Maximum iteration reached.")
        
    return f0, timel, i