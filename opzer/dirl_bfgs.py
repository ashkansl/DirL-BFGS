import sys
import time

# append "optim" path sys.path.append('path/to/torch/optim/')  # TODO
sys.path.append("/home/user/pyenvs/env1/lib/python3.12/site-packages/torch/optim/")
sys.path.append("/home/sadeghi/python_envs/env2/lib/python3.12/site-packages/torch/optim/")



import torch
from functools import reduce
from optimizer import Optimizer
import numpy as np

DEVICE = None

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


class DirLBFGS(Optimizer):
    """Implements L-BFGS algorithm, heavily inspired by `minFunc
    <https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html>`_.

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.

    Args:
        lr (float): learning rate (default: 1)
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-5).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 100).
        line_search_fn (str): either 'strong_wolfe' or None (default: None).
    """

    def __init__(self,
                 params,
                 lr=1,
                 restart_lr=0.1,
                 max_iter=20,
                 max_eval=None,
                 tolerance_grad=1e-7,  # 1e-7, # TODO
                 tolerance_proj_grad=10e-5,
                 tolerance_change=1e-9,
                 tolerance_change_gtd=True,
                 history_size=100,
                 preallocate_memory=0,
                 line_search_fn=None,
                 fast_version=True,
                 update_gamma=True,
                 restart=False,
                 return_time=False,
                 debug_time=False,
                 dtype=torch.double):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        # if preallocate_memory<history_size and history_size < 100: 
        #     preallocate_memory = history_size + 1
        #     print('WARNING: The pre-allocated memory size increased to history size + 10.')
        defaults = dict(
            lr=lr,
            restart_lr=restart_lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_proj_grad=tolerance_proj_grad,
            tolerance_change=tolerance_change,
            tolerance_change_gtd=tolerance_change_gtd,
            history_size=history_size,
            preallocate_memory=preallocate_memory,
            line_search_fn=line_search_fn,
            fast_version=fast_version,
            update_gamma=update_gamma,
            restart=restart,
            return_time=return_time,
            debug_time=debug_time,
            dtype=dtype)
        super(DirLBFGS, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self.dtype = self._params[0].dtype
        # global USE_CUDA
        # USE_CUDA = self._params[0].is_cuda
        global DEVICE
        if self._params[0].is_cuda:
            DEVICE = 'cuda'
        else:
            DEVICE = 'cpu'
        
        self._numel_cache = None
        self.total_time = 0
        self.debug_time = np.zeros(15)


    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self): # changed
        return self._params[0].grad
        # views = []
        # for p in self._params:
        #     if p.grad is None:
        #         view = p.new(p.numel()).zero_()
        #     elif p.grad.is_sparse:
        #         view = p.grad.to_dense().view(-1)
        #     else:
        #         view = p.grad.view(-1)
        #     views.append(view)
        # return torch.cat(views, 0)

    def _add_grad(self, step_size, update): # changed
        offset = 0
        # for p in self._params:
        numel = self._params[0].numel()
        # view as to avoid deprecated pointwise semantics
        self._params[0].add_(update[offset:offset + numel].view_as(self._params[0]), alpha=step_size)
        offset += numel
        # assert offset == self._numel()

    def _clone_param(self): # changed
        return [self._params[0].clone(memory_format=torch.contiguous_format)]
        # return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data): # changed
        self._params[0].copy_(params_data[0])
        # for p, pdata in zip(self._params, params_data):
        #     p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        # print(f'-------\np: {self._params[0].data} \ng: {self._params[0].grad}')
        self._add_grad(t, d)
        # print(f'-------\np: {self._params[0].data} \ng: {self._params[0].grad}')
        loss = float(closure())
        flat_grad = self._gather_flat_grad()
        # print(f'params before set params: \np: {self._params[0].data} \ng: {self._params[0].grad}')
        self._set_param(x)
        # print(f'-------\np: {self._params[0].data} \ng: {self._params[0].grad}')
        # print(20*'-')
        return loss, flat_grad

      

    @torch.no_grad()
    def step(self, closure):
        """Performs a single optimization step.

        Args:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        t0 = time.time()
        group = self.param_groups[0]
        lr = group['lr']
        restart_lr = group['restart_lr']
        max_iter = group['max_iter']
        max_eval = group['max_eval']
        tolerance_grad = group['tolerance_grad']
        tolerance_proj_grad = group['tolerance_proj_grad']       
        tolerance_change = group['tolerance_change']
        tolerance_change_gtd = group['tolerance_change_gtd']
        line_search_fn = group['line_search_fn']
        history_size = group['history_size']
        preallocate_memory = group['preallocate_memory']
        fast_version = group['fast_version']
        update_gamma = group['update_gamma']
        restart = group['restart']
        return_time = group['return_time']
        debug_time = group['debug_time']
        dtype = self.dtype

        # show = group['show']
        # stop_loss = group['stop_loss']

        # NOTE: LBFGS has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[self._params[0]]
        state.setdefault('func_evals', 0)
        state.setdefault('n_iter', 0)
        state.setdefault('all_iter', 0)
        
        # self.debug_time[0] = self.debug_time[0] + (time.time()-t0)


        # t0 = time.time()
        # evaluate initial f(x) and df/dx
        orig_loss = closure()
        # tt1 = time.time()
        self.debug_time[1] = self.debug_time[1] + (time.time()-t0)
        
        # t0 = time.time()
        
        loss = float(orig_loss)
        # if loss > 10000:
        #     dddd = 0
        current_evals = 1
        state['func_evals'] += 1

        flat_grad = self._gather_flat_grad()
        opt_cond = flat_grad.abs().max() <= tolerance_grad
        # self.debug_time[2] = self.debug_time[2] + (time.time()-t0)
       
        # optimal condition

        if opt_cond:
            return orig_loss, opt_cond



        # tensors cached in state (for tracing)
        d = state.get('d')
        t = state.get('t')
        # global old_dirs, old_stps, ro, H_diag, prev_flat_grad, prev_loss, betas, gbetas, v_vectors, gv_vectors, u_vectors, gu_vectors, gama, gama_list
        old_dirs = state.get('old_dirs')
        old_stps = state.get('old_stps')
        ro = state.get('ro')
        H_diag = state.get('H_diag')
        prev_flat_grad = state.get('prev_flat_grad')
        prev_loss = state.get('prev_loss')
        betas = state.get('betas')
        gbetas = state.get('gbetas')
        v_vectors = state.get('v_vectors')
        gv_vectors = state.get('gv_vectors')
        u_vectors = state.get('u_vectors')
        gu_vectors = state.get('gu_vectors')
        gama = state.get('gama')
        gama_list = state.get('gama_list')
     
        
        ys = 0
        n_iter = 0
        sniter = state['n_iter']
        tempH_G = 0
        push_iter = 10
        gamma_push_iter = 5
        nogamma_push_iter = 3
        end_mindx = nogamma_push_iter*(sniter)
        gend_mindx = gamma_push_iter*(sniter)

        # optimize for a max of max_iter iterations
        ys = 100
        ys_threshold = 1e-10
        ys_restart = 0

        

        while n_iter < max_iter:
            # keep track of nb of iterations
            n_iter += 1
            state['n_iter'] += 1
            state['all_iter'] += 1
            ############################################################
            # compute gradient descent direction
            ############################################################
            if state['n_iter'] == 1:
                
                d = flat_grad.neg()
                old_dirs = []
                old_stps = []
                ro = []
                H_diag = 1  
                gama_list = []
                gama_list.append(H_diag)
                if preallocate_memory == 0:
                    betas = torch.zeros((nogamma_push_iter, 1), device=DEVICE, dtype=dtype)
                    gbetas = torch.zeros((gamma_push_iter, 1), device=DEVICE, dtype=dtype)
                    v_vectors = torch.zeros((nogamma_push_iter, d.shape[0]), device=DEVICE, dtype=dtype)
                    gv_vectors = torch.zeros((gamma_push_iter, d.shape[0]), device=DEVICE, dtype=dtype)
                    u_vectors = torch.zeros((nogamma_push_iter, d.shape[0]), device=DEVICE, dtype=dtype)
                    gu_vectors = torch.zeros((gamma_push_iter, d.shape[0]), device=DEVICE, dtype=dtype)
                else:
                    del v_vectors, gv_vectors, u_vectors, gu_vectors, betas, gbetas
                    torch.cuda.empty_cache()
                    mem = nogamma_push_iter*preallocate_memory
                    gmem = gamma_push_iter*preallocate_memory
                    betas = torch.zeros((mem, 1), device=DEVICE, dtype=dtype)
                    gbetas = torch.zeros((gmem, 1), device=DEVICE, dtype=dtype)
                    v_vectors = torch.zeros((mem, d.shape[0]), device=DEVICE, dtype=dtype)
                    gv_vectors = torch.zeros((gmem, d.shape[0]), device=DEVICE, dtype=dtype)
                    u_vectors = torch.zeros((mem, d.shape[0]), device=DEVICE, dtype=dtype)
                    gu_vectors = torch.zeros((gmem, d.shape[0]), device=DEVICE, dtype=dtype)

                # h_main = torch.eye(d.shape[0], d.shape[0], device=DEVICE)

            
            else:
                # t0 = time.time()
                y = flat_grad.sub(prev_flat_grad)
                s = d.mul(t)
                ys = y.dot(s)  # y*s
                
                newbeta = 0
                newu = 0
                newv = 0
                gnewbeta = 0
                gnewu = 0
                gnewv = 0
                
                

                if ys > ys_threshold:
                    # updating memory
                    if len(old_dirs) >= history_size:
                        # shift history by one (limited-memory)
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)


                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1. / ys)

                    if len(gama_list) >= history_size:  # TODO history_size+1
                        if len(betas) > 1:
                            betas = betas[nogamma_push_iter:]
                            gbetas = gbetas[gamma_push_iter:]
                            v_vectors = v_vectors[nogamma_push_iter:]
                            gv_vectors = gv_vectors[gamma_push_iter:]
                            u_vectors = u_vectors[nogamma_push_iter:]
                            gu_vectors = gu_vectors[gamma_push_iter:]
                            gama_list.pop(0)
                    
                    if len(gama_list)+1 >= len(betas)/3:
                         
                        mem = nogamma_push_iter*preallocate_memory
                        gmem = gamma_push_iter*preallocate_memory
                        newbeta = torch.zeros((mem, 1), device=DEVICE, dtype=dtype)
                        gnewbeta = torch.zeros((gmem, 1), device=DEVICE, dtype=dtype)
                        newv = torch.zeros((mem, d.shape[0]), device=DEVICE, dtype=dtype)
                        gnewv = torch.zeros((gmem, d.shape[0]), device=DEVICE, dtype=dtype)
                        newu = torch.zeros((mem, d.shape[0]), device=DEVICE, dtype=dtype)
                        gnewu = torch.zeros((gmem, d.shape[0]), device=DEVICE, dtype=dtype)

                        betas = torch.cat((betas, newbeta), 0)
                        u_vectors = torch.cat((u_vectors, newu), 0)
                        v_vectors = torch.cat((v_vectors, newv), 0)
                        gbetas = torch.cat((gbetas, gnewbeta), 0)
                        gu_vectors = torch.cat((gu_vectors, gnewu), 0)
                        gv_vectors = torch.cat((gv_vectors, gnewv), 0)
               



                    if state['n_iter'] == 2 or update_gamma:
                        H_diag = ys / y.dot(y)  
                        gama = H_diag

                    gama_list.append(gama)

                    
                    
                        
                    # t0 = time.time()

                    u_tensor = u_vectors
                    gu_tensor = gu_vectors
                    v_tensor = v_vectors
                    gv_tensor = gv_vectors
                    betas_tensor = betas
                    gbetas_tensor = gbetas

                    yk = old_dirs[-1]
                    k_indx = -1
                    # self.debug_time[3] = self.debug_time[3] + (time.time()-t0)
                    
                    # t0 = time.time()
                    memlen = len(gama_list)
                    dimn = d.shape[0]

                    yk_uT = yk @ u_tensor[0:end_mindx].T # n*3m
                    yk_guT = yk @ gu_tensor[0:gend_mindx].T # n*5m
                    tempv = torch.sum((betas_tensor[0:end_mindx].T * yk_uT) * v_tensor[0:end_mindx].T, 1).view(-1) #  3m+(n*3m)
                    gtempv = torch.sum((gbetas_tensor[0:gend_mindx].T * yk_guT) * gv_tensor[0:gend_mindx].T, 1).view(-1) #  5m+(n*5m)
                    tempu = tempv # torch.sum(betas_tensor.T * ((v_tensor @ yk.T) * u_tensor.T), 1).view(-1, 1)
                    gtempu = gtempv # torch.sum(gbetas_tensor.T * ((gv_tensor @ yk.T) * gu_tensor.T), 1).view(-1, 1)
                    tempb = torch.sum((betas_tensor[0:end_mindx].T * yk_uT) * (v_tensor[0:end_mindx] @ yk.T), 1) # 3m + 3m*n + 3m
                    gtempb = torch.sum((gbetas_tensor[0:gend_mindx].T * yk_guT) * (gv_tensor[0:gend_mindx] @ yk.T), 1) # 5m + 5m*n + 5m

                    # self.debug_time[4] = self.debug_time[4] + (time.time()-t0)

                    # t0 = time.time()
                    

                    gbetas[gend_mindx:gend_mindx+4] = -ro[k_indx]
                    gu_vectors[gend_mindx] = old_stps[k_indx]
                    gv_vectors[gend_mindx] = old_dirs[k_indx]

                    # gbetas[gend_mindx+1] = -ro[k_indx]
                    gu_vectors[gend_mindx+1] = old_dirs[k_indx]
                    gv_vectors[gend_mindx+1] = old_stps[k_indx]

                    # gbetas[gend_mindx+2] = -ro[k_indx]
                    gu_vectors[gend_mindx+2] = old_stps[k_indx]
                    gv_vectors[gend_mindx+2] = gtempv  

                    # gbetas[gend_mindx+3] = -ro[k_indx]
                    gu_vectors[gend_mindx+3] = gtempu  
                    gv_vectors[gend_mindx+3] = old_stps[k_indx]

                    gbetas[gend_mindx+4] = ro[k_indx] * ro[k_indx] * ((old_dirs[k_indx] @ old_dirs[k_indx]) + gtempb)
                    gu_vectors[gend_mindx+4] = old_stps[k_indx]
                    gv_vectors[gend_mindx+4] = old_stps[k_indx]


                    betas[end_mindx:end_mindx+2] = -ro[k_indx]
                    u_vectors[end_mindx] = old_stps[k_indx]
                    v_vectors[end_mindx] = tempv  

                    # betas[end_mindx+1] = -ro[k_indx]
                    u_vectors[end_mindx+1] = tempu  
                    v_vectors[end_mindx+1] = old_stps[k_indx]

                    betas[end_mindx+2] = (ro[k_indx] * ro[k_indx] * tempb) + ro[k_indx] 
                    u_vectors[end_mindx+2] = old_stps[k_indx]
                    v_vectors[end_mindx+2] = old_stps[k_indx]

                
                    # self.debug_time[5] = self.debug_time[5] + (time.time()-t0)

               
                u_tensor = u_vectors
                v_tensor = v_vectors
                betas_tensor = betas
                gu_tensor = gu_vectors
                gv_tensor = gv_vectors
                gbetas_tensor = gbetas

                tempH_G11 = 0
                tempH_G22 = 0
                
                # if fast_version:
                emx = end_mindx+nogamma_push_iter
                gemx = gend_mindx+gamma_push_iter
                # t8 = time.time()

                tempH_G11 = torch.sum((gbetas_tensor[0:gemx]*(gv_tensor[0:gemx] @ flat_grad.reshape(-1, 1)))*gu_tensor[0:gemx], 0) # 5m + 5mn +  5mn
                tempH_G22 = torch.sum((betas_tensor[0:emx]*(v_tensor[0:emx] @ flat_grad.reshape(-1, 1)))*u_tensor[0:emx], 0) # 3m + 3mn + 3mn
                tempH_G = tempH_G22 + (tempH_G11 + flat_grad) * gama_list[-1]
                
                d = -tempH_G

                # if ys == 0: # TODO
                #     ys_restart = 1

                # self.debug_time[8] = self.debug_time[8] + (time.time()-t8)


            if prev_flat_grad is None:
                prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
            else:
                prev_flat_grad.copy_(flat_grad)
            prev_loss = loss


            ############################################################
            # compute step length
            ############################################################
            
            # t9 = time.time()

            # reset initial guess for step size
            if state['n_iter'] == 1 and state['all_iter'] == 1:
                t = min(1., 1. / flat_grad.abs().sum()) * lr
            elif state['n_iter'] == 1 and state['all_iter'] > 1:
                t = min(restart_lr, 1. / flat_grad.abs().sum()) * lr
                # t = restart_lr * lr
            else:
                t = lr
            

            if  (state['n_iter'] >= history_size-1  and restart==True) or ys_restart:
                state['n_iter'] = 0         
            
        
            

            # directional derivative
            gtd = flat_grad.dot(d)  
            # directional derivative is below tolerance
            # if tolerance_change_gtd:
            
            # if flat_grad.dot(d) > -tolerance_change: # TODO
            #     print('gtd < -tolerance')
            #     break
                    
          
            # self.debug_time[9] = self.debug_time[9] + (time.time()-t9)
            


            # optional line search: user function
            ls_func_evals = 0
            # t13 = time.time()
            if line_search_fn is not None:
                # perform line search, using user function
                if line_search_fn != "strong_wolfe":
                    raise RuntimeError("only 'strong_wolfe' is supported")
                else:
                    x_init = self._clone_param()

                    def obj_func(x, t, d):
                        
                        return self._directional_evaluate(closure, x, t, d)
                    
                    loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                        obj_func, x_init, t, d, loss, flat_grad, gtd)
                self._add_grad(t, d)
                l_t = t
                opt_cond = flat_grad.abs().max() <= tolerance_grad
            else:
                # no line search, simply move with fixed-step
                self._add_grad(t, d)
                l_t = t
                if n_iter != max_iter:
                    # re-evaluate function only if not in last iteration
                    # the reason we do this: in a stochastic setting,
                    # no use to re-evaluate that function here
                    with torch.enable_grad():
                        loss = float(closure())
                    flat_grad = self._gather_flat_grad()
                    opt_cond = flat_grad.abs().max() <= tolerance_grad
                    ls_func_evals = 1
            # self.debug_time[13] = self.debug_time[13] + (time.time()-t13)

            # update func eval
            current_evals += ls_func_evals
            state['func_evals'] += ls_func_evals
            # print(f'{state['all_iter']}:{len(u_vectors)/3}')


            ############################################################
            # check conditions
            ############################################################
            
            # if show is not None:
            #     all_iter = state['all_iter']
            #     if all_iter % show == 0:
            #         print(f'iter{all_iter}: {loss}')
            # if stop_loss is not None:
            #     if loss < stop_loss:
            #         break

            if n_iter == max_iter:
                break

            if current_evals >= max_eval:
                break

            # optimal condition
            if opt_cond:
                break

            # lack of progress
            # if d.mul(t).abs().max() <= tolerance_change:
            #     break

            # if abs(loss - prev_loss) < tolerance_change:
            #     break

        state['d'] = d
        state['t'] = t
        state['old_dirs'] = old_dirs
        state['old_stps'] = old_stps
        state['ro'] = ro
        state['H_diag'] = H_diag
        state['prev_flat_grad'] = prev_flat_grad
        state['prev_loss'] = prev_loss
        state['betas'] = betas
        state['u_vectors'] = u_vectors
        state['v_vectors'] = v_vectors
        state['gbetas'] = gbetas
        state['gu_vectors'] = gu_vectors
        state['gv_vectors'] = gv_vectors
        state['gama'] = gama
        state['gama_list'] = gama_list
      
      
        # tott = time.time() - tt1
        if return_time :
            return orig_loss, self.debug_time
        
        return orig_loss, opt_cond



       
# if state['n_iter'] > 1 and torch.abs(torch.max(flat_grad-d)) < tolerance_change:
            #     print('max(flat_grad-d) < tolerance_change', torch.max(flat_grad-d))
            #     stop_condition = 1
            #     break