import sys
import time


sys.path.append('path/to/your/torch/Lib/site-packages/torch/optim')

import torch
from functools import reduce
from optimizer import Optimizer


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


class NewLBFGS(Optimizer):
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
                 tolerance_change=1e-9,
                 history_size=100,
                 line_search_fn=None,
                 fast_version=True,
                 update_gamma=True,
                 restart=False,
                 return_time=False):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(
            lr=lr,
            restart_lr=restart_lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size+1,
            line_search_fn=line_search_fn,
            fast_version=fast_version,
            update_gamma=update_gamma,
            restart=restart,
            return_time=return_time)
        super(NewLBFGS, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        
        # global USE_CUDA
        # USE_CUDA = self._params[0].is_cuda
        global DEVICE
        if self._params[0].is_cuda:
            DEVICE = 'cuda'
        else:
            DEVICE = 'cpu'
        
        self._numel_cache = None
        self.total_time = 0
        self.debug_time1 = 0
        self.debug_time2 = 0

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

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

       
    def _compute_buv2(self, d, push_iter, gamma_push_iter, old_stps, old_dirs, ro, betas, u_vectors, v_vectors, gb, gu, gv):

        nogamma_pi = push_iter - gamma_push_iter

        newbeta = torch.zeros((nogamma_pi, 1), device=DEVICE)
        newv = torch.zeros((nogamma_pi, d.shape[0]), device=DEVICE)
        newu = torch.zeros((nogamma_pi, d.shape[0]), device=DEVICE)
        gnewbeta = torch.zeros((gamma_push_iter, 1), device=DEVICE)
        gnewv = torch.zeros((gamma_push_iter, d.shape[0]), device=DEVICE)
        gnewu = torch.zeros((gamma_push_iter, d.shape[0]), device=DEVICE)

        k_indx = -1
        tempb4 = 0
        tempb7 = 0
        tempu5 = torch.zeros((d.shape[0], 1), device=DEVICE)
        tempu8 = torch.zeros((d.shape[0], 1), device=DEVICE)
        tempv3 = torch.zeros((d.shape[0], 1), device=DEVICE)
        tempv6 = torch.zeros((d.shape[0], 1), device=DEVICE)
        # time1 = time.time()
        for k in range(len(u_vectors)):
            for i in range(gamma_push_iter):
                tempb4 = tempb4 + gb[k][i] * (old_dirs[k_indx] @ gu[k][i]) * (gv[k][i] @ old_dirs[k_indx])
                t1 = gb[k][i] * (gv[k][i] @ old_dirs[k_indx]) * gu[k][i]
                tempu5 = tempu5 + t1.reshape((d.shape[0], 1))
                t2 = gb[k][i] * (old_dirs[k_indx] @ gu[k][i]) * gv[k][i]
                tempv3 = tempv3 + t2.reshape((d.shape[0], 1))
            for i in range(nogamma_pi):
                tempb7 = tempb7 + betas[k][i] * (old_dirs[k_indx] @ u_vectors[k][i]) * (v_vectors[k][i] @ old_dirs[k_indx])
                t1 = betas[k][i] * (v_vectors[k][i] @ old_dirs[k_indx]) * u_vectors[k][i]
                tempu8 = tempu8 + t1.reshape((d.shape[0], 1))
                t2 = betas[k][i] * (old_dirs[k_indx] @ u_vectors[k][i]) * v_vectors[k][i]
                tempv6 = tempv6 + t2.reshape((d.shape[0], 1))

        gnewbeta[0] = -ro[k_indx]
        gnewu[0] = old_stps[k_indx]
        gnewv[0] = old_dirs[k_indx]

        gnewbeta[1] = -ro[k_indx]
        gnewu[1] = old_dirs[k_indx]
        gnewv[1] = old_stps[k_indx]

        gnewbeta[2] = ro[k_indx] * ro[k_indx] * (old_dirs[k_indx] @ old_dirs[k_indx])
        gnewu[2] = old_stps[k_indx]
        gnewv[2] = old_stps[k_indx]

        gnewbeta[3] = -ro[k_indx]
        gnewu[3] = old_stps[k_indx]
        gnewv[3] = tempv3[:, 0]

        gnewbeta[4] = ro[k_indx] * ro[k_indx] * tempb4
        gnewu[4] = old_stps[k_indx]
        gnewv[4] = old_stps[k_indx]

        gnewbeta[5] = -ro[k_indx]
        gnewu[5] = tempu5[:, 0]
        gnewv[5] = old_stps[k_indx]

        newbeta[0] = -ro[k_indx]
        newu[0] = old_stps[k_indx]
        newv[0] = tempv6[:, 0]

        newbeta[1] =ro[k_indx] * ro[k_indx] * tempb7
        newu[1] = old_stps[k_indx]
        newv[1] = old_stps[k_indx]

        newbeta[2] = -ro[k_indx]
        newu[2] = tempu8[:, 0]
        newv[2] = old_stps[k_indx]

        newbeta[3] = ro[k_indx]
        newu[3] = old_stps[k_indx]
        newv[3] = old_stps[k_indx]

        return newbeta, newu, newv, gnewbeta, gnewu, gnewv

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

        group = self.param_groups[0]
        lr = group['lr']
        restart_lr = group['restart_lr']
        max_iter = group['max_iter']
        max_eval = group['max_eval']
        tolerance_grad = group['tolerance_grad']
        tolerance_change = group['tolerance_change']
        line_search_fn = group['line_search_fn']
        history_size = group['history_size']
        fast_version = group['fast_version']
        update_gamma = group['update_gamma']
        restart = group['restart']
        return_time = group['return_time']

        # NOTE: LBFGS has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[self._params[0]]
        state.setdefault('func_evals', 0)
        state.setdefault('n_iter', 0)
        state.setdefault('all_iter', 0)

        # evaluate initial f(x) and df/dx
        orig_loss = closure()
        tt1 = time.time()
        
        loss = float(orig_loss)
        if loss > 10000:
            dddd = 0
        current_evals = 1
        state['func_evals'] += 1

        flat_grad = self._gather_flat_grad()
        opt_cond = flat_grad.abs().max() <= tolerance_grad
       
        # optimal condition
        if opt_cond:
            # print('opt cond new!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            return orig_loss, 0

        l_d = state.get('l_d')
        l_t = state.get('l_t')
        l_old_dirs = state.get('l_old_dirs')
        l_old_stps = state.get('l_old_stps')
        l_ro = state.get('l_ro')
        l_H_diag = state.get('l_H_diag')
        l_prev_flat_grad = state.get('l_prev_flat_grad')
        l_prev_loss = state.get('l_prev_loss')

        # tensors cached in state (for tracing)
        d = state.get('d')
        t = state.get('t')
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
        h_main = state.get('h_main')

        params_dict = 0
        ys = 0
        d_new = 0
        n_iter = 0
        tempH_G = 0
        push_iter = 10
        gamma_push_iter = 6
        nogamma_push_iter = 4

        # optimize for a max of max_iter iterations
        sniter = state['n_iter']
        ys = 100
        ys_threshold = 1e-12
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
                betas = [torch.zeros((nogamma_push_iter, 1), device=DEVICE)]
                gbetas = [torch.zeros((gamma_push_iter, 1), device=DEVICE)]
                v_vectors = [torch.zeros((nogamma_push_iter, d.shape[0]), device=DEVICE)]
                gv_vectors = [torch.zeros((gamma_push_iter, d.shape[0]), device=DEVICE)]
                u_vectors = [torch.zeros((nogamma_push_iter, d.shape[0]), device=DEVICE)]
                gu_vectors = [torch.zeros((gamma_push_iter, d.shape[0]), device=DEVICE)]
                # h_main = torch.eye(d.shape[0], d.shape[0], device=DEVICE)

            
            else:
                t0 = time.time()
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
                    # print(f'{sniter} new {y} * {s} = {ys} (ys > 1e-10) ')
                    # updating memory
                    if len(old_dirs) >= history_size:
                        # shift history by one (limited-memory)
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)


                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1. / ys)


                    if len(betas) >= history_size+1:  # TODO history_size+1
                        if len(betas) > 1:
                            betas.pop(0)
                            gbetas.pop(0)
                            v_vectors.pop(0)
                            gv_vectors.pop(0)
                            u_vectors.pop(0)
                            gu_vectors.pop(0)
                            gama_list.pop(0)


                    if state['n_iter'] == 2 or update_gamma:
                        H_diag = ys / y.dot(y)  # (y*y) TODO H_diag in if or before if
                        gama = H_diag

                    gama_list.append(gama)

                    
                    if fast_version:

                        newbeta = torch.zeros((push_iter, 1), device=DEVICE)
                        newv = torch.zeros((push_iter, d.shape[0]), device=DEVICE)
                        newu = torch.zeros((push_iter, d.shape[0]), device=DEVICE)

                        u_tensor = torch.cat(u_vectors)
                        gu_tensor = torch.cat(gu_vectors)
                        v_tensor = torch.cat(v_vectors)
                        gv_tensor = torch.cat(gv_vectors)
                        betas_tensor = torch.cat(betas)
                        gbetas_tensor = torch.cat(gbetas)

                        yk = old_dirs[-1]
                        k_indx = -1

                        tempv = torch.sum(betas_tensor.T * ((yk @ u_tensor.T) * v_tensor.T), 1).view(-1, 1)
                        gtempv = torch.sum(gbetas_tensor.T * ((yk @ gu_tensor.T) * gv_tensor.T), 1).view(-1, 1)
                        tempu = torch.sum(betas_tensor.T * ((v_tensor @ yk.T) * u_tensor.T), 1).view(-1, 1)
                        gtempu = torch.sum(gbetas_tensor.T * ((gv_tensor @ yk.T) * gu_tensor.T), 1).view(-1, 1)
                        tempb = torch.sum(betas_tensor.T * ((yk @ u_tensor.T) * (v_tensor @ yk.T)), 1)
                        gtempb = torch.sum(gbetas_tensor.T * ((yk @ gu_tensor.T) * (gv_tensor @ yk.T)), 1)

                        newbeta = torch.zeros((nogamma_push_iter, 1), device=DEVICE)
                        newv = torch.zeros((nogamma_push_iter, d.shape[0]), device=DEVICE)
                        newu = torch.zeros((nogamma_push_iter, d.shape[0]), device=DEVICE)
                        gnewbeta = torch.zeros((gamma_push_iter, 1), device=DEVICE)
                        gnewv = torch.zeros((gamma_push_iter, d.shape[0]), device=DEVICE)
                        gnewu = torch.zeros((gamma_push_iter, d.shape[0]), device=DEVICE)

                        gnewbeta[0] = -ro[k_indx]
                        gnewu[0] = old_stps[k_indx]
                        gnewv[0] = old_dirs[k_indx]

                        gnewbeta[1] = -ro[k_indx]
                        gnewu[1] = old_dirs[k_indx]
                        gnewv[1] = old_stps[k_indx]

                        gnewbeta[2] = ro[k_indx] * ro[k_indx] * (old_dirs[k_indx] @ old_dirs[k_indx])
                        gnewu[2] = old_stps[k_indx]
                        gnewv[2] = old_stps[k_indx]

                        gnewbeta[3] = -ro[k_indx]
                        gnewu[3] = old_stps[k_indx]
                        gnewv[3] = gtempv[:, 0]  ##################

                        gnewbeta[4] = ro[k_indx] * ro[k_indx] * gtempb  ###################
                        gnewu[4] = old_stps[k_indx]
                        gnewv[4] = old_stps[k_indx]

                        gnewbeta[5] = -ro[k_indx]
                        gnewu[5] = gtempu[:, 0]  ##################
                        gnewv[5] = old_stps[k_indx]

                        newbeta[0] = -ro[k_indx]
                        newu[0] = old_stps[k_indx]
                        newv[0] = tempv[:, 0]  ##################

                        newbeta[1] = ro[k_indx] * ro[k_indx] * tempb  ################
                        newu[1] = old_stps[k_indx]
                        newv[1] = old_stps[k_indx]

                        newbeta[2] = -ro[k_indx]
                        newu[2] = tempu[:, 0]  ####################
                        newv[2] = old_stps[k_indx]

                        newbeta[3] = ro[k_indx]
                        newu[3] = old_stps[k_indx]
                        newv[3] = old_stps[k_indx]

                       
                    else:  # elif ~fast_version:
                       
                        newbeta, newu, newv, gnewbeta, gnewu, gnewv = self._compute_buv2(d, push_iter,
                                                                 gamma_push_iter,old_stps, old_dirs, ro,
                                                                 betas, u_vectors, v_vectors,
                                                                 gbetas, gu_vectors, gv_vectors)

                    # print(f'beta= {newbeta2 - newbeta}')
                    # print(f'gbeta= {gnewbeta2 - gnewbeta}')
                    # print(f'newu= {newu2 - newu}')
                    # print(f'gnewu= {gnewu2 - gnewu}')
                    # print(f'newv= {newv2 - newv}')
                    # print(f'gnewv= {gnewv2 - gnewv}')

                    betas.append(newbeta)
                    u_vectors.append(newu)
                    v_vectors.append(newv)
                    gbetas.append(gnewbeta)
                    gu_vectors.append(gnewu)
                    gv_vectors.append(gnewv)

                    if len(betas) >= history_size:  # TODO
                        betas.pop(0)
                        u_vectors.pop(0)
                        v_vectors.pop(0)
                        gbetas.pop(0)
                        gu_vectors.pop(0)
                        gv_vectors.pop(0)

               
                u_tensor = torch.cat(u_vectors)
                v_tensor = torch.cat(v_vectors)
                betas_tensor = torch.cat(betas)
                gu_tensor = torch.cat(gu_vectors)
                gv_tensor = torch.cat(gv_vectors)
                gbetas_tensor = torch.cat(gbetas)

                tempH_G1 = 0
                tempH_G2 = 0
                tempH_G11 = 0
                tempH_G22 = 0
                            
                
                if fast_version:
                    tempH_G11 = torch.sum(gbetas_tensor * gu_tensor * (gv_tensor @ flat_grad.reshape(-1, 1)), 0)
                    tempH_G22 = torch.sum(betas_tensor * u_tensor * (v_tensor @ flat_grad.reshape(-1, 1)), 0)
                    tempH_G = tempH_G22 + (tempH_G11 * gama_list[-1]) + (gama_list[-1] * flat_grad)
                    # tempH_G = tempH_G22 + (tempH_G11 + flat_grad) * gama_list[-1]

                    d = -tempH_G
                    # tempH_G = torch.eye(d.shape[0], d.shape[0], device=DEVICE)  # TODO

                else:
                    # d = (torch.sum(betas_tensor * u_tensor * (v_tensor @ flat_grad.reshape(-1, 1)), 0) + (H_diag * flat_grad)) * -1
                    for k in range(len(u_vectors)):
                        for i in range(gamma_push_iter):
                            tempH_G1 = tempH_G1 + gbetas[k][i] * gu_vectors[k][i].reshape(d.shape[0], 1) @ gv_vectors[k][i].reshape(d.shape[0],1).T
                    for k in range(len(u_vectors)):
                        for i in range(nogamma_push_iter):
                            tempH_G2 = tempH_G2 + betas[k][i] * u_vectors[k][i].reshape(d.shape[0], 1) @ v_vectors[k][i].reshape(d.shape[0], 1).T
                    I = torch.eye(d.shape[0], d.shape[0], device=DEVICE)
                    tempH_G = (tempH_G1 * gama_list[-1]) + tempH_G2 + gama_list[-1] * I
                    d = -tempH_G @ flat_grad
                
                # d_new = d
                # print(f'dddd{d - d2}') #####################

            if prev_flat_grad is None:
                prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
            else:
                prev_flat_grad.copy_(flat_grad)
            prev_loss = loss

            ############################################################
            # compute step length
            ############################################################
            # reset initial guess for step size
            if state['n_iter'] == 1 and state['all_iter'] == 1:
                t = min(1., 1. / flat_grad.abs().sum()) * lr
            elif state['n_iter'] == 1 and state['all_iter'] > 1:
                t = min(restart_lr, 1. / flat_grad.abs().sum()) * lr
                # t = restart_lr * lr
                
                # gama_list[0] = t
            else:
                t = lr

            # if ys <= ys_threshold and restart==True:
            if  len(betas) >= history_size-1  and restart==True:
                # sssssssssssssss = state['n_iter']
                # print(f'{sssssssssssssss} - {len(betas)}')
                
                # t = min(1., 1. / flat_grad.abs().sum()) * lr
                state['n_iter'] = 0
                
            # directional derivative
            gtd = flat_grad.dot(d)  # g * d

            # directional derivative is below tolerance
            if gtd > -tolerance_change:
                break

            # optional line search: user function
            ls_func_evals = 0
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

            # update func eval
            current_evals += ls_func_evals
            state['func_evals'] += ls_func_evals

            ############################################################
            # check conditions
            ############################################################
            if n_iter == max_iter:
                break

            if current_evals >= max_eval:
                break

            # optimal condition
            if opt_cond:
                break

            # lack of progress
            if d.mul(t).abs().max() <= tolerance_change:
                break

            if abs(loss - prev_loss) < tolerance_change:
                break

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
      
      
        tott = time.time() - tt1
        if return_time :
            return orig_loss, tott
        
        return orig_loss  # , d_new, tempH_G, ys



       