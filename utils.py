import os
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Needed for generating classification, regression and clustering datasets
import sklearn.datasets as dt

from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification, make_blobs
from sklearn.model_selection import train_test_split
from collections import OrderedDict

from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from scipy.signal import lfilter
from scipy.interpolate import make_interp_spline
from glob import glob
from natsort import natsorted
from scipy import interpolate
# from mlxtend.data import iris_data
# from mlxtend.preprocessing import standardize
# from mlxtend.feature_extraction import RBFKernelPCA as KPCA


def make_random_quad():
    n_prob = 20
    quad_size = 2
    quad_samples = 2

    w = torch.zeros(n_prob, quad_samples, quad_size)
    y = torch.zeros(n_prob, quad_samples)
    torch.manual_seed(27)

    for i in range(n_prob):
        w[i,:,:] = torch.randn(quad_samples, quad_size)
        y[i,:] = torch.randn(quad_samples)

    # --------------------------------------- enable to save
    name = "w" + str(n_prob) + "_" + str(quad_size) + "d_" + str(quad_samples) + "s.pt"
    torch.save(w, name)
    name = "y" + str(n_prob) + "_" + str(quad_size) + "d_" + str(quad_samples) + "s.pt"
    torch.save(y, name)


def make_cn_quad_rand():
    n_prob = 10
    quad_size = 100
    quad_samples = 100
    condition_cumber = 625

    w = torch.zeros(n_prob, quad_samples, quad_size)
    y = torch.zeros(n_prob, quad_samples)
    torch.manual_seed(27)
    
    i = 0
    while i < n_prob:
        w1 = torch.randn(quad_samples, quad_size)
        cn = np.linalg.cond(w1)
        print(f'{cn}, {i}')
        if condition_cumber-0.5 < cn and cn < condition_cumber+0.5:
            w[i,:,:] = w1
            y[i,:] = torch.randn(quad_samples)
            i = i + 1
    # --------------------------------------- enable to save
    name = "w" + str(n_prob) + "_" + str(quad_size) + "d_" + str(quad_samples) + "s_"+ str(condition_cumber) +"cn.pt"
    torch.save(w, name)
    name = "y" + str(n_prob) + "_" + str(quad_size) + "d_" + str(quad_samples) + "s_"+ str(condition_cumber) +"cn.pt"
    torch.save(y, name)


def make_cn_quad():
    n_prob = 3
    quad_size = 5000
    quad_samples = quad_size
    condition_cumber = 10000

    w = torch.zeros(n_prob, quad_samples, quad_size)
    y = torch.zeros(n_prob, quad_samples)
    torch.manual_seed(27)
    

    for i in range(n_prob):
        cond_P = condition_cumber    # Condition number
        n = quad_size
        log_cond_P = np.log(cond_P)
        exp_vec = np.arange(-log_cond_P/4., log_cond_P * (n)/(4 * (n - 1)), log_cond_P/(2.*(n-1)))
        s = np.exp(exp_vec)
        S = np.diag(s)
        U, _ = np.linalg.qr((np.random.rand(n, n) - 5.) * 200)
        V, _ = np.linalg.qr((np.random.rand(n, n) - 5.) * 200)
        P = U.dot(S).dot(V.T)
        P = P.dot(P.T)
        Ptensor = torch.tensor(P)
        w[i,:,:] = Ptensor
        y[i,:] = torch.randn(quad_samples)
        print(i)

    # --------------------------------------- enable to save
    name = "w" + str(n_prob) + "_" + str(quad_size) + "d_" + str(quad_samples) + "s_"+ str(condition_cumber) +"cn.pt"
    torch.save(w, name)
    name = "y" + str(n_prob) + "_" + str(quad_size) + "d_" + str(quad_samples) + "s_"+ str(condition_cumber) +"cn.pt"
    torch.save(y, name)
    

def make_random_quad_rank1():
    n_prob = 100
    quad_size = 2
    quad_samples = 2

    w = torch.zeros(n_prob, quad_samples, quad_size)
    y = torch.zeros(n_prob, quad_samples)
    torch.manual_seed(27)

    for i in range(n_prob):

        s1 = torch.randn(1, quad_size)
        s2 = s1 * (i+1)
        w[i,0,:] = s1
        w[i,1,:] = s2
        y[i,:] = torch.randn(quad_samples)

    # --------------------------------------- enable to save
    name = "w" + str(n_prob) + "_" + str(quad_size) + "d_" + str(quad_samples) + "s_r1.pt"
    torch.save(w, name)
    name = "y" + str(n_prob) + "_" + str(quad_size) + "d_" + str(quad_samples) + "s_r1.pt"
    torch.save(y, name)

    
if __name__ == "__main__":
    # make_random_quad()
    make_cn_quad()
    # make_cn_quad_rand()
    # make_random_quad_rank1()