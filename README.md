# DirL-BFGS
## How to use
Our proposed method "DirL-BFGS" is in "opzer" directory: 
- dirl_bfgs.py
- dirl_bfgs_nn.py
  
In "dirl_bfgs_nn.py" we optimize implementation for neural networks.
Our codes are based on PyTorch library. So at fisrt you shoule install PyTorch with pip or Conda command. Then you should add optim directory of PyTorch library to "dirl_bfgs.py" like this:

``` sys.path.append("/home/user/pyenvs/env1/lib/python3.12/site-packages/torch/optim/")  ```

The same approach can be applied to the other optimizers located in the folder.

Additional examples can be found in the root directory. The file quad_torch.py provides a straightforward demonstration of how to utilize the optimizers.

<!-- ![_avg_time_plot - Copy](https://github.com/user-attachments/assets/930c2a98-c50c-44a5-b3a0-2ef571977624) -->
