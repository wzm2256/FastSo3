# FastSo3: An efficient implement of escn

FastSo3 provides a fast implement of the equivariant layers introduced in the paper [escn](https://arxiv.org/abs/2302.03655). These layers properly handle 3D rigid transformations (rotation and translation), so they are useful for modelling 3D data such as point clouds or molecules. FastSo3 is ``torch.compile`` compatible.


### How is FastSo3 different from the [official implement](https://github.com/FAIR-Chem/fairchem/tree/main/src/fairchem/core/models/escn)? 

FastSo3 is faster, and it can be even faster when combined with ``torch.compile``. Computation time (us): 

[b, c, l] |   escn   |  so2_efficient  |  so2_efficient_compile
--- | --- | --- | ---
1000 128 1 |  1545.0  |       776.4     |           246.8 
1000 128 2 |  2143.3  |       724.7     |           430.5 
1000 128 3 |  1848.9  |       724.0     |           664.3 
1000 128 4 |  2324.5  |       962.0     |           996.3 
1000 128 5 |  2838.1  |      1480.7     |          1534.6 
1000 128 6 |  3142.7  |      2314.4     |          2033.4

* The code is run on one NVidia A100 GPU. `escn`: the official escn implement. `so2_efficient`: the FastSo3 layer `SO2_conv_e`. `so2_efficient_compile`: the compiled `SO2_conv_e` layer. `b`: number of edges. 'c': number of channels. 'l': number of degrees. Hidden channel is set to `c`. For escn all possible `m`s are used. 

#### Why is FastSo3 faster?
FastSo3 pads features of different degrees to the same size, so that they can be processed efficiently. Specifically,
1. There is no for-loops in FastSo3, so the overhead is avoided.
2. There is no dynamic indexing, which means that ``torch.compile`` can be used to acclerate the modules. 


### Usage
Features of channels `C` and maximum degree `L`  are represented by tensors of shape `(B, C, L, 2L+1)`. For all degree `l` smaller than `L`, corresponding features are zero padded to length `2L+1`. For example, Let L=3.
A degree-1 feature f (length=3) will be padded as [0, 0, f, 0, 0] (length=7);
A degree-2 feature g (length=5) will be padded as [0, g, 0] (length=7).


```
import FastSo3.so3 as so3
import FastSo3.so2 as so2
import torch


c_in = 4 # input channel
c_out = 2 # output channel
c_hidden = 2 # hidden channel
L = 2 # degree
c_L0_in =2 # additional channel of degree 0 input 
c_L0_out =4 # additional channel for degree 0 output
b = 100 # number of edges

so2_conv = SO2_conv(c_in, c_out, L, c_L0_in=c_L0_in, c_L0_out=c_L0_out)

# prepare input
x_ = torch.randn((b, c_in, L, L * 2 + 1))
M = so2.get_mask(L)
x = x_* M

# extra degree 0 input
x_L0 = torch.randn((b, c_L0_in))

# weight
w = torch.randn((b, c_in, L, L * 2, c_out, L))
w_L0 = torch.randn((b, c_L0_in + c_in * L, c_L0_out + c_out * L))

# edge vector
edges_vec = torch.randn((b, 3))

# a forward pass
message, message_0 = so2.get_message(x, x_L0, w, w_L0, edges_vec, so2_conv, L)
```

### More details

1. FastSo3 provides two layers``SO2_conv_e`` and ``SO2_conv``:
    1. ``SO2_conv_e`` is the efficient version which is equivalent to the official implement. It maps the feature to lower dimensional space to achieve better efficiency.
    2. ``SO2_conv`` is the linear map between the input and ouput. It is conceptually simpler, but it can be expensive for large graphs, high degrees or large channels. 
* Note these two layers use weights of different shapes. See the source code and the script `check_equi.py` for details and more examples.

2. The equivariance of the layer can be checked by running the `check_equi.py` script:
```
>>> python check_equi.py

# output
# The equivariance error: 5.020955995860277e-06
# The invariance error: 3.5846128412231337e-06
```

### LICENSE
FastSo3 is available under a [MIT license](LICENSE).