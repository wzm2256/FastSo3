import pdb

import FastSo3.so3 as so3
import FastSo3.so2 as so2
import torch
from scipy.spatial.transform import Rotation

c_in = 4
c_out = 2
c_hidden = 2
L = 2
c_L0_in =2
c_L0_out =4
b = 100

# preparing input
x_ = torch.randn((b, c_in, L, L * 2 + 1))
M = so2.get_mask(L)
x = x_* M
x_L0 = torch.randn((b, c_L0_in))

# defining edge vector
edges_vec = torch.randn((b, 3))

#### comment out to test SO2_conv
w = torch.randn((b, c_in, L, L * 2, c_out, L))
w_L0 = torch.randn((b, c_L0_in + c_in * L, c_L0_out + c_out * L))
so2_conv = so2.SO2_conv(c_in, c_out, L, c_L0_in=c_L0_in, c_L0_out=c_L0_out)
#### comment out to test SO2_conv_e
# w = torch.randn((b, c_hidden, 2 * L))
# w_L0 = torch.randn((b, c_hidden))
# so2_conv = so2.SO2_conv_e(c_in, c_out, c_hidden, L, c_L0_in=c_L0_in, c_L0_out=c_L0_out)
####

# a forward pass on the original graph
message, message_0 = so2.get_message(x, x_L0, w, w_L0, edges_vec, so2_conv, L)

# get a random rotation
R = torch.tensor(Rotation.random(num=1).as_matrix()).type(torch.float32)
R_w = so3.RotationToWignerDMatrix(R, end_lmax=L)

# rotate the whole graph: edges, node features
x_R = so3.rotate(R_w, x)
edges_vec_R = (R @ edges_vec.unsqueeze(-1)).squeeze(-1)

# a forwad pass on the rotated graph
message_R, message_0R = so2.get_message(x_R, x_L0, w, w_L0, edges_vec_R, so2_conv, L)


print(f'The equivariance error: {(so3.rotate(R_w, message) - message_R).norm()}')
print(f'The invariance error: {(message_0 - message_0R).norm()}')














# pdb.set_trace()
# def get_message(x, x_L0, w, w_L0, edges_vec, conv):
#     rotation = so3.init_edge_rot_mat(edges_vec)
#     W = so3.RotationToWignerDMatrix(rotation, L)
#     x_W = so3.rotate(W, x)
#     message_so2, message_L0 = conv(x_W, x_L0, w, w_L0)
#
#     message = so3.rotate(W.mT, message_so2)
#     return  message, message_so2, message_L0, W
# message, message_so2, message_0, D = get_message(x, x_L0, w, w_L0, edges_vec, so2_conv)

# R_y = D_R @ R_w @ D.mT

# so3.rotate(R_y_inv, x_W) - x_WR
# print((so3.rotate(R_y.mT, message_so2R) - message_so2).norm())
