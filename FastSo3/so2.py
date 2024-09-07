import pdb

import torch
import torch.nn as nn
import einops
import math
from . import so3

def get_mask(L: int, device='cpu') -> torch.Tensor:
    '''
    Generate mask for data.
    For example, when L=2, mask is
    [[0, 1, 1, 1, 0]
     [1, 1, 1, 1, 1]]

    Args:
        L: The highest degree
        device: device

    Returns:
        Mask: 1, 1, L, 2L+1

    '''
    Mask_1 = torch.tril(torch.ones((L, L), dtype=torch.int, device=device))
    Mask_2 = torch.flip(Mask_1, [1])
    Mask_3 = torch.ones((L, 1), dtype=torch.int, device=device)
    Mask = torch.cat([Mask_2, Mask_3, Mask_1], 1) # L, 2L+1
    return Mask.unsqueeze(0).unsqueeze(0) # 1, 1, L, 2L+1


class SO2_conv_e(torch.nn.Module):
    '''
    Computing SO3 equivariant message in the reference frame where the edge is rotated to y axis.
    The efficient version.
    Features are first compressed and then decompressed.
    Weights are multiplied to low dim features.
    '''

    def __init__(self, c_in: int, c_out: int, c_hidden:int, L: int, c_L0_in=0, c_L0_out=0):
        '''

        Args:
            c_in: input channel for L>0 degree features
            c_out: ouput channel for L>0 degree features
            L: maximum feature degree
            c_L0_in: input channel for 0-degree features
            c_L0_out: output channel for 0-degree features
            c_hidden: c_in * L elements will be compressed to c_hidden elements
        '''
        super().__init__()
        assert c_in >= 0, 'c_in must be >= 0'
        assert c_out >= 0, 'c_out must be >= 0'
        assert L >= 0, 'L must be >= 0'
        assert c_L0_in >= 0, 'c_L0_in must be >= 0'
        assert c_L0_out >= 0, 'c_L0_out must be >= 0'

        self.register_buffer('Mask_out', get_mask(L))
        self.L = L
        self.c_in = c_in
        self.c_out = c_out
        self.c_L0_in = c_L0_in
        self.c_L0_out = c_L0_out
        self.c_hidden = c_hidden


        self.en0 = nn.Parameter(torch.empty((c_in * L + c_L0_in, c_hidden)))
        self.de0 = nn.Parameter(torch.empty((c_hidden, c_out * L + c_L0_out)))

        self.en1 = nn.Parameter(torch.empty((c_in, L, c_hidden, L)))
        self.de1 = nn.Parameter(torch.empty((c_hidden, c_out, L, L)))
        self.en2 = nn.Parameter(torch.empty((c_in, L, c_hidden, L)))
        self.de2 = nn.Parameter(torch.empty((c_hidden, c_out, L, L)))

        torch.nn.init.kaiming_uniform_(self.en0, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.de0, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.en1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.de1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.en2, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.de2, a=math.sqrt(5))


    def forward(self, x: torch.Tensor, x_L0: torch.Tensor, w: torch.Tensor, w_L0: torch.Tensor) -> \
            (torch.Tensor, torch.Tensor):
        '''
        x and x_L0 are the equivariant and invariant features of the source nodes.
        w and w_L0 are the learnable weights for x and x_L0.
        The output y and y_L0 are the equivariant and invariant messages.

        x:    B, c_in, L, 2L+1
        x_L0: B, c_L0_in
        w:    B, c_hidden, 2 * L
        w_L0: B, c_hidden

        output:
        y:    B, c_out, L, 2L+1
        y_L0: B, c_L0_out
        '''

        # check shape
        assert x.shape[1:] == (self.c_in, self.L, self.L * 2 + 1), 'Inconsistent x shape'
        assert x_L0.shape[1] == self.c_L0_in, 'Inconsistent x_L0 shape'
        assert w.shape[1:] == (self.c_hidden, self.L * 2), 'Inconsistent w shape'
        assert w_L0.shape[1:] == (self.c_hidden, ), 'Inconsistent w_0 shape'
        assert x.shape[0] == x_L0.shape[0] == w.shape[0] == w_L0.shape[0], 'Inconsistent edge number'

        x_minus, x_m0, x_plus = torch.split(x, [self.L, 1, self.L], dim=-1) # x_minus and x_plus B, c_in, L, L x_m0:B, c_in, L
        w_plus = torch.cat([torch.flip(w[:, :, self.L:], [2]), w[:, :, self.L:]], -1)
        w_minus = torch.cat([w[:, :, :self.L], torch.flip(w[:, :, :self.L], [2])], -1)

        ### compute m=0
        all_x_m0 = torch.cat([x_L0, torch.flatten(x_m0, start_dim=1)], 1) # B, c_in * L + c_L0_in
        en_x0 = einops.einsum(self.en0, all_x_m0, 'c h, b c -> b h')
        new_x0 = en_x0 * w_L0
        de_x0 = einops.einsum(self.de0, new_x0, 'h c, b h -> b c')
        y_m0 = de_x0[:, self.c_L0_out:].view((-1, self.c_out, self.L, 1))
        y_L0 = de_x0[:, :self.c_L0_out]

        ### compute m> 0
        x_L = torch.cat([x_minus, x_plus], -1)
        en_weight1 = torch.cat([torch.flip(self.en1, [3]), self.en1], -1)
        de_weight1 = torch.cat([torch.flip(self.de1, [3]), self.de1], -1)
        en_weight2 = torch.cat([torch.flip(self.en2, [3]), self.en2], -1)
        de_weight2 = torch.cat([torch.flip(self.de2, [3]), self.de2], -1)

        en_x1 = einops.einsum(en_weight1, x_L, 'c l h m, b c l m -> b h m')
        new_x_plus = en_x1 * w_plus
        de_x_plus = einops.einsum(de_weight1, new_x_plus, 'h c l m, b h m -> b c l m')

        en_x2 = einops.einsum(en_weight2, x_L, 'c l h m, b c l m -> b h m')
        new_x_minus = en_x2 * w_minus
        de_x_minus = einops.einsum(de_weight2, new_x_minus, 'h c l m, b h m -> b c l m')

        y_plus = de_x_plus[:, :, :, self.L:] - torch.flip(de_x_minus[:, :, :, :self.L], [3])
        y_minus = de_x_plus[:, :, :, :self.L] +  torch.flip(de_x_minus[:, :, :, self.L:], [3])

        y = torch.cat([y_minus, y_m0, y_plus], axis=-1) * self.Mask_out
        return y, y_L0


class SO2_conv(torch.nn.Module):
    '''
    Computing SO3 equivariant message in the reference frame where the edge is rotated to y axis.
    '''
    def __init__(self, c_in: int, c_out: int, L: int, c_L0_in=0, c_L0_out=0):
        '''

        Args:
            c_in: input channel for L>0 degree features
            c_out: ouput channel for L>0 degree features
            L: maximum feature degree
            c_L0_in: input channel for 0-degree features
            c_L0_out: output channel for 0-degree features
        '''
        super().__init__()
        assert c_in >= 0, 'c_in must be >= 0'
        assert c_out >= 0, 'c_out must be >= 0'
        assert L >= 0, 'L must be >= 0'
        assert c_L0_in >= 0, 'c_L0_in must be >= 0'
        assert c_L0_out >= 0, 'c_L0_out must be >= 0'


        self.register_buffer('Mask_out', get_mask(L))
        self.L = L
        self.c_in  = c_in
        self.c_out = c_out
        self.c_L0_in = c_L0_in
        self.c_L0_out = c_L0_out

    def forward(self, x: torch.Tensor, x_L0: torch.Tensor, w: torch.Tensor, w_L0: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        '''
        x and x_L0 are the equivariant and invariant features of the source nodes.
        w and w_L0 are the learnable weights for x and x_L0.
        The output y and y_L0 are the equivariant and invariant messages.

        x:    B, c_in, L, 2L+1
        x_L0: B, c_L0_in
        w:    B, c_in, L, 2*L, c_out, L
        w_L0: B, c_L0_in + c_in * L, c_L0_out + c_out * L
        
        output:
        y:    B, c_out, L, 2L+1
        y_L0: B, c_L0_out
        '''

        # check shape
        assert x.shape[1:] == (self.c_in, self.L, self.L * 2 + 1), 'Inconsistent x shape'
        assert x_L0.shape[1] == self.c_L0_in, 'Inconsistent x_L0 shape'
        assert w.shape[1:] == (self.c_in, self.L, self.L * 2, self.c_out, self.L), 'Inconsistent w shape'
        assert w_L0.shape[1:] == (self.c_L0_in + self.c_in * self.L, self.c_L0_out + self.c_out * self.L), 'Inconsistent w_0 shape'
        assert x.shape[0] == x_L0.shape[0] == w.shape[0] == w_L0.shape[0], 'Inconsistent edge number'

        w_minus, w_plus = torch.split(w, self.L, dim=3) # w_minus and w_plus: c_in, L, L, c_out, L
        x_minus, x_m0, x_plus = torch.split(x, [self.L, 1, self.L], dim=-1) # x_minus and x_plus B, c_in, L, L x_m0:B, c_in, L 


        y_plus = einops.einsum(w_plus, x_plus, 'b c l m d o, b c l m -> b d o m') - \
                 torch.flip(einops.einsum(w_minus, x_minus, 'b c l m d o, b c l m -> b d o m'), [3])
        y_minus = einops.einsum(torch.flip(w_plus, [3]), x_minus, 'b c l m d o, b c l m -> b d o m') + \
                  einops.einsum(w_minus, torch.flip(x_plus, [3]), 'b c l m d o, b c l m -> b d o m')

        all_x_m0 = torch.cat([x_L0, torch.flatten(x_m0, start_dim=1)], 1) # B, c_in * L + c_L0_in
        all_y_m0 = einops.einsum(w_L0, all_x_m0, 'b c d, b c -> b d') # B, c_L0_out + c_out * L

        y_m0 = all_y_m0[:, self.c_L0_out:].view((-1, self.c_out, self.L, 1))
        y_L0 = all_y_m0[:, :self.c_L0_out]

        y = torch.cat([y_minus, y_m0, y_plus], axis=-1) * self.Mask_out
        return y, y_L0
    

def get_message(x, x_L0, w, w_L0, edges_vec, conv, L):
    rotation = so3.init_edge_rot_mat(edges_vec)
    W = so3.RotationToWignerDMatrix(rotation, L)
    x_W = so3.rotate(W, x)
    message_so2, message_L0 = conv(x_W, x_L0, w, w_L0)

    message = so3.rotate(W.mT, message_so2)
    return  message, message_L0


if __name__ == '__main__':
    import so3

    c_in = 4
    c_out = 2
    L = 2
    c_L0_in=4
    c_L0_out=8
    b = 10

    so2_conv = SO2_conv(c_in, c_out, L, c_L0_in=c_L0_in, c_L0_out=c_L0_out)

    x_ = torch.randn((b, c_in, L, L * 2 + 1))
    M = get_mask(L)
    x = x_* M

    # pdb.set_trace()
    x_L0 = torch.randn((b, c_L0_in))

    w = torch.randn((b, c_in, L, L * 2, c_out, L))
    w_L0 = torch.randn((b, c_L0_in + c_in * L, c_L0_out + c_out * L))

    edges_vec = torch.randn((b, 3))


    #####
    # rotation_e = so3.init_edge_rot_mat(edges_vec)
    # W = so3.RotationToWignerDMatrix(rotation_e, L)
    # x_W = so3.rotate(W, x)
    # message_so2, message_L0 = so2_conv(x_W, x_L0, w, w_L0)
    # message = so3.rotate(W.mT, message_so2)
    # #####
    # # message = get_message(x, x_L0, w, w_L0, edges_vec)
    # from scipy.spatial.transform import Rotation
    # R = torch.tensor(Rotation.random(num=1).as_matrix()).type(torch.float32)
    # R_w = so3.RotationToWignerDMatrix(R, end_lmax=L)
    #
    # x_R = so3.rotate(R_w, x)
    # edges_vec_R = (R @ edges_vec.unsqueeze(-1)).squeeze(-1)
    #
    # rotation_eR = so3.init_edge_rot_mat(edges_vec_R)
    # W_eR = so3.RotationToWignerDMatrix(rotation_eR, L)
    # x_WR = so3.rotate(W_eR, x_R)
    # message_so2R, message_L0R = so2_conv(x_WR, x_L0, w, w_L0)
    #
    # message_R = so3.rotate(W_eR.mT, message_so2R)
    # #####
    # R_y = rotation_eR @ R @ rotation_e.mT
    # WR_y = W_eR @ R_w @ W.mT
    # so3.rotate(WR_y, x_W) - x_WR
    #
    # so3.rotate(WR_y.mT, message_so2R) - message_so2
    #
    # pdb.set_trace()
    # w_minus, w_plus = torch.split(w, L, dim=2)  # w_minus and w_plus: c_in, L, L, c_out, L
    # x_minus, x_m0, x_plus = torch.split(x_W, [L, 1, L], dim=-1)  # x_minus and x_plus B, c_in, L, L x_m0:B, c_in, L
    # y_plus = einops.einsum(w_plus, x_plus, 'c l m d o, b c l m -> b d o m') - torch.flip(einops.einsum(w_minus, x_minus, 'c l m d o, b c l m -> b d o m'), [3])
    #
    #
    # x_W[0, 0, 1, 4] * w[0, 1, 3, 0, 1] - x_W[0, 0, 1, 0] * w[0, 1, 0, 0, 1]
    #
    # x_W[0, 0, 1, 4] - x_plus[0, 0, 1, 1]
    # x_W[0, 0, 1, 0] - x_minus[0, 0, 1, 0]
    # w[0, 1, 3, 0, 1] - w_plus[0, 1, 1, 0, 1]
    # w[0, 1, 0, 0, 1] - w_minus[0, 1, 0, 0, 1]
    #
    # pdb.set_trace()
    #

    # pdb.set_trace()
    ######
    message, message_0 = get_message(x, x_L0, w, w_L0, edges_vec, so2_conv)

    from scipy.spatial.transform import Rotation
    R = torch.tensor(Rotation.random(num=1).as_matrix()).type(torch.float32)
    R_w = so3.RotationToWignerDMatrix(R, end_lmax=L)

    x_R = so3.rotate(R_w, x)
    edges_vec_R = (R @ edges_vec.unsqueeze(-1)).squeeze(-1)
    message_R, message_0R = get_message(x_R, x_L0, w, w_L0, edges_vec_R, so2_conv)


    message_timesR = so3.rotate(R_w, message)
    print((message_timesR - message_R).norm())
    print((message_0 - message_0R).norm())
    # pdb.set_trace()

    # message_1, message_L0 = so2_conv(x, x_L0, w, w_L0)
    #
    # R = torch.tensor(Rotation.from_rotvec(np.array([[0, 0.3, 0]])).as_matrix()).type(torch.float32)
    # R_w = so3.RotationToWignerDMatrix(R, end_lmax=L)
    # pdb.set_trace()
    # message_1, message_L0 = so2_conv(x, x_L0, w, w_L0)

