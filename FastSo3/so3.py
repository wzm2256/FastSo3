import pdb

import torch
import logging
import torch.nn.functional as F
import os
import einops
from . import e3nn_rotate


_Jd = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"))


def init_edge_rot_mat(edge_distance_vec):
	edge_vec_0 = edge_distance_vec
	edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0**2, dim=1))

	# Make sure the atoms are far enough apart
	# assert torch.min(edge_vec_0_distance) < 0.0001
	if torch.min(edge_vec_0_distance) < 0.0001:
		logging.error(f"Error edge_vec_0_distance: {torch.min(edge_vec_0_distance)}")

	norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))

	edge_vec_2 = torch.rand_like(edge_vec_0) - 0.5
	edge_vec_2 = edge_vec_2 / (torch.sqrt(torch.sum(edge_vec_2**2, dim=1)).view(-1, 1))
	# Create two rotated copys of the random vectors in case the random vector is aligned with norm_x
	# With two 90 degree rotated vectors, at least one should not be aligned with norm_x
	edge_vec_2b = edge_vec_2.clone()
	edge_vec_2b[:, 0] = -edge_vec_2[:, 1]
	edge_vec_2b[:, 1] = edge_vec_2[:, 0]
	edge_vec_2c = edge_vec_2.clone()
	edge_vec_2c[:, 1] = -edge_vec_2[:, 2]
	edge_vec_2c[:, 2] = edge_vec_2[:, 1]
	vec_dot_b = torch.abs(torch.sum(edge_vec_2b * norm_x, dim=1)).view(-1, 1)
	vec_dot_c = torch.abs(torch.sum(edge_vec_2c * norm_x, dim=1)).view(-1, 1)

	vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
	edge_vec_2 = torch.where(torch.gt(vec_dot, vec_dot_b), edge_vec_2b, edge_vec_2)
	vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
	edge_vec_2 = torch.where(torch.gt(vec_dot, vec_dot_c), edge_vec_2c, edge_vec_2)

	vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1))
	# Check the vectors aren't aligned
	assert torch.max(vec_dot) < 0.99

	norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
	norm_z = norm_z / (torch.sqrt(torch.sum(norm_z**2, dim=1, keepdim=True)))
	norm_z = norm_z / (torch.sqrt(torch.sum(norm_z**2, dim=1)).view(-1, 1))
	norm_y = torch.cross(norm_x, norm_z, dim=1)
	norm_y = norm_y / (torch.sqrt(torch.sum(norm_y**2, dim=1, keepdim=True)))

	# Construct the 3D rotation matrix
	norm_x = norm_x.view(-1, 3, 1)
	norm_y = -norm_y.view(-1, 3, 1)
	norm_z = norm_z.view(-1, 3, 1)

	edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
	edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)

	return edge_rot_mat.detach()


# Compute Wigner matrices from rotation matrix
def RotationToWignerDMatrix(edge_rot_mat: torch.Tensor, end_lmax: int, start_lmax: int=1) -> torch.Tensor:
	x = edge_rot_mat @ edge_rot_mat.new_tensor([0.0, 1.0, 0.0])
	alpha, beta = e3nn_rotate.xyz_to_angles(x)
	R = (e3nn_rotate.angles_to_matrix(alpha, beta, torch.zeros_like(alpha)).transpose(-1, -2) @ edge_rot_mat)
	gamma = torch.atan2(R[..., 0, 2], R[..., 0, 0])

	W = []
	for l in range(start_lmax, end_lmax + 1):
		W.append(F.pad(wigner_D(l, alpha, beta, gamma), [end_lmax - l] * 4, mode='constant', value=0))

	W_full = torch.stack(W, 1) # B, L, 2L+1, 2L+1
	return W_full

# Borrowed from e3nn @ 0.4.0:
# https://github.com/e3nn/e3nn/blob/0.4.0/e3nn/o3/_wigner.py#L37
#
# In 0.5.0, e3nn shifted to torch.matrix_exp which is significantly slower:
# https://github.com/e3nn/e3nn/blob/0.5.0/e3nn/o3/_wigner.py#L92
def wigner_D(lval, alpha, beta, gamma):
	if not lval < len(_Jd):
		raise NotImplementedError(
			f"wigner D maximum l implemented is {len(_Jd) - 1}, send us an email to ask for more"
		)

	alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
	J = _Jd[lval].to(dtype=alpha.dtype, device=alpha.device)
	Xa = _z_rot_mat(alpha, lval)
	Xb = _z_rot_mat(beta, lval)
	Xc = _z_rot_mat(gamma, lval)
	return Xa @ J @ Xb @ J @ Xc

def _z_rot_mat(angle: torch.Tensor, lv: int) -> torch.Tensor:
	shape, device, dtype = angle.shape, angle.device, angle.dtype
	M = angle.new_zeros((*shape, 2 * lv + 1, 2 * lv + 1))
	inds = torch.arange(0, 2 * lv + 1, 1, device=device)
	reversed_inds = torch.arange(2 * lv, -1, -1, device=device)
	frequencies = torch.arange(lv, -lv - 1, -1, dtype=dtype, device=device)
	M[..., inds, reversed_inds] = torch.sin(frequencies * angle[..., None])
	M[..., inds, inds] = torch.cos(frequencies * angle[..., None])
	return M



def rotate(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
	'''
	Args:
		W: B, L, 2L+1, 2L+1
		x: B, C, L, 2L+1

	Returns:
		y: B, C, L, 2L+1
	'''

	return einops.einsum(W, x, 'b l d e, b c l e -> b c l d')






if __name__ == '__main__':
	# from scipy.spatial.transform import Rotation
	# edge_rot_mat = torch.tensor(Rotation.random(num=5).as_matrix()).type(torch.float32)
	# RotationToWignerDMatrix(edge_rot_mat, end_lmax=3)

	# import so2
	edges_vec = torch.randn((100, 3))
	source_feature = torch.randn((100, 32, 4, 9))
	source_feature_m0 = torch.randn((100, 32))
	rotation = init_edge_rot_mat(edges_vec)


	# so2.so2_conv()
	rotated = rotation @ edges_vec.unsqueeze(-1)
	pdb.set_trace()


