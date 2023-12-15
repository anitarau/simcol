"""
Bimodal Camera Pose Prediction for Endoscopy.

Anita Rau, a.rau.16@ucl.ac.uk, 2023
"""

from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from inverse_warp import get_proj_idx

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



def compute_proj_idx(intrinsics, tgt_depth, ref_depths, poses, poses_inv, embedded_size=16, resize=False):

    for ref_depth, pose, pose_inv in zip(ref_depths, poses, poses_inv):

        # upsample depth
        b, _, h, w = tgt_depth.size()

        tgt_depth_scaled = tgt_depth
        ref_depth_scaled = ref_depth

        intrinsic_scaled = intrinsics

        src_pixel_coords_tgt_in_ref, valid_tgt_in_ref = get_proj_idx(tgt_depth_scaled, ref_depth_scaled, pose, intrinsic_scaled, 'zeros')

        src_pixel_coords_ref_in_tgt, valid_ref_in_tgt = get_proj_idx(ref_depth_scaled, tgt_depth_scaled, pose_inv, intrinsic_scaled, 'zeros')

        ref_in_tgt = ((src_pixel_coords_ref_in_tgt))
        tgt_in_ref = ((src_pixel_coords_tgt_in_ref))


    map_ref_in_tgt = ref_in_tgt.permute((0, 3, 1, 2)).float()
    map_tgt_in_ref = tgt_in_ref.permute((0, 3, 1, 2)).float()

    if resize:
        map_ref_in_tgt = F.interpolate(map_ref_in_tgt, size=(embedded_size, embedded_size))
        map_tgt_in_ref = F.interpolate(map_tgt_in_ref, size=(embedded_size, embedded_size))
        valid_tgt_in_ref = F.interpolate(valid_tgt_in_ref.unsqueeze(1).float(), size=(embedded_size, embedded_size))
        valid_ref_in_tgt = F.interpolate(valid_ref_in_tgt.unsqueeze(1).float(), size=(embedded_size, embedded_size))
    map_ref_in_tgt = map_ref_in_tgt * valid_ref_in_tgt
    map_tgt_in_ref = map_tgt_in_ref * valid_tgt_in_ref
    return map_tgt_in_ref, map_ref_in_tgt, valid_tgt_in_ref, valid_ref_in_tgt


def logq_to_quaternion(q):
    # return: quaternion with w, x, y, z
    #from geomap paper
    n = torch.norm(q, p=2, dim=1, keepdim=True)
    n = torch.clamp(n, min=1e-8)
    q = q * torch.sin(n)
    q = q / n
    q = torch.cat((torch.cos(n), q), dim=1)
    return q



def quat2mat(q):

    import nibabel.quaternions as nq
    return nq.quat2mat(q)

def unity_quaternion_to_logq(q):
    u = q[:, -1]
    v = q[:, :-1]
    norm = torch.norm(v, dim=1, keepdim=True).clamp(min=1e-8)
    out = v * (torch.acos(torch.clamp(u, min=-1.0, max=1.0)).reshape(-1, 1) / norm)


    return out

class LogQuatLoss(nn.Module):
    def __init__(self, criterion=nn.L1Loss()):
        super(LogQuatLoss, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor([-3]).cuda(), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor([0]).cuda(), requires_grad=True)
        self.criterion = criterion
        self.rot_criterion = nn.L1Loss()

    def forward(self, q_hat, q):
        t = q[:, :3]
        log_q = unity_quaternion_to_logq(q[:, 3:])
        t_hat = q_hat[:, :3]
        log_q_hat = q_hat[:, 3:]

        loss = self.criterion(t_hat, t)*torch.exp(-self.beta) + self.beta + self.rot_criterion(log_q_hat, log_q)*torch.exp(-self.gamma) + self.gamma
        return loss