from __future__ import division
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def logq_to_quaternion(q):
    # return: quaternion with w, x, y, z
    #from geomap paper
    n = torch.norm(q, p=2, dim=1, keepdim=True)
    n = torch.clamp(n, min=1e-8)
    q = q * torch.sin(n)
    q = q / n
    q = torch.cat((torch.cos(n), q), dim=1)
    return q
    #return torch.cat([first, second])



def quat2mat(q):

    import nibabel.quaternions as nq
    return nq.quat2mat(q)