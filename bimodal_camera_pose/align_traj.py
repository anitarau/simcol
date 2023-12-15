"""
Bimodal Camera Pose Prediction for Endoscopy.

Anita Rau, a.rau.16@ucl.ac.uk, 2023
"""

import sys
import numpy as np
import argparse


def ralign(X,Y):
    m, n = X.shape

    mx = X.mean(1)
    my = Y.mean(1)
    Xc =  X - np.tile(mx, (n, 1)).T
    Yc =  Y - np.tile(my, (n, 1)).T

    sx = np.mean(np.sum(Xc*Xc, 0))
    sy = np.mean(np.sum(Yc*Yc, 0))

    Sxy = np.dot(Yc, Xc.T) / n

    U,D,V = np.linalg.svd(Sxy,full_matrices=True,compute_uv=True)
    V=V.T.copy()
    #print U,"\n\n",D,"\n\n",V
    r = np.linalg.matrix_rank(Sxy)
    d = np.linalg.det(Sxy)
    S = np.eye(m)
    if r > (m - 1):
        if ( np.det(Sxy) < 0 ):
            S[m, m] = -1;
        elif (r == m - 1):
            if (np.det(U) * np.det(V) < 0):
                S[m, m] = -1
        else:
            R = np.eye(2)
            c = 1
            t = np.zeros(2)
            return R,c,t

    R = np.dot( np.dot(U, S ), V.T)

    c = np.trace(np.dot(np.diag(D), S)) / sx
    t = my - c * np.dot(R, mx)

    return R,c,t

def get_traj(first, rots, trans, direction='forward'):
    traj = []
    traj_4x4 = []
    next = first
    traj.append(next[:3, -1])
    traj_4x4.append(first)
    Ps = []

    if direction=='forward':
        for i in range(0, rots.shape[0]):
            ri = rots[i, :, :]

            Pi = np.concatenate((ri, trans[i].reshape((3, 1))), 1)
            Pi = np.concatenate((Pi, np.array([0.0, 0.0, 0.0, 1.0]).reshape((1, 4))), 0)

            next = np.matmul(next, Pi)
            traj.append(next[:3, -1])
            traj_4x4.append(next)
            Ps.append(Pi)
    elif direction == 'backward':
        for i in range(rots.shape[0] - 1, -1, -1):
            ri = rots[i, :, :]

            Pi = np.concatenate((ri, trans[i].reshape((3, 1))), 1)
            Pi = np.concatenate((Pi, np.array([0.0, 0.0, 0.0, 1.0]).reshape((1, 4))), 0)

            next = np.matmul(next, Pi)
            traj.append(next[:3, -1])
            traj_4x4.append(next)
            Ps.append(Pi)


    traj = np.array(traj)
    traj_4x4 = np.array(traj_4x4)
    Ps= np.array(Ps)
    return traj, traj_4x4, Ps


def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).

    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)

    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)

    """
    np.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1, keepdims=True)
    data_zerocentered = data - data.mean(1, keepdims=True)

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])

    X = model
    Y = data
    m, n = X.shape

    mx = X.mean(1)
    my = Y.mean(1)
    Xc = X - np.tile(mx, (n, 1)).T
    Yc = Y - np.tile(my, (n, 1)).T

    sx = np.mean(np.sum(Xc * Xc, 0))
    sy = np.mean(np.sum(Yc * Yc, 0))

    Sxy = np.dot(Yc, Xc.T) / n
    U, D, V = np.linalg.svd(Sxy, full_matrices=True, compute_uv=True)
    V=V.T.copy()
    #print U,"\n\n",D,"\n\n",V
    r = np.linalg.matrix_rank(Sxy)
    d = np.linalg.det(Sxy)
    S = np.eye(m)
    if r > (m - 1):
        if ( np.linalg.det(Sxy) < 0 ):
            S[m-1, m-1] = -1
        elif (r == m - 1):
            if (np.linalg.det(U) * np.linalg.det(V) < 0):
                S[m-1, m-1] = -1


    R = np.dot( np.dot(U, S ), V.T)
    c = np.trace(np.dot(np.diag(D), S)) / sx
    t = my - c * np.dot(R, mx)


    U, d, Vh = np.linalg.linalg.svd(W.transpose())

    S = np.eye(3)
    if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U @ S @ Vh

    rotmodel = rot @ model_zerocentered
    dots = 0.0
    norms = 0.0

    for column in range(data_zerocentered.shape[1]):
        dots += np.dot(data_zerocentered[:, column].transpose(), rotmodel[:, column])
        normi = np.linalg.norm(model_zerocentered[:, column])
        norms += normi * normi

    s = float(dots / norms)

    print
    "scale: %f " % s

    trans = data.mean(1) - s * rot @ model.mean(1)

    model_aligned = s * rot @ model + trans.reshape(3,1)

    trans_error = 0

    P = np.concatenate([s * rot, trans.reshape(3,1)], 1)

    return rot, trans, trans_error, s, P
