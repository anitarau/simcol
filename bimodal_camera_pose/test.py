"""
Bimodal Camera Pose Prediction for Endoscopy.

Anita Rau, a.rau.16@ucl.ac.uk, 2023
"""

import torch
import numpy as np
import models
from datasets.sequence_folders_quat import SequenceFolder

import matplotlib.pyplot as plt
from inverse_warp import get_bins_quat
from loss_functions import logq_to_quaternion, quat2mat
import torch.nn as nn
from options import OptionsTest
from align_traj import align
from mpl_toolkits.axes_grid1 import make_axes_locatable

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@torch.no_grad()
def main():
    args = OptionsTest().parse()
    args.pretrained_posenet = 'bimodal_camera_pose/trained_models/posenet_binned/posenet.tar'

    args.fs = 256
    args.dataset = 'S'
    args.frames_apart = 5
    args.binned = 1
    args.im_size = 256
    ATEs_for = []
    ATEs_back = []
    RTEs_for = []
    RTEs_back = []
    accus = []
    accus_back= []
    rots_for = []
    rots_back = []
    avg_translations = []
    total_translations = []
    for i in range(5):  # image pairs are five frames apart by default, so average over five runs with start index i = 0,...,4
        atef, ateb, rtef, rteb, rotf, rotb, acc_for, acc_back ,avg_trans,total_trans = eval(args, i)
        ATEs_for.append(atef)
        ATEs_back.append(ateb)
        RTEs_for.append(rtef)
        RTEs_back.append(rteb)
        accus.append(acc_for)
        accus_back.append(acc_back)
        rots_for.append(rotf)
        rots_back.append(rotb)
        avg_translations.append(avg_trans)
        total_translations.append(total_trans)
    print('\n AVERAGED RESULTS:\n')
    print('ATE forward average: ', np.mean(ATEs_for))
    print('ATE backward average: ', np.mean(ATEs_back))
    print('RTE forward average: ', np.mean(RTEs_for))
    print('RTE backward average: ', np.mean(RTEs_back))
    print('Rot forward average: ', np.mean(rots_for)*180/np.pi)
    print('Rot backward average: ', np.mean(rots_back)*180/np.pi)
    print('Accuracy for average: ', np.mean(accus))
    print('Accuracy back average: ', np.mean(accus_back))
    print('avg step size', np.mean(avg_translations))
    print('avg total length', np.mean(total_translations))

def eval(args, offset):
    print('offset:', offset)
    args.with_resize = True
    weights = torch.load(args.pretrained_posenet)

    pose_net = models.PoseCorrNet(fs=args.fs, pose_decoder=args.pose_decoder).to(device)
    pose_net.load_state_dict(weights['state_dict'], strict=True)
    pose_net.eval()

    val_set = SequenceFolder(
        args.data,
        seed=1,
        train=False,
        sequence_length=args.sequence_length,
        dataset=args.dataset,
        frames_apart=args.frames_apart,
        resize=args.with_resize,
        gap=args.frames_apart,
        offset=offset,
        test_file=args.test_file,
        im_size=args.im_size,
    )

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    gts = []
    preds = []
    gts_inv = []
    preds_inv = []

    gt_rot, gt_rot_inv, pred_rot, pred_rot_inv = [], [],[],[]
    acc_for = []
    acc_back = []
    confidence = []
    bins = get_bins_quat(args.frames_apart)
    softmax = nn.Softmax(dim=1)
    first_pose = np.eye(4)

    for i, (tgt_img, ref_imgs, _, _, pose_gt, pose_inv_gt, _, _, tgt_name,ref_name,pose_mat,pose_inv_mat)in enumerate(val_loader):

        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        pose_gt = pose_gt.to(device).float()
        pose_inv_gt = pose_inv_gt.to(device).float()

        poses, poses_inv, confs_raw, confs_inv_raw, scores_AB, max_indices = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)

        confs_sm = softmax(confs_raw)
        confs_inv_sm = softmax(confs_inv_raw)
        confidence.append(torch.max(confs_sm).detach().cpu().numpy())

        if args.binned==1:
            binned_poses = bins + poses
            binned_poses_inv = bins + poses_inv

            confidence_b0 = confs_sm[:, 0].view(-1, 1).repeat(1, 6) 
            confidence_b1 = confs_sm[:, 1].view(-1, 1).repeat(1, 6)
            pred_poses = binned_poses[:, 0, :] * confidence_b0 + binned_poses[:, 1, :] * confidence_b1

            confidence_inv_b0 = confs_inv_sm[:, 0].view(-1, 1).repeat(1, 6) 
            confidence_inv_b1 = confs_inv_sm[:, 1].view(-1, 1).repeat(1, 6)
            pred_poses_inv = binned_poses_inv[:, 0, :] * confidence_inv_b0 + binned_poses_inv[:, 1, :] * confidence_inv_b1
        else:
            pred_poses = poses[:, 0, :]
            pred_poses_inv = poses_inv[:, 0, :]

        pred_direction = (pred_poses[:, 2] < 0).float() 
        pred_inv_direction = (pred_poses_inv[:, 2] < 0).float() 

        gt_direction = (pose_gt[:, 2] < 0).float()
        gt_inv_direction = (pose_inv_gt[:, 2] < 0).float()
        accuracy_pose = 1 - torch.mean(torch.abs(gt_direction - pred_direction))
        accuracy_pose_inv = 1 - torch.mean(torch.abs(gt_inv_direction - pred_inv_direction))
        acc_for.append(accuracy_pose_inv.cpu().numpy())
        acc_back.append(accuracy_pose.cpu().numpy())

        gts.append(pose_gt.cpu().numpy()[:, :3])
        preds.append(pred_poses.detach().cpu().numpy()[:, :3])
        gts_inv.append(pose_inv_gt.cpu().numpy()[:, :3])
        preds_inv.append(pred_poses_inv.detach().cpu().numpy()[:, :3])

        pred_rot_inv_quat = logq_to_quaternion(pred_poses_inv[:, 3:]).detach().cpu().numpy()
        gt_rot_inv_quat = torch.cat([pose_inv_gt[:, -1:], pose_inv_gt[:, 3:-1]], 1).detach().cpu().numpy()
        pred_rot_quat = logq_to_quaternion(pred_poses[:, 3:]).detach().cpu().numpy()
        gt_rot_quat = torch.cat([pose_gt[:, -1:], pose_gt[:, 3:-1]], 1).detach().cpu().numpy()

        pred_rot_inv.append(quat2mat(pred_rot_inv_quat[0, :]))
        gt_rot_inv.append(quat2mat(gt_rot_inv_quat[0, :]))
        pred_rot.append(quat2mat(pred_rot_quat[0, :]))
        gt_rot.append(quat2mat(gt_rot_quat[0, :]))


    gts = np.stack(gts).squeeze()
    preds = np.stack(preds).squeeze()
    gts_inv = np.stack(gts_inv).squeeze()
    preds_inv = np.stack(preds_inv).squeeze()
    gt_rot_inv = np.stack(gt_rot_inv).squeeze()
    pred_rot_inv = np.stack(pred_rot_inv).squeeze()
    gt_rot = np.stack(gt_rot).squeeze()
    pred_rot = np.stack(pred_rot).squeeze()
    confidence = np.stack(confidence)
    acc_for = np.mean(acc_for)
    acc_back = np.mean(acc_back)


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



    ## Forward
    gt_traj, gt_traj_4x4, Ps = get_traj(first_pose, gt_rot_inv, gts_inv)
    pred_traj_accumulated, pred_traj_accumulated_4x4, rels = get_traj(first_pose, pred_rot_inv, preds_inv)

    ## Back
    gt_traj_back, gt_traj_4x4_back, Ps_back = get_traj(gt_traj_4x4[-1], gt_rot, gts, 'backward')
    pred_traj_accumulated_back, pred_traj_accumulated_4x4_back, _ = get_traj(gt_traj_4x4[-1], pred_rot, preds, 'backward')

    ## Forward
    rot, trans, trans_error, scale, P_world = align(pred_traj_accumulated.transpose(), gt_traj.transpose())

    ## Back
    rot, trans, trans_error, scale_b, P_world_back = align(pred_traj_accumulated_back.transpose(), gt_traj_back.transpose())
    model_aligned_back = scale_b * rot @ pred_traj_accumulated_back.transpose() + trans.reshape(3,1)
    pred_xyz_aligned_back = model_aligned_back.transpose()

    ## Forward
    ATE, RTE, errs, ROT, gt_rot_mag = compute_ate_rte(gt_traj_4x4, pred_traj_accumulated_4x4)
    error_names = ['ATE', 'RTE', 'Accuracy', 'ROT']
    print('')
    print("Results forward ", args.pretrained_posenet.split('checkpoints/')[-1])
    print("\t {:>10}, {:>10}, {:>15}, {:>10}".format(*error_names))
    print("mean \t {:10.4f}cm, {:10.4f}cm, {:10.2f}%, {:10.2f}deg".format(ATE, RTE, acc_for*100, ROT*180/np.pi))
    print('')

    #Back
    ATEb, RTEb, errs_back, ROTb, gt_rot_magb = compute_ate_rte(gt_traj_4x4_back,pred_traj_accumulated_4x4_back, plot=False)
    error_names = ['ATE', 'RTE', 'Accuracy', 'ROT']
    print("Results backward", args.pretrained_posenet.split('checkpoints/')[-1])
    print("\t {:>10}, {:>10}, {:>15}, {:>10}".format(*error_names))
    print("mean \t {:10.4f}cm, {:10.4f}cm, {:10.2f}%, {:10.2f}deg".format(ATEb, RTEb, acc_back*100, ROTb*180/np.pi))
    print('')
    print('The average translation in cm per delta step is ', np.mean(np.linalg.norm(gts_inv, ord=2, axis=1)))
    print('The average rotation in degrees per delta step is ', gt_rot_magb)
    print('The total length of the trajectory in cm is ', np.sum(np.linalg.norm(gts_inv, ord=2, axis=1)))

    if False:  # True if you want to plot trajectories
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
        ax[0, 0].axis('off')
        ax[0, 1].axis('off')
        ax[1, 0].axis('off')
        ax[1, 1].axis('off')

        ang = 0
        for i in range(2):
            for j in range(2):

                ax = fig.add_subplot(2, 2, ang+1, projection='3d')
                ax.view_init(elev=30*ang, azim=90*ang)

                ax.plot(gt_traj_back[:-1, 0], gt_traj_back[:-1, 1], gt_traj_back[:-1, 2], '-', c='blue')
                ax.plot(pred_xyz_aligned_back[:, 0], pred_xyz_aligned_back[:, 1], pred_xyz_aligned_back[:, 2], '-', c='green')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')

                ang += 1

        fig.suptitle('Predicted and Ground Truth Trajectory')
        fig.savefig('traj_real' + str(offset) +'.png')

    return ATE, ATEb, RTE, RTEb, ROT, ROTb, acc_for, acc_back, np.mean(np.linalg.norm(gts_inv, ord=2, axis=1)), np.sum(np.linalg.norm(gts_inv, ord=2, axis=1))


def compute_pose_with_inv(pose_net, tgt_img, ref_imgs):

    i = 1
    pose, conf, _, _ = pose_net(tgt_img, ref_imgs[i])
    pose_inv, conf_inv, scores_AB, max_indices = pose_net(ref_imgs[i], tgt_img)

    return pose, pose_inv, conf, conf_inv, scores_AB, max_indices


def compute_ate_rte(gt, pred, delta=1, plot=True):

    errs = []
    rot_err = []
    rot_gt = []
    trans_gt = []
    for i in range(pred.shape[0]-delta):
        Q = np.linalg.inv(gt[i, :, :]) @ gt[i+delta, :, :]
        P = np.linalg.inv(pred[i, :, :]) @ pred[i+delta, :, :]
        E = np.linalg.inv(Q) @ P
        t = E[:3, -1]
        t_gt = Q[:3, -1]
        trans = np.linalg.norm(t, ord=2)
        errs.append(trans)
        tr = np.arccos((np.trace(E[:3, :3]) -1)/2)
        gt_tr = np.arccos((np.trace(Q[:3, :3]) -1)/2)
        rot_err.append(tr)
        rot_gt.append(gt_tr)
        trans_gt.append(np.linalg.norm(t_gt, ord=2))

    errs = np.array(errs)

    scale_factor = np.sum(gt[:, :, -1] * pred[:, :, -1]) / np.sum(pred[:, :, -1] ** 2)
    ATE_endo = np.median(np.linalg.norm((gt[:, :, -1] - scale_factor * pred[:, :, -1]), ord=2, axis=1))
    ATE = np.median(np.linalg.norm((gt[:, :, -1] - pred[:, :, -1]), ord=2, axis=1))
    RTE = np.median(errs)
    ROT = np.median(rot_err)

    return ATE, RTE, errs, ROT, np.mean(rot_gt) * 180 / np.pi



if __name__ == '__main__':
    main()
