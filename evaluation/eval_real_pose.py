import os
import numpy as np
import glob
from scipy.spatial.transform import Rotation as R
from camera_visualizer_eval import CameraPoseVisualizerTotalTraj
import collections
import sys

warning1 = False
warning2 = False
warning3 = False

INPUT_PATH = sys.argv[1]
GT_PATH = sys.argv[2]


def check_pose(pred):
    global warning1, warning2, warning3
    assert pred.shape == (16,), \
        "Wrong size of predicted pose, expected (16,), got {}".format(list(pred.shape))
    if np.max(pred.reshape((4,4))[:3,:3]) > 1:
        if not warning1:
            print("Warning: Rotation matrix element > 1 found")
        warning1 = True
    if np.max(pred.reshape((4,4))[:3,:3]) < -1:
        if not warning2:
            print("Warning: Rotation matrix element < -1 found")
        warning2 = True
    if (pred.reshape((4,4))[3,:] != np.array([0, 0, 0, 1])).any():
        if not warning3:
            print("Warning: The last row of any relative pose should be [0, 0, 0, 1]")
        warning3 = True

    return pred


def load_pose(pred_file):
    for line in open(pred_file, 'r'):
        pose = np.array(list(map(float, line.split())))
        break

    pred = check_pose(pose)
    return pred


def process_poses(test_folders, INPUT_PATH, GT_PATH):
    delta = 1

    for traj in test_folders:
        print(traj)
        assert os.path.exists(INPUT_PATH+traj+'/pose/'), 'No input folder found'

    for traj in test_folders:
        input_file_list = sorted(glob.glob(INPUT_PATH + traj + "/pose/out*.txt"))
        forward = input_file_list[0::2]
        backward = input_file_list[1::2]
        backward.reverse()
        preds = []
        folder = traj.split('ims_')[-1].split('_')[0]
        sequence = traj.split('ims_')[-1].split('_')[1]
        colmap_traj = get_colmap_poses(sequence, folder, GT_PATH, forward)

        colmap_traj = np.stack(colmap_traj)
        normalizer = np.linalg.inv(colmap_traj[0])  # set first pose to np.eye(4)
        colmap_abs = []
        for k in range(len(colmap_traj)):
            colmap_abs.append(normalizer @ colmap_traj[k])
        colmap_abs = np.stack(colmap_abs)

        # Absolute colmap poses to relative poses
        colmap_rel_poses = []
        for i in range(0,colmap_abs.shape[0]-1,delta):
            out = get_relative_pose(colmap_abs[i],colmap_abs[i+delta])
            colmap_rel_poses.append(out)

        first_pose = np.eye(4)

        for i in range(0,len(forward),delta):
            im1_path = forward[i]
            pred_rel_pose = load_pose(im1_path)
            preds.append(pred_rel_pose.reshape((4,4)))

        pred_traj, pred_traj_4x4 = get_traj(first_pose, np.array(preds))
        gt_traj, gt_traj_4x4 = get_traj(first_pose, np.array(colmap_rel_poses))  # translate back to absolute poses
        if (colmap_abs - gt_traj_4x4 < 10e-12).all():
            print('Switch from absolute to relative to absolute poses correct')
        accuracy = np.mean((np.array(preds)[:,2,-1] * np.array(colmap_rel_poses)[:,2,-1])>0)
        scale = get_scale(colmap_abs[:,:3,:], pred_traj_4x4[:, :3, :])
        pred_traj_4x4[:,:3,-1] = pred_traj_4x4[:,:3,-1] * scale
        ATE, RTE, errs, ROT, gt_rot_mag = compute_errors(gt_traj_4x4, pred_traj_4x4)

        step_size = []
        for g in range(len(colmap_traj) - 1):
            diff = np.linalg.norm(colmap_traj[g, :3, -1] - colmap_traj[g + 1, :3, -1], ord=2)
            step_size.append(diff)
        total_length = np.sum(step_size)
        avg_length = np.mean(step_size)
        plot_pred_trajectory(pred_traj_4x4, gt_traj_4x4, traj.split('/')[-1])
        print('------------------------')
        print('Results {} {}'.format(folder, sequence))
        print('FORWARD')
        print("Scale {:8.4f}".format(scale))
        print("ATE {:10.4f}".format(ATE))
        print("RTE {:10.4f}".format(RTE))
        print("ROT {:10.4f} degrees".format(ROT))
        print("Accuracy in z-direction {:2.2f}".format(accuracy))
        print('The average translation per delta step is ', avg_length)
        print('The average rotation in degrees per delta step is ', gt_rot_mag)
        print('The total length of the trajectory is ', total_length)

def plot_pred_trajectory(pred_traj_ours, gt_traj, name):

    for j in range(len(gt_traj) - 2, len(gt_traj) - 1):
        visualizer = CameraPoseVisualizerTotalTraj()

        visualizer.extrinsic2lineAbs(gt_traj, 'c')
        visualizer.extrinsic2lineAbs(pred_traj_ours, 'orange')

        for i in range(j + 2):
            visualizer.extrinsic2pyramidAbs(pred_traj_ours[i], 'orange', 0.3, model='ours')
            visualizer.extrinsic2pyramidAbs(gt_traj[i], 'c', 0.3)

        visualizer.customize_legend()
        visualizer.show(name)

def get_scale(gt, pred):
    # Negative scales are not allowed
    scale_factor = np.sum(np.abs(gt[:, :, -1] * pred[:, :, -1])) / np.sum(pred[:, :, -1] ** 2)
    return scale_factor

def get_colmap_poses(scene, folder, root, rel_pose_names):
    # Note that the colmap outputs have to be exported as .txt files with colmap running the line below:
    # colmap model_converter --input_path /path/to/082/sparse/6/ --output_path /path/to/082/sparse/6 --output_type TXT
    txt = open(root + folder + '/sparse/' + scene +'/images.txt', 'r')
    poses = dict()
    for i, line in enumerate(txt):
        if i > 3 and i % 2 == 0:
            #print(line)
            pose = line.split()[1:8]
            name = line.split()[9]
            #print(name)
            pose = np.array(list(map(float, pose)))
            qxyzw = np.concatenate([pose[1:4], pose[0:1]])
            txyz = pose[4:].reshape(3, 1)
            Rot = R.from_quat(qxyzw).as_dcm()  # 3 x 3
            P = np.concatenate([Rot, txyz], 1)
            ones = np.zeros((1, 4))
            ones[0, -1] = 1
            P = np.concatenate([P, ones], 0)
            poses[name] = np.linalg.inv(P)  # need the inverse to get camera position, as P is world to cam proj

    od = collections.OrderedDict(sorted(poses.items()))

    abs_pose_names = []
    abs_poses = []
    for i in range(len(rel_pose_names)):
        im_name = rel_pose_names[i].split('_to_')[0].split('/')[-1] + '.png'
        abs_pose_names.append(im_name)
        abs_poses.append(od[im_name])
    im_name = rel_pose_names[-1].split('_to_')[1].replace('txt','png')
    abs_pose_names.append(im_name)
    abs_poses.append(od[im_name])
    print('')
    return abs_poses


def get_relative_pose(pose_t0, pose_t1):
    """
    :param pose_tx: 4x4 camera pose describing camera to world frame projection of camera x.
    :return: Position of camera 1's origin in camera 0's frame.
    """
    return np.matmul(np.linalg.inv(pose_t0), pose_t1)

def get_traj(first, P):
    traj, traj_4x4 = [], []
    next = first
    traj.append(next[:3, -1])
    traj_4x4.append(first)

    for i in range(0, P.shape[0]):
        Pi = P[i]
        next = np.matmul(next, Pi)
        traj.append(next[:3, -1])
        traj_4x4.append(next)

    traj = np.array(traj)
    traj_4x4 = np.array(traj_4x4)
    return traj, traj_4x4

def compute_errors(gt, pred, delta=1):
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
        tr = np.arccos((np.trace(E[:3, :3]).clip(-3,3) -1)/2)
        gt_tr = np.arccos((np.trace(Q[:3, :3]) -1)/2)
        rot_err.append(tr)
        rot_gt.append(gt_tr)
        trans_gt.append(np.linalg.norm(t_gt, ord=2))

    errs = np.array(errs)

    ATE = np.median(np.linalg.norm((gt[:, :, -1] - pred[:, :, -1]), ord=2, axis=1))
    RTE = np.median(errs)
    ROT = np.median(rot_err)

    return ATE, RTE, errs, ROT * 180 / np.pi, np.mean(rot_gt) * 180 / np.pi

def main():

    test_folders = ['/RealColon_082/ims_082_6_OP'
                    '/RealColon_082/ims_082_12_OP',
                    '/RealColon_082/ims_082_25_OP',
                    '/RealColon_084/ims_084_25_OP',
                    '/RealColon_084/ims_084_26_OP',
                    '/RealColon_084/ims_084_30_OP',
                    '/RealColon_086/ims_086_18_OP']


    process_poses(test_folders, INPUT_PATH, GT_PATH)

if __name__ == "__main__":
    main()



