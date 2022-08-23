import os
import numpy as np
import glob
from scipy.spatial.transform import Rotation as R
import sys

INPUT_PATH = sys.argv[1]
GT_PATH = sys.argv[2]

warning1 = False
warning2 = False
warning3 = False

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

    # First check if all test sequences are present
    for traj in test_folders:
        print(traj)
        assert os.path.exists(INPUT_PATH+traj+'/pose/'), 'No input folder found'
        input_file_list = np.sort(glob.glob(INPUT_PATH + traj + "/pose/FrameBuffer*.txt"))
        if traj[18] == 'I':
            assert len(input_file_list) == 600, 'Predictions missing in {}'.format(traj)
        else:
            assert len(input_file_list) == 1200, 'Predictions missing in {}'.format(traj)

    # Load and evaluate predictions
    for traj in test_folders:
        input_file_list = np.sort(glob.glob(INPUT_PATH + traj + "/pose/FrameBuffer*.txt"))
        preds = []
        sequence = traj.split('Frames_')[-1].strip('_OP')
        gt_abs_poses = get_gt_poses(sequence, GT_PATH)  # load ground truth poses from .txt file
        gt_rel_poses = []
        ## Absolute gt poses --> relative gt poses
        for i in range(0,gt_abs_poses.shape[0]-1,delta):
            out = get_relative_pose(gt_abs_poses[i],gt_abs_poses[i+delta])
            out[:3,-1] = out[:3,-1]
            gt_rel_poses.append(out)

        first_pose = np.eye(4)  # define arbitrarily

        # read predictions from files
        for i in range(0,len(input_file_list),delta):
            im1_path = input_file_list[i]
            pred_rel_pose = load_pose(im1_path)
            preds.append(pred_rel_pose.reshape((4,4)))

        ## Relative poses --> absolute poses
        gt_traj, gt_traj_4x4 = get_traj(first_pose, np.array(gt_rel_poses))  # get_traj() should map relative gt poses back to gt_abs_poses
        pred_traj, pred_traj_4x4 = get_traj(first_pose, np.array(preds))

        if (gt_abs_poses - gt_traj_4x4 < 10e-12).all():
            print('Correctly switching between absolute and relative ground truth poses.')

        ## Uncomment below for debugging
        #pred_traj_4x4 = gt_traj_4x4 #+ np.random.rand(gt_traj_4x4.shape[0],4,4)/1000
        #preds = np.array(gt_rel_poses) #+ np.random.rand(gt_traj_4x4.shape[0]-1,4,4)/1000

        scale = get_scale(np.array(gt_rel_poses), np.array(preds))

        pred_traj_4x4[:,:3,-1] = pred_traj_4x4[:,:3,-1] * scale
        ATE, RTE, errs, ROT, gt_rot_mag = compute_errors(gt_traj_4x4, pred_traj_4x4)
        print('------------------------')
        print('Results sequence ', sequence)
        print("Scale {:8.4f}".format(scale))
        print("ATE {:10.4f} cm".format(ATE))
        print("RTE {:10.4f} cm".format(RTE))
        print("ROT {:10.4f} degrees".format(ROT))

def get_scale(gt, pred):
    scale_factor = np.sum(gt[:, :3, -1] * pred[:, :3, -1])/np.sum(pred[:, :3, -1] ** 2)
    return scale_factor

def get_gt_poses(scene, root):
    """
    :param scene: Index of trajectory
    :param root: Root folder of dataset
    :return: all camera poses as quaternion vector and 4x4 projection matrix
    """
    locations = []
    rotations = []
    loc_reader = open(root + 'SavedPosition_' + scene + '.txt', 'r')
    rot_reader = open(root + 'SavedRotationQuaternion_' + scene + '.txt', 'r')
    for line in loc_reader:
        locations.append(list(map(float, line.split())))

    for line in rot_reader:
        rotations.append(list(map(float, line.split())))

    locations = np.array(locations)
    rotations = np.array(rotations)
    poses = np.concatenate([locations, rotations], 1)

    r = R.from_quat(rotations).as_dcm()

    TM = np.eye(4)
    TM[1, 1] = -1

    poses_mat = []
    for i in range(locations.shape[0]):
        ri = r[i]
        Pi = np.concatenate((ri, locations[i].reshape((3, 1))), 1)
        Pi = np.concatenate((Pi, np.array([0.0, 0.0, 0.0, 1.0]).reshape((1, 4))), 0)
        Pi_left = TM @ Pi @ TM   # Translate between left and right handed systems
        poses_mat.append(Pi_left)

    return np.array(poses_mat)


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
        if np.isnan(tr):
            print('')
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
    test_folders = ['/SyntheticColon_I_Test/Frames_S5_OP',
                   '/SyntheticColon_I_Test/Frames_S10_OP',
                   '/SyntheticColon_I_Test/Frames_S15_OP',
                   '/SyntheticColon_II_Test/Frames_B5_OP',
                   '/SyntheticColon_II_Test/Frames_B10_OP',
                   '/SyntheticColon_II_Test/Frames_B15_OP',
                   '/SyntheticColon_III_Test/Frames_O1_OP',
                   '/SyntheticColon_III_Test/Frames_O2_OP',
                   '/SyntheticColon_III_Test/Frames_O3_OP']

    process_poses(test_folders, INPUT_PATH, GT_PATH)

if __name__ == "__main__":
    main()



