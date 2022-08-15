import os
import numpy as np
import glob
from PIL import Image
import sys

INPUT_PATH = sys.argv[1]
GT_PATH = sys.argv[2]

warning1 = False
warning2 = False

def check_depth(pred):
    global warning1, warning2
    assert pred.shape == (475,475), \
        "Wrong size of predicted depth, expected [475,475], got {}".format(list(pred.shape))
    assert pred.dtype == np.float16, \
        "Wrong data type, expected float16, got {}".format(pred.dtype)
    if np.max(pred) > 1:
        if not warning1:
            print("Warning: Depths > 20cm encountered")
        warning1 = True
    if np.min(pred) < 0:
        if not warning2:
            print("Warning: Depths < 0cm encountered")
        warning2 = True

    return pred.clip(0,1)  # depths are clipped to (0,1) to avoid invalid depths

def load_depth(pred_file, gt_file):
    gt_depth = np.array(Image.open(gt_file.replace('FrameBuffer','Depth')))/255/256 # please use this to load ground truth depth during training and testing
    pred = np.load(pred_file)
    pred = check_depth(pred)
    return pred, gt_depth

def eval_depth(pred, gt_depth):
    # * 20 to get centimeters
    L1_error = np.mean(np.abs(pred * 20 - gt_depth * 20))
    rel_error = np.median(np.abs((pred * 20 - gt_depth * 20)/(gt_depth * 20 + 10e-5))) * 100
    RMSE_error = np.sqrt(np.mean((pred * 20 - gt_depth * 20)**2))
    return L1_error, rel_error, RMSE_error

def process_depths(test_folders, INPUT_PATH, GT_PATH):
    # first check if all the data is there
    for traj in test_folders:
        print(traj)
        assert os.path.exists(INPUT_PATH+traj+'/depth/'), 'No input folder found'
        input_file_list = np.sort(glob.glob(INPUT_PATH + traj + "/depth/FrameBuffer*.npy"))
        if traj[18] == 'I':
            assert len(input_file_list) == 601, 'Predictions missing in {}'.format(traj)
        else:
            assert len(input_file_list) == 1201, 'Predictions missing in {}'.format(traj)

    # loop through predictions
    for traj in test_folders:
        print('Processing ', traj)
        input_file_list = np.sort(glob.glob(INPUT_PATH + traj + "/depth/FrameBuffer*.npy"))
        L1_errors, rel_errors, rmses = [], [], []
        preds, gts = [], []
        for i in range(len(input_file_list)):
            file_name1 = input_file_list[i].split("/")[-1]
            #print(file_name1)
            im1_path = input_file_list[i]
            gt_depth_path = GT_PATH + traj.split('/')[-1].strip('_OP') +'/' + file_name1.replace('npy','png')
            pred_depth, gt_depth = load_depth(im1_path, gt_depth_path)
            preds.append(pred_depth)
            gts.append(gt_depth)
        gts_ = np.mean(np.mean(np.array(gts),1),1)
        preds_ = np.mean(np.mean(np.array(preds),1),1)
        scale = np.sum(preds_ * gts_) / np.sum(preds_ * preds_)  # monocular methods predict depth up to scale
        print('Scale: ', scale)

        for i in range(len(input_file_list)):
            L1_error, rel_error, rmse = eval_depth(preds[i]*scale, gts[i])
            L1_errors.append(L1_error)
            rel_errors.append(rel_error)
            rmses.append(rmse)
        print('Mean L1 error in cm: ', np.mean(L1_errors))
        print('Median relative error in %: ', np.mean(rel_errors))
        print('Mean RMSE in cm: ', np.mean(rmses))

def main():
    # The 9 test sequences have to organized in the submission .zip file as follows:
    test_folders = ['/SyntheticColon_I_Test/Frames_S5_OP',
                   '/SyntheticColon_I_Test/Frames_S10_OP',
                   '/SyntheticColon_I_Test/Frames_S15_OP',
                   '/SyntheticColon_II_Test/Frames_B5_OP',
                   '/SyntheticColon_II_Test/Frames_B10_OP',
                   '/SyntheticColon_II_Test/Frames_B15_OP',
                   '/SyntheticColon_III_Test/Frames_O1_OP',
                   '/SyntheticColon_III_Test/Frames_O2_OP',
                   '/SyntheticColon_III_Test/Frames_O3_OP']

    process_depths(test_folders, INPUT_PATH, GT_PATH)

if __name__ == "__main__":
    main()



