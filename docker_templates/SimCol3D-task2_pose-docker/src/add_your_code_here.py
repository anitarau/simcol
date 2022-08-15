"""
SimCol-to-3D challenge - MICCAI 2022
Challenge link: https://www.synapse.org/#!Synapse:syn28548633/wiki/617126
Task 1: Depth prediction in simulated colonoscopy
Task 2: Camera pose estimation in simulated colonoscopy

This is a dummy example to illustrate how participants should format their prediction outputs.
Please direct questions to the discussion forum: https://www.synapse.org/#!Synapse:syn28548633/discussion/default
"""
import numpy as np
import glob
from PIL import Image
import torch
from torchvision import transforms
import os

to_tensor = transforms.ToTensor()

def predict_pose(im1_path, im2_path, output_folder):
    """
    param im1_path: Path to image 1
    param im2_path: Path to image 2
    param output_folder: Path to folder where output will be saved
    predict the relative pose between the image pair and save it to a .txt file.
    """

    ### replace below with your own prediction pipeline ###
    # We generate random poses as an example
    predicted_pose = np.random.rand(4,4)
    # Output should be a 4x4 relative camera pose [R t; 0001], where R is a 3x3 rotation matrix, and t is a 3x1
    # translation vector between the two input images. The output should represent the position of camera 2's origin in
    # camera 1's frame. Please see the file read_poses.py provided by the challenge organizers to find out how to
    # compute the relative pose between two cameras:  https://www.synapse.org/#!Synapse:syn29430445
    # Note: R should be a valid rotation matrix.

    ### Output and save your prediction in the correct format ###
    out_file = im1_path.split('/')[-1].strip('.png') + '_to_' + im2_path.split('/')[-1].strip('.png') + '.txt'
    assert predicted_pose.shape == (4,4), \
        "Wrong size of predicted pose, expected (7,) got {}".format(list(predicted_pose.shape))
    write_file =  open(output_folder + out_file, 'w')
    write_file.write(" ".join(map(str, predicted_pose.flatten())))
    write_file.close()
    print(out_file + ' saved')
    ### Double check that the organizers' evaluation pipeline will correctly reload your poses (uncomment below) ###
    """
    read_file = open(output_folder + out_file, 'r')
    reloaded_pose = []
    for line in read_file:
        reloaded_pose.append(list(map(float, line.split())))
    read_file.close()
    reloaded_pose = np.array(reloaded_pose).reshape(4,4)
    if np.sum(np.abs(reloaded_pose - predicted_pose)) == 0:
        print('Prediction will be correctly reloaded by organizers')
    """

def predict_depth(im_path, output_folder):
    """
    param im_path: Input path for a single image
    param output_folder: Path to folder where output will be saved
    predict the depth for an image and save in the correct formatting as .npy file
    """
    ### replace below with your own prediction pipeline ###
    # We apply noise to the labels as an example. During testing labels will not be available.
    gt_depth = to_tensor(Image.open(im_path.replace('FrameBuffer','Depth'))).squeeze()/255/256 # please use this to load ground truth depth during training
    predicted_depth = gt_depth.numpy() + np.random.normal(scale=0.01,size=gt_depth.shape)
    # The output depth should be in the range [0,1] similar to the input format. Note that a depth of 1 of the output
    # depth should correspond to 20cm in the world. The organizers will clip all values <0 and >1 to a valid range [0,1]
    # and multiply the output by 20 to obtain depth in centimeters.


    ### Output and save your prediction in the correct format ###
    out_file = im_path.split('/')[-1]
    assert predicted_depth.shape == (475,475), \
        "Wrong size of predicted depth, expected [475,475], got {}".format(list(predicted_depth.shape))

    np.save(output_folder + out_file, np.float16(predicted_depth))  # save a np.float16() to reduce file size
    # Note: Saving as np.float16() will lead to minor loss in precision


    ### Double check that the organizers' evaluation pipeline will correctly reload your depths (uncomment below) ###
    """
    reloaded_prediction = torch.from_numpy(np.load(output_folder + out_file + '.npy'))
    print('Half precision error: {} cm'.format(np.max((reloaded_prediction.numpy() - predicted_depth)) * 20))
    error = torch.mean(torch.abs(reloaded_prediction - gt_depth)) * 20  #Note that this is for illustration only. Validation pipeline will be published separately.
    print('Mean error: {} cm'.format(error.numpy()))
    """

def process_test_set(test_file,input_folder):
    """
    :param test_file: .txt file including all test folders
    :param input_folder: Path to root folder

    Loop through all test folders to find all RGB images. Process all image pairs that are k=1 frames apart.
    Predict pose in forward AND backward direction.
    For an input folder with 1201 RGB images, you should obtain 1201 .npy-files and 2400 .txt-files in the output folder
    """

    test_folders = []
    for line in open(test_file,'r'):
        test_folders.append(line.strip('\n'))

    for folder in test_folders:
        output_folder = input_folder.replace('test_data','outputs') + folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(output_folder + ' created')
        else:
            print(output_folder + ' exists')
        all_files = glob.glob(input_folder + folder + '/FrameBuffer*')
        all_files.sort()
        print(str(len(all_files)) + ' images found in ' + folder)
        for idx in range(len(all_files) -1):
            im1_path = all_files[idx]
            im2_path = all_files[idx+1]
            predict_pose(im1_path, im2_path, output_folder)  # Predict pose in forward direction.
            predict_pose(im2_path, im1_path, output_folder)  # Predict pose in backward direction.
            predict_depth(im1_path, output_folder)
        predict_depth(all_files[-1], output_folder) # evaluate last element omitted in loop above

#def main():

    # root_folder = '/path/to/simcolValidation/'
    # test_file = root_folder + 'src/dummy_test_file.txt'
    # data_folder = root_folder + 'test_data/'


    # process_test_set(test_file,data_folder)

#if __name__ == "__main__":
#    main()