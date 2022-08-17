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

