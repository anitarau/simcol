"""
SimCol-to-3D challenge - MICCAI 2022
Challenge link: https://www.synapse.org/#!Synapse:syn28548633/wiki/617126
Task 1: Depth prediction in simulated colonoscopy
Task 2: Camera pose estimation in simulated colonoscopy

This is a dummy example to illustrate how participants should format their prediction outputs.
Please direct questions to the discussion forum: https://www.synapse.org/#!Synapse:syn28548633/discussion/default

Task 2 - dummy docker script
"""

import numpy as np
import os
import glob
from add_your_code_here import predict_pose
import sys  

INPUT_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2]

## uncomment below to run without docker
#INPUT_PATH = '../images/input'
#OUTPUT_PATH = '../images/output'

if __name__ == "__main__":

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        print(OUTPUT_PATH +' created')
        os.makedirs(OUTPUT_PATH+'/pose/')
    else:
        print(OUTPUT_PATH +' exists')

    if not os.path.exists(OUTPUT_PATH+'/pose/'):
        os.makedirs(OUTPUT_PATH+'/pose/')

    if not os.path.exists(INPUT_PATH):
        print('No input folder found')
    else:
        print(INPUT_PATH +' exists')
        glob.__file__
        input_file_list = np.sort(glob.glob(INPUT_PATH + "/FrameBuffer*.png"))


    for i in range(len(input_file_list)-1):
        file_name1 = input_file_list[i].split("/")[-1]
        im1_path = INPUT_PATH + '/' + file_name1

        file_name2 = input_file_list[i+1].split("/")[-1]
        im2_path = INPUT_PATH + '/' + file_name2

        predict_pose(im1_path, im2_path, OUTPUT_PATH+'/pose/')

