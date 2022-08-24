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
    out_file = out_file.replace('.png','')
    assert predicted_depth.shape == (475,475), \
        "Wrong size of predicted depth, expected [475,475], got {}".format(list(predicted_depth.shape))

    np.save(output_folder + out_file, np.float16(predicted_depth))  # save a np.float16() to reduce file size
    print(output_folder + out_file + '.npy saved')
    # Note: Saving as np.float16() will lead to minor loss in precision


    ### Double check that the organizers' evaluation pipeline will correctly reload your depths (uncomment below) ###
    """
    reloaded_prediction = torch.from_numpy(np.load(output_folder + out_file + '.npy'))
    print('Half precision error: {} cm'.format(np.max((reloaded_prediction.numpy() - predicted_depth)) * 20))
    error = torch.mean(torch.abs(reloaded_prediction - gt_depth)) * 20  #Note that this is for illustration only. Validation pipeline will be published separately.
    print('Mean error: {} cm'.format(error.numpy()))
    """
