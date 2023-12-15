# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

"""
Bimodal Camera Pose Prediction for Endoscopy.

Edited by Anita Rau, a.rau.16@ucl.ac.uk, 2023
"""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn import functional as F
import torchvision.models as tvmodels

def featureL2Norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature,norm)


class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features=1, num_frames_to_predict_for=1, stride=1, out=6):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features
        self.out_dim = out
        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, self.out_dim  * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU(inplace=False)

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)
        pose = 0.01 * out.view(-1, self.out_dim)

        return pose



class PoseConfidenceVgg(nn.Module):
    def __init__(self, fc_layer=1024*12*16, num_input_features=1, num_frames_to_predict_for=1, stride=1, out=2):
        super(PoseConfidenceVgg, self).__init__()
        self.fc_layer = fc_layer
        self.confidence = nn.Sequential(
            nn.Linear(fc_layer, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, out),
            # nn.Softmax()
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, self.fc_layer)
        return self.confidence(x)


class PoseCorrNet(nn.Module):
    """incorporate NC layer"""
    def __init__(self, fs=2, pose_decoder='conv',resnet_fs=16):
        super(PoseCorrNet, self).__init__()
        self.confidence = PoseConfidenceVgg(fc_layer=2*16**4) 
        if pose_decoder == 'conv':
            self.decoder_b1 = PoseDecoder([fs*2])
            self.decoder_b2 = PoseDecoder([fs*2])
        else:
            raise NotImplementedError


        resnet_feature_layers = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']
        self.resnet_features = tvmodels.resnet18(weights='IMAGENET1K_V1')
        last_layer = 'layer3'
        resnet_module_list = [getattr(self.resnet_features, l) for l in resnet_feature_layers]
        last_layer_idx = resnet_feature_layers.index(last_layer)
        self.resnet_features = nn.Sequential(*resnet_module_list[:last_layer_idx + 1])

        self.softmax = nn.Softmax(dim=1)
        self.correlation = FeatureCorrelation(shape='4D', normalization=False)
        self.fs = fs

    def init_weights(self):
        pass

    def forward(self, img1, img2):

        resnet_f1 = self.resnet_features(img1)
        resnet_f2 = self.resnet_features(img2)

        corr4d = self.correlation(resnet_f1, resnet_f2)
        corr4d = MutualMatching(corr4d)

        batch_size = corr4d.size(0)
        feature_size_h = corr4d.size(2)
        feature_size_w = corr4d.size(3)
        nc_B_Avec = corr4d.view(
            batch_size, feature_size_h * feature_size_w, feature_size_h, feature_size_w
        ) 
        nc_A_Bvec = corr4d.view(
            batch_size, feature_size_h, feature_size_w, feature_size_h * feature_size_w
        ).permute(
            0, 3, 1, 2
        ) 

        nc_B_Avec = nn.functional.softmax(nc_B_Avec, 1) 
        nc_A_Bvec = nn.functional.softmax(nc_A_Bvec, 1)

        # compute matching scores
        max_scores_A, _ = torch.max(nc_B_Avec, dim=1)
        max_scores_B, _ = torch.max(nc_A_Bvec, dim=1)
        scores_A = torch.argmax(nc_B_Avec, dim=1).float()
        scores_B = torch.argmax(nc_A_Bvec, dim=1).float()

        scores_A_odo_i = scores_A // feature_size_w
        scores_A_odo_j = torch.fmod(scores_A, feature_size_w)
        scores_B_odo_i = scores_B // feature_size_w
        scores_B_odo_j = torch.fmod(scores_B, feature_size_w)
        scores_A_odo = torch.stack([scores_A_odo_i, scores_A_odo_j],1)
        scores_B_odo = torch.stack([scores_B_odo_i, scores_B_odo_j], 1)

        i_indeces_in_B = scores_A[0, :, :] // feature_size_w
        j_indeces_in_B = torch.fmod(scores_A[0, :, :], feature_size_w)
        sorted_scores = torch.argsort(max_scores_A.view(batch_size, feature_size_h * feature_size_w), 1, descending=True)

        max_i_indeces_in_A = sorted_scores[0, :20] // feature_size_w
        max_j_indeces_in_A = torch.fmod(sorted_scores[0, :20], feature_size_w)
        max_i_indeces_in_B = i_indeces_in_B[max_i_indeces_in_A, max_j_indeces_in_A]
        max_j_indeces_in_B = j_indeces_in_B[max_i_indeces_in_A, max_j_indeces_in_A]
        max_indices = dict()
        max_indices['i_in_A'] = max_i_indeces_in_A
        max_indices['j_in_A'] = max_j_indeces_in_A
        max_indices['i_in_B'] = max_i_indeces_in_B
        max_indices['j_in_B'] = max_j_indeces_in_B

        feature_A = resnet_f1
        feature_B = resnet_f2

        confidence = self.confidence(torch.cat([nc_A_Bvec, nc_B_Avec], 1))

        pose_b1 = self.decoder_b1([[torch.cat([feature_A, feature_B], 1)]])
        pose_b2 = self.decoder_b2([[torch.cat([feature_A, feature_B], 1)]])
        poses = torch.stack([pose_b1, pose_b2], 1)

        return poses, confidence, [scores_A_odo, scores_B_odo, max_scores_A, max_scores_B], max_indices


class FeatureCorrelation(torch.nn.Module):
    # From https://github.com/ignacio-rocco/ncnet/blob/master/lib/model.py, edited by AR 2023
    def __init__(self, shape='3D', normalization=True):
        super(FeatureCorrelation, self).__init__()
        self.normalization = normalization
        self.shape = shape
        self.ReLU = nn.ReLU()

    def forward(self, feature_A, feature_B):
        if self.shape == '3D':
            raise NotImplementedError
        elif self.shape == '4D':
            b, c, hA, wA = feature_A.size()
            b, c, hB, wB = feature_B.size()
            # reshape features for matrix multiplication
            feature_A = feature_A.view(b, c, hA * wA).transpose(1, 2)  # size [b,c,h*w]
            feature_B = feature_B.view(b, c, hB * wB)  # size [b,c,h*w]
            # perform matrix mult.
            feature_mul = torch.bmm(feature_A, feature_B)
            correlation_tensor = feature_mul.view(b, hA, wA, hB, wB).unsqueeze(1)

        if self.normalization:
            correlation_tensor = featureL2Norm(self.ReLU(correlation_tensor))

        return correlation_tensor


def MutualMatching(corr4d):
    # From https://github.com/ignacio-rocco/ncnet/blob/master/lib/model.py
    # mutual matching
    batch_size, ch, fs1, fs2, fs3, fs4 = corr4d.size()

    corr4d_B = corr4d.view(batch_size, fs1 * fs2, fs3, fs4) 
    corr4d_A = corr4d.view(batch_size, fs1, fs2, fs3 * fs4)

    # get max
    corr4d_B_max, _ = torch.max(corr4d_B, dim=1, keepdim=True)
    corr4d_A_max, _ = torch.max(corr4d_A, dim=3, keepdim=True)

    eps = 1e-5
    corr4d_B = corr4d_B / (corr4d_B_max + eps)
    corr4d_A = corr4d_A / (corr4d_A_max + eps)

    corr4d_B = corr4d_B.view(batch_size, 1, fs1, fs2, fs3, fs4)
    corr4d_A = corr4d_A.view(batch_size, 1, fs1, fs2, fs3, fs4)

    corr4d = corr4d * (corr4d_A * corr4d_B)  

    return corr4d

