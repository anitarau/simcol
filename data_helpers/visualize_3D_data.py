import torch
from skimage.transform import resize as imresize
import numpy as np

import matplotlib.pyplot as plt
from camera_visualizer import CameraPoseVisualizerSlim
from scipy.spatial.transform import Rotation as R
import collections
from PIL import Image
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_growing_cloud(datasetname, scene):
    locations = []
    rotations = []
    loc_reader = open('/media/anita/DATA/tracking_data/SavedPosition_' + datasetname + scene + '.txt', 'r')
    rot_reader = open('/media/anita/DATA/tracking_data/SavedRotationQuaternion_' + datasetname + scene + '.txt',
                      'r')
    for line in loc_reader:
        locations.append(list(map(float, line.split())))
    loc_reader.close

    for line in rot_reader:
        rotations.append(list(map(float, line.split())))
    rot_reader.close

    locations = np.array(locations)  # in cm
    rotations = np.array(rotations)
    poses = np.concatenate([locations, rotations], 1)

    r = R.from_quat(rotations).as_dcm()

    TM = np.eye(4)
    TM[1, 1] = -1

    poses_mat = []
    for i in range(locations.shape[0]):
        ri = r[i]  # np.linalg.inv(r[0])
        Pi = np.concatenate((ri, locations[i].reshape((3, 1))), 1)
        Pi = np.concatenate((Pi, np.array([0.0, 0.0, 0.0, 1.0]).reshape((1, 4))), 0)
        Pi_left = TM @ Pi @ TM
        poses_mat.append(Pi_left)

    visualizer = CameraPoseVisualizerSlim([np.min(np.array(poses_mat)[:,0,-1]), np.max(np.array(poses_mat)[:,0,-1])],
                                          [np.min(np.array(poses_mat)[:,1,-1]), np.max(np.array(poses_mat)[:,1,-1])],
                                          [np.min(np.array(poses_mat)[:,2,-1]), np.max(np.array(poses_mat)[:,2,-1])])

    if datasetname == 'S':
        visualizer.ax3d1.view_init(elev=10, azim=80)
    elif datasetname == 'B':
        visualizer.ax3d1.view_init(elev=-30, azim=30)
    else:
        visualizer.ax3d1.view_init(elev=10, azim=30)

    intrinsics = np.eye(3)
    intrinsics[0, 0] = 227.6
    intrinsics[0, 2] = 227.6
    intrinsics[1, 1] = 237.5
    intrinsics[1, 2] = 237.5

    for j in range(0, len(poses_mat), 5):
        gt = poses_mat[j]
        depth0 = np.array(Image.open('/media/anita/DATA/tracking_data/Frames_'+datasetname + scene + '/Depth_' + str(j).zfill(4) + '.png'))/256/255 * 20

        im0 = plt.imread(
            '/media/anita/DATA/tracking_data/Frames_'+datasetname + scene + '/FrameBuffer_' + str(j).zfill(4) + '.png')
        im0 = im0[:, :, :3].reshape((-1, 3))

        cam_coords0 = pixel2cam(torch.tensor(depth0.reshape((1, 475, 475))).float(),
                                torch.tensor(intrinsics).inverse().float())
        cam_coords_flat0 = cam_coords0.reshape(1, 3, -1).numpy()

        rot_gt, tr_gt = gt[:3, :3], gt[:3, -1:]
        cloud_gt = rot_gt @ cam_coords_flat0 + tr_gt
        indeces = np.random.choice(225625, size=1000, replace=False)

        visualizer.ax3d1.scatter(cloud_gt[0, 0, indeces], cloud_gt[0, 1, indeces], cloud_gt[0, 2, indeces],
                                 c=im0[indeces, :],
                                 s=1)
        if j % 20 == 0:
            visualizer.extrinsic2pyramidAbs(poses_mat[j], 'b', 0.7)
    visualizer.show('pointcloud_' + datasetname + '_' + scene)

def pixel2cam(depth, intrinsics_inv):
    b, h, w = depth.size()
    pixel_coords = set_id_grid(depth)
    current_pixel_coords = pixel_coords[:, :, :h, :w].expand(
        b, 3, h, w).reshape(b, 3, -1)  # [B, 3, H*W]
    cam_coords = (intrinsics_inv @ current_pixel_coords).reshape(b, 3, h, w)
    return cam_coords * depth.unsqueeze(1)

def set_id_grid(depth):
    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(
        1, h, w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(
        1, h, w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1, h, w).type_as(depth)
    return torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


if __name__ == "__main__":
    datasetname = 'S'
    scene = '1'
    plot_growing_cloud(datasetname, scene)