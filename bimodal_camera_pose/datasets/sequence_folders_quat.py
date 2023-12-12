import torch.utils.data as data
import numpy as np
from path import Path
import random
from PIL import Image
from torchvision import transforms
from scipy.spatial.transform import Rotation as R


def load_as_float(path):
    return Image.open(path)


class SequenceFolder(data.Dataset):

    def __init__(self, root, seed=None, train=True, sequence_length=3, custom_transform=None, skip_frames=1,
                 dataset='S', resize=True, frames_apart=1, gap=1, offset=0,
                 train_file='train_file.txt', test_file='test_file.txt', im_size=256, depth=False):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root / train_file if train else self.root / test_file
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.custom_transform = custom_transform
        self.k = skip_frames
        self.frames_apart = frames_apart
        self.train = train
        self.gap = gap
        self.offset = offset
        self.depth = depth
        self.dataset = dataset

        self.crawl_folders(sequence_length, datasetname=dataset)

        self.to_norm_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])#, transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.GaussianBlur(3, sigma=(0.1, 2.0))])
        self.to_tensor = transforms.ToTensor()

        if self.train:
            self.resizer = transforms.Compose([transforms.Resize((im_size, im_size)), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)])
        else:
            self.resizer = transforms.Resize((im_size, im_size))

        self.depth_resizer = transforms.Resize((im_size, im_size))
        self.resize = resize
        self.scale_x = im_size / 475
        self.scale_y = im_size / 475
        self.Tx = np.eye(4)
        self.Tx[0, 0] = -1
        self.Ty = np.eye(4)
        self.Ty[1, 1] = -1



    def rescale_matrix(self, intrinsics_matrix):
        intrinsics_matrix[0, :] = intrinsics_matrix[0, :] * self.scale_x
        intrinsics_matrix[1, :] = intrinsics_matrix[1, :] * self.scale_y
        return intrinsics_matrix


    def crawl_folders(self, sequence_length, datasetname):
        # k skip frames
        sequence_set = []
        demi_length = (sequence_length - 1) // 2
        shifts = list(range(-demi_length * self.k, demi_length * self.k + 1, self.k))
        shifts.pop(demi_length)
        for scene in self.scenes:
            #print(scene)
            intrinsics = np.genfromtxt('/'.join(scene.split('/')[:-1]) + '/cam.txt').astype(np.float32).reshape((3, 3))
            imgs = sorted(scene.files('F*.png'))
            print('get: ', scene.split('Frames_' + self.dataset)[-1])
            abs_poses, abs_poses_mat = self.get_poses(scene.split('Frames_' + self.dataset)[-1], datasetname)

            if len(imgs) < sequence_length:
                continue

            if self.train:
                for i in range(demi_length * self.k + self.frames_apart - 1 + 2,
                               len(imgs) - demi_length * self.k - self.frames_apart + 1 - 2):

                    sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': [],
                              'ref_poses': [], 'tgt_pose': [abs_poses[i], abs_poses_mat[i]]}
                    for j in shifts:
                        sample['ref_imgs'].append(imgs[i + j * self.frames_apart])
                        sample['ref_poses'].append([abs_poses[i + j * self.frames_apart],
                                                    abs_poses_mat[i + j * self.frames_apart]])  # *2 for x=2
                    sequence_set.append(sample)

                    sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': [],
                              'ref_poses': [], 'tgt_pose': [abs_poses[i], abs_poses_mat[i]]}
                    for j in shifts:
                        sample['ref_imgs'].append(imgs[i + j * (self.frames_apart + 1)])
                        sample['ref_poses'].append([abs_poses[i + j * (self.frames_apart + 1)],
                                                    abs_poses_mat[i + j * (self.frames_apart + 1)]])
                    sequence_set.append(sample)

                    sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': [],
                              'ref_poses': [], 'tgt_pose': [abs_poses[i], abs_poses_mat[i]]}
                    for j in shifts:
                        sample['ref_imgs'].append(imgs[i + j * (self.frames_apart - 1)])
                        sample['ref_poses'].append([abs_poses[i + j * (self.frames_apart - 1)],
                                                    abs_poses_mat[i + j * (self.frames_apart - 1)]])
                    sequence_set.append(sample)


            else:
                shifts = [-1, 1]
                for i in range(self.offset + demi_length * self.k - 1,
                               len(imgs) - demi_length * self.k - self.frames_apart + 1,
                               self.gap):
                    sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': [], 'ref_poses': [],
                              'tgt_pose': [abs_poses[i], abs_poses_mat[i]]}
                    for j in shifts:
                        sample['ref_imgs'].append(imgs[i + j * self.frames_apart])
                        sample['ref_poses'].append([abs_poses[i + j * self.frames_apart],
                                                    abs_poses_mat[i + j * self.frames_apart]])  # *2 for x=2
                    sequence_set.append(sample)

                    

        # random.shuffle(sequence_set)
        self.samples = sequence_set

    def get_poses(self, scene, datasetname):
        locations = []
        rotations = []
        loc_reader = open(self.root / 'SavedPosition_' + datasetname + scene + '.txt', 'r')
        rot_reader = open(self.root / 'SavedRotationQuaternion_' + datasetname + scene + '.txt', 'r')
        for line in loc_reader:
            locations.append(list(map(float, line.split())))
        loc_reader.close

        for line in rot_reader:
            rotations.append(list(map(float, line.split())))
        rot_reader.close

        locations = np.array(locations)  # in cm
        rotations = np.array(rotations)
        poses = np.concatenate([locations, rotations], 1)

        #r = R.from_quat(rotations).as_dcm()
        r = R.from_quat(rotations).as_matrix()

        TM = np.eye(4)
        TM[1, 1] = -1

        poses_mat = []
        for i in range(locations.shape[0]):
            ri = r[i]  # np.linalg.inv(r[0])
            Pi = np.concatenate((ri, locations[i].reshape((3, 1))), 1)
            Pi = np.concatenate((Pi, np.array([0.0, 0.0, 0.0, 1.0]).reshape((1, 4))), 0)
            Pi_left = TM @ Pi @ TM
            poses_mat.append(Pi_left)

        return poses, np.array(poses_mat)

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_depth_path = str(sample['tgt']).replace('FrameBuffer', 'Depth').replace('_Frames', '')

        tgt_img = []
        ref_imgs = []
        intrinsics = np.copy(sample['intrinsics'])


        if self.resize:
            tgt_img = self.resizer(Image.open(sample['tgt']).convert('RGB'))
            ref_imgs = [self.resizer(Image.open(ref_img).convert('RGB')) for ref_img in sample['ref_imgs']]
            intrinsics = self.rescale_matrix(np.copy(sample['intrinsics']))

        else:
            tgt_img = Image.open(sample['tgt']).convert('RGB')
            ref_imgs = [Image.open(ref_img).convert('RGB') for ref_img in sample['ref_imgs']]
            intrinsics = np.copy(sample['intrinsics'])



        # pose_tm1 = sample['ref_poses'][0]
        pose_tp1 = sample['ref_poses'][1]
        tgt_pose = sample['tgt_pose']


        tgt_img = self.to_norm_tensor(np.array(tgt_img))[:3, :, :]
        ref_imgs = [self.to_norm_tensor(np.array(ref_img))[:3, :, :] for ref_img in ref_imgs]


        if self.depth:
            if self.resize:
                tgt_gt_depth = self.depth_resizer(Image.open(tgt_depth_path))  # get depth in cm
                ref_gt_depths = [
                    self.depth_resizer(Image.open(ref_depth.replace('FrameBuffer', 'Depth').replace('_Frames', ''))) for
                    ref_depth in sample['ref_imgs']]

            else:
                tgt_gt_depth = Image.open(tgt_depth_path)  # get depth in cm
                ref_gt_depths = [Image.open(ref_depth.replace('FrameBuffer', 'Depth').replace('_Frames', '')) for
                                 ref_depth in sample['ref_imgs']]


            tgt_gt_depth = self.to_tensor(np.array(tgt_gt_depth)) / 65000. * 20.  # get depth in cm
            ref_gt_depths = [self.to_tensor(np.array(ref_depth)) / 65000. * 20. for ref_depth in ref_gt_depths]

        # ground truth (tested by pplaying to images and warping)

        pose_tp1 = pose_tp1[1]
        tgt_pose = tgt_pose[1]

        pose_mat = np.matmul(np.linalg.inv(pose_tp1), tgt_pose)
        pose_inv_mat = np.matmul(np.linalg.inv(tgt_pose), pose_tp1)

        #quat_diff = R.from_dcm(pose_mat[:3, :3]).as_quat()
        #quat_diff_inv = R.from_dcm(pose_inv_mat[:3, :3]).as_quat()
        quat_diff = R.from_matrix(pose_mat[:3, :3]).as_quat()
        quat_diff_inv = R.from_matrix(pose_inv_mat[:3, :3]).as_quat()
        trans_from_quat = pose_mat[:3, -1]
        trans_from_quat_inv = pose_inv_mat[:3, -1]

        int_inv = np.linalg.inv(intrinsics)

        pose = np.concatenate([trans_from_quat, quat_diff])
        pose_inv = np.concatenate([trans_from_quat_inv, quat_diff_inv])

        if self.depth:
            return tgt_img, ref_imgs, intrinsics, int_inv, pose, pose_inv, tgt_gt_depth, ref_gt_depths, \
                   sample['tgt'].split('/')[-1].split('.')[0], sample['ref_imgs'][1].split('/')[-1].split('.')[
                       0], pose_mat, pose_inv_mat

        return tgt_img, ref_imgs, intrinsics, int_inv, pose, pose_inv, [], [], \
               sample['tgt'].split('/')[-1].split('.')[0], sample['ref_imgs'][1].split('/')[-1].split('.')[
                   0], pose_mat, pose_inv_mat



    def __len__(self):
        if self.train:
            return len(self.samples)
        else:
            return len(self.samples)
