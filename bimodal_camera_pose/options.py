"""
Bimodal Camera Pose Prediction for Endoscopy.

Anita Rau, a.rau.16@ucl.ac.uk, 2023
"""

import argparse


class OptionsTrain:
    def __init__(self):
        self.options = None
        self.parser = argparse.ArgumentParser(
            description='Colonoscopy Pose Trainer',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('data', metavar='DIR', help='path to dataset')
        self.parser.add_argument('--folder-type', type=str, choices=['sequence', 'pair'], default='sequence',
                            help='the dataset dype to train')
        self.parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
        self.parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
        self.parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
        self.parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                            help='manual epoch size (will match dataset size if not set)')
        self.parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='mini-batch size')
        self.parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR',
                            help='initial learning rate')
        self.parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum for sgd, alpha parameter for adam')
        self.parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
        self.parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
        self.parser.add_argument('--print_freq', default=10, type=int, metavar='N', help='print frequency')
        self.parser.add_argument('--frames-apart', default=1, type=int,
                            help='number of subsequent frames between an image pair')
        self.parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
        self.parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                            help='csv where to save per-epoch train and valid stats')
        self.parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH',
                            help='csv where to save per-gradient descent train stats')
        self.parser.add_argument('--log-output', action='store_true', help='will log dispnet outputs at validation step')
        self.parser.add_argument('--resnet-layers', type=int, default=18, choices=[18, 50],
                            help='number of ResNet layers for depth estimation.')
        self.parser.add_argument('--num-scales', '--number-of-scales', type=int, help='the number of scales', metavar='W',
                            default=1)
        self.parser.add_argument('-p', '--photo-loss-weight', type=float, help='weight for photometric loss', metavar='W',
                            default=1)
        self.parser.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss',
                            metavar='W', default=0.1)
        self.parser.add_argument('-c', '--geometry-consistency-weight', type=float, help='weight for depth consistency loss',
                            metavar='W', default=0.5)
        self.parser.add_argument('-e', '--endo-slam-weight', type=float, help='weight for total EndoSLAM loss', metavar='W',
                            default=1)
        self.parser.add_argument('--class-w', '--cross-entropy-loss-weight', type=float, help='weight for class net', default=0.1)

        self.parser.add_argument('--with-ssim', type=int, default=1, help='with ssim or not')
        self.parser.add_argument('--with-mask', type=int, default=1,
                            help='with the the mask for moving objects and occlusions or not')
        self.parser.add_argument('--cross-num', type=str, default='1', help='which cross validation index')

        self.parser.add_argument('--with-auto-mask', type=int, default=0, help='with the the mask for stationary points')
        self.parser.add_argument('--with-pretrain', type=int, default=1, help='with or without imagenet pretrain for resnet')
        self.parser.add_argument('--with-resize', type=int, default=1, help='resize input images yes/no')
        self.parser.add_argument('--dataset', type=str, choices=['kitti', 'S'], default='S',
                            help='the dataset to train')
        self.parser.add_argument('--val-dataset', type=str, choices=['kitti','S'],
                            default='S', help='the dataset to train')

        self.parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH',
                            help='path to pre-trained dispnet model')
        self.parser.add_argument('--pretrained-pose', dest='pretrained_pose', default=None, metavar='PATH',
                            help='path to pre-trained Pose net model')
        self.parser.add_argument('--name', dest='name', type=str, required=True,
                            help='name of the experiment, checkpoints are stored in checpoints/name')
        self.parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                            help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                                 ' zeros will null gradients outside target image.'
                                 ' border will only null gradients of the coordinate outside (x or y)')
        self.parser.add_argument('--with-gt', action='store_true', help='use ground truth for validation. \
                            You need to store it in npy 2D arrays see data/kitti_raw_loader.py for an example')
        self.parser.add_argument('--train-file', type=str, default='train_M.txt', help='name of training file')
        self.parser.add_argument('--val-file', type=str, default='val.txt', help='name of val file')
        self.parser.add_argument('--pose-model', type=str, default='posecorrnet', help='Choose network to predict pose')
        self.parser.add_argument('--pose-decoder', type=str, default='conv', choices=['conv','fc','resnet'], help='Choose network to decode pose')
        self.parser.add_argument('--fs', type=int, default=256, choices=[256], help='number of features in PoseNet')
        self.parser.add_argument('--im-size', type=int, default=256, help='Target input size into network')
        self.parser.add_argument('--embedded-size', type=int, default=None, help='optical flow map size')
        self.parser.add_argument('--binned', type=int, default=1, help='bimodal pose flag')




    def parse(self, *args, **kwargs):
        self.options = self.parser.parse_args(*args, **kwargs)
        return self.options



class OptionsTest:
    def __init__(self):
        self.options = None

        self.parser = argparse.ArgumentParser(description='Colonoscopy Pose Tester', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('data', metavar='DIR', help='path to dataset')

        self.parser.add_argument("--pretrained_posenet", type=str, help="pretrained PoseNet path")
        self.parser.add_argument('-j', '--workers', default=1, type=int, metavar='N', help='number of data loading workers')
        self.parser.add_argument('--with-resize', type=int,  default=1, help='resize input images yes/no')
        self.parser.add_argument('--frames-apart', default=1, type=int, help='number of subsequent frames between an image pair')
        self.parser.add_argument("--dataset-dir", type=str, help="Dataset directory")
        self.parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for testing', default=3)
        self.parser.add_argument("--output-dir", default=None, type=str, help="Output directory for saving predictions in a big 3D numpy file")
        self.parser.add_argument('--dataset', type=str, choices=['kitti','S'], default='S', help='the dataset to test')
        self.parser.add_argument('--test-file', type=str, default='test_file.txt', help='name of validation file')
        self.parser.add_argument('--pose-model', type=str, default='posecorrnet', help='Choose network to predict pose')
        self.parser.add_argument('--pose-decoder', type=str, default='conv', choices=['conv','fc','resnet'], help='Choose network to decode pose')
        self.parser.add_argument('--fs', type=int, default=256, choices=[256], help='number of features in PoseNet')
        self.parser.add_argument('--im-size', type=int, default=256, help='Target input size into network')
        self.parser.add_argument('--isTrain', type=bool, default=0)
        self.parser.add_argument('--input_nc', type=int, default=3, help='number of input channels to depth net')
        self.parser.add_argument('--output_nc', type=bool, default=1, help='number of output channels of depth net')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--binned', type=int, default=1, help='bimodal pose flag')
        self.parser.add_argument('--resnet-layers', type=int, default=18, choices=[18, 50],
                            help='number of ResNet layers for depth estimation.')
        self.parser.add_argument('--with-pretrain', type=int, default=1, help='with or without imagenet pretrain for resnet')
        self.parser.add_argument('--with-ssim', type=int, default=1, help='with ssim or not')
        self.parser.add_argument('--with-mask', type=int, default=1,
                            help='with the the mask for moving objects and occlusions or not')
        self.parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                            help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                                 ' zeros will null gradients outside target image.'
                                 ' border will only null gradients of the coordinate outside (x or y)')



    def parse(self, *args, **kwargs):
        self.options = self.parser.parse_args(*args, **kwargs)
        return self.options