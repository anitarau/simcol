"""
Bimodal Camera Pose Prediction for Endoscopy.

Anita Rau, a.rau.16@ucl.ac.uk, 2023
"""

import time
import csv
import datetime
import torch
import torch.optim
import torch.utils.data

import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from path import Path
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter

import models
from utils import args_to_yaml
from datasets.sequence_folders_quat import SequenceFolder
from loss_functions import logq_to_quaternion, quat2mat, LogQuatLoss, compute_proj_idx
from inverse_warp import get_bins_quat
from options import OptionsTrain
from align_traj import align, get_traj

best_error = -1
n_iter = 0

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
torch.autograd.set_detect_anomaly(True)


def main():
    global best_error, n_iter, device
    args = OptionsTrain().parse()
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    save_path = Path(args.name)
    args.save_path = 'checkpoints'/save_path/timestamp
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    training_writer = SummaryWriter(args.save_path)


    train_set = SequenceFolder(
        args.data,
        seed=args.seed,
        train=True,
        sequence_length=args.sequence_length,
        dataset=args.dataset,
        frames_apart=args.frames_apart,
        resize=args.with_resize,
        train_file=args.train_file,
        im_size=args.im_size,
        depth=True
    )


    val_set = SequenceFolder(
        args.data,
        seed=args.seed,
        train=False,
        sequence_length=args.sequence_length,
        dataset=args.val_dataset,
        frames_apart=args.frames_apart,
        resize=args.with_resize,
        test_file=args.val_file,
        im_size=args.im_size,
        gap=args.frames_apart,
    )

    print('Images are {} frames apart'.format(args.frames_apart))
    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    if not args.embedded_size:
        if args.im_size == 256:
            args.embedded_size = 16
        elif args.im_size == 480:
            args.embedded_size = 30
        elif args.im_size == 128:
            args.embedded_size = 8
        else:
            raise ValueError('What is the feature size?')

    # create model
    pose_net = models.PoseCorrNet(fs=args.fs, pose_decoder=args.pose_decoder).to(device)

    log_quat_loss = LogQuatLoss()

    print("=> done creating models")

    if args.pretrained_pose:
        print("=> using pre-trained weights for PoseResNet")
        weights = torch.load(args.pretrained_pose)
        pose_net.load_state_dict(weights['state_dict'], strict=False)


    pose_net = torch.nn.DataParallel(pose_net, device_ids=[0])

    print('=> setting adam solver')
    optim_params = [
        {'params': pose_net.parameters(), 'lr': args.lr},
        {'params': log_quat_loss.parameters(), 'lr': args.lr}
    ]
    optimizer = torch.optim.Adam(optim_params,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.97)


    logger = TermLogger(n_epochs=args.epochs, train_size=min(len(train_loader), args.epoch_size), valid_size=len(val_loader))
    logger.epoch_bar.start()
    args_to_yaml(args.save_path + '/settings.yml', args)


    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)

        # train for one epoch
        logger.reset_train_bar()
        train_loss = train(args, train_loader, pose_net, optimizer, args.epoch_size, logger, training_writer, log_quat_loss)
        logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))

        # evaluate and log on validation set
        logger.reset_valid_bar()
        errors, error_names, ATE, RTE = validate(args, val_loader, pose_net, epoch, logger)
        error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))
        logger.valid_writer.write(' * Avg {}'.format(error_string))

        for error, name in zip(errors, error_names):
            training_writer.add_scalar(name, error, epoch)
        training_writer.add_scalar('ATE', ATE, epoch)
        training_writer.add_scalar('RTE', RTE, epoch)


        pose_net_state = {
            'epoch': epoch + 1,
            'state_dict': pose_net.module.state_dict()
        }
        torch.save(pose_net_state, args.save_path / '{}_{}'.format('exp_pose', 'checkpoint.pth.tar'))
        scheduler.step()

    logger.epoch_bar.finish()


def train(args, train_loader, pose_net, optimizer, epoch_size, logger, train_writer, log_quat_loss):
    global n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)

    softmax = nn.Softmax(dim=1)
    crossX = nn.CrossEntropyLoss(reduction='none')

    pose_net.train()
    end = time.time()
    logger.train_bar.update(0)
    bins = get_bins_quat(args.frames_apart)

    for i, (tgt_img, ref_imgs, intrinsics, int_inv, pose_gt, pose_inv_gt, tgt_gt_depth, ref_gt_depths, tgt_name, ref_name, pose_gt_mat, pose_inv_gt_mat) in enumerate(train_loader):
        log_losses = i > 0 and n_iter % args.print_freq == 0

        data_time.update(time.time() - end)
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device).float()
        pose_gt = pose_gt.to(device).float()
        pose_inv_gt = pose_inv_gt.to(device).float()
        pose_gt_mat = pose_gt_mat.to(device).float()
        pose_inv_gt_mat = pose_inv_gt_mat.to(device).float()

        poses, poses_inv, confs_raw, confs_inv_raw, scores_AB, max_indices = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)

        pred_direction = torch.argmax(confs_raw, 1).float().detach() 
        pred_inv_direction = torch.argmax(confs_inv_raw, 1).float().detach() 
        gt_direction = (pose_gt[:, 2] < 0).float()
        gt_inv_direction = (pose_inv_gt[:, 2] < 0).float()
        accuracy_pose = 1 - torch.mean(torch.abs(gt_direction - pred_direction))
        accuracy_pose_inv = 1 - torch.mean(torch.abs(gt_inv_direction - pred_inv_direction))

        lossX1 = crossX(confs_raw, gt_direction.long())
        lossX2 = crossX(confs_inv_raw, gt_inv_direction.long())
        loss_X = torch.mean(lossX1 + lossX2)

        confs_sm = softmax(confs_raw)
        confs_inv_sm = softmax(confs_inv_raw)

        if args.binned==1:
            binned_poses = bins + poses
            binned_poses_inv = bins + poses_inv

            confidence_b0 = confs_sm[:, 0].view(-1, 1).repeat(1, 6)  
            confidence_b1 = confs_sm[:, 1].view(-1, 1).repeat(1, 6)
            pred_poses = binned_poses[:, 0, :] * confidence_b0 + binned_poses[:, 1, :] * confidence_b1

            confidence_inv_b0 = confs_inv_sm[:, 0].view(-1, 1).repeat(1, 6) 
            confidence_inv_b1 = confs_inv_sm[:, 1].view(-1, 1).repeat(1, 6)
            pred_poses_inv = binned_poses_inv[:, 0, :] * confidence_inv_b0 + binned_poses_inv[:, 1, :] * confidence_inv_b1
        else:
            pred_poses = poses[:, 0, :]
            pred_poses_inv = poses_inv[:, 0, :]

        pose_loss = 0.5 * (log_quat_loss(pred_poses, pose_gt) + log_quat_loss(pred_poses_inv, pose_inv_gt))
        pose_error = 0.5 * (torch.mean(torch.sum(torch.abs(pred_poses[:, :3].detach() - pose_gt[:, :3].detach()),1)) + \
                    torch.mean(torch.sum(torch.abs(pred_poses_inv[:, :3].detach() - pose_inv_gt[:, :3].detach()), 1))).detach()

        abs_error = torch.sqrt(torch.sum((pred_poses[:, :3] - pose_gt[:, :3])**2, 1))
        gt_length = torch.sqrt(torch.sum((pose_gt[:, :3])**2, 1))
        rel_pose_error = torch.median(abs_error / gt_length)

        loss_total = args.class_w * loss_X + pose_loss

        pred_rot_quat = logq_to_quaternion(pred_poses[:, 3:]).detach().cpu().numpy()
        gt_rot_quat = torch.cat([pose_gt[:, -1:], pose_gt[:, 3:-1]], 1).detach().cpu().numpy()

        thetas = []
        for j in range(pred_rot_quat.shape[0]):
            rot_diff = quat2mat(pred_rot_quat[j, :]) @ np.linalg.inv(quat2mat(gt_rot_quat[j, :]))
            rot = (np.trace(rot_diff) - 1) / 2
            thetas.append(np.arccos(rot) * 180 / np.pi)
        rot_error = np.median(np.stack(thetas))


        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        
        if log_losses:
            train_writer.add_scalar('loss_X', loss_X.item(), n_iter)
            train_writer.add_scalar('accuracy pose', accuracy_pose.item(), n_iter)
            train_writer.add_scalar('accuracy pose inv', accuracy_pose_inv.item(), n_iter)
            train_writer.add_scalar('trans pose loss', pose_error.item(), n_iter)
            train_writer.add_scalar('loss', pose_loss.item(), n_iter)
            train_writer.add_scalar('beta', log_quat_loss.beta.item(), n_iter)
            train_writer.add_scalar('gamma', log_quat_loss.gamma.item(), n_iter)
            train_writer.add_scalar('rel trans poss loss', rel_pose_error.item(), n_iter)
            train_writer.add_scalar('rot loss', rot_error, n_iter)
            del pose_error, rel_pose_error

        if n_iter % 1000 == 0:
            tgt_gt_depth = tgt_gt_depth.to(device)
            ref_gt_depths = [depths.to(device) for depths in ref_gt_depths]
            tgt_flow, ref_flow, tgt_mask, ref_mask = compute_proj_idx(intrinsics, tgt_gt_depth, [ref_gt_depths[1]], [pose_gt_mat],
                                                [pose_inv_gt_mat], embedded_size=args.embedded_size, resize=True)  # args.embedded_size


            fig, ax = plt.subplots(3, 1, figsize=(10, 13))
            ttl = pose_inv_gt.cpu().detach().numpy()[0, :3]
            plt.title('{:1.2f}  {:1.2f}  {:1.2f}'.format(ttl[0], ttl[1], ttl[2]))
            h = args.embedded_size
            w = args.embedded_size
            im_width = args.im_size
            patch_size = im_width//w
            im1 = tgt_img.cpu().detach().numpy()[0].transpose((1, 2, 0)) * 0.225 + 0.45
            im2 = ref_imgs[1][0].cpu().detach().numpy().transpose((1, 2, 0)) * 0.225 + 0.45
            map1 = (scores_AB[0]*tgt_mask).cpu().detach().numpy()[0]
            map2 = (scores_AB[1]*ref_mask).cpu().detach().numpy()[0]
            map1 = ((scores_AB[1][:,1,:,:] * args.embedded_size +scores_AB[1][:,0,:,:])).cpu().detach().numpy()[0]
            map2 = ((scores_AB[0][:,1,:,:] * args.embedded_size +scores_AB[0][:,0,:,:])).cpu().detach().numpy()[0]
            tgt = (((tgt_flow[:,1,:,:] * args.embedded_size +tgt_flow[:,0,:,:]))[0]).detach().cpu().numpy()
            ref = (((ref_flow[:, 1, :, :] * args.embedded_size + ref_flow[:, 0, :, :]))[0]).detach().cpu().numpy()
            max1 = scores_AB[2].cpu().detach().numpy()[0]
            max2 = scores_AB[3].cpu().detach().numpy()[0]
            y_in_A = max_indices['i_in_A'].cpu().detach().numpy() * patch_size + patch_size // 2
            x_in_A = max_indices['j_in_A'].cpu().detach().numpy() * patch_size + patch_size // 2
            y_in_B = max_indices['i_in_B'].cpu().detach().numpy() * patch_size + patch_size // 2
            x_in_B = max_indices['j_in_B'].cpu().detach().numpy() * patch_size + im_width + patch_size // 2
            ax[0].imshow(np.concatenate([im1, im2], 1))
            ax[0].axis('off')
            ax[0].plot(x_in_A[:15], y_in_A[:15], 'o')
            ax[0].plot(x_in_B[:15], y_in_B[:15], 'o')
            for b in range(0, 15):
                ax[0].plot([x_in_A[b], x_in_B[b]], [y_in_A[b], y_in_B[b]], '-')
            ax[1].imshow(np.concatenate([map1, map2], 1), vmin=0, vmax=h * w - 1)
            ax[1].axis('off')
            ax[2].imshow(np.concatenate([tgt, ref], 1), vmin=0, vmax=h * w - 1)
            ax[2].axis('off')

            train_writer.add_figure('corr', fig, n_iter)


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        logger.train_bar.update(i+1)
        n_iter += 1
        if i % args.print_freq == 0:
            logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
        if i >= epoch_size - 1:
            break

    return losses.avg[0]


@torch.no_grad()
def validate(args, val_loader, pose_net, epoch, logger):
    global device
    batch_time = AverageMeter()
    losses = AverageMeter(i=7, precision=4)
    softmax = nn.Softmax(dim=1)

    pose_net.eval()
    crossX = nn.CrossEntropyLoss(reduction='none')
    bins = get_bins_quat(args.frames_apart)
    end = time.time()
    logger.valid_bar.update(0)
    gts_inv = []
    preds_inv = []

    gt_rot_inv, pred_rot_inv = [], []

    for i, (tgt_img, ref_imgs, intrinsics, int_inv, pose_gt, pose_inv_gt, _, _, tgt_name, ref_name, pose_gt_mat, pose_inv_gt_mat) in enumerate(val_loader):

        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device).float()
        pose_gt = pose_gt.to(device).float()
        pose_inv_gt = pose_inv_gt.to(device).float()
        pose_gt_mat = pose_gt_mat.to(device).float()
        pose_inv_gt_mat = pose_inv_gt_mat.to(device).float()

        # compute output
        poses, poses_inv, confs_raw, confs_inv_raw, _, _ = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)

        pred_direction = torch.argmax(confs_raw, 1).float()  
        pred_inv_direction = torch.argmax(confs_inv_raw, 1).float() 
        accuracy_pose = 1 - torch.mean(torch.abs(torch.ones_like(pred_direction) - pred_direction.detach()))
        accuracy_pose_inv = 1 - torch.mean(torch.abs(torch.zeros_like(pred_direction) - pred_inv_direction.detach()))
        accuracy = 0.5 * (accuracy_pose_inv + accuracy_pose)

        lossX1 = crossX(confs_raw, torch.ones_like(confs_raw[:, 0]).cuda().long())
        lossX2 = crossX(confs_inv_raw, torch.zeros_like(confs_raw[:, 0]).cuda().long())
        loss_X = torch.mean(lossX1 + lossX2)

        confs_sm = softmax(confs_raw)
        confs_inv_sm = softmax(confs_inv_raw)

        if args.binned==1:
            binned_poses = bins + poses
            binned_poses_inv = bins + poses_inv

            confidence_b0 = confs_sm[:, 0].view(-1, 1).repeat(1, 6)  # b,4,4
            confidence_b1 = confs_sm[:, 1].view(-1, 1).repeat(1, 6)
            pred_poses = binned_poses[:, 0, :] * confidence_b0 + binned_poses[:, 1, :] * confidence_b1

            confidence_inv_b0 = confs_inv_sm[:, 0].view(-1, 1).repeat(1, 6)  # b,
            confidence_inv_b1 = confs_inv_sm[:, 1].view(-1, 1).repeat(1, 6)
            pred_poses_inv = binned_poses_inv[:, 0, :] * confidence_inv_b0 + binned_poses_inv[:, 1, :] * confidence_inv_b1
        else:
            pred_poses = poses[:, 0, :]
            pred_poses_inv = poses_inv[:, 0, :]

        abs_error = torch.sqrt(torch.sum((pred_poses[:, :3] - pose_gt[:, :3])**2, 1))
        gt_length = torch.sqrt(torch.sum((pose_gt[:, :3])**2, 1))
        rel_pose_error = torch.median(abs_error / gt_length)

        abs_pose_error = 0.5 * (torch.mean(torch.sum(torch.abs(pred_poses[:, :3].detach() - pose_gt[:, :3].detach()),1)) + \
                    torch.mean(torch.sum(torch.abs(pred_poses_inv[:, :3].detach() - pose_inv_gt[:, :3].detach()), 1))).detach()
        pred_rot_quat = logq_to_quaternion(pred_poses[:, 3:]).detach().cpu().numpy()
        gt_rot_quat = torch.cat([pose_gt[:, -1:], pose_gt[:, 3:-1]], 1).detach().cpu().numpy()

        thetas = []
        for j in range(pred_rot_quat.shape[0]):
            rot_diff = quat2mat(pred_rot_quat[j, :]) @ np.linalg.inv(quat2mat(gt_rot_quat[j, :]))
            rot = (np.trace(rot_diff) - 1) / 2
            thetas.append(np.arccos(rot) * 180 / np.pi)
        rot_error = np.median(np.stack(thetas))

        losses.update([loss_X, accuracy, accuracy_pose, accuracy_pose_inv, abs_pose_error, rel_pose_error, rot_error])

        gts_inv.append(pose_inv_gt.cpu().numpy()[:, :3])
        preds_inv.append(pred_poses_inv.detach().cpu().numpy()[:, :3])
        pred_rot_inv_quat = logq_to_quaternion(pred_poses_inv[:, 3:]).detach().cpu().numpy()
        gt_rot_inv_quat = torch.cat([pose_inv_gt[:, -1:], pose_inv_gt[:, 3:-1]], 1).detach().cpu().numpy()
        pred_rot_inv.append(quat2mat(pred_rot_inv_quat[0, :]))
        gt_rot_inv.append(quat2mat(gt_rot_inv_quat[0, :]))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Loss {}'.format(batch_time, losses))

    gts_inv = np.stack(gts_inv).squeeze()
    preds_inv = np.stack(preds_inv).squeeze()
    gt_rot_inv = np.stack(gt_rot_inv).squeeze()
    pred_rot_inv = np.stack(pred_rot_inv).squeeze()

    ## Forward
    gt_traj, gt_traj_4x4, Ps = get_traj(np.eye(4), gt_rot_inv, gts_inv)
    pred_traj_accumulated, pred_traj_accumulated_4x4, _ = get_traj(np.eye(4), pred_rot_inv, preds_inv)

    rot, trans, trans_error, scale, P_world = align(pred_traj_accumulated.transpose(), gt_traj.transpose())

    P_world = np.concatenate((P_world, np.array([0.0, 0.0, 0.0, 1.0]).reshape((1, 4))), 0)
    world_poses = (P_world @ pred_traj_accumulated_4x4) 

    ATE, RTE = compute_ate_rte(gt_traj_4x4, world_poses)

    return losses.avg, ['Loss_X', 'Accuracy', 'Accuracy Pose', 'Accuracy Pose_inv', 'TransLoss', 'RelTransLoss', 'RotLoss'], ATE, RTE


def compute_pose_with_inv(pose_net, tgt_img, ref_imgs):
    i = 1
    pose, conf, scores_AB, max_indices = pose_net(tgt_img, ref_imgs[i])
    pose_inv, conf_inv, _, _ = pose_net(ref_imgs[i], tgt_img)

    return pose, pose_inv, conf, conf_inv, scores_AB, max_indices


def compute_ate_rte(gt, pred, delta=1):
    errs = []
    for i in range(pred.shape[0]-delta):
        Q = np.linalg.inv(gt[i,:,:]) @ gt[i+delta,:,:]
        P = np.linalg.inv(pred[i,:,:]) @ pred[i+delta,:,:]
        E = np.linalg.inv(Q) @ P
        t = E[:3, -1]
        trans = np.linalg.norm(t, ord=2)
        errs.append(trans)

    errs = np.array(errs)

    ATE = np.median(np.linalg.norm((gt[:, :, -1] - pred[:, :, -1]), ord=2, axis=1))
    RTE = np.median(errs)

    return ATE, RTE

if __name__ == '__main__':
    main()
