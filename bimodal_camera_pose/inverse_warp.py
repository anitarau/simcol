from __future__ import division
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


pixel_coords = None


def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(
        1, h, w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(
        1, h, w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1, h, w).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i, size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(
        input_name, 'x'.join(expected), list(input.size()))


def pixel2cam(depth, intrinsics_inv):
    global pixel_coords
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(2) < h:
        set_id_grid(depth)
    current_pixel_coords = pixel_coords[:, :, :h, :w].expand(
        b, 3, h, w).reshape(b, 3, -1)  # [B, 3, H*W]
    cam_coords = (intrinsics_inv @ current_pixel_coords).reshape(b, 3, h, w)
    return cam_coords * depth.unsqueeze(1)


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    X_norm = 2*(X / Z)/(w-1) - 1
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.reshape(b, h, w, 2)


def euler2mat(angle):
    """Convert euler angles to rotation matrix.
     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat


def transquat2mat(p):
    ### TODO THIS IS NOT DONE
    """Convert 7D vector B x [tx,ty,tz, qx, qy, qz, qw] to pose matrix.
    Args:
        unity ground truth: x, y, z, w [:, 3:] and trans vector [:,:3]
    Returns:
        [B,3,4] pose matrix
    """
    quat = p[:, 3:]
    norm_quat = torch.cat([quat[:, :1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    x, y, z, w = norm_quat[:, 0], norm_quat[:,
                                            1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:, :1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:,
                                            1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:, 3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat

def pose_vec2mat4(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:, 3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    zeros = torch.zeros_like(transform_mat[:, 1:2, :])
    zeros[:,-1,-1] = 1.
    transform_mat = torch.cat([transform_mat, zeros], dim=1)
    return transform_mat

def get_bins(scale=1):
    bin1 = torch.eye(4).unsqueeze(0)
    bin1[:, 2, -1] = 0.1 * scale
    bin2 = torch.eye(4).unsqueeze(0)
    bin2[:, 2, -1] = -0.1 * scale
    bins = torch.cat([bin1, bin2], 0).cuda()
    return bins

def get_bins_chess(scale=1):
    bin1 = torch.eye(4).unsqueeze(0)
    bin1[:, 0, -1] = 0.6 * scale
    bin2 = torch.eye(4).unsqueeze(0)
    bin2[:, 0, -1] = -0.6 * scale
    bins = torch.cat([bin1, bin2], 0).cuda()
    return bins

def get_bins_quat(scale=1):
    bins = torch.zeros(1, 2, 6)
    bins[:,0,2] = 0.1 * scale
    bins[:,1,2] = -0.1 * scale
    return bins.cuda()


def pose_2vec2mat4(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 2, 6]
    Returns:
        A transformation matrix -- [B, 2, 4, 4]
    """
    B = vec.shape[0]
    vec = torch.cat([vec[:, 0, :], vec[:, 1, :]], 0)  # B*2, 6
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:, 3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)
    zeros = torch.zeros_like(transform_mat[:, 1:2, :])
    zeros[:,-1,-1] = 1.
    transform_mat = torch.cat([transform_mat, zeros], dim=1) # B*2, 4, 4
    transform_mat = torch.cat([transform_mat[:B, :, :].unsqueeze(1), transform_mat[B:, :, :].unsqueeze(1)], 1) # B, 2, 4, 4
    return transform_mat

def inverse_warp(img, depth, pose, intrinsics, rotation_mode='euler', padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_points: Boolean array indicating point validity
    """
    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'BHW')
    check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = img.size()

    cam_coords = pixel2cam(depth, intrinsics.inverse())  # [B,3,H,W]

    pose_mat = pose_vec2mat(pose, rotation_mode)  # [B,3,4]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    src_pixel_coords = cam2pixel(
        cam_coords, rot, tr, padding_mode)  # [B,H,W,2]
    projected_img = F.grid_sample(
        img, src_pixel_coords, padding_mode=padding_mode)

    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1

    return projected_img, valid_points

def backproject_depth(cam_coords, pose):
    """Transform coordinates in the camera frame to other camera frame
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    ones = torch.ones(b, 1, h * w).cuda()
    pix_coords = torch.cat([cam_coords_flat, ones], 1)
    cam_space = torch.matmul(pose, pix_coords)
    return cam_space

def cam2pixel2(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    X_norm = 2*(X / Z)/(w-1) - 1
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]
    if padding_mode == 'zeros':
        X_mask = ((X_norm > 1)+(X_norm < -1)).detach()
        # make sure that no point in warped image is a combinaison of im and gray
        X_norm[X_mask] = 2
        Y_mask = ((Y_norm > 1)+(Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    valid_mask = ((X_mask.float() + Y_mask.float()) == 0).reshape(b, h, w)
    return pixel_coords.reshape(b, h, w, 2), Z.reshape(b, 1, h, w), valid_mask


def inverse_warp2_orig(img, depth, ref_depth, pose, intrinsics, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, 1, H, W]
        ref_depth: the source depth map (where to sample depth) -- [B, 1, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_mask: Float array indicating point validity
        projected_depth: sampled depth from source image
        computed_depth: computed depth of source image using the target depth
    """
    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'B1HW')
    check_sizes(ref_depth, 'ref_depth', 'B1HW')
    check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = img.size()

    cam_coords = pixel2cam(depth.squeeze(1), intrinsics.inverse())  # [B,3,H,W]

    pose_mat = pose_vec2mat(pose)  # [B,3,4]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    src_pixel_coords, computed_depth,_ = cam2pixel2(cam_coords, rot, tr, padding_mode)  # [B,H,W,2]
    projected_img = F.grid_sample(img, src_pixel_coords, padding_mode=padding_mode, align_corners=False)

    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1
    valid_mask = valid_points.unsqueeze(1).float()

    projected_depth = F.grid_sample(ref_depth, src_pixel_coords, padding_mode=padding_mode, align_corners=False)

    return projected_img, valid_mask, projected_depth, computed_depth

def inverse_warp2(img_to_warp, target_depth, depth_to_warp, pose, intrinsics, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img_to_warp: the source image (where to sample pixels) -- [B, 3, H, W]
        target_depth: depth map of the target image -- [B, 1, H, W]
        depth_to_warp: the source depth map (where to sample depth) -- [B, 1, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_mask: Float array indicating point validity
        projected_depth: sampled depth from source image  
        computed_depth: computed depth of source image using the target depth
    """
    check_sizes(img_to_warp, 'img', 'B3HW')
    check_sizes(target_depth, 'depth', 'B1HW')
    check_sizes(depth_to_warp, 'ref_depth', 'B1HW')
    #check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = img_to_warp.size()

    cam_coords = pixel2cam(target_depth.squeeze(1), intrinsics.inverse())  # [B,3,H,W]
    if pose.shape[1] == 6:
        pose_mat = pose_vec2mat(pose)  # [B,3,4]
    elif pose.shape[1] == 4:
        pose_mat = pose[:, :3, :]
    elif pose.shape[1] == 7:
        pose_mat = quat2mat(pose)
    else:
        raise TypeError

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    src_pixel_coords, computed_depth, _ = cam2pixel2(cam_coords, rot, tr, padding_mode)  # [B,H,W,2]
    # no clue wth computed_depth is

    projected_img = F.grid_sample(img_to_warp, src_pixel_coords, padding_mode=padding_mode, align_corners=False)

    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1
    valid_mask = valid_points.unsqueeze(1).float()

    projected_depth = F.grid_sample(depth_to_warp, src_pixel_coords, padding_mode=padding_mode, align_corners=False)
    # projected depth should ideally look like target_depth but wrong scale

    return projected_img, valid_mask, projected_depth, computed_depth


def get_proj_idx(target_depth, depth_to_warp, pose, intrinsics, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img_to_warp: the source image (where to sample pixels) -- [B, 3, H, W]
        target_depth: depth map of the target image -- [B, 1, H, W]
        depth_to_warp: the source depth map (where to sample depth) -- [B, 1, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_mask: Float array indicating point validity
        projected_depth: sampled depth from source image
        computed_depth: computed depth of source image using the target depth
    """

    check_sizes(target_depth, 'depth', 'B1HW')
    check_sizes(depth_to_warp, 'ref_depth', 'B1HW')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = target_depth.size()

    cam_coords = pixel2cam(target_depth.squeeze(1), intrinsics.inverse())  # [B,3,H,W]

    pose_mat = pose[:, :3, :]  # [B,3,4]

    # print('pose_mat: ', pose_mat)

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    src_pixel_coords, computed_depth, valid_mask = cam2pixel2(cam_coords, rot, tr, padding_mode)  # [B,H,W,2]

    #projected_img = F.grid_sample(img_to_warp, src_pixel_coords, padding_mode=padding_mode, align_corners=False)

    #plt.imsave('test_proj.png', (projected_img[0].detach().cpu().numpy().transpose((1, 2, 0)) + 2.7) / 5.4)

    return src_pixel_coords, valid_mask


def inverse_warp3(img_to_warp, target_depth, depth_to_warp, pose, intrinsics, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img_to_warp: the source image (where to sample pixels) -- [B, 3, H, W]
        target_depth: depth map of the target image -- [B, 1, H, W]
        depth_to_warp: the source depth map (where to sample depth) -- [B, 1, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_mask: Float array indicating point validity
        projected_depth: sampled depth from source image
        computed_depth: computed depth of source image using the target depth
    """
    check_sizes(img_to_warp, 'img', 'B3HW')
    check_sizes(target_depth, 'depth', 'B1HW')
    check_sizes(depth_to_warp, 'ref_depth', 'B1HW')
    #check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = img_to_warp.size()

    cam_coords = pixel2cam(target_depth.squeeze(1), intrinsics.inverse())  # [B,3,H,W]
    if pose.shape[1] == 6:
        pose_mat = pose_vec2mat(pose)  # [B,3,4]
    elif pose.shape[1] == 4:
        pose_mat = pose[:, :3, :]
    else:
        raise TypeError

    b_, _, _ = pose_mat.shape
    zeros = torch.zeros((b_,1,4)).cuda()
    zeros[:,0,-1] = 1
    pose_mat_sq = torch.cat([pose_mat, zeros], 1)
    #print('sq: ', pose_mat_sq.shape)

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    src_pixel_coords, computed_depth,_ = cam2pixel2(cam_coords, rot, tr, padding_mode)  # [B,H,W,2]
    # TODO computed_depth is target depth (t+1) warped to t.
    # no clue wth computed_depth is

    orig = torch.zeros((4, 1)).cuda()
    orig[-1, 0] = 1
    orig[2, 0] = 1

    proj_cam_orig = proj_cam_to_src_pixel.squeeze() @ orig
    #print('proj_cam_orig', proj_cam_orig)
    proj_cam_orig = proj_cam_orig / proj_cam_orig[-1, 0]
    #print('proj_cam_orig', proj_cam_orig)
    test = (intrinsics @ (pose_mat_sq[:,:3,:3] @ pose_mat_sq[:,:3,-1:])).squeeze(0)
    test_orig = test / test[-1,0]
    projected_img = F.grid_sample(img_to_warp, src_pixel_coords, padding_mode=padding_mode, align_corners=False)

    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1
    valid_mask = valid_points.unsqueeze(1).float()

    projected_depth = F.grid_sample(depth_to_warp, src_pixel_coords, padding_mode=padding_mode, align_corners=False)
    # projected depth should ideally look like target_depth but wrong scale
    projected_depths, depth_masks = warp_depth(target_depth, depth_to_warp, pose_mat_sq, intrinsics, batch_size, img_height, img_width)


    return projected_img, valid_mask, projected_depth, computed_depth, proj_cam_orig, pose_mat_sq.squeeze(), projected_depths[:,1,:,], depth_masks[:,1,:,]


def inverse_warp4(img_to_warp, other_image, target_depth, depth_to_warp, pose, intrinsics, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img_to_warp: the source image (where to sample pixels) -- [B, 3, H, W]
        target_depth: depth map of the target image -- [B, 1, H, W]
        depth_to_warp: the source depth map (where to sample depth) -- [B, 1, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_mask: Float array indicating point validity
        projected_depth: sampled depth from source image
        computed_depth: computed depth of source image using the target depth
    """
    check_sizes(img_to_warp, 'img', 'B3HW')
    check_sizes(target_depth, 'depth', 'B1HW')
    check_sizes(depth_to_warp, 'ref_depth', 'B1HW')
    #check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = img_to_warp.size()

    cam_coords = pixel2cam(target_depth.squeeze(1), intrinsics.inverse())  # [B,3,H,W]
    if pose.shape[1] == 6:
        pose_mat = pose_vec2mat(pose)  # [B,3,4]
    elif pose.shape[1] == 4:
        pose_mat = pose[:, :3, :]
    else:
        raise TypeError

    b_, _, _ = pose_mat.shape
    zeros = torch.zeros((b_,1,4)).cuda()
    zeros[:,0,-1] = 1
    pose_mat_sq = torch.cat([pose_mat, zeros], 1)
    #print('sq: ', pose_mat_sq.shape)

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    src_pixel_coords, computed_depth, _ = cam2pixel2(cam_coords, rot, tr, padding_mode)  # [B,H,W,2]
    # TODO computed_depth is target depth (t+1) warped to t.
    # no clue wth computed_depth is

    orig = torch.zeros((4, 1)).cuda()
    orig[-1, 0] = 1
    orig[2, 0] = 1

    proj_cam_orig = proj_cam_to_src_pixel.squeeze() @ orig
    #print('proj_cam_orig', proj_cam_orig)
    proj_cam_orig = proj_cam_orig / proj_cam_orig[-1, 0]
    #print('proj_cam_orig', proj_cam_orig)
    test = (intrinsics @ (pose_mat_sq[:, :3, :3] @ pose_mat_sq[:, :3, -1:])).squeeze(0)
    test_orig = test / test[-1, 0]
    projected_img = F.grid_sample(img_to_warp, src_pixel_coords, padding_mode=padding_mode, align_corners=False)
    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1
    valid_mask = valid_points.unsqueeze(1).float()

    b, h, w = depth_to_warp[0,:,:,:].size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(
        1, h, w).type_as(depth_to_warp)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(
        1, h, w).type_as(depth_to_warp)  # [1, H, W]
    pixel_coords = torch.stack((j_range, i_range), dim=1)  # [1, 3, H, W]

    xy_left = (src_pixel_coords[0,:,:,:].reshape((-1,2)) +1) * 0.5 * 256
    xy_right = pixel_coords[0,:,:,:].reshape((2,-1))
    import numpy as np
    #rdm_idx = np.random.choice(65536, size=10, replace=False)
    rdm_idx = [35180, 29479, 16684, 12936, 1131, 48007, 62609, 15595, 34628, 4905]
    rdm_idx = [29479]
    xy_left_short = xy_left.clone().cpu().numpy()
    xy_right_short = xy_right.clone().cpu().numpy().transpose()
    xy_left_short_ = xy_left_short[rdm_idx,:]
    xy_right_short_ = xy_right_short[rdm_idx,:]

    fig, ax = plt.subplots(2, 1, figsize=(8,8))
    for ax_idx in range(2):
        ax[ax_idx].axis('off')

    ax[0].imshow(torch.cat([img_to_warp, other_image], 3).cpu().detach().numpy()[0].transpose((1, 2, 0)))
    for i in range(1):
        #x_left, x_right, y_left, y_right = 100, 120+256, 200, 250
        ax[0].plot([xy_left_short_[:,0], xy_right_short_[:,0]+256], [xy_left_short_[:,1], xy_right_short_[:,1]], '-')

    ax[1].imshow(torch.cat([img_to_warp, projected_img], 3).cpu().detach().numpy()[0].transpose((1, 2, 0)))
    for i in range(1):
        #x_left, x_right, y_left, y_right = 100, 120+256, 200, 250
        ax[1].plot([xy_left_short_[:,0], xy_right_short_[:,0]+256], [xy_left_short_[:,1], xy_right_short_[:,1]], '-')

    #ax[2].add_subplot(projection='3d')
    plt.savefig('flying_warps' +str(np.random.randint(300)) +'.png', bbox_inches="tight")


    projected_depth = F.grid_sample(depth_to_warp, src_pixel_coords, padding_mode=padding_mode, align_corners=False)
    # projected depth should ideally look like target_depth but wrong scale
    projected_depths, depth_masks = warp_depth(target_depth, depth_to_warp, pose_mat_sq, intrinsics, batch_size, img_height, img_width)


    return projected_img, valid_mask, projected_depth, computed_depth, proj_cam_orig, pose_mat_sq.squeeze(), projected_depths[:,1,:,], depth_masks[:,1,:,]


def inverse_warp_P(img_to_warp, target_depth, depth_to_warp, pose, intrinsics, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img_to_warp: the source image (where to sample pixels) -- [B, 3, H, W]
        target_depth: depth map of the target image -- [B, 1, H, W]
        depth_to_warp: the source depth map (where to sample depth) -- [B, 1, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 4, 4]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_mask: Float array indicating point validity
        projected_depth: sampled depth from source image
        computed_depth: computed depth of source image using the target depth
    """
    check_sizes(img_to_warp, 'img', 'B3HW')
    check_sizes(target_depth, 'depth', 'B1HW')
    check_sizes(depth_to_warp, 'ref_depth', 'B1HW')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = img_to_warp.size()

    cam_coords = pixel2cam(target_depth.squeeze(1), intrinsics.inverse())  # [B,3,H,W]

    pose_mat = pose[:, :3, :] # [B,3,4]
    zeros = torch.zeros((1, 1, 4)).cuda()
    zeros[0, 0, -1] = 1
    pose_mat_sq = torch.cat([pose_mat, zeros], 1)
    #print('pose_mat: ', pose_mat)

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    src_pixel_coords, computed_depth,_ = cam2pixel2(cam_coords, rot, tr, padding_mode)  # [B,H,W,2]
    # no clue wth computed_depth is
    orig = torch.zeros((4, 1)).cuda()
    orig[-1, 0] = 1
    orig[2, 0] = 1

    proj_cam_orig = proj_cam_to_src_pixel.squeeze() @ orig
    #print('proj_cam_orig', proj_cam_orig)
    proj_cam_orig = proj_cam_orig / proj_cam_orig[-1, 0]
    #print('proj_cam_orig', proj_cam_orig)
    test = (intrinsics @ (pose[:,:3,:3] @ pose[:,:3,-1:])).squeeze(0)
    test_orig = test / test[-1,0]

    projected_img = F.grid_sample(img_to_warp, src_pixel_coords, padding_mode=padding_mode, align_corners=False)

    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1
    valid_mask = valid_points.unsqueeze(1).float()

    warped_depth = F.grid_sample(depth_to_warp, src_pixel_coords, padding_mode=padding_mode, align_corners=False)
    # projected depth should ideally look like target_depth but wrong scale
    projected_depths, depth_masks = warp_depth(target_depth, depth_to_warp, pose_mat_sq, intrinsics, batch_size, img_height, img_width)


    #fig, ax = plt.subplots(1, 4, figsize=(12, 3))
    #ax[0].imshow((projected_depths[:, 1, :, :]).abs().cpu().detach().numpy()[0], vmin=0, vmax=20)
    #ax[1].imshow((projected_depths[:, 0, :, :]).abs().cpu().detach().numpy()[0], vmin=0, vmax=20)
    #ax[2].imshow((depth_to_warp.squeeze(1)).cpu().detach().numpy()[0], vmin=0, vmax=20)
    #ax[3].imshow((target_depth.squeeze(1)).cpu().detach().numpy()[0], vmin=0, vmax=20)
    #ax[0].title.set_text('depth t warped to t+1')
    #ax[1].title.set_text('depth t+1 warped to t')
    #ax[2].title.set_text('depth t')
    #ax[3].title.set_text('depth t+1')
    #plt.savefig('test_depths.png')

    return projected_img, valid_mask, warped_depth, computed_depth, proj_cam_orig, projected_depths[:,1,:,], depth_masks[:,1,:,]

def project3D(batch_size, height, width, points, K, eps=1e-7):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """

    cam_points = torch.matmul(K, points[:, :3, :])
    pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + torch.tensor(eps).cuda())
    pix_coords = pix_coords.view(batch_size, 2, height, width)
    pix_coords = pix_coords.permute(0, 2, 3, 1)
    pix_coords[..., 0] /= width - 1
    pix_coords[..., 1] /= height - 1
    pix_coords = (pix_coords - 0.5) * 2
    return pix_coords



def warp_depth(target_depth, depth_to_warp, pose_mat, intrinsics, b, h, w):
    cam_coords_target = pixel2cam(target_depth.squeeze(1), intrinsics.inverse())
    cam_coords_source = pixel2cam(depth_to_warp.squeeze(1), intrinsics.inverse())
    space_points_from_target_in_source = backproject_depth(cam_coords_target, pose_mat)
    space_points_from_source_in_target = backproject_depth(cam_coords_source, pose_mat.inverse())

    world_depths_from_source = space_points_from_source_in_target[:, 2, :].view(b, h, w).unsqueeze(1)
    image2_from1 = project3D(b, h, w, space_points_from_target_in_source[:, :3, :], intrinsics)
    #image_1_warped_to_2 = F.grid_sample(world_depths_from_source, image2_from1, padding_mode="border", align_corners=False)
    image_1_warped_to_2 = F.grid_sample(world_depths_from_source, image2_from1, padding_mode="zeros",
                                        align_corners=False)

    world_depths_from_target = space_points_from_target_in_source[:, 2, :].view(b, h, w).unsqueeze(1)
    image1_from2 = project3D(b, h, w, space_points_from_source_in_target[:, :3, :], intrinsics)
    #image_2_warped_to_1 = F.grid_sample(world_depths_from_target, image1_from2, padding_mode="border", align_corners=False)
    image_2_warped_to_1 = F.grid_sample(world_depths_from_target, image1_from2, padding_mode="zeros",
                                        align_corners=False)

    im_1_mask = ((((image1_from2 > 1) + (image1_from2 < -1)) * 1.).sum(3).unsqueeze(1) == 0)

    im_2_mask = ((((image2_from1 > 1) + (image2_from1 < -1)) * 1.).sum(3).unsqueeze(
        1) == 0)  # torch.where(image_1_in_2 > 0, y, x)

    #plt.imsave('deletem.png', im_1_mask[0, 0, :, :].cpu().detach().numpy())
    warp_mask = torch.cat((im_1_mask, im_2_mask), 1)
    A_warped = torch.cat((image_2_warped_to_1, image_1_warped_to_2), 1)
    return A_warped, warp_mask