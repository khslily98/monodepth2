# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import transformation_from_parameters
from utils import readlines, load_training_option
from options import MonodepthOptions
from datasets import KITTIOdomDataset
from kitti_odometry import KittiEvalOdom

import networks

def pose_to_traj(rel_poses):
    traj = np.zeros((len(rel_poses)+1, 4, 4))
    init_pose = np.eye(4)
    traj[0] = init_pose

    for idx, pose in enumerate(rel_poses):
        prev_pose = traj[idx]
        traj[idx+1] = prev_pose @ pose
    
    # Algin axis with groundtruth
    traj[:, 0, 1:3] *= -1
    traj[:, 1:3, 0] *= -1
    traj[:, 1:3, 3] *= -1

    return np.reshape(traj[:, :3, :], (-1, 12))


# from https://github.com/tinghuiz/SfMLearner
def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        xyzs.append(cam_to_world[:3, 3])
    return xyzs


# from https://github.com/tinghuiz/SfMLearner
def compute_ate(gtruth_xyz, pred_xyz_o):

    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz_o[0]
    pred_xyz = pred_xyz_o + offset[None, :]

    # Optimize the scaling factor
    scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz ** 2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sqrt(np.sum(alignment_error ** 2)) / gtruth_xyz.shape[0])
    return rmse


def evaluate(opt):
    """Evaluate odometry on the KITTI dataset
    """
    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    assert opt.eval_split == "odom_9" or opt.eval_split == "odom_10", \
        "eval_split should be either odom_9 or odom_10"
    
    opt = load_training_option(opt)
    sequence_id = int(opt.eval_split.split("_")[1])

    filenames = readlines(
        os.path.join(os.path.dirname(__file__), "splits", "odom",
                     "test_files_{:02d}.txt".format(sequence_id)))

    dataset = KITTIOdomDataset(opt.data_path, filenames, opt.height, opt.width,
                               [0, 1], 4, is_train=False)
    dataloader = DataLoader(dataset, opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True, drop_last=False)

    if opt.pose_model_type is not 'posecnn':
        pose_encoder_path = os.path.join(opt.load_weights_folder, "pose_encoder.pth")
        pose_decoder_path = os.path.join(opt.load_weights_folder, "pose.pth")

        pose_encoder = networks.ResnetEncoder(opt.num_layers, False, 2)
        pose_encoder.load_state_dict(torch.load(pose_encoder_path))

        pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, 1,
                                            opt.deformable_conv,
                                            opt.uncertainty,
                                            num_frames_to_predict_for=1)
        pose_decoder.load_state_dict(torch.load(pose_decoder_path))

        pose_encoder.cuda()
        pose_encoder.eval()
        pose_decoder.cuda()
        pose_decoder.eval()
    else:
        pose_path = os.path.join(opt.load_weights_folder, "pose.pth")
        pose_network = networks.PoseCNN(opt.deformable_conv, opt.uncertainty, 2)
        pose_network.load_state_dict(torch.load(pose_path))

        pose_network.cuda()
        pose_network.eval()

    if opt.uncertainty:
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)
        encoder = networks.ResnetEncoder(opt.num_layers, False)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

    pred_poses = []

    print("-> Computing pose predictions")

    opt.frame_ids = [0, 1]  # pose network only takes two frames as input

    with torch.no_grad():
        for inputs in dataloader:
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()

            if opt.uncertainty:
                input_color = inputs[("color_aug", 0, 0)]
                N = input_color.shape[0]

                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
                outputs = depth_decoder(encoder(input_color))
                uncertainty = []
                for s in opt.scales:
                    disp, disp_rev = outputs[('disp', s)][:N], torch.flip(outputs[('disp', s)][N:], [3])
                    uncertainty.append(torch.abs(disp - disp_rev))
            else:
                uncertainty = None

            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in opt.frame_ids], 1)
            if opt.pose_model_type is not 'posecnn':
                features = [pose_encoder(all_color_aug)]
                axisangle, translation = pose_decoder(features)
            else:
                axisangle, translation = pose_network(all_color_aug, uncertainty)

            pred_poses.append(
                transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy())

    pred_poses = np.concatenate(pred_poses)
    pose_save_path = os.path.join(opt.load_weights_folder, f"{sequence_id:02d}.npy")
    np.save(pose_save_path, pred_poses)
    print("-> Predictions saved to", pose_save_path)

    pred_traj = pose_to_traj(pred_poses)
    traj_save_path = os.path.join(opt.load_weights_folder, f"{sequence_id:02d}.txt")
    np.savetxt(traj_save_path, pred_traj, fmt="%1.6e")
    print("-> Trajectory saved to", traj_save_path)
    
    gt_poses_path = os.path.join(opt.data_path, "poses")

    eval_tool = KittiEvalOdom()
    eval_tool.eval(gt_poses_path, opt.load_weights_folder, alignment='scale', seqs=[sequence_id], plot=True)

    # Compute 5-frame snippet ATE
    gt_poses_path = os.path.join(opt.data_path, "poses", "{:02d}.txt".format(sequence_id))
    gt_global_poses = np.loadtxt(gt_poses_path).reshape(-1, 3, 4)
    gt_global_poses = np.concatenate(
        (gt_global_poses, np.zeros((gt_global_poses.shape[0], 1, 4))), 1)
    gt_global_poses[:, 3, 3] = 1
    gt_xyzs = gt_global_poses[:, :3, 3]

    gt_local_poses = []
    for i in range(1, len(gt_global_poses)):
        gt_local_poses.append(
            np.linalg.inv(np.dot(np.linalg.inv(gt_global_poses[i - 1]), gt_global_poses[i])))

    ates = []
    num_frames = gt_xyzs.shape[0]
    track_length = 5
    for i in range(0, num_frames - 1):
        local_xyzs = np.array(dump_xyz(pred_poses[i:i + track_length - 1]))
        gt_local_xyzs = np.array(dump_xyz(gt_local_poses[i:i + track_length - 1]))

        ates.append(compute_ate(gt_local_xyzs, local_xyzs))

    print("\n   Trajectory error: {:0.3f}, std: {:0.3f}\n".format(np.mean(ates), np.std(ates)))

if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
