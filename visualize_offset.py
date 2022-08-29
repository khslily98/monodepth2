# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.transforms import ToPILImage
import cv2

from utils import readlines, load_training_option
from options import MonodepthOptions
from datasets import KITTIOdomDataset

import networks


class SaveOutput:
    def __init__(self):
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        
    def clear(self):
        self.outputs = []

def plot_offsets(img, save_output, roi_x, roi_y):
    cv2.circle(img, center=(roi_x, roi_y), color=(0, 255, 0), radius=1, thickness=-1)
    input_img_h, input_img_w = img.shape[:2]
    for offsets in save_output.outputs:
        offset_tensor_h, offset_tensor_w = offsets.shape[2:]
        resize_factor_h, resize_factor_w = input_img_h/offset_tensor_h, input_img_w/offset_tensor_w

        offsets_y = offsets[:, ::2]
        offsets_x = offsets[:, 1::2]

        grid_y = np.arange(0, offset_tensor_h)
        grid_x = np.arange(0, offset_tensor_w)

        grid_x, grid_y = np.meshgrid(grid_x, grid_y)

        sampling_y = grid_y + offsets_y.detach().cpu().numpy()
        sampling_x = grid_x + offsets_x.detach().cpu().numpy()

        sampling_y *= resize_factor_h
        sampling_x *= resize_factor_w

        sampling_y = sampling_y[0] # remove batch axis
        sampling_x = sampling_x[0] # remove batch axis

        sampling_y = sampling_y.transpose(1, 2, 0) # c, h, w -> h, w, c
        sampling_x = sampling_x.transpose(1, 2, 0) # c, h, w -> h, w, c

        sampling_y = np.clip(sampling_y, 0, input_img_h)
        sampling_x = np.clip(sampling_x, 0, input_img_w)

        sampling_y = cv2.resize(sampling_y, dsize=None, fx=resize_factor_w, fy=resize_factor_h)
        sampling_x = cv2.resize(sampling_x, dsize=None, fx=resize_factor_w, fy=resize_factor_h)

        sampling_y = sampling_y[roi_y, roi_x]
        sampling_x = sampling_x[roi_y, roi_x]
        
        for y, x in zip(sampling_y, sampling_x):
            y = round(y)
            x = round(x)
            cv2.circle(img, center=(x, y), color=(0, 0, 255), radius=1, thickness=-1)

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
    dataloader = DataLoader(dataset, 1, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True, drop_last=False)

    save_output = SaveOutput()
    to_pil_image = ToPILImage()

    if not opt.pose_model_type == 'posecnn':
        pose_encoder_path = os.path.join(opt.load_weights_folder, "pose_encoder.pth")
        pose_decoder_path = os.path.join(opt.load_weights_folder, "pose.pth")

        pose_encoder = networks.ResnetEncoder(opt.num_layers, False, 2)
        pose_encoder.load_state_dict(torch.load(pose_encoder_path))

        pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, 1,
                                            opt.deformable_conv,
                                            opt.uncertainty_input,
                                            num_frames_to_predict_for=1)
        pose_decoder.load_state_dict(torch.load(pose_decoder_path))
        
        for name, layer in pose_decoder.named_modules():
            if 'offset' in name and isinstance(layer, nn.Conv2d):
                layer.register_forward_hook(save_output)

        pose_encoder.cuda()
        pose_encoder.eval()
        pose_decoder.cuda()
        pose_decoder.eval()
    else:
        pose_path = os.path.join(opt.load_weights_folder, "pose.pth")
        pose_network = networks.PoseCNN(opt.deformable_conv, opt.uncertainty_input, 2)
        pose_network.load_state_dict(torch.load(pose_path))

        for name, layer in pose_network.named_modules():
            if 'offset' in name and isinstance(layer, nn.Conv2d):
                layer.register_forward_hook(save_output)
        
        pose_network.cuda()
        pose_network.eval()

    if opt.uncertainty_input:
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

    print("-> Computing deformable convolution offsets")

    opt.frame_ids = [0, 1]  # pose network only takes two frames as input

    with torch.no_grad():
        for idx, inputs in enumerate(dataloader):
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()

            if opt.uncertainty_input:
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

            # RGB to BGR transformation for cv2
            img1, img2 = inputs[("color_aug", 0, 0)][:, [2, 1, 0], :, :], inputs[("color_aug", 1, 0)][:, [2, 1, 0], :, :]

            if not opt.pose_model_type == 'posecnn':
                features = [pose_encoder(all_color_aug)]
                _ = pose_decoder(features, uncertainty)
            else:
                _ = pose_network(all_color_aug, uncertainty)
            
            # CHW to HWC by PIL Image transformation
            img1 = np.asarray(to_pil_image(img1.squeeze().cpu()))
            img2 = np.asarray(to_pil_image(img2.squeeze().cpu()))
            roi_y, roi_x = opt.height//2, opt.width//2

            plot_offsets(img1, save_output, roi_x=roi_x, roi_y=roi_y)
            plot_offsets(img2, save_output, roi_x=roi_x, roi_y=roi_y)
            save_output.clear()

            cv2.imwrite(f"visualizations/{idx:06d}_0.jpg", img1)
            cv2.imwrite(f"visualizations/{idx:06d}_1.jpg", img2)

if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
