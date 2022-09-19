# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from collections import OrderedDict

from layers import *

class PoseCNN(nn.Module):
    def __init__(self, deformable_conv, uncertainty, num_input_frames):
        super(PoseCNN, self).__init__()
        self.deformable_conv = deformable_conv
        self.uncertainty = uncertainty
        self.num_input_frames = num_input_frames

        self.convs = OrderedDict()
        
        if not deformable_conv:
            # HS add
            self.convs[0] = nn.Conv2d(3 * num_input_frames + 1, 16, 3, 2, 1)
            # self.convs[0] = nn.Conv2d(3 * num_input_frames, 16, 3, 2, 1)
            self.convs[1] = nn.Conv2d(16, 32, 3, 2, 1)
            self.convs[2] = nn.Conv2d(32, 64, 3, 2, 1)
            self.convs[3] = nn.Conv2d(64, 128, 3, 2, 1)
        else:
            self.offset_convs = OrderedDict()
            self.mask_convs = OrderedDict()

            self.convs[0] = ops.DeformConv2d(3 * num_input_frames, 16, 7, 2, 3)
            self.offset_convs[0] = nn.Conv2d(3 * num_input_frames, 2*7*7, 7, 2, 3)
            self.mask_convs[0]   = nn.Conv2d(3 * num_input_frames, 7*7, 7, 2, 3)

            self.convs[1] = ops.DeformConv2d(16, 32, kernel_size=5, stride=2, padding=2)
            self.offset_convs[1] = nn.Conv2d(16, 2*5*5, 5, 2, 2)
            self.mask_convs[1]   = nn.Conv2d(16, 5*5, 5, 2, 2)

            self.convs[2]  = ops.DeformConv2d(32, 64, kernel_size=3, stride=2, padding=1)
            self.offset_convs[2] = nn.Conv2d(32, 2*3*3, 3, 2, 1)
            self.mask_convs[2]   = nn.Conv2d(32, 3*3, 3, 2, 1)

            self.convs[3]  = ops.DeformConv2d(64, 128, kernel_size=3, stride=2, padding=1)
            self.offset_convs[3] = nn.Conv2d(64, 2*3*3, 3, 2, 1)
            self.mask_convs[3]   = nn.Conv2d(64, 3*3, 3, 2, 1)

            for i in range(len(self.offset_convs)):
                nn.init.constant_(self.offset_convs[i].weight, 0.)
                nn.init.constant_(self.offset_convs[i].bias, 0.)
                nn.init.constant_(self.mask_convs[i].weight, 0.)
                nn.init.constant_(self.mask_convs[i].bias, 0.)
            
            self.offset = nn.ModuleList(list(self.offset_convs.values()))
            self.mask = nn.ModuleList(list(self.mask_convs.values()))

        self.convs[4] = nn.Conv2d(128, 256, 3, 2, 1)
        self.convs[5] = nn.Conv2d(256, 512, 3, 2, 1)
        self.convs[6] = nn.Conv2d(512, 1024, 3, 1, 1)

        if uncertainty:
            self.upconvs = OrderedDict()
            self.upconvs["0_0"] = ConvBlock(1024, 512)
            self.upconvs["0_1"] = ConvBlock(1024, 512)

            self.upconvs["1_0"] = ConvBlock(512, 256)
            self.upconvs["1_1"] = ConvBlock(512, 256)
            
            self.upconvs["2_0"] = ConvBlock(256, 128)
            self.upconvs["2_1"] = ConvBlock(256, 128)

            self.upconvs["3_0"] = ConvBlock(128, 64)
            self.upconvs["3_1"] = ConvBlock(128, 64)
            self.upconvs["3_out"] = Conv3x3(64, 1)

            self.upconvs["4_0"] = ConvBlock(64, 32)
            self.upconvs["4_1"] = ConvBlock(64, 32)
            self.upconvs["4_out"] = Conv3x3(32, 1)

            self.upconvs["5_0"] = ConvBlock(32, 16)
            self.upconvs["5_1"] = ConvBlock(32, 16)
            self.upconvs["5_out"] = Conv3x3(16, 1)

            self.upconvs["6_0"] = ConvBlock(16, 16)
            self.upconvs["6_1"] = ConvBlock(16, 16)
            self.upconvs["6_out"] = Conv3x3(16, 1)

            self.upconv = nn.ModuleList(list(self.upconvs.values()))

        self.pose_conv = nn.Conv2d(1024, 6 * (num_input_frames - 1), 1)

        self.relu = nn.ReLU(True)

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, out, depth_uncertainty):
        if self.deformable_conv:
            feat0 = self.relu(self.convs[0](out, self.offset_convs[0](out), torch.sigmoid(self.mask_convs[0](out))))
            feat1 = self.relu(self.convs[1](feat0, self.offset_convs[1](feat0), torch.sigmoid(self.mask_convs[1](feat0))))
            feat2 = self.relu(self.convs[2](feat1, self.offset_convs[2](feat1), torch.sigmoid(self.mask_convs[2](feat1))))
            feat3 = self.relu(self.convs[3](feat2, self.offset_convs[3](feat2), torch.sigmoid(self.mask_convs[3](feat2))))
        else:
            feat0 = self.relu(self.convs[0](out))
            feat1 = self.relu(self.convs[1](feat0))
            feat2 = self.relu(self.convs[2](feat1))
            feat3 = self.relu(self.convs[3](feat2))
        
        feat4 = self.relu(self.convs[4](feat3))
        feat5 = self.relu(self.convs[5](feat4))
        feat6 = self.relu(self.convs[6](feat5))

        if self.uncertainty:
            pose_uncertainty = []
            x = self.upconvs["0_0"](feat6)
            x = torch.cat([x, feat5], 1)
            x = self.upconvs["0_1"](x)

            x = self.upconvs["1_0"](x)
            x = torch.cat([upsample(x), feat4], 1)
            x = self.upconvs["1_1"](x)

            x = self.upconvs["2_0"](x)
            x = torch.cat([upsample(x), feat3], 1)
            x = self.upconvs["2_1"](x)

            x = self.upconvs["3_0"](x)
            x = torch.cat([upsample(x), feat2], 1)
            x = self.upconvs["3_1"](x)
            pose_uncertainty.append(torch.sigmoid(self.upconvs["3_out"](x)))

            x = self.upconvs["4_0"](x)
            x = torch.cat([upsample(x), feat1], 1)
            x = self.upconvs["4_1"](x)
            pose_uncertainty.append(torch.sigmoid(self.upconvs["4_out"](x)))

            x = self.upconvs["5_0"](x)
            x = torch.cat([upsample(x), feat0], 1)
            x = self.upconvs["5_1"](x)
            pose_uncertainty.append(torch.sigmoid(self.upconvs["5_out"](x)))

            x = self.upconvs["6_0"](x)
            x = upsample(x)
            x = self.upconvs["6_1"](x)
            pose_uncertainty.append(torch.sigmoid(self.upconvs["6_out"](x)))
            pose_uncertainty.reverse()

        pose = self.pose_conv(feat6)
        pose = pose.mean(3).mean(2)

        pose = 0.01 * pose.view(-1, self.num_input_frames - 1, 1, 6)

        axisangle = pose[..., :3]
        translation = pose[..., 3:]

        if self.uncertainty:
            return axisangle, translation, pose_uncertainty
        else:
            return axisangle, translation, None
