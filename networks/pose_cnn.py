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

class PoseCNN(nn.Module):
    def __init__(self, deformable_conv, uncertainty, num_input_frames):
        super(PoseCNN, self).__init__()
        self.deformable_conv = deformable_conv
        self.uncertainty = uncertainty
        self.num_input_frames = num_input_frames

        self.convs = OrderedDict()
        
        if not deformable_conv:
            self.convs[0] = nn.Conv2d(3 * num_input_frames, 16, 3, 2, 1)
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
            self.num_deforms = len(self.offset_convs)

        self.convs[4] = nn.Conv2d(128, 256, 3, 2, 1)
        self.convs[5] = nn.Conv2d(256, 512, 3, 2, 1)
        self.convs[6] = nn.Conv2d(512, 1024, 3, 2, 1)

        self.pose_conv = nn.Conv2d(1024, 6 * (num_input_frames - 1), 1)

        self.num_convs = len(self.convs)

        self.relu = nn.ReLU(True)

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, out, uncertainty):
        for i in range(self.num_convs):
            if self.deformable_conv:
                if i <= self.num_deforms - 1:
                    att = 1 if uncertainty is None else torch.sigmoid(uncertainty[i])
                    out = self.convs[i](out, self.offset_convs[i](out*att), torch.sigmoid(self.mask_convs[i](out*att)))
            else:
                out = self.convs[i](out)
            out = self.relu(out)

        out = self.pose_conv(out)
        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_input_frames - 1, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation
