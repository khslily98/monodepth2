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
    def __init__(self, deformable_conv, uncertainty_input, num_input_frames):
        super(PoseCNN, self).__init__()
        self.deformable_conv = deformable_conv
        self.uncertainty_input = uncertainty_input
        self.num_input_frames = num_input_frames

        self.convs = OrderedDict()
        
        if not deformable_conv:
            self.convs[('pose', 0)] = nn.Conv2d(3 * num_input_frames, 16, 7, 2, 3)
            self.convs[('pose', 1)] = nn.Conv2d(16, 32, 5, 2, 2)
            self.convs[('pose', 2)] = nn.Conv2d(32, 64, 3, 2, 1)
            self.convs[('pose', 3)] = nn.Conv2d(64, 128, 3, 2, 1)
        else:
            self.deformable_convs = OrderedDict()

            self.convs[('pose', 0)] = ops.DeformConv2d(3 * num_input_frames, 16, 7, 2, 3)
            self.deformable_convs[('offset', 0)] = nn.Conv2d(3 * num_input_frames, 2*7*7, 7, 2, 3)
            self.deformable_convs[('mask', 0)]   = nn.Conv2d(3 * num_input_frames, 7*7, 7, 2, 3)

            self.convs[('pose', 1)] = ops.DeformConv2d(16, 32, kernel_size=5, stride=2, padding=2)
            self.deformable_convs[('offset', 1)] = nn.Conv2d(16, 2*5*5, 5, 2, 2)
            self.deformable_convs[('mask', 1)]   = nn.Conv2d(16, 5*5, 5, 2, 2)

            self.convs[('pose', 2)]  = ops.DeformConv2d(32, 64, kernel_size=3, stride=2, padding=1)
            self.deformable_convs[('offset', 2)] = nn.Conv2d(32, 2*3*3, 3, 2, 1)
            self.deformable_convs[('mask', 2)]   = nn.Conv2d(32, 3*3, 3, 2, 1)

            self.convs[('pose', 3)]  = ops.DeformConv2d(64, 128, kernel_size=3, stride=2, padding=1)
            self.deformable_convs[('offset', 2)] = nn.Conv2d(64, 2*3*3, 3, 2, 1)
            self.deformable_convs[('mask', 2)]   = nn.Conv2d(64, 3*3, 3, 2, 1)

            for i in range(4):
                nn.init.constant_(self.deformable_convs[('offset', i)].weight, 0.)
                nn.init.constant_(self.deformable_convs[('offset', i)].bias, 0.)
                nn.init.constant_(self.deformable_convs[('mask', i)].weight, 0.)
                nn.init.constant_(self.deformable_convs[('mask', i)].bias, 0.)
            
            self.deform = nn.ModuleList(list(self.deformable_convs.values()))

        self.convs[('pose', 4)] = nn.Conv2d(128, 256, 3, 2, 1)
        self.convs[('pose', 5)] = nn.Conv2d(256, 256, 3, 2, 1)
        self.convs[('pose', 6)] = nn.Conv2d(256, 256, 3, 2, 1)

        self.convs[('out')] = nn.Conv2d(256, 6 * (num_input_frames - 1), 1)

        self.num_convs = len(self.convs)

        self.relu = nn.ReLU(True)

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, out, uncertainty):
        for i in range(self.num_convs):
            if i <= 3 and self.deformable_conv:
                if self.uncertainty_input:
                    attention = F.sigmoid(uncertainty[i])
                    out = self.convs[i](out, self.offset[i](out * attention), F.sigmoid(self.mask[i](out * attention)))
                else:
                    out = self.convs[i](out, self.offset[i](out), F.sigmoid(self.mask[i](out)))
            else:
                out = self.convs[i](out)
            out = self.relu(out)

        out = self.pose_conv(out)
        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_input_frames - 1, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation
