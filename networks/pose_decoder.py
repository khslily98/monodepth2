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


class PoseDecoder(nn.Module):
    def __init__(self, 
                num_ch_enc, 
                num_input_features, 
                deformable_conv=False,
                uncertainty=False,
                num_frames_to_predict_for=None):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features
        self.deformable_conv = deformable_conv
        self.uncertainty = uncertainty

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.squeeze = nn.Conv2d(self.num_ch_enc[-1], 256, 1)

        if deformable_conv:
            self.offset_convs = OrderedDict()
            self.mask_convs = OrderedDict()

            self.convs[0]   = ops.DeformConv2d(num_input_features*256, 256, 3, 1, 1)
            self.offset_convs[0] = nn.Conv2d(num_input_features*256, 2*3*3, 3, 1, 1)
            self.mask_convs[0]   = nn.Conv2d(num_input_features*256, 3*3, 3, 1, 1)

            self.convs[1] = ops.DeformConv2d(256, 256, 3, 1, 1)
            self.offset_convs[1] = nn.Conv2d(256, 2*3*3, 3, 1, 1)
            self.mask_convs[1]   = nn.Conv2d(256, 3*3, 3, 1, 1)

            self.convs[2] = ops.DeformConv2d(256, 6*num_frames_to_predict_for, 1, 1, 0)
            self.offset_convs[2] = nn.Conv2d(256, 2*1*1, 3, 1, 1)
            self.mask_convs[2]   = nn.Conv2d(256, 1*1, 3, 1, 1)

            for i in range(3):
                nn.init.constant_(self.offset_convs[("offset", i)].weight, 0.)
                nn.init.constant_(self.offset_convs[("offset", i)].bias, 0.)
                nn.init.constant_(self.mask_convs[("mask", i)].weight, 0.)
                nn.init.constant_(self.mask_convs[("mask", i)].bias, 0.)

            self.offset = nn.ModuleList(list(self.offset_convs.values()))
            self.mask = nn.ModuleList(list(self.mask_convs.values()))
        else:
            self.convs[0] = nn.Conv2d(num_input_features * 256, 256, 3, 1, 1)
            self.convs[1] = nn.Conv2d(256, 256, 3, 1, 1)
            self.convs[2] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.num_convs = len(self.convs)
        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features, uncertainty):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.squeeze(f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            if self.deformable_conv:
                offset = self.offset_convs[i](out)
                mask = torch.sigmoid(self.mask_convs[i](out))
                out = self.convs[i](out, offset, mask)
            else:
                out = self.convs[i](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation
