import numpy as np
from pathlib import Path
import torch
import os
import random

import torch.optim as optim
import pandas as pd
import torch.nn as nn
import torch
def conv3x3(in_channels, out_channels, kernel_size=3):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)]


def add_feature_maps(x1, x2):
    x1 = torch.nn.functional.interpolate(x1, x2.shape[2])
    if x1.shape[1] == x2.shape[1]:
        x = x1 + x2
    else:
        x = x1 + x2[:, 0 - x1.shape[1]:, :, ]
        x = torch.cat([x, x2[:, x1.shape[1] - x2.shape[1]:, :, ]], dim=1)
    return x


class DeepMedic(nn.Module):
    def __init__(self, out = 1):
        super(DeepMedic, self).__init__()

        self.first_res = nn.Sequential(
            *conv3x3(4, 30),
            *conv3x3(30, 30)
        )

        self.second_res = nn.Sequential(
            *conv3x3(30, 40),
            *conv3x3(40, 40)
        )
        self.third_res = nn.Sequential(
            *conv3x3(40, 40),
            *conv3x3(40, 40)
        )
        self.fourth_res = nn.Sequential(
            *conv3x3(40, 50),
            *conv3x3(50, 50)
        )

        self.final_conv = nn.Sequential(
            *conv3x3(100, 150, kernel_size=1),
            *conv3x3(150, 150, kernel_size=1)
        )

        self.classification = nn.Sequential(
            nn.Conv2d(150, out, kernel_size=1)
        )

    def one_branch(self, my_input):
        x1 = self.first_res(my_input)
        x2 = self.second_res(x1)
        x = add_feature_maps(x1, x2)
        x3 = self.third_res(x)
        x = add_feature_maps(x, x3)
        x4 = self.fourth_res(x)
        output = add_feature_maps(x, x4)
        return output

    def forward(self, inputs):
        high_res, low_res = inputs
        high_output = self.one_branch(high_res)
        low_output = self.one_branch(low_res)
        upsampled_low_output = torch.nn.functional.interpolate(low_output, high_output.shape[2], mode='bilinear')
        concatenated = torch.cat([high_output, upsampled_low_output], dim=1)
        # ---------------------------------------------------------------------------------------------
        final = self.final_conv(concatenated)
        x = add_feature_maps(concatenated, final)
        x = self.classification(x)
        return x
